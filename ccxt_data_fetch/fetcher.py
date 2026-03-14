import ccxt
import pandas as pd
import numpy as np
import os
import zipfile
import time
import logging
import requests
import io
from datetime import date, datetime, timezone, timedelta
from tqdm import tqdm
from ccxt_data_fetch.config import DATA_LOCATION, PROXIES, REQUEST_TIMEOUT
from ccxt_data_fetch.utils import format_symbol, get_ms_from_midnight

logger = logging.getLogger(__name__)


# Retry settings for Binance Vision depth downloads
_DEPTH_DOWNLOAD_MAX_RETRIES = 3
_DEPTH_DOWNLOAD_RETRY_DELAY_SEC = 5


# Timeout for Binance Vision file downloads (ZIPs can be large; longer than API timeout)
_VISION_DOWNLOAD_TIMEOUT = 90


def _download_with_retries(url: str) -> requests.Response | None:
    """
    Download a URL with bounded retries for transient network issues.

    - Tries at most `_DEPTH_DOWNLOAD_MAX_RETRIES` times
    - Sleeps `_DEPTH_DOWNLOAD_RETRY_DELAY_SEC` seconds between attempts
    - Uses proxy if configured; for proxy-related failures also tries once without proxy per attempt
    """
    last_error: Exception | None = None

    for attempt in range(1, _DEPTH_DOWNLOAD_MAX_RETRIES + 1):
        try:
            return requests.get(url, proxies=PROXIES, timeout=_VISION_DOWNLOAD_TIMEOUT)
        except (requests.exceptions.ProxyError, OSError) as e:
            last_error = e
            # If proxy is flaky, try once without proxy in the same attempt.
            try:
                logger.warning(
                    "Proxy-related error downloading %s (attempt %s/%s): %s. Retrying once without proxy...",
                    url, attempt, _DEPTH_DOWNLOAD_MAX_RETRIES, e,
                )
                return requests.get(url, timeout=_VISION_DOWNLOAD_TIMEOUT)
            except Exception as no_proxy_e:
                last_error = no_proxy_e
        except (requests.exceptions.Timeout, requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            last_error = e
            # Also try once without proxy (common when proxy is slow or drops SSL for Vision).
            if PROXIES:
                try:
                    logger.warning(
                        "Network error downloading %s (attempt %s/%s): %s. Retrying once without proxy...",
                        url, attempt, _DEPTH_DOWNLOAD_MAX_RETRIES, e,
                    )
                    return requests.get(url, timeout=_VISION_DOWNLOAD_TIMEOUT)
                except Exception as no_proxy_e:
                    last_error = no_proxy_e

        if attempt < _DEPTH_DOWNLOAD_MAX_RETRIES:
            logger.warning(
                f"Transient network error downloading {url} (attempt {attempt}/{_DEPTH_DOWNLOAD_MAX_RETRIES}): "
                f"{last_error}. Waiting {_DEPTH_DOWNLOAD_RETRY_DELAY_SEC}s before retry..."
            )
            time.sleep(_DEPTH_DOWNLOAD_RETRY_DELAY_SEC)

    logger.error(f"Failed to download after {_DEPTH_DOWNLOAD_MAX_RETRIES} attempts: {url}. Last error: {last_error}")
    return None


def fetch_ohlcv_range(symbol, timeframe, since_ms, until_ms, asset_class="cryptofuture"):
    options = {}
    if asset_class == "cryptofuture":
        options["defaultType"] = "future"
    
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "proxies": PROXIES,
            "options": options,
        }
    )

    # Map timeframe to ms
    if timeframe == "1m":
        ms_per_step = 60000
    elif timeframe == "1h":
        ms_per_step = 3600000
    elif timeframe == "1d":
        ms_per_step = 86400000
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    all_ohlcv = []
    current_since = since_ms
    pbar = tqdm(total=until_ms - since_ms, desc=f"Fetching {timeframe} {symbol}")

    while current_since < until_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=current_since, limit=1000
            )
            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            # Ensure we move forward
            next_step = last_ts + ms_per_step
            
            # Update progress
            progress = min(next_step, until_ms) - current_since
            if progress > 0:
                pbar.update(progress)
            
            current_since = max(next_step, current_since + 1) # Prevent infinite loop if exchange returns same data

            if current_since >= until_ms:
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            logger.error(f"Error for {symbol}: {e}")
            time.sleep(10)
            continue

    pbar.close()
    return all_ohlcv


def fetch_funding_rates(symbol, since_ms, until_ms):
    """
    Fetch funding rate history for futures.
    Returns list of [timestamp, fundingRate, ...]
    """
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "proxies": PROXIES,
            "options": {"defaultType": "future"},
        }
    )
    
    all_rates = []
    current_since = since_ms
    pbar = tqdm(total=until_ms - since_ms, desc=f"Fetching Funding Rates {symbol}")
    
    while current_since < until_ms:
        try:
            rates = exchange.fetch_funding_rate_history(
                symbol, since=current_since, limit=1000
            )
            if not rates:
                break
                
            all_rates.extend(rates)
            last_ts = rates[-1]['timestamp']
            
            # Progress
            progress = min(last_ts, until_ms) - current_since
            if progress > 0:
                pbar.update(progress)
                
            current_since = last_ts + 1 # ccxt usually returns inclusive, need to step forward
            
            if current_since >= until_ms:
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            logger.error(f"Error fetching funding rates for {symbol}: {e}")
            time.sleep(10)
            break
            
    pbar.close()
    return all_rates

def get_existing_depth_dates(
    symbol: str,
    asset_class: str = "cryptofuture",
    resolution: str = "minute",
) -> set[str]:
    """
    Return set of date strings (YYYYMMDD) for which depth data already exists on disk.
    Used for resume (skip these) or redownload (refetch and overwrite these).
    """
    formatted_symbol = format_symbol(symbol)
    symbol_dir = os.path.join(
        DATA_LOCATION, asset_class, "binance", resolution, formatted_symbol
    )
    if not os.path.isdir(symbol_dir):
        return set()
    existing = set()
    for f in os.listdir(symbol_dir):
        if f.endswith("_depth.zip"):
            date_part = f.split("_")[0]
            if len(date_part) == 8 and date_part.isdigit():
                existing.add(date_part)
    return existing


def fetch_and_save_depth_range(
    symbol: str,
    since_ms: int,
    until_ms: int,
    asset_class: str = "cryptofuture",
    margin_type: str = "um",
    data_source: str = "binancevision",
    resolution: str = "minute",
    force_redownload: bool = False,
) -> None:
    """
    Fetch depth data for the given range and save to LEAN format under binance/<resolution>/<symbol>/.
    resolution: minute | hour | daily. Minute = all snapshots; hour = one per hour; daily = one per day.
    """
    if resolution not in ("minute", "hour", "daily"):
        resolution = "minute"
    if force_redownload:
        fetch_since = since_ms
    else:
        existing_dates = get_existing_depth_dates(symbol, asset_class, resolution)
        if existing_dates:
            latest_yyyymmdd = max(existing_dates)
            latest_dt = datetime.strptime(latest_yyyymmdd, "%Y%m%d").replace(tzinfo=timezone.utc)
            resume_ms = int((latest_dt + timedelta(days=1)).timestamp() * 1000)
            if resume_ms >= until_ms:
                logger.info(
                    "Depth for %s already up to date (latest %s), skipping.", symbol, latest_yyyymmdd
                )
                return
            fetch_since = max(since_ms, resume_ms)
        else:
            fetch_since = since_ms

    data = fetch_depth_range_cryptofuture(
        symbol, fetch_since, until_ms, margin_type=margin_type, data_source=data_source
    )
    if data:
        for date_str, depth_df in data.items():
            save_depth_data(symbol, date_str, depth_df, asset_class=asset_class, resolution=resolution)
        logger.info("Saved depth data for %s (%s day(s)) at %s resolution.", symbol, len(data), resolution)
    else:
        logger.debug("No depth data fetched for %s in range.", symbol)


def fetch_depth_range_cryptofuture(symbol, since_ms, until_ms, margin_type="um", data_source="binancevision"):
    """fetch order book snapshots for crypto futures from [BinanceVision](https://data.binance.vision/?prefix=data/futures/margin_type/daily/bookDepth/)
    depth data has no resolution parameter, and is always about 30s interval snapshots. you should align the timestamps to the nearest next minute boundary (e.g., 60s) to ensure consistency, and fill missing snapshots at the start of the day if needed to ensure continuity at midnight.
    
    :return: (pd.DataFrame)
    """

    if data_source != "binancevision":
        raise NotImplementedError("Only binancevision data source is supported for depth data.")

    all_dfs_dict = {}
    processed_dates = set()
    current_since = since_ms - 24*60*60*1000 # start from the previous day to fill the 0:00 snapshot of since_ms
        
    pbar = tqdm(total=until_ms - since_ms, desc=f"Fetching Depth {symbol}")

    while current_since < until_ms:
        dt = datetime.fromtimestamp(current_since / 1000, tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
        date_str_yyyymmdd = dt.strftime("%Y%m%d")
        
        if date_str in processed_dates:
            next_day = (dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
            current_since = int(next_day.timestamp() * 1000)
            continue
            
        formatted_symbol = format_symbol(symbol).upper()
        url = f"https://data.binance.vision/data/futures/{margin_type}/daily/bookDepth/{formatted_symbol}/{formatted_symbol}-bookDepth-{date_str}.zip"
        
        try:
            response = _download_with_retries(url)
            if response is None:
                raise requests.exceptions.RequestException("Depth download failed after retries")
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as f:
                        df = pd.read_csv(f, low_memory=False)
                        all_dfs_dict[date_str_yyyymmdd] = df
                        
            elif response.status_code == 404:
                logger.debug(f"No depth data for {symbol} on {date_str} (404)")
            else:
                logger.warning(f"Failed to download depth data for {symbol} on {date_str}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching depth data for {symbol} on {date_str}: {e}")
            
        processed_dates.add(date_str)
        next_day = (dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
        new_since = int(next_day.timestamp() * 1000)
        
        progress = new_since - current_since
        if progress > 0:
            pbar.update(min(progress, until_ms - current_since))
        current_since = new_since

    pbar.close()
    if not all_dfs_dict:
        return {}
    return all_dfs_dict


def save_depth_data(
    symbol: str,
    date_str: str,
    depth_data: pd.DataFrame,
    asset_class: str = "cryptofuture",
    resolution: str = "minute",
) -> None:
    """
    Save depth data to LEAN format under binance/<resolution>/<symbol>/.
    resolution: minute = all snapshots; hour = one per hour (00:00, 01:00, ...); daily = one per day (midnight).
    Format: ms_midnight, percentage, depth, notional
    """
    if depth_data.empty:
        return
    if resolution not in ("minute", "hour", "daily"):
        resolution = "minute"
    df = pd.DataFrame(depth_data)
    formatted_symbol = format_symbol(symbol)
    symbol_dir = os.path.join(
        DATA_LOCATION, asset_class, "binance", resolution, formatted_symbol
    )
    os.makedirs(symbol_dir, exist_ok=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ms_midnight"] = df["timestamp"].apply(get_ms_from_midnight)

    if resolution == "hour":
        # Keep one snapshot per hour (nearest to hour boundary)
        hour_ms = (df["ms_midnight"] // 3_600_000) * 3_600_000
        df = df.groupby(hour_ms, as_index=False).first()
    elif resolution == "daily":
        # Keep one snapshot per day (earliest of the day)
        if len(df) > 0:
            df = df.loc[[df["ms_midnight"].idxmin()]]

    lean_df = df[["ms_midnight", "percentage", "depth", "notional"]]
    zip_path = os.path.join(symbol_dir, f"{date_str}_depth.zip")
    zf_name = f"{date_str}_depth.csv"
    csv_content = lean_df.to_csv(index=False, header=False)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zf_name, csv_content)
    logger.info("Wrote %s depth to %s", resolution, zip_path)


def get_lean_df(df, asset_class, tick_type, time_col):
    """
    Returns a dataframe with columns in the order LEAN expects for the given tick_type.
    """
    if tick_type == "trade":
        # Standard Trade format: Time, Open, High, Low, Close, Volume
        return df[[time_col, "open", "high", "low", "close", "volume"]]

    elif tick_type == "quote":
        # Standard Quote format: Time, BidOpen, BidHigh, BidLow, BidClose, BidSize, AskOpen, AskHigh, AskLow, AskClose, AskSize
        # Fallback to OHLC if Bid/Ask is missing
        for side in ["bid", "ask"]:
            for price in ["open", "high", "low", "close"]:
                col = f"{side}_{price}"
                if col not in df.columns:
                    df[col] = df[price]
        
        if asset_class in ["forex", "cfd"]:
            # Forex/CFD Quote usually omits sizes in some formats, but LEAN often expects Bid/Ask OHLC
            return df[[time_col, "bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close"]]
        else:
            if "bid_size" not in df.columns: df["bid_size"] = pd.NA
            if "ask_size" not in df.columns: df["ask_size"] = pd.NA
            return df[[time_col, "bid_open", "bid_high", "bid_low", "bid_close", "bid_size", "ask_open", "ask_high", "ask_low", "ask_close", "ask_size"]]

    elif tick_type == "open_interest":
        # Standard Open Interest format: Time, Open Interest
        if "open_interest" not in df.columns:
            df["open_interest"] = df["volume"]
        return df[[time_col, "open_interest"]]

    return df[[time_col, "open", "high", "low", "close", "volume"]]


def save_ohlcv_data(symbol, ohlcv_data, resolution, asset_class="cryptofuture", tick_type="trade"):
    """
    Unified function to save OHLCV data to LEAN format for any resolution.
    """
    assert resolution in ["minute", "hour", "daily"], "Unsupported resolution for OHLCV saving"
    if not ohlcv_data:
        logger.warning(f"No OHLCV data for {symbol}")
        return
        
    df = pd.DataFrame(
        ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    formatted_symbol = format_symbol(symbol)
    base_dir = os.path.join(DATA_LOCATION, asset_class, "binance", resolution)
    
    if resolution in ["minute", "second", "tick"]:
        # Daily partitioned files in symbol subdirectory
        symbol_dir = os.path.join(base_dir, formatted_symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        df["date_str"] = df["dt"].dt.strftime("%Y%m%d")
        for date_str, group in df.groupby("date_str"):
            group = group.copy()
            group["time_val"] = group["dt"].apply(get_ms_from_midnight)
            lean_df = get_lean_df(group, asset_class, tick_type, "time_val")

            zip_path = os.path.join(symbol_dir, f"{date_str}_{tick_type}.zip")
            zf_name = f"{date_str}_{tick_type}.csv"
            csv_content = lean_df.to_csv(index=False, header=False)

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(zf_name, csv_content)
    else:
        # Consolidated files for Hour and Daily
        os.makedirs(base_dir, exist_ok=True)
        
        if resolution == "daily":
            df["time_val"] = df["dt"].dt.strftime("%Y%m%d 00:00")
            zip_path = os.path.join(base_dir, f"{formatted_symbol}_{tick_type}.zip")
        elif resolution == "hour":
            df["time_val"] = df["dt"].dt.strftime("%Y%m%d %H:%M")
            zip_path = os.path.join(base_dir, f"{formatted_symbol}_{tick_type}.zip")
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")
            
        csv_filename = f"{formatted_symbol}.csv"
        lean_df = get_lean_df(df, asset_class, tick_type, "time_val")
        
        logger.info(f"writing file to {zip_path}/{csv_filename}")
        csv_content = lean_df.to_csv(index=False, header=False)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(csv_filename, csv_content)
    
    logger.debug(f"Saved {resolution} data for {symbol}")



def save_margin_interest(symbol, rates_data):
    """
    Save funding rates as margin interest.
    LEAN requests margin interest as a direct CSV (not inside a zip):
    data/cryptofuture/binance/margin_interest/<symbol>.csv
    Format: Time (yyyyMMdd HH:mm:ss per LEAN MarginInterestRate.Reader), Interest Rate (no header).
    """
    if not rates_data:
        return

    # CCXT fetchFundingRateHistory structure:
    # {'timestamp': 1600000000000, 'fundingRate': 0.0001, ...}
    data_list = []
    for r in rates_data:
        data_list.append([r["timestamp"], r["fundingRate"]])

    df = pd.DataFrame(data_list, columns=["timestamp", "rate"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    # LEAN MarginInterestRate.Reader expects exact "yyyyMMdd HH:mm:ss" (with seconds)
    df["formatted_time"] = df["dt"].dt.strftime("%Y%m%d %H:%M:%S").str.strip()

    formatted_symbol = format_symbol(symbol)
    base_dir = os.path.join(DATA_LOCATION, "cryptofuture", "binance", "margin_interest")
    os.makedirs(base_dir, exist_ok=True)

    # LEAN expects a plain CSV at margin_interest/<symbol>.csv (not .zip)
    csv_path = os.path.join(base_dir, f"{formatted_symbol}.csv")
    # Write without headers; no trailing spaces so LEAN's Reader parses correctly
    with open(csv_path, "w") as f:
        for _, row in df.iterrows():
            t = str(row["formatted_time"]).strip()
            r = f"{row['rate']:.10g}".strip()
            f.write(f"{t},{r}\n")
    logger.debug("Saved margin interest %s", csv_path)


def fix_margin_interest_format(base_dir: str | None = None, data_folder: str | None = None) -> int:
    """
    Normalize existing margin_interest CSV files so LEAN can read them.
    LEAN's MarginInterestRate.Reader expects "yyyyMMdd HH:mm:ss". If a file
    has "yyyyMMdd HH:mm" (no seconds), this appends ":00" and rewrites.
    Returns the number of files fixed.

    Args:
        base_dir: Full path to margin_interest folder. If None, built from data_folder or DATA_LOCATION.
        data_folder: Top-level data folder (e.g. project "data"); margin_interest path is
                     data_folder/cryptofuture/binance/margin_interest. Ignored if base_dir is set.
    """
    import re
    if base_dir is None:
        root = data_folder if data_folder is not None else DATA_LOCATION
        base_dir = os.path.join(root, "cryptofuture", "binance", "margin_interest")
    if not os.path.isdir(base_dir):
        return 0
    # LEAN expects "yyyyMMdd HH:mm:ss". If file has "yyyyMMdd HH:mm", add ":00"
    time_no_seconds_re = re.compile(r"^\d{8}\s+\d{1,2}:\d{2}$")
    fixed = 0
    for name in os.listdir(base_dir):
        if not name.endswith(".csv"):
            continue
        path = os.path.join(base_dir, name)
        try:
            with open(path) as f:
                lines = f.readlines()
        except OSError as e:
            logger.warning("Could not read %s: %s", path, e)
            continue
        new_lines = []
        changed = False
        for line in lines:
            line = line.rstrip("\n\r")
            if not line.strip():
                new_lines.append(line)
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                new_lines.append(line)
                continue
            time_part, rest = parts[0].strip(), parts[1].strip()
            if time_no_seconds_re.match(time_part):
                time_part = time_part + ":00"
                changed = True
            new_lines.append(f"{time_part},{rest}")
        if changed:
            try:
                with open(path, "w") as f:
                    for ln in new_lines:
                        f.write(ln + "\n")
                fixed += 1
                logger.info("Fixed margin interest format: %s", path)
            except OSError as e:
                logger.warning("Could not write %s: %s", path, e)
    return fixed


# ---------------------------------------------------------------------------
# Cryptofuture quote (minute QuoteBar) — fetch from Binance API / Vision
# ---------------------------------------------------------------------------
# LEAN minute quote format: Time (ms since midnight), BidOpen, BidHigh, BidLow, BidClose, BidSize,
# AskOpen, AskHigh, AskLow, AskClose, AskSize. Saved under data/cryptofuture/binance/minute/<symbol>/YYYYMMDD_quote.zip
# Source: Binance Vision (futures/um/daily/bookTicker) when available; else Binance REST /fapi/v1/ticker/bookTicker
# for current snapshot only (no historical from REST).


def _normalize_bookticker_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw bookTicker CSV to columns: timestamp (ms), bid_price, bid_qty, ask_price, ask_qty."""
    df = df.copy()
    # Accept common column names (Binance stream: E=event time, b/B=bid price/qty, a/A=ask price/qty)
    col_map = {}
    for c in df.columns:
        c_lower = str(c).strip().lower()
        if c_lower in ("timestamp", "time", "e", "event_time", "t", "transacttime"):
            col_map[c] = "timestamp"
        elif c_lower in ("bid", "bidprice", "best_bid", "b"):
            col_map[c] = "bid_price"
        elif c_lower in ("bidqty", "bid_qty", "best_bid_qty", "bqty", "B"):
            col_map[c] = "bid_qty"
        elif c_lower in ("ask", "askprice", "best_ask", "a"):
            col_map[c] = "ask_price"
        elif c_lower in ("askqty", "ask_qty", "best_ask_qty", "aqty", "A"):
            col_map[c] = "ask_qty"
    df = df.rename(columns=col_map)
    # Ensure numeric and timestamp in ms
    for col in ("bid_price", "bid_qty", "ask_price", "ask_qty"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.max() < 1e12:
            ts *= 1000  # assume seconds
        df["timestamp"] = ts.astype("Int64")
    return df


def _aggregate_bookticker_to_bars(raw_df: pd.DataFrame, align_ms: int) -> pd.DataFrame:
    """
    Aggregate raw bookTicker rows into QuoteBars at the given alignment (ms since midnight).
    align_ms: 60_000 = minute, 3_600_000 = hour, 86_400_000 = daily (one bar per day).
    Returns DataFrame with: time_val, bid_open, bid_high, bid_low, bid_close, bid_size, ask_*, ask_size.
    """
    need = {"timestamp", "bid_price", "bid_qty", "ask_price", "ask_qty"}
    if not need.issubset(raw_df.columns):
        missing = need - set(raw_df.columns)
        logger.warning("aggregate_bookticker: missing columns %s", missing)
        return pd.DataFrame()
    df = raw_df.dropna(subset=["timestamp", "bid_price", "ask_price"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["ms_midnight"] = df["timestamp"] % (24 * 60 * 60 * 1000)
    df["bucket_ms"] = (np.floor(df["ms_midnight"] / align_ms) * align_ms).astype(int)
    df = df[df["bucket_ms"] < 86400000]
    agg = df.groupby("bucket_ms", as_index=False).agg(
        bid_open=("bid_price", "first"),
        bid_high=("bid_price", "max"),
        bid_low=("bid_price", "min"),
        bid_close=("bid_price", "last"),
        bid_size=("bid_qty", "last"),
        ask_open=("ask_price", "first"),
        ask_high=("ask_price", "max"),
        ask_low=("ask_price", "min"),
        ask_close=("ask_price", "last"),
        ask_size=("ask_qty", "last"),
    )
    agg["time_val"] = agg["bucket_ms"]
    return agg


def aggregate_bookticker_to_minute_bars(raw_df: pd.DataFrame, align_minute_ms: int = 60_000) -> pd.DataFrame:
    """Aggregate raw bookTicker into minute QuoteBars (LEAN format)."""
    return _aggregate_bookticker_to_bars(raw_df, align_minute_ms)


def aggregate_bookticker_to_hour_bars(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw bookTicker into hour QuoteBars (one per hour, time_val = ms since midnight)."""
    return _aggregate_bookticker_to_bars(raw_df, 3_600_000)


def aggregate_bookticker_to_daily_bars(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw bookTicker into one daily QuoteBar (time_val = 0 = midnight)."""
    return _aggregate_bookticker_to_bars(raw_df, 86_400_000)


def fetch_bookticker_range_binance_vision(
    symbol: str,
    since_ms: int,
    until_ms: int,
    margin_type: str = "um",
) -> dict[str, pd.DataFrame]:
    """
    Download cryptofuture bookTicker (best bid/ask) from Binance Vision for each day in range.

    Listing (browse) URL: https://data.binance.vision/?prefix=data/futures/um/daily/bookTicker/{SYMBOL}/
    Download (direct file) URL: https://data.binance.vision/data/futures/{um|cm}/daily/bookTicker/{SYMBOL}/{SYMBOL}-bookTicker-{YYYY-MM-DD}.zip

    Returns dict date_str (YYYYMMDD) -> DataFrame with columns timestamp, bid_price, bid_qty, ask_price, ask_qty.
    Returns empty dict when Vision returns 404/500 (historical bookTicker may not be available for USD-M).
    """
    out = {}
    start_dt = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(until_ms / 1000, tz=timezone.utc)
    sym_upper = format_symbol(symbol).upper()
    current = start_dt
    while current < end_dt:
        date_str = current.strftime("%Y-%m-%d")
        date_compact = current.strftime("%Y%m%d")
        url = (
            f"https://data.binance.vision/data/futures/{margin_type}/daily/bookTicker/"
            f"{sym_upper}/{sym_upper}-bookTicker-{date_str}.zip"
        )
        try:
            response = _download_with_retries(url)
            if response is None or response.status_code != 200:
                if response and response.status_code == 404:
                    logger.debug("No Vision bookTicker for %s on %s (404)", symbol, date_str)
                current += timedelta(days=1)
                continue
            # Binance Vision can return 200 with an XML error body (NoSuchKey) when the file does not exist
            body = response.content[:1000] if response.content else b""
            if b"<Error>" in body or b"NoSuchKey" in body or body.strip().startswith(b"<?xml"):
                logger.debug("No Vision bookTicker for %s on %s (no such key)", symbol, date_str)
                current += timedelta(days=1)
                continue
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                names = zf.namelist()
                entry = names[0] if names else None
                if not entry:
                    current += timedelta(days=1)
                    continue
                with zf.open(entry) as f:
                    raw = pd.read_csv(f, low_memory=False)
            raw = _normalize_bookticker_df(raw)
            if "timestamp" in raw.columns and "bid_price" in raw.columns and "ask_price" in raw.columns:
                out[date_compact] = raw
                logger.info("Fetched Vision bookTicker for %s on %s (%d rows)", symbol, date_str, len(raw))
        except zipfile.BadZipFile:
            logger.debug("No Vision bookTicker for %s on %s (invalid zip / no such key)", symbol, date_str)
        except Exception as e:
            logger.warning("Vision bookTicker %s %s: %s", symbol, date_str, e)
        current += timedelta(days=1)
    return out


def fetch_bookticker_rest(symbol: str, asset_class: str = "cryptofuture") -> dict | None:
    """
    Fetch current best bid/ask from Binance Futures REST: GET /fapi/v1/ticker/bookTicker.
    Returns one snapshot: {"timestamp": ms, "bid_price", "bid_qty", "ask_price", "ask_qty"} or None.
    """
    if asset_class != "cryptofuture":
        return None
    url = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
    try:
        r = requests.get(url, params={"symbol": symbol}, proxies=PROXIES, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return {
            "timestamp": int(data.get("time", datetime.now(timezone.utc).timestamp() * 1000)),
            "bid_price": float(data.get("bidPrice", 0)),
            "bid_qty": float(data.get("bidQty", 0)),
            "ask_price": float(data.get("askPrice", 0)),
            "ask_qty": float(data.get("askQty", 0)),
        }
    except Exception as e:
        logger.debug("fetch_bookticker_rest %s: %s", symbol, e)
        return None


def save_quote_bars(
    symbol: str,
    date_str: str,
    quote_bars_df: pd.DataFrame,
    resolution: str,
    asset_class: str = "cryptofuture",
) -> None:
    """
    Save QuoteBars to LEAN layout: data/<asset_class>/binance/<resolution>/<symbol>/YYYYMMDD_quote.zip
    resolution: minute | hour | daily.
    CSV (no header): Time (ms since midnight), BidOpen, BidHigh, BidLow, BidClose, BidSize,
    AskOpen, AskHigh, AskLow, AskClose, AskSize.
    """
    if quote_bars_df is None or quote_bars_df.empty:
        return
    if resolution not in ("minute", "hour", "daily"):
        logger.warning("save_quote_bars: resolution %s not in minute/hour/daily, using minute", resolution)
        resolution = "minute"
    formatted_symbol = format_symbol(symbol)
    base_dir = os.path.join(DATA_LOCATION, asset_class, "binance", resolution)
    symbol_dir = os.path.join(base_dir, formatted_symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    df = quote_bars_df.copy()
    time_col = "time_val" if "time_val" in df.columns else "ms_midnight"
    lean_df = get_lean_df(df, asset_class, "quote", time_col)
    zip_path = os.path.join(symbol_dir, f"{date_str}_quote.zip")
    zf_name = f"{date_str}_quote.csv"
    csv_content = lean_df.to_csv(index=False, header=False, float_format="%.8f")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zf_name, csv_content)
    logger.info("Wrote %s quote to %s", resolution, zip_path)


def get_existing_quote_dates(
    symbol: str,
    asset_class: str = "cryptofuture",
    resolution: str = "minute",
) -> set[str]:
    """Return set of date strings (YYYYMMDD) for which quote zip exists under binance/<resolution>/<symbol>/."""
    formatted_symbol = format_symbol(symbol)
    symbol_dir = os.path.join(DATA_LOCATION, asset_class, "binance", resolution, formatted_symbol)
    if not os.path.isdir(symbol_dir):
        return set()
    existing = set()
    for f in os.listdir(symbol_dir):
        if f.endswith("_quote.zip"):
            date_part = f.split("_")[0]
            if len(date_part) == 8 and date_part.isdigit():
                existing.add(date_part)
    return existing


def fetch_and_save_quote_range(
    symbol: str,
    since_ms: int,
    until_ms: int,
    asset_class: str = "cryptofuture",
    margin_type: str = "um",
    resolution: str = "minute",
    force_redownload: bool = False,
) -> None:
    """
    Fetch cryptofuture quote (best bid/ask) from Binance Vision bookTicker (not depth).
    Aggregate to the requested resolution (minute | hour | daily) and save under binance/<resolution>/<symbol>/.
    """
    logger.info(
        "fetch_and_save_quote_range start: symbol=%s, asset_class=%s, margin_type=%s, "
        "resolution=%s, since_ms=%s, until_ms=%s, force_redownload=%s",
        symbol,
        asset_class,
        margin_type,
        resolution,
        since_ms,
        until_ms,
        force_redownload,
    )

    if resolution not in ("minute", "hour", "daily"):
        logger.warning(
            "fetch_and_save_quote_range: unsupported resolution %s, defaulting to 'minute'",
            resolution,
        )
        resolution = "minute"

    existing = (
        get_existing_quote_dates(symbol, asset_class, resolution)
        if not force_redownload
        else set()
    )
    logger.info(
        "fetch_and_save_quote_range existing quote dates for %s (%s, %s): %s",
        symbol,
        asset_class,
        resolution,
        sorted(existing),
    )

    # 1) Binance Vision (historical bookTicker; may 404 for futures/um)
    logger.info(
        "fetch_and_save_quote_range: fetching Vision bookTicker for %s from %s to %s (margin_type=%s)",
        symbol,
        since_ms,
        until_ms,
        margin_type,
    )
    vision_data = fetch_bookticker_range_binance_vision(
        symbol, since_ms, until_ms, margin_type=margin_type
    )
    logger.info(
        "fetch_and_save_quote_range: Vision returned %d day(s) for %s: %s",
        len(vision_data),
        symbol,
        sorted(vision_data.keys()),
    )

    for date_str, raw_df in vision_data.items():
        logger.debug(
            "fetch_and_save_quote_range: processing Vision day %s for %s, raw rows=%d",
            date_str,
            symbol,
            len(raw_df) if raw_df is not None else -1,
        )

        if not force_redownload and date_str in existing:
            logger.info(
                "fetch_and_save_quote_range: skipping %s on %s because quote zip already exists "
                "(resolution=%s, force_redownload=%s)",
                symbol,
                date_str,
                resolution,
                force_redownload,
            )
            continue

        if resolution == "minute":
            bars = aggregate_bookticker_to_minute_bars(raw_df)
        elif resolution == "hour":
            bars = aggregate_bookticker_to_hour_bars(raw_df)
        else:
            bars = aggregate_bookticker_to_daily_bars(raw_df)
        logger.debug(
            "fetch_and_save_quote_range: aggregated Vision data for %s on %s into %d bar(s) "
            "(resolution=%s)",
            symbol,
            date_str,
            0 if bars is None else len(bars),
            resolution,
        )

        if bars is not None and not bars.empty:
            logger.info(
                "fetch_and_save_quote_range: saving %d quote bar(s) for %s on %s "
                "(resolution=%s, asset_class=%s)",
                len(bars),
                symbol,
                date_str,
                resolution,
                asset_class,
            )
            save_quote_bars(symbol, date_str, bars, resolution, asset_class=asset_class)
        else:
            logger.warning(
                "fetch_and_save_quote_range: no bars aggregated for %s on %s (resolution=%s); "
                "raw rows may be empty or invalid",
                symbol,
                date_str,
                resolution,
            )

    # 2) If no Vision data and user wants at least one bar: append current REST snapshot as "today" (optional)
    if not vision_data and not existing:
        logger.info(
            "fetch_and_save_quote_range: no Vision data and no existing files for %s, "
            "attempting REST snapshot fallback",
            symbol,
        )
        snap = fetch_bookticker_rest(symbol, asset_class=asset_class)
        logger.debug("fetch_and_save_quote_range: REST snapshot for %s: %s", symbol, snap)

        if snap and snap.get("bid_price") and snap.get("ask_price"):
            now = datetime.now(timezone.utc)
            today = now.strftime("%Y%m%d")
            ms_midnight = get_ms_from_midnight(now)

            if resolution == "minute":
                bucket_ms = int(np.floor(ms_midnight / 60_000) * 60_000)
            elif resolution == "hour":
                bucket_ms = int(np.floor(ms_midnight / 3_600_000) * 3_600_000)
            else:
                bucket_ms = 0

            logger.debug(
                "fetch_and_save_quote_range: building single REST bar for %s on %s at bucket_ms=%s "
                "(resolution=%s)",
                symbol,
                today,
                bucket_ms,
                resolution,
            )

            one_bar = pd.DataFrame(
                [
                    {
                        "time_val": bucket_ms,
                        "bid_open": snap["bid_price"],
                        "bid_high": snap["bid_price"],
                        "bid_low": snap["bid_price"],
                        "bid_close": snap["bid_price"],
                        "bid_size": snap["bid_qty"],
                        "ask_open": snap["ask_price"],
                        "ask_high": snap["ask_price"],
                        "ask_low": snap["ask_price"],
                        "ask_close": snap["ask_price"],
                        "ask_size": snap["ask_qty"],
                    }
                ]
            )

            save_quote_bars(symbol, today, one_bar, resolution, asset_class=asset_class)
            logger.info(
                "fetch_and_save_quote_range: saved one REST bookTicker bar for %s as %s (no Vision data)",
                symbol,
                today,
            )
        else:
            logger.warning(
                "fetch_and_save_quote_range: REST snapshot fallback for %s failed or returned empty prices",
                symbol,
            )

    logger.info(
        "fetch_and_save_quote_range done: symbol=%s, asset_class=%s, resolution=%s",
        symbol,
        asset_class,
        resolution,
    )


def _normalize_aggtrades_for_quotes(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Binance aggTrades CSV to columns suitable for quote reconstruction.

    Expected Binance futures aggTrades layout (no header, 7+ columns):
        0: agg_trade_id
        1: price
        2: qty
        3: first_trade_id
        4: last_trade_id
        5: timestamp (Unix ms or ms since midnight)
        6: is_buyer_maker (True if sell aggressor, False if buy aggressor)
        [7: is_best_match]  # often dropped
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    # If columns are positional (0,1,2,...) assign Binance aggTrades semantics.
    cols = list(df.columns)
    col_map: dict[object, str] = {}
    if all(isinstance(c, int) for c in cols):
        if len(cols) >= 2:
            col_map[cols[1]] = "price"
        if len(cols) >= 3:
            col_map[cols[2]] = "qty"
        if len(cols) >= 6:
            col_map[cols[5]] = "timestamp"
        if len(cols) >= 7:
            col_map[cols[6]] = "is_buyer_maker"
    else:
        # Fallback: try to infer from string column names.
        for c in cols:
            c_lower = str(c).strip().lower()
            if c_lower in ("price", "p"):
                col_map[c] = "price"
            elif c_lower in ("qty", "quantity", "q"):
                col_map[c] = "qty"
            elif c_lower in ("timestamp", "time", "t", "transact_time"):
                col_map[c] = "timestamp"
            elif c_lower in ("isbuyermaker", "is_buyer_maker", "buyer_maker"):
                col_map[c] = "is_buyer_maker"

    df = df.rename(columns=col_map)

    required = {"price", "qty", "timestamp"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning("normalize_aggtrades_for_quotes: missing columns %s", missing)
        return pd.DataFrame()

    # Price and quantity as numeric.
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")

    # Timestamp → ms since midnight UTC.
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    # Heuristic: values < 1 day are already "ms since midnight"; otherwise assume Unix ms.
    one_day_ms = 24 * 60 * 60 * 1000
    if ts.max(skipna=True) < one_day_ms:
        ms_midnight = ts
    else:
        ms_midnight = ts % one_day_ms
    df["ms_midnight"] = ms_midnight.astype("Int64")

    # is_buyer_maker: True = seller aggressor → trade executes at bid.
    if "is_buyer_maker" in df.columns:
        s = df["is_buyer_maker"].astype(str).str.strip().str.lower()
        df["is_buyer_maker"] = s.isin(("true", "1", "t", "yes", "y"))
    else:
        # Unknown aggressor side: mark as NA; quotes will treat such trades as neutral.
        df["is_buyer_maker"] = pd.NA

    df = df.dropna(subset=["ms_midnight", "price", "qty"])
    return df[["ms_midnight", "price", "qty", "is_buyer_maker"]]


def build_minute_quote_bars_from_aggtrades(
    aggtrades_df: pd.DataFrame,
    static_spread: float = 0.5,
) -> pd.DataFrame:
    """
    Build minute-level QuoteBars from Binance aggTrades.

    Logic:
    - Classify each trade as bid-side or ask-side using is_buyer_maker:
        * is_buyer_maker = True  -> seller aggressor -> trade occurs at best bid.
        * is_buyer_maker = False -> buyer aggressor -> trade occurs at best ask.
    - Within each minute bucket:
        * Collect bid and ask price sequences.
        * If one side is missing, infer it from the other side plus/minus a static spread.
        * Aggregate each side to OHLC and sum quantities as BidSize / AskSize.

    Returns DataFrame with LEAN QuoteBar columns:
        time_val, bid_open, bid_high, bid_low, bid_close, bid_size,
        ask_open, ask_high, ask_low, ask_close, ask_size
    """
    norm = _normalize_aggtrades_for_quotes(aggtrades_df)
    if norm.empty:
        return pd.DataFrame()

    df = norm.copy()
    df["minute_bucket"] = (df["ms_midnight"] // 60_000) * 60_000

    # Side labels based on is_buyer_maker.
    is_bid = df["is_buyer_maker"] == True  # noqa: E712
    is_ask = df["is_buyer_maker"] == False  # noqa: E712

    df["bid_price"] = np.where(is_bid, df["price"], np.nan)
    df["ask_price"] = np.where(is_ask, df["price"], np.nan)
    df["bid_qty"] = np.where(is_bid, df["qty"], 0.0)
    df["ask_qty"] = np.where(is_ask, df["qty"], 0.0)

    rows: list[dict[str, float]] = []
    for bucket_ms, g in df.groupby("minute_bucket"):
        # Skip empty buckets (should not happen after grouping).
        if g.empty:
            continue

        bid_prices = g["bid_price"].dropna()
        ask_prices = g["ask_price"].dropna()

        # Derive bid OHLC.
        if not bid_prices.empty:
            bid_open = bid_prices.iloc[0]
            bid_high = float(bid_prices.max())
            bid_low = float(bid_prices.min())
            bid_close = bid_prices.iloc[-1]
        elif not ask_prices.empty:
            # No explicit bid trades; approximate bid from ask minus spread.
            ask_open0 = ask_prices.iloc[0]
            ask_high0 = float(ask_prices.max())
            ask_low0 = float(ask_prices.min())
            ask_close0 = ask_prices.iloc[-1]
            bid_open = ask_open0 - static_spread
            bid_high = ask_high0 - static_spread
            bid_low = ask_low0 - static_spread
            bid_close = ask_close0 - static_spread
        else:
            # No trades at all in this minute.
            continue

        # Derive ask OHLC.
        if not ask_prices.empty:
            ask_open = ask_prices.iloc[0]
            ask_high = float(ask_prices.max())
            ask_low = float(ask_prices.min())
            ask_close = ask_prices.iloc[-1]
        elif not bid_prices.empty:
            # No explicit ask trades; approximate ask from bid plus spread.
            bid_open0 = bid_prices.iloc[0]
            bid_high0 = float(bid_prices.max())
            bid_low0 = float(bid_prices.min())
            bid_close0 = bid_prices.iloc[-1]
            ask_open = bid_open0 + static_spread
            ask_high = bid_high0 + static_spread
            ask_low = bid_low0 + static_spread
            ask_close = bid_close0 + static_spread
        else:
            # Already handled by previous branch; keep for clarity.
            continue

        bid_size = float(g["bid_qty"].sum())
        ask_size = float(g["ask_qty"].sum())
        total_qty = float(g["qty"].sum())

        # If both sides have zero size (e.g. no aggressor info), split volume evenly.
        if bid_size == 0.0 and ask_size == 0.0 and total_qty > 0.0:
            bid_size = total_qty / 2.0
            ask_size = total_qty / 2.0

        rows.append(
            {
                "time_val": int(bucket_ms),
                "bid_open": float(bid_open),
                "bid_high": float(bid_high),
                "bid_low": float(bid_low),
                "bid_close": float(bid_close),
                "bid_size": bid_size,
                "ask_open": float(ask_open),
                "ask_high": float(ask_high),
                "ask_low": float(ask_low),
                "ask_close": float(ask_close),
                "ask_size": ask_size,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("time_val").reset_index(drop=True)


def build_minute_quote_from_aggtrades_files(
    symbol: str,
    date_str: str,
    asset_class: str = "cryptofuture",
    static_spread: float = 0.5,
) -> None:
    """
    Rebuild LEAN-compatible minute QuoteBars for one symbol/day from local aggTrades files.

    Input:
        data/<asset_class>/binance/aggtrades/<symbol>/<YYYYMMDD>_aggtrades.zip

    Output:
        data/<asset_class>/binance/minute/<symbol>/<YYYYMMDD>_quote.zip
        with CSV columns:
            Time(ms since midnight), BidOpen, BidHigh, BidLow, BidClose, BidSize,
            AskOpen, AskHigh, AskLow, AskClose, AskSize.
    """
    formatted_symbol = format_symbol(symbol)
    agg_dir = os.path.join(
        DATA_LOCATION,
        asset_class,
        "binance",
        "aggtrades",
        formatted_symbol,
    )
    zip_name = f"{date_str}_aggtrades.zip"
    zip_path = os.path.join(agg_dir, zip_name)

    if not os.path.isfile(zip_path):
        logger.warning(
            "build_minute_quote_from_aggtrades_files: no aggTrades file for %s on %s at %s",
            symbol,
            date_str,
            zip_path,
        )
        return

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            if not names:
                logger.warning(
                    "build_minute_quote_from_aggtrades_files: empty zip for %s on %s at %s",
                    symbol,
                    date_str,
                    zip_path,
                )
                return
            entry = names[0]
            with zf.open(entry) as f:
                raw = pd.read_csv(f, header=None, low_memory=False)
    except Exception as e:
        logger.error(
            "build_minute_quote_from_aggtrades_files: failed to read %s: %s",
            zip_path,
            e,
        )
        return

    bars = build_minute_quote_bars_from_aggtrades(raw, static_spread=static_spread)
    if bars is None or bars.empty:
        logger.warning(
            "build_minute_quote_from_aggtrades_files: no quote bars built for %s on %s",
            symbol,
            date_str,
        )
        return

    save_quote_bars(
        symbol=symbol,
        date_str=date_str,
        quote_bars_df=bars,
        resolution="minute",
        asset_class=asset_class,
    )
    logger.info(
        "build_minute_quote_from_aggtrades_files: wrote minute quote for %s on %s from %s",
        symbol,
        date_str,
        zip_path,
    )