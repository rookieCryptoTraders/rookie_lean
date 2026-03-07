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
            return requests.get(url, proxies=PROXIES, timeout=30)
        except (requests.exceptions.ProxyError, OSError) as e:
            last_error = e
            # If proxy is flaky, try once without proxy in the same attempt.
            try:
                logger.warning(
                    f"Proxy-related error downloading {url} (attempt {attempt}/{_DEPTH_DOWNLOAD_MAX_RETRIES}): {e}. "
                    "Retrying once without proxy..."
                )
                return requests.get(url, timeout=30)
            except Exception as no_proxy_e:
                last_error = no_proxy_e
        except (requests.exceptions.Timeout, requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            last_error = e

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

def get_existing_depth_dates(symbol: str, asset_class: str = "cryptofuture") -> set[str]:
    """
    Return set of date strings (YYYYMMDD) for which depth data already exists on disk.
    Used for resume (skip these) or redownload (refetch and overwrite these).
    """
    formatted_symbol = format_symbol(symbol)
    symbol_dir = os.path.join(
        DATA_LOCATION, asset_class, "binance", "minute", formatted_symbol
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
    force_redownload: bool = False,
) -> None:
    """
    Fetch depth data for the given range and save to LEAN format.
    When force_redownload=False, skips dates that already have depth zip files (resume).
    When force_redownload=True, fetches and overwrites all dates in range (redownload).
    """
    if force_redownload:
        # Redownload: use full range as-is
        fetch_since = since_ms
    else:
        # Resume: skip existing dates
        existing_dates = get_existing_depth_dates(symbol, asset_class)
        if existing_dates:
            # Move start past the latest existing date
            latest_yyyymmdd = max(existing_dates)
            latest_dt = datetime.strptime(latest_yyyymmdd, "%Y%m%d").replace(tzinfo=timezone.utc)
            resume_ms = int((latest_dt + timedelta(days=1)).timestamp() * 1000)
            if resume_ms >= until_ms:
                logger.info(
                    f"Depth for {symbol} already up to date (latest {latest_yyyymmdd}), skipping."
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
            save_depth_data(symbol, date_str, depth_df, asset_class=asset_class)
        logger.info(f"Saved depth data for {symbol} ({len(data)} day(s)).")
    else:
        logger.debug(f"No depth data fetched for {symbol} in range.")


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


def save_depth_data(symbol,date_str:str, depth_data:pd.DataFrame, asset_class="cryptofuture"):
    """
    Save depth data to LEAN format.
    Format: ms_midnight, percentage, depth, notional
    """
    if depth_data.empty:
        return
    df = pd.DataFrame(depth_data)

    formatted_symbol = format_symbol(symbol)
    symbol_dir = os.path.join(
        DATA_LOCATION, asset_class, "binance", "minute", formatted_symbol
    )
    os.makedirs(symbol_dir, exist_ok=True)

    # LEAN depth format: ms_midnight, percentage, depth, notional
    df["timestamp"]=pd.to_datetime(df["timestamp"], utc=True)
    df["ms_midnight"] = df["timestamp"].apply(get_ms_from_midnight)

    # Filename: YYYYMMDD_depth.zip
    zip_path = os.path.join(symbol_dir, f"{date_str}_depth.zip")
    zf_name = f"{date_str}_depth.csv"
    
    lean_df = df[["ms_midnight", "percentage", "depth", "notional"]]
    
    logger.info(f"writing file to {zip_path}/{zf_name}")
    csv_content = lean_df.to_csv(index=False, header=False)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zf_name, csv_content)


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
        else: # hour
            df["time_val"] = df["dt"].dt.strftime("%Y%m%d %H:%M")
            primary_tick = "quote" if asset_class in ["forex", "cfd"] else "trade"
            suffix = f"_{tick_type}" if tick_type != primary_tick else ""
            zip_path = os.path.join(base_dir, f"{formatted_symbol}{suffix}.zip")
            
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
    LEAN format for margin_rate: Time (YYYYMMDD HH:mm), Interest Rate
    """
    if not rates_data:
        return
        
    # CCXT fetchFundingRateHistory structure:
    # {'timestamp': 1600000000000, 'fundingRate': 0.0001, ...}
    
    data_list = []
    for r in rates_data:
        data_list.append([r['timestamp'], r['fundingRate']])
        
    df = pd.DataFrame(data_list, columns=["timestamp", "rate"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    # Format: YYYYMMDD HH:mm
    df["formatted_time"] = df["dt"].dt.strftime("%Y%m%d %H:%M")
    
    formatted_symbol = format_symbol(symbol)
    # Per instructions: data/cryptofuture/binance/margin_interest/
    base_dir = os.path.join(DATA_LOCATION, "cryptofuture", "binance", "margin_interest")
    os.makedirs(base_dir, exist_ok=True)
    
    zip_path = os.path.join(base_dir, f"{formatted_symbol}.zip")
    csv_filename = f"{formatted_symbol}.csv"
    
    # LEAN Margin Rate file: Time, Interest Rate
    lean_df = df[["formatted_time", "rate"]]
    csv_content = lean_df.to_csv(index=False, header=False)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_filename, csv_content)
    logger.debug(f"Saved margin interest {zip_path}")


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


def aggregate_bookticker_to_minute_bars(raw_df: pd.DataFrame, align_minute_ms: int = 60_000) -> pd.DataFrame:
    """
    Aggregate raw bookTicker rows (timestamp, bid_price, bid_qty, ask_price, ask_qty) into minute QuoteBars.
    Returns DataFrame with: time_val (ms since midnight), bid_open, bid_high, bid_low, bid_close, bid_size,
    ask_open, ask_high, ask_low, ask_close, ask_size (LEAN minute quote format).
    """
    need = {"timestamp", "bid_price", "bid_qty", "ask_price", "ask_qty"}
    if not need.issubset(raw_df.columns):
        missing = need - set(raw_df.columns)
        logger.warning("aggregate_bookticker_to_minute_bars: missing columns %s", missing)
        return pd.DataFrame()
    df = raw_df.dropna(subset=["timestamp", "bid_price", "ask_price"]).copy()
    if df.empty:
        return pd.DataFrame()
    # Align to minute boundary (ms since midnight of the day)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
    df["ms_midnight"] = df["timestamp"] % (24 * 60 * 60 * 1000)
    df["minute_ms"] = (np.floor(df["ms_midnight"] / align_minute_ms) * align_minute_ms).astype(int)
    df = df[df["minute_ms"] < 86400000]
    agg = df.groupby("minute_ms", as_index=False).agg(
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
    agg["time_val"] = agg["minute_ms"]
    return agg


def fetch_bookticker_range_binance_vision(
    symbol: str,
    since_ms: int,
    until_ms: int,
    margin_type: str = "um",
) -> dict[str, pd.DataFrame]:
    """
    Download cryptofuture bookTicker (best bid/ask) from Binance Vision for each day in range.
    URL: https://data.binance.vision/data/futures/{um|cm}/daily/bookTicker/{SYMBOL}/{SYMBOL}-bookTicker-{date}.zip
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


def save_quote_bars_minute(
    symbol: str,
    date_str: str,
    quote_bars_df: pd.DataFrame,
    asset_class: str = "cryptofuture",
) -> None:
    """
    Save minute QuoteBars to LEAN layout: data/cryptofuture/binance/minute/<symbol>/YYYYMMDD_quote.zip
    CSV (no header): Time (ms since midnight), BidOpen, BidHigh, BidLow, BidClose, BidSize,
    AskOpen, AskHigh, AskLow, AskClose, AskSize.
    """
    if quote_bars_df is None or quote_bars_df.empty:
        return
    formatted_symbol = format_symbol(symbol)
    base_dir = os.path.join(DATA_LOCATION, asset_class, "binance", "minute")
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
    logger.info("Wrote minute quote to %s", zip_path)


def get_existing_quote_dates(symbol: str, asset_class: str = "cryptofuture") -> set[str]:
    """Return set of date strings (YYYYMMDD) for which minute quote zip exists under binance/minute/<symbol>/."""
    formatted_symbol = format_symbol(symbol)
    symbol_dir = os.path.join(DATA_LOCATION, asset_class, "binance", "minute", formatted_symbol)
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
    force_redownload: bool = False,
) -> None:
    """
    Fetch cryptofuture quote from Binance (Vision first, then REST for current snapshot if needed),
    aggregate to minute QuoteBars, and save in QuantConnect format under data/cryptofuture/binance/minute/<symbol>/.

    Minute resolution: one row per minute with Bid O/H/L/C, BidSize, Ask O/H/L/C, AskSize (LEAN QuoteBar).
    """
    existing = get_existing_quote_dates(symbol, asset_class) if not force_redownload else set()
    start_dt = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(until_ms / 1000, tz=timezone.utc)

    # 1) Binance Vision (historical bookTicker; may 404 for futures/um)
    vision_data = fetch_bookticker_range_binance_vision(symbol, since_ms, until_ms, margin_type=margin_type)
    for date_str, raw_df in vision_data.items():
        if not force_redownload and date_str in existing:
            continue
        bars = aggregate_bookticker_to_minute_bars(raw_df)
        if not bars.empty:
            save_quote_bars_minute(symbol, date_str, bars, asset_class=asset_class)

    # 2) If no Vision data and user wants at least one bar: append current REST snapshot as "today" (optional)
    if not vision_data and not existing:
        snap = fetch_bookticker_rest(symbol, asset_class=asset_class)
        if snap and snap.get("bid_price") and snap.get("ask_price"):
            now = datetime.now(timezone.utc)
            today = now.strftime("%Y%m%d")
            ms_midnight = get_ms_from_midnight(now)
            minute_ms = int(np.floor(ms_midnight / 60_000) * 60_000)
            one_bar = pd.DataFrame([{
                "time_val": minute_ms,
                "bid_open": snap["bid_price"], "bid_high": snap["bid_price"], "bid_low": snap["bid_price"], "bid_close": snap["bid_price"],
                "bid_size": snap["bid_qty"],
                "ask_open": snap["ask_price"], "ask_high": snap["ask_price"], "ask_low": snap["ask_price"], "ask_close": snap["ask_price"],
                "ask_size": snap["ask_qty"],
            }])
            save_quote_bars_minute(symbol, today, one_bar, asset_class=asset_class)
            logger.info("Saved one REST bookTicker bar for %s as %s (no Vision data)", symbol, today)