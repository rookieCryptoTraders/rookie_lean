"""
Data loading utilities for as_winningrate strategy.
Loads trade, quote, hourly OHLCV, and depth data from ccxt_data_fetch / local storage.
"""

from datetime import date, datetime, timedelta, timezone
import io
import json
import logging
import os
import sys
import time
import zipfile

from AlgorithmImports import *
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from config import (
    ASSET_CLASS,
    BASE_DATA_PATH,
    CUSTOM_DEPTH_MAP,
    CUSTOM_QUOTE_MAP,
    CUSTOM_RESOLUTION_FOLDER,
    DATA_LOCATION,
    EXCHANGE,
    PROXIES,
)

logger = logging.getLogger(__name__)


# LEAN's data folder (set by engine; use for custom data so Docker mount paths resolve).
# See https://www.quantconnect.com/docs/v2/lean-cli/datasets/custom-data
def _lean_data_folder():
    try:
        from QuantConnect import Globals

        folder = getattr(Globals, "data_folder", None) or getattr(
            Globals, "DataFolder", None
        )
        if folder:
            return folder
    except Exception:
        pass
    return DATA_LOCATION


def format_symbol(symbol):
    """Convert symbols like BTC/USDT, BTC/USDT:USDT, or BTCUSDT.depth to btcusdt."""
    # Strip custom data suffixes first
    for suffix in (".depth", ".quote"):
        if symbol.lower().endswith(suffix):
            symbol = symbol[: -len(suffix)]
    return symbol.split(":")[0].replace("/", "").lower()


class CryptoFutureQuoteData(PythonData):
    """
    Custom quote data for Binance crypto futures.
    Manual extraction to /tmp/ to fix LEAN 2.5 ZIP bugs.
    """

    def GetSource(self, config, date, is_live_mode):
        return self.get_source(config, date, is_live_mode)

    def Reader(self, config, line, date, is_live_mode):
        return self.reader(config, line, date, is_live_mode)

    def get_source(self, config, date, is_live_mode):
        if getattr(config, "tick_type", None) != TickType.TRADE:
            return None

        ticker = format_symbol(config.symbol.value)
        if is_live_mode:
            return SubscriptionDataSource(
                f"http://localhost:8080/quote/{ticker}",
                SubscriptionTransportMedium.REMOTE_FILE,
            )

        # Backtest - ZIP extraction
        base = "/data"
        try:
            from QuantConnect import Globals

            base = Globals.DataFolder or Globals.data_folder or "/data"
        except:
            base = _lean_data_folder()

        date_str = date.strftime("%Y%m%d")
        zip_path = os.path.join(
            base,
            "custom",
            "cryptofuture-quote",
            ticker,
            "minute",
            f"{date_str}_quote.zip",
        ).replace("\\", "/")

        # DUMMY FILE mechanism to avoid recursion if missing
        dummy_file = os.path.join("/tmp", "dummy_empty.csv")
        if not os.path.exists(dummy_file):
            with open(dummy_file, "w") as f:
                f.write("time,bid,ask,bsize,asize\n")

        if not os.path.exists(zip_path):
            return SubscriptionDataSource(
                dummy_file, SubscriptionTransportMedium.LOCAL_FILE, FileFormat.CSV
            )

        # Extract to cache
        cache_dir = os.path.join("/tmp", "lean_quote_cache", ticker)
        os.makedirs(cache_dir, exist_ok=True)
        csv_filename = f"{date_str}_quote.csv"
        extracted_csv = os.path.join(cache_dir, csv_filename)

        if not os.path.exists(extracted_csv):
            import zipfile

            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extract(csv_filename, cache_dir)
            except:
                return SubscriptionDataSource(
                    dummy_file, SubscriptionTransportMedium.LOCAL_FILE, FileFormat.CSV
                )

        return SubscriptionDataSource(
            extracted_csv, SubscriptionTransportMedium.LOCAL_FILE, FileFormat.CSV
        )

    def reader(self, config, line, date, is_live_mode):
        parts = line.split(",")
        if len(parts) < 4 or line.startswith("time"):  # Skip header
            return None
        try:
            ms_midnight = int(parts[0])
            quote = CryptoFutureQuoteData()
            quote.symbol = config.symbol
            quote.time = date + timedelta(milliseconds=ms_midnight)
            quote.end_time = quote.time + timedelta(milliseconds=60000)
            quote.bid = float(parts[1])
            quote.ask = float(parts[2])
            quote.bid_size = float(parts[3])
            quote.ask_size = float(parts[4])
            return quote
        except:
            return None


class CryptoFutureDepthData(PythonData):
    """
    Robust implementation of custom depth data for LEAN.
    Backtest (CSV inside ZIP): aggregates multiple rows (levels) per timestamp.
    Live (JSON from REST/WebSocket): parses snapshot directly.
    """

    def GetSource(self, config, date, is_live_mode):
        """PascalCase alias for C# bridge."""
        return self.get_source(config, date, is_live_mode)

    def get_source(self, config, date, is_live_mode):
        """
        Source builder. Uses dummy fallback for missing files to kill LEAN 2.5 recursions.
        """
        if getattr(config, "tick_type", None) != TickType.TRADE:
            return None

        ticker = format_symbol(config.symbol.value)
        if is_live_mode:
            return SubscriptionDataSource(
                f"http://localhost:8080/depth/{ticker}",
                SubscriptionTransportMedium.REMOTE_FILE,
            )

        # Backtest
        base = "/data"
        try:
            from QuantConnect import Globals

            base = Globals.DataFolder or Globals.data_folder or "/data"
        except:
            base = _lean_data_folder()

        date_str = date.strftime("%Y%m%d")
        zip_rel = os.path.join(
            "custom", "cryptofuture-depth", ticker, "minute", f"{date_str}_depth.zip"
        )
        zip_path = os.path.join(base, zip_rel).replace("\\", "/")

        dummy_file = os.path.join("/tmp", "dummy_empty_depth.csv")
        if not os.path.exists(dummy_file):
            with open(dummy_file, "w") as f:
                f.write("time,side,price,amount,qty\n")

        if not os.path.exists(zip_path):
            return SubscriptionDataSource(
                dummy_file, SubscriptionTransportMedium.LOCAL_FILE, FileFormat.CSV
            )

        # Extraction logic
        cache_dir = os.path.join("/tmp", "lean_depth_cache", ticker)
        os.makedirs(cache_dir, exist_ok=True)
        csv_filename = f"{date_str}_depth.csv"
        extracted_csv = os.path.join(cache_dir, csv_filename)

        if not os.path.exists(extracted_csv):
            import zipfile

            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extract(csv_filename, cache_dir)
            except:
                return SubscriptionDataSource(
                    dummy_file, SubscriptionTransportMedium.LOCAL_FILE, FileFormat.CSV
                )

        return SubscriptionDataSource(
            extracted_csv, SubscriptionTransportMedium.LOCAL_FILE, FileFormat.CSV
        )

    def Reader(self, config, line, date, is_live_mode):
        """PascalCase alias for C# bridge."""
        return self.reader(config, line, date, is_live_mode)

    def reader(self, config, line, date, is_live_mode):
        """
        Parses depth CSV into aggregated DepthSnapshot objects.
        Uses timestamp-change detection to handle variable row counts
        (10 or 12 rows per snapshot depending on whether ±0.2% levels exist).
        """
        if not line or not line.strip():
            return None

        # Skip header lines
        if line.startswith("time") or line.startswith("#"):
            return None

        if is_live_mode:
            try:
                payload = json.loads(line)
                obj = CryptoFutureDepthData()
                obj.symbol = config.symbol
                obj.time = date
                obj.percentages = payload.get("percentages", [])
                obj.depths = payload.get("depths", [])
                obj.notionals = payload.get("notionals", [])
                obj.value = sum(obj.depths)
                return obj
            except Exception:
                return None

        # BACKTEST CSV format: ms_midnight, percentage, qty, notional_usd
        parts = line.split(",")
        if len(parts) < 3:
            return None

        try:
            ms_midnight = int(parts[0])
            pct = float(parts[1])
            depth = float(parts[2])
            notional = float(parts[3]) if len(parts) > 3 else 0.0
        except (ValueError, IndexError):
            return None

        sym_key = config.symbol.value
        # One depth snapshot = 12 lines (percentages -5,-4,-3,-2,-1,-0.2,0.2,1,2,3,4,5 or similar)
        DEPTH_LEVELS_FULL = 12

        result_obj = None

        # If we have a buffer from a DIFFERENT timestamp, emit it only when it's a full snapshot (12 levels)
        if sym_key in _DEPTH_SNAPSHOT_BUFFER:
            prev_buf = _DEPTH_SNAPSHOT_BUFFER[sym_key]
            prev_ms = prev_buf["ms"]
            if prev_ms != ms_midnight:
                if len(prev_buf["p"]) >= 10:  # full snapshot (10–12 levels)
                    result_obj = self._emit_snapshot(config, date, prev_buf)
                del _DEPTH_SNAPSHOT_BUFFER[sym_key]

        # Start or reuse buffer for current timestamp
        if sym_key not in _DEPTH_SNAPSHOT_BUFFER:
            _DEPTH_SNAPSHOT_BUFFER[sym_key] = {
                "ms": ms_midnight,
                "p": [],
                "d": [],
                "n": [],
            }

        _DEPTH_SNAPSHOT_BUFFER[sym_key]["p"].append(pct)
        _DEPTH_SNAPSHOT_BUFFER[sym_key]["d"].append(depth)
        _DEPTH_SNAPSHOT_BUFFER[sym_key]["n"].append(notional)

        buf = _DEPTH_SNAPSHOT_BUFFER[sym_key]
        # Emit current snapshot as soon as we have 12 lines for this timestamp
        if len(buf["p"]) >= DEPTH_LEVELS_FULL:
            result_obj = self._emit_snapshot(config, date, buf)
            del _DEPTH_SNAPSHOT_BUFFER[sym_key]

        return result_obj

    @staticmethod
    def _emit_snapshot(config, date, buf):
        """Build a CryptoFutureDepthData object from the accumulated buffer."""
        combined = sorted(zip(buf["p"], buf["d"], buf["n"]), key=lambda x: x[0])
        p, d, n = zip(*combined)

        obj = CryptoFutureDepthData()
        obj.symbol = config.symbol

        align_ms = 60000
        ms_midnight = buf["ms"]
        snapped_t = date + timedelta(
            milliseconds=int(np.ceil(ms_midnight / align_ms) * align_ms)
        )
        obj.end_time = snapped_t
        obj.time = snapped_t - timedelta(milliseconds=align_ms)

        obj.percentages = list(p)
        obj.depths = list(d)
        obj.notionals = list(n)
        obj.value = sum(obj.depths)
        return obj

    def supported_resolutions(self):
        return [Resolution.MINUTE, Resolution.HOUR]

    def default_resolution(self):
        return Resolution.MINUTE


def _resolve_data_dir(data_dir=None):
    """Resolve data directory; default is DATA_LOCATION (repo data folder)."""
    if data_dir is not None:
        return os.path.normpath(os.path.abspath(data_dir))
    return DATA_LOCATION


def load_trade_data(ticker, start_date, end_date, interval=None, data_dir=None):
    """Load minute trade (OHLCV) data, with optional resampling."""
    resolved = _resolve_data_dir(data_dir)
    ticker_dir = os.path.join(resolved, ticker.lower())
    if not os.path.exists(ticker_dir):
        logger.warning("Data directory %s does not exist.", ticker_dir)
        return None
    logger.info("Loading trade data for %s from %s", ticker, ticker_dir)

    all_dfs = []
    files = sorted([f for f in os.listdir(ticker_dir) if f.endswith("_trade.zip")])
    for f in files:
        date_str = f.split("_")[0]
        try:
            file_date = datetime.strptime(date_str, "%Y%m%d")
            if start_date <= file_date <= end_date:
                df = pd.read_csv(
                    os.path.join(ticker_dir, f), header=None, compression="zip"
                )
                df.columns = ["ms", "open", "high", "low", "close", "volume"]
                df["time"] = file_date + pd.to_timedelta(df["ms"], unit="ms")
                df.set_index("time", inplace=True)
                all_dfs.append(df[["open", "high", "low", "close", "volume"]])
        except Exception as e:
            raise e

    if not all_dfs:
        return None

    df = pd.concat(all_dfs).sort_index().drop_duplicates()

    if interval:
        df = (
            df.resample(interval)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

    return df


def load_quote_data(ticker, start_date, end_date, interval=None, data_dir=None):
    """
    Load minute quote (L1 bid/ask) data, with optional resampling.

    Supports:
    - Custom layout (L1 from ccxt_data_fetch): data/custom/cryptofuture-quote/<ticker>/minute/<date>_quote.zip
      CSV: ms_midnight, bid, ask, bid_size, ask_size (5 columns).
    - Legacy layout: <data_dir>/<ticker>/_quote.zip with 11 columns (QuoteBar: bid/ask OHLC + sizes).

    Returns DataFrame with bid_open, bid_high, bid_low, bid_close, bid_size, ask_open, ..., ask_size.
    """
    resolved = _resolve_data_dir(data_dir)
    ticker_lower = ticker.lower() if isinstance(ticker, str) else str(ticker).lower()
    # 1) Try custom L1 quote path first
    custom_quote_dir = os.path.join(
        resolved, "custom", CUSTOM_QUOTE_MAP, ticker_lower, CUSTOM_RESOLUTION_FOLDER
    )
    quote_dir = (
        custom_quote_dir
        if os.path.isdir(custom_quote_dir)
        else os.path.join(resolved, ticker_lower)
    )
    if not os.path.exists(quote_dir):
        logger.warning("Quote data directory %s does not exist.", quote_dir)
        return None
    logger.info("Loading quote data for %s from %s", ticker, quote_dir)
    all_dfs = []
    files = sorted([f for f in os.listdir(quote_dir) if f.endswith("_quote.zip")])
    for f in files:
        date_str = f.split("_")[0]
        try:
            file_date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            continue
        if not (start_date <= file_date <= end_date):
            continue
        path = os.path.join(quote_dir, f)
        try:
            df = pd.read_csv(path, header=None, compression="zip")
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
            continue
        ncol = len(df.columns)
        if ncol >= 11:
            df.columns = [
                "ms",
                "bid_open",
                "bid_high",
                "bid_low",
                "bid_close",
                "bid_size",
                "ask_open",
                "ask_high",
                "ask_low",
                "ask_close",
                "ask_size",
            ][:ncol]
        elif ncol >= 5:
            # L1 snapshot: ms_midnight, bid, ask, bid_size, ask_size -> synthesize OHLC
            df.columns = ["ms", "bid", "ask", "bid_size", "ask_size"][:ncol]
            df["bid_open"] = df["bid_high"] = df["bid_low"] = df["bid_close"] = df[
                "bid"
            ]
            df["ask_open"] = df["ask_high"] = df["ask_low"] = df["ask_close"] = df[
                "ask"
            ]
        else:
            continue
        df["time"] = file_date + pd.to_timedelta(df["ms"], unit="ms")
        df.set_index("time", inplace=True)
        cols = [
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
            "bid_size",
            "ask_open",
            "ask_high",
            "ask_low",
            "ask_close",
            "ask_size",
        ]
        all_dfs.append(df[[c for c in cols if c in df.columns]])
    if not all_dfs:
        return None
    df = pd.concat(all_dfs).sort_index().drop_duplicates()
    if interval:
        df = (
            df.resample(interval)
            .agg(
                {
                    "bid_open": "first",
                    "bid_high": "max",
                    "bid_low": "min",
                    "bid_close": "last",
                    "bid_size": "sum",
                    "ask_open": "first",
                    "ask_high": "max",
                    "ask_low": "min",
                    "ask_close": "last",
                    "ask_size": "sum",
                }
            )
            .dropna()
        )
    return df


def load_hourly_ohlcv(ticker, start_date, end_date, data_dir=None):
    """
    Load hourly OHLCV from 1h directory (ccxt_data_fetch format).
    Falls back to resampling minute data if 1h files missing.
    """
    resolved = _resolve_data_dir(data_dir)
    base_dir = os.path.dirname(resolved)
    hourly_dir = os.path.join(base_dir, "1h")
    ticker_key = ticker.lower()
    if not ticker_key.endswith("usdt"):
        ticker_key = ticker_key + "usdt"
    paths = [
        os.path.join(hourly_dir, f"{ticker_key}.csv"),
        os.path.join(hourly_dir, f"{ticker_key}.zip"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                df = (
                    pd.read_csv(p)
                    if p.endswith(".csv")
                    else pd.read_csv(p, compression="zip")
                )
                time_col = "time" if "time" in df.columns else df.columns[0]
                df[time_col] = pd.to_datetime(df[time_col], utc=True)
                df = df.set_index(time_col)
                mask = (df.index >= pd.Timestamp(start_date)) & (
                    df.index <= pd.Timestamp(end_date)
                )
                return df.loc[mask].sort_index().drop_duplicates()
            except Exception as e:
                logger.warning("Failed to load hourly %s: %s", p, e)
    logger.info("No hourly data for %s; falling back to minute resample.", ticker)
    return load_trade_data(
        ticker.upper(), start_date, end_date, interval="1h", data_dir=data_dir
    )


# Module-level cache: (symbol, date_str) -> last_snapshot DataFrame; avoids reloading prev day
_DEPTH_LAST_SNAPSHOT_CACHE = {}


def _load_depth_day_from_disk_only(
    symbol: str, date_str: str, data_base: str | None = None
) -> pd.DataFrame:
    """
    Load one day's depth from disk (custom layout). No fetch - returns empty if file missing.
    Use in LEAN/backtest when fetch_depth_range_cryptofuture is not available.
    """
    base = data_base or DATA_LOCATION
    formatted_symbol = format_symbol(symbol)
    symbol_dir = os.path.join(
        base, "custom", CUSTOM_DEPTH_MAP, formatted_symbol, CUSTOM_RESOLUTION_FOLDER
    )
    zip_path = os.path.join(symbol_dir, f"{date_str}_depth.zip")
    if not os.path.exists(zip_path):
        return pd.DataFrame()
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_name = f"{date_str}_depth.csv"
            if csv_name not in zf.namelist():
                return pd.DataFrame()
            with zf.open(csv_name) as f:
                return pd.read_csv(
                    f, names=["ms_midnight", "percentage", "depth", "notional"]
                )
    except Exception as e:
        logger.debug("_load_depth_day_from_disk_only %s %s: %s", symbol, date_str, e)
        return pd.DataFrame()


# Module-level buffer for aggregating 10 depth rows into one snapshot
_DEPTH_SNAPSHOT_BUFFER: dict[tuple, dict] = {}


def _load_depth_data_pandas(
    symbol: str,
    date_str: str,
    pivot=False,
    ffill=False,
    align_ms=None,
    fill_first=True,
    asset_class="cryptofuture",
    exchange="binance",
    resolution="minute",
    prev_last_snapshot=None,
):
    """
    Load depth data for a single day.

    Args:
        symbol (str): Symbol name (e.g., BTCUSDT)
        date_str (str): Date string in YYYYMMDD format.
        pivot (bool): Whether to pivot the data. Defaults to False.
        ffill (bool): Whether to forward fill missing data. Defaults to False.
        align_ms (int): Alignment interval in milliseconds. Defaults to None.
        fill_first (bool): Whether to fill the first snapshot with previous day's last. Defaults to True.
        asset_class (str): Asset class directory. Defaults to "cryptofuture".
        exchange (str): Exchange directory. Defaults to 'binance'.
        prev_last_snapshot (pd.DataFrame, optional): Cached last snapshot from previous day; used instead of loading prev file.
    """
    assert resolution == "minute", (
        "Currently only minute resolution is supported for depth data loading."
    )

    def _load_day_file(d_str):
        formatted_symbol = format_symbol(symbol)
        # Custom layout: data/custom/cryptofuture-depth/<symbol>/minute/
        symbol_dir = os.path.join(
            DATA_LOCATION,
            "custom",
            CUSTOM_DEPTH_MAP,
            formatted_symbol,
            CUSTOM_RESOLUTION_FOLDER,
        )
        zip_path = os.path.join(symbol_dir, f"{d_str}_depth.zip")

        if not os.path.exists(zip_path):
            # Attempt to download (saves into custom layout when implemented)
            if exchange == "binance" and asset_class == "cryptofuture":
                try:
                    dt_obj = datetime.strptime(d_str, "%Y%m%d").replace(
                        tzinfo=timezone.utc
                    )
                    since_ms = int(dt_obj.timestamp() * 1000)
                    until_ms = since_ms + 86400000
                    depth_data_dict = fetch_depth_range_cryptofuture(
                        symbol, since_ms, until_ms
                    )
                    if depth_data_dict:
                        for day_str, day_df in depth_data_dict.items():
                            save_depth_data(
                                symbol, day_str, day_df, asset_class=asset_class
                            )

                    if d_str not in (depth_data_dict or {}):
                        return pd.DataFrame()
                except Exception as e:
                    logger.error(
                        f"Error downloading depth data for {symbol} on {d_str}: {e}"
                    )
                    return pd.DataFrame()
            else:
                return pd.DataFrame()

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                csv_name = f"{d_str}_depth.csv"
                if csv_name in zf.namelist():
                    with zf.open(csv_name) as f:
                        df = pd.read_csv(
                            f, names=["ms_midnight", "percentage", "depth", "notional"]
                        )
                        # df['ms_original'] = df['ms_midnight']+datetime.strptime(d_str, "%Y%m%d").timestamp()*1000
                        return df
        except Exception as e:
            logger.error(f"Error reading depth data file {zip_path}: {e}")
            return pd.DataFrame()
        return pd.DataFrame()

    df = _load_day_file(date_str)

    if fill_first and df is not None and not df.empty:
        if prev_last_snapshot is not None and not prev_last_snapshot.empty:
            last_snapshot = prev_last_snapshot.copy()
            last_snapshot.index = pd.RangeIndex(len(last_snapshot))
            last_snapshot = last_snapshot.rename_axis(None)
            last_snapshot["ms_midnight"] = 0
            df = pd.concat([last_snapshot, df], ignore_index=True)
        else:
            prev_date = (
                datetime.strptime(date_str, "%Y%m%d") - timedelta(days=1)
            ).strftime("%Y%m%d")
            prev_df = _load_day_file(prev_date)
            if not prev_df.empty:
                last_ms = prev_df["ms_midnight"].max()
                last_snapshot = prev_df[prev_df["ms_midnight"] == last_ms].copy()
                last_snapshot["ms_midnight"] = 0
                df = pd.concat([last_snapshot, df], ignore_index=True)

    if df is None or df.empty:
        return pd.DataFrame()

    # Initial sort and drop duplicates
    df = df.sort_values(["ms_midnight", "percentage"]).drop_duplicates(
        ["ms_midnight", "percentage"], keep="last"
    )

    # Cache last snapshot for _load_depth_data_range to avoid reloading prev day
    last_ms = (
        df[df["ms_midnight"] > 0]["ms_midnight"].max()
        if (df["ms_midnight"] == 0).any()
        else df["ms_midnight"].max()
    )
    if not pd.isna(last_ms):
        _DEPTH_LAST_SNAPSHOT_CACHE[(format_symbol(symbol), date_str)] = df[
            df["ms_midnight"] == last_ms
        ][["ms_midnight", "percentage", "depth", "notional"]].copy()

    if align_ms is not None:
        # Align to the nearest NEXT boundary (ceil)
        df["ms_midnight"] = (np.ceil(df["ms_midnight"] / align_ms) * align_ms).astype(
            int
        )
        df = df[df["ms_midnight"] < 86400000]
        df = df.sort_values(["ms_midnight", "percentage"]).drop_duplicates(
            ["ms_midnight", "percentage"], keep="last"
        )

        ms_range = np.arange(0, 86400000, align_ms)
        if pivot:
            df_depth = df.pivot(
                index="ms_midnight", columns="percentage", values="depth"
            )
            df_depth = df_depth.reindex(ms_range)
            df_depth.columns = [f"depth_{p}%" for p in df_depth.columns]
            df_notional = df.pivot(
                index="ms_midnight", columns="percentage", values="notional"
            )
            df_notional = df_notional.reindex(ms_range)
            df_notional.columns = [f"notional_{p}%" for p in df_notional.columns]
            df = pd.concat([df_depth, df_notional], axis=1, join="outer")
        else:
            percentages = df["percentage"].unique()
            new_index = pd.MultiIndex.from_product(
                [ms_range, percentages], names=["ms_midnight", "percentage"]
            )
            df = (
                df.set_index(["ms_midnight", "percentage"])
                .reindex(new_index)
                .reset_index()
            )
    elif pivot:
        df_depth = df.pivot(index="ms_midnight", columns="percentage", values="depth")
        df_depth.columns = [f"depth_{p}%" for p in df_depth.columns]
        df_notional = df.pivot(
            index="ms_midnight", columns="percentage", values="notional"
        )
        df_notional.columns = [f"notional_{p}%" for p in df_notional.columns]
        df = pd.concat([df_depth, df_notional], axis=1, join="outer")

    if ffill:
        if pivot:
            df = df.ffill()
        else:
            df = df.sort_values(["percentage", "ms_midnight"])
            df[["depth", "notional"]] = df.groupby("percentage")[
                ["depth", "notional"]
            ].ffill()
            df = df.sort_values(["ms_midnight", "percentage"])

    df = df.reset_index().set_index("ms_midnight")

    return df


def _load_depth_data_range_pandas(
    symbol: str,
    resolution: str,
    start: datetime | int,
    end: datetime | int,
    pivot=False,
    ffill=False,
    align_ms=None,
    fill_first=True,
    asset_class="cryptofuture",
    exchange="binance",
):
    """Load depth data for a date range using _load_depth_data_pandas per day. Caches last snapshot to avoid reloading.

    Args:
        symbol (str): Symbol name (e.g., BTCUSDT)
        resolution (str): Data resolution ('minute', 'hour', 'daily')
        start (datetime | int): Start timestamp or datetime
        end (datetime | int): End timestamp or datetime
        pivot (bool, optional): Whether to pivot the data. Defaults to False.
        ffill (bool, optional): Whether to forward fill missing data. Defaults to False.
        align_ms (int, optional): Alignment interval in milliseconds. Defaults to None.
        fill_first (bool, optional): Whether to fill the first snapshot with previous day's last. Defaults to True.
        asset_class (str, optional): Asset class directory. Defaults to "cryptofuture".
        exchange (str, optional): Exchange directory. Defaults to 'binance'.
    """
    if isinstance(start, (int, np.integer)):
        start = datetime.fromtimestamp(start / 1000, tz=timezone.utc)
    if isinstance(end, (int, np.integer)):
        end = datetime.fromtimestamp(end / 1000, tz=timezone.utc)

    formatted_symbol = format_symbol(symbol)
    start_date = start.date()
    end_date = end.date()
    load_start = start_date - timedelta(days=1) if fill_first else start_date

    all_dfs = []
    curr = load_start
    while curr <= end_date:
        date_str = curr.strftime("%Y%m%d")
        prev_date_str = (curr - timedelta(days=1)).strftime("%Y%m%d")
        prev_snap = _DEPTH_LAST_SNAPSHOT_CACHE.get((formatted_symbol, prev_date_str))

        day_df = _load_depth_data_pandas(
            symbol,
            date_str,
            pivot=False,
            ffill=False,
            align_ms=None,
            fill_first=fill_first,
            asset_class=asset_class,
            exchange=exchange,
            resolution="minute",
            prev_last_snapshot=prev_snap,
        )
        if day_df.empty:
            curr += timedelta(days=1)
            continue

        day_df = day_df.reset_index()
        day_df["date"] = pd.to_datetime(curr)
        all_dfs.append(day_df)
        curr += timedelta(days=1)

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    if fill_first:
        prev_day_mask = df["date"].dt.date < start_date
        if prev_day_mask.any():
            prev_day_data = df[prev_day_mask]
            last_ms = prev_day_data["ms_midnight"].max()
            last_snapshot = prev_day_data[
                prev_day_data["ms_midnight"] == last_ms
            ].copy()
            last_snapshot["date"] = pd.to_datetime(start_date)
            last_snapshot["ms_midnight"] = 0
            df = pd.concat(
                [last_snapshot, df[df["date"].dt.date >= start_date]], ignore_index=True
            )
        else:
            df = df[df["date"].dt.date >= start_date]
    else:
        df = df[df["date"].dt.date >= start_date]

    if align_ms is not None:
        dates = df["date"].unique()
        ms_range = np.arange(0, 86400000, align_ms)
        if pivot:
            df = df.pivot_table(
                index=["date", "ms_midnight"], columns="percentage", values="depth"
            )
            new_index = pd.MultiIndex.from_product(
                [dates, ms_range], names=["date", "ms_midnight"]
            )
            df = df.reindex(new_index)
        else:
            percentages = df["percentage"].unique()
            new_index = pd.MultiIndex.from_product(
                [dates, ms_range, percentages],
                names=["date", "ms_midnight", "percentage"],
            )
            df = (
                df.set_index(["date", "ms_midnight", "percentage"])
                .reindex(new_index)
                .reset_index()
            )

    if pivot and not isinstance(df.index, pd.MultiIndex):
        df = df.pivot_table(
            index=["date", "ms_midnight"], columns="percentage", values="depth"
        )

    if ffill:
        if pivot:
            df = df.groupby(level=0).ffill()
        else:
            df = df.sort_values(["percentage", "date", "ms_midnight"])
            df[["depth", "notional"]] = df.groupby("percentage")[
                ["depth", "notional"]
            ].ffill()

    res_map = {"minute": 60000, "hour": 3600000, "daily": 86400000}
    target_ms = res_map.get(resolution)
    if target_ms:
        if pivot:
            df = df[df.index.get_level_values("ms_midnight") % target_ms == 0]
        else:
            df = df[df["ms_midnight"] % target_ms == 0]

    return df


def load_depth_snapshot_for_minute(
    symbol_value: str,
    dt: datetime,
    data_base: str | None = None,
) -> object | None:
    """
    Load a single minute's depth snapshot from disk (custom layout).
    Returns a simple object with .percentages, .depths, .notionals, .value for OBI etc.
    Use when subscription depth is unavailable (e.g. OpenInterest worker crash).
    When data_base is None, uses _lean_data_folder() in LEAN context so Docker path matches.
    """
    base = data_base if data_base is not None else _lean_data_folder()
    date_str = dt.strftime("%Y%m%d")
    ms_midnight = int(
        (dt.hour * 3600 + dt.minute * 60 + dt.second) * 1000 + dt.microsecond / 1000
    )
    align_ms = 60000
    snapped_ms = int(np.ceil(ms_midnight / align_ms) * align_ms)
    if snapped_ms >= 86400000:
        snapped_ms = 86400000 - align_ms

    try:
        df = _load_depth_day_from_disk_only(symbol_value, date_str, data_base=base)
        if df is None or df.empty:
            return None
        df = df.sort_values(["ms_midnight", "percentage"]).drop_duplicates(
            ["ms_midnight", "percentage"], keep="last"
        )
        df["ms_midnight"] = (
            np.ceil(df["ms_midnight"].astype(float) / align_ms) * align_ms
        ).astype(int)
        df = df[df["ms_midnight"] < 86400000]
        sub = df[df["ms_midnight"] == snapped_ms]
        if sub.empty:
            return None
        if "percentage" in sub.columns and "depth" in sub.columns:
            sub = sub.sort_values("percentage")
            percentages = sub["percentage"].astype(float).tolist()
            depths = sub["depth"].astype(float).tolist()
            notionals = (
                sub["notional"].astype(float).tolist()
                if "notional" in sub.columns
                else [0.0] * len(depths)
            )
        else:
            return None
        total = sum(depths)
        out = type("DepthSnapshot", (), {})()
        out.percentages = percentages
        out.depths = depths
        out.notionals = notionals
        out.value = total
        return out
    except Exception as e:
        logger.debug("load_depth_snapshot_for_minute %s %s: %s", symbol_value, dt, e)
        return None


def load_depth_data_range(
    symbol,
    start_date,
    end_date,
    pivot=False,
    ffill=False,
    align_ms=60000,
    data_dir=None,
):
    """
    Load depth data for a date range via ccxt_data_fetch.fetcher.
    Uses fetcher's cached last snapshot to avoid frequent reloading.
    """
    global _DEPTH_LAST_SNAPSHOT_CACHE
    _DEPTH_LAST_SNAPSHOT_CACHE = {}
    try:
        project_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
        )
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
    except ImportError:
        logger.warning("ccxt_data_fetch not found; depth loading unavailable.")
        return pd.DataFrame()
    start_dt = (
        start_date
        if isinstance(start_date, datetime)
        else datetime.combine(start_date, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )
    )
    end_dt = (
        end_date
        if isinstance(end_date, datetime)
        else datetime.combine(end_date, datetime.max.time()).replace(
            tzinfo=timezone.utc
        )
    )
    return _load_depth_data_range_pandas(
        symbol,
        "minute",
        start_dt,
        end_dt,
        pivot=pivot,
        ffill=ffill,
        align_ms=align_ms,
        fill_first=True,
        asset_class="cryptofuture",
        exchange="binance",
    )


__all__ = ["CryptoFutureDepthData"]
