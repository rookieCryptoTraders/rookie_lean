"""
Unified Binance data fetch from Binance Vision (data.binance.vision).
Stores data under the project data folder following LEAN folder structure.
Timestamps are stored in the format chosen by --timestamp-format (default: Unix milliseconds UTC).

Usage:
  python -m qc_datafeed.binance.run_binance_fetch [asset_class] [data_type] [resolution] [--start YYYYMMDD] [--end YYYYMMDD] [--redownload] [--margin-type um|cm] [--timestamp-format unix_ms|ms_midnight|datetime_string]

  asset_class:  cryptofuture | spot  (default: cryptofuture)
  data_type:    klines | aggTrades | bookDepth | indexPriceKlines | markPriceKlines | premiumIndexKlines | metrics  (futures)
                klines | aggTrades  (spot)
  resolution:   minute | hour | daily  (for klines and *Klines; ignored for aggTrades, bookDepth, metrics). Default: hour
  --timestamp-format: unix_ms (default) | ms_midnight | datetime_string
  --margin-type: um | cm  (futures only; default: um)

Examples:
  python -m qc_datafeed.binance.run_binance_fetch cryptofuture klines hour --start 20260101 --end 20260201
  python -m qc_datafeed.binance.run_binance_fetch cryptofuture aggTrades daily --start 20260101 --redownload
  python -m qc_datafeed.binance.run_binance_fetch cryptofuture bookDepth minute --start 20260101
  python -m qc_datafeed.binance.run_binance_fetch cryptofuture klines hour --timestamp-format ms_midnight
  python -m qc_datafeed.binance.run_binance_fetch spot klines daily --start 20260101

Data is written under: data/<asset_class>/binance/<subpath>/<symbol>/  (see binance README).
"""

import argparse
import io
import logging
import os
import zipfile
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
from tqdm import tqdm

from qc_datafeed.binance.config import (
    DATA_ROOT,
    DEFAULT_SYMBOLS,
    DEFAULT_MARGIN_TYPE,
    BINANCE_VISION_BASE,
    DOWNLOAD_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY_SEC,
    PROXIES,
)
from qc_datafeed.binance.quotes import build_minute_quote_from_aggtrades_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Optional: use ccxt_data_fetch for symbol list (no logic change there)
try:
    from ccxt_data_fetch.utils import get_top_200_symbols, format_symbol as _format_symbol
    from ccxt_data_fetch.config import TOP_N_SYMBOL
    def get_symbols(asset_class: str) -> list[str]:
        try:
            symbols = get_top_200_symbols(asset_class)
            return symbols[:TOP_N_SYMBOL]
        except Exception:
            return DEFAULT_SYMBOLS.copy()
    def format_symbol(s: str) -> str:
        return _format_symbol(s)
except ImportError:
    def get_symbols(asset_class: str) -> list[str]:
        return DEFAULT_SYMBOLS.copy()
    def format_symbol(s: str) -> str:
        return s.replace("/", "").replace(":", "").upper().strip().lower()


def _download_with_retries(url: str, proxies: dict | None = None) -> requests.Response | None:
    proxies = proxies if proxies is not None else PROXIES
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, proxies=proxies, timeout=DOWNLOAD_TIMEOUT)
            return r
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                logger.warning("Download attempt %s/%s failed: %s. Retrying in %ss...", attempt, MAX_RETRIES, e, RETRY_DELAY_SEC)
                import time
                time.sleep(RETRY_DELAY_SEC)
    logger.error("Failed to download after %s attempts: %s", MAX_RETRIES, last_error)
    return None


def _date_range(start_yyyymmdd: str, end_yyyymmdd: str) -> list[str]:
    start = datetime.strptime(start_yyyymmdd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_yyyymmdd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    out = []
    d = start
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


# Resolution: only minute | hour | daily (folder names); map to Binance interval for URLs
RESOLUTION_CHOICES = ("minute", "hour", "daily")


def _resolution_to_interval(resolution: str) -> str:
    """Map resolution (minute|hour|daily) to Binance Vision interval (1m|1h|1d)."""
    m = {"minute": "1m", "hour": "1h", "daily": "1d"}
    return m.get(resolution, "1h")


def _resolution_to_subdir(resolution: str) -> str:
    """Resolution is already minute|hour|daily; use as subdir name."""
    return resolution if resolution in RESOLUTION_CHOICES else "hour"


# Timestamp format: unix_ms | ms_midnight | datetime_string
TS_FORMAT_UNIX_MS = "unix_ms"
TS_FORMAT_MS_MIDNIGHT = "ms_midnight"
TS_FORMAT_DATETIME_STR = "datetime_string"
TS_FORMAT_CHOICES = (TS_FORMAT_UNIX_MS, TS_FORMAT_MS_MIDNIGHT, TS_FORMAT_DATETIME_STR)


def _format_timestamp(unix_ms: int, ts_format: str) -> int | str:
    """Convert Unix ms (UTC) to the requested storage format."""
    if ts_format == TS_FORMAT_UNIX_MS:
        return int(unix_ms)
    if ts_format == TS_FORMAT_MS_MIDNIGHT:
        return int(unix_ms % (24 * 60 * 60 * 1000))
    if ts_format == TS_FORMAT_DATETIME_STR:
        dt = datetime.fromtimestamp(unix_ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return int(unix_ms)


def _format_timestamp_from_midnight(date_str: str, ms_midnight: int, ts_format: str) -> int | str:
    """Convert (date, ms_since_midnight) to the requested storage format."""
    start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    unix_ms = int(start.timestamp() * 1000) + ms_midnight
    return _format_timestamp(unix_ms, ts_format)


# ---------------------------------------------------------------------------
# Futures
# ---------------------------------------------------------------------------


def _fetch_futures_klines(
    symbol: str,
    interval: str,
    date_str: str,
    margin_type: str,
    redownload: bool,
    ts_format: str,
) -> None:
    sym = format_symbol(symbol).upper()
    interval = _resolution_to_interval(interval)
    subdir = _resolution_to_subdir(interval)
    out_dir = os.path.join(DATA_ROOT, "cryptofuture", "binance", subdir, format_symbol(symbol))
    os.makedirs(out_dir, exist_ok=True)
    date_yyyymmdd = date_str.replace("-", "")
    zip_name = f"{date_yyyymmdd}_trade.zip"
    zip_path = os.path.join(out_dir, zip_name)
    if not redownload and os.path.isfile(zip_path):
        return
    url = f"{BINANCE_VISION_BASE}/futures/{margin_type}/daily/klines/{sym}/{interval}/{sym}-{interval}-{date_str}.zip"
    resp = _download_with_retries(url)
    if resp is None or resp.status_code != 200:
        if resp and resp.status_code == 404:
            logger.debug("No klines for %s on %s (404)", symbol, date_str)
        return
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if not names:
            return
        with zf.open(names[0]) as f:
            df = pd.read_csv(f, header=None)
    if df.empty:
        return
    df = df.iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    unix_ms = pd.to_numeric(df["time"], errors="coerce")
    df["time"] = unix_ms.apply(lambda x: _format_timestamp(int(x), ts_format) if pd.notna(x) else x)
    csv_name = f"{date_yyyymmdd}_trade.csv"
    content = df.to_csv(index=False, header=False)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, content)
    logger.debug("Wrote %s", zip_path)


def _fetch_futures_agg_trades(
    symbol: str,
    date_str: str,
    margin_type: str,
    redownload: bool,
    ts_format: str,
) -> None:
    sym = format_symbol(symbol).upper()
    out_dir = os.path.join(DATA_ROOT, "cryptofuture", "binance", "aggtrades", format_symbol(symbol))
    os.makedirs(out_dir, exist_ok=True)
    date_yyyymmdd = date_str.replace("-", "")
    zip_name = f"{date_yyyymmdd}_aggtrades.zip"
    zip_path = os.path.join(out_dir, zip_name)
    if not redownload and os.path.isfile(zip_path):
        return
    url = f"{BINANCE_VISION_BASE}/futures/{margin_type}/daily/aggTrades/{sym}/{sym}-aggTrades-{date_str}.zip"
    resp = _download_with_retries(url)
    if resp is None or resp.status_code != 200:
        if resp and resp.status_code == 404:
            logger.debug("No aggTrades for %s on %s (404)", symbol, date_str)
        return
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if not names:
            return
        with zf.open(names[0]) as f:
            # Keep all columns as str to avoid dtype conversion (e.g. int -> float)
            df = pd.read_csv(f, header=None, low_memory=False, dtype=str)
    if len(df.columns) >= 7:
        df = df.iloc[:, :7].copy()
        # Format only column 5 (transact_time); keep as str so output has no float
        def fmt_ts(s: str):
            if not s or (isinstance(s, float) and pd.isna(s)):
                return s
            try:
                v = _format_timestamp(int(float(s)), ts_format)
                return str(v) if isinstance(v, int) else v
            except (ValueError, TypeError):
                return s
        df.iloc[:, 5] = [fmt_ts(x) for x in df.iloc[:, 5]]
    csv_name = f"{date_yyyymmdd}_aggtrades.csv"
    content = df.to_csv(index=False, header=False)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, content)
    logger.debug("Wrote %s", zip_path)


def _ms_from_midnight(dt) -> int:
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int((dt - midnight).total_seconds() * 1000)


def _fetch_futures_book_depth(
    symbol: str,
    date_str: str,
    margin_type: str,
    redownload: bool,
    ts_format: str,
) -> None:
    """Download bookDepth and save with first column in chosen timestamp format (unix_ms, ms_midnight, or datetime_string)."""
    sym = format_symbol(symbol).upper()
    out_dir = os.path.join(DATA_ROOT, "cryptofuture", "binance", "depth", format_symbol(symbol))
    os.makedirs(out_dir, exist_ok=True)
    date_yyyymmdd = date_str.replace("-", "")
    zip_name = f"{date_yyyymmdd}_depth.zip"
    zip_path = os.path.join(out_dir, zip_name)
    if not redownload and os.path.isfile(zip_path):
        return
    url = f"{BINANCE_VISION_BASE}/futures/{margin_type}/daily/bookDepth/{sym}/{sym}-bookDepth-{date_str}.zip"
    resp = _download_with_retries(url)
    if resp is None or resp.status_code != 200:
        if resp and resp.status_code == 404:
            logger.debug("No bookDepth for %s on %s (404)", symbol, date_str)
        return
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if not names:
            return
        with zf.open(names[0]) as f:
            df = pd.read_csv(f, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    col_lower = {c.lower(): c for c in df.columns}
    ts_col = col_lower.get("timestamp") or col_lower.get("update_time") or col_lower.get("time") or df.columns[0]
    ts_vals = df[ts_col]
    if pd.api.types.is_numeric_dtype(ts_vals):
        df["_dt"] = pd.to_datetime(ts_vals, unit="ms", utc=True)
    else:
        df["_dt"] = pd.to_datetime(ts_vals, utc=True)
    df["ms_midnight"] = df["_dt"].apply(_ms_from_midnight)
    for name in ("percentage", "depth", "notional"):
        if name not in df.columns and name in col_lower:
            df[name] = df[col_lower[name]]
    # First column: timestamp in chosen format
    df["_ts_out"] = df["ms_midnight"].apply(lambda ms: _format_timestamp_from_midnight(date_str, int(ms), ts_format))
    lean = df[["_ts_out", "percentage", "depth", "notional"]].rename(columns={"_ts_out": "time"})
    csv_name = f"{date_yyyymmdd}_depth.csv"
    content = lean.to_csv(index=False, header=False)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, content)
    logger.debug("Wrote %s", zip_path)


def _fetch_futures_extra_klines(
    symbol: str,
    interval: str,
    date_str: str,
    margin_type: str,
    data_type: str,
    redownload: bool,
) -> None:
    """indexPriceKlines, markPriceKlines, premiumIndexKlines."""
    sym = format_symbol(symbol).upper()
    interval = _resolution_to_interval(interval)
    subdir = _resolution_to_subdir(interval)
    out_dir = os.path.join(DATA_ROOT, "cryptofuture", "binance", data_type, subdir, format_symbol(symbol))
    os.makedirs(out_dir, exist_ok=True)
    date_yyyymmdd = date_str.replace("-", "")
    zip_name = f"{date_yyyymmdd}.zip"
    zip_path = os.path.join(out_dir, zip_name)
    if not redownload and os.path.isfile(zip_path):
        return
    url = f"{BINANCE_VISION_BASE}/futures/{margin_type}/daily/{data_type}/{sym}/{interval}/{sym}-{interval}-{date_str}.zip"
    resp = _download_with_retries(url)
    if resp is None or resp.status_code != 200:
        if resp and resp.status_code == 404:
            logger.debug("No %s for %s on %s (404)", data_type, symbol, date_str)
        return
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if not names:
            return
        with zf.open(names[0]) as f:
            raw = f.read()
    csv_name = f"{date_yyyymmdd}.csv"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, raw)
    logger.debug("Wrote %s", zip_path)


def _fetch_futures_metrics(symbol: str, date_str: str, margin_type: str, redownload: bool) -> None:
    sym = format_symbol(symbol).upper()
    out_dir = os.path.join(DATA_ROOT, "cryptofuture", "binance", "metrics", format_symbol(symbol))
    os.makedirs(out_dir, exist_ok=True)
    date_yyyymmdd = date_str.replace("-", "")
    zip_name = f"{date_yyyymmdd}_metrics.zip"
    zip_path = os.path.join(out_dir, zip_name)
    if not redownload and os.path.isfile(zip_path):
        return
    url = f"{BINANCE_VISION_BASE}/futures/{margin_type}/daily/metrics/{sym}/{sym}-metrics-{date_str}.zip"
    resp = _download_with_retries(url)
    if resp is None or resp.status_code != 200:
        if resp and resp.status_code == 404:
            logger.debug("No metrics for %s on %s (404)", symbol, date_str)
        return
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if not names:
            return
        with zf.open(names[0]) as f:
            raw = f.read()
    csv_name = f"{date_yyyymmdd}_metrics.csv"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, raw)
    logger.debug("Wrote %s", zip_path)


# ---------------------------------------------------------------------------
# Spot
# ---------------------------------------------------------------------------


def _fetch_spot_klines(symbol: str, interval: str, date_str: str, redownload: bool, ts_format: str) -> None:
    sym = format_symbol(symbol).upper()
    interval = _resolution_to_interval(interval)
    subdir = _resolution_to_subdir(interval)
    out_dir = os.path.join(DATA_ROOT, "spot", "binance", subdir, format_symbol(symbol))
    os.makedirs(out_dir, exist_ok=True)
    date_yyyymmdd = date_str.replace("-", "")
    zip_name = f"{date_yyyymmdd}_trade.zip"
    zip_path = os.path.join(out_dir, zip_name)
    if not redownload and os.path.isfile(zip_path):
        return
    url = f"{BINANCE_VISION_BASE}/spot/daily/klines/{sym}/{interval}/{sym}-{interval}-{date_str}.zip"
    resp = _download_with_retries(url)
    if resp is None or resp.status_code != 200:
        if resp and resp.status_code == 404:
            logger.debug("No spot klines for %s on %s (404)", symbol, date_str)
        return
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if not names:
            return
        with zf.open(names[0]) as f:
            df = pd.read_csv(f, header=None)
    if df.empty:
        return
    df = df.iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    unix_ms = pd.to_numeric(df["time"], errors="coerce")
    df["time"] = unix_ms.apply(lambda x: _format_timestamp(int(x), ts_format) if pd.notna(x) else x)
    csv_name = f"{date_yyyymmdd}_trade.csv"
    content = df.to_csv(index=False, header=False)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, content)
    logger.debug("Wrote %s", zip_path)


def _fetch_spot_agg_trades(symbol: str, date_str: str, redownload: bool, ts_format: str) -> None:
    sym = format_symbol(symbol).upper()
    out_dir = os.path.join(DATA_ROOT, "spot", "binance", "aggtrades", format_symbol(symbol))
    os.makedirs(out_dir, exist_ok=True)
    date_yyyymmdd = date_str.replace("-", "")
    zip_name = f"{date_yyyymmdd}_aggtrades.zip"
    zip_path = os.path.join(out_dir, zip_name)
    if not redownload and os.path.isfile(zip_path):
        return
    url = f"{BINANCE_VISION_BASE}/spot/daily/aggTrades/{sym}/{sym}-aggTrades-{date_str}.zip"
    resp = _download_with_retries(url)
    if resp is None or resp.status_code != 200:
        if resp and resp.status_code == 404:
            logger.debug("No spot aggTrades for %s on %s (404)", symbol, date_str)
        return
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if not names:
            return
        with zf.open(names[0]) as f:
            df = pd.read_csv(f, header=None, low_memory=False, dtype=str)
    if len(df.columns) >= 7:
        df = df.iloc[:, :7].copy()
        def fmt_ts(s: str):
            if not s or (isinstance(s, float) and pd.isna(s)):
                return s
            try:
                v = _format_timestamp(int(float(s)), ts_format)
                return str(v) if isinstance(v, int) else v
            except (ValueError, TypeError):
                return s
        df.iloc[:, 5] = [fmt_ts(x) for x in df.iloc[:, 5]]
    csv_name = f"{date_yyyymmdd}_aggtrades.csv"
    content = df.to_csv(index=False, header=False)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, content)
    logger.debug("Wrote %s", zip_path)


# ---------------------------------------------------------------------------
# Dispatch and main
# ---------------------------------------------------------------------------

FUTURES_DATA_TYPES = ("klines", "aggTrades", "bookDepth", "indexPriceKlines", "markPriceKlines", "premiumIndexKlines", "metrics", "quote")
SPOT_DATA_TYPES = ("klines", "aggTrades", "quote")
RESOLUTION_OPTIONAL = ("aggTrades", "bookDepth", "metrics", "quote")


def run_fetch(
    asset_class: str,
    data_type: str,
    resolution: str,
    start_str: str,
    end_str: str,
    redownload: bool,
    margin_type: str,
    ts_format: str,
) -> None:
    resolution = resolution if resolution in RESOLUTION_CHOICES else "hour"
    dates = _date_range(start_str, end_str)
    symbols = get_symbols(asset_class)

    if asset_class == "cryptofuture":
        if data_type not in FUTURES_DATA_TYPES:
            raise ValueError(f"data_type must be one of {FUTURES_DATA_TYPES}")
        for symbol in tqdm(symbols, desc=data_type, unit="symbol"):
            for date_str in dates:
                try:
                    if data_type == "klines":
                        _fetch_futures_klines(symbol, resolution, date_str, margin_type, redownload, ts_format)
                    elif data_type == "aggTrades":
                        _fetch_futures_agg_trades(symbol, date_str, margin_type, redownload, ts_format)
                    elif data_type == "bookDepth":
                        _fetch_futures_book_depth(symbol, date_str, margin_type, redownload, ts_format)
                    elif data_type in ("indexPriceKlines", "markPriceKlines", "premiumIndexKlines"):
                        _fetch_futures_extra_klines(symbol, resolution, date_str, margin_type, data_type, redownload)
                    elif data_type == "metrics":
                        _fetch_futures_metrics(symbol, date_str, margin_type, redownload)
                    elif data_type == "quote":
                        # Quote bars are rebuilt from aggTrades. Ensure underlying aggTrades
                        # exist first (this call is idempotent and respects redownload).
                        _fetch_futures_agg_trades(symbol, date_str, margin_type, redownload, ts_format)

                        # Resolution is effectively "minute" for output quote bars.
                        date_yyyymmdd = date_str.replace("-", "")
                        build_minute_quote_from_aggtrades_files(
                            symbol=symbol,
                            date_str=date_yyyymmdd,
                            asset_class=asset_class,
                            redownload=redownload,
                        )
                except Exception as e:
                    logger.warning("Failed %s %s %s: %s", symbol, date_str, data_type, e)
    elif asset_class == "spot":
        if data_type not in SPOT_DATA_TYPES:
            raise ValueError(f"data_type must be one of {SPOT_DATA_TYPES}")
        for symbol in tqdm(symbols, desc=data_type, unit="symbol"):
            for date_str in dates:
                try:
                    if data_type == "klines":
                        _fetch_spot_klines(symbol, resolution, date_str, redownload, ts_format)
                    elif data_type == "aggTrades":
                        _fetch_spot_agg_trades(symbol, date_str, redownload, ts_format)
                    else:  # quote
                        # Ensure underlying aggTrades exist first (idempotent).
                        _fetch_spot_agg_trades(symbol, date_str, redownload, ts_format)

                        date_yyyymmdd = date_str.replace("-", "")
                        build_minute_quote_from_aggtrades_files(
                            symbol=symbol,
                            date_str=date_yyyymmdd,
                            asset_class=asset_class,
                            redownload=redownload,
                        )
                except Exception as e:
                    logger.warning("Failed %s %s %s: %s", symbol, date_str, data_type, e)
    else:
        raise ValueError("asset_class must be cryptofuture or spot")

    logger.info("Done: %s %s %s from %s to %s (ts_format=%s)", asset_class, data_type, resolution, start_str, end_str, ts_format)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Binance data from Binance Vision into LEAN data folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("asset_class", nargs="?", default="cryptofuture", help="cryptofuture | spot")
    parser.add_argument(
        "data_type",
        nargs="?",
        default="klines",
        help="Data type (e.g. klines, aggTrades, bookDepth, indexPriceKlines, markPriceKlines, premiumIndexKlines, metrics for futures; klines, aggTrades for spot)",
    )
    parser.add_argument(
        "resolution",
        nargs="?",
        default="hour",
        choices=RESOLUTION_CHOICES,
        help="Resolution for klines/*Klines: minute | hour | daily (ignored for aggTrades, bookDepth, metrics)",
    )
    parser.add_argument("--start", default=None, help="Start date YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--redownload", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--timestamp-format",
        choices=TS_FORMAT_CHOICES,
        default=TS_FORMAT_UNIX_MS,
        help="How to store timestamps: unix_ms (default) | ms_midnight | datetime_string",
    )
    parser.add_argument(
        "--margin-type",
        choices=("um", "cm"),
        default=DEFAULT_MARGIN_TYPE,
        help="Futures margin type (default: um)",
    )
    args = parser.parse_args()

    start_str = args.start or "2026-01-01"
    end_str = args.end or "2026-01-31"
    if len(start_str) == 8 and start_str.isdigit():
        start_str = f"{start_str[:4]}-{start_str[4:6]}-{start_str[6:8]}"
    if len(end_str) == 8 and end_str.isdigit():
        end_str = f"{end_str[:4]}-{end_str[4:6]}-{end_str[6:8]}"

    run_fetch(
        asset_class=args.asset_class or "cryptofuture",
        data_type=args.data_type or "klines",
        resolution=args.resolution or "hour",
        start_str=start_str,
        end_str=end_str,
        redownload=args.redownload,
        margin_type=args.margin_type,
        ts_format=args.timestamp_format,
    )


if __name__ == "__main__":
    main()
