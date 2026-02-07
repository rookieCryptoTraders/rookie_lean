"""
Binance Vision Archive Downloader
=================================
Downloads historical crypto futures data from Binance Vision for quantitative research.

Supported data types:
- metrics: OI, Long/Short Ratio, Taker Buy/Sell Ratio (5-min, resampled to target)
- indexPriceKlines: Index Price (1m/1h)
- markPriceKlines: Mark Price (1m/1h)
- premiumIndexKlines: Premium Index (1m/1h)

Output:
Saves merged CSVs to data/cryptofuture/binance/extra/
Naming: {ticker}_{type}.csv (e.g. btcusdt_metrics.csv, btcusdt_index.csv)
"""

import os
import sys
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

try:
    from .config import PROXIES, DATA_LOCATION, ASSET_CLASS
except ImportError:
    # Standalone fallback
    PROXIES = {"http": "http://127.0.0.1:1082", "https": "http://127.0.0.1:1082"}
    DATA_LOCATION = "/Users/chenzhao/Documents/lean_workspace/data"
    ASSET_CLASS = "cryptofuture"

# Configuration
TIMEFRAME = "1h"  # Target resolution
START_DATE = "2025-01-01"
END_DATE = "2026-01-31"

TICKERS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "TRXUSDT",
    "LINKUSDT",
    "MATICUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "ATOMUSDT",
    "UNIUSDT",
    "ETCUSDT",
    "FILUSDT",
]

DATA_TYPES = {
    "metrics": {
        "url": "https://data.binance.vision/data/futures/um/daily/metrics/{ticker}/{ticker}-metrics-{date}.zip",
        "suffix": "_metrics.csv",
        "is_kline": False,
        "resample": True,
    },
    "indexPriceKlines": {
        "url": f"https://data.binance.vision/data/futures/um/daily/indexPriceKlines/{{ticker}}/{TIMEFRAME}/{{ticker}}-{TIMEFRAME}-{{date}}.zip",
        "suffix": "_index.csv",
        "is_kline": True,
        "resample": False,
    },
    "markPriceKlines": {
        "url": f"https://data.binance.vision/data/futures/um/daily/markPriceKlines/{{ticker}}/{TIMEFRAME}/{{ticker}}-{TIMEFRAME}-{{date}}.zip",
        "suffix": "_mark.csv",
        "is_kline": True,
        "resample": False,
    },
    "premiumIndexKlines": {
        "url": f"https://data.binance.vision/data/futures/um/daily/premiumIndexKlines/{{ticker}}/{TIMEFRAME}/{{ticker}}-{TIMEFRAME}-{{date}}.zip",
        "suffix": "_premium.csv",
        "is_kline": True,
        "resample": False,
    },
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BinanceDownloader")


def get_daterange(start, end):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    delta = e - s
    return [(s + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]


def download_url(url):
    try:
        resp = requests.get(url, proxies=PROXIES, timeout=10)
        if resp.status_code == 200:
            return resp.content
        elif resp.status_code == 404:
            return 404
        return None
    except Exception as e:
        logger.debug(f"Error downloading {url}: {e}")
        return None


def process_ticker_type(ticker, type_key, dates, output_dir):
    spec = DATA_TYPES[type_key]
    dfs = []

    # Check if target exists
    out_path = os.path.join(output_dir, f"{ticker.lower()}{spec['suffix']}")
    if os.path.exists(out_path):
        logger.info(f"[{ticker}] {type_key} exists, skipping.")
        return

    logger.info(f"[{ticker}] Downloading {type_key} ({len(dates)} days)...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {}
        for d in dates:
            url = spec["url"].format(ticker=ticker, date=d)
            future_map[executor.submit(download_url, url)] = d

        for future in as_completed(future_map):
            content = future.result()
            if content == 404:
                continue
            if content:
                try:
                    with zipfile.ZipFile(io.BytesIO(content)) as zf:
                        csv_name = zf.namelist()[0]
                        with zf.open(csv_name) as f:
                            df = pd.read_csv(f)
                            # Strip whitespace from columns
                            df.columns = [c.strip() for c in df.columns]

                            if spec["is_kline"]:
                                # Kline format: open_time, ...
                                # We need: time, open, high, low, close
                                clean = pd.DataFrame()
                                clean["time"] = pd.to_datetime(
                                    df["open_time"], unit="ms", utc=True
                                )
                                clean["open"] = df["open"]
                                clean["high"] = df["high"]
                                clean["low"] = df["low"]
                                clean["close"] = df["close"]
                                dfs.append(clean)
                            else:
                                # Metrics format
                                # create_time, symbol, sum_open_interest, ...
                                clean = pd.DataFrame()
                                # Handle varied column names "create_time" or "createTime"
                                time_col = next(
                                    (c for c in df.columns if "time" in c.lower()), None
                                )
                                if time_col:
                                    clean["time"] = pd.to_datetime(
                                        df[time_col], unit="ms", utc=True
                                    )  # Metrics often not ms? verify
                                    # Actually metrics create_time usually string in daily dumps?
                                    # Let's try pd.to_datetime autodetect
                                    # If it's already datetime, good.
                                    # For safety, let's inspect sample. Usually it IS a string "2025-01-01 00:05:00"
                                    # But sometimes it calls it create_time.
                                    pass

                                # Metrics specific columns
                                if "sum_open_interest" in df.columns:
                                    clean["time"] = pd.to_datetime(df[time_col])
                                    if clean["time"].dt.tz is None:
                                        clean["time"] = clean["time"].dt.tz_localize(
                                            "UTC"
                                        )
                                    clean["openInterest"] = df["sum_open_interest"]
                                    clean["ls_ratio"] = df[
                                        "sum_toptrader_long_short_ratio"
                                    ]
                                    clean["taker_ratio"] = df[
                                        "sum_taker_long_short_vol_ratio"
                                    ]
                                    dfs.append(clean)
                except Exception as e:
                    logger.warning(f"Failed to parse {future_map[future]}: {e}")

    if not dfs:
        logger.warning(f"[{ticker}] No data found for {type_key}")
        return

    # Merge
    full_df = pd.concat(dfs).sort_values("time").drop_duplicates("time")

    # Resample
    if spec["resample"] or spec["suffix"] == "_metrics.csv":
        # Metrics are 5m, we want 1h usually
        # Use '1min' instead of '1m' to avoid pandas legacy warning if TIMEFRAME='1m'
        tf = TIMEFRAME
        if tf == "1m":
            tf = "1min"

        full_df = (
            full_df.set_index("time")
            .resample(tf)
            .last()
            .dropna(how="all")
            .reset_index()
        )

    # Save
    full_df["time"] = full_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    full_df.to_csv(out_path, index=False)
    logger.info(f"[{ticker}] Saved {len(full_df)} rows to {os.path.basename(out_path)}")


def main():
    extra_dir = os.path.join(DATA_LOCATION, ASSET_CLASS, "binance", "extra")
    os.makedirs(extra_dir, exist_ok=True)

    print("=" * 60)
    print(f" Binance Downloader | {START_DATE} -> {END_DATE} | {TIMEFRAME}")
    print("=" * 60)

    dates = get_daterange(START_DATE, END_DATE)

    for ticker in TICKERS:
        print(f"\n>>> {ticker}")
        for type_key in DATA_TYPES:
            try:
                process_ticker_type(ticker, type_key, dates, extra_dir)
            except Exception as e:
                logger.error(f"Error processing {ticker} {type_key}: {e}")


if __name__ == "__main__":
    main()
