import ccxt
import pandas as pd
import os
import zipfile
import time
import logging
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from ccxt_data_fetch.config import DATA_LOCATION, ASSET_CLASS, PROXIES
from ccxt_data_fetch.utils import format_symbol, get_ms_from_midnight

logger = logging.getLogger(__name__)


def fetch_ohlcv_range(symbol, timeframe, since_ms, until_ms):
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "proxies": PROXIES,
            "options": {"defaultType": "future"},
        }
    )

    ms_per_step = {"1m": 60000, "1h": 3600000}[timeframe]

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
            pbar.update(min(last_ts + ms_per_step, until_ms) - current_since)
            current_since = last_ts + ms_per_step

            if current_since >= until_ms:
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            logger.error(f"Error for {symbol}: {e}")
            time.sleep(10)
            continue

    pbar.close()
    return all_ohlcv


def save_hour_data(symbol, ohlcv_data):
    if not ohlcv_data:
        return
    df = pd.DataFrame(
        ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["formatted_time"] = df["dt"].dt.strftime("%Y%m%d %H:%M")

    formatted_symbol = format_symbol(symbol)
    base_dir = os.path.join(DATA_LOCATION, ASSET_CLASS, "binance", "hour")
    os.makedirs(base_dir, exist_ok=True)

    zip_path = os.path.join(base_dir, f"{formatted_symbol}.zip")
    csv_filename = f"{formatted_symbol}.csv"

    lean_df = df[["formatted_time", "open", "high", "low", "close", "volume"]]
    csv_content = lean_df.to_csv(index=False, header=False)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_filename, csv_content)
    logger.debug(f"Saved {zip_path}")


def save_minute_data(symbol, ohlcv_data):
    if not ohlcv_data:
        return
    df = pd.DataFrame(
        ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["date_str"] = df["dt"].dt.strftime("%Y%m%d")

    formatted_symbol = format_symbol(symbol)
    symbol_dir = os.path.join(
        DATA_LOCATION, ASSET_CLASS, "binance", "minute", formatted_symbol
    )
    os.makedirs(symbol_dir, exist_ok=True)

    for date_str, group in df.groupby("date_str"):
        day_start = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        mask = (group["dt"] >= day_start) & (group["dt"] < day_end)
        day_group = group.loc[mask].copy()
        if day_group.empty:
            continue

        day_group["ms_midnight"] = day_group["dt"].apply(get_ms_from_midnight)
        lean_df = day_group[["ms_midnight", "open", "high", "low", "close", "volume"]]

        zip_path = os.path.join(symbol_dir, f"{date_str}_trade.zip")
        zf_name = f"{date_str}_trade.csv"
        csv_content = lean_df.to_csv(index=False, header=False)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(zf_name, csv_content)
