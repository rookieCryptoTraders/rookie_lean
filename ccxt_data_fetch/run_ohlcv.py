"""
Fetch OHLCV (trade or quote tick type) for crypto/cryptofuture from Binance via CCXT.
Saves to LEAN format under data/<asset_class>/binance/<resolution>/.
"""
import argparse
import logging
import os
import sys
import time as ttime
from datetime import datetime, timezone, timedelta

from ccxt_data_fetch.config import START_DATE, END_DATE, DATA_LOCATION, TOP_N_SYMBOL
from ccxt_data_fetch.utils import get_top_200_symbols, format_symbol
from ccxt_data_fetch.fetcher import fetch_ohlcv_range, save_ohlcv_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
os.environ["TZ"] = "UTC"
ttime.tzset()


def run_fetch_ohlcv(
    asset_class: str,
    resolution: str,
    tick_type: str = "trade",
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    start_str = start_date or START_DATE
    end_str = end_date or END_DATE
    start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    timeframe_map = {"minute": "1m", "hour": "1h", "daily": "1d"}
    if resolution not in timeframe_map:
        logger.error("Unknown resolution: %s (use minute, hour, daily)", resolution)
        return

    symbols = get_top_200_symbols(asset_class)
    symbols = symbols[:TOP_N_SYMBOL]
    logger.info(
        "Starting OHLCV fetch: %s | %s | %s for %s symbols...",
        asset_class,
        resolution,
        tick_type,
        len(symbols),
    )

    for symbol in symbols:
        try:
            current_since = since_ms
            formatted_symbol = format_symbol(symbol)

            if resolution == "minute":
                symbol_dir = os.path.join(DATA_LOCATION, asset_class, "binance", "minute", formatted_symbol)
                if os.path.exists(symbol_dir):
                    existing = [f for f in os.listdir(symbol_dir) if f.endswith(f"_{tick_type}.zip")]
                    if existing:
                        latest_zip = sorted(existing)[-1]
                        latest_date = latest_zip.split("_")[0]
                        try:
                            latest_dt = datetime.strptime(latest_date, "%Y%m%d").replace(tzinfo=timezone.utc)
                            resume_ms = int((latest_dt + timedelta(days=1)).timestamp() * 1000)
                            if resume_ms >= until_ms:
                                logger.info("Skipping %s, already up to date.", symbol)
                                continue
                            current_since = max(current_since, resume_ms)
                        except ValueError:
                            pass

            data = fetch_ohlcv_range(symbol, timeframe_map[resolution], current_since, until_ms, asset_class)
            save_ohlcv_data(symbol, data, resolution, asset_class, tick_type)
        except Exception as e:
            logger.error("Failed to process %s: %s", symbol, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OHLCV data for crypto/cryptofuture.")
    parser.add_argument("asset_class", nargs="?", default="cryptofuture", help="Asset class")
    parser.add_argument("resolution", nargs="?", default="minute", help="Resolution: minute, hour, daily")
    parser.add_argument("tick_type", nargs="?", default="trade", help="Tick type: trade or quote")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: config.START_DATE)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: config.END_DATE)")
    args = parser.parse_args()

    logger.info("Downloading OHLCV %s %s %s to %s", args.asset_class, args.resolution, args.tick_type, DATA_LOCATION)
    run_fetch_ohlcv(
        args.asset_class,
        args.resolution,
        args.tick_type,
        start_date=args.start,
        end_date=args.end,
    )
