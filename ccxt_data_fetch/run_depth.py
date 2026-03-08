import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta

from ccxt_data_fetch.config import START_DATE, END_DATE, DATA_LOCATION, TOP_N_SYMBOL
from ccxt_data_fetch.utils import get_top_200_symbols, format_symbol
from ccxt_data_fetch.fetcher import fetch_and_save_depth_range

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# datetime set time zone to UTC
os.environ['TZ'] = 'UTC'
time.tzset()

def run_fetch_depth(start_dt, end_dt, asset_class: str = "cryptofuture", force_redownload: bool = False):
    """
    Fetches and saves book depth data for top symbols.
    When force_redownload=False, resumes from the day after the latest existing depth file.
    When force_redownload=True, redownloads and overwrites all depth data in the date range.
    """

    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    symbols = get_top_200_symbols(asset_class)
    symbols = symbols[:TOP_N_SYMBOL]
    mode = "REDOWNLOAD" if force_redownload else "fetch"
    logger.info(
        "Starting DEPTH %s: %s from %s to %s for %s symbols...",
        mode,
        asset_class,
        start_dt.date(),
        end_dt.date(),
        len(symbols),
    )

    for symbol in symbols:
        try:
            fetch_and_save_depth_range(
                symbol,
                since_ms,
                until_ms,
                asset_class=asset_class,
                force_redownload=force_redownload,
            )
        except Exception as e:
            logger.error(f"Failed to process depth for {symbol}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch depth data for cryptofuture.")
    parser.add_argument("asset_class", nargs="?", default="cryptofuture", help="Asset class (default: cryptofuture)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: config.START_DATE)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: config.END_DATE)")
    parser.add_argument("--redownload", action="store_true", help="Force redownload and overwrite")
    args = parser.parse_args()

    start_str = args.start or START_DATE
    end_str = args.end or END_DATE
    start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    logger.info(
        "%s depth data for %s (minute) to %s",
        "Redownloading" if args.redownload else "Downloading",
        args.asset_class,
        DATA_LOCATION,
    )
    run_fetch_depth(start_dt, end_dt, asset_class=args.asset_class, force_redownload=args.redownload)
