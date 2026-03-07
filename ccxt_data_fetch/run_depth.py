import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from config import START_DATE, END_DATE, DATA_LOCATION, TOP_N_SYMBOL
from utils import get_top_200_symbols, format_symbol
from fetcher import fetch_and_save_depth_range

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
        f"Starting DEPTH {mode}: {asset_class} from {START_DATE} to {END_DATE} for {len(symbols)} symbols..."
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
    # Usage: python -m ccxt_data_fetch.run_depth [asset_class] [--redownload]
    args = [a for a in sys.argv[1:] if a != "--redownload"]
    force_redownload = "--redownload" in sys.argv
    asset_class = args[0] if args else "cryptofuture"

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(
        f"{'Redownloading' if force_redownload else 'Downloading'} depth data for {asset_class} (minute) to {DATA_LOCATION}"
    )
    run_fetch_depth(start_dt, end_dt, asset_class=asset_class, force_redownload=force_redownload)
