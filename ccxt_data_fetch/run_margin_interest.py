"""
Fetch margin interest (funding rates) for cryptofuture from Binance via CCXT.
Saves to LEAN format under data/cryptofuture/binance/margin_interest/.
"""
import logging
import os
import sys
import time as ttime
from datetime import datetime, timezone, timedelta

from ccxt_data_fetch.config import START_DATE, END_DATE, DATA_LOCATION, TOP_N_SYMBOL
from ccxt_data_fetch.utils import get_top_200_symbols
from ccxt_data_fetch.fetcher import (
    fetch_funding_rates,
    save_margin_interest,
    fix_margin_interest_format,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
os.environ["TZ"] = "UTC"
ttime.tzset()


def run_fetch_margin_interest(asset_class: str = "cryptofuture") -> None:
    if asset_class != "cryptofuture":
        logger.warning("Margin interest only valid for cryptofuture.")
        return

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    symbols = get_top_200_symbols(asset_class)
    symbols = symbols[:TOP_N_SYMBOL]
    logger.info(
        "Starting margin interest fetch for %s symbols (%s to %s)...",
        len(symbols),
        START_DATE,
        END_DATE,
    )

    for symbol in symbols:
        try:
            rates = fetch_funding_rates(symbol, since_ms, until_ms)
            save_margin_interest(symbol, rates)
        except Exception as e:
            logger.error("Failed to process %s: %s", symbol, e)


if __name__ == "__main__":
    # Usage:
    #   python -m ccxt_data_fetch.run_margin_interest [asset_class]
    #   python -m ccxt_data_fetch.run_margin_interest fix [data_folder]
    # Example: python -m ccxt_data_fetch.run_margin_interest cryptofuture
    # Example: python -m ccxt_data_fetch.run_margin_interest fix data
    args = sys.argv[1:]
    if args and args[0].lower() == "fix":
        data_folder = os.path.abspath(args[1]) if len(args) > 1 else None
        n = fix_margin_interest_format(data_folder=data_folder)
        logger.info("Fixed %s margin interest CSV(s).", n)
        sys.exit(0)

    asset_class = args[0] if args else "cryptofuture"
    logger.info("Downloading margin interest for %s to %s", asset_class, DATA_LOCATION)
    run_fetch_margin_interest(asset_class)
