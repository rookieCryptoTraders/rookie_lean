"""
Fetch margin interest (funding rates) for cryptofuture from Binance via CCXT.
Saves to LEAN format under data/cryptofuture/binance/margin_interest/.
"""
import argparse
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


def run_fetch_margin_interest(
    asset_class: str = "cryptofuture",
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    if asset_class != "cryptofuture":
        logger.warning("Margin interest only valid for cryptofuture.")
        return

    start_str = start_date or START_DATE
    end_str = end_date or END_DATE
    start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    symbols = get_top_200_symbols(asset_class)
    symbols = symbols[:TOP_N_SYMBOL]
    logger.info(
        "Starting margin interest fetch for %s symbols (%s to %s)...",
        len(symbols),
        start_str,
        end_str,
    )

    for symbol in symbols:
        try:
            rates = fetch_funding_rates(symbol, since_ms, until_ms)
            save_margin_interest(symbol, rates)
        except Exception as e:
            logger.error("Failed to process %s: %s", symbol, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch margin interest (funding rates) for cryptofuture.")
    parser.add_argument("asset_class", nargs="?", default="cryptofuture", help="Asset class")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: config.START_DATE)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: config.END_DATE)")
    parser.add_argument("fix", nargs="?", help="Pass 'fix' to fix margin interest format; optional data_folder")
    args = parser.parse_args()

    if args.asset_class.lower() == "fix":
        data_folder = os.path.abspath(args.fix) if args.fix else None
        n = fix_margin_interest_format(data_folder=data_folder)
        logger.info("Fixed %s margin interest CSV(s).", n)
        sys.exit(0)

    logger.info("Downloading margin interest for %s to %s", args.asset_class, DATA_LOCATION)
    run_fetch_margin_interest(
        args.asset_class,
        start_date=args.start,
        end_date=args.end,
    )
