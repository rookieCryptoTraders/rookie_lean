"""
Fetch cryptofuture quote (best bid/ask) from Binance and save as QuantConnect minute QuoteBar.

- Tries Binance Vision first: data.binance.vision/data/futures/um/daily/bookTicker/{SYMBOL}/
- If Vision 404 (e.g. not available for USD-M), falls back to one current snapshot via REST.
- Output: data/cryptofuture/binance/minute/<symbol>/YYYYMMDD_quote.zip
  CSV (no header): Time (ms since midnight), BidOpen, BidHigh, BidLow, BidClose, BidSize,
  AskOpen, AskHigh, AskLow, AskClose, AskSize (LEAN minute quote format).

Usage:
  python -m ccxt_data_fetch.run_quote [asset_class] [--redownload]
"""
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta

from ccxt_data_fetch.config import START_DATE, END_DATE, DATA_LOCATION, TOP_N_SYMBOL
from ccxt_data_fetch.utils import get_top_200_symbols, format_symbol
from ccxt_data_fetch.fetcher import fetch_and_save_quote_range

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
os.environ["TZ"] = "UTC"
time.tzset()


def run_fetch_quote(
    start_dt: datetime,
    end_dt: datetime,
    asset_class: str = "cryptofuture",
    force_redownload: bool = False,
) -> None:
    """
    Build L1 quote from depth + trade for each symbol in the date range.
    Skips dates that already have quote zip when force_redownload=False.
    """
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    symbols = get_top_200_symbols(asset_class)
    symbols = symbols[:TOP_N_SYMBOL]
    mode = "REDOWNLOAD" if force_redownload else "build"
    logger.info(
        "Starting L1 QUOTE %s: %s from %s to %s for %s symbols...",
        mode,
        asset_class,
        start_dt.date(),
        end_dt.date(),
        len(symbols),
    )

    for symbol in symbols:
        try:
            fetch_and_save_quote_range(
                symbol,
                since_ms,
                until_ms,
                asset_class=asset_class,
                force_redownload=force_redownload,
            )
        except Exception as e:
            logger.error("Failed to process quote for %s: %s", symbol, e)


if __name__ == "__main__":
    # Usage: python -m ccxt_data_fetch.run_quote [asset_class] [--redownload]
    args = [a for a in sys.argv[1:] if a != "--redownload"]
    force_redownload = "--redownload" in sys.argv
    asset_class = args[0] if args else "cryptofuture"

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    logger.info(
        "%s L1 quote data for %s (minute) to %s",
        "Redownloading" if force_redownload else "Building",
        asset_class,
        DATA_LOCATION,
    )
    run_fetch_quote(
        start_dt,
        end_dt,
        asset_class=asset_class,
        force_redownload=force_redownload,
    )
