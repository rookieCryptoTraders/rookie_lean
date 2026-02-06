#!/usr/bin/env python3
"""
ccxt_data_fetch CLI
===================
Command-line interface for the Binance data downloader.

Usage:
    # Download all data for default tickers
    python -m ccxt_data_fetch

    # Download specific tickers
    python -m ccxt_data_fetch BTCUSDT ETHUSDT

    # Download only OHLCV
    python -m ccxt_data_fetch --ohlcv-only BTCUSDT

    # Download only metrics (no OHLCV)
    python -m ccxt_data_fetch --no-ohlcv BTCUSDT

    # Change timeframe
    python -m ccxt_data_fetch --timeframe 4h BTCUSDT

    # Custom date range
    python -m ccxt_data_fetch --start 2025-06-01 --end 2025-12-31 BTCUSDT
"""

import argparse
import logging
import sys
from typing import List, Optional

from .downloader import BinanceDataDownloader
from .config import (
    DEFAULT_TICKERS,
    DEFAULT_TIMEFRAME,
    START_DATE,
    END_DATE,
    VALID_TIMEFRAMES,
)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Binance Futures data for quantitative research.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Positional: tickers
    parser.add_argument(
        "tickers",
        nargs="*",
        default=None,
        help="Tickers to download (e.g., BTCUSDT ETHUSDT). If not specified, downloads all default tickers.",
    )

    # Timeframe
    parser.add_argument(
        "-t",
        "--timeframe",
        default=DEFAULT_TIMEFRAME,
        choices=VALID_TIMEFRAMES,
        help=f"Kline timeframe (default: {DEFAULT_TIMEFRAME})",
    )

    # Date range
    parser.add_argument(
        "--start",
        default=START_DATE,
        help=f"Start date YYYY-MM-DD (default: {START_DATE})",
    )
    parser.add_argument(
        "--end", default=END_DATE, help=f"End date YYYY-MM-DD (default: {END_DATE})"
    )

    # Data type selection
    parser.add_argument(
        "--ohlcv-only",
        action="store_true",
        help="Download only OHLCV klines (skip metrics/index/mark/premium)",
    )
    parser.add_argument(
        "--no-ohlcv",
        action="store_true",
        help="Skip OHLCV, download only Vision archive data",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip metrics (OI, LS ratio, taker ratio)",
    )
    parser.add_argument(
        "--no-index", action="store_true", help="Skip index price klines"
    )
    parser.add_argument("--no-mark", action="store_true", help="Skip mark price klines")
    parser.add_argument(
        "--no-premium", action="store_true", help="Skip premium index klines"
    )

    # Other options
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose (debug) logging"
    )
    parser.add_argument(
        "--list-default", action="store_true", help="List default tickers and exit"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle --list-default
    if args.list_default:
        print("Default tickers:")
        for i, ticker in enumerate(DEFAULT_TICKERS, 1):
            print(f"  {i:2d}. {ticker}")
        return 0

    setup_logging(args.verbose)

    # Determine tickers
    tickers: Optional[List[str]] = args.tickers if args.tickers else None

    # Determine data types
    if args.ohlcv_only:
        include_ohlcv = True
        include_metrics = include_index = include_mark = include_premium = False
    else:
        include_ohlcv = not args.no_ohlcv
        include_metrics = not args.no_metrics
        include_index = not args.no_index
        include_mark = not args.no_mark
        include_premium = not args.no_premium

    # Create downloader
    downloader = BinanceDataDownloader(
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        include_ohlcv=include_ohlcv,
        include_metrics=include_metrics,
        include_index=include_index,
        include_mark=include_mark,
        include_premium=include_premium,
    )

    # Run download
    results = downloader.download(tickers)

    # Summary
    success = sum(1 for r in results if "error" not in r)
    failed = len(results) - success

    print(f"\nSummary: {success} succeeded, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
