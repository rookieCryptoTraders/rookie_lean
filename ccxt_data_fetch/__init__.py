"""
ccxt_data_fetch
===============
Professional data fetching package for Binance Futures.

Quick Start:
    from ccxt_data_fetch import BinanceDataDownloader

    # Download all data for specific tickers
    downloader = BinanceDataDownloader(timeframe="1h")
    downloader.download(["BTCUSDT", "ETHUSDT"])

    # Download only OHLCV
    downloader = BinanceDataDownloader(include_metrics=False, include_index=False,
                                        include_mark=False, include_premium=False)
    downloader.download(["BTCUSDT"])

CLI Usage:
    python -m ccxt_data_fetch BTCUSDT ETHUSDT
    python -m ccxt_data_fetch --timeframe 4h --no-ohlcv BTCUSDT
    python -m ccxt_data_fetch --help
"""

from .downloader import BinanceDataDownloader
from .config import (
    DEFAULT_TICKERS,
    DEFAULT_TIMEFRAME,
    BASE_DATA_PATH,
    START_DATE,
    END_DATE,
    VALID_TIMEFRAMES,
)

__version__ = "2.0.0"
__all__ = [
    "BinanceDataDownloader",
    "DEFAULT_TICKERS",
    "DEFAULT_TIMEFRAME",
    "BASE_DATA_PATH",
    "START_DATE",
    "END_DATE",
    "VALID_TIMEFRAMES",
]
