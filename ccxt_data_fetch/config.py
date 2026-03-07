"""
ccxt_data_fetch Configuration
=============================
Central configuration for the data fetching package.
"""

import os
# load .env variables from a .env file if present
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Root directory for all data storage
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))   # ccxt_data_fetch/
_PROJECT_ROOT = os.path.dirname(_CONFIG_DIR)               # rookie_lean/
DATA_LOCATION = os.path.join(_PROJECT_ROOT, os.getenv("DATA_RELATIVE_LOCATION", "data"))        # rookie_lean/data
BASE_DATA_PATH = DATA_LOCATION

# Asset class subdirectory
ASSET_CLASS = "cryptofuture"

# Exchange
EXCHANGE = "binance"

# ============================================================================
# DATE RANGE
# ============================================================================

START_DATE = "2026-01-01"
END_DATE = "2026-02-22"

# ============================================================================
# NETWORK
# ============================================================================

# Proxy: set PROXY in .env or USE_PROXY=false to disable. Use PROXY= to run without proxy.
_PROXY_RAW = os.getenv("PROXY", "http://127.0.0.1:1082").strip()
USE_PROXY = os.getenv("USE_PROXY", "true").lower() in ("1", "true", "yes")
PROXY = _PROXY_RAW if (USE_PROXY and _PROXY_RAW) else ""
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else {}
# When proxy is disabled, requests in fetcher/downloader use no proxy (direct connection).

# Request timeout in seconds
REQUEST_TIMEOUT = 15

# ============================================================================
# PERFORMANCE
# ============================================================================

# Maximum parallel downloads per ticker
MAX_WORKERS_FILES = 10

# Maximum parallel tickers (be careful with API rate limits for CCXT)
MAX_WORKERS_TICKERS = 3

# ============================================================================
# DEFAULT TICKERS
# ============================================================================

DEFAULT_TICKERS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "POLUSDT",
    "LTCUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "ETCUSDT",
    "FILUSDT",
    "APTUSDT",
    "NEARUSDT",
    "ARBUSDT",
    "OPUSDT",
    "INJUSDT",
    "SUIUSDT",
    "TIAUSDT",
    "SEIUSDT",
    "STXUSDT",
    "IMXUSDT",
    "RUNEUSDT",
    "AAVEUSDT",
    "MKRUSDT",
    "LDOUSDT",
]
TOP_N_SYMBOL=10

# Ticker name mapping (for rebranded assets)
TICKER_ALIASES = {
    "POLUSDT": "maticusdt",
    "MATICUSDT": "maticusdt",
}

# ============================================================================
# DATA TYPES
# ============================================================================

# Available timeframes for kline data
VALID_TIMEFRAMES = [
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
]

# Default timeframe
DEFAULT_TIMEFRAME = "1h"

# ============================================================================
# L1 QUOTE (custom layout for LEAN strategy)
# ============================================================================
# Saves to DATA_LOCATION/custom/<QUOTE_CUSTOM_MAP>/<symbol>/minute/<date>_quote.zip
# Strategy expects CUSTOM_QUOTE_MAP = "cryptofuture-quote" (see strategies/.../config.py).
QUOTE_CUSTOM_MAP = "cryptofuture-quote"
QUOTE_RESOLUTION_FOLDER = "minute"
