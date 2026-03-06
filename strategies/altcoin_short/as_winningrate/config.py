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
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))   # strategies/altcoin_short/as_winningrate/
_PROJECT_ROOT = os.path.join(os.path.dirname(_CONFIG_DIR),"../../")               # /
DATA_LOCATION = os.path.join(_PROJECT_ROOT, os.getenv("DATA_RELATIVE_LOCATION", "data"))        # data
BASE_DATA_PATH = DATA_LOCATION

# Asset class subdirectory
ASSET_CLASS = "cryptofuture"

# Exchange
EXCHANGE = "binance"

# ============================================================================
# DATE RANGE
# ============================================================================

START_DATE = "2026-02-03"
END_DATE = "2026-02-03"

# ============================================================================
# NETWORK
# ============================================================================

PROXY = "http://127.0.0.1:1082"
PROXIES = {"http": PROXY, "https": PROXY}

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
    # "SOLUSDT",
    # "XRPUSDT",
    # "DOGEUSDT",
    # "ADAUSDT",
    # "AVAXUSDT",
    # "DOTUSDT",
    # "LINKUSDT",
    # "POLUSDT",
    # "LTCUSDT",
    # "UNIUSDT",
    # "ATOMUSDT",
    # "ETCUSDT",
    # "FILUSDT",
    # "APTUSDT",
    # "NEARUSDT",
    # "ARBUSDT",
    # "OPUSDT",
    # "INJUSDT",
    # "SUIUSDT",
    # "TIAUSDT",
    # "SEIUSDT",
    # "STXUSDT",
    # "IMXUSDT",
    # "RUNEUSDT",
    # "AAVEUSDT",
    # "MKRUSDT",
    # "LDOUSDT",
]
TOP_N_SYMBOL=3

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
