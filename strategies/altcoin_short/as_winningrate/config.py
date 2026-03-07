"""
Strategy configuration for as_winningrate (crypto futures).
All data paths use QuantConnect custom layout; LEAN uses Globals.data_folder in Docker.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# LEAN backtest: custom data paths are under Globals.data_folder (set by lean.json
# "data-folder": "data"). Ensure the project's data directory is mounted in Docker.
# Research/scripts: _PROJECT_ROOT and DATA_LOCATION point to repo data folder.

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_project_root():
    """Find directory that contains 'data' so paths work locally and in Docker."""
    candidate = _CONFIG_DIR
    while candidate and candidate != os.path.dirname(candidate):
        if os.path.isdir(os.path.join(candidate, "data")):
            return candidate
        candidate = os.path.dirname(candidate)
    return os.path.normpath(os.path.join(_CONFIG_DIR, "..", "..", ".."))

_PROJECT_ROOT = _find_project_root()
DATA_LOCATION = os.path.join(_PROJECT_ROOT, os.getenv("DATA_RELATIVE_LOCATION", "data"))
BASE_DATA_PATH = DATA_LOCATION

# ============================================================================
# CUSTOM DATA LAYOUT (QuantConnect)
# ============================================================================
# Depth and quote data: data/custom/<MAP>/<TICKER>/minute/<YYYYMMDD>_*.zip
# Use this layout so LEAN's FileSystemDataFeed and Docker mounts resolve correctly.
# See DATA_LAYOUT.md in this directory.

CUSTOM_DEPTH_MAP = "cryptofuture-depth"
CUSTOM_QUOTE_MAP = "cryptofuture-quote"
CUSTOM_RESOLUTION_FOLDER = "minute"

# Legacy (for reference / research scripts only; PythonData get_source uses custom layout)
ASSET_CLASS = "cryptofuture"
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
