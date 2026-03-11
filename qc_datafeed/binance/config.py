"""
Binance datafeed configuration.
Data is stored under DATA_ROOT following LEAN folder structure.
"""

import os

# Project root (parent of qc_datafeed)
_QC_DATAFEED_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_QC_DATAFEED_DIR))

# Root directory for all data (LEAN data folder; relative to project root)
DATA_ROOT = os.path.join(_PROJECT_ROOT, os.getenv("DATA_RELATIVE_LOCATION", "data"))

# Default symbols (Binance futures/spot USDT pairs)
DEFAULT_SYMBOLS = [
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
    "MATICUSDT",
    "LTCUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "ETCUSDT",
    "FILUSDT",
    "APTUSDT",
    "NEARUSDT",
    "ARBUSDT",
    "OPUSDT",
]

# Margin type for futures: um (USD-M) or cm (COIN-M). Default um.
DEFAULT_MARGIN_TYPE = "um"

# Binance Vision base URL
BINANCE_VISION_BASE = "https://data.binance.vision/data"

# Download timeout (seconds)
DOWNLOAD_TIMEOUT = 90
MAX_RETRIES = 3
RETRY_DELAY_SEC = 5

# Optional proxy (set PROXY in env, e.g. http://127.0.0.1:1082)
_PROXY = os.getenv("PROXY", "").strip()
USE_PROXY = os.getenv("USE_PROXY", "true").lower() in ("1", "true", "yes")
PROXIES = {"http": _PROXY, "https": _PROXY} if (USE_PROXY and _PROXY) else {}
