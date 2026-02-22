from datetime import datetime, timedelta

# --- Configuration ---
DATA_DIR = "../../../data/cryptofuture/binance/minute"
START_DATE = datetime(2026, 2, 3)  # Training: 2025 Full Year
END_DATE = datetime(2026, 2, 6)  # OOT Test: 2026 Jan

TICKERS = [
    "btcusdt",
    "ethusdt",
    # "bnbusdt",
    # "solusdt",
    # "xrpusdt",
    # "dogeusdt",
    # "adausdt",
    # "avaxusdt",
    # "dotusdt",
    # "linkusdt",
    # "maticusdt",
    # "ltcusdt",
    # "uniusdt",
    # "atomusdt",
    # "etcusdt",
    # "filusdt",
    # "aptusdt",
    # "nearusdt",
    # "arbusdt",
    # "opusdt",
    # "injusdt",
    # "suiusdt",
    # "tiausdt",
    # "seiusdt",
    # "stxusdt",
    # "imxusdt",
    # "runeusdt",
    # "aaveusdt",
    # "mkrusdt",
    # "ldousdt",
]
