# ccxt_data_fetch – Data Fetching Package v2.0

**Updated:** 2026-02-07  
**Version:** 2.0.0

## OVERVIEW

Professional data fetching package for Binance Futures, supporting:
- **OHLCV Klines** via CCXT API
- **Microstructure Data** via Binance Vision archives (Metrics, Index, Mark, Premium)

## QUICK START

### Command Line
```bash
# Download all data for specific tickers
python -m ccxt_data_fetch BTCUSDT ETHUSDT

# Download only OHLCV (skip Vision data)
python -m ccxt_data_fetch --ohlcv-only BTCUSDT

# Download only Vision data (skip OHLCV)
python -m ccxt_data_fetch --no-ohlcv BTCUSDT

# Change timeframe
python -m ccxt_data_fetch --timeframe 4h BTCUSDT

# Custom date range
python -m ccxt_data_fetch --start 2025-06-01 --end 2025-12-31 BTCUSDT

# Download all default tickers
python -m ccxt_data_fetch
```

### Python API
```python
from ccxt_data_fetch import BinanceDataDownloader

# Download everything for specific tickers
downloader = BinanceDataDownloader(timeframe="1h")
downloader.download(["BTCUSDT", "ETHUSDT"])

# Download only OHLCV
downloader = BinanceDataDownloader(
    include_metrics=False,
    include_index=False,
    include_mark=False,
    include_premium=False
)
downloader.download(["BTCUSDT"])

# Custom date range
downloader = BinanceDataDownloader(
    start_date="2025-06-01",
    end_date="2025-12-31"
)
downloader.download(["BTCUSDT"])
```

## PACKAGE STRUCTURE

```
ccxt_data_fetch/
├── __init__.py          # Package exports
├── __main__.py          # Module entry point
├── cli.py               # Command-line interface
├── config.py            # Configuration constants
├── downloader.py        # ★ Main downloader class
├── fetcher.py           # Legacy OHLCV fetcher (deprecated)
├── run.py               # Legacy entry point (deprecated)
├── utils.py             # Utility functions
└── AGENTS.md            # This documentation
```

## OUTPUT STRUCTURE

```
data/cryptofuture/binance/
├── 1h/                  # OHLCV Klines
│   ├── btcusdt.csv
│   └── ethusdt.csv
├── extra/               # Metrics (OI, LS Ratio, Taker Ratio)
│   ├── btcusdt_extra.csv
│   └── ethusdt_extra.csv
├── extra_index/         # Index Price OHLC
│   ├── btcusdt_index.csv
│   └── ethusdt_index.csv
├── extra_mark/          # Mark Price OHLC
│   ├── btcusdt_mark.csv
│   └── ethusdt_mark.csv
├── extra_premium/       # Premium Index OHLC
│   ├── btcusdt_premium.csv
│   └── ethusdt_premium.csv
└── raw_*/               # Raw downloaded files (can be deleted)
```

## DATA COLUMNS

| File Type    | Columns                                  |
|--------------|------------------------------------------|
| OHLCV        | time, open, high, low, close, volume     |
| Metrics      | time, openInterest, ls_ratio, taker_ratio|
| Index/Mark   | time, open, high, low, close             |
| Premium      | time, open, high, low, close             |

**Time format:** `YYYY-MM-DD HH:MM:SS` (UTC, no timezone suffix)

## CONFIGURATION

Edit `config.py` to change:

| Variable              | Default           | Description                    |
|-----------------------|-------------------|--------------------------------|
| `DATA_ROOT`           | `.../lean_workspace/data` | Data storage root    |
| `START_DATE`          | `2025-01-01`      | Download start date            |
| `END_DATE`            | `2026-01-31`      | Download end date              |
| `PROXY`               | `http://127.0.0.1:1082` | Network proxy          |
| `MAX_WORKERS_FILES`   | `10`              | Parallel file downloads        |
| `MAX_WORKERS_TICKERS` | `3`               | Parallel ticker processing     |
| `DEFAULT_TICKERS`     | 30 symbols        | Default ticker list            |

## CLI OPTIONS

```
usage: python -m ccxt_data_fetch [-h] [-t TIMEFRAME] [--start DATE] [--end DATE]
                                 [--ohlcv-only] [--no-ohlcv] [--no-metrics]
                                 [--no-index] [--no-mark] [--no-premium]
                                 [-v] [--list-default] [tickers ...]

positional arguments:
  tickers         Tickers to download (e.g., BTCUSDT ETHUSDT)

options:
  -t, --timeframe Kline timeframe: 1m, 5m, 15m, 1h, 4h, 1d (default: 1h)
  --start         Start date YYYY-MM-DD
  --end           End date YYYY-MM-DD
  --ohlcv-only    Download only OHLCV klines
  --no-ohlcv      Skip OHLCV, download only Vision data
  --no-metrics    Skip metrics (OI, LS ratio, taker ratio)
  --no-index      Skip index price klines
  --no-mark       Skip mark price klines
  --no-premium    Skip premium index klines
  -v, --verbose   Enable debug logging
  --list-default  List default tickers and exit
```

## FEATURE ENGINEERING IDEAS

```python
import pandas as pd

# Load data
ohlcv = pd.read_csv("data/cryptofuture/binance/1h/btcusdt.csv")
index = pd.read_csv("data/cryptofuture/binance/extra_index/btcusdt_index.csv")
extra = pd.read_csv("data/cryptofuture/binance/extra/btcusdt_extra.csv")

# Parse time
for df in [ohlcv, index, extra]:
    df['time'] = pd.to_datetime(df['time'])

# Merge on time
merged = ohlcv.merge(index[['time', 'close']], on='time', suffixes=('', '_index'))
merged = merged.merge(extra, on='time')

# Basis (Futures Premium)
merged['basis'] = (merged['close'] - merged['close_index']) / merged['close_index']

# OI Change
merged['oi_change_12h'] = merged['openInterest'].pct_change(12)

# Crowding Signal
merged['crowding'] = merged['ls_ratio'] - merged['ls_ratio'].rolling(24).mean()
```

## NOTES

- **Proxy required:** Default `http://127.0.0.1:1082`; adjust in `config.py`
- **Resume support:** Existing files are skipped automatically
- **Data availability:** Binance Vision archives start from ~2020
- **Metrics frequency:** Native 5-min, resampled to match OHLCV timeframe