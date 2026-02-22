# CCXT Data Fetch Module

## Purpose

This module automates the download and conversion of cryptocurrency futures market data from exchanges (primarily Binance) into LEAN-compatible format for local backtesting.

## Module Structure

```
ccxt_data_fetch/
├── config.py           # Configuration (API keys, paths, date ranges, proxies)
├── fetcher.py          # Core fetching logic using CCXT
├── utils.py            # Utility functions (symbol formatting, time conversion)
├── run.py              # Entry point for data download
├── __init__.py         # Module initialization
└── GEMINI.md           # This file
```

## Core Components

### 1. Configuration (`config.py`)

Defines project settings:
- **DATA_LOCATION**: Path to output data folder
- **ASSET_CLASS**: Type of asset (e.g., "cryptofuture")
- **START_DATE / END_DATE**: Date range for downloads
- **PROXIES**: Optional proxy configuration
- **EXCHANGES**: Exchange configuration

**Key Settings**:
```python
ASSET_CLASS = "cryptofuture"
START_DATE = "2026-01-01"
END_DATE = "2026-01-31"
```

### 2. Fetcher (`fetcher.py`)

Main data retrieval engine using CCXT library.

**Key Functions**:

#### `fetch_ohlcv_range(symbol, timeframe, since_ms, until_ms)`
- Fetches OHLCV (Open, High, Low, Close, Volume) data for a given symbol and timeframe
- Handles rate limiting and retries
- Parameters:
  - `symbol`: Trading pair (e.g., "BTC/USDT")
  - `timeframe`: "1m" for minute, "1h" for hourly
  - `since_ms`: Start timestamp in milliseconds
  - `until_ms`: End timestamp in milliseconds
- Returns: List of OHLCV candles

#### `save_minute_data(symbol, data)`
- Converts fetched data to LEAN minute-bar format
- Creates zip files organized by date
- Saves to: `data/cryptofuture/binance/minute/{symbol}/{YYYYMMDD}_trade.zip`
- Format: Time (ms from midnight), Open, High, Low, Close, Volume

#### `save_hour_data(symbol, data)`
- Converts fetched data to LEAN hourly format
- Consolidates all data in single file per symbol
- Saves to: `data/cryptofuture/binance/hour/{symbol}.zip`
- Format: Time (YYYYMMDD HH:mm), Open, High, Low, Close, Volume

### 3. Utilities (`utils.py`)

Helper functions for data processing:

#### `get_top_200_symbols()`
- Retrieves list of top 200 trading pairs by volume from Binance
- Used for batch downloading

#### `format_symbol(symbol)`
- Standardizes symbol naming for file paths
- Example: "BTC/USDT" → "BTCUSDT"

#### `get_ms_from_midnight(timestamp_ms)`
- Converts timestamp to milliseconds from midnight
- Used for LEAN's minute/second data format

### 4. Entry Point (`run.py`)

Main script to execute data downloads.

**Usage**:
```bash
# Download hourly data (default)
python -m ccxt_data_fetch.run

# Download minute data
python -m ccxt_data_fetch.run minute

# Download hourly data (explicit)
python -m ccxt_data_fetch.run hour
```

**Features**:
- Resume capability: Detects existing data and continues from latest date
- Progress tracking: Shows download progress with tqdm
- Error handling: Logs failures without stopping entire process
- Logging: Timestamped output to console

## Workflow

### Step 1: Configure Settings
Edit `config.py` to set:
- Output path (`DATA_LOCATION`)
- Date range (`START_DATE`, `END_DATE`)
- Asset class (`ASSET_CLASS`)
- Proxy if needed (`PROXIES`)

### Step 2: Run Download
```bash
python -m ccxt_data_fetch.run minute
```

### Step 3: Monitor Output
- Console shows progress for each symbol
- Download can be safely interrupted and resumed
- Data is automatically saved to `data/cryptofuture/binance/`

### Step 4: Verify Data
Check output structure:
```
data/cryptofuture/binance/
├── minute/
│   ├── BTCUSDT/
│   │   ├── 20260101_trade.zip
│   │   ├── 20260102_trade.zip
│   │   └── ...
│   └── ETHUSDT/
│       └── ...
└── hour/
    ├── BTCUSDT.zip
    └── ETHUSDT.zip
```

## Data Format Details

### Minute/Second Resolution
**File Organization**: Daily ZIP files with name format `YYYYMMDD_trade.zip`

**CSV Content** (no headers):
```
Time (ms from midnight), Open, High, Low, Close, Volume
100000, 50000.0, 50100.5, 49950.0, 50050.0, 123.45
101000, 50050.0, 50200.0, 50000.0, 50150.0, 456.78
```

### Hour/Daily Resolution
**File Organization**: Single consolidated ZIP per symbol

**CSV Content** (no headers):
```
Time (YYYYMMDD HH:mm), Open, High, Low, Close, Volume
20260101 00:00, 50000.0, 50100.5, 49950.0, 50050.0, 1234.56
20260101 01:00, 50050.0, 50200.0, 50000.0, 50150.0, 2345.67
```

## Advanced Usage

### Resume Interrupted Downloads
The module automatically detects existing data:
```bash
# If download stopped on day 15
python -m ccxt_data_fetch.run minute
# Automatically resumes from day 16
```

### Using Proxy
Edit `config.py`:
```python
PROXY = "http://your.proxy.com:port"
PROXIES = {"http": PROXY, "https": PROXY}
```

### Extending to New Symbols
Modify `utils.py`'s `get_top_200_symbols()`:
```python
def get_top_200_symbols():
    # Add custom symbols here
    return ["BTC/USDT", "ETH/USDT", "YOUR/SYMBOL"]
```

### Extending to New Exchanges
Modify `fetcher.py`:
```python
# Change from ccxt.binance to ccxt.bybit, etc.
exchange = ccxt.bybit({
    "enableRateLimit": True,
    "proxies": PROXIES,
})
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Rate limiting errors | Reduce number of symbols or increase delays in `fetcher.py` |
| Network timeouts | Enable proxy in `config.py` or check internet connection |
| Proxy connection failed | Verify proxy URL format and ensure proxy is running |
| Missing data days | Some exchanges don't have data for certain dates; check source |
| Symbolic permissions error | Ensure write permissions to `DATA_LOCATION` directory |

## Performance Notes

- **Rate Limits**: Binance has API rate limits; default configuration respects these
- **Download Speed**: Typically 50-100 symbols per hour (varies with resolution)
- **Storage**: Minute data is ~100-200 MB per symbol per month; Hour data is much smaller
- **Resume**: Designed to be interrupted and resumed safely without data loss

## Integration with LEAN

Once data is downloaded and formatted by this module:

1. Place in `data/cryptofuture/binance/` directory
2. LEAN engine automatically discovers the data
3. Reference in strategy code:
```python
# In your QCAlgorithm strategy
self.AddCryptoFuture("BTCUSDT", Resolution.Minute)
```

4. Data is ready for backtesting with `lean backtest`

## Dependencies

- **ccxt**: Unified cryptocurrency exchange API
- **pandas**: Data manipulation and analysis
- **tqdm**: Progress bar visualization
- **python-dotenv**: Environment variable loading

## Future Enhancements

- Support for tick data (currently minute/hour only)
- Multi-exchange parallel downloading
- Alternative resolution formats (second-level data)
- Margin rate data fetching
- Quote data (bid/ask) in addition to trades
