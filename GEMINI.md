# Crypto Quantitative Trading Project (LEAN-based)

## Project Overview

This is a quantitative trading strategy project built on **QuantConnect's LEAN engine**. It focuses on algorithmic trading of cryptocurrency futures, equities, and other asset classes with local backtesting capabilities.

The project architecture consists of three core components:

1. **Data Layer** (`data/`) - Multi-asset financial data storage in LEAN format
2. **Data Fetching** (`ccxt_data_fetch/`) - Python utilities for downloading and converting market data
3. **Strategy Development** (`strategies/` & `example/`) - Algorithm implementation and backtesting

## Quick Start

### Local Backtesting with LEAN

```bash
# Run a backtest on a local strategy
lean backtest "My Project"

# Pull cloud projects to local
lean cloud pull

# Create new project from template
lean create-project "New Strategy"
```

### Configuration

The `lean.json` file contains the LEAN engine configuration:
- `data-folder`: Points to local data directory (`data/`)
- Data providers, logging, and API handlers are pre-configured
- Supports multiple brokerages: Binance, Interactive Brokers, OANDA, FXCM, Tradier, etc.

## Project Structure

### `/data` - Financial Data Storage
Contains market data organized by asset class, market, and resolution:

```
data/
├── crypto/              # Spot cryptocurrencies (Binance, Bitfinex, Bybit, Coinbase)
├── cryptofuture/        # Crypto futures (Binance, Bybit, dYdX)
├── equity/              # Stocks (USA, India)
├── forex/               # Foreign exchange (OANDA, FXCM)
├── future/              # Traditional futures (CME, COMEX, EUREX, etc.)
├── option/              # Options (USA)
├── index/               # Index data
├── cfd/                 # CFDs (OANDA)
├── market-hours/        # Market hours database
└── symbol-properties/   # Security database and symbol properties
```

**Data Organization**: `data/<assetClass>/<market>/<resolution>/<symbol>/`

**Resolutions**: `tick`, `second`, `minute`, `hour`, `daily`

**File Format**: CSV files compressed as ZIP, following LEAN standards

### `/ccxt_data_fetch` - Data Fetching Pipeline
Python-based data downloader using CCXT library to fetch market data from exchanges.

**Key Features**:
- Fetches OHLCV data from Binance Futures
- Converts to LEAN format automatically
- Supports minute and hourly resolution
- Resume capability for interrupted downloads
- Proxy support for geographical restrictions

**Usage**: See [ccxt_data_fetch/GEMINI.md](ccxt_data_fetch/GEMINI.md)

### `/strategies` - Strategy Development
Directory for custom trading algorithms and strategies.

**Current Strategies**:
- `altcoin_short/` - Altcoin short-selling strategy with winning rate analysis

**Usage**: See [strategies/GEMINI.md](strategies/GEMINI.md)

### `/example` - Reference Implementations
Pre-built example strategies for learning and reference:

- **PayoffAsymmetry**: Strategy analyzing cross-sectional payoff asymmetry across top 30 crypto futures
  - Uses Huber regression for robust statistical analysis
  - Backtests on Binance Futures data
  - Demonstrates LEAN algorithm structure with QCAlgorithm base class

## Key Technologies

| Component | Technology |
|-----------|-----------|
| Backtesting Engine | QuantConnect LEAN (C#) |
| Strategy Development | Python, C# |
| Data Fetching | Python, CCXT Library |
| Data Format | CSV (ZIP compressed) |
| Configuration | JSON |
| Supported Assets | Crypto, Equities, Forex, Futures, Options |

## Common Workflows

### 1. Download Crypto Futures Data
```bash
cd ccxt_data_fetch
python -m ccxt_data_fetch.run minute  # or 'hour'
```
Data is automatically formatted and saved to `data/cryptofuture/binance/`

### 2. Create New Strategy
```bash
lean create-project "My Strategy"
# Edit the main algorithm file in MyStrategy/main.py
# Reference example strategies in /example or /strategies
```

### 3. Backtest Locally
```bash
lean backtest "My Strategy"
# Results appear in backtest output
```

### 4. Cloud Synchronization
```bash
lean cloud push   # Upload to QuantConnect
lean cloud pull   # Download from QuantConnect
```

## Development Notes

- **Resolution Hierarchy**: Minute → Hourly → Daily. Higher resolution data can be aggregated to lower resolutions.
- **Data Formats**: Different asset classes may have slight variations in CSV format (e.g., crypto vs. equity vs. futures).
- **Historical Data**: Some resolutions are limited by exchange availability (e.g., older crypto data may only have hourly).
- **Margin Calculation**: Crypto futures include special `margin_interest` rate data in dedicated folders.

## Documentation Resources

- [LEAN Data Formats](https://www.quantconnect.com/data/tree/)
- [LEAN CLI Guide](https://www.lean.io/docs/v2/lean-cli)
- [CCXT Library Documentation](https://docs.ccxt.com/)
- [QuantConnect Docs](https://www.quantconnect.com/docs)

## Next Steps

1. Review strategy examples in `/example/PayoffAsymmetry/`
2. Explore data structures in `/data/` to understand available assets
3. Implement custom strategies in `/strategies/`
4. Use `ccxt_data_fetch/` to extend data coverage
5. Run local backtests with `lean backtest`
