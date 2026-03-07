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

[naming]
# QuantConnect/LEAN naming conventions (PascalCase for classes, snake_case for methods/variables in Python)
class_naming = pascal_case
method_naming = snake_case
variable_naming = snake_case
constant_naming = upper_snake_case
algorithm_class_name = *Algorithm  # All algorithm classes must end with "Algorithm" (e.g., EquityMomentumAlgorithm)
indicator_name_format = {IndicatorType}_{AssetSymbol}_{Period}  # e.g., EMA_SPY_20, RSI_AAPL_14
lean_constant_examples = BrokerageName.BINANCE_FUTURES, AccountType.MARGIN, SecurityType.CRYPTO_FUTURE, Market.BINANCE, OrderStatus.filled, Resolution.MINUTE
lean_function_examples = Symbol.create, 
lean_attribute_examples = self.settings.rebalance_portfolio_on_insight_changes, self.settings.rebalance_portfolio_on_security_changes, self.portfolio.values(), self.portfolio.total_portfolio_value, OrderEvent.status, OrderEvent.fill_quantity, OrderEvent.symbol.value, self.time,self.schedule.on(), self.date_rules.every_day(),self.time_rules.every(timedelta(hours=1))


# QuantConnect LEAN Project Rules (Quant Developer)
# Purpose: Enforce QuantConnect API usage, performance, and quant-specific best practices

[natual_language]
language = english

[core]
# Base language and framework constraints
language = python3.11  # This workspace uses LEAN Python algorithms
framework = .NET 6+  # Minimum .NET version for modern LEAN
qc_api_version = 10000+  # Target latest LEAN API version (QC's versioning)

[naming]

**NOTE: maybe there are some functions naming in pascal_case or camel case, they are out of date. In the newest version of lean, only contains snake_case functions. If you are not sure, read the source code.**

# QuantConnect/LEAN Python naming conventions
# - PascalCase 仅用于类型（类名、枚举名等）
# - snake_case 用于所有 Python 侧的方法、属性、变量
class_naming = pascal_case            # e.g. AltcoinShortAlgorithm, AltcoinShortAlphaModel
method_naming = snake_case            # e.g. initialize, on_data, on_order_event, update, on_securities_changed
variable_naming = snake_case          # e.g. max_positions, selection_weight_power, coin_data
constant_naming = upper_snake_case    # e.g. REGIME_BULL, MAX_SPREAD_PERCENT
algorithm_class_name = *Algorithm     # 所有算法类以 "Algorithm" 结尾 (e.g., AltcoinShortAlgorithm)
indicator_name_format = {IndicatorType}_{AssetSymbol}_{Period}  # e.g. EMA_SPY_20, RSI_AAPL_14
lean_type_hint_format = use modern python type hint. e.g. list[int], dict[str, int], tuple[int, str], set[int], tuple[int, str] | set[int] etc.

# LEAN API Python 访问约定（尽量使用 snake_case 别名）
lean_constant_examples = BrokerageName.BINANCE_FUTURES, AccountType.MARGIN, SecurityType.CRYPTO_FUTURE, Market.BINANCE, OrderStatus.filled, Resolution.MINUTE, InsightDirection.DOWN
lean_function_examples = Insight.price, Symbol.create, Algorithm.debug,  self.schedule.on, self.date_rules.every_day, self.time_rules.every, self.add_risk_management(), self.set_execution(), self.set_portfolio_construction(), self.set_alpha(), list(self.portfolio.values())[0].unrealized_profit , list(self.portfolio.values())[0].holdings_value ,self.portfolio.total_unrealised_profit, self.settings.rebalance_portfolio_on_insight_changes, changes.added_securities, changes.removed_securities, self.settings.rebalance_portfolio_on_security_changes, self.portfolio.values(), self.portfolio.total_portfolio_value, OrderEvent.status, OrderEvent.fill_quantity, OrderEvent.symbol.value, self.time
lean_attribute_examples = slice.contains_key


[structure]
# Enforce LEAN algorithm structure (required methods and properties)
# Python algorithms use snake_case lifecycle methods; keep PascalCase only as QC framework aliases.
required_algorithm_methods = initialize, on_data  # Core LEAN lifecycle methods
optional_algorithm_methods = on_order_event, on_end_of_day, on_securities_changed, on_error
required_algorithm_properties = cash, portfolio, securities  # Core LEAN algorithm properties
data_feed_integration = alpha_streams, market_hours, extended_hours  # Validate data feed compliance

[quantconnect_api]
# Mandatory QC API usage rules (avoid anti-patterns)
forbidden_methods = System.Threading.Thread.Sleep, System.Timers.Timer  # LEAN is event-driven; avoid blocking
required_api_patterns = 
  Selection:QCAlgorithm.add_universe,
  Execution:QCAlgorithm.market_order/limit_order/stop_order,
  Risk:QCAlgorithm.set_risk_management,
  Indicators:QCAlgorithm.INDICATOR_NAME(period)  # e.g., ema(20), rsi(14)
indicator_calculation = use_lean_indicators  # Forbid custom indicator implementations (use QC's optimized versions)
data_access = use_subscription_manager  # Use QC's SubscriptionManager instead of direct data downloads
order_sizing = use_quantity_or_target  # Enforce QC's order sizing (Quantity/TargetPercent)

[performance]
# Quant-specific performance rules (critical for backtesting/live trading)
max_backtest_data_points = 1000000  # Prevent excessive data loading
max_indicators_per_asset = 5  # Avoid over-calculation
forbidden_loops = nested_loops_over_historical_data  # Prevent O(n²) slowdowns
async_patterns = use_qc_async  # Use LEAN's async methods (e.g., DownloadDataAsync)

[risk_management]
# Mandatory risk controls (QC compliance)
required_risk_checks = 
  MaxDrawdown(0.1),  # 10% max drawdown
  MaxPositionSize(0.05),  # 5% of portfolio per asset
  StopLoss(0.02)  # 2% stop-loss per trade
order_validation = enforce_slippage_and_commission  # Must set Slippage/Commission models
forbidden_trading = naked_options, leverage_over_2x  # High-risk strategies prohibited

[testing]
# QuantConnect backtesting/validation rules
required_backtest_parameters = 
  StartDate(2020-01-01),
  EndDate(2025-01-01),
  InitialCapital(100000),
  Resolution(Hour)  # Standard resolution for backtesting
required_metrics = 
  SharpeRatio(1),  # Min Sharpe Ratio > 1
  SortinoRatio(1.5),  # Min Sortino Ratio > 1.5
  WinRate(0.5)  # Min win rate 50%
live_trading_prechecks = paper_trading_validation, log_verbosity(Info)  # Require paper trading first

[logging]
# QC logging standards (avoid sensitive data)
log_level = Info  # Default log level (Debug for development only)
forbidden_log_data = api_keys, personal_data, raw_order_data  # No sensitive info in logs
required_log_events = order_filled, position_opened/closed, error_occurred  # Critical events to log

[compliance]
# QuantConnect platform compliance
lean_config_validation = true  # Validate lean.json configuration
qc_cloud_compatibility = true  # Ensure compatibility with QC Cloud
version_control = track_algorithm_versions  # Require versioning for algorithm changes

[resources]
- [LEAN Data Formats](https://www.quantconnect.com/data/tree/)
- [LEAN CLI Guide](https://www.lean.io/docs/v2/lean-cli)
- [QuantConnect Docs](https://www.quantconnect.com/docs)