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