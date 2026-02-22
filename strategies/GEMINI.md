# Strategies Directory

## Purpose

This directory contains custom quantitative trading strategies built on QuantConnect's LEAN engine. Each strategy is a complete algorithm implementation that can be backtested against historical market data.

## Directory Structure

```
strategies/
├── altcoin_short/              # Altcoin short-selling strategy
│   └── as_winningrate/         # Winning rate analysis variant
└── GEMINI.md                   # This file
```

## Strategy Development Guide

### Creating a New Strategy

1. **Create Strategy Folder**:
   ```bash
   mkdir strategies/my_strategy
   cd strategies/my_strategy
   ```

2. **Create Main Algorithm File** (`main.py`):
   ```python
   from AlgorithmImports import *
   
   class MyStrategy(QCAlgorithm):
       def Initialize(self):
           # Set backtest parameters
           self.SetStartDate(2025, 1, 1)
           self.SetEndDate(2025, 12, 31)
           self.SetCash("USDT", 100000)
           
           # Add securities
           self.AddCryptoFuture("BTCUSDT", Resolution.Minute)
           
       def OnData(self, data):
           # Trading logic here
           pass
   ```

3. **Register with LEAN** (if using cloud):
   ```bash
   lean cloud push
   ```

4. **Backtest Locally**:
   ```bash
   lean backtest "my_strategy"
   ```

## Existing Strategies

### Altcoin Short Strategy (`altcoin_short/`)

**Purpose**: Implements a systematic short-selling strategy for altcoins based on momentum and volatility patterns.

**Variants**:
- `as_winningrate/`: Analyzes win rate metrics and performance statistics

**Key Characteristics**:
- Targets mid-cap and large-cap altcoins
- Uses technical indicators for entry/exit signals
- Implements risk management with position sizing
- Suitable for crypto futures markets

## Strategy Development Framework

### Algorithm Structure (QCAlgorithm)

Every strategy inherits from `QCAlgorithm`:

```python
class MyStrategy(QCAlgorithm):
    def Initialize(self):
        """Called once at backtest start"""
        pass
    
    def OnData(self, data):
        """Called on each data point"""
        pass
    
    def OnEndOfDay(self, symbol):
        """Called at end of each trading day"""
        pass
```

### Essential Methods

#### Initialize()
Set backtest parameters:
```python
def Initialize(self):
    self.SetStartDate(2025, 1, 15)      # Backtest start date
    self.SetEndDate(2025, 2, 1)         # Backtest end date
    self.SetCash("USDT", 100000)        # Initial cash
    self.SetBrokerageModel(BrokerageName.BinanceFutures)
    self.AddCryptoFuture("BTCUSDT", Resolution.Minute)
```

#### OnData(data)
Main trading logic triggered on data updates:
```python
def OnData(self, data):
    # Check if we have data
    if data["BTCUSDT"] is None:
        return
    
    # Execute trades
    current_price = data["BTCUSDT"].Close
    
    if not self.Portfolio.Invested:
        self.Buy("BTCUSDT", 10)  # Buy 10 contracts
    elif current_price > target_price:
        self.Liquidate("BTCUSDT")  # Close position
```

### Common Indicators

LEAN provides built-in indicators:

```python
# Simple Moving Average
sma = self.SMA("BTCUSDT", 20, Resolution.Daily)

# Relative Strength Index
rsi = self.RSI("BTCUSDT", 14)

# Bollinger Bands
bb = self.BB("BTCUSDT", 20, 2)

# MACD
macd = self.MACD("BTCUSDT", 12, 26, 9)
```

### Portfolio Management

```python
# Check if invested
if not self.Portfolio.Invested:
    self.Buy("BTCUSDT", quantity)

# Get position details
position = self.Portfolio["BTCUSDT"]
quantity = position.Quantity
entry_price = position.AveragePrice
unrealized_pnl = position.UnrealizedProfit

# Close position
self.Liquidate("BTCUSDT")

# Set stop loss
self.StopMarketOrder("BTCUSDT", -quantity, stop_price)
```

### Asset Classes Available

Depending on data in `/data` folder:

```python
# Crypto Futures
self.AddCryptoFuture("BTCUSDT", Resolution.Minute)

# Spot Crypto
self.AddCrypto("BTC", currency="USD", Resolution.Daily)

# Equities
self.AddEquity("AAPL", Resolution.Daily)

# Forex
self.AddForex("EURUSD", Resolution.Hour)

# Futures
self.AddFuture(Futures.Indices.SP500EMini)

# Options
self.AddOption("SPY")
```

## Example: Simple Moving Average Crossover

```python
from AlgorithmImports import *

class MovingAverageCrossover(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2025, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash("USDT", 100000)
        self.SetBrokerageModel(BrokerageName.BinanceFutures)
        
        self.symbol = "BTCUSDT"
        self.AddCryptoFuture(self.symbol, Resolution.Hour)
        
        # Indicators
        self.sma_fast = self.SMA(self.symbol, 10, Resolution.Hour)
        self.sma_slow = self.SMA(self.symbol, 30, Resolution.Hour)
    
    def OnData(self, data):
        if not self.sma_fast.IsReady or not self.sma_slow.IsReady:
            return
        
        # Golden cross
        if self.sma_fast.Current.Value > self.sma_slow.Current.Value:
            if not self.Portfolio.Invested:
                self.Buy(self.symbol, 10)
        
        # Death cross
        elif self.sma_fast.Current.Value < self.sma_slow.Current.Value:
            if self.Portfolio.Invested:
                self.Liquidate(self.symbol)
```

## Testing and Optimization

### Local Backtesting

```bash
# Run backtest
lean backtest "my_strategy"

# Output includes:
# - Total Return %
# - Sharpe Ratio
# - Drawdown metrics
# - Trade statistics
# - Equity curve
```

### Parameter Optimization

For parameter sweeps, use LEAN's optimization:

```python
# In config.json
{
    "parameters": {
        "fast_period": {"min": 5, "max": 20, "step": 5},
        "slow_period": {"min": 20, "max": 60, "step": 10}
    }
}
```

### Performance Metrics

Monitor these metrics:
- **Sharpe Ratio**: Risk-adjusted return (target > 1.0)
- **Sortino Ratio**: Return vs downside volatility
- **Calmar Ratio**: Return vs max drawdown
- **Win Rate**: % of winning trades
- **Profit Factor**: Gross profit / Gross loss

## Best Practices

### 1. Risk Management
```python
# Implement position sizing
max_position_size = 0.05  # 5% of portfolio per position

# Set stop losses
self.StopMarketOrder(symbol, -quantity, current_price * 0.95)

# Implement profit targets
self.LimitOrder(symbol, -quantity, current_price * 1.05)
```

### 2. Data Validation
```python
# Always check if data exists
if data is None or symbol not in data:
    return

# Verify indicator readiness
if not indicator.IsReady:
    return
```

### 3. Logging and Debugging
```python
# Log important events
self.Log(f"Buying {quantity} at {current_price}")

# Track performance
self.Plot("Strategy", "Sharpe", self.portfolio.sharpe_ratio)
```

### 4. Avoid Common Pitfalls
- **Look-ahead bias**: Only use data available at decision time
- **Survivorship bias**: Include delisted securities in backtest
- **Optimization bias**: Test on out-of-sample data
- **Overfitting**: Keep strategies simple and robust

## Deployment

### Paper Trading
Test on live market without real capital:
```bash
lean live "my_strategy" --paper
```

### Live Trading
Deploy with real capital (use with caution):
```bash
lean live "my_strategy"
```

## Resources

- [QCAlgorithm API Reference](https://www.quantconnect.com/docs/api-reference)
- [Algorithm Structure Guide](https://www.quantconnect.com/tutorials)
- [Indicator Library](https://www.quantconnect.com/indicators)
- [Community Strategies](https://www.quantconnect.com/community)

## Reference Example

See `/example/PayoffAsymmetry/` for a complete reference implementation:
- Cross-sectional analysis strategy
- Multiple crypto futures positions
- Advanced statistical methods (Huber regression)
- Proper data handling and logging

## Support

For issues or questions:
1. Check LEAN documentation
2. Review reference strategies
3. Consult QuantConnect community forum
4. Check ccxt data availability for symbols
