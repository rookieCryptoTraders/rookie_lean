# Lean Binance Trading Workspace

This workspace is set up for algorithmic trading on Binance using the QuantConnect Lean Engine.

## Setup Details
- **Python Environment**: `venv` is active.
- **Lean CLI**: Installed and configured.
- **Organization**: Chen Zhao (ID: `32428d7ae019266d4f87b05be51fb41c`)
- **Credentials**: Stored in `~/.lean/credentials` and `.env`.

## Directory Structure
- `BinanceBot/`: Your main trading algorithm.
- `data/`: Local data storage (currently empty/minimal).
- `lean.json`: Main Lean configuration.

## How to Run

### 1. Backtesting (Docker)
To run a backtest for the `BinanceBot` project:
```bash
source venv/bin/activate
lean backtest BinanceBot
```
*Note: This will automatically pull the `quantconnect/lean` Docker image if it's not present.*

### 2. Live Trading (Binance)
To run the algorithm live on Binance, you first need to add your API keys to `lean.json`:
1. Open `lean.json`.
2. Find `"binance-api-key"` and `"binance-api-secret"`.
3. Enter your Binance API credentials.
4. Run:
```bash
source venv/bin/activate
lean live BinanceBot
```
Follow the prompts to select the Binance brokerage.

### 3. Downloading Data
Since this setup is minimal, you can download Binance data using:
```bash
lean data download --project BinanceBot --data-type-crypto --exchange Binance --tick-type Quote --resolution Hour
```

## Proxy Configuration
If you encounter network issues, the workspace is configured to use the proxy on port `1082`. You can set it in your terminal:
```bash
export http_proxy=http://127.0.0.1:1082
export https_proxy=http://127.0.0.1:1082
```
