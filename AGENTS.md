# PROJECT KNOWLEDGE BASE

**Generated:** 2026-02-06 22:56  
**Commit:** cb2ba63  
**Branch:** main  

## OVERVIEW
Lean Binance trading workspace for algorithmic trading using QuantConnect Lean Engine. Python environment with venv, data fetching via ccxt, and ML-driven strategies.

## STRUCTURE
```
./
├── BinanceBot/           # Main trading algorithm (Lean QCAlgorithm)
├── ccxt_data_fetch/      # Data fetching package (ccxt → Lean format)
├── PayoffAsymmetry/      # ML-driven trading strategies
├── data/                 # Local data storage (crypto/futures)
├── venv/                 # Python virtual environment
└── storage/              # Temporary storage
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Data fetching | `ccxt_data_fetch/` | Fetches Binance futures data, saves as Lean-compatible ZIP/CSV |
| Lean algorithm | `BinanceBot/main.py` | Simple BTCUSDT strategy |
| ML strategy | `PayoffAsymmetry/main.py` | Cross-sectional payoff asymmetry model |
| Research/backtest | `PayoffAsymmetry/research_script.py` | Feature engineering, labeling, model training |
| Configuration | `lean.json` | Lean engine config (brokerages, parameters) |
| Environment | `.env` | Credentials (referenced by lean.json) |
| Data location | `data/cryptofuture/binance/` | Hourly/minute data organized by symbol |

## CODE MAP
**Key modules:**
- `ccxt_data_fetch/fetcher.py` – Core data fetching with ccxt, rate limiting, progress bars
- `ccxt_data_fetch/utils.py` – Symbol formatting, time utilities, top-200 symbol list
- `ccxt_data_fetch/config.py` – Centralized constants (DATA_LOCATION, ASSET_CLASS, PROXIES)
- `ccxt_data_fetch/run.py` – Orchestration with resume logic for minute data

**Lean patterns:**
- `BinanceBot/main.py` – Standard QCAlgorithm structure (Initialize, OnData)
- `PayoffAsymmetry/main.py` – Multi-symbol, feature-based, rolling window strategy

## CONVENTIONS
- **Data storage**: `DATA_LOCATION/ASSET_CLASS/binance/{hour,minute}/{symbol}/`
  - Hourly: `<symbol>.zip` → `<symbol>.csv` (no headers)
  - Minute: per-day `<date>_trade.zip` → `<date>_trade.csv`
- **Symbol handling**: `format_symbol(symbol)` → lowercase, no pair suffix
- **Time handling**: UTC throughout, `get_ms_from_midnight()` for minute indexing
- **Logging**: `logger = logging.getLogger(__name__)` per module
- **Proxy**: `http://127.0.0.1:1082` configured in config.py and lean.json

## ANTI-PATTERNS (THIS PROJECT)
- No standard Python packaging (no pyproject.toml, setup.py)
- Entry points as standalone scripts (not console_scripts)
- No test directory or test scaffolding
- Multiple top-level packages without unified monorepo structure
- Empty `__init__.py` in ccxt_data_fetch (intentional but minimal)

## UNIQUE STYLES
- Region blocks (`# region imports` / `# endregion`) for code organization
- Compact CSV exports (no headers) to minimize storage
- Resume logic for minute data (checks existing ZIPs, skips completed dates)
- Feature naming: `HighPressure`, `VWAP_ZScore`, `VolumeProfile` (PayoffAsymmetry)

## COMMANDS
```bash
# Activate environment
source venv/bin/activate

# Backtest BinanceBot
lean backtest BinanceBot

# Live trading (requires API keys in lean.json)
lean live BinanceBot

# Download Binance data
lean data download --project BinanceBot --data-type-crypto --exchange Binance --tick-type Quote --resolution Hour

# Run data fetcher (hourly)
cd ccxt_data_fetch && python run.py hour

# Run data fetcher (minute)
cd ccxt_data_fetch && python run.py minute
```

## NOTES
- **Proxy required**: Workspace configured for proxy on port 1082
- **Data organization**: Follows Lean's expected structure for crypto futures
- **Module depth**: Shallow (2–3 levels), no deep nesting (>4)
- **Shared utilities**: `ccxt_data_fetch/utils.py` serves as common helpers
- **ML strategy**: PayoffAsymmetry uses 12h payoff labeling, 10 features, Huber/RandomForest
- **Credentials**: Stored in `~/.lean/credentials` and `.env` (not in repo)