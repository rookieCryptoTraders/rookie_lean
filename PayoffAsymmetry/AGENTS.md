# PayoffAsymmetry – ML‑Driven Trading Strategies

**Generated:** 2026-02-06  
**Commit:** cb2ba63  

## OVERVIEW
Cross‑sectional payoff‑asymmetry strategy using machine learning (Huber/RandomForest) to rank crypto futures and select top‑N positions.

## STRUCTURE
```
PayoffAsymmetry/
├── main.py              # Lean QCAlgorithm: multi‑symbol, rolling‑window ML strategy
├── research_script.py   # Offline feature engineering, labeling, model training
└── backtests/           # Generated backtest outputs (date‑stamped subfolders)
    └── YYYY‑MM‑DD_HH‑MM‑SS/code/
        ├── main.py      # Copy of strategy code for that backtest
        └── research_script.py
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Live/backtest strategy | `main.py` | `CrossSectionPayoffAsymmetry` class; uses `Train`, `Schedule`, `OnData` |
| Feature engineering research | `research_script.py` | Loads raw data, computes 10 features, labels with 12‑hour payoff |
| Model training | `research_script.py` | `RandomForestRegressor` (research) / `HuberRegressor` (live) |
| Generated backtests | `backtests/` | Each subfolder contains code snapshot and results |

## CONVENTIONS (PayoffAsymmetry specific)
- **Feature set**: 10 engineered features per symbol (e.g., `HighPressure`, `VWAP_ZScore`, `VolumeProfile`)
- **Labeling**: 12‑hour forward return (`payoff`) used as regression target
- **Rolling windows**: Training on past 30 days, prediction on current day
- **Selection logic**: Rank symbols by predicted payoff, take top‑N, enforce entry threshold
- **Region blocks**: Code organized with `# region imports` / `# endregion` markers
- **Symbol universe**: Top‑200 Binance futures (same list as `ccxt_data_fetch`)

## ANTI‑PATTERNS
- **No separate model serialization**: Model re‑trained daily in‑memory; no pickled models saved
- **Hard‑coded feature list**: 10 features defined inline; not configurable
- **Backtest code duplication**: Each backtest copies source files into dated folders
- **LSP errors**: Lean‑specific methods (`SetStartDate`, `AddCryptoFuture`, etc.) trigger type‑check warnings (ignore)

## ML PIPELINE (research_script.py)
1. **Load data**: Reads CSV/ZIP from `data/cryptofuture/binance/minute/`
2. **Resample**: 5‑minute bars → 1‑hour bars
3. **Feature engineering**: Compute 10 features per symbol
4. **Labeling**: `payoff` = future 12‑hour return
5. **Train/test split**: Chronological split (80/20)
6. **Model training**: `RandomForestRegressor` with `n_estimators=100`
7. **Evaluation**: R², MSE, feature importance

## LIVE STRATEGY (main.py)
- **Schedule**: Trains daily (Sunday 00:00), predicts daily (00:05)
- **Model**: `HuberRegressor` (robust to outliers)
- **Portfolio**: Equal weight across top‑N symbols, rebalanced daily
- **Brokerage**: Binance Futures with margin account

## USAGE
```bash
# Run research script (feature engineering + model training)
python research_script.py

# Backtest via Lean CLI (from workspace root)
lean backtest PayoffAsymmetry

# Generated backtests appear under:
#   PayoffAsymmetry/backtests/YYYY‑MM‑DD_HH‑MM‑SS/
```

## NOTES
- **Feature names**: `HighPressure`, `VWAP_ZScore`, `VolumeProfile`, `Momentum`, `Volatility`, `Skew`, `Kurtosis`, `VolumeRatio`, `Spread`, `Liquidity`
- **Label window**: 12‑hour forward return; adjust in `research_script.py` if needed
- **Top‑N & threshold**: `top_n = 10`, `entry_threshold = 0.001` (symbols with predicted payoff > 0.1%)
- **Data dependency**: Requires minute‑level data from `ccxt_data_fetch`