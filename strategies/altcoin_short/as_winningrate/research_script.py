"""
Cross-Section Payoff Asymmetry Research Script (Consolidated & Optimized)
Features: 10 predictive factors for downside payoff asymmetry.
Model: RandomForestRegressor.
Evaluation: Short-only strategy on Top 15 ranked assets (OOT testing).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import logging
import warnings
from settings import DATA_DIR, START_DATE, END_DATE, TICKERS
from utils import load_trade_data

# --- Aesthetics ---
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-muted")
# sns.set_palette("viridis")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def construct_features():
    pass

def run_research():
    # Load BTC data first for beta features
    logger.info("Loading BTC reference data...")
    btc_df = load_trade_data("btcusdt", START_DATE, END_DATE)

    # Pre-calculate BTC features on an hourly grid to avoid redundant slices
    logger.info("Pre-calculating BTC Beta features...")
    btc_df["log_ret"] = np.log(btc_df["close"] / btc_df["close"].shift(1))
    btc_hourly = btc_df.resample("1H").last().dropna()
    btc_features = {}

    for ts in btc_hourly.index:
        try:
            bw1h = btc_df.loc[ts - timedelta(hours=1) : ts]
            bw12h = btc_df.loc[ts - timedelta(hours=12) : ts]
            if bw1h.empty or bw12h.empty:
                continue

            b_ret_1h = (bw1h["close"].iloc[-1] - bw1h["close"].iloc[0]) / (
                bw1h["close"].iloc[0] + 1e-6
            )
            b_mom_12h = (bw12h["close"].iloc[-1] - bw12h["close"].iloc[0]) / (
                bw12h["close"].iloc[0] + 1e-6
            )
            br12 = np.log(bw12h["close"] / bw12h["close"].shift(1))
            bs_down, bs_up = br12[br12 < 0].std(), br12[br12 > 0].std()
            b_vol_ratio = (bs_down or 0) / ((bs_up or 0) + 1e-6)

            btc_features[ts] = {
                "BTC_Ret_1h": b_ret_1h,
                "BTC_Mom_12h": b_mom_12h,
                "BTC_Vol_Ratio": b_vol_ratio,
            }
        except:
            continue

    all_samples = []
    for ticker in TICKERS:
        logger.info(f"Processing {ticker}...")
        df = load_trade_data(ticker, START_DATE, END_DATE)
        ticker_samples = construct_features(df, ticker, btc_features=btc_features)
        all_samples.extend(ticker_samples)
        logger.info(f"  {ticker}: {len(ticker_samples)} samples")

    data = pd.DataFrame(all_samples).dropna().sort_values("time")
    logger.info(f"Total dataset size: {len(data)}")

    # Split: 2025 Training, 2026 Jan OOT
    train = data[data["time"] < datetime(2026, 1, 1)]
    test = data[data["time"] >= datetime(2026, 1, 1)]

    if train.empty or test.empty:
        logger.error("Insufficient split data. Check data dates.")
        return

    X_cols = [
        "HighPressure",
        "Skewness",
        "VolRatio",
        "VWAP_ZScore",
        "VolSurge",
        "Momentum",
        "RSI",
        "ATR_Ratio",
        "MA_Deviation",
        "VolPriceCorr",
        "BTC_Ret_1h",
        "BTC_Mom_12h",
        "BTC_Vol_Ratio",
    ]

    logger.info("Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
    )
    model.fit(train[X_cols], train["y"])

    # Feature Importance
    importances = pd.Series(model.feature_importances_, index=X_cols).sort_values(
        ascending=False
    )
    logger.info(f"\nFeature Importances:\n{importances}")

    # Evaluate Short-Only (Top 15 ranking per hour)
    test["y_pred"] = model.predict(test[X_cols])
    results = []
    for ts, group in test.groupby("time", observed=True):
        if len(group) < 20:
            continue
        top_15 = group.sort_values("y_pred").tail(15)
        results.append({"time": ts, "avg_y": top_15["y"].mean()})

    results_df = pd.DataFrame(results)
    avg_y = results_df["avg_y"].mean()
    win_rate = (results_df["avg_y"] > 0).mean()
    stability = avg_y / (results_df["avg_y"].std() + 1e-6)

    logger.info("=" * 50)
    logger.info("OOT PERFORMANCE (JAN 2026)")
    logger.info(f"Average Realized Payoff (Y): {avg_y:.4f}")
    logger.info(f"Short Win Rate (Y > 0):      {win_rate:.2%}")
    logger.info(f"Payoff Stability (Sharpe):   {stability:.4f}")
    logger.info("=" * 50)

    # Save
    data.to_csv("PayoffAsymmetry/research_dataset.csv", index=False)
    logger.info("Dataset saved. Research Complete.")
