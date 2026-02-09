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
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import logging
import warnings

# --- Aesthetics ---
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-muted")
sns.set_palette("viridis")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIR = "/Users/chenzhao/Documents/lean_workspace/data/cryptofuture/binance/minute"
START_DATE = datetime(2025, 1, 1)  # Training: 2025 Full Year
END_DATE = datetime(2026, 1, 31)  # OOT Test: 2026 Jan

TICKERS = [
    "btcusdt",
    "ethusdt",
    "bnbusdt",
    "solusdt",
    "xrpusdt",
    "dogeusdt",
    "adausdt",
    "avaxusdt",
    "dotusdt",
    "linkusdt",
    "maticusdt",
    "ltcusdt",
    "uniusdt",
    "atomusdt",
    "etcusdt",
    "filusdt",
    "aptusdt",
    "nearusdt",
    "arbusdt",
    "opusdt",
    "injusdt",
    "suiusdt",
    "tiausdt",
    "seiusdt",
    "stxusdt",
    "imxusdt",
    "runeusdt",
    "aaveusdt",
    "mkrusdt",
    "ldousdt",
]


def load_ticker_data(ticker, start_date, end_date):
    """Efficiently load and concatenate minute data for a ticker."""
    ticker_dir = os.path.join(DATA_DIR, ticker)
    if not os.path.exists(ticker_dir):
        return pd.DataFrame()

    all_dfs = []
    files = sorted([f for f in os.listdir(ticker_dir) if f.endswith("_trade.zip")])
    for f in files:
        date_str = f.split("_")[0]
        try:
            file_date = datetime.strptime(date_str, "%Y%m%d")
            if start_date <= file_date <= end_date:
                df = pd.read_csv(
                    os.path.join(ticker_dir, f), header=None, compression="zip"
                )
                df.columns = ["ms", "open", "high", "low", "close", "volume"]
                df["time"] = file_date + pd.to_timedelta(df["ms"], unit="ms")
                df.set_index("time", inplace=True)
                all_dfs.append(df[["open", "high", "low", "close", "volume"]])
        except:
            continue

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs).sort_index().drop_duplicates()


def construct_features(df, ticker, btc_features=None):
    """Construct 10 optimized features + BTC beta features and 12h payoff label."""
    if df.empty or len(df) < 1440:
        return []

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    hourly = df.resample("1H").last().dropna()

    samples = []
    for ts in hourly.index:
        try:
            w1h = df.loc[ts - timedelta(hours=1) : ts]
            w12h = df.loc[ts - timedelta(hours=12) : ts]
            w24h = df.loc[ts - timedelta(hours=24) : ts]

            if len(w1h) < 50 or len(w12h) < 600:
                continue

            price = w1h["close"].iloc[-1]
            sigma_1h = w1h["log_ret"].std() + 1e-6

            # 1. High Pressure (normalized distance from 24h high)
            high_pressure = (w24h["high"].max() - price) / (price * sigma_1h)
            # 2. Skewness (1h)
            skewness = w1h["log_ret"].skew()
            # 3. Vol Ratio (12h Down vs Up Vol)
            r12 = w12h["log_ret"]
            s_down, s_up = r12[r12 < 0].std(), r12[r12 > 0].std()
            vol_ratio = (s_down or 0) / ((s_up or 0) + 1e-6)
            # 4. VWAP Z-Score (12h)
            vwap_12h = (w12h["close"] * w12h["volume"]).sum() / (
                w12h["volume"].sum() + 1e-6
            )
            vwap_zscore = (price - vwap_12h) / (price * r12.std() + 1e-6)
            # 5. Volume Surge (1h avg vs 24h avg)
            vol_surge = w1h["volume"].mean() / (w24h["volume"].mean() + 1e-6)
            # 6. Momentum Overextension (12h Return / Sigma)
            momentum = (price - w12h["close"].iloc[0]) / (
                w12h["close"].iloc[0] * r12.std() + 1e-6
            )
            # 7. RSI-like (12h)
            delta = w12h["close"].diff()
            gain, loss = (
                delta.where(delta > 0, 0).mean(),
                -delta.where(delta < 0, 0).mean(),
            )
            rsi = 100 - (100 / (1 + (gain / (loss + 1e-6))))
            # 8. ATR-Ratio (1h)
            tr = np.maximum(
                w1h["high"] - w1h["low"],
                np.maximum(
                    abs(w1h["high"] - w1h["close"].shift(1)),
                    abs(w1h["low"] - w1h["close"].shift(1)),
                ),
            )
            atr_ratio = tr.mean() / (price + 1e-6)
            # 9. MA-Deviation (Dist from 1h MA)
            ma_deviation = (price - w1h["close"].mean()) / (w1h["close"].mean() + 1e-6)
            # 10. Volume-Price correlation (1h)
            vol_price_corr = w1h["close"].pct_change().corr(w1h["volume"].pct_change())

            # --- BTC Market Beta Features (Pre-calculated lookup) ---
            btc_ret_1h = 0
            btc_mom_12h = 0
            btc_vol_ratio = 1
            if btc_features is not None and ts in btc_features:
                b = btc_features[ts]
                btc_ret_1h = b["BTC_Ret_1h"]
                btc_mom_12h = b["BTC_Mom_12h"]
                btc_vol_ratio = b["BTC_Vol_Ratio"]

            # Label (Y): 12h Payoff Asymmetry
            f_start, f_end = ts + timedelta(minutes=1), ts + timedelta(hours=12)
            if f_end > df.index[-1]:
                continue
            future = df.loc[f_start:f_end]
            if future.empty:
                continue

            max_down = max(price - future["low"].min(), price * 0.001)
            max_up = max(future["high"].max() - price, price * 0.001)
            y = np.log(max_down / max_up)

            samples.append(
                {
                    "ticker": ticker,
                    "time": ts,
                    "y": y,
                    "HighPressure": high_pressure,
                    "Skewness": skewness,
                    "VolRatio": vol_ratio,
                    "VWAP_ZScore": vwap_zscore,
                    "VolSurge": vol_surge,
                    "Momentum": momentum,
                    "RSI": rsi,
                    "ATR_Ratio": atr_ratio,
                    "MA_Deviation": ma_deviation,
                    "VolPriceCorr": vol_price_corr,
                    "BTC_Ret_1h": btc_ret_1h,
                    "BTC_Mom_12h": btc_mom_12h,
                    "BTC_Vol_Ratio": btc_vol_ratio,
                }
            )
        except:
            continue
    return samples


def run_research():
    # Load BTC data first for beta features
    logger.info("Loading BTC reference data...")
    btc_df = load_ticker_data("btcusdt", START_DATE, END_DATE)

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
        df = load_ticker_data(ticker, START_DATE, END_DATE)
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


if __name__ == "__main__":
    run_research()
