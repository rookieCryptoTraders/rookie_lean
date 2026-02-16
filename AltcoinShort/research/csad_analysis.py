"""
CSAD Herding Analysis
=====================
Computes Cross-Sectional Absolute Deviation (CSAD) and Gamma2 coefficient
to detect herding behavior.
CSAD_t = alpha + gamma1 * |Rm_t| + gamma2 * Rm_t^2

Hypothesis: Gamma2 < 0 (significant) indicates herding, often preceding trend reversals.
"""

import os
import sys
import zipfile
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

LEAN_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "cryptofuture", "binance", "minute"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ic_research")
START_DATE = "2025-01-01"
END_DATE = "2026-02-01"

# Use top liquid tickers for market representation
TICKERS = [
    "btcusdt",
    "ethusdt",
    "bnbusdt",
    "solusdt",
    "xrpusdt",
    "dogeusdt",
    "adausdt",
    "avaxusdt",
    "trxusdt",
    "dotusdt",
    "linkusdt",
    "ltcusdt",
    "uniusdt",
    "nearusdt",
    "suiusdt",
    "aptusdt",
    "arbusdt",
    "opusdt",
    "injusdt",
    "seiusdt",
    "tiausdt",
    "atomusdt",
    "etcusdt",
    "filusdt",
    "xlmusdt",
    "hbarusdt",
    "vetusdt",
    "icpusdt",
    "stxusdt",
    "aaveusdt",
    "ldousdt",
    "crvusdt",
    "bchusdt",
    "algousdt",
    "xtzusdt",
    "neousdt",
    "grtusdt",
    "sandusdt",
    "manausdt",
    "axsusdt",
    "galausdt",
    "fetusdt",
    "jasmyusdt",
    "chzusdt",
    "qntusdt",
    "cakeusdt",
    "iotausdt",
    "dashusdt",
    "zecusdt",
    "wifusdt",
]

CSAD_LOOKBACK = 48  # Lookback window for regression (hours)
ROLLING_STEP = 6  # Step size for rolling calculation (hours)

# Regime Dates (for comparison)
REGIME_DATES = [
    ("bull", "2025-01-01", "2025-08-31"),
    ("altseason", "2025-09-01", "2025-10-06"),
    ("transition", "2025-10-07", "2025-10-24"),
    ("bear", "2025-10-25", "2026-02-01"),
]


# ============================================================================
# DATA LOADING
# ============================================================================


def read_lean_minute_zip(zip_path, date_str):
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(
                    f,
                    header=None,
                    names=["ms", "open", "high", "low", "close", "volume"],
                )
        if df.empty:
            return None
        base = pd.Timestamp(date_str, tz="UTC")
        df["time"] = base + pd.to_timedelta(df["ms"], unit="ms")
        return df.drop(columns=["ms"])
    except Exception:
        return None


def load_ticker_minutes(ticker):
    ticker_dir = os.path.join(LEAN_DATA_DIR, ticker)
    if not os.path.isdir(ticker_dir):
        return None
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    all_dfs = []
    current = start
    while current <= end:
        zp = os.path.join(ticker_dir, f"{current.strftime('%Y%m%d')}_trade.zip")
        if os.path.exists(zp):
            df = read_lean_minute_zip(zp, current.strftime("%Y-%m-%d"))
            if df is not None:
                all_dfs.append(df)
        current += timedelta(days=1)
    if not all_dfs:
        return None
    combined = pd.concat(all_dfs, ignore_index=True)
    return (
        combined.sort_values("time")
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )


def resample_to_hourly(df_min):
    df = df_min.set_index("time")
    return (
        df.resample("1h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )


def load_hourly_returns():
    print(f"\n{'=' * 60}")
    print(f" Loading hourly data for CSAD: {len(TICKERS)} tickers")
    print(f"{'=' * 60}")

    returns_dict = {}

    for i, ticker in enumerate(TICKERS):
        df_min = load_ticker_minutes(ticker)
        if df_min is not None and len(df_min) > 1000:
            df_h = resample_to_hourly(df_min)
            # Calculate simple returns
            ret = df_h["close"].pct_change()
            returns_dict[ticker] = ret
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(TICKERS)}] loaded...")

    # Combine into a wide DataFrame (index=time, columns=tickers)
    wide_df = pd.DataFrame(returns_dict).dropna(how="all")
    print(f"  Shape: {wide_df.shape}")
    return wide_df


# ============================================================================
# CSAD CALCULATION
# ============================================================================


def calculate_csad_gamma2(prices_window):
    """
    Compute Gamma2 for a given window of returns (TxN).
    prices_window: DataFrame of returns
    """
    # Filter out columns with too many NaNs in this window
    df = prices_window.dropna(
        axis=1, how="any"
    )  # Drop cols with ANY nan in this window
    if df.shape[1] < 10 or df.shape[0] < 20:
        return np.nan, np.nan, np.nan

    # 1. Market Return (equal weighted)
    rm = df.mean(axis=1)  # Series of length T

    # 2. CSAD
    # |Ri - Rm|
    abs_diff = np.abs(df.sub(rm, axis=0))
    csad = abs_diff.mean(axis=1)  # Series of length T

    # 3. Regression: CSAD = alpha + g1*|Rm| + g2*Rm^2
    Y = csad.values
    X1 = np.abs(rm.values)
    X2 = rm.values**2

    # Check for constant input to avoid singular matrix
    if np.var(X1) < 1e-12 or np.var(X2) < 1e-12:
        return np.nan, np.nan, np.nan

    X = np.column_stack([np.ones(len(Y)), X1, X2])

    try:
        beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        gamma2 = beta[2]

        # Calculate t-stat for gamma2
        # MSE = sum(residuals) / (n - p)
        if len(residuals) > 0:
            mse = residuals[0] / (len(Y) - 3)
            cov = mse * np.linalg.inv(X.T @ X)
            se_gamma2 = np.sqrt(cov[2, 2])
            t_stat = gamma2 / se_gamma2 if se_gamma2 > 0 else 0
        else:
            t_stat = 0

        return gamma2, t_stat, rm.iloc[-1]  # Return last market return too
    except Exception:
        return np.nan, np.nan, np.nan


# ============================================================================
# MAIN
# ============================================================================


def main():
    returns_df = load_hourly_returns()

    # BTC returns for forward comparison
    if "btcusdt" not in returns_df.columns:
        print("Error: btcusdt not found")
        return
    btc_fwd = returns_df["btcusdt"].shift(-24).rolling(24).sum()  # Next 24h return

    results = []

    # Rolling calculation
    times = returns_df.index
    # Start from lookback
    valid_starts = range(CSAD_LOOKBACK, len(times) - 1, ROLLING_STEP)

    print(f"\nComputing Gamma2 (window={CSAD_LOOKBACK}h, step={ROLLING_STEP}h)...")

    for i in valid_starts:
        end_idx = i
        start_idx = i - CSAD_LOOKBACK

        window = returns_df.iloc[start_idx:end_idx]
        t_current = times[end_idx]

        g2, t_stat, last_rm = calculate_csad_gamma2(window)

        if np.isnan(g2):
            continue

        # Get forward market return (next 24h)
        # Using simple mean of top coins as market proxy
        fwd_ret_24h = np.mean(
            [
                returns_df[c].iloc[end_idx + 1 : end_idx + 25].sum()
                for c in returns_df.columns
                if pd.notna(returns_df[c].iloc[end_idx])
            ]
        )

        results.append(
            {
                "time": t_current,
                "gamma2": g2,
                "t_stat": t_stat,
                "fwd_ret_24h": fwd_ret_24h,
                "last_rm": last_rm,
            }
        )

    res_df = pd.DataFrame(results)
    res_df.set_index("time", inplace=True)

    # ── Regime Analysis ──
    regime_series = pd.Series("unknown", index=res_df.index)
    for name, start, end in REGIME_DATES:
        mask = (res_df.index >= pd.Timestamp(start, tz="UTC")) & (
            res_df.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
        )
        regime_series[mask] = name
    res_df["regime"] = regime_series

    print(f"\n{'=' * 60}")
    print(" CSAD Gamma2 Analysis Results")
    print(f"{'=' * 60}")

    # 1. Distribution of Gamma2
    pct_neg = (res_df["gamma2"] < 0).mean() * 100
    pct_sig_neg = ((res_df["gamma2"] < 0) & (res_df["t_stat"] < -1.96)).mean() * 100
    print(f"  Total samples: {len(res_df)}")
    print(f"  Negative Gamma2: {pct_neg:.1f}%")
    print(f"  Signif. Negative (Herding): {pct_sig_neg:.1f}% (t < -1.96)")

    # 2. Predictive Power by Regime
    # Does significant negative gamma2 predict reversals?
    print(f"\n  Predictive Power: Herding (g2 < 0, t < -2) -> Next 24h Return")

    for r in ["bull", "altseason", "transition", "bear"]:
        subset = res_df[res_df["regime"] == r]
        if subset.empty:
            continue

        # Filter for herding
        herding = subset[(subset["gamma2"] < 0) & (subset["t_stat"] < -2.0)]
        normal = subset[subset["t_stat"] >= -2.0]

        avg_ret_herding = herding["fwd_ret_24h"].mean() * 100
        avg_ret_normal = normal["fwd_ret_24h"].mean() * 100
        n_herding = len(herding)

        print(
            f"    {r.upper():<12s}: N={n_herding:3d} | Herd Fwd={avg_ret_herding:+.2f}% vs Normal={avg_ret_normal:+.2f}%"
        )

    # 3. Correlation
    print(f"\n  Correlation (Gamma2 vs Fwd Return):")
    ic = res_df["gamma2"].corr(res_df["fwd_ret_24h"])
    print(f"    Global IC: {ic:.4f}")

    # Save
    out = os.path.join(OUTPUT_DIR, "csad_results.csv")
    res_df.to_csv(out)
    print(f"\n  Results -> {out}")


if __name__ == "__main__":
    main()
