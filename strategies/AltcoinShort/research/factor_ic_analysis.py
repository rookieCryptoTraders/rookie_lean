"""
Single-Factor IC (Information Coefficient) Analysis
====================================================
Reads existing QuantConnect-format minute data from disk,
resamples to 1h for factor calculation,
and uses minute-level forward returns for IC precision.

Usage:
    python3 factor_ic_analysis.py
"""

import os
import sys
import zipfile
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

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

# Top-50 market cap (excluding stablecoins), lowercase matching LEAN dirs
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

# Factor params (matching alpha.py — in HOURLY bars)
LOOKBACK = 168  # 7d of hourly
CO_LOOKBACK = 48
CO_DECAY = 0.95
VWAP_LOOKBACK = 72

# IC analysis
FORWARD_HORIZONS = [1, 6, 12, 24, 48]  # hours
REBALANCE_FREQ = 6  # eval every 6h
MIN_HISTORY = 168  # min hourly bars needed


# ============================================================================
# READ LEAN FORMAT DATA
# ============================================================================


def read_lean_minute_zip(zip_path: str, date_str: str) -> Optional[pd.DataFrame]:
    """
    Read a LEAN minute trade zip file.
    Format: ms_since_midnight,open,high,low,close,volume (no header)
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(
                    f,
                    header=None,
                    names=["ms", "open", "high", "low", "close", "volume"],
                )
        if df.empty:
            return None

        # Convert ms_since_midnight to datetime
        base = pd.Timestamp(date_str, tz="UTC")
        df["time"] = base + pd.to_timedelta(df["ms"], unit="ms")
        df = df.drop(columns=["ms"])
        return df
    except Exception:
        return None


def load_ticker_minutes(ticker: str) -> Optional[pd.DataFrame]:
    """Load all minute data for a ticker from LEAN zip files."""
    ticker_dir = os.path.join(LEAN_DATA_DIR, ticker)
    if not os.path.isdir(ticker_dir):
        return None

    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")

    all_dfs = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        zip_path = os.path.join(ticker_dir, f"{date_str}_trade.zip")
        if os.path.exists(zip_path):
            df = read_lean_minute_zip(zip_path, current.strftime("%Y-%m-%d"))
            if df is not None:
                all_dfs.append(df)
        current += timedelta(days=1)

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("time").drop_duplicates(subset=["time"])
    return combined.reset_index(drop=True)


def resample_to_hourly(df_min: pd.DataFrame) -> pd.DataFrame:
    """Resample minute OHLCV → hourly OHLCV."""
    df = df_min.set_index("time")
    hourly = (
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
    return hourly.reset_index()


def load_all_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Load minute data from LEAN zips, resample to hourly.
    Returns (minute_data, hourly_data).
    """
    print(f"\n{'=' * 60}")
    print(f" Loading LEAN minute data: {len(TICKERS)} tickers")
    print(f" Path: {LEAN_DATA_DIR}")
    print(f" Range: {START_DATE} to {END_DATE}")
    print(f"{'=' * 60}")

    minute_data = {}
    hourly_data = {}

    for i, ticker in enumerate(TICKERS):
        df_min = load_ticker_minutes(ticker)
        if df_min is None or len(df_min) < MIN_HISTORY * 60:
            print(
                f"  [{i + 1:2d}/{len(TICKERS)}] {ticker:>15s}: SKIP (insufficient data)"
            )
            continue

        df_hourly = resample_to_hourly(df_min)
        if len(df_hourly) < MIN_HISTORY:
            print(
                f"  [{i + 1:2d}/{len(TICKERS)}] {ticker:>15s}: SKIP ({len(df_hourly)} hourly bars)"
            )
            continue

        minute_data[ticker] = df_min
        hourly_data[ticker] = df_hourly
        print(
            f"  [{i + 1:2d}/{len(TICKERS)}] {ticker:>15s}: "
            f"{len(df_min):>8,} min → {len(df_hourly):>6,} hourly"
        )

    print(f"\n  Total: {len(minute_data)}/{len(TICKERS)} tickers loaded")
    return minute_data, hourly_data


# ============================================================================
# FACTOR CALCULATION (mirrors alpha.py — hourly bars)
# ============================================================================


def calculate_factors(
    closes: np.ndarray, volumes: np.ndarray
) -> Optional[Dict[str, float]]:
    """All 7 factors for one symbol's lookback window (hourly)."""
    if len(closes) < 48:
        return None

    returns = np.diff(closes) / closes[:-1]
    if len(returns) < 48:
        return None

    down_ret = returns[returns < 0]
    up_ret = returns[returns > 0]
    if len(down_ret) < 10 or len(up_ret) < 10:
        return None

    down_dev = np.std(down_ret)
    up_dev = np.std(up_ret)
    down_vol = down_dev * np.sqrt(24) * 100
    up_vol = up_dev * np.sqrt(24) * 100
    total_vol = down_vol + up_vol
    downward_bias = down_vol / total_vol if total_vol > 0 else 0.5

    asymmetry = (
        (down_dev - up_dev) / (down_dev + up_dev) if (down_dev + up_dev) > 0 else 0.0
    )

    skewness = float(sp_stats.skew(returns))

    vol_recent = np.std(returns[-24:]) * np.sqrt(24) * 100
    vol_full = np.std(returns) * np.sqrt(24) * 100
    vol_expansion = vol_recent / vol_full if vol_full > 0 else 1.0

    co_w = returns[-CO_LOOKBACK:]
    weights = np.array([CO_DECAY**k for k in range(len(co_w) - 1, -1, -1)])
    co_score = float(np.sum(np.sign(co_w) * weights))

    if volumes is not None and len(volumes) >= VWAP_LOOKBACK:
        vc = closes[-VWAP_LOOKBACK:]
        vv = volumes[-VWAP_LOOKBACK:]
        tvv = np.sum(vv)
        vwap = np.sum(vc * vv) / tvv if tvv > 0 else closes[-1]
        disposition = (closes[-1] - vwap) / vwap if vwap > 0 else 0.0
    else:
        p_ref = np.mean(closes[-VWAP_LOOKBACK:])
        disposition = (closes[-1] - p_ref) / p_ref if p_ref > 0 else 0.0

    return {
        "downward_vol": down_vol,
        "upward_vol": up_vol,
        "downward_bias": downward_bias,
        "asymmetry": asymmetry,
        "skewness": skewness,
        "vol_expansion": vol_expansion,
        "co_score": co_score,
        "disposition": disposition,
    }


# ============================================================================
# IC ANALYSIS ENGINE
# ============================================================================


def build_matrices(hourly_data, minute_data):
    """Build aligned DataFrames."""
    hc, hv = {}, {}
    for t, df in hourly_data.items():
        di = df.set_index("time")
        hc[t] = di["close"]
        hv[t] = di["volume"]
    hc_df = pd.DataFrame(hc)
    hv_df = pd.DataFrame(hv)

    mc = {}
    for t, df in minute_data.items():
        di = df.set_index("time")
        mc[t] = di["close"]
    mc_df = pd.DataFrame(mc)

    return hc_df, hv_df, mc_df


def compute_cross_sectional_ic(
    hc_df,
    hv_df,
    mc_df,
    factor_name: str,
    forward_h: int = 6,
) -> pd.DataFrame:
    """Factors on hourly, forward returns on minute → Spearman IC."""
    fwd_min = forward_h * 60
    m_fwd = mc_df.shift(-fwd_min) / mc_df - 1

    valid_times = hc_df.index[MIN_HISTORY : -max(forward_h, 1) : REBALANCE_FREQ]
    ic_records = []

    for t in valid_times:
        t_idx = hc_df.index.get_loc(t)
        if t_idx < MIN_HISTORY:
            continue

        factor_vals = {}
        for ticker in hc_df.columns:
            if pd.isna(hc_df.iloc[t_idx][ticker]):
                continue
            si = max(0, t_idx - LOOKBACK)
            wc = hc_df.iloc[si : t_idx + 1][ticker].dropna().values
            wv = hv_df.iloc[si : t_idx + 1][ticker].dropna().values
            if len(wc) < 48:
                continue
            f = calculate_factors(wc, wv)
            if f is None:
                continue
            factor_vals[ticker] = f[factor_name]

        fwd = {}
        for ticker in factor_vals:
            if t in m_fwd.index:
                fr = m_fwd.loc[t, ticker]
            else:
                nearest = m_fwd.index.asof(t)
                fr = m_fwd.loc[nearest, ticker] if nearest is not pd.NaT else np.nan
            if not pd.isna(fr):
                fwd[ticker] = fr

        common = set(factor_vals) & set(fwd)
        if len(common) < 10:
            continue

        f_arr = np.array([factor_vals[c] for c in common])
        r_arr = np.array([fwd[c] for c in common])
        ic, _ = sp_stats.spearmanr(f_arr, r_arr)

        ic_records.append({"datetime": t, "ic": ic, "n_symbols": len(common)})

    return pd.DataFrame(ic_records)


def quintile_analysis(hc_df, hv_df, mc_df, factor_name, forward_h=6, n_groups=5):
    """Quintile spread analysis."""
    fwd_min = forward_h * 60
    m_fwd = mc_df.shift(-fwd_min) / mc_df - 1

    valid_times = hc_df.index[MIN_HISTORY : -max(forward_h, 1) : REBALANCE_FREQ]
    group_returns = {i: [] for i in range(n_groups)}

    for t in valid_times:
        t_idx = hc_df.index.get_loc(t)
        if t_idx < MIN_HISTORY:
            continue

        scores = {}
        for ticker in hc_df.columns:
            if pd.isna(hc_df.iloc[t_idx][ticker]):
                continue
            si = max(0, t_idx - LOOKBACK)
            wc = hc_df.iloc[si : t_idx + 1][ticker].dropna().values
            wv = hv_df.iloc[si : t_idx + 1][ticker].dropna().values
            if len(wc) < 48:
                continue
            f = calculate_factors(wc, wv)
            if f is None:
                continue
            if t in m_fwd.index:
                fr = m_fwd.loc[t, ticker]
            else:
                nearest = m_fwd.index.asof(t)
                fr = m_fwd.loc[nearest, ticker] if nearest is not pd.NaT else np.nan
            if pd.isna(fr):
                continue
            scores[ticker] = (f[factor_name], fr)

        if len(scores) < n_groups * 2:
            continue

        sorted_items = sorted(scores.items(), key=lambda x: x[1][0])
        gs = len(sorted_items) // n_groups
        for g in range(n_groups):
            s = g * gs
            e = s + gs if g < n_groups - 1 else len(sorted_items)
            group_returns[g].append(np.mean([it[1][1] for it in sorted_items[s:e]]))

    result = {}
    for g in range(n_groups):
        arr = np.array(group_returns[g])
        result[f"Q{g + 1}_mean"] = np.mean(arr) * 100 if len(arr) else 0
        result[f"Q{g + 1}_std"] = np.std(arr) * 100 if len(arr) else 0
    result["Q1_Q5_spread"] = result["Q1_mean"] - result["Q5_mean"]
    result["n_periods"] = len(group_returns[0])
    return result


# ============================================================================
# MAIN
# ============================================================================


def main():
    minute_data, hourly_data = load_all_data()
    if len(hourly_data) < 15:
        print("ERROR: Not enough tickers loaded.")
        sys.exit(1)

    print("\nBuilding matrices...", flush=True)
    hc_df, hv_df, mc_df = build_matrices(hourly_data, minute_data)
    print(f"  Hourly matrix: {hc_df.shape}, Minute matrix: {mc_df.shape}")

    factor_names = [
        "downward_vol",
        "downward_bias",
        "asymmetry",
        "skewness",
        "vol_expansion",
        "co_score",
        "disposition",
    ]

    print(f"\n{'=' * 60}")
    print(
        f" IC Analysis: {len(factor_names)} factors × {len(FORWARD_HORIZONS)} horizons"
    )
    print(
        f" Rebalance: {REBALANCE_FREQ}h | Lookback: {LOOKBACK}h | Symbols: {len(hourly_data)}"
    )
    print(f"{'=' * 60}")

    results = []
    for factor in factor_names:
        print(f"\n── {factor} ──", flush=True)
        row = {"factor": factor}
        for h in FORWARD_HORIZONS:
            ic_df = compute_cross_sectional_ic(hc_df, hv_df, mc_df, factor, h)
            if len(ic_df) < 10:
                print(f"  h={h:2d}h: insufficient data")
                for k in ["IC_mean", "IC_IR", "IC_dir"]:
                    row[f"{k}_{h}h"] = np.nan
                continue
            ic_mean = ic_df["ic"].mean()
            ic_std = ic_df["ic"].std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
            direction = (ic_df["ic"] < 0).mean() * 100
            row[f"IC_mean_{h}h"] = round(ic_mean, 4)
            row[f"IC_IR_{h}h"] = round(ic_ir, 3)
            row[f"IC_dir_{h}h"] = round(direction, 1)
            row[f"IC_n_{h}h"] = len(ic_df)
            print(
                f"  h={h:2d}h: IC={ic_mean:+.4f} | IR={ic_ir:+.3f} | "
                f"Dir={direction:.0f}% neg | N={len(ic_df)}",
                flush=True,
            )
        results.append(row)

    results_df = pd.DataFrame(results)

    # Summary
    print(f"\n{'=' * 60}")
    print(" SUMMARY (h=6h)")
    print(f"{'=' * 60}")
    cols = [
        c
        for c in ["factor", "IC_mean_6h", "IC_IR_6h", "IC_dir_6h"]
        if c in results_df.columns
    ]
    print(results_df[cols].to_string(index=False))

    # Quintile
    print(f"\n{'=' * 60}")
    print(" QUINTILE ANALYSIS (h=6h)")
    print(f"{'=' * 60}")
    for factor in factor_names:
        q = quintile_analysis(hc_df, hv_df, mc_df, factor, forward_h=6)
        print(f"\n  {factor}:")
        for g in range(5):
            print(f"    Q{g + 1}: {q.get(f'Q{g + 1}_mean', 0):+.4f}%")
        print(f"    Q1-Q5 spread: {q.get('Q1_Q5_spread', 0):+.4f}%")

    # IC Decay
    print(f"\n{'=' * 60}")
    print(" IC DECAY")
    print(f"{'=' * 60}")
    print(f"{'Factor':<18} ", end="")
    for h in FORWARD_HORIZONS:
        print(f"{'h=' + str(h) + 'h':>8}", end="")
    print()
    print("-" * 60)
    for _, row in results_df.iterrows():
        print(f"{row['factor']:<18} ", end="")
        for h in FORWARD_HORIZONS:
            v = row.get(f"IC_mean_{h}h", np.nan)
            print(f"{v:>+8.4f}" if not pd.isna(v) else f"{'N/A':>8}", end="")
        print()

    # Grading
    print(f"\n{'=' * 60}")
    print(" FACTOR GRADES")
    print(f"{'=' * 60}")
    for _, row in results_df.iterrows():
        ic_m = abs(row.get("IC_mean_6h", 0) or 0)
        ic_ir = abs(row.get("IC_IR_6h", 0) or 0)
        if ic_m > 0.05 and ic_ir > 0.5:
            g = "★★★ STRONG"
        elif ic_m > 0.03 and ic_ir > 0.3:
            g = "★★  MODERATE"
        elif ic_m > 0.015:
            g = "★   WEAK"
        else:
            g = "✗   NOISE"
        print(f"  {row['factor']:<18} → {g}")

    out = os.path.join(OUTPUT_DIR, "ic_analysis_results.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"\n  Results → {out}")


if __name__ == "__main__":
    main()
