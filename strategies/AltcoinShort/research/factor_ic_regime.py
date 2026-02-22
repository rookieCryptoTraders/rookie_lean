"""
Factor IC by Market Regime (Bull / Bear / Sideways)
=====================================================
Uses BTC 30-day rolling return to classify regimes,
then computes IC for each factor within each regime.

Usage:
    python3 factor_ic_regime.py
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
# CONFIG (same as factor_ic_analysis.py)
# ============================================================================

LEAN_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "cryptofuture", "binance", "minute"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ic_research")
START_DATE = "2025-01-01"
END_DATE = "2026-02-01"

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

LOOKBACK = 168
CO_LOOKBACK = 48
CO_DECAY = 0.95
VWAP_LOOKBACK = 72
REBALANCE_FREQ = 6
MIN_HISTORY = 168

# ── Market Regime Dates (based on actual 2025 market review) ──
# Bull:       2025-01-01 ~ 2025-08-31  (ETF inflows, BTC dominance rising)
# AltSeason:  2025-09-01 ~ 2025-10-06  (altcoin rotation, BTC ATH ~$126k)
# Transition: 2025-10-07 ~ 2025-10-24  (flash crash Oct 10, top confirmed)
# Bear:       2025-10-25 ~ 2026-02-01  (trend reversal, deleveraging)
REGIME_DATES = [
    ("bull", "2025-01-01", "2025-08-31"),
    ("altseason", "2025-09-01", "2025-10-06"),
    ("transition", "2025-10-07", "2025-10-24"),
    ("bear", "2025-10-25", "2026-02-01"),
]

FORWARD_H = 6


# ============================================================================
# DATA LOADING (reuse from factor_ic_analysis.py)
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
        .reset_index()
    )


def load_all_data():
    print(f"\n{'=' * 60}")
    print(f" Loading data: {len(TICKERS)} tickers")
    print(f"{'=' * 60}")
    minute_data, hourly_data = {}, {}
    for i, ticker in enumerate(TICKERS):
        df_min = load_ticker_minutes(ticker)
        if df_min is not None and len(df_min) > MIN_HISTORY * 60:
            minute_data[ticker] = df_min
            hourly_data[ticker] = resample_to_hourly(df_min)
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(TICKERS)}] loaded...")
    print(f"  Total: {len(minute_data)}/{len(TICKERS)} tickers")
    return minute_data, hourly_data


# ============================================================================
# FACTOR CALCULATION (same as before)
# ============================================================================


def calculate_factors(closes, volumes):
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
        "asymmetry": asymmetry,
        "skewness": skewness,
        "vol_expansion": vol_expansion,
        "co_score": co_score,
        "disposition": disposition,
    }


# ============================================================================
# REGIME CLASSIFICATION
# ============================================================================


def classify_regimes(hc_df):
    """
    Classify each hourly timestamp by actual market regime dates.
    """
    regime = pd.Series("unknown", index=hc_df.index)
    for name, start, end in REGIME_DATES:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
        mask = (regime.index >= start_ts) & (regime.index <= end_ts)
        regime[mask] = name
    return regime


# ============================================================================
# IC BY REGIME
# ============================================================================


def compute_ic_by_regime(hc_df, hv_df, mc_df, regime_series, factor_name, forward_h):
    """Compute IC separately for each regime."""
    fwd_min = forward_h * 60
    m_fwd = mc_df.shift(-fwd_min) / mc_df - 1

    valid_times = hc_df.index[MIN_HISTORY : -max(forward_h, 1) : REBALANCE_FREQ]

    regime_names = [r[0] for r in REGIME_DATES]
    records = {r: [] for r in regime_names}

    for t in valid_times:
        t_idx = hc_df.index.get_loc(t)
        if t_idx < MIN_HISTORY:
            continue

        regime = regime_series.loc[t]

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

        if regime in records:
            records[regime].append(ic)

    return records


# ============================================================================
# MAIN
# ============================================================================


def main():
    minute_data, hourly_data = load_all_data()
    if len(hourly_data) < 15:
        print("ERROR: Not enough tickers.")
        sys.exit(1)

    # Build matrices
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

    # Classify regimes
    regime = classify_regimes(hc_df)
    regime_names = [r[0] for r in REGIME_DATES]
    regime_counts = regime.value_counts()
    print(f"\n{'=' * 60}")
    print(" Market Regime Distribution")
    print(f"{'=' * 60}")
    for name, start, end in REGIME_DATES:
        n = regime_counts.get(name, 0)
        pct = n / len(regime) * 100
        print(f"  {name:>12s}: {n:>5d} hours ({pct:.1f}%)  [{start} ~ {end}]")

    # BTC price at regime transitions
    btc = hc_df["btcusdt"].dropna()
    print(f"\n  BTC range: ${btc.min():,.0f} - ${btc.max():,.0f}")
    print(f"  BTC start: ${btc.iloc[0]:,.0f} → end: ${btc.iloc[-1]:,.0f}")

    # Show BTC price at each regime boundary
    print("\n  BTC at regime boundaries:")
    for name, start, end in REGIME_DATES:
        ts_s = pd.Timestamp(start, tz="UTC")
        ts_e = pd.Timestamp(end, tz="UTC")
        btc_s = btc.asof(ts_s)
        btc_e = btc.asof(ts_e)
        ret = (btc_e / btc_s - 1) * 100 if btc_s > 0 else 0
        print(f"    {name:>12s}: ${btc_s:>9,.0f} → ${btc_e:>9,.0f}  ({ret:+.1f}%)")

    # Run IC by regime for each factor at multiple horizons
    factor_names = [
        "downward_vol",
        "asymmetry",
        "skewness",
        "vol_expansion",
        "co_score",
        "disposition",
    ]
    horizons = [1, 6, 12, 24, 48]

    print(f"\n{'=' * 60}")
    print(f" IC by Regime: {len(factor_names)} factors × {len(horizons)} horizons")
    print(f"{'=' * 60}")

    all_results = []

    for factor in factor_names:
        print(f"\n{'─' * 60}")
        print(f" {factor}")
        print(f"{'─' * 60}")

        header = f"  {'Horizon':>8s}  "
        for rn in regime_names:
            header += f"  {rn.upper():>12s}"
        header += f"  {'ALL':>12s}"
        print(header)
        subhdr = f"  {'':>8s}  "
        for _ in regime_names:
            subhdr += f"  {'IC  (IR)':>12s}"
        subhdr += f"  {'IC  (IR)':>12s}"
        print(subhdr)

        for h in horizons:
            ic_regimes = compute_ic_by_regime(hc_df, hv_df, mc_df, regime, factor, h)

            row = {"factor": factor, "horizon": h}
            line = f"  h={h:2d}h     "

            for r in regime_names:
                arr = np.array(ic_regimes.get(r, []))
                if len(arr) > 5:
                    ic_mean = np.mean(arr)
                    ic_ir = ic_mean / np.std(arr) if np.std(arr) > 0 else 0
                    n = len(arr)
                    row[f"IC_{r}"] = round(ic_mean, 4)
                    row[f"IR_{r}"] = round(ic_ir, 3)
                    row[f"N_{r}"] = n
                    line += f"  {ic_mean:+.3f}({ic_ir:+.2f})"
                else:
                    row[f"IC_{r}"] = np.nan
                    row[f"IR_{r}"] = np.nan
                    row[f"N_{r}"] = len(arr)
                    line += f"  {'N/A':>12s}"

            # All combined
            all_ics = []
            for r in regime_names:
                all_ics.extend(ic_regimes.get(r, []))
            all_arr = np.array(all_ics)
            if len(all_arr) > 5:
                ic_all = np.mean(all_arr)
                ir_all = ic_all / np.std(all_arr) if np.std(all_arr) > 0 else 0
                line += f"  {ic_all:+.3f}({ir_all:+.2f})"
                row["IC_all"] = round(ic_all, 4)
                row["IR_all"] = round(ir_all, 3)

            print(line, flush=True)
            all_results.append(row)

    results_df = pd.DataFrame(all_results)

    # Summary: best regime for each factor
    print(f"\n{'=' * 60}")
    print(" KEY FINDINGS: Factor × Regime Interaction (h=6h)")
    print(f"{'=' * 60}")

    for factor in factor_names:
        fdf = results_df[
            (results_df["factor"] == factor) & (results_df["horizon"] == 6)
        ]
        if fdf.empty:
            continue
        r = fdf.iloc[0]

        ics = {}
        line = f"  {factor:<16s} "
        for rn in regime_names:
            ic_val = r.get(f"IC_{rn}", 0) or 0
            ics[rn] = ic_val
            line += f" {rn[:4]}={ic_val:+.4f}"

        # Check direction flips
        signs = [np.sign(v) for v in ics.values() if v != 0]
        flips = len(set(signs)) > 1

        best = max(ics.items(), key=lambda x: abs(x[1]))
        line += f"  Best={best[0]}  {'⚠️ FLIPS' if flips else '✅ STABLE'}"
        print(line)

    # Save
    out = os.path.join(OUTPUT_DIR, "ic_regime_results.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"\n  Results → {out}")


if __name__ == "__main__":
    main()
