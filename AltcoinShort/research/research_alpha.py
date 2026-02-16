#!/usr/bin/env python3
"""
research_alpha.py  –  AltcoinShort 赔率/波动率/行为 Alpha 综合研究
===================================================================

独立脚本，不依赖 QuantConnect 引擎。
直接读取本地 minute 数据，计算 7 大 alpha 因子，
模拟做空信号并生成 PnL / 因子分析。

因子列表:
  A 类 (波动率/赔率):
    1. Downside Asymmetry
    2. Return Skewness
    3. Volatility Expansion
    4. Adverse Spike Filter
  B 类 (行为金融):
    5. TS-CO   (持续过度反应)
    6. PGR/PLR (处置效应 VWAP 代理)
    7. CSAD    (横截面羊群效应)

用法:
  /Users/chenzhao/Documents/lean_workspace/venv/bin/python research_alpha.py

回测区间: 2025‑11‑01 – 2026‑02‑01
"""

from __future__ import annotations

import os
import zipfile
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ====================================================================
# 配置
# ====================================================================
DATA_DIR = Path(
    "/Users/chenzhao/Documents/lean_workspace/data/cryptofuture/binance/minute"
)
START_DATE = datetime(2025, 11, 1)
END_DATE = datetime(2026, 2, 1)

# A 类因子参数
VOL_LOOKBACK_HOURS = 168
VOL_REFRESH_HOURS = 6
SKEW_THRESHOLD = -0.3
VOL_EXPANSION_CAP = 3.0
ADVERSE_SPIKE_MULT = 2.0
MAX_ADVERSE_RATIO = 2.0
WEIGHT_POWER = 1.5

# B 类因子参数
CO_LOOKBACK = 24  # CO 回望小时
CO_EXTREME_PCT = 0.98  # CO 极端分位 (Tuned)
CO_DECAY = 0.95  # CO 指数衰减
VWAP_LOOKBACK = 168  # PGR/PLR VWAP 回望
DISPOSITION_HIGH = 0.03  # 高浮盈阈值 3%
DISPOSITION_LOW = -0.03  # 高浮亏阈值 -3%
CSAD_LOOKBACK = 48  # CSAD 回归窗口 (Tuned)

# 交易参数
MAX_POSITIONS = 5
HOLD_HOURS = 24
INITIAL_CAPITAL = 100_000
LEVERAGE = 2
TRAILING_STOP_PCT = 0.05
HARD_STOP_PCT = 0.10

# 排除清单
EXCLUSION = {
    "ONDO",
    "PENDLE",
    "OM",
    "CFG",
    "POLYX",
    "MPL",
    "CPOOL",
    "MAPLE",
    "RIO",
    "PROPS",
    "RSR",
    "MKR",
    "COMP",
    "FET",
    "RENDER",
    "TAO",
    "WLD",
    "ARKM",
    "OCEAN",
    "AGIX",
    "ASI",
    "AKT",
    "RNDR",
    "GLM",
    "NMR",
    "CTXC",
    "PHB",
    "MDT",
    "VIRTUAL",
    "AI16Z",
    "GRIFFAIN",
    "PRIME",
    "AIOZ",
    "NFP",
    "ALI",
    "RSS3",
    "MASA",
    "IO",
    "PAAL",
    "SLEEPLESS",
    "0X0",
    "USDC",
    "USDT",
    "DAI",
    "TUSD",
    "USDP",
    "GUSD",
    "FRAX",
    "LUSD",
    "SUSD",
    "FDUSD",
    "PYUSD",
    "EURC",
    "PAXG",
    "XAUT",
    "BTC",
    "ETH",
}


# ====================================================================
# 数据加载
# ====================================================================
def _load_minute_zip(zip_path: Path) -> Optional[pd.DataFrame]:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            name = zf.namelist()[0]
            df = pd.read_csv(
                zf.open(name),
                header=None,
                names=["ms", "open", "high", "low", "close", "volume"],
            )
        return df
    except Exception:
        return None


def load_hourly_data(
    ticker_dir: Path, start: datetime, end: datetime
) -> Optional[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    d = start
    while d < end:
        fname = f"{d.strftime('%Y%m%d')}_trade.zip"
        zp = ticker_dir / fname
        if zp.exists():
            df = _load_minute_zip(zp)
            if df is not None and len(df) == 1440:
                base_ts = pd.Timestamp(d)
                df["datetime"] = base_ts + pd.to_timedelta(df["ms"], unit="ms")
                frames.append(df)
        d += timedelta(days=1)

    if not frames:
        return None

    full = pd.concat(frames, ignore_index=True)
    full.set_index("datetime", inplace=True)
    hourly = (
        full.resample("1h")
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
    return hourly


# ====================================================================
# Alpha 因子计算  (per-symbol)
# ====================================================================
def compute_factors(
    hourly: pd.DataFrame,
    co_history: deque,
) -> List[Dict]:
    """
    滑动窗口计算全部 7 因子。
    返回 [{datetime, 因子值..., weight, close}, ...]
    """
    closes = hourly["close"].values
    volumes = hourly["volume"].values
    returns = np.diff(closes) / closes[:-1]

    results = []

    for i in range(VOL_LOOKBACK_HOURS, len(returns), VOL_REFRESH_HOURS):
        window = returns[i - VOL_LOOKBACK_HOURS : i]
        if len(window) < 48:
            continue

        down_ret = window[window < 0]
        up_ret = window[window > 0]
        if len(down_ret) < 10 or len(up_ret) < 10:
            continue

        down_dev = np.std(down_ret)
        up_dev = np.std(up_ret)
        down_vol = down_dev * np.sqrt(24) * 100
        up_vol = up_dev * np.sqrt(24) * 100

        if (down_dev + up_dev) == 0:
            continue

        # ── A1: Asymmetry ──
        asymmetry = (down_dev - up_dev) / (down_dev + up_dev)

        # ── A2: Skewness ──
        skewness = float(sp_stats.skew(window))

        # ── A3: Vol Expansion ──
        recent = window[-24:]
        vol_recent = np.std(recent) * np.sqrt(24) * 100
        vol_full = np.std(window) * np.sqrt(24) * 100
        vol_expansion = min(
            vol_recent / vol_full if vol_full > 0 else 1.0, VOL_EXPANSION_CAP
        )

        # ── A4: Adverse Spike ──
        recent_up = recent[recent > 0]
        adverse_spike = False
        if len(recent_up) > 0:
            if np.max(recent_up) > ADVERSE_SPIKE_MULT * np.mean(np.abs(up_ret)):
                adverse_spike = True

        # Adverse vol ratio filter
        if up_vol > 0 and down_vol > 0 and (up_vol / down_vol) > MAX_ADVERSE_RATIO:
            continue
        if adverse_spike:
            continue

        # ── B1: TS-CO (持续过度反应) ──
        co_window = returns[max(0, i - CO_LOOKBACK) : i]
        decay_weights = np.array(
            [CO_DECAY**k for k in range(len(co_window) - 1, -1, -1)]
        )
        co_score = float(np.sum(np.sign(co_window) * decay_weights))
        co_history.append(co_score)

        co_extreme = False
        if len(co_history) > 100:
            pct_val = np.percentile(list(co_history), CO_EXTREME_PCT * 100)
            if abs(co_score) > abs(pct_val):
                co_extreme = True

        # ── B2: PGR/PLR Proxy (VWAP 乖离率) ──
        vwap_start = max(0, i + 1 - VWAP_LOOKBACK)
        vwap_end = i + 1
        vwap_closes = closes[vwap_start:vwap_end]
        vwap_vols = volumes[vwap_start:vwap_end]
        total_vv = np.sum(vwap_vols)
        if total_vv > 0:
            vwap = np.sum(vwap_closes * vwap_vols) / total_vv
        else:
            vwap = np.mean(vwap_closes)
        current_close = closes[i] if i < len(closes) else closes[-1]
        disposition = (current_close - vwap) / vwap if vwap > 0 else 0.0

        # ── 综合权重 ──
        base_score = max(asymmetry, 0.0)

        skew_bonus = 1.0
        if skewness < SKEW_THRESHOLD:
            skew_bonus = 1.0 + min(abs(skewness) * 0.5, 1.0)

        vol_mult = max(vol_expansion, 1.0)

        # CO modifier (Tuned)
        co_mult = 1.0
        if co_extreme:
            if co_score < 0:
                co_mult = 0.5  # 连续暴跌到极端 → 适度降权
            else:
                co_mult = 1.5  # 连续暴涨到极端 → 崩盘信号 → 做空加分
        else:
            if co_score < -2:
                co_mult = 1.2  # 持续下跌中 → 趋势延续 → 小幅加分

        # Disposition modifier (Tuned)
        disp_mult = 1.0
        if disposition > DISPOSITION_HIGH:
            disp_mult = 1.0 + min(disposition * 5, 1.0)  # 高浮盈 → 卖压
        elif disposition < DISPOSITION_LOW:
            disp_mult = max(1.0 + disposition * 2, 0.3)  # 高浮亏 → 支撑

        weight = (
            (base_score**WEIGHT_POWER) * skew_bonus * vol_mult * co_mult * disp_mult
        )

        dt_idx = i + 1
        if dt_idx < len(hourly):
            results.append(
                {
                    "datetime": hourly.index[dt_idx],
                    "asymmetry": round(asymmetry, 4),
                    "skewness": round(skewness, 3),
                    "vol_expansion": round(vol_expansion, 3),
                    "co_score": round(co_score, 3),
                    "co_extreme": co_extreme,
                    "disposition": round(disposition, 4),
                    "weight": round(weight, 4),
                    "close": closes[dt_idx] if dt_idx < len(closes) else closes[-1],
                    "down_vol": round(down_vol, 2),
                    "up_vol": round(up_vol, 2),
                }
            )

    return results


# ====================================================================
# CSAD 横截面计算
# ====================================================================
def compute_csad_series(
    all_hourly: Dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    计算横截面 CSAD 及回归系数 γ2 的滚动值。
    返回 DataFrame: [datetime, csad, r_market, gamma2, herding]
    """
    # 构建统一时间索引的 return panel
    return_dfs = {}
    for ticker, h in all_hourly.items():
        c = h["close"]
        r = c.pct_change().dropna()
        r = r[(r.index >= pd.Timestamp(start)) & (r.index < pd.Timestamp(end))]
        if len(r) > 100:
            return_dfs[ticker] = r

    if len(return_dfs) < 5:
        return pd.DataFrame()

    panel = pd.DataFrame(return_dfs)
    panel.dropna(axis=0, how="all", inplace=True)
    # 至少 5 个 ticker 有数据的行
    panel = panel.dropna(thresh=5)

    if len(panel) < CSAD_LOOKBACK * 2:
        return pd.DataFrame()

    r_market = panel.mean(axis=1)
    csad = panel.sub(r_market, axis=0).abs().mean(axis=1)

    # 滚动回归
    results = []
    for end_idx in range(CSAD_LOOKBACK, len(panel)):
        start_idx = end_idx - CSAD_LOOKBACK
        y = csad.iloc[start_idx:end_idx].values
        rm = r_market.iloc[start_idx:end_idx].values

        X = np.column_stack([np.ones(CSAD_LOOKBACK), np.abs(rm), rm**2])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            gamma2 = float(beta[2])
            resid = y - X @ beta
            s2 = np.sum(resid**2) / (CSAD_LOOKBACK - 3)
            var_b = s2 * np.linalg.inv(X.T @ X)
            t_stat = gamma2 / np.sqrt(var_b[2, 2]) if var_b[2, 2] > 0 else 0.0
            herding = gamma2 < 0 and t_stat < -2.0
        except Exception:
            gamma2 = 0.0
            herding = False

        results.append(
            {
                "datetime": panel.index[end_idx],
                "csad": float(csad.iloc[end_idx]),
                "r_market": float(r_market.iloc[end_idx]),
                "gamma2": gamma2,
                "herding": herding,
            }
        )

    return pd.DataFrame(results)


# ====================================================================
# 回测引擎 (Optimized Exit)
# ====================================================================
class Position:
    __slots__ = (
        "ticker",
        "entry_price",
        "entry_time",
        "initial_size",
        "current_size",
        "best_price",
        "pnl",
        "closed",
        "close_reason",
        "partial_taken",
        "max_pnl_pct",
    )

    def __init__(self, ticker, entry_price, entry_time, size):
        self.ticker = ticker
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.initial_size = size
        self.current_size = size
        self.best_price = entry_price
        self.pnl = 0.0
        self.closed = False
        self.close_reason = ""
        self.partial_taken = False
        self.max_pnl_pct = 0.0

    def update(self, current_price, current_time) -> bool:
        self.best_price = min(self.best_price, current_price)

        # Current Unrealized PnL % (Short)
        current_pnl_pct = (self.entry_price - current_price) / self.entry_price
        self.max_pnl_pct = max(self.max_pnl_pct, current_pnl_pct)

        elapsed_hours = (current_time - self.entry_time).total_seconds() / 3600

        # 1. Flash Crash TP: Close 50% if >8% profit in <4h
        if not self.partial_taken and elapsed_hours <= 4 and current_pnl_pct > 0.08:
            realized = self.current_size * 0.5 * current_pnl_pct
            self.pnl += realized
            self.current_size *= 0.5
            self.partial_taken = True
            # Note: We don't return True here, trade continues with half size

        # 2. Hard Stop (10%)
        # Check against entry price
        loss_pct = (current_price - self.entry_price) / self.entry_price
        if loss_pct > HARD_STOP_PCT:
            # Close remaining
            self.pnl += -loss_pct * self.current_size
            self.closed = True
            self.close_reason = "hard_stop"
            return True

        # 3. Dynamic Trailing Stop
        # Default 5%. If profit > 10%, tighten to 2%.
        trailing_limit = 0.02 if self.max_pnl_pct > 0.10 else 0.05

        # Calculate bounce from best price
        bounce = (current_price - self.best_price) / self.best_price
        if bounce > trailing_limit:
            # Close remaining
            final_pnl_pct = (self.entry_price - current_price) / self.entry_price
            self.pnl += final_pnl_pct * self.current_size
            self.closed = True
            if self.partial_taken:
                self.close_reason = "trailing_partial_tp"
            else:
                self.close_reason = "trailing_stop"
            return True

        # 4. Timeout (24h)
        if elapsed_hours >= HOLD_HOURS:
            final_pnl_pct = (self.entry_price - current_price) / self.entry_price
            self.pnl += final_pnl_pct * self.current_size
            self.closed = True
            if self.partial_taken:
                self.close_reason = "timeout_partial_tp"
            else:
                self.close_reason = "timeout"
            return True

        return False


def run_backtest():
    print("=" * 70)
    print("  AltcoinShort – 赔率/波动率/行为 Alpha 综合研究")
    print(f"  exit optimization: Flash Crash TP & Dynamic Trailing")
    print(f"  Period: {START_DATE.date()} → {END_DATE.date()}")
    print("=" * 70)

    # ── 发现 tickers ──
    ticker_dirs = sorted(
        [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.endswith("usdt")]
    )
    base_tickers = {d.name.replace("usdt", "").upper(): d for d in ticker_dirs}
    eligible = {k: v for k, v in base_tickers.items() if k not in EXCLUSION}
    print(f"\n  发现 {len(base_tickers)} 个 tickers，排除后剩余 {len(eligible)} 个")

    warmup_start = START_DATE - timedelta(hours=VOL_LOOKBACK_HOURS + 24)

    # ── 加载所有 hourly 数据 ──
    all_hourly: Dict[str, pd.DataFrame] = {}
    print(f"\n  加载数据中 (warmup from {warmup_start.date()})...")
    loaded = 0
    for base, ddir in eligible.items():
        hourly = load_hourly_data(ddir, warmup_start, END_DATE + timedelta(days=2))
        if hourly is not None and len(hourly) >= VOL_LOOKBACK_HOURS + 48:
            all_hourly[base] = hourly
            loaded += 1
            if loaded % 20 == 0:
                print(f"    已加载 {loaded} 个 tickers...")
    print(f"  共加载 {loaded} 个 tickers")

    # ── 计算 CSAD ──
    print("  计算 CSAD 横截面羊群效应...")
    csad_df = compute_csad_series(all_hourly, START_DATE, END_DATE)
    herding_times = set()
    if not csad_df.empty:
        herding_rows = csad_df[csad_df["herding"]]
        herding_times = set(herding_rows["datetime"])
        herding_pct = len(herding_rows) / len(csad_df) * 100
        avg_gamma2 = csad_df["gamma2"].mean()
        print(
            f"  CSAD 结果: γ2 均值={avg_gamma2:.4f} | "
            f"羊群效应时段: {len(herding_rows)}/{len(csad_df)} ({herding_pct:.1f}%)"
        )
    else:
        print("  CSAD: 数据不足，跳过")

    # ── 计算 per-symbol 因子 ──
    print("  计算 per-symbol 因子...")
    co_history: deque = deque(maxlen=10000)
    all_signals: List[Dict] = []
    computed = 0

    for base, hourly in all_hourly.items():
        factors = compute_factors(hourly, co_history)
        for f in factors:
            if f["datetime"] >= START_DATE:
                f["ticker"] = base

                # 应用 CSAD herding 加成
                if f["datetime"] in herding_times:
                    f["weight"] = round(f["weight"] * 1.3, 4)
                    f["csad_herding"] = True
                else:
                    f["csad_herding"] = False

                all_signals.append(f)
        computed += 1
        if computed % 20 == 0:
            print(f"    已计算 {computed} 个 tickers...")

    print(f"  共生成 {len(all_signals)} 条原始信号")

    if not all_signals:
        print("  ❌ 没有信号")
        return

    sig_df = pd.DataFrame(all_signals)
    sig_df.sort_values(["datetime", "weight"], ascending=[True, False], inplace=True)

    signal_times = sig_df["datetime"].unique()

    # ── 价格查询表 ──
    print("  构建价格查询表...")
    price_cache: Dict[str, pd.Series] = {}
    for base, hourly in all_hourly.items():
        price_cache[base] = hourly["close"]

    # ── 模拟交易 ──
    print("  模拟交易中...")
    trades: List[Dict] = []
    active_positions: List[Position] = []
    equity_curve = []
    capital = INITIAL_CAPITAL
    position_per_slot = (INITIAL_CAPITAL * LEVERAGE) / MAX_POSITIONS

    for t in sorted(signal_times):
        for pos in active_positions:
            if pos.closed:
                continue
            if pos.ticker in price_cache:
                prices = price_cache[pos.ticker]
                mask = prices.index == t
                if mask.any():
                    pos.update(prices.loc[mask].iloc[0], t)

        newly_closed = [p for p in active_positions if p.closed]
        for p in newly_closed:
            capital += p.pnl
            trades.append(
                {
                    "ticker": p.ticker,
                    "entry_time": p.entry_time,
                    "exit_time": t,
                    "entry_price": p.entry_price,
                    "pnl": p.pnl,
                    "pnl_pct": p.pnl / p.initial_size * 100,
                    "reason": p.close_reason,
                }
            )
        active_positions = [p for p in active_positions if not p.closed]

        open_tickers = {p.ticker for p in active_positions}
        available = MAX_POSITIONS - len(active_positions)

        if available > 0:
            window = sig_df[sig_df["datetime"] == t]
            top = window[
                (~window["ticker"].isin(open_tickers)) & (window["weight"] > 0.01)
            ].head(available)

            for _, row in top.iterrows():
                active_positions.append(
                    Position(row["ticker"], row["close"], t, position_per_slot)
                )

        unrealized = 0.0
        for p in active_positions:
            if p.ticker in price_cache:
                prices = price_cache[p.ticker]
                mask = prices.index == t
                if mask.any():
                    cp = prices.loc[mask].iloc[0]
                    # Calculate unrealized PnL: realized + current floating of remaining
                    current_floating = (
                        -(cp - p.entry_price) / p.entry_price * p.current_size
                    )
                    unrealized += p.pnl + current_floating

        equity_curve.append(
            {
                "datetime": t,
                "equity": capital + unrealized,
                "positions": len(active_positions),
            }
        )

    # 强制平仓
    for p in active_positions:
        if not p.closed and p.ticker in price_cache:
            prices = price_cache[p.ticker]
            if len(prices) > 0:
                lp = prices.iloc[-1]
                p.pnl += -(lp - p.entry_price) / p.entry_price * p.current_size
                capital += p.pnl
                trades.append(
                    {
                        "ticker": p.ticker,
                        "entry_time": p.entry_time,
                        "exit_time": prices.index[-1],
                        "entry_price": p.entry_price,
                        "pnl": p.pnl,
                        "pnl_pct": p.pnl / p.initial_size * 100,
                        "reason": "force_close",
                    }
                )

    # ================================================================
    # 输出
    # ================================================================
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    print("\n" + "=" * 70)
    print("  📊 回测结果 (Optimized Exit: Flash TP + Dynamic Trailing)")
    print("=" * 70)

    if trades_df.empty:
        print("  没有交易")
        return

    total_pnl = trades_df["pnl"].sum()
    win = trades_df[trades_df["pnl"] > 0]
    lose = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(win) / len(trades_df) * 100

    avg_win = win["pnl_pct"].mean() if len(win) > 0 else 0
    avg_loss = lose["pnl_pct"].mean() if len(lose) > 0 else 0
    pf = (
        abs(win["pnl"].sum() / lose["pnl"].sum())
        if len(lose) > 0 and lose["pnl"].sum() != 0
        else float("inf")
    )

    print(f"  总交易数:     {len(trades_df)}")
    print(f"  胜率:         {win_rate:.1f}%")
    print(f"  盈利交易:     {len(win)}  (平均 {avg_win:+.2f}%)")
    print(f"  亏损交易:     {len(lose)}  (平均 {avg_loss:+.2f}%)")
    print(f"  盈亏比 (PF):  {pf:.2f}")
    print(f"  总 PnL:       ${total_pnl:,.2f}")
    print(f"  总收益率:     {total_pnl / INITIAL_CAPITAL * 100:+.2f}%")

    if not equity_df.empty:
        peak = equity_df["equity"].expanding().max()
        dd = (equity_df["equity"] - peak) / peak
        max_dd = dd.min() * 100
        print(f"  最大回撤:     {max_dd:.2f}%")

    # ── 平仓原因 ──
    print(f"\n  平仓原因分布:")
    for reason, count in trades_df["reason"].value_counts().items():
        sub = trades_df[trades_df["reason"] == reason]
        print(f"    {reason:20s}: {count:4d} 笔  (avg {sub['pnl_pct'].mean():+.2f}%)")

    # ── Top/Bottom tickers ──
    if len(trades_df) > 0:
        tp = trades_df.groupby("ticker")["pnl"].sum().sort_values()
        print(f"\n  📉 亏损最多 Top 5 tickers:")
        for t, pnl in tp.head(5).items():
            print(f"    {t:10s}: ${pnl:+,.2f}")
        print(f"\n  📈 盈利最多 Top 5 tickers:")
        for t, pnl in tp.tail(5).items():
            print(f"    {t:10s}: ${pnl:+,.2f}")

    # ── 保存 ──
    out_dir = Path(__file__).parent / "backtests"
    out_dir.mkdir(exist_ok=True)

    tp = out_dir / "alpha_exit_trades.csv"
    ep = out_dir / "alpha_exit_equity.csv"
    sp = out_dir / "alpha_exit_signals.csv"

    trades_df.to_csv(tp, index=False)
    equity_df.to_csv(ep, index=False)
    sig_df.to_csv(sp, index=False)

    print(f"\n  结果已保存:")
    print(f"    交易记录:  {tp}")
    print(f"    权益曲线:  {ep}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_backtest()
