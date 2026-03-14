# region imports
from AlgorithmImports import *
import os
import csv
import zipfile
from collections import deque
from datetime import datetime, timedelta

# TODO: use modern python type hint. e.g. list[int], dict[str, int], tuple[int, str], set[int], tuple[int, str] | set[int] etc.
from typing import List, Dict, Set, Optional, Tuple, Callable, Any

import numpy as np
from scipy import stats as sp_stats

from config import BASE_DATA_PATH, ASSET_CLASS, EXCHANGE
from utils import CryptoFutureDepthData
# endregion


# ================================================================
# Custom data feature calculators (reusable, decoupled)
# ================================================================
# Add new features by implementing a calculator function and registering
# it in AltcoinShortAlphaModel._custom_feature_calculators.
# See ADDING_FEATURES.md for step-by-step instructions.


def compute_depth_features(
    depth_bar: Optional[CryptoFutureDepthData],
) -> Dict[str, float]:
    """
    Compute order-book / depth features from a single snapshot (L5 or aggregated).
    Used by the custom data feature pipeline in AltcoinShortAlphaModel.
    Returns dict: depth_obi_l5, depth_spread, depth_micro_price_divergence.
    Doc §2.2: OBI L5, spread; §4.2: WMP/micro-price divergence (proxy from depth-weighted level).
    """
    out: Dict[str, float] = {
        "depth_obi_l5": 0.0,
        "depth_spread": 0.0,
        "depth_micro_price_divergence": 0.0,
    }
    if depth_bar is None:
        return out
    try:
        percentages = list(getattr(depth_bar, "percentages", []))
        depths = list(getattr(depth_bar, "depths", []))
        if not percentages or not depths or len(percentages) != len(depths):
            return out
        bid_depth = sum(d for d, p in zip(depths, percentages) if p < 0)
        ask_depth = sum(d for d, p in zip(depths, percentages) if p > 0)
        total = bid_depth + ask_depth
        if total > 0:
            out["depth_obi_l5"] = (bid_depth - ask_depth) / total
        # Spread proxy: distance between best bid and best ask levels (in % terms)
        bid_pcts = [p for p in percentages if p < 0]
        ask_pcts = [p for p in percentages if p > 0]
        if bid_pcts and ask_pcts:
            out["depth_spread"] = float(min(ask_pcts) - max(bid_pcts))
        # Micro-price divergence proxy: depth-weighted average level = implied pressure
        # Positive = ask-heavy (WMP above mid), negative = bid-heavy. Normalize to ~[-1,1].
        weighted_sum = sum(p * d for p, d in zip(percentages, depths))
        if total > 0:
            wmp_offset = weighted_sum / total
            out["depth_micro_price_divergence"] = max(-1.0, min(1.0, wmp_offset / 5.0))
    except Exception:
        pass
    return out


def compute_ofi_lcr(
    prev_depth: Optional[CryptoFutureDepthData],
    curr_depth: Optional[CryptoFutureDepthData],
) -> Dict[str, float]:
    """
    Order Flow Imbalance (OFI) and Liquidity Consumption Rate (LCR) proxy.

    §4.1 / §4.3 of the design doc define OFI/LCR on event-level order flow.
    Here we approximate them from consecutive L5 depth snapshots:

    - OFI: net signed depth change (bid additions minus ask additions) normalised by total depth.
    - LCR: total depth removed within 1 minute (sum of negative depth deltas) normalised by total depth.
    """
    out: Dict[str, float] = {
        "depth_ofi_1m": 0.0,
        "depth_lcr_1m": 0.0,
    }
    if prev_depth is None or curr_depth is None:
        return out

    try:
        prev_pcts = list(getattr(prev_depth, "percentages", []))
        prev_depths = list(getattr(prev_depth, "depths", []))
        curr_pcts = list(getattr(curr_depth, "percentages", []))
        curr_depths = list(getattr(curr_depth, "depths", []))

        if not prev_pcts or not prev_depths or not curr_pcts or not curr_depths:
            return out

        prev_map = {float(p): float(d) for p, d in zip(prev_pcts, prev_depths)}
        curr_map = {float(p): float(d) for p, d in zip(curr_pcts, curr_depths)}

        ofi = 0.0
        lcr = 0.0

        for pct, curr_d in curr_map.items():
            prev_d = prev_map.get(pct, 0.0)
            delta = curr_d - prev_d
            side = -1.0 if pct < 0 else 1.0  # negative % = bid side, positive % = ask side
            ofi += side * delta
            if delta < 0:
                lcr += abs(delta)

        total_depth = sum(curr_map.values())
        if total_depth > 0:
            out["depth_ofi_1m"] = float(ofi / total_depth)
            out["depth_lcr_1m"] = float(lcr / total_depth)
    except Exception:
        pass

    return out


# ── OHLCV-based feature helpers (per 加密货币量化模型设计.md §2.1, §3, §4, §5.2, §6) ──

def compute_parkinson_volatility(high: np.ndarray, low: np.ndarray) -> float:
    """Parkinson volatility (range-based). §2.1: High/Low → volatility. Daily ann. %."""
    if high is None or low is None or len(high) < 2 or len(low) < 2:
        return 0.0
    ln_hl = np.log(np.maximum(high / np.maximum(low, 1e-12), 1e-12))
    # sigma^2 = (1/(4*ln2)) * (ln(H/L))^2 per period
    coeff = 1.0 / (4.0 * np.log(2.0))
    var = coeff * np.mean(ln_hl**2)
    return float(np.sqrt(var * 24) * 100)  # hourly -> daily ann. %


def compute_rogers_satchell_volatility(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> float:
    """Rogers-Satchell volatility (OHLC range). §2.1. Daily ann. %."""
    n = min(len(open_), len(high), len(low), len(close))
    if n < 2:
        return 0.0
    o, h, l_, c = open_[-n:], high[-n:], low[-n:], close[-n:]
    ln_hc = np.log(np.maximum(h / np.maximum(c, 1e-12), 1e-12))
    ln_ho = np.log(np.maximum(h / np.maximum(o, 1e-12), 1e-12))
    ln_lc = np.log(np.maximum(l_ / np.maximum(c, 1e-12), 1e-12))
    ln_lo = np.log(np.maximum(l_ / np.maximum(o, 1e-12), 1e-12))
    rs = ln_hc * ln_ho + ln_lc * ln_lo
    var = np.mean(rs)
    if var <= 0:
        return 0.0
    return float(np.sqrt(var * 24) * 100)


def compute_shadow_features(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    lookback: int = 24,
) -> Tuple[float, float]:
    """§2.1: Upper/lower shadow (wick) length. Returns (avg_upper_ratio, avg_lower_ratio)."""
    n = min(len(open_), len(high), len(low), len(close), lookback)
    if n < 2:
        return 0.0, 0.0
    o, h, l_, c = open_[-n:], high[-n:], low[-n:], close[-n:]
    range_ = h - l_
    range_ = np.where(range_ < 1e-12, 1e-12, range_)
    max_oc = np.maximum(o, c)
    min_oc = np.minimum(o, c)
    upper = (h - max_oc) / range_
    lower = (min_oc - l_) / range_
    return float(np.mean(upper)), float(np.mean(lower))


def compute_volume_features(
    close: np.ndarray, volume: np.ndarray, lookback: int = 24
) -> Tuple[float, float]:
    """§2.1: Volume SMA ratio; volume-price divergence (price up + volume down = bearish)."""
    if volume is None or len(volume) < lookback or len(close) < lookback:
        return 1.0, 0.0
    vol = volume[-lookback:]
    cl = close[-lookback:]
    vol_ma = np.mean(vol)
    vol_sma_ratio = float(vol[-1] / vol_ma) if vol_ma > 0 else 1.0
    ret = np.diff(cl) / np.maximum(cl[:-1], 1e-12)
    vol_chg = np.diff(vol) / np.maximum(vol[:-1], 1e-12)
    if len(ret) < 2 or len(vol_chg) < 2:
        return vol_sma_ratio, 0.0
    corr = np.corrcoef(ret, vol_chg[: len(ret)])[0, 1] if len(ret) > 1 else 0.0
    vol_price_divergence = float(corr) if not np.isnan(corr) else 0.0
    return vol_sma_ratio, vol_price_divergence


def compute_vpin_proxy(
    open_: np.ndarray, close: np.ndarray, volume: np.ndarray, lookback: int = 24
) -> float:
    """§6.2: VPIN proxy from OHLCV — buy/sell volume proxy by bar direction."""
    n = min(len(open_), len(close), len(volume), lookback)
    if n < 2:
        return 0.0
    o, c, v = open_[-n:], close[-n:], volume[-n:]
    buy_vol = np.where(c >= o, v, 0.0).sum()
    sell_vol = np.where(c < o, v, 0.0).sum()
    total = buy_vol + sell_vol
    if total <= 0:
        return 0.0
    return float(abs(buy_vol - sell_vol) / total)


def compute_lfi_proxy(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int = 48
) -> Tuple[float, float]:
    """§6.1: Liquidity Fragility Index proxy — spread proxy (H-L)/C, then skewness & kurtosis."""
    n = min(len(high), len(low), len(close), lookback)
    if n < 10:
        return 0.0, 0.0
    h, l_, c = high[-n:], low[-n:], close[-n:]
    spread_proxy = (h - l_) / np.maximum(c, 1e-12)
    sk = float(sp_stats.skew(spread_proxy))
    ku = float(sp_stats.kurtosis(spread_proxy))
    return sk, ku


def compute_benford_deviation(volume: np.ndarray, lookback: int = 168) -> float:
    """§5.2: Benford's law deviation for first digit of volume — wash-trading proxy."""
    if volume is None or len(volume) < 10:
        return 0.0
    vol = volume[-lookback:]
    first_digits = []
    for v in vol:
        if v <= 0:
            continue
        s = f"{v:.0f}".lstrip("0") or "0"
        d = int(s[0]) if s and s[0].isdigit() else 0
        if 1 <= d <= 9:
            first_digits.append(d)
    if len(first_digits) < 10:
        return 0.0
    observed = np.bincount(first_digits, minlength=10)[1:10].astype(float)
    total = observed.sum()
    if total <= 0:
        return 0.0
    observed = observed / total
    benford = np.array([np.log10(1.0 + 1.0 / d) for d in range(1, 10)])
    deviation = np.sum((observed - benford) ** 2)
    return float(np.sqrt(deviation))


class AltcoinShortAlphaModel(AlphaModel):
    """
    Altcoin Short Alpha Model — IC-Research-Driven v3.0
    ══════════════════════════════════════════════════════

    Architecture based on factor_ic_analysis_report.md findings:

    Layer 1 — SELECTION (h=24-48h factors, choose WHAT to short)
    ─────────────────────────────────────────────────────────────
      • downward_vol   IC=-0.073 (Bear: -0.106)  ★★★ CORE
      • vol_expansion  IC=-0.028 (Bull only)      ★   WEAK
      • asymmetry      IC=+0.022 (Transition: +0.126!) ★/★★★
      • skewness       IC=-0.017 (Transition: -0.121!) ✗/★★★

    Layer 2 — ENTRY FILTER (h=1-6h factors, choose WHEN to short)
    ─────────────────────────────────────────────────────────────
      • co_score       IC=+0.165 @ 1h, IR=0.77   ★★★ FILTER
        → CO > 0 means upward momentum → DON'T short
        → CO < 0 means downward momentum → GO short
      • disposition    IC=+0.167 @ 1h             ★★★ FILTER
        → Disp > 0 means above VWAP → short-term squeeze risk
        → Disp < 0 means below VWAP → safe to short

    Layer 3 — REGIME ADAPTATION (market-level)
    ─────────────────────────────────────────────────────────────
      • BTC 30d return  → Bull / Sideways / Bear / Transition
      • CSAD herding    → Rare top-finding signal (Bull: 9.0x)

    Key corrections from IC analysis:
      1. co_score is MOMENTUM, not mean-reversion
      2. disposition flips direction at h=12h
      3. asymmetry + skewness are "crisis alpha" (sleeping giants)
      4. vol_expansion fails in Bear market
      5. downward_bias ≡ asymmetry (removed duplicate)

    Additional OHLCV/depth factors (加密货币量化模型设计.md):
      §2.1: parkinson_vol, rogers_satchell_vol, upper/lower_shadow, volume_sma_ratio,
             volume_price_divergence
      §2.2/4.2: depth_spread, depth_micro_price_divergence (WMP proxy)
      §5.2: benford_deviation (wash-trading proxy)
      §6.1: lfi_skew, lfi_kurtosis (liquidity fragility proxy)
      §6.2: vpin_proxy (informed flow toxicity) — used for risk_penalty
    ══════════════════════════════════════════════════════
    """

    # ── Regime enum ──
    REGIME_BULL = "bull"
    REGIME_SIDEWAYS = "sideways"
    REGIME_BEAR = "bear"
    REGIME_TRANSITION = "transition"

    # Helpful name for framework/hybrid algorithms
    name = "AltcoinShortAlphaModel"

    def __init__(
        self,
        max_positions: int = 30,
        volatility_lookback_hours: int = 168,
        volatility_cache_hours: int = 6,
        selection_weight_power: float = 1.5,
        max_adverse_vol_ratio: float = 3.0,  # Relaxed from 2.0 (§8.3)
        insight_duration_hours: int = 48,  # Extended from 24h (§8.3: downward_vol peaks at 48h)
        # ── Volatility factor params ──
        vol_expansion_cap: float = 3.0,
        adverse_spike_mult: float = 2.0,
        # ── Behavioral factor params ──
        co_lookback: int = 24,
        co_decay: float = 0.95,
        vwap_lookback: int = 168,
        csad_lookback: int = 48,
        # ── Regime Detection ──
        btc_ticker: str = "BTCUSDT",
        regime_lookback_days: int = 30,
        regime_bull_threshold: float = 0.10,  # +10% → Bull
        regime_bear_threshold: float = -0.10,  # -10% → Bear
    ):
        self.max_positions = max_positions
        self.volatility_lookback_hours = volatility_lookback_hours
        self.volatility_cache_hours = volatility_cache_hours
        self.selection_weight_power = selection_weight_power
        self.max_adverse_vol_ratio = max_adverse_vol_ratio
        self.insight_duration = timedelta(hours=insight_duration_hours)

        self.vol_expansion_cap = vol_expansion_cap
        self.adverse_spike_mult = adverse_spike_mult

        self.co_lookback = co_lookback
        self.co_decay = co_decay
        self.vwap_lookback = vwap_lookback
        self.csad_lookback = csad_lookback

        # Regime
        self.btc_ticker = btc_ticker
        self.regime_lookback_days = regime_lookback_days
        self.regime_bull_threshold = regime_bull_threshold
        self.regime_bear_threshold = regime_bear_threshold
        self.current_regime: str = self.REGIME_SIDEWAYS
        self._btc_symbol: Optional[Symbol] = None

        # Exclusion list
        self.exclusion_list = self._build_exclusion_list()

        # State
        self.coin_data: Dict[Symbol, dict] = {}
        self._all_symbols: Set = set()
        self.last_volatility_update: Optional[datetime] = None
        self.active_insights: Set[Symbol] = set()

        # CSAD (cross-sectional)
        self.csad_gamma2: float = 0.0
        self.csad_herding: bool = False

        # CO distribution history
        self._co_history: deque = deque(maxlen=5000)

        # Depth data (LOB L5) from custom PythonData
        # Mapping: trading symbol -> depth custom symbol
        self.depth_custom_symbols: Dict[Symbol, Symbol] = {}
        # Latest depth snapshot per trading symbol (updated from Slice in update())
        self.latest_depth: Dict[Symbol, CryptoFutureDepthData] = {}
        # Simple depth-flow features (OFI/LCR) from consecutive snapshots
        self._depth_flow_features: Dict[Symbol, Dict[str, float]] = {}
        self._prev_depth_snapshot: Dict[Symbol, CryptoFutureDepthData] = {}

        # Last generated insights (for hybrid algorithms reading signals in on_data)
        self.latest_insights: List[Insight] = []

        # Custom data feature pipeline: list of (get_bar_fn, compute_fn).
        # get_bar_fn(algorithm, symbol) -> bar or None; compute_fn(bar) -> dict of feature name -> value.
        self._custom_feature_calculators: List[
            Tuple[Callable[..., Any], Callable[[Any], Dict[str, float]]]
        ] = [(self._get_depth_bar, compute_depth_features)]

    # ================================================================
    # Exclusion list
    # ================================================================
    def _build_exclusion_list(self) -> Set[str]:
        stablecoin = {
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
        }
        return stablecoin

    # ================================================================
    # Update (main loop)
    # ================================================================
    def update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        insights: List[Insight] = []

        # 1. Capture streaming depth data from Slice (Works in Backtest and Live)
        # We look for objects of type CryptoFutureDepthData
        for kvp in data.get(CryptoFutureDepthData):
            depth_symbol = kvp.key
            depth_data = kvp.value

            # Map "BTCUSDT.DEPTH" -> "BTCUSDT"
            base_ticker = depth_symbol.value.replace(".DEPTH", "")
            # Find the actual base symbol object tracked in coin_data
            for s in self.coin_data.keys():
                if s.value == base_ticker:
                    # Update latest snapshot
                    self.latest_depth[s] = depth_data

                    # Compute OFI / LCR proxy from consecutive snapshots
                    prev_snapshot = self._prev_depth_snapshot.get(s)
                    flow_feats = compute_ofi_lcr(prev_snapshot, depth_data)
                    if flow_feats:
                        self._depth_flow_features[s] = flow_feats
                    self._prev_depth_snapshot[s] = depth_data

                    break

        # 2. Only run the rest of the alpha logic on full-hour bars
        if algorithm.time.minute != 0:
            self.latest_insights = []
            return insights

        # Refresh factors every cache period (or if empty)
        if self._should_refresh(algorithm.time) or not self.coin_data:
            self._detect_regime(algorithm)
            self._refresh_all_factors(algorithm)

        # Gather candidates with valid data
        candidates: List[Symbol] = []
        for symbol in self.coin_data:
            if not data.contains_key(symbol) or data[symbol] is None:
                continue
            candidates.append(symbol)

        if not candidates:
            self.latest_insights = []
            return insights

        raw_insights: List[Tuple[float, Insight]] = []

        for symbol in candidates:
            d = self.coin_data.get(symbol, {})
            weight = d.get("selection_weight", 0.0)

            # Relaxed selection threshold
            if weight < 0.01:
                continue

            # ── Entry Filter (Layer 2) ──
            co_score = d.get("co_score", 0.0)
            disposition = d.get("disposition", 0.0)

            # Depth-based micro-structure filters
            depth_obi_l5 = d.get("depth_obi_l5", 0.0)
            micro_price_divergence = d.get("depth_micro_price_divergence", 0.0)

            # Relaxed filters
            if co_score > 2.0:
                continue

            if disposition > 0.10:
                continue

            if depth_obi_l5 > 0.8:
                continue

            if micro_price_divergence > 0.8:
                continue

            # Soft momentum penalty (gradual, not binary)
            momentum_penalty = 1.0
            if co_score > 0:
                momentum_penalty *= max(1.0 - co_score * 0.5, 0.0)
            elif co_score < -0.5:
                momentum_penalty *= 1.0 + min(abs(co_score) * 0.2, 0.4)

            if disposition > 0:
                momentum_penalty *= max(1.0 - disposition * 5, 0.2)
            elif disposition < -0.03:
                momentum_penalty *= 1.0 + min(abs(disposition) * 3, 0.3)

            # Depth: strong ask-side imbalance / micro-price below mid → boost
            depth_boost = 1.0
            if depth_obi_l5 < -0.2:
                depth_boost *= 1.0 + min(abs(depth_obi_l5) * 0.5, 0.5)
            if micro_price_divergence < -0.2:
                depth_boost *= 1.0 + min(abs(micro_price_divergence) * 0.5, 0.5)

            # §6.2 VPIN high = toxicity / flash-crash risk; §5.2 Benford high = wash-trading
            risk_penalty = 1.0
            vpin_proxy = d.get("vpin_proxy", 0.0)
            benford_deviation = d.get("benford_deviation", 0.0)
            if vpin_proxy > 0.7:
                risk_penalty *= max(0.0, 1.0 - (vpin_proxy - 0.7) * 2.0)
            if benford_deviation > 0.15:
                risk_penalty *= max(0.5, 1.0 - (benford_deviation - 0.15) * 2.0)

            effective_weight = weight * momentum_penalty * depth_boost * risk_penalty

            # algorithm.debug(
            #     f"[Alpha-Weight] {symbol.value} base_w={weight:.3f} co={co_score:.3f} disp={disposition:.3f} obi={depth_obi_l5:.3f} mp={momentum_penalty:.3f} db={depth_boost:.3f} eff_w={effective_weight:.3f}"
            # )

            # Relaxed effective weight threshold
            if effective_weight < 0.01:
                continue

            confidence = min(
                d.get("downward_vol", 5.0) / 50.0,
                0.99,
            )

            insight = Insight.price(
                symbol,
                self.insight_duration,
                InsightDirection.DOWN,
                d.get("downward_vol", 5.0) / 100,
                confidence,
                None,
                effective_weight,
            )

            tag_parts = [
                f"R:{self.current_regime[0].upper()}",
                f"dv:{d.get('downward_vol', 0):.1f}",
                f"asym:{d.get('asymmetry', 0):.3f}",
                f"skew:{d.get('skewness', 0):.2f}",
                f"vexp:{d.get('vol_expansion', 1):.2f}",
                f"co:{co_score:.2f}",
                f"disp:{disposition:.3f}",
                f"obi:{depth_obi_l5:.2f}",
                f"mp_div:{micro_price_divergence:.2f}",
                f"vp:{d.get('vpin_proxy', 0):.2f}",
                f"lfi:{d.get('lfi_kurtosis', 0):.2f}",
                f"ben:{d.get('benford_deviation', 0):.2f}",
                f"mp:{momentum_penalty:.2f}",
                f"db:{depth_boost:.2f}",
                f"w:{effective_weight:.3f}",
            ]
            if self.csad_herding:
                tag_parts.append("HERD")
            insight.Tag = "|".join(tag_parts)

            raw_insights.append((effective_weight, insight))

        raw_insights.sort(key=lambda x: x[0], reverse=True)
        insights = [ins for _, ins in raw_insights[: self.max_positions]]

        # CSAD Herding Global Boost (Layer 3)
        if self.csad_herding:
            for insight in insights:
                insight.Weight *= 1.5
                insight.Tag += "|HERD_BOOST"

        # Cache latest insights for hybrid algorithms that execute in on_data
        self.latest_insights = insights
        return insights

    # ================================================================
    # Regime Detection (Layer 3)
    # ================================================================
    def _detect_regime(self, algorithm: QCAlgorithm) -> None:
        """
        Simple regime detector based on BTC 30-day rolling return.
        Report §10.3:
          BTC 30d return >  +10% → Bull
          BTC 30d return < -10% → Bear
          In between → Sideways
          Bear preceded by recent Bull peak → Transition
        """
        if self._btc_symbol is None:
            # Find BTC in our securities
            for symbol in self._all_symbols:
                if self.btc_ticker in symbol.Value:
                    self._btc_symbol = symbol
                    break
            if self._btc_symbol is None:
                # BTC not in universe — try to get it from algorithm
                for kvp in algorithm.securities:
                    if self.btc_ticker in kvp.key.value:
                        self._btc_symbol = kvp.key
                        break

        if self._btc_symbol is None:
            self.current_regime = self.REGIME_SIDEWAYS
            return

        try:
            btc_hist = algorithm.history(
                self._btc_symbol,
                self.regime_lookback_days * 24,
                Resolution.Hour,
            )
            if btc_hist.empty or len(btc_hist) < 48:
                return

            closes = btc_hist["close"].values
            ret_30d = (closes[-1] / closes[0]) - 1.0

            # Also check recent drawdown from peak (transition detection)
            peak = np.max(closes)
            drawdown_from_peak = (closes[-1] / peak) - 1.0

            old_regime = self.current_regime

            if ret_30d > self.regime_bull_threshold:
                self.current_regime = self.REGIME_BULL
            elif ret_30d < self.regime_bear_threshold:
                # Check if this is a fresh transition (just started falling)
                if drawdown_from_peak < -0.05 and drawdown_from_peak > -0.15:
                    self.current_regime = self.REGIME_TRANSITION
                else:
                    self.current_regime = self.REGIME_BEAR
            else:
                self.current_regime = self.REGIME_SIDEWAYS

            if old_regime != self.current_regime:
                algorithm.debug(
                    f"[Alpha] Regime: {old_regime} → {self.current_regime} "
                    f"| BTC 30d: {ret_30d:+.1%} | DD from peak: {drawdown_from_peak:.1%}"
                )

        except Exception as e:
            algorithm.debug(f"[Alpha] Regime detection error: {e}")

    # ================================================================
    # Factor refresh
    # ================================================================
    def _should_refresh(self, current_time: datetime) -> bool:
        if self.last_volatility_update is None:
            return True
        elapsed = (current_time - self.last_volatility_update).total_seconds() / 3600
        return elapsed >= self.volatility_cache_hours

    def _refresh_all_factors(self, algorithm: QCAlgorithm) -> None:
        """Refresh all per-symbol factors + cross-sectional CSAD."""
        updated = 0
        removed = 0

        cross_returns: Dict[Symbol, np.ndarray] = {}

        for symbol in list(self._all_symbols):
            if not algorithm.securities.contains_key(symbol):
                continue

            result = self._calculate_all_factors(algorithm, symbol)
            if result is None:
                continue

            factor_data, hourly_returns = result

            # Adverse vol ratio filter (relaxed to 3.0 per §8.3)
            if factor_data["upward_vol"] > 0 and factor_data["downward_vol"] > 0:
                ratio = factor_data["upward_vol"] / factor_data["downward_vol"]
                if ratio > self.max_adverse_vol_ratio:
                    if symbol in self.coin_data:
                        del self.coin_data[symbol]
                    removed += 1
                    continue

            # Adverse spike filter
            if factor_data.get("adverse_spike", False):
                if symbol in self.coin_data:
                    del self.coin_data[symbol]
                removed += 1
                continue

            self.coin_data[symbol] = factor_data
            cross_returns[symbol] = hourly_returns
            updated += 1

        # CSAD cross-sectional
        self._compute_csad(cross_returns)

        self.last_volatility_update = algorithm.time

    # ================================================================
    # Per-symbol factor calculation (+ depth features)
    # ================================================================
    def _calculate_all_factors(
        self,
        algorithm: QCAlgorithm,
        symbol: Symbol,
    ) -> Optional[Tuple[dict, np.ndarray]]:
        """
        Compute all factors for a single symbol.
        Custom data features (depth, quote, etc.) are computed via _custom_feature_calculators.
        Returns (factor_dict, hourly_returns) or None.
        """
        try:
            history = algorithm.history(
                symbol, self.volatility_lookback_hours, Resolution.HOUR
            )

            if history.empty or len(history) < 12:
                return None

            closes = history["close"].values
            opens = history["open"].values if "open" in history.columns else closes
            highs = history["high"].values if "high" in history.columns else closes
            lows = history["low"].values if "low" in history.columns else closes
            volumes = history["volume"].values if "volume" in history.columns else None

            returns = np.diff(closes) / closes[:-1]
            if len(returns) < 12:
                return None

            down_returns = returns[returns < 0]
            up_returns = returns[returns > 0]
            if len(down_returns) < 2 or len(up_returns) < 2:
                return None

            # ══════════════════════════════════════════════════════════
            # A1: Downward Volatility — ★★★ CORE SELECTION FACTOR
            # IC=-0.073 (all), -0.106 (bear) at h=48h
            # Higher down_vol → lower future returns → short-friendly
            # ══════════════════════════════════════════════════════════
            down_dev = np.std(down_returns)
            up_dev = np.std(up_returns)
            down_vol = down_dev * np.sqrt(24) * 100  # daily annualized %
            up_vol = up_dev * np.sqrt(24) * 100

            # ══════════════════════════════════════════════════════════
            # A2: Downside Asymmetry — ★/★★★ CRISIS ALPHA
            # IC=+0.022 (all), but +0.126 in Transition!
            # Positive = more downside vol → short-friendly
            # ══════════════════════════════════════════════════════════
            asymmetry = (
                (down_dev - up_dev) / (down_dev + up_dev)
                if (down_dev + up_dev) > 0
                else 0.0
            )

            # ══════════════════════════════════════════════════════════
            # A3: Return Skewness — ✗/★★★ CRISIS ALPHA
            # IC=-0.017 (all), but -0.121 in Transition!
            # Negative skew → left tail → short-friendly (in crisis)
            # ══════════════════════════════════════════════════════════
            skewness = float(sp_stats.skew(returns))

            # ══════════════════════════════════════════════════════════
            # A4: Volatility Expansion — ★ WEAK (Bull only)
            # IC=-0.028 (all), -0.041 (bull), -0.000 (bear)
            # vol_recent / vol_full > 1.0 → expanding → bearish
            # ══════════════════════════════════════════════════════════
            recent_returns = returns[-24:]
            vol_recent = np.std(recent_returns) * np.sqrt(24) * 100
            vol_full = np.std(returns) * np.sqrt(24) * 100
            vol_expansion = vol_recent / vol_full if vol_full > 0 else 1.0
            vol_expansion = min(vol_expansion, self.vol_expansion_cap)

            # A5: Adverse spike filter
            recent_up = recent_returns[recent_returns > 0]
            adverse_spike = False
            if len(recent_up) > 0:
                max_up_spike = np.max(recent_up)
                avg_up = np.mean(np.abs(up_returns))
                adverse_spike = max_up_spike > self.adverse_spike_mult * avg_up

            # ══════════════════════════════════════════════════════════
            # OHLCV-based features (Doc §2.1, §5.2, §6)
            # ══════════════════════════════════════════════════════════
            parkinson_vol = compute_parkinson_volatility(highs, lows)
            rogers_satchell_vol = compute_rogers_satchell_volatility(
                opens, highs, lows, closes
            )
            upper_shadow, lower_shadow = compute_shadow_features(
                opens, highs, lows, closes, lookback=min(24, len(closes) - 1)
            )
            volume_sma_ratio, volume_price_divergence = compute_volume_features(
                closes, volumes, lookback=min(24, len(closes) - 1)
            )
            vpin_proxy = compute_vpin_proxy(
                opens, closes, volumes if volumes is not None else np.zeros_like(closes),
                lookback=min(24, len(closes) - 1),
            )
            lfi_skew, lfi_kurtosis = compute_lfi_proxy(
                highs, lows, closes, lookback=min(48, len(closes) - 1)
            )
            benford_deviation = compute_benford_deviation(
                volumes, lookback=min(168, len(closes) - 1)
            ) if volumes is not None else 0.0

            # ══════════════════════════════════════════════════════════
            # B1: TS-CO (Continuous Overreaction) — ★★★ ENTRY FILTER
            # IC=+0.165 @ 1h → MOMENTUM, not mean-reversion!
            # High CO = strong upward momentum → DON'T short
            # Low CO = downward momentum → safe to short
            # ══════════════════════════════════════════════════════════
            co_window = returns[-self.co_lookback :]
            weights = np.array(
                [self.co_decay**k for k in range(len(co_window) - 1, -1, -1)]
            )
            co_score = float(np.sum(np.sign(co_window) * weights))

            self._co_history.append(co_score)

            # ══════════════════════════════════════════════════════════
            # B2: Disposition Effect (VWAP Deviation) — ★★★ DUAL SIGNAL
            # h=1-6h: IC=+0.167 → above VWAP = more upside (momentum)
            # h=24-48h: IC=-0.022 → above VWAP = revert (mean-reversion)
            # Used as ENTRY FILTER (short-term) in Layer 2
            # ══════════════════════════════════════════════════════════
            if volumes is not None and len(volumes) >= self.vwap_lookback:
                vwap_closes = closes[-self.vwap_lookback :]
                vwap_vols = volumes[-self.vwap_lookback :]
                total_vv = np.sum(vwap_vols)
                if total_vv > 0:
                    vwap = np.sum(vwap_closes * vwap_vols) / total_vv
                else:
                    vwap = closes[-1]
                disposition = (closes[-1] - vwap) / vwap if vwap > 0 else 0.0
            else:
                p_ref = np.mean(closes[-self.vwap_lookback :])
                disposition = (closes[-1] - p_ref) / p_ref if p_ref > 0 else 0.0

            # Custom data feature pipeline (depth, quote, etc.): merge all calculator outputs
            custom_features: Dict[str, float] = {}
            for get_bar_fn, compute_fn in self._custom_feature_calculators:
                bar = get_bar_fn(algorithm, symbol)
                custom_features.update(compute_fn(bar))
            depth_obi_l5 = custom_features.get("depth_obi_l5", 0.0)
            depth_spread = custom_features.get("depth_spread", 0.0)
            depth_micro_price_divergence = custom_features.get(
                "depth_micro_price_divergence", 0.0
            )

            # Depth-flow features from consecutive L5 snapshots (Doc §4.1 / §4.3)
            flow_features = self._depth_flow_features.get(symbol, {}) or {}
            depth_ofi_1m = float(flow_features.get("depth_ofi_1m", 0.0))
            depth_lcr_1m = float(flow_features.get("depth_lcr_1m", 0.0))

            # SELECTION WEIGHT (Layer 1) — Regime-Adaptive + depth
            selection_weight = self._compute_selection_weight(
                down_vol,
                asymmetry,
                skewness,
                vol_expansion,
                disposition,
                depth_obi_l5,
                depth_micro_price_divergence,
            )

            factor_dict = {
                "downward_vol": round(down_vol, 2),
                "upward_vol": round(up_vol, 2),
                "asymmetry": round(asymmetry, 4),
                "skewness": round(skewness, 3),
                "vol_expansion": round(vol_expansion, 3),
                "adverse_spike": adverse_spike,
                "co_score": round(co_score, 3),
                "disposition": round(disposition, 4),
                "selection_weight": round(selection_weight, 4),
                "depth_obi_l5": round(depth_obi_l5, 4),
                "depth_spread": round(depth_spread, 6),
                "depth_micro_price_divergence": round(depth_micro_price_divergence, 4),
                "depth_ofi_1m": round(depth_ofi_1m, 4),
                "depth_lcr_1m": round(depth_lcr_1m, 4),
                # OHLCV-based (Doc §2.1, §5.2, §6)
                "parkinson_vol": round(parkinson_vol, 2),
                "rogers_satchell_vol": round(rogers_satchell_vol, 2),
                "upper_shadow": round(upper_shadow, 4),
                "lower_shadow": round(lower_shadow, 4),
                "volume_sma_ratio": round(volume_sma_ratio, 4),
                "volume_price_divergence": round(volume_price_divergence, 4),
                "vpin_proxy": round(vpin_proxy, 4),
                "lfi_skew": round(lfi_skew, 3),
                "lfi_kurtosis": round(lfi_kurtosis, 3),
                "benford_deviation": round(benford_deviation, 4),
            }

            return factor_dict, returns

        except Exception as e:
            algorithm.Debug(f"[Alpha] Error for {symbol.Value}: {e}")
            return None

    # ================================================================
    # Selection Weight Calculation (Regime-Adaptive + depth)
    # ================================================================
    def _compute_selection_weight(
        self,
        down_vol: float,
        asymmetry: float,
        skewness: float,
        vol_expansion: float,
        disposition: float,
        depth_obi_l5: float,
        depth_micro_price_divergence: float,
    ) -> float:
        """
        Compute regime-adaptive selection weight per §10.2 and add depth features.
        """
        regime = self.current_regime

        # Normalize each factor to [0, 1]
        s_dvol = min(down_vol / 20.0, 1.0)
        s_vexp = min(max(vol_expansion - 0.8, 0.0) / 1.5, 1.0)
        s_asym = min(max(asymmetry + 0.1, 0.0) / 0.5, 1.0)
        s_skew = min(max(-skewness, 0.0) / 1.5, 1.0)
        s_disp = min(max(disposition + 0.02, 0.0) / 0.08, 1.0)

        # Depth: negative OBI (ask-dominant) and micro-price below mid are short-friendly
        s_obi = min(max(-depth_obi_l5, 0.0) / 1.0, 1.0)
        s_mp = min(max(-depth_micro_price_divergence, 0.0) / 1.0, 1.0)

        if regime == self.REGIME_TRANSITION:
            w = (
                0.15 * s_dvol
                + 0.10 * s_vexp
                + 0.30 * s_asym
                + 0.25 * s_skew
                + 0.20 * s_disp
                + 0.10 * s_obi
                + 0.10 * s_mp
            )
        elif regime == self.REGIME_BEAR:
            w = (
                0.55 * s_dvol
                + 0.00 * s_vexp
                + 0.05 * s_asym
                + 0.05 * s_skew
                + 0.00 * s_disp
                + 0.15 * s_obi
                + 0.10 * s_mp
            )
        else:
            w = (
                0.40 * s_dvol
                + 0.20 * s_vexp
                + 0.05 * s_asym
                + 0.00 * s_skew
                + 0.20 * s_disp
                + 0.10 * s_obi
                + 0.05 * s_mp
            )

        selection_weight = max(w, 0.0) ** self.selection_weight_power
        return selection_weight

    # ================================================================
    # CSAD — Cross-Sectional Herding
    # ================================================================
    def _compute_csad(self, cross_returns: Dict[Symbol, np.ndarray]) -> None:
        """
        CSAD regression: CSAD_t = α + γ1·|R_m,t| + γ2·R_m,t²
        γ2 significantly negative → herding → reversal signal

        Report §12: Only 1.2% trigger rate, but 9.0x predictive in Bull.
        """
        if len(cross_returns) < 5:
            self.csad_gamma2 = 0.0
            self.csad_herding = False
            return

        min_len = min(len(r) for r in cross_returns.values())
        min_len = min(min_len, self.volatility_lookback_hours)

        if min_len < self.csad_lookback:
            self.csad_gamma2 = 0.0
            self.csad_herding = False
            return

        all_returns = np.array([r[-min_len:] for r in cross_returns.values()])
        r_market = np.mean(all_returns, axis=0)
        csad_series = np.mean(np.abs(all_returns - r_market[np.newaxis, :]), axis=0)

        T = len(r_market)
        X = np.column_stack(
            [
                np.ones(T),
                np.abs(r_market),
                r_market**2,
            ]
        )

        try:
            beta = np.linalg.lstsq(X, csad_series, rcond=None)[0]
            gamma2 = float(beta[2])

            residuals = csad_series - X @ beta
            s2 = np.sum(residuals**2) / (T - 3)
            var_beta = s2 * np.linalg.inv(X.T @ X)
            t_stat = gamma2 / np.sqrt(var_beta[2, 2]) if var_beta[2, 2] > 0 else 0.0

            self.csad_gamma2 = gamma2
            self.csad_herding = gamma2 < 0 and t_stat < -2.0

        except Exception:
            self.csad_gamma2 = 0.0
            self.csad_herding = False

    def _get_depth_bar(self, algorithm: QCAlgorithm, symbol: Symbol) -> Optional[CryptoFutureDepthData]:
        """Provide latest depth snapshot for the given symbol to the custom feature pipeline."""
        return self.latest_depth.get(symbol)

    # ================================================================
    # Universe changes
    # ================================================================
    def register_symbol(self, algorithm: QCAlgorithm, symbol: Symbol) -> None:
        """
        Lightweight helper for non-Framework algorithms to register symbols.
        Mirrors the logic of on_securities_changed(add) for a single symbol.
        """
        base = symbol.value.replace("USDT", "").replace("BUSD", "")

        if base in self.exclusion_list:
            return

        self._all_symbols.add(symbol)
        self.coin_data[symbol] = {
            "downward_vol": 0.0,
            "upward_vol": 0.0,
            "asymmetry": 0.0,
            "skewness": 0.0,
            "vol_expansion": 1.0,
            "adverse_spike": False,
            "co_score": 0.0,
            "disposition": 0.0,
            "selection_weight": 0.0,
            "depth_obi_l5": 0.0,
            "depth_spread": 0.0,
            "depth_micro_price_divergence": 0.0,
            "depth_ofi_1m": 0.0,
            "depth_lcr_1m": 0.0,
            "parkinson_vol": 0.0,
            "rogers_satchell_vol": 0.0,
            "upper_shadow": 0.0,
            "lower_shadow": 0.0,
            "volume_sma_ratio": 1.0,
            "volume_price_divergence": 0.0,
            "vpin_proxy": 0.0,
            "lfi_skew": 0.0,
            "lfi_kurtosis": 0.0,
            "benford_deviation": 0.0,
        }
        if self.btc_ticker in symbol.value:
            self._btc_symbol = symbol

    def register_depth_symbol(self, symbol: Symbol, depth_symbol: Symbol) -> None:
        """
        Register mapping from trading symbol -> depth custom data symbol.
        Depth snapshots are then read from Slice in update().
        """
        self.depth_custom_symbols[symbol] = depth_symbol

    def on_securities_changed(
        self, algorithm: QCAlgorithm, changes: SecurityChanges
    ) -> None:
        for security in changes.added_securities:
            symbol = security.symbol
            base = symbol.value.replace("USDT", "").replace("BUSD", "")

            if base not in self.exclusion_list:
                self._all_symbols.add(symbol)
                self.coin_data[symbol] = {
                    "downward_vol": 5.0,
                    "upward_vol": 5.0,
                    "asymmetry": 0.0,
                    "skewness": 0.0,
                    "vol_expansion": 1.0,
                    "adverse_spike": False,
                    "co_score": 0.0,
                    "disposition": 0.0,
                    "selection_weight": 0.5,
                    "depth_obi_l5": 0.0,
                    "depth_spread": 0.0,
                    "depth_micro_price_divergence": 0.0,
                    "depth_ofi_1m": 0.0,
                    "depth_lcr_1m": 0.0,
                    "parkinson_vol": 0.0,
                    "rogers_satchell_vol": 0.0,
                    "upper_shadow": 0.0,
                    "lower_shadow": 0.0,
                    "volume_sma_ratio": 1.0,
                    "volume_price_divergence": 0.0,
                    "vpin_proxy": 0.0,
                    "lfi_skew": 0.0,
                    "lfi_kurtosis": 0.0,
                    "benford_deviation": 0.0,
                }
                # Track BTC symbol for regime detection
                if self.btc_ticker in symbol.value:
                    self._btc_symbol = symbol

        for security in changes.removed_securities:
            symbol = security.symbol
            self._all_symbols.discard(symbol)
            if symbol in self.coin_data:
                del self.coin_data[symbol]
