# region imports
from AlgorithmImports import *
import os
import csv
import zipfile
from collections import deque
from datetime import datetime, timedelta
# TODO: use modern python type hint. e.g. list[int], dict[str, int], tuple[int, str], set[int], tuple[int, str] | set[int] etc.
from typing import List, Dict, Set, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from config import BASE_DATA_PATH, ASSET_CLASS, EXCHANGE
from utils import CryptoFutureDepthData
# endregion


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

        # Last generated insights (for hybrid algorithms reading signals in on_data)
        self.latest_insights: List[Insight] = []

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

        # 1. ALWAYS capture depth data if present in the current slice, regardless of the minute.
        # This ensures we have the freshest microstructure data available when we do refresh.
        # NOTE: Slice.get expects a data type (e.g., CryptoFutureDepthData), not a Symbol.
        #       We first fetch the depth dictionary, then index by the mapped depth Symbol.
        depth_captured = 0
        depth_dict = data.get(CryptoFutureDepthData)

        if depth_dict is None:
            algorithm.debug(f"[Alpha-Depth] No CryptoFutureDepthData in slice at {algorithm.time}")
        else:
            algorithm.debug(f"[Alpha-Depth] depth_dict keys={list(depth_dict.keys())} values={list(depth_dict.values())}")
            for symbol, depth_symbol in self.depth_custom_symbols.items():
                bar = depth_dict.get(depth_symbol)
                if bar is not None:
                    self.latest_depth[symbol] = bar
                    depth_captured += 1
                    # DEBUG: Trace the arrival of depth data in the alpha model
                    algorithm.log(
                        f"[Alpha-Depth-Flow] Captured depth for {symbol.value} at {algorithm.time} | Total Depth Value: {bar.value:.2f}"
                    )
                else:
                    # algorithm.debug(
                    #     f"[Alpha-Depth] No depth data for {symbol.value} at {algorithm.time}"
                    # )
                    pass

        # Optional periodic heartbeat to confirm depth reception
        if depth_captured > 0 and algorithm.time.minute % 15 == 0:
            algorithm.debug(f"[Alpha-Depth] Captured depth for {depth_captured} symbols at {algorithm.time}")

        # 2. Only run the rest of the alpha logic on full-hour bars to reduce noise and cost
        if algorithm.time.minute != 0:
            # Lightweight heartbeat every 30 minutes
            if algorithm.time.minute % 30 == 0:
                algorithm.debug(
                    f"[Alpha] heartbeat at {algorithm.time} (min={algorithm.time.minute}), pool={len(self.coin_data)}"
                )
            self.latest_insights = []
            return insights

        algorithm.debug(
            f"[Alpha] update at {algorithm.time} | current_regime={self.current_regime} | "
            f"tracked_symbols={len(self._all_symbols)}"
        )

        # Refresh factors every cache period (or if empty)
        if self._should_refresh(algorithm.time) or not self.coin_data:
            algorithm.debug("[Alpha] refreshing all factors and CSAD")
            self._detect_regime(algorithm)
            self._refresh_all_factors(algorithm)
        else:
            algorithm.debug(
                f"[Alpha] skip factor refresh | last_refresh={self.last_volatility_update}"
            )

        # Gather candidates with valid data
        candidates: List[Symbol] = []
        for symbol in self.coin_data:
            if not data.contains_key(symbol) or data[symbol] is None:
                continue
            candidates.append(symbol)

        if not candidates:
            algorithm.debug(
                f"[Alpha] no candidates at {algorithm.time} | coin_data={len(self.coin_data)}"
            )
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

            effective_weight = weight * momentum_penalty * depth_boost

            algorithm.debug(
                f"[Alpha-Weight] {symbol.value} base_w={weight:.3f} co={co_score:.3f} disp={disposition:.3f} obi={depth_obi_l5:.3f} mp={momentum_penalty:.3f} db={depth_boost:.3f} eff_w={effective_weight:.3f}"
            )

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

        # Debug summary of generated insights
        if not insights:
            algorithm.debug(
                f"[Alpha] update produced 0 insights | candidates={len(candidates)} | "
                f"pool={len(self.coin_data)}"
            )
        else:
            sample = ", ".join(
                f"{ins.symbol.value}:{(ins.weight or 0):.3f}"
                for ins in insights[: min(5, len(insights))]
            )
            algorithm.debug(
                f"[Alpha] update produced {len(insights)} insights "
                f"(top: {sample}) | pool={len(self.coin_data)} | regime={self.current_regime}"
            )

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

            # Latest depth snapshot for this symbol (may be absent)
            depth_bar = self.latest_depth.get(symbol)

            algorithm.debug(f"[test] test debug")
            if depth_bar is None:
                algorithm.debug(f"[Alpha-Depth] No depth snapshot for {symbol.value} at refresh at {algorithm.time}")
            else:
                algorithm.debug(f"[depth_bar] symbol={symbol.value} depth_bar={depth_bar} at {algorithm.time}")
                

            result = self._calculate_all_factors(algorithm, symbol, depth_bar)
            if result is None:
                # Keep existing data if available, don't delete
                algorithm.debug(f"[Alpha] Skip refresh for {symbol.Value} (insufficient history)")
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
        algorithm.debug(
            f"[Alpha] Refresh: {updated} ok, {removed} filtered | "
            f"Pool: {len(self.coin_data)} | Regime: {self.current_regime} | "
            f"CSAD γ2={self.csad_gamma2:.4f} herd={self.csad_herding}"
        )

    # ================================================================
    # Per-symbol factor calculation (+ depth features)
    # ================================================================
    def _calculate_all_factors(
        self,
        algorithm: QCAlgorithm,
        symbol: Symbol,
        depth_bar: Optional[CryptoFutureDepthData],
    ) -> Optional[Tuple[dict, np.ndarray]]:
        """
        Compute all factors for a single symbol.
        Returns (factor_dict, hourly_returns) or None.
        """
        try:
            history = algorithm.history(
                symbol, self.volatility_lookback_hours, Resolution.HOUR
            )

            if history.empty or len(history) < 12:
                return None

            closes = history["close"].values
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

            # Depth features from latest CryptoFutureDepthData snapshot
            depth_obi_l5 = 0.0
            depth_spread = 0.0
            depth_micro_price_divergence = 0.0
            

            if depth_bar is not None:
                try:
                    percentages = list(getattr(depth_bar, "percentages", []))
                    depths = list(getattr(depth_bar, "depths", []))
                    if percentages and depths and len(percentages) == len(depths):
                        bid_depth = sum(
                            d for d, p in zip(depths, percentages) if p < 0
                        )
                        ask_depth = sum(
                            d for d, p in zip(depths, percentages) if p > 0
                        )
                        total = bid_depth + ask_depth
                        if total > 0:
                            depth_obi_l5 = (bid_depth - ask_depth) / total
                        algorithm.debug(
                            f"[Alpha-Depth] calc {symbol.value} "
                            f"bid_depth={bid_depth:.2f} ask_depth={ask_depth:.2f} "
                            f"obi={depth_obi_l5:.4f}"
                        )
                except Exception as _:
                    depth_obi_l5 = 0.0

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
                "depth_micro_price_divergence": round(
                    depth_micro_price_divergence, 4
                ),
            }

            algorithm.debug(
                f"[Alpha-Factors] {symbol.value} dv={factor_dict['downward_vol']:.2f} "
                f"asym={factor_dict['asymmetry']:.3f} skew={factor_dict['skewness']:.2f} "
                f"vexp={factor_dict['vol_expansion']:.2f} disp={factor_dict['disposition']:.3f} "
                f"obi={factor_dict['depth_obi_l5']:.3f} sel_w={factor_dict['selection_weight']:.3f}"
            )

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

    # (old file-based depth loader removed; depth now comes from CryptoFutureDepthData)

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
        }
        algorithm.debug(f"[Alpha] Added: {symbol.value}")

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
                }
                algorithm.debug(f"[Alpha] Added: {symbol.value}")

                # Track BTC symbol for regime detection
                if self.btc_ticker in symbol.value:
                    self._btc_symbol = symbol

        for security in changes.removed_securities:
            symbol = security.symbol
            self._all_symbols.discard(symbol)
            if symbol in self.coin_data:
                del self.coin_data[symbol]
            algorithm.debug(f"[Alpha] Removed: {symbol.value}")
