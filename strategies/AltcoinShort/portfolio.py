# region imports
from AlgorithmImports import *
from typing import List, Dict, Set, Optional
# endregion


class EnterAndHoldPCM(PortfolioConstructionModel):
    """
    Enter-and-Hold Portfolio Construction Model.

    逻辑:
    - 收到新 Insight → 只在有空余 slot 时开仓
    - 固定仓位大小: (Cash × Leverage) / MaxPositions
    - 已持仓 symbol 不产生任何 target (不调仓)
    - 出场完全交给 RiskManagementModel (DynamicExitRiskModel)
    """

    def __init__(
        self,
        max_positions: int = 5,
        leverage: float = 2.0,
        rebalance: Resolution = Resolution.Hour,
    ):
        super().__init__()
        self.max_positions = max_positions
        self.leverage = leverage
        self._rebalance = rebalance

        # 追踪 PCM 主动开仓的 symbol (区别于 Risk Model 平仓后的 symbol)
        self._active_symbols: Set = set()

    def CreateTargets(
        self, algorithm: QCAlgorithm, insights: List[Insight]
    ) -> List[PortfolioTarget]:
        """
        仅对新 symbol 生成开仓 target, 不触碰已持仓 symbol.
        """
        targets: List[PortfolioTarget] = []

        # ── 同步: 检测 Risk Model 平仓的 symbol, 从追踪中移除 ──
        closed_symbols = set()
        for sym in list(self._active_symbols):
            holding = algorithm.Portfolio.get(sym)
            if holding is None or not holding.Invested:
                closed_symbols.add(sym)
        self._active_symbols -= closed_symbols

        # ── 当前可用 slot ──
        current_positions = len(self._active_symbols)
        available_slots = self.max_positions - current_positions

        if available_slots <= 0:
            return targets

        # ── 计算固定仓位大小 (notional) ──
        total_equity = algorithm.Portfolio.TotalPortfolioValue
        position_notional = (total_equity * self.leverage) / self.max_positions

        # ── 按 weight 降序排列, 只取 Down direction 的 insight ──
        valid_insights = [
            i
            for i in insights
            if i.Direction == InsightDirection.Down
            and i.Weight is not None
            and i.Weight > 0.01
        ]
        valid_insights.sort(key=lambda i: i.Weight, reverse=True)

        # ── 开新仓 ──
        for insight in valid_insights:
            if available_slots <= 0:
                break

            symbol = insight.Symbol

            # 跳过已持仓
            if symbol in self._active_symbols:
                continue

            # 跳过已有持仓但不在追踪中 (可能是刚被 Risk 平仓过的)
            holding = algorithm.Portfolio.get(symbol)
            if holding is not None and holding.Invested:
                continue

            # 计算做空数量
            price = algorithm.Securities[symbol].Price
            if price <= 0:
                continue

            quantity = position_notional / price
            # 做空 → 负数
            target_quantity = -quantity

            targets.append(PortfolioTarget(symbol, target_quantity))
            self._active_symbols.add(symbol)
            available_slots -= 1

            algorithm.Debug(
                f"[PCM] NEW SHORT {symbol.Value}: "
                f"qty={target_quantity:.4f} @ ${price:.4f} | "
                f"notional=${position_notional:.0f} | "
                f"slots={self.max_positions - available_slots}/{self.max_positions}"
            )

        return targets

    def OnSecuritiesChanged(
        self, algorithm: QCAlgorithm, changes: SecurityChanges
    ) -> None:
        """清理被移除的 symbol."""
        for security in changes.RemovedSecurities:
            self._active_symbols.discard(security.Symbol)
