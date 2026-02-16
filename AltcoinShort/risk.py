# region imports
from AlgorithmImports import *
from typing import List, Dict
# endregion


class DynamicExitRiskModel(RiskManagementModel):
    """
    Dynamic Exit Risk Management Model

    动态出场风控模型，替代标准 TrailingStop + MaxDDPerSecurity:

    1. Flash Crash TP (急跌止盈):
       开仓 ≤4h 且浮盈 >8% → 减仓 50%，落袋为安

    2. Dynamic Trailing (动态追踪止损):
       默认 5% trailing; 最大浮盈 >10% 时收紧至 2%

    3. Hard Stop (硬止损):
       亏损 >10% → 全平

    参数:
    - flash_tp_threshold: 触发急跌止盈的浮盈阈值 (默认 0.08 = 8%)
    - flash_tp_hours:     急跌止盈生效的时间窗口 (默认 4h)
    - trailing_default:   默认追踪止损比例 (默认 0.05 = 5%)
    - trailing_tight:     收紧后的追踪止损比例 (默认 0.02 = 2%)
    - tight_threshold:    收紧追踪的浮盈阈值 (默认 0.10 = 10%)
    - hard_stop:          硬止损比例 (默认 0.10 = 10%)
    """

    def __init__(
        self,
        flash_tp_threshold: float = 0.08,
        flash_tp_hours: float = 4.0,
        trailing_default: float = 0.05,
        trailing_tight: float = 0.02,
        tight_threshold: float = 0.10,
        hard_stop: float = 0.10,
    ):
        super().__init__()
        self.flash_tp_threshold = flash_tp_threshold
        self.flash_tp_hours = flash_tp_hours
        self.trailing_default = trailing_default
        self.trailing_tight = trailing_tight
        self.tight_threshold = tight_threshold
        self.hard_stop = hard_stop

        # Per-symbol 状态追踪
        self._state: Dict[Symbol, dict] = {}

    def ManageRisk(
        self, algorithm: QCAlgorithm, targets: List[PortfolioTarget]
    ) -> List[PortfolioTarget]:
        """
        核心风控方法 — 每次 Portfolio 重平衡时被调用。
        遍历所有持仓，检查是否需要减仓/平仓。
        """
        risk_targets: List[PortfolioTarget] = []

        for kvp in algorithm.Securities:
            symbol = kvp.Key
            security = kvp.Value

            # 只处理有持仓的 symbol
            holding = algorithm.Portfolio[symbol]
            if not holding.Invested:
                # 清理已平仓的状态
                if symbol in self._state:
                    del self._state[symbol]
                continue

            current_price = security.Price
            if current_price <= 0:
                continue

            quantity = holding.Quantity
            avg_price = holding.AveragePrice
            current_time = algorithm.Time

            # ── 初始化状态 ──
            if symbol not in self._state:
                self._state[symbol] = {
                    "entry_time": current_time,
                    "entry_price": avg_price,
                    "best_price": current_price,
                    "max_pnl_pct": 0.0,
                    "partial_taken": False,
                }

            state = self._state[symbol]

            # 用 LEAN 的 AveragePrice 作为入场价 (更准确)
            entry_price = avg_price if avg_price > 0 else state["entry_price"]

            # ── 计算当前 PnL % ──
            # Short: 入场高卖, 当前低买 → profit = (entry - current) / entry
            # Long:  入场低买, 当前高卖 → profit = (current - entry) / entry
            is_short = quantity < 0
            if is_short:
                current_pnl_pct = (entry_price - current_price) / entry_price
                # 追踪最低价 (空头的最佳价格)
                state["best_price"] = min(state["best_price"], current_price)
            else:
                current_pnl_pct = (current_price - entry_price) / entry_price
                # 追踪最高价 (多头的最佳价格)
                state["best_price"] = max(state["best_price"], current_price)

            state["max_pnl_pct"] = max(state["max_pnl_pct"], current_pnl_pct)

            elapsed_hours = (current_time - state["entry_time"]).total_seconds() / 3600

            # ═══════════════════════════════════════════════════════
            # 1. Flash Crash TP: 快速止盈 (减仓 50%)
            # ═══════════════════════════════════════════════════════
            if (
                not state["partial_taken"]
                and elapsed_hours <= self.flash_tp_hours
                and current_pnl_pct > self.flash_tp_threshold
            ):
                # 减仓一半
                new_quantity = quantity * 0.5
                risk_targets.append(PortfolioTarget(symbol, new_quantity))
                state["partial_taken"] = True
                algorithm.Debug(
                    f"[Risk] FLASH_TP: {symbol.Value} | "
                    f"PnL={current_pnl_pct * 100:+.2f}% in {elapsed_hours:.1f}h | "
                    f"Reducing 50%"
                )
                continue  # 本轮不再检查其他条件

            # ═══════════════════════════════════════════════════════
            # 2. Hard Stop: 硬止损 (全平)
            # ═══════════════════════════════════════════════════════
            if current_pnl_pct < -self.hard_stop:
                risk_targets.append(PortfolioTarget(symbol, 0))
                algorithm.Debug(
                    f"[Risk] HARD_STOP: {symbol.Value} | "
                    f"Loss={current_pnl_pct * 100:+.2f}%"
                )
                if symbol in self._state:
                    del self._state[symbol]
                continue

            # ═══════════════════════════════════════════════════════
            # 3. Dynamic Trailing: 动态追踪止损 (全平)
            # ═══════════════════════════════════════════════════════
            # 选择 trailing 比例
            trailing_pct = (
                self.trailing_tight
                if state["max_pnl_pct"] > self.tight_threshold
                else self.trailing_default
            )

            # 计算从最佳价格的回撤
            best_price = state["best_price"]
            if is_short:
                # 空头: 从最低价反弹多少
                bounce = (
                    (current_price - best_price) / best_price if best_price > 0 else 0.0
                )
            else:
                # 多头: 从最高价回撤多少
                bounce = (
                    (best_price - current_price) / best_price if best_price > 0 else 0.0
                )

            if bounce > trailing_pct:
                tight_tag = " (TIGHT)" if trailing_pct == self.trailing_tight else ""
                risk_targets.append(PortfolioTarget(symbol, 0))
                algorithm.Debug(
                    f"[Risk] TRAILING{tight_tag}: {symbol.Value} | "
                    f"MaxPnL={state['max_pnl_pct'] * 100:+.2f}% | "
                    f"Bounce={bounce * 100:.2f}% > {trailing_pct * 100:.1f}%"
                )
                if symbol in self._state:
                    del self._state[symbol]
                continue

        return risk_targets
