# region imports
from AlgorithmImports import *
from typing import List, Dict
from datetime import datetime, timedelta
# endregion


class SmartSpreadExecutionModel(ExecutionModel):
    """
    Smart Spread Execution Model

    智能执行模型：
    - 入场（增加敞口）：检查 Spread，如果过大则等待
    - 出场（平仓/减仓）：立即执行，保命优先

    参数：
    - max_spread_percent: 入场时可接受的最大买卖价差 (默认 0.5%)
    - max_wait_hours: 最长等待时间，超过后放弃该信号 (默认 1 小时)
    """

    def __init__(self, max_spread_percent: float = 0.005, max_wait_hours: float = 1.0):

        super().__init__()
        self.max_spread_percent = max_spread_percent
        self.max_wait_hours = max_wait_hours

        # 跟踪 pending targets
        self.targets_collection = PortfolioTargetCollection()

        # 记录每个 target 首次出现的时间（用于超时判断）
        self.target_first_seen: Dict[Symbol, datetime] = {}

    def Execute(self, algorithm: QCAlgorithm, targets: List[PortfolioTarget]) -> None:
        """
        核心执行方法

        逻辑：
        1. 判断是入场还是出场
        2. 出场：立即执行
        3. 入场：检查 spread，满足条件才执行
        """
        # 添加新 targets 到集合
        self.targets_collection.AddRange(targets)

        # 当前时间
        current_time = algorithm.Time

        for target in self.targets_collection:
            symbol = target.Symbol

            # 检查 symbol 是否可交易
            if not algorithm.Securities.ContainsKey(symbol):
                continue

            security = algorithm.Securities[symbol]

            if not security.HasData:
                continue

            # 获取当前持仓
            holding = algorithm.Portfolio[symbol]
            current_quantity = holding.Quantity

            # 计算目标数量
            target_quantity = target.Quantity

            # 如果是百分比形式，转换为实际数量
            if abs(target_quantity) < 1:
                portfolio_value = algorithm.Portfolio.TotalPortfolioValue
                target_value = portfolio_value * abs(target_quantity)
                price = security.Price

                if price > 0:
                    target_quantity = (
                        -target_value / price
                        if target_quantity < 0
                        else target_value / price
                    )
                else:
                    continue

            # 计算需要交易的数量
            quantity_to_trade = target_quantity - current_quantity

            # 最小交易价值检查
            min_trade_value = 5.0
            trade_value = abs(quantity_to_trade * security.Price)

            if trade_value < min_trade_value:
                # 移除已满足的 target
                self.targets_collection.Remove(symbol)
                if symbol in self.target_first_seen:
                    del self.target_first_seen[symbol]
                continue

            # =========================================================
            # 核心逻辑：判断是入场还是出场
            # =========================================================
            is_closing = self._is_closing_position(current_quantity, quantity_to_trade)

            if is_closing:
                # =====================================================
                # 出场/平仓：立即执行，不管滑点
                # =====================================================
                self._execute_immediately(algorithm, symbol, quantity_to_trade, "CLOSE")
                self.targets_collection.Remove(symbol)
                if symbol in self.target_first_seen:
                    del self.target_first_seen[symbol]
            else:
                # =====================================================
                # 入场/加仓：检查 Spread
                # =====================================================

                # 记录首次看到时间
                if symbol not in self.target_first_seen:
                    self.target_first_seen[symbol] = current_time

                first_seen = self.target_first_seen[symbol]
                time_waited = (current_time - first_seen).total_seconds() / 3600

                # 检查是否超时
                if time_waited >= self.max_wait_hours:
                    # 超时：放弃该信号
                    algorithm.Debug(
                        f"[Exec] TIMEOUT: {symbol.Value} waited {time_waited:.2f}h, abandoning signal"
                    )
                    self.targets_collection.Remove(symbol)
                    del self.target_first_seen[symbol]
                    continue

                # 检查 Spread
                spread_ok = self._check_spread(algorithm, security)

                if spread_ok:
                    # Spread 可接受：执行
                    self._execute_immediately(
                        algorithm, symbol, quantity_to_trade, "OPEN"
                    )
                    self.targets_collection.Remove(symbol)
                    if symbol in self.target_first_seen:
                        del self.target_first_seen[symbol]
                else:
                    # Spread 过大：等待下一个 tick
                    bid = security.BidPrice
                    ask = security.AskPrice
                    mid = (bid + ask) / 2 if (bid + ask) > 0 else security.Price
                    spread_pct = (ask - bid) / mid if mid > 0 else 0

                    algorithm.Debug(
                        f"[Exec] WAITING: {symbol.Value} | Spread: {spread_pct * 100:.3f}% > {self.max_spread_percent * 100:.2f}% | "
                        f"Waited: {time_waited:.2f}h"
                    )

    def _is_closing_position(
        self, current_quantity: float, quantity_to_trade: float
    ) -> bool:
        """
        判断是否是平仓/减仓操作

        平仓定义：
        - 当前持有空头 (quantity < 0)，正在买回 (trade > 0)
        - 或当前持有多头 (quantity > 0)，正在卖出 (trade < 0)
        - 或目标是完全平仓 (target = 0)
        """
        # 如果没有持仓，这是入场
        if current_quantity == 0:
            return False

        # 如果交易方向与持仓方向相反，是平仓
        if current_quantity < 0 and quantity_to_trade > 0:
            return True  # 空头平仓
        if current_quantity > 0 and quantity_to_trade < 0:
            return True  # 多头平仓

        return False

    def _check_spread(self, algorithm: QCAlgorithm, security) -> bool:
        """
        检查买卖价差是否在可接受范围内
        """
        bid = security.BidPrice
        ask = security.AskPrice

        # 如果没有 bid/ask 数据，直接通过（使用 last price）
        if bid <= 0 or ask <= 0:
            return True

        mid = (bid + ask) / 2
        if mid <= 0:
            return True

        spread_percent = (ask - bid) / mid

        return spread_percent <= self.max_spread_percent

    def _execute_immediately(
        self, algorithm: QCAlgorithm, symbol: Symbol, quantity: float, action: str
    ) -> None:
        """
        立即执行市价单
        """
        if quantity == 0:
            return

        ticket = algorithm.MarketOrder(symbol, quantity)

        direction = "SHORT" if quantity < 0 else "BUY/COVER"
        trade_value = abs(quantity * algorithm.Securities[symbol].Price)

        algorithm.Debug(
            f"[Exec] {action} {direction} {symbol.Value}: "
            f"Qty={quantity:.4f} (${trade_value:.2f})"
        )

    def OnSecuritiesChanged(
        self, algorithm: QCAlgorithm, changes: SecurityChanges
    ) -> None:
        """处理 Universe 变更"""
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if self.targets_collection.Contains(symbol):
                self.targets_collection.Remove(symbol)
            if symbol in self.target_first_seen:
                del self.target_first_seen[symbol]
