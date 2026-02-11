# region imports
from AlgorithmImports import *
import random
import numpy as np
from typing import List, Dict, Set, Optional
# endregion


class AltcoinShortAlphaModel(AlphaModel):
    """
    Altcoin Short Alpha Model

    职责：生成做空信号 (Insight)
    逻辑：
    1. 筛选币种池（排除 RWA, AI, Stablecoin）
    2. 计算波动率指标
    3. 根据下行波动率偏好加权选择币种
    4. 生成做空 Insight
    """

    def __init__(
        self,
        max_positions: int = 30,
        volatility_lookback_hours: int = 168,
        volatility_cache_hours: int = 6,
        selection_weight_power: float = 1.5,
        max_adverse_vol_ratio: float = 2.0,
        insight_duration_hours: int = 24,
    ):

        self.max_positions = max_positions
        self.volatility_lookback_hours = volatility_lookback_hours
        self.volatility_cache_hours = volatility_cache_hours
        self.selection_weight_power = selection_weight_power
        self.max_adverse_vol_ratio = max_adverse_vol_ratio
        self.insight_duration = timedelta(hours=insight_duration_hours)

        # 排除列表
        self.exclusion_list = self._build_exclusion_list()

        # 状态
        self.coin_data: Dict[Symbol, dict] = {}  # symbol -> volatility data
        self.last_volatility_update: Optional[datetime] = None
        self.active_insights: Set[Symbol] = set()  # 持有活跃 insight 的 symbols

    def _build_exclusion_list(self) -> Set[str]:
        """构建排除列表"""
        rwa = {
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
        }

        ai = {
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
        }

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

        return rwa | ai | stablecoin

    def Update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        """
        核心方法：生成 Insights

        每次数据更新时调用，返回做空信号列表
        """
        insights = []

        # 每天只运行一次核心逻辑 (例如在 00:00)
        # 或者每小时运行一次
        if algorithm.Time.minute != 0:
            return insights

        # 检查是否需要刷新波动率
        if self._should_refresh_volatility(algorithm.Time):
            self._refresh_volatility(algorithm)

        # 计算可用仓位数
        current_active = len(self.active_insights)

        # 获取所有可交易的币种
        candidates = []
        for symbol in self.coin_data.keys():
            if not data.ContainsKey(symbol) or data[symbol] is None:
                continue
            candidates.append(symbol)

        if not candidates:
            return insights

        # 生成做空 Insights (带有权重)
        # 权重 = selection_weight (基于下行波动率偏好)
        raw_insights = []
        for symbol in candidates:
            vol_data = self.coin_data.get(symbol, {})
            weight = vol_data.get("selection_weight", 0.0)

            # 如果权重太小，忽略
            if weight < 0.1:
                continue

            # 创建做空 Insight
            # Weight 参数非常重要：它会被 Portfolio Model 用来分配资金
            insight = Insight.Price(
                symbol,
                self.insight_duration,
                InsightDirection.Down,  # 做空方向
                vol_data.get("downward_vol", 5.0) / 100,  # magnitude
                vol_data.get("downward_bias", 0.5),  # confidence
                None,  # sourceModel
                weight,  # weight -> 用于 Portfolio Construction
            )

            # 附加波动率数据到 insight 的 tag
            insight.Tag = f"down_vol:{vol_data.get('downward_vol', 0):.2f}|up_vol:{vol_data.get('upward_vol', 0):.2f}"

            raw_insights.append((weight, insight))

        # 按权重降序排列，只取前 max_positions 个信号
        raw_insights.sort(key=lambda x: x[0], reverse=True)
        insights = [ins for _, ins in raw_insights[: self.max_positions]]

        return insights

    def _should_refresh_volatility(self, current_time: datetime) -> bool:
        """判断是否需要刷新波动率"""
        if self.last_volatility_update is None:
            return True
        return (
            current_time - self.last_volatility_update
        ).total_seconds() / 3600 >= self.volatility_cache_hours

    def _refresh_volatility(self, algorithm: QCAlgorithm) -> None:
        """刷新所有币种的波动率数据"""
        updated = 0
        removed = 0

        for symbol in list(self.coin_data.keys()):
            # 确保 symbol 在证券池中
            if not algorithm.Securities.ContainsKey(symbol):
                continue

            vol_data = self._calculate_volatility(algorithm, symbol)

            if vol_data is None:
                continue

            # 检查不利波动率比
            if vol_data["upward_vol"] > 0 and vol_data["downward_vol"] > 0:
                ratio = vol_data["upward_vol"] / vol_data["downward_vol"]
                if ratio > self.max_adverse_vol_ratio:
                    # 不利波动太大，移除并不产生信号
                    if symbol in self.coin_data:
                        del self.coin_data[symbol]
                    removed += 1
                    continue

            self.coin_data[symbol] = vol_data
            updated += 1

        self.last_volatility_update = algorithm.Time
        algorithm.Debug(
            f"[Alpha] Volatility refreshed: {updated} updated, {removed} removed"
        )

    def _calculate_volatility(
        self, algorithm: QCAlgorithm, symbol: Symbol
    ) -> Optional[dict]:
        """计算单个币种的波动率"""
        try:
            history = algorithm.History(
                symbol, self.volatility_lookback_hours, Resolution.Hour
            )

            if history.empty or len(history) < 24:
                return None

            closes = history["close"].values
            returns = np.diff(closes) / closes[:-1]

            down_returns = returns[returns < 0]
            up_returns = returns[returns > 0]

            if len(down_returns) < 5 or len(up_returns) < 5:
                return None

            # 转换为日波动率百分比
            down_vol = abs(np.std(down_returns)) * np.sqrt(24) * 100
            up_vol = abs(np.std(up_returns)) * np.sqrt(24) * 100

            downward_bias = (
                down_vol / (down_vol + up_vol) if (down_vol + up_vol) > 0 else 0.5
            )

            # 核心权重计算
            selection_weight = downward_bias**self.selection_weight_power

            return {
                "downward_vol": round(down_vol, 2),
                "upward_vol": round(up_vol, 2),
                "downward_bias": round(downward_bias, 3),
                "selection_weight": round(selection_weight, 3),
            }
        except Exception as e:
            # algorithm.Debug(f"Error cal vol for {symbol}: {e}")
            return None

    def OnSecuritiesChanged(
        self, algorithm: QCAlgorithm, changes: SecurityChanges
    ) -> None:
        """处理 Universe 变更"""
        # 新增的证券
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            ticker = symbol.Value

            # 简单检查 ticker (假设格式为 XXXUSDT)
            base = ticker.replace("USDT", "").replace("BUSD", "")

            # 过滤排除列表
            if base not in self.exclusion_list:
                self.coin_data[symbol] = {
                    "downward_vol": 5.0,
                    "upward_vol": 5.0,
                    "downward_bias": 0.5,
                    "selection_weight": 1.0,
                }
                algorithm.Debug(f"[Alpha] Added: {symbol.Value}")

        # 移除的证券
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.coin_data:
                del self.coin_data[symbol]
            algorithm.Debug(f"[Alpha] Removed: {symbol.Value}")
