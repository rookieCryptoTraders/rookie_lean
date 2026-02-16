# region imports
from AlgorithmImports import *

# 导入自定义模块
from alpha import AltcoinShortAlphaModel
from execution import SmartSpreadExecutionModel
from risk import DynamicExitRiskModel
from portfolio import EnterAndHoldPCM
# endregion


class AltcoinShortAlgorithm(QCAlgorithm):
    """
    Altcoin Short Strategy - QuantConnect Algorithm Framework 实现

    策略概述：
    1. 做空市值排名 15-300 的 Altcoin（排除 RWA、AI、Stablecoin）
    2. 同时持有最多 30 个空仓
    3. 动态仓位 = Insight Weight (基于下行波动率)
    4. 风险管理：
       - 移动止损：5% (TrailingStopRiskManagementModel)
       - 硬止损：10% (MaximumDrawdownPercentPerSecurity)
       - 全局熔断：15% (MaximumDrawdownPercentPortfolio)
    5. 智能执行：
       - 入场：检查价差，过大则等待 (最长1小时)
       - 出场：立即执行，保命优先

    模块化架构：
    - Alpha Model: 生成做空信号 (Insight with Weight)
    - AddCryptoFuture: 直接添加 Binance 永续合约
    - Portfolio Construction Model: 官方 InsightWeighting 模型
    - Risk Management Model: 标准模型组合
    - Execution Model: 智能价差执行
    """

    def initialize(self) -> None:
        """初始化策略"""

        # =====================================================================
        # 基本设置
        # =====================================================================
        self.SetStartDate(2025, 10, 20)
        self.SetEndDate(2026, 2, 1)
        self.SetAccountCurrency("USDT")
        self.SetCash(100000)  # 100k USDT

        # 设置券商模型 - Binance Futures (永续合约)
        self.SetBrokerageModel(BrokerageName.BinanceFutures, AccountType.Margin)

        # =====================================================================
        # 重要：禁用 Insight/Security 变化时的自动重平衡
        # 只在定时器触发时重平衡 (每小时一次)
        # 风险管理订单仍然会正常触发
        # =====================================================================
        self.Settings.RebalancePortfolioOnInsightChanges = False
        self.Settings.RebalancePortfolioOnSecurityChanges = False

        # =====================================================================
        # 策略参数
        # =====================================================================
        MAX_POSITIONS = 5  # 100k / 5 = 20k per pos * 2x = 40k exposure
        self._max_positions = MAX_POSITIONS
        LEVERAGE = 2

        # =====================================================================
        # 添加 Crypto Futures - 使用 AddCryptoFuture
        # =====================================================================
        self._add_crypto_futures(leverage=LEVERAGE)

        # =====================================================================
        # 设置 Algorithm Framework 模块
        # =====================================================================

        # 1. Alpha Model - 生成做空信号
        self.SetAlpha(
            AltcoinShortAlphaModel(
                max_positions=MAX_POSITIONS,
                volatility_lookback_hours=168,  # 7 days lookback
                volatility_cache_hours=6,  # Refresh every 6h
                selection_weight_power=1.5,
                max_adverse_vol_ratio=3.0,  # Relaxed from 2.0 (IC report §8.3)
                insight_duration_hours=48,  # Extended from 24h (downward_vol peaks at 48h)
            )
        )

        # 2. Portfolio Construction Model - Enter-and-Hold
        # 固定仓位大小, 只开新仓不调仓, 出场由 Risk Model 控制
        self.SetPortfolioConstruction(
            EnterAndHoldPCM(
                max_positions=MAX_POSITIONS,
                leverage=LEVERAGE,
            )
        )

        # 3. Risk Management Model - 动态出场风控
        # 策略：自定义模型 + 全局熔断

        # A. 动态出场模型 (替代标准 TrailingStop + MaxDDPerSecurity)
        #    - Flash Crash TP: ≤4h 浮盈>8% → 减仓50%
        #    - Dynamic Trailing: 最大浮盈>10% → trailing 5%收紧至2%
        #    - Hard Stop: 亏损>10% → 全平
        self.AddRiskManagement(
            DynamicExitRiskModel(
                flash_tp_threshold=0.08,
                flash_tp_hours=4.0,
                trailing_default=0.05,
                trailing_tight=0.02,
                tight_threshold=0.10,
                hard_stop=0.10,
            )
        )

        # B. 全局熔断 (防止系统性风险) - 15% 总资金回撤
        # 当总权益回撤超过 15% 时，平掉所有仓位，防止账户爆仓
        self.AddRiskManagement(MaximumDrawdownPercentPortfolio(0.15))

        # 4. Execution Model - 智能价差执行
        # 入场时检查 spread (最大 0.5%)，超时 1 小时放弃
        # 出场时立即执行，不管滑点
        self.SetExecution(
            SmartSpreadExecutionModel(
                max_spread_percent=0.005,  # 0.5% 最大可接受价差
                max_wait_hours=1.0,  # 最长等待 1 小时
            )
        )

        # =====================================================================
        # 定时任务
        # =====================================================================
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(hours=1)),
            self._log_status,
        )

        self.Debug("=" * 60)
        self.Debug("Altcoin Short Strategy Initialized")
        self.Debug(f"Max Positions: {MAX_POSITIONS} | Leverage: {LEVERAGE}x")
        self.Debug(
            "Using DynamicExitRiskModel (FlashTP=8%/4h | Trail=5%→2% | HardStop=10%)"
        )
        self.Debug("=" * 60)

    def _add_crypto_futures(self, leverage: int = 2) -> None:
        """
        添加 Binance 永续合约交易标的

        使用 AddCryptoFuture 是添加永续合约的标准方式，
        它会自动配置正确的数据路径和合约属性。
        """
        # Altcoin 列表 - 市值排名 15-300 的代币
        # 注意：排除 BTC/ETH (太大)、Stablecoin、RWA 类
        tickers = [
            # Top Layer 1s (排除 BTC/ETH)
            "BNBUSDT",
            "XRPUSDT",
            "ADAUSDT",
            "SOLUSDT",
            "DOTUSDT",
            "LTCUSDT",
            "AVAXUSDT",
            "TRXUSDT",
            # Layer 2s
            "OPUSDT",
            "ARBUSDT",
            # Newer High Cap
            "APTUSDT",
            "SUIUSDT",
            "SEIUSDT",
            "TIAUSDT",
            # "EOSUSDT",  # no trade data available
            "NEARUSDT",
            "ATOMUSDT",
            "ETCUSDT",
            "XLMUSDT",
            "FILUSDT",
            "HBARUSDT",
            "VETUSDT",
            "ICPUSDT",
            "INJUSDT",
            "STXUSDT",
            # DeFi Blue Chips
            "LINKUSDT",
            "UNIUSDT",
            "AAVEUSDT",
            "CRVUSDT",
            "LDOUSDT",
            # Gaming / Metaverse
            "AXSUSDT",
            "SANDUSDT",
            "MANAUSDT",
            "GALAUSDT",
            # Infrastructure
            "GRTUSDT",
            "FETUSDT",
            "JASMYUSDT",
            # Older Gen
            "BCHUSDT",
            "ALGOUSDT",
            "XTZUSDT",
            "CHZUSDT",
            "NEOUSDT",
            "IOTAUSDT",
            "DASHUSDT",
            "ZECUSDT",
            "XMRUSDT",
            # Meme & Community
            "DOGEUSDT",
            "WIFUSDT",
            # Others / High Volatility candidates
            "CAKEUSDT",
            "QNTUSDT",
        ]

        added_count = 0
        for ticker in tickers:
            try:
                # AddCryptoFuture: 添加永续合约，使用分钟级K线数据
                crypto_future = self.AddCryptoFuture(
                    ticker,
                    Resolution.Minute,
                    Market.Binance,
                    fillForward=True,
                    leverage=leverage,
                )

                added_count += 1
                self.Debug(f"[Init] Added CryptoFuture: {ticker}")
            except Exception as e:
                self.Debug(f"[Init] Failed to add {ticker}: {e}")

        self.Debug(f"[Init] Total CryptoFutures added: {added_count}/{len(tickers)}")

    def _log_status(self) -> None:
        """定时状态日志"""
        holdings = [h for h in self.Portfolio.Values if h.Invested]

        total_pnl = sum(h.UnrealizedProfit for h in holdings)
        margin_used = sum(abs(h.HoldingsValue) for h in holdings)

        self.Debug(
            f"[Status] Positions: {len(holdings)}/{self._max_positions} | "
            f"PnL: ${total_pnl:.2f} | "
            f"Margin: ${margin_used:.2f} | "
            f"Equity: ${self.Portfolio.TotalPortfolioValue:.2f}"
        )

    def OnOrderEvent(self, order_event: OrderEvent) -> None:
        """订单事件回调"""
        if order_event.Status == OrderStatus.Filled:
            direction = "SHORT" if order_event.FillQuantity < 0 else "COVER"
            self.Debug(
                f"[Order] {direction} {order_event.Symbol.Value}: "
                f"Qty={order_event.FillQuantity:.4f} @ ${order_event.FillPrice:.4f}"
            )

    def OnEndOfAlgorithm(self) -> None:
        """策略结束回调"""
        self.Debug("=" * 60)
        self.Debug("Altcoin Short Strategy Completed")
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Debug("=" * 60)
