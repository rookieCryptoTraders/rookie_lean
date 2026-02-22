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

        # 设置基准 - 显式指定为 Binance 的 BTCUSDT 永续合约
        # 避免字符串歧义，直接使用 Symbol 对象
        btc_benchmark = Symbol.Create(
            "BTCUSDT", SecurityType.CryptoFuture, Market.Binance
        )
        self.SetBenchmark(btc_benchmark)

        # =====================================================================
        # 重要：禁用 Insight/Security 变化时的自动重平衡
        # 只在定时器触发时重平衡 (每小时一次)
        # 风险管理订单仍然会正常触发
        # =====================================================================
        self.Settings.RebalancePortfolioOnInsightChanges = False
        self.Settings.RebalancePortfolioOnSecurityChanges = False

        # =====================================================================
        # 策略参数 — 通过 GetParameter 读取, 支持 lean optimize 调参
        # config.json 中以字符串形式定义默认值
        # =====================================================================
        MAX_POSITIONS = int(self.GetParameter("max-positions", 5))
        LEVERAGE = int(self.GetParameter("leverage", 2))
        self._max_positions = MAX_POSITIONS

        # ── Alpha 参数 ──
        insight_duration_hours = int(self.GetParameter("insight-duration-hours", 48))
        volatility_lookback_hours = int(
            self.GetParameter("volatility-lookback-hours", 168)
        )
        volatility_cache_hours = int(self.GetParameter("volatility-cache-hours", 6))
        selection_weight_power = float(self.GetParameter("selection-weight-power", 1.5))
        max_adverse_vol_ratio = float(self.GetParameter("max-adverse-vol-ratio", 3.0))
        co_lookback = int(self.GetParameter("co-lookback", 24))
        co_decay = float(self.GetParameter("co-decay", 0.95))

        # ── Risk 参数 ──
        flash_tp_threshold = float(self.GetParameter("flash-tp-threshold", 0.08))
        flash_tp_hours = float(self.GetParameter("flash-tp-hours", 4.0))
        trailing_default = float(self.GetParameter("trailing-default", 0.05))
        trailing_tight = float(self.GetParameter("trailing-tight", 0.02))
        tight_threshold = float(self.GetParameter("tight-threshold", 0.10))
        hard_stop = float(self.GetParameter("hard-stop", 0.10))
        portfolio_max_dd = float(self.GetParameter("portfolio-max-dd", 0.15))

        # ── Execution 参数 ──
        max_spread_percent = float(self.GetParameter("max-spread-percent", 0.005))
        max_wait_hours = float(self.GetParameter("max-wait-hours", 1.0))

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
                volatility_lookback_hours=volatility_lookback_hours,
                volatility_cache_hours=volatility_cache_hours,
                selection_weight_power=selection_weight_power,
                max_adverse_vol_ratio=max_adverse_vol_ratio,
                insight_duration_hours=insight_duration_hours,
                co_lookback=co_lookback,
                co_decay=co_decay,
            )
        )

        # 2. Portfolio Construction Model - Enter-and-Hold
        self.SetPortfolioConstruction(
            EnterAndHoldPCM(
                max_positions=MAX_POSITIONS,
                leverage=LEVERAGE,
            )
        )

        # 3. Risk Management Model - 动态出场风控
        self.AddRiskManagement(
            DynamicExitRiskModel(
                flash_tp_threshold=flash_tp_threshold,
                flash_tp_hours=flash_tp_hours,
                trailing_default=trailing_default,
                trailing_tight=trailing_tight,
                tight_threshold=tight_threshold,
                hard_stop=hard_stop,
            )
        )

        # 全局熔断
        self.AddRiskManagement(MaximumDrawdownPercentPortfolio(portfolio_max_dd))

        # 4. Execution Model - 智能价差执行
        self.SetExecution(
            SmartSpreadExecutionModel(
                max_spread_percent=max_spread_percent,
                max_wait_hours=max_wait_hours,
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
        self.Debug("Altcoin Short Strategy Initialized (Parameterized)")
        self.Debug(
            f"Positions: {MAX_POSITIONS} | Leverage: {LEVERAGE}x | Duration: {insight_duration_hours}h"
        )
        self.Debug(
            f"Alpha: vol_lookback={volatility_lookback_hours}h | weight_pow={selection_weight_power} | co_lookback={co_lookback}"
        )
        self.Debug(
            f"Risk: flash_tp={flash_tp_threshold:.0%} | trail={trailing_default:.0%}→{trailing_tight:.0%} | stop={hard_stop:.0%} | fuse={portfolio_max_dd:.0%}"
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
