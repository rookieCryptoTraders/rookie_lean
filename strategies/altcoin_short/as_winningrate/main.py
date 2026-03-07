from AlgorithmImports import *
from datetime import timedelta, datetime
import logging

from utils import CryptoFutureDepthData, CryptoFutureQuoteData
from alpha import AltcoinShortAlphaModel
import time as ttime
import os

from config import ASSET_CLASS, BASE_DATA_PATH, EXCHANGE
from config import DATA_LOCATION, PROXIES, _CONFIG_DIR, _PROJECT_ROOT

# datetime set time zone to UTC
os.environ["TZ"] = "UTC"
ttime.tzset()


class MyMarginInterestRateModel:
    def __init__(self, algorithm: QCAlgorithm, symbol: Symbol):
        self.algorithm = algorithm
        self.symbol = symbol

    def apply_margin_interest_rate(
        self, margin_interest_rate_parameters: MarginInterestRateParameters
    ) -> None:
        holdings = margin_interest_rate_parameters.security.holdings
        position_value = holdings.get_quantity_value(holdings.quantity)
        position_value.cash.add_amount(-1)


class AltcoinShortWinningRateAlgorithm(QCAlgorithm):
    """
    Altcoin Short Strategy Winning Rate - on_data 驱动实现（不使用 Algorithm Framework）

    思路（参考备份版本与官方 BasicTemplateCryptoAlgorithm 模式）：
    - initialize 中完成合约订阅与指标初始化
    - on_data 中基于简单趋势信号（EMA 快慢线）做空 Altcoin
    - 不加载深度数据，仅使用行情 K 线（分钟）
    """

    def initialize(self) -> None:
        log_level = (
            (self.get_parameter("log-level", "info") or "info").strip().upper()
        )
        if log_level in ("DEBUG", "TRACE", "VERBOSE", "1", "0"):
            logging.getLogger().setLevel(logging.DEBUG)
            for name in ("__main__", "utils", "alpha"):
                logging.getLogger(name).setLevel(logging.DEBUG)

        self.counter = 0
        self.set_time_zone("UTC")
        self.set_start_date(2026, 1, 15)
        self.set_end_date(2026, 1, 15)
        self.set_account_currency("USDT")
        self.set_cash(100000)  # 100k USDT

        # Configuration
        self.symbols = []
        self.depth_symbols = {}
        self.quote_symbols = {}
        self.regime_model = None  # Placeholder if you add regime detection later
        self.alpha_model = None

        # Binance 永续合约
        self.set_brokerage_model(BrokerageName.BINANCE_FUTURES, AccountType.MARGIN)

        # 策略参数（支持 config.json 中通过 get_parameter 调参）
        max_positions = int(self.get_parameter("max-positions", 5))
        leverage = int(self.get_parameter("leverage", 2))
        cooldown_minutes = float(self.get_parameter("cooldown-minutes", 60.0))

        self._max_positions = max_positions
        self._position_size = 1.0 / max_positions
        self._cooldown_minutes = cooldown_minutes

        # 订阅合约并为每个交易标的创建简单指标容器
        self.symbols: list[Symbol] = []
        self.depth_symbols: dict[Symbol, Symbol] = {}
        self.quote_symbols: dict[
            Symbol, Symbol
        ] = {}  # L1 quote (top-of-book) for backtest/execution

        # Alpha model: all feature engineering lives inside AltcoinShortAlphaModel
        self.alpha_model = AltcoinShortAlphaModel(
            max_positions=max_positions,
        )
        # Register AlphaModel with the Algorithm Framework so LEAN
        # drives its update() calls and manages Insight lifecycle.
        self.add_alpha(self.alpha_model)

        # Altcoin 列表（BTC 仅作为基准，不参与交易）
        tickers = [
            "BTCUSDT",  # debug: only BTCUSDT to validate depth custom data
        ]
        added_count = 0
        alpha_symbols_to_register = []  # List to hold all symbols (base, depth, quote) for alpha model
        for ticker in tickers:
            try:
                # 1. Base Security (CryptoFuture)
                crypto_future = self.add_crypto_future(
                    ticker,
                    resolution=Resolution.MINUTE,
                    market=Market.BINANCE,
                    fill_forward=True,
                    leverage=leverage,
                )
                # 设置一个固定的年化利率模型
                symbol = crypto_future.symbol
                crypto_future.set_margin_interest_rate_model(
                    MyMarginInterestRateModel(self, symbol)
                )
                self.symbols.append(symbol)
                alpha_symbols_to_register.append(symbol)

                # 2. Custom Depth Data (Ticker suffix prevents Symbol collision)
                depth_ticker_str = f"{ticker}.DEPTH"
                depth_security = self.add_data(
                    CryptoFutureDepthData, depth_ticker_str, Resolution.MINUTE
                )
                self.depth_symbols[symbol] = depth_security.symbol
                alpha_symbols_to_register.append(depth_security.symbol)

                # 3. Custom Quote Data (Optional, but let's keep it for completeness)
                quote_ticker_str = f"{ticker}.QUOTE"
                quote_security = self.add_data(
                    CryptoFutureQuoteData, quote_ticker_str, Resolution.MINUTE
                )
                self.quote_symbols[symbol] = quote_security.symbol
                alpha_symbols_to_register.append(quote_security.symbol)

                added_count += 1
            except Exception as e:
                self.debug(f"[Init] Failed to add {ticker}: {e}")

        # Benchmark: use lambda over our subscribed BTCUSDT so series align with performance
        # (avoids StatisticsBuilder "1 misaligned values" from separate Hour benchmark feed)
        btc_sym = next(
            (s for s in self.symbols if s.value and str(s.value).upper() == "BTCUSDT"),
            self.symbols[0] if self.symbols else None,
        )
        if btc_sym is not None:

            def _benchmark_value(dt: datetime) -> float:
                if btc_sym in self.securities and self.securities[btc_sym].price:
                    return float(self.securities[btc_sym].price)
                return 1.0

            self.set_benchmark(_benchmark_value)
        else:
            self.set_benchmark(lambda dt: 1.0)

        # 简单 warm-up，确保 EMA 等指标准备就绪
        self.set_warm_up(timedelta(days=1))
        self._last_trade_time: dict[Symbol, datetime] = {}
        # L1 quote cache (bid, ask) from CryptoFutureQuoteData for backtest when Security.BidPrice/AskPrice are 0
        self._last_quote: dict[Symbol, tuple[float, float]] = {}

        # 定时状态日志（每小时一次）
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(timedelta(hours=1)),
            self._log_status,
        )

        self.debug(
            f"Altcoin Short initialized | Positions: {max_positions} | Leverage: {leverage}x"
        )

    def _log_status(self) -> None:
        """定时状态日志"""
        holdings = [h for h in self.portfolio.values() if h.invested]

        total_pnl = sum(h.unrealized_profit for h in holdings)
        margin_used = sum(abs(h.holdings_value) for h in holdings)

        self.debug(
            f"[Status] Positions: {len(holdings)}/{self._max_positions} | "
            f"PnL: ${total_pnl:.2f} | "
            f"Margin: ${margin_used:.2f} | "
            f"Equity: ${self.portfolio.total_portfolio_value:.2f}"
        )

    def on_data(self, data: Slice) -> None:
        """
        主交易逻辑：
        - 将行情切片传递给 AlphaModel 生成 Insights
        - on_data 只负责根据 Insights 执行持仓调整，不再做复杂特征计算
        """
        if self.is_warming_up:
            return

        self.counter += 1
        now = self.time.replace(second=0, microsecond=0)

        # Update L1 quote cache from custom quote data (for execution spread check in backtest)
        quote_dict = data.get(CryptoFutureQuoteData)
        if quote_dict and self.quote_symbols:
            for sym, qsym in self.quote_symbols.items():
                bar = quote_dict.get(qsym)
                if bar is not None and hasattr(bar, "bid") and hasattr(bar, "ask"):
                    self._last_quote[sym] = (float(bar.bid), float(bar.ask))

        # 由 AlphaModel 生成做空信号（InsightDirection.Down），
        # AlphaModel 由框架通过 add_alpha 自动调用 update()，
        # 这里只读取最新的 insights，不再手动触发 update。
        insights = list(self.alpha_model.latest_insights)
        if not insights:
            return

        # 只保留做空类 insight，按权重排序，限制最大标的数量
        short_insights = [i for i in insights if i.direction == InsightDirection.DOWN]
        # if not short_insights:
        #     return

        short_insights.sort(key=lambda x: abs(x.weight or 0.0), reverse=True)
        desired_symbols = {i.symbol for i in short_insights[: self._max_positions]}

        # DEBUG: Trace the execution flow and the depth signals in the active insights
        for ins in short_insights[: self._max_positions]:
            depth_info = [
                part for part in ins.Tag.split("|") if "obi" in part or "mp_div" in part
            ]
        #  self.log(f"[Main-Depth-Flow] Processing Insight for {ins.symbol.value} | Weight: {ins.weight:.3f} | DepthSignals: {depth_info}")

        # self.debug(
        #     f"[OnData] insights={len(insights)} short_insights={len(short_insights)} "
        #     f"desired={len(desired_symbols)} | "
        #     f"symbols={[s.value for s in list(desired_symbols)[:5]]}"
        # )

        # 当前已持有的空头（用于限制最大持仓数 & 管理平仓）
        active_shorts = {
            h.symbol for h in self.portfolio.values() if h.invested and h.quantity < 0
        }

        # 平掉不再在候选集合中的空头
        for symbol in list(active_shorts):
            if symbol not in desired_symbols:
                last = self._last_trade_time.get(symbol)
                if (
                    last is not None
                    and (now - last).total_seconds() < self._cooldown_minutes * 60
                ):
                    continue

                self.set_holdings(symbol, 0.0)
                self._last_trade_time[symbol] = now

        # 为候选集合中标的建立 / 维持均匀仓位空头
        for symbol in desired_symbols:
            last = self._last_trade_time.get(symbol)
            if (
                last is not None
                and (now - last).total_seconds() < self._cooldown_minutes * 60
            ):
                continue

            target_weight = -self._position_size
            current_weight = 0.0
            total_value = self.portfolio.total_portfolio_value
            if total_value > 0:
                current_weight = self.portfolio[symbol].holdings_value / total_value

            if abs(target_weight - current_weight) < 0.01:
                continue

            self.set_holdings(symbol, target_weight)
            self._last_trade_time[symbol] = now

    def on_order_event(self, order_event: OrderEvent) -> None:
        """订单事件回调"""
        if order_event.status == OrderStatus.FILLED:
            direction = "SHORT" if order_event.fill_quantity < 0 else "COVER"
            self.debug(
                f"[Order] {direction} {order_event.symbol.value}: "
                f"Qty={order_event.fill_quantity:.4f} @ ${order_event.fill_price:.4f}"
            )
        elif order_event.status == OrderStatus.INVALID:
            self.debug(f"[Order] Rejected: {order_event.message}")

    def on_end_of_algorithm(self) -> None:
        """策略结束回调"""
        self.debug(
            f"Altcoin Short completed | Final value: ${self.portfolio.total_portfolio_value:,.2f}"
        )
