# region imports
from AlgorithmImports import *
from sklearn.linear_model import HuberRegressor
import numpy as np
import pandas as pd
from datetime import timedelta,datetime
# endregion

# Top 30 Binance Futures by volume/liquidity
TICKERS = [
    "BTCUSDT",
    "ETHUSDT",
    # "BNBUSDT",
    # "SOLUSDT",
    # "XRPUSDT",
    # "DOGEUSDT",
    # "ADAUSDT",
    # "AVAXUSDT",
    # "DOTUSDT",
    # "LINKUSDT",
    # "MATICUSDT",
    # "LTCUSDT",
    # "UNIUSDT",
    # "ATOMUSDT",
    # "ETCUSDT",
    # "FILUSDT",
    # "APTUSDT",
    # "NEARUSDT",
    # "ARBUSDT",
    # "OPUSDT",
    # "INJUSDT",
    # "SUIUSDT",
    # "TIAUSDT",
    # "SEIUSDT",
    # "STXUSDT",
    # "IMXUSDT",
    # "RUNEUSDT",
    # "AAVEUSDT",
    # "MKRUSDT",
    # "LDOUSDT",
]

class SymbolData:
    def __init__(self, algo:QCAlgorithm, symbol):
        self.algo = algo
        self.symbol = symbol
        
        # --- 初始化指标 ---
        # 1. 布林带 (Close)
        self.bb = algo.bb(symbol, 20, 2, MovingAverageType.SIMPLE, Resolution.MINUTE)
        
        # 2. Close SMA
        self.sma_close = algo.sma(symbol, 20, Resolution.MINUTE)
        
        # 3. High SMA (手动注册)
        self.sma_high = SimpleMovingAverage(20)
        algo.register_indicator(symbol, self.sma_high, Resolution.MINUTE, Field.HIGH)
        
        # 4. Volume SMA (手动注册)
        self.sma_vol = SimpleMovingAverage(20)
        algo.register_indicator(symbol, self.sma_vol, Resolution.MINUTE, Field.VOLUME)
        
        # --- 初始化滚动窗口 (RollingWindow) ---
        # 记录 SMA High 的历史数据
        self.sma_high_window = RollingWindow[IndicatorDataPoint](5)
        self.sma_high.updated += lambda sender, updated: self.sma_high_window.add(updated)
        
        # # 3. 手动触发热身（解决 Forward Only 报错的最有效方法）
        # # 这会立即抓取历史数据填充指标，而不会等待回测数据流
        # res=Resolution.MINUTE
        # algo.warm_up_indicator(symbol, self.bb, res)
        # algo.warm_up_indicator(symbol, self.sma_close, res)
        # algo.warm_up_indicator(symbol, self.sma_high, res)
        # algo.warm_up_indicator(symbol, self.sma_vol, res)

    @property
    def is_ready(self):
        # 必须所有指标和窗口都填满数据
        return (self.bb.is_ready and 
                self.sma_close.is_ready and 
                self.sma_high.is_ready and 
                self.sma_vol.is_ready and 
                self.sma_high_window.is_ready)

class Aswinningrate(QCAlgorithm):

    def initialize(self):
        # self._can_trade = False
        self.settings.automatic_indicator_warm_up = False


        self.set_time_zone("UTC")
        self.set_start_date(datetime(year=2026,month=2,day=4,hour=12))
        self.set_end_date(datetime(year=2026,month=2,day=7,hour=0))
        
        self.set_cash("USDT", 100000)
        self.set_brokerage_model(BrokerageName.BINANCE_FUTURES, AccountType.MARGIN)
        
        # Create some indicators to detect trading opportunities.
        # self._macd=self.macd

            
        # Add all symbols
        self.symbols = []
        self.targets : dict[str,SymbolData]= {}
        self.consolidators_trade = dict()
        self.consolidators_quote = dict()
        for ticker in TICKERS:
            try:
                crypto_future = self.add_crypto_future(ticker, Resolution.MINUTE, Market.BINANCE)
                self.debug(f"added {crypto_future.symbol}")
                # crypto_future.sma_fast = self.sma(crypto_future.symbol, 5, Resolution.Minute)  # default close
                # crypto_future.sma_slow = self.sma(crypto_future.symbol, 30, Resolution.Minute)  # default close
                # crypto_future.close_window = RollingWindow[float](window_size)
                # crypto_future.high_window = RollingWindow[float](window_size)
                # crypto_future.low_window = RollingWindow[float](window_size)
                # crypto_future.volume_window = RollingWindow[float](window_size)
                # crypto_future.sma_fast.updated += lambda sender, updated: crypto_future.close_window.add(updated)
                # crypto_future.sma_slow.Updated += lambda sender, updated: crypto_future.high_window.add(updated)
                self.symbols.append(crypto_future.symbol)
                self.targets[crypto_future.symbol] = SymbolData(self, crypto_future.symbol)   
            except:
                self.debug(f"Failed to add {ticker}")
                
        # for symbol in self.symbols:
        #     try:
        #         # # Consolidate data into 60-minute bars.
        #         # consolidator = self.consolidate(symbol, timedelta(minutes=60), self._consolidation_handler) 
        #         # self.consolidators_trade[symbol] = consolidator               
        #         # Hook up the indicators to be updated with the 60-min bars.
        #         for indicator in [self._close_windows ,self._volume_windows ,self._high_windows ]:
        #             # self.register_indicator(symbol, indicator, consolidator)
        #             self.register_indicator(symbol, indicator)
        #     except:
        #         self.debug(f"Failed to create consolidator for {symbol}")
            


        self.debug(f"Added {len(self.symbols)} symbols")

        # warmup
        self.warm_up_period = timedelta(days=1)
        self.set_warm_up(self.warm_up_period)

        # # Schedule
        # self.train(self.train_model)
        # self.schedule.on(
        #     self.date_rules.every(DayOfWeek.Sunday),
        #     self.time_rules.at(0, 0),
        #     self.train_model,
        # )
        # self.schedule.on(
        #     self.date_rules.every_day(),
        #     self.time_rules.every(timedelta(hours=1)),
        #     self.run_prediction,
        # )
        # self.schedule.on(
        #     self.date_rules.every_day(),
        #     self.time_rules.every(interval=timedelta(hours=1)),
            
        # )

        # self.set_warm_up(timedelta(days=7))
        self.debug("Initialized")

    # def train_model(self):
    #     """Train on pooled data from ALL symbols."""
    #     all_features = []
    #     all_labels = []

    #     for sym in self.symbols:
    #         history = self.history(
    #             sym, timedelta(days=self.lookback_days), Resolution.Minute
    #         )
    #         if history.empty:
    #             continue

    #         try:
    #             df = history.loc[sym]
    #         except:
    #             continue

    #         hourly = df.resample("1H").last().dropna()
    #         df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    #         for ts in hourly.index:
    #             try:
    #                 w12h = df.loc[ts - timedelta(hours=12) : ts]
    #                 w1h = df.loc[ts - timedelta(hours=1) : ts]
    #                 w24h = df.loc[ts - timedelta(hours=24) : ts]

    #                 if len(w1h) < 30:
    #                     continue

    #                 price = w1h.iloc[-1]["close"]
    #                 sigma_1h = w1h["log_ret"].std()
    #                 if sigma_1h == 0:
    #                     continue

    #                 high_24h = w24h["high"].max()
    #                 feat_short_dist = (high_24h - price) / (price * sigma_1h)
    #                 feat_skew = w1h["log_ret"].skew()

    #                 rets = w12h["log_ret"]
    #                 s_down = rets[rets < 0].std()
    #                 s_up = rets[rets > 0].std()
    #                 s_down = 0 if pd.isna(s_down) else s_down
    #                 s_up = 0 if pd.isna(s_up) else s_up
    #                 feat_vol_ratio = s_down / (s_up + 1e-6)

    #                 vwap = (w12h["close"] * w12h["volume"]).sum() / (
    #                     w12h["volume"].sum() + 1e-6
    #                 )
    #                 feat_vwap = (price - vwap) / vwap

    #                 # Label
    #                 future_start = ts + timedelta(minutes=1)
    #                 future_end = ts + timedelta(hours=24)
    #                 if future_end > df.index[-1]:
    #                     continue
    #                 future = df.loc[future_start:future_end]
    #                 if future.empty:
    #                     continue

    #                 max_down = max(price - future["low"].min(), price * 0.001)
    #                 max_up = max(future["high"].max() - price, price * 0.001)
    #                 y = np.log(max_down / max_up)

    #                 row = np.nan_to_num(
    #                     [feat_short_dist, feat_skew, feat_vol_ratio, feat_vwap]
    #                 )
    #                 all_features.append(row)
    #                 all_labels.append(y)
    #             except:
    #                 continue

    #     if len(all_features) > 100:
    #         self.model.fit(all_features, all_labels)
    #         self.model_ready = True
    #         self.debug(
    #             f"Model Trained on {len(all_features)} samples. Coefs: {self.model.coef_}"
    #         )
    #     else:
    #         self.debug(f"Not enough samples: {len(all_features)}")

    def on_data(self, data: Slice):
        if self.is_warming_up:
            return

        # single symbol strategy
        for sym in self.symbols:
            if sym in self.targets and self.targets[sym].is_ready:
                if sym in data.bars:
                    # self.debug(f"on_data for {data.bars[sym]}")
                    # Scan for long trades.
                    open_price = data.bars[sym].open
                    close_price = data.bars[sym].close
                    high_price = data.bars[sym].high
                    low_price = data.bars[sym].low
                    volume = data.bars[sym].volume
                    
                    sma_close = self.targets[sym].sma_close.current.value
                    sma_high = self.targets[sym].sma_high.current.value
                    sma_vol = self.targets[sym].sma_vol.current.value
                    sma_high_window = self.targets[sym].sma_high_window[0].value
                    # self.debug(f" sma_close {sma_close } sma_high {sma_high } sma_vol {sma_vol } sma_high_window {sma_high_window }")
                    self.log(f" open_price {open_price } high_price {high_price } low_price {low_price } close_price {close_price } volume {volume }  sma_close {sma_close } sma_high {sma_high } sma_vol {sma_vol } sma_high_window {sma_high_window }")

                    # Example logic: if close SMA < high SMA and volume SMA increasing, go short
                    if sma_close < sma_high:
                        # self.set_holdings(sym, -0.05)  # Short 5% of portfolio
                        self.market_order(sym, -0.1)
                        self.log(f"Shorting {sym}")
                        
                        self.log(f"Portfolio {self.portfolio.positions.to_string()} Value: {self.portfolio.total_portfolio_value}, Cash: {self.portfolio.cash}")
                    # elif sma_close > sma_high and sma_vol > 0 and sma_high > sma_high_window:
                    #     self.set_holdings(sym, 0.05)  # Long 5% of portfolio
                    #     self.debug(f"Longing {sym}")
                    # else:
                    #     self.set_holdings(sym, 0)  # Exit position
                    #     self.debug(f"Exiting position for {sym}")
                else:
                    self.error(f"no bar data {sym}")
            else:
                self.error(f"data not ready {sym}")
                                    

    # def run_prediction(self):
    #     if not self.model_ready or self.is_warming_up:
    #         return

    #     predictions = {}

    #     for sym in self.symbols:
    #         if not self.close_windows[sym].is_ready:
    #             continue

    #         closes = np.array([x for x in self.close_windows[sym]])[::-1]
    #         volumes = np.array([x for x in self.volume_windows[sym]])[::-1]
    #         highs = np.array([x for x in self.high_windows[sym]])[::-1]

    #         if len(closes) < 720:
    #             continue

    #         w1h = closes[-60:]
    #         w12h = closes[-720:]
    #         w12h_vol = volumes[-720:]

    #         ret_1h = np.diff(np.log(w1h))
    #         ret_12h = np.diff(np.log(w12h))

    #         price = closes[-1]
    #         high_24h = np.max(highs)
    #         sigma_1h = np.std(ret_1h)
    #         if sigma_1h == 0:
    #             continue

    #         feat_short_dist = (high_24h - price) / (price * sigma_1h)
    #         mean_ret = np.mean(ret_1h)
    #         feat_skew = (
    #             np.sum((ret_1h - mean_ret) ** 3) / (len(ret_1h) * sigma_1h**3)
    #             if sigma_1h > 0
    #             else 0
    #         )

    #         down_rets = ret_12h[ret_12h < 0]
    #         up_rets = ret_12h[ret_12h > 0]
    #         s_down = np.std(down_rets) if len(down_rets) > 0 else 0
    #         s_up = np.std(up_rets) if len(up_rets) > 0 else 0
    #         feat_vol_ratio = s_down / (s_up + 1e-6)

    #         vwap = np.sum(w12h * w12h_vol) / (np.sum(w12h_vol) + 1e-6)
    #         feat_vwap = (price - vwap) / vwap

    #         row = np.nan_to_num([feat_short_dist, feat_skew, feat_vol_ratio, feat_vwap])
    #         y_pred = self.model.predict([row])[0]
    #         predictions[sym] = y_pred

    #     if len(predictions) < 10:
    #         return

    #     # Rank and select
    #     sorted_preds = sorted(predictions.items(), key=lambda x: x[1])
    #     bottom_15 = [
    #         s for s, _ in sorted_preds[: self.top_n]
    #     ]  # Most negative = LONG (expect up)
    #     top_15 = [
    #         s for s, _ in sorted_preds[-self.top_n :]
    #     ]  # Most positive = SHORT (expect down)

    #     # Rebalance
    #     for sym in self.symbols:
    #         if sym in top_15:
    #             target = -self.position_size_per_asset
    #         elif sym in bottom_15:
    #             target = self.position_size_per_asset
    #         else:
    #             target = 0

    #         current = (
    #             self.portfolio[sym].holdings_value / self.portfolio.total_portfolio_value
    #             if self.portfolio.total_portfolio_value > 0
    #             else 0
    #         )

    #         if abs(target - current) > 0.01:
    #             self.set_holdings(sym, target)

    #     self.debug(f"Rebalanced: Long {len(bottom_15)}, Short {len(top_15)}")

    def on_order_event(self, order_event):
        if order_event.status == OrderStatus.Filled:
            self.debug(
                f"Filled: {order_event.symbol} @ {order_event.fill_price} x {order_event.fill_quantity}"
            )
        elif order_event.status == OrderStatus.Invalid:
            self.debug(f"Rejected: {order_event.message}")
