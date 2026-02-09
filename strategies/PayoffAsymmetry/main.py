# region imports
from AlgorithmImports import *
from sklearn.linear_model import HuberRegressor
from scipy import stats
import numpy as np
import pandas as pd
# endregion

# Top 30 Binance Futures by volume/liquidity
TICKERS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "MATICUSDT",
    "LTCUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "ETCUSDT",
    "FILUSDT",
    "APTUSDT",
    "NEARUSDT",
    "ARBUSDT",
    "OPUSDT",
    "INJUSDT",
    "SUIUSDT",
    "TIAUSDT",
    "SEIUSDT",
    "STXUSDT",
    "IMXUSDT",
    "RUNEUSDT",
    "AAVEUSDT",
    "MKRUSDT",
    "LDOUSDT",
]


class CrossSectionPayoffAsymmetry(QCAlgorithm):
    def Initialize(self):
        # Using 2026 Jan for OOT backtest
        self.SetStartDate(2026, 1, 1)
        self.SetEndDate(2026, 1, 31)

        self.SetCash("USDT", 200000)
        self.SetBrokerageModel(BrokerageName.BinanceFutures, AccountType.Margin)

        # Assets
        self.symbols = []
        for ticker in TICKERS:
            try:
                crypto = self.AddCryptoFuture(ticker, Resolution.Minute, Market.Binance)
                self.symbols.append(crypto.Symbol)
                self.Debug(f"Added {ticker}")
            except Exception as e:
                self.Debug(f"Failed to add {ticker}: {e}")

        self.top_n = 15
        self.position_size_per_asset = 0.05
        self.entry_threshold = 0.1  # Realized Payoff (Y) must be predicted > 0.1

        # Model and State
        self.model = HuberRegressor(epsilon=1.35)
        self.model_ready = False

        # Windows
        self.close_windows = {}
        self.volume_windows = {}
        self.high_windows = {}
        window_size = 24 * 60 + 100

        for sym in self.symbols:
            self.close_windows[sym] = RollingWindow[float](window_size)
            self.volume_windows[sym] = RollingWindow[float](window_size)
            self.high_windows[sym] = RollingWindow[float](window_size)

        # Scheduling
        self.Train(self.TrainModel)
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Sunday),
            self.TimeRules.At(0, 0),
            self.TrainModel,
        )
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(TimeSpan.FromHours(4)),
            self.RunPrediction,
        )

        self.SetWarmUp(timedelta(days=7))
        self.Debug("Short-Only PayoffAsymmetry Strategy Initialized")

    def TrainModel(self):
        """Train on pooled data from ALL symbols using the 6 optimized features."""
        all_features = []
        all_labels = []

        for sym in self.symbols:
            history = self.History(
                sym, timedelta(days=20), Resolution.Minute
            )  # Train on recent slice for speed in backtest
            if history.empty:
                continue

            try:
                df = history.loc[sym]
                df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
                hourly = df.resample("1H").last().dropna()

                for ts in hourly.index:
                    w12h = df.loc[ts - timedelta(hours=12) : ts]
                    w1h = df.loc[ts - timedelta(hours=1) : ts]
                    w24h = df.loc[ts - timedelta(hours=24) : ts]

                    if len(w1h) < 30 or len(w12h) < 100:
                        continue

                    price = w1h.iloc[-1]["close"]
                    sigma_1h = w1h["log_ret"].std() + 1e-6

                    # 6 Features
                    f1 = (w24h["high"].max() - price) / (price * sigma_1h)
                    f2 = w1h["log_ret"].skew()

                    r12 = w12h["log_ret"]
                    s_down = r12[r12 < 0].std()
                    s_up = r12[r12 > 0].std()
                    f3 = (s_down or 0) / ((s_up or 0) + 1e-6)

                    vwap_12h = (w12h["close"] * w12h["volume"]).sum() / (
                        w12h["volume"].sum() + 1e-6
                    )
                    sigma_12h = r12.std() + 1e-6
                    f4 = (price - vwap_12h) / (price * sigma_12h)

                    avg_vol_24h = w24h["volume"].mean() + 1e-6
                    f5 = w1h["volume"].mean() / avg_vol_24h

                    ret_12h_total = (price - w12h.iloc[0]["close"]) / w12h.iloc[0][
                        "close"
                    ]
                    f6 = ret_12h_total / sigma_12h

                    # Label
                    future_end = ts + timedelta(hours=12)
                    if future_end > df.index[-1]:
                        continue
                    future = df.loc[ts + timedelta(minutes=1) : future_end]
                    if future.empty:
                        continue

                    max_down = max(price - future["low"].min(), price * 0.001)
                    max_up = max(future["high"].max() - price, price * 0.001)
                    y = np.log(max_down / max_up)

                    all_features.append(np.nan_to_num([f1, f2, f3, f4, f5, f6]))
                    all_labels.append(y)
            except:
                continue

        if len(all_features) > 500:
            self.model.fit(all_features, all_labels)
            self.model_ready = True
            self.Debug(f"Model Trained on {len(all_features)} samples.")
        else:
            self.Debug(f"Insufficient training samples: {len(all_features)}")

    def OnData(self, data: Slice):
        for sym in self.symbols:
            if sym in data.Bars:
                bar = data.Bars[sym]
                self.close_windows[sym].Add(float(bar.Close))
                self.volume_windows[sym].Add(float(bar.Volume))
                self.high_windows[sym].Add(float(bar.High))

    def RunPrediction(self):
        if not self.model_ready or self.IsWarmingUp:
            return

        predictions = {}

        for sym in self.symbols:
            if not self.close_windows[sym].IsReady:
                continue

            closes = np.array([x for x in self.close_windows[sym]])[::-1]
            volumes = np.array([x for x in self.volume_windows[sym]])[::-1]
            highs = np.array([x for x in self.high_windows[sym]])[::-1]

            w1h, w12h = closes[-60:], closes[-720:]
            w12h_vol = volumes[-720:]
            w24h_high = np.max(highs[-1440:])

            ret_1h = np.diff(np.log(w1h))
            sigma_1h = np.std(ret_1h) + 1e-6
            price = closes[-1]

            f1 = (w24h_high - price) / (price * sigma_1h)
            f2 = stats.skew(ret_1h) if len(ret_1h) > 30 else 0

            ret_12h = np.diff(np.log(w12h))
            s_down = (
                np.std(ret_12h[ret_12h < 0]) if len(ret_12h[ret_12h < 0]) > 0 else 0
            )
            s_up = np.std(ret_12h[ret_12h > 0]) if len(ret_12h[ret_12h > 0]) > 0 else 0
            f3 = s_down / (s_up + 1e-6)

            vwap_12h = np.sum(w12h * w12h_vol) / (np.sum(w12h_vol) + 1e-6)
            sigma_12h = np.std(ret_12h) + 1e-6
            f4 = (price - vwap_12h) / (price * sigma_12h)

            f5 = np.mean(volumes[-60:]) / (np.mean(volumes[-1440:]) + 1e-6)

            ret_12h_total = (price - w12h[0]) / w12h[0]
            f6 = ret_12h_total / sigma_12h

            row = np.nan_to_num([f1, f2, f3, f4, f5, f6])
            predictions[sym] = self.model.predict([row])[0]

        if len(predictions) < 10:
            return

        # --- SHORT-ONLY REBALANCING ---
        # Sort by predicted Y (highest Y = highest downside payoff)
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        # Filter: Must be in Top N AND above the entry threshold
        candidates = [
            s for s, y in sorted_preds[: self.top_n] if y >= self.entry_threshold
        ]

        targets = []
        for sym in self.symbols:
            if sym in candidates:
                targets.append(PortfolioTarget(sym, -self.position_size_per_asset))
            else:
                targets.append(PortfolioTarget(sym, 0))

        self.SetHoldings(targets)
        self.Debug(
            f"Short-Only Rebalance: {len(candidates)} assets met threshold {self.entry_threshold}."
        )

    def OnOrderEvent(self, orderEvent):
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Filled: {orderEvent.Symbol} x {orderEvent.FillQuantity}")
