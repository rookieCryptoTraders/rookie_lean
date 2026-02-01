from AlgorithmImports import *


class BinanceBot(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2023, 1, 1)  # Set Start Date
        self.set_cash(100000)  # Set Strategy Cash

        # Set Brokerage to Binance
        self.set_brokerage_model(BrokerageName.BINANCE, AccountType.CASH)

        # Add Crypto data from Binance
        self.add_crypto("BTCUSDT", Resolution.HOUR, Market.BINANCE)

        self.debug("Binance Bot Initialized")

    def on_data(self, data: Slice):
        if not self.portfolio.invested:
            if data.contains_key("BTCUSDT"):
                self.set_holdings("BTCUSDT", 0.5)
                self.debug(f"Purchased BTC at {self.time}")
