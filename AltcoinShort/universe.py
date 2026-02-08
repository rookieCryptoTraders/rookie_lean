# region imports
from AlgorithmImports import *
# endregion


class AltcoinFuturesUniverseSelectionModel(ManualUniverseSelectionModel):
    """
    Altcoin Futures 手动选币模型

    选择前 15-300 市值的代币作为基础证券池。
    注意：具体的过滤（排除 RWA/AI 等）由 Alpha Model 处理。
    """

    def __init__(self):
        tickers = ["BTCUSDT"]

        symbols = []
        for ticker in tickers:
            # 关键：创建 CryptoFuture 类型的 Symbol
            # 必须指定 Market.Binance
            symbol = Symbol.Create(ticker, SecurityType.CryptoFuture, Market.Binance)
            symbols.append(symbol)
            # Debug symbol creation
            print(
                f"DEBUG: Created Symbol: {symbol} | Type: {symbol.SecurityType} | ID: {symbol.ID}"
            )

        super().__init__(symbols)

    def OnSecuritiesChanged(
        self, algorithm: QCAlgorithm, changes: SecurityChanges
    ) -> None:
        """
        当证券变更时触发（手动模型只会在初始化时触发一次添加）
        可以在这里设置证券的特定属性，如 Leverage
        但推荐使用 Algorithm.UniverseSettings 统一设置
        """
        super().OnSecuritiesChanged(algorithm, changes)

        for security in changes.AddedSecurities:
            # 确保 Leverage 设置正确
            security.SetLeverage(2)
