# data format

## cryptofuture

在 QuantConnect 的 LEAN 引擎中，`CryptoFuture`（加密货币期货）的数据格式主要遵循 LEAN 的标准数据存储结构。根据数据分辨率（Resolution）的不同，其文件组织和 CSV 列格式会有所区别。

以下是详细的数据格式说明：

---

### 1. 目录结构

数据通常存储在 `data/cryptofuture/` 目录下，层级如下：

> `data/cryptofuture/<market>/<resolution>/<symbol>/<date>_<type>.zip`

* **market**: 交易所名称（如 `binance`）。
* **resolution**: 分辨率（`tick`, `second`, `minute`, `hour`, `daily`）。
* **symbol**: 交易对名称（如 `btcusdt`）。
* **date**: 对应日期（YYYYMMDD）。
* **type**: 数据类型（`trade` 代表成交，`quote` 代表报价）。

---

### 2. 具体文件格式 (CSV 内部列)

#### **A. 分钟线与秒线 (Minute / Second)**

这些分辨率的文件通常按天打包成 ZIP。CSV 文件内部不包含表头，列顺序如下：
| 列索引 | 字段名称 | 说明 |
| :--- | :--- | :--- |
| 0 | **Time** | 距离当日午夜的毫秒数 (Milliseconds since midnight) |
| 1 | **Open** | 开盘价 (乘以  或以十进制表示，取决于版本) |
| 2 | **High** | 最高价 |
| 3 | **Low** | 最低价 |
| 4 | **Close** | 收盘价 |
| 5 | **Volume** | 成交量 |

#### **B. 小时线与日线 (Hour / Daily)**

这些长周期数据通常一个文件包含所有历史记录。
| 列索引 | 字段名称 | 说明 |
| :--- | :--- | :--- |
| 0 | **Time** | 格式为 `yyyyMMdd HH:mm` |
| 1 | **Open** | 开盘价 |
| 2 | **High** | 最高价 |
| 3 | **Low** | 最低价 |
| 4 | **Close** | 收盘价 |
| 5 | **Volume** | 成交量 |

#### **C. Tick 数据 (Tick)**

Tick 数据分为 `trade` 和 `quote` 两种文件。

* **Trade Tick**:
`Time (ms), Price, Quantity, Exchange, Sale Condition, Suspicious`
* **Quote Tick** (买卖盘报价):
`Time (ms), BidPrice, BidSize, AskPrice, AskSize, Exchange, Suspicious`

---

### 3. 特殊字段：保证金率 (Margin Interest Rate)

由于加密期货涉及持仓费用，QuantConnect 还有专门的 `margin_rate` 数据集：

* **路径**: `data/cryptofuture/binance/margin_rate/`
* **格式**: 通常包含 `Time` 和 `Interest Rate`（万分之几或百分比，依交易所而定）。

---

### 4. 编程中的对象表达

在 Python 或 C# 代码中访问时，你会得到以下对象：

* **TradeBar**: 包含 `Open`, `High`, `Low`, `Close`, `Volume`。
* **QuoteBar**: 包含 `Bid` 和 `Ask` 两个子 Bar（各自有 OHLC 结构）。
* **CryptoFuture**: 这是一个特殊的 Security 类型，比普通 Crypto 多了 `SymbolProperties`（如乘数、最小交易量）和 `MarginModel`。

> **注意：** QuantConnect 官方数据通常会对价格进行缩放（除以  或 ）以存储为整数节省空间，但在加载进 LEAN 引擎后，你会看到的都是正常的浮点数价格。

您是需要转换现有数据到这种格式，还是在编写代码时无法正确读取数据？如果是后者，我可以为您提供一个 `AddCryptoFuture` 的代码范例。


#   How to Use

  You can now run the fetcher with various configurations:

  1. Futures Data (Default)

```python
# Minute Resolution
python -m ccxt_data_fetch.run cryptofuture minute

# Hourly Resolution
python -m ccxt_data_fetch.run cryptofuture hour

# Daily Resolution
python -m ccxt_data_fetch.run cryptofuture daily

# Margin Interest (Funding Rates)
python -m ccxt_data_fetch.run cryptofuture margin_interest
```

  2. Spot Data

```python
# Minute Resolution
python -m ccxt_data_fetch.run crypto minute

# Hourly Resolution
python -m ccxt_data_fetch.run crypto hour

# Daily Resolution
python -m ccxt_data_fetch.run crypto daily
```


  3. Specifying Tick Type (e.g., for folder structure compliance)

```python
# Saves as ..._quote.zip (Note: Data is still OHLCV trades)
python -m ccxt_data_fetch.run cryptofuture minute quote
```
