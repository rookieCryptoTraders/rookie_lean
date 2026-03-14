# Binance datafeed

Unified fetch from **Binance Vision** (data.binance.vision) into the project data folder. All timestamps are **Unix milliseconds (UTC)**. Data is stored under the LEAN folder structure so backtests can resolve paths via `lean.json` → `"data-folder"` (e.g. `data`).

---

## 1. Unified entry

**Script:** `run_binance_fetch.py`

**Usage:**
```bash
python -m qc_datafeed.binance.run_binance_fetch [asset_class] [data_type] [resolution] [--start YYYYMMDD] [--end YYYYMMDD] [--redownload] [--margin-type um|cm] [--timestamp-format unix_ms|ms_midnight|datetime_string]
```

---

## 2. Allowed parameters

| Parameter      | Allowed values | Default   | Notes |
|----------------|----------------|-----------|--------|
| **asset_class** | `cryptofuture`, `spot` | `cryptofuture` | Futures (USD-M/COIN-M) vs spot. |
| **data_type**   | See below      | `klines`  | Per-asset list in next section. |
| **resolution**  | `minute`, `hour`, `daily` | `hour` | Used only for klines and *Klines types; ignored for aggTrades, bookDepth, metrics. |
| **--start**     | `YYYYMMDD` or `YYYY-MM-DD` | `2026-01-01` | Start date (inclusive). |
| **--end**       | `YYYYMMDD` or `YYYY-MM-DD` | `2026-01-31` | End date (inclusive). |
| **--redownload** | flag         | off       | Overwrite existing ZIPs in range. |
| **--timestamp-format** | `unix_ms`, `ms_midnight`, `datetime_string` | `unix_ms` | How timestamps are stored in output CSVs (see below). |
| **--margin-type** | `um`, `cm`  | `um`      | Futures only: USD-M (`um`) or COIN-M (`cm`). |

### Timestamp format

- **unix_ms** (default): Unix time in milliseconds (UTC).
- **ms_midnight**: Milliseconds since midnight UTC (same calendar day).
- **datetime_string**: UTC string like `YYYY-MM-DD HH:MM:SS.mmm`.

### data_type by asset_class

- **cryptofuture:** `klines`, `aggTrades`, `bookDepth`, `indexPriceKlines`, `markPriceKlines`, `premiumIndexKlines`, `metrics`
- **spot:** `klines`, `aggTrades`

---

## 3. Data types and Binance Vision paths

| data_type           | Binance Vision path (futures um / spot) | resolution |
|---------------------|------------------------------------------|------------|
| klines              | `futures/um/daily/klines/<symbol>/<1m\|1h\|1d>/` or `spot/daily/klines/...` | 1m, 1h, 1d |
| aggTrades           | `futures/um/daily/aggTrades/<symbol>/` or `spot/daily/aggTrades/...` | N/A (daily) |
| bookDepth           | `futures/um/daily/bookDepth/<symbol>/`    | N/A (daily) |
| indexPriceKlines    | `futures/um/daily/indexPriceKlines/<symbol>/<1m\|1h\|1d>/` | 1m, 1h, 1d |
| markPriceKlines     | `futures/um/daily/markPriceKlines/<symbol>/<1m\|1h\|1d>/`  | 1m, 1h, 1d |
| premiumIndexKlines  | `futures/um/daily/premiumIndexKlines/<symbol>/<1m\|1h\|1d>/` | 1m, 1h, 1d |
| metrics             | `futures/um/daily/metrics/<symbol>/`     | N/A (daily) |

---

## 4. Output paths (LEAN structure)

All under `data/` (or `DATA_ROOT` from env `DATA_RELATIVE_LOCATION`). Symbol is lowercase (e.g. `btcusdt`).

| data_type           | Output path pattern | File pattern |
|---------------------|---------------------|--------------|
| klines (futures)    | `data/cryptofuture/binance/<minute\|hour\|daily>/<symbol>/` | `YYYYMMDD_trade.zip` → `YYYYMMDD_trade.csv` |
| klines (spot)       | `data/spot/binance/<minute\|hour\|daily>/<symbol>/` | `YYYYMMDD_trade.zip` → `YYYYMMDD_trade.csv` |
| aggTrades (futures) | `data/cryptofuture/binance/aggtrades/<symbol>/` | `YYYYMMDD_aggtrades.zip` → `YYYYMMDD_aggtrades.csv` |
| aggTrades (spot)    | `data/spot/binance/aggtrades/<symbol>/` | `YYYYMMDD_aggtrades.zip` → `YYYYMMDD_aggtrades.csv` |
| bookDepth           | `data/cryptofuture/binance/depth/<symbol>/` | `YYYYMMDD_depth.zip` → `YYYYMMDD_depth.csv` |
| indexPriceKlines    | `data/cryptofuture/binance/indexPriceKlines/<minute\|hour\|daily>/<symbol>/` | `YYYYMMDD.zip` → `YYYYMMDD.csv` |
| markPriceKlines    | `data/cryptofuture/binance/markPriceKlines/<minute\|hour\|daily>/<symbol>/` | `YYYYMMDD.zip` → `YYYYMMDD.csv` |
| premiumIndexKlines  | `data/cryptofuture/binance/premiumIndexKlines/<minute\|hour\|daily>/<symbol>/` | `YYYYMMDD.zip` → `YYYYMMDD.csv` |
| metrics             | `data/cryptofuture/binance/metrics/<symbol>/` | `YYYYMMDD_metrics.zip` → `YYYYMMDD_metrics.csv` |

---

## 5. Data columns and meanings

### 5.1 klines (trade bars)

- **Path:** `data/<asset_class>/binance/<resolution>/<symbol>/YYYYMMDD_trade.zip`
- **CSV:** No header. Columns in order:

| Column | Meaning | Unit |
|--------|---------|------|
| time   | Candle open time (Unix ms, UTC) | milliseconds |
| open   | Open price | quote asset |
| high   | High price | quote asset |
| low    | Low price  | quote asset |
| close  | Close price | quote asset |
| volume | Volume (base asset) | base asset |

### 5.2 aggTrades

- **Path:** `data/<asset_class>/binance/aggtrades/<symbol>/YYYYMMDD_aggtrades.zip`
- **CSV:** No header. Columns in order (Binance aggTrades):

| Column         | Meaning | Unit |
|----------------|---------|------|
| agg_trade_id   | Aggregate trade ID | - |
| price          | Trade price | quote asset |
| quantity       | Trade quantity | base asset |
| first_trade_id | First trade ID in aggregate | - |
| last_trade_id  | Last trade ID in aggregate | - |
| transact_time  | Trade time (Unix ms, UTC) | milliseconds |
| is_buyer_maker | True if maker was seller (buyer was taker) | boolean (0/1) |

### 5.3 bookDepth

- **Path:** `data/cryptofuture/binance/depth/<symbol>/YYYYMMDD_depth.zip`
- **Format:** Same as `ccxt_data_fetch` `fetch_depth_range_cryptofuture` output. CSV: no header. Columns:

| Column     | Meaning | Unit |
|------------|---------|------|
| ms_midnight | Milliseconds since midnight UTC for the snapshot | ms |
| percentage | Depth level (e.g. -5, -4, …, 0, …, 5) | % |
| depth     | Depth at that level | - |
| notional  | Notional at that level | quote asset |

### 5.4 indexPriceKlines / markPriceKlines / premiumIndexKlines

- **Path:** `data/cryptofuture/binance/<data_type>/<resolution>/<symbol>/YYYYMMDD.zip`
- **CSV:** Raw Binance Vision format (typically open_time, open, high, low, close, …). Timestamps in Unix ms, UTC.

### 5.5 metrics

- **Path:** `data/cryptofuture/binance/metrics/<symbol>/YYYYMMDD_metrics.zip`
- **CSV:** Raw Binance Vision daily metrics (OI, long/short ratio, taker buy/sell, etc.). Timestamps in Unix ms or date string, UTC.

---

## 6. Examples

```bash
# Futures hour klines, Jan 2026 (timestamps in Unix ms)
python -m qc_datafeed.binance.run_binance_fetch cryptofuture klines hour --start 20260101 --end 20260201

# Futures aggTrades, force redownload
python -m qc_datafeed.binance.run_binance_fetch cryptofuture aggTrades daily --start 20260101 --redownload

# Futures bookDepth (depth snapshots)
python -m qc_datafeed.binance.run_binance_fetch cryptofuture bookDepth minute --start 20260101

# Klines with ms_since_midnight or datetime string
python -m qc_datafeed.binance.run_binance_fetch cryptofuture klines hour --timestamp-format ms_midnight --start 20260101
python -m qc_datafeed.binance.run_binance_fetch cryptofuture klines hour --timestamp-format datetime_string --start 20260101

# Futures index/mark/premium klines
python -m qc_datafeed.binance.run_binance_fetch cryptofuture indexPriceKlines hour --start 20260101
python -m qc_datafeed.binance.run_binance_fetch cryptofuture markPriceKlines hour --start 20260101
python -m qc_datafeed.binance.run_binance_fetch cryptofuture premiumIndexKlines hour --start 20260101

# Futures daily metrics
python -m qc_datafeed.binance.run_binance_fetch cryptofuture metrics daily --start 20260101

# Spot klines
python -m qc_datafeed.binance.run_binance_fetch spot klines daily --start 20260101
```

---

## 7. Symbols and config

- Symbols are taken from `qc_datafeed.binance.config.DEFAULT_SYMBOLS` or, if `ccxt_data_fetch` is available, from `get_top_200_symbols(asset_class)` capped by `TOP_N_SYMBOL`.
- Data root is set by `DATA_RELATIVE_LOCATION` (default `data` relative to project root).
- Optional proxy: set `PROXY` (e.g. `http://127.0.0.1:1082`) and `USE_PROXY=true` in env.
