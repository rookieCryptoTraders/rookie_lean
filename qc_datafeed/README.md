# qc_datafeed

Data feed scripts for the LEAN engine. Each subfolder is organized **by data source** (e.g. `binance`, `polymarket`). Scripts download or produce data that lives under the project **data folder** and follows the **LEAN folder structure** so backtests and live runs can resolve paths correctly.

---

## 1. Folder organization

- **One folder per data source**  
  - `qc_datafeed/binance/` – Binance (spot, USD‑M futures, etc.)  
  - `qc_datafeed/polymarket/` – Polymarket CLOB, etc.  
  - Add `qc_datafeed/<source>/` for each new provider.

- **Inside each source folder**  
  - **Unified entry script**: e.g. `run_<source>_fetch.py` (single CLI for all data types from that source).  
  - **README.md**: usage, allowed params, data types, column specs, and storage paths.  
  - **Helper modules** (optional): e.g. `config.py`, `client.py`, shared helpers used only within that source.

- **No cross-source logic in the entry script**  
  - Each `run_<source>_fetch.py` only handles that source; shared conventions are documented here and in each README.

---

## 2. Data storage and LEAN folder structure

- **Root directory**: All output goes under the **data directory** (e.g. project-relative `data/` or path set in `lean.json` → `"data-folder"`). Scripts must not hardcode an absolute path; use a configurable root (default `data` relative to project root).

- **Structure rule** (LEAN engine layout, not the internal CSV/ZIP layout):  
  - `data/<asset_class>/<exchange_or_source>/<data_type_or_resolution>/<symbol>/`  
  - Examples:  
    - `data/cryptofuture/binance/hour/btcusdt/`  
    - `data/cryptofuture/binance/depth/btcusdt/`  
    - `data/spot/binance/aggtrades/btcusdt/`  
  - File naming: typically `YYYYMMDD_<suffix>.zip` (e.g. `20260101_trade.zip`, `20260101_depth.zip`).  
  - No headers in CSV inside ZIP when LEAN expects headerless (document in source README).

- **Time and timezone**  
  - Timestamps in stored data: **Unix milliseconds** (e.g. `1772841600000`).  
  - All times **UTC**.

---

## 3. Unified entry script requirements

Each data source must provide a **single entry script** (e.g. `run_binance_fetch.py`) that:

### 3.1 CLI signature

- **Positional arguments** (order and meaning must be documented in the script docstring and in the source README):  
  - `[asset_class]` – e.g. `cryptofuture`, `spot`.  
  - `[data_type]` – e.g. `klines`, `aggTrades`, `bookDepth`, `quote` (replaces any legacy “tick_type” naming).  
  - `[resolution]` – where applicable: e.g. `1m`, `1h`, `1d` or `minute`, `hour`, `daily` (document the allowed set per data_type).

- **Optional arguments** (document in script and README):  
  - `--start YYYYMMDD` – start date (inclusive).  
  - `--end YYYYMMDD` – end date (inclusive).  
  - `--redownload` – force overwrite of existing files for the requested range.  
  - Source-specific flags (e.g. `--margin-type um|cm`) with a clear default (e.g. `um`).

### 3.2 Usage example in the script

The script file must contain a **top-level docstring** with:

- One-line description.
- **Usage example(s)** with the exact CLI, e.g.:  
  `python -m qc_datafeed.binance.run_binance_fetch cryptofuture klines 1h --start 20260101 --end 20260201`
- List of **allowed `data_type`** and **resolution** values and how they map to the source API (e.g. Binance Vision paths).
- Short note on where data is written (data folder and path pattern).

### 3.3 Allowed params in the unified entry

Each source README must include a **table or list** of:

- **asset_class**: allowed values and meaning.  
- **data_type**: all supported types (e.g. klines, aggTrades, bookDepth, indexPriceKlines, markPriceKlines, premiumIndexKlines, metrics for futures).  
- **resolution**: allowed resolutions per data_type (e.g. 1m/1h/1d for klines; N/A for aggTrades).  
- **Default values** (e.g. margin_type=um, date range from config or env).  
- **Behavior of `--redownload`**: which files are overwritten (e.g. per-date ZIPs in the requested range).

---

## 4. Data columns and meanings

Each source README must document, **per data_type**:

- **Output path**: under `data/`, e.g. `data/cryptofuture/binance/depth/btcusdt/`.  
- **File naming**: e.g. `YYYYMMDD_depth.zip` containing `YYYYMMDD_depth.csv`.  
- **CSV format**:  
  - Header or no header (as required by LEAN or downstream).  
  - **Column names and order**.  
  - **Meaning of each column** (e.g. `transact_time`: trade time in Unix ms UTC; `is_buyer_maker`: true if maker was seller).  
  - **Units**: e.g. price in quote asset, quantity in base asset, time in milliseconds.

This allows strategies and data loaders to rely on a single place for the contract of each data type.

---

## 5. Conventions

- **Logging**: Use `logging.getLogger(__name__)`; no sensitive data (API keys, raw orders) in logs.  
- **Errors**: Prefer clear messages and non-zero exit code on fatal errors.  
- **Idempotence**: Without `--redownload`, skip dates (or files) that already exist.  
- **Resume**: Prefer resuming from the last missing date instead of re-downloading from start.  
- **Dependencies**: Prefer minimal deps; document any in the source README or a requirements snippet.  
- **No changes to other packages**: Datafeed scripts may *refer* to e.g. `ccxt_data_fetch` for symbol lists or utilities, but must not change that package’s logic; all new logic stays under `qc_datafeed/<source>/`.

---

## 6. Summary checklist for a new source

1. Add `qc_datafeed/<source>/` and a README that describes folder layout, CLI, params, and column specs.  
2. Implement `run_<source>_fetch.py` with the unified CLI, usage docstring, and storage under `data/` following LEAN structure.  
3. Use Unix ms and UTC for all timestamps.  
4. Document allowed `asset_class`, `data_type`, `resolution`, and defaults.  
5. Document output path, file names, and CSV columns/meanings for each data_type.
