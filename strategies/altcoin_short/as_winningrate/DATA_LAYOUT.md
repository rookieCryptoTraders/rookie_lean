# Data layout (as_winningrate)

This strategy uses **QuantConnect’s custom data layout** so paths resolve correctly when running LEAN in Docker.

## Layout

All custom data lives under the LEAN **data folder** (set by `lean.json` → `"data-folder": "data"`). Inside that folder:

- **Depth (LOB):** `data/custom/cryptofuture-depth/<TICKER>/minute/<YYYYMMDD>_depth.zip`  
  - Example: `data/custom/cryptofuture-depth/btcusdt/minute/20260201_depth.zip`  
  - ZIP contains: `20260201_depth.csv` (no header; columns: `ms_midnight, percentage, depth, notional`).

- **Quote (minute QuoteBar):** Standard LEAN path `data/cryptofuture/binance/minute/<TICKER>/<YYYYMMDD>_quote.zip`  
  - CSV (no header): `Time (ms since midnight), BidOpen, BidHigh, BidLow, BidClose, BidSize, AskOpen, AskHigh, AskLow, AskClose, AskSize`.  
  - Fetched by `ccxt_data_fetch.run_fetch cryptofuture quote minute` from Binance (Vision bookTicker or REST `/fapi/v1/ticker/bookTicker`).  
  - Optional custom layout for strategies: `data/custom/cryptofuture-quote/...` (if you copy/symlink from the standard path).

Tickers are **lowercase**, e.g. `btcusdt`, `ethusdt`.

## Docker

When you run `lean backtest`, the CLI mounts your project (including the `data` directory). LEAN sets `Globals.data_folder` to that mounted path, so custom data is read from:

- `Globals.data_folder/custom/cryptofuture-depth/<symbol>/minute/...`

Ensure:

1. `lean.json` has `"data-folder": "data"` (relative to project root).
2. Your repo’s `data` directory is the one mounted (default with `lean backtest`).

## Setting up depth data

If you already have depth files under the old layout (`data/cryptofuture/binance/minute/<symbol>/`), either:

- **Copy or move** files into the custom layout:
  - From: `data/cryptofuture/binance/minute/btcusdt/20260201_depth.zip`
  - To:   `data/custom/cryptofuture-depth/btcusdt/minute/20260201_depth.zip`
- **Symlink** the minute directory (from repo root):
  - `mkdir -p data/custom/cryptofuture-depth/btcusdt`
  - `ln -s ../../../cryptofuture/binance/minute/btcusdt data/custom/cryptofuture-depth/btcusdt/minute`

Same pattern for other symbols (`ethusdt`, `solusdt`, etc.).

## Config (config.py)

- `CUSTOM_DEPTH_MAP = "cryptofuture-depth"`
- `CUSTOM_QUOTE_MAP = "cryptofuture-quote"`
- `CUSTOM_RESOLUTION_FOLDER = "minute"`

PythonData classes (`CryptoFutureDepthData`, `CryptoFutureQuoteData`) build paths from `Globals.data_folder` and these constants only; no legacy path fallback.
