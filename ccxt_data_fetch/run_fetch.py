"""
Unified fetch entrypoint for ccxt_data_fetch.

Single CLI and implementation for all data types: OHLCV, quote (L1), depth, margin_interest.

Usage:
  python -m ccxt_data_fetch.run_fetch [asset_class] [tick_type] [resolution] [--start YYYYMMDD] [--end YYYYMMDD] [--redownload] [--ohlcv-tick trade|quote]

  tick_type: ohlcv | quote | depth | margin_interest
  resolution: minute | hour | daily (default: minute; applies to ohlcv, quote, depth)
  --ohlcv-tick: for ohlcv only, trade or quote (default: trade)

  Margin interest format fix: python -m ccxt_data_fetch.run_fetch fix [data_folder]

Examples:
  python -m ccxt_data_fetch.run_fetch cryptofuture ohlcv minute --start 20260101 --end 20260201
  python -m ccxt_data_fetch.run_fetch cryptofuture ohlcv hour --ohlcv-tick quote
  python -m ccxt_data_fetch.run_fetch cryptofuture quote minute --start 20260101 --redownload
  python -m ccxt_data_fetch.run_fetch cryptofuture quote hour --start 20260101
  python -m ccxt_data_fetch.run_fetch cryptofuture depth minute --redownload
  python -m ccxt_data_fetch.run_fetch cryptofuture margin_interest --start 20260101
  python -m ccxt_data_fetch.run_fetch fix
  
Notes:
what's the difference between tick_type quote and --ohlcv-tick quote, ohlcv and --ohlcv-tick trade @ccxt_data_fetch/run_fetch.py @ccxt_data_fetch/fetcher.py 

Here’s the distinction:

---

## 1. **`tick_type`** (positional) = **which data product to fetch**

Second argument: one of **`ohlcv` | `quote` | `depth` | `margin_interest`**. It chooses the **source** and **type** of data.

| `tick_type` | Meaning | Source | Output |
|-------------|--------|--------|--------|
| **`quote`** | L1 best bid/ask (QuoteBars) | Binance Vision **bookTicker** | True quote data: Bid/Ask O/H/L/C + sizes, saved as `*_quote.zip` under `binance/<resolution>/<symbol>/`. |
| **`ohlcv`** | Candles (OHLCV) | CCXT **fetch_ohlcv** (Binance candle API) | Candle data: Open, High, Low, Close, Volume. Sub-type (trade vs quote) is then controlled by `--ohlcv-tick`. |

So:

- **`tick_type quote`** → Fetch **real quote** (bookTicker), aggregate to bars, save as **QuoteBars**.
- **`tick_type ohlcv`** → Fetch **candles** (OHLCV API). What you do with those (trade vs quote **format**) is what `--ohlcv-tick` controls.

---

## 2. **`--ohlcv-tick`** = **only when `tick_type=ohlcv`** (how to name/format those candles)

Used only for **OHLCV** runs. It does **not** change the API (you still get candles from the same OHLCV endpoint). It only changes:

- **File names**: `YYYYMMDD_trade.zip` vs `YYYYMMDD_quote.zip`
- **CSV column layout** for LEAN: trade format (Time, O, H, L, C, V) vs quote-style (Time, BidO, BidH, …, AskO, …) via `get_lean_df(..., tick_type, ...)` in `fetcher.py` (e.g. around 354–370 and 388–418).

So:

- **`ohlcv` + `--ohlcv-tick trade`** (default): Fetch OHLCV candles, save as **trade** bars (standard candle columns and `*_trade.zip`).
- **`ohlcv` + `--ohlcv-tick quote`**: Same OHLCV candles, but saved with **quote**-style filenames and column layout (for LEAN/strategies that expect quote-bar naming while still using candle data).

---

## 3. Short comparison

- **`tick_type quote`** → **Real** L1 quote data (bookTicker → QuoteBars).
- **`tick_type ohlcv`** → Candle data (OHLCV API).
  - **`--ohlcv-tick trade`** → Save those candles as trade bars.
  - **`--ohlcv-tick quote`** → Save the same candles in quote-bar naming/format.

So: **`tick_type`** = *what to fetch* (quote vs ohlcv vs depth vs margin_interest). **`--ohlcv-tick`** = *how to label/format OHLCV output* (trade vs quote), only when you’re already fetching OHLCV.
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta

from tqdm import tqdm

from ccxt_data_fetch.config import (
    START_DATE,
    END_DATE,
    DATA_LOCATION,
    TOP_N_SYMBOL,
)
from ccxt_data_fetch.utils import get_top_200_symbols, format_symbol
from ccxt_data_fetch.fetcher import (
    fetch_ohlcv_range,
    save_ohlcv_data,
    fetch_and_save_quote_range,
    fetch_and_save_depth_range,
    fetch_funding_rates,
    save_margin_interest,
    fix_margin_interest_format,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
os.environ["TZ"] = "UTC"
time.tzset()


def _normalize_date(s: str) -> str:
    """Accept YYYYMMDD or YYYY-MM-DD; return YYYY-MM-DD."""
    s = s.strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def _print_task_summary(
    task_name: str,
    asset_class: str,
    n_symbols: int,
    start_str: str,
    end_str: str,
    extra: str | None = None,
) -> None:
    """Print a one-line summary of the fetch task (multi-asset / multi-task info)."""
    parts = [
        f"Task: {task_name}",
        f"asset_class={asset_class}",
        f"symbols={n_symbols}",
        f"range={start_str} to {end_str}",
    ]
    if extra:
        parts.append(extra)
    logger.info(" | ".join(parts))


# ---------------------------------------------------------------------------
# Fetch implementations
# ---------------------------------------------------------------------------


def _run_fetch_ohlcv(
    asset_class: str,
    resolution: str,
    tick_type: str,
    start_str: str,
    end_str: str,
    force_redownload: bool,
) -> None:
    start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    timeframe_map = {"minute": "1m", "hour": "1h", "daily": "1d"}
    if resolution not in timeframe_map:
        logger.error("Unknown resolution: %s (use minute, hour, daily)", resolution)
        return

    symbols = get_top_200_symbols(asset_class)
    symbols = symbols[:TOP_N_SYMBOL]
    _print_task_summary(
        "OHLCV",
        asset_class,
        len(symbols),
        start_str,
        end_str,
        extra=f"resolution={resolution} tick_type={tick_type}",
    )

    failed: list[tuple[str, str]] = []
    for symbol in tqdm(symbols, desc="OHLCV", unit="symbol"):
        try:
            current_since = since_ms
            formatted_symbol = format_symbol(symbol)

            if resolution == "minute" and not force_redownload:
                symbol_dir = os.path.join(
                    DATA_LOCATION, asset_class, "binance", "minute", formatted_symbol
                )
                if os.path.exists(symbol_dir):
                    existing = [
                        f
                        for f in os.listdir(symbol_dir)
                        if f.endswith(f"_{tick_type}.zip")
                    ]
                    if existing:
                        latest_zip = sorted(existing)[-1]
                        latest_date = latest_zip.split("_")[0]
                        try:
                            latest_dt = datetime.strptime(
                                latest_date, "%Y%m%d"
                            ).replace(tzinfo=timezone.utc)
                            resume_ms = int(
                                (latest_dt + timedelta(days=1)).timestamp() * 1000
                            )
                            if resume_ms >= until_ms:
                                continue
                            current_since = max(current_since, resume_ms)
                        except ValueError:
                            pass

            data = fetch_ohlcv_range(
                symbol, timeframe_map[resolution], current_since, until_ms, asset_class
            )
            save_ohlcv_data(symbol, data, resolution, asset_class, tick_type)
        except Exception as e:
            logger.error("Failed to process %s: %s", symbol, e)
            failed.append((symbol, str(e)))
    n_ok = len(symbols) - len(failed)
    if failed:
        logger.warning("OHLCV done. %s/%s symbols OK, %s failed: %s", n_ok, len(symbols), len(failed), [s for s, _ in failed])
    else:
        logger.info("OHLCV done. All %s symbols OK.", len(symbols))


def _run_fetch_quote(
    asset_class: str,
    resolution: str,
    start_str: str,
    end_str: str,
    force_redownload: bool,
) -> None:
    start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    ) + timedelta(days=1)
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    symbols = get_top_200_symbols(asset_class)
    symbols = symbols[:TOP_N_SYMBOL]
    mode = "REDOWNLOAD" if force_redownload else "build"
    _print_task_summary(
        f"QUOTE ({mode})",
        asset_class,
        len(symbols),
        start_str,
        end_str,
        extra=f"resolution={resolution}",
    )

    failed: list[tuple[str, str]] = []
    for symbol in tqdm(symbols, desc="Quote", unit="symbol"):
        try:
            fetch_and_save_quote_range(
                symbol,
                since_ms,
                until_ms,
                asset_class=asset_class,
                resolution=resolution,
                force_redownload=force_redownload,
            )
        except Exception as e:
            logger.error("Failed to process quote for %s: %s", symbol, e)
            failed.append((symbol, str(e)))
    n_ok = len(symbols) - len(failed)
    if failed:
        logger.warning("Quote done. %s/%s symbols OK, %s failed: %s", n_ok, len(symbols), len(failed), [s for s, _ in failed])
    else:
        logger.info("Quote done. All %s symbols OK.", len(symbols))


def _run_fetch_depth(
    asset_class: str,
    resolution: str,
    start_str: str,
    end_str: str,
    force_redownload: bool,
) -> None:
    start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    symbols = get_top_200_symbols(asset_class)
    symbols = symbols[:TOP_N_SYMBOL]
    mode = "REDOWNLOAD" if force_redownload else "fetch"
    _print_task_summary(
        f"DEPTH ({mode})",
        asset_class,
        len(symbols),
        start_str,
        end_str,
        extra=f"resolution={resolution}",
    )

    failed: list[tuple[str, str]] = []
    for symbol in tqdm(symbols, desc="Depth", unit="symbol"):
        try:
            fetch_and_save_depth_range(
                symbol,
                since_ms,
                until_ms,
                asset_class=asset_class,
                resolution=resolution,
                force_redownload=force_redownload,
            )
        except Exception as e:
            logger.error("Failed to process depth for %s: %s", symbol, e)
            failed.append((symbol, str(e)))
    n_ok = len(symbols) - len(failed)
    if failed:
        logger.warning("Depth done. %s/%s symbols OK, %s failed: %s", n_ok, len(symbols), len(failed), [s for s, _ in failed])
    else:
        logger.info("Depth done. All %s symbols OK.", len(symbols))


def _run_fetch_margin_interest(
    asset_class: str,
    start_str: str,
    end_str: str,
) -> None:
    if asset_class != "cryptofuture":
        logger.warning("Margin interest only valid for cryptofuture.")
        return

    start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    ) + timedelta(days=1)
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    symbols = get_top_200_symbols(asset_class)
    symbols = symbols[:TOP_N_SYMBOL]
    _print_task_summary("MARGIN_INTEREST", asset_class, len(symbols), start_str, end_str)

    failed: list[tuple[str, str]] = []
    for symbol in tqdm(symbols, desc="Margin interest", unit="symbol"):
        try:
            rates = fetch_funding_rates(symbol, since_ms, until_ms)
            save_margin_interest(symbol, rates)
        except Exception as e:
            logger.error("Failed to process %s: %s", symbol, e)
            failed.append((symbol, str(e)))
    n_ok = len(symbols) - len(failed)
    if failed:
        logger.warning("Margin interest done. %s/%s symbols OK, %s failed: %s", n_ok, len(symbols), len(failed), [s for s, _ in failed])
    else:
        logger.info("Margin interest done. All %s symbols OK.", len(symbols))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified fetch: OHLCV, quote, depth, margin_interest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "asset_class",
        nargs="?",
        default="cryptofuture",
        help="Asset class (default: cryptofuture); or 'fix' for margin_interest format fix",
    )
    parser.add_argument(
        "tick_type",
        nargs="?",
        default="ohlcv",
        help="Data to fetch: ohlcv | quote | depth | margin_interest (default: ohlcv)",
    )
    parser.add_argument(
        "resolution",
        nargs="?",
        default="minute",
        help="Resolution: minute | hour | daily (default: minute; for ohlcv, quote, depth)",
    )
    parser.add_argument(
        "--ohlcv-tick",
        choices=("trade", "quote"),
        default="trade",
        help="For OHLCV only: trade or quote tick type (default: trade)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date YYYYMMDD or YYYY-MM-DD (default: config.START_DATE)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYYMMDD or YYYY-MM-DD (default: config.END_DATE)",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Force redownload and overwrite (quote, depth, ohlcv)",
    )
    args = parser.parse_args()

    # Margin interest format fix: run_fetch fix [data_folder]
    if (args.asset_class or "").lower() == "fix":
        data_folder = None
        if args.tick_type and args.tick_type not in (
            "ohlcv",
            "quote",
            "depth",
            "margin_interest",
        ):
            data_folder = os.path.abspath(args.tick_type)
        n = fix_margin_interest_format(data_folder=data_folder)
        logger.info("Fixed %s margin interest CSV(s).", n)
        return

    start_str = _normalize_date(args.start) if args.start else START_DATE
    end_str = _normalize_date(args.end) if args.end else END_DATE

    data_type = (args.tick_type or "ohlcv").lower()
    asset_class = args.asset_class or "cryptofuture"
    resolution = (args.resolution or "minute").lower()

    if data_type == "ohlcv":
        _run_fetch_ohlcv(
            asset_class,
            resolution,
            getattr(args, "ohlcv_tick", "trade"),
            start_str,
            end_str,
            args.redownload,
        )
    elif data_type == "quote":
        _run_fetch_quote(
            asset_class,
            resolution,
            start_str,
            end_str,
            args.redownload,
        )
    elif data_type == "depth":
        _run_fetch_depth(
            asset_class,
            resolution,
            start_str,
            end_str,
            args.redownload,
        )
    elif data_type == "margin_interest":
        _run_fetch_margin_interest(asset_class, start_str, end_str)
    else:
        logger.error(
            "Unknown tick_type %r. Use: ohlcv | quote | depth | margin_interest",
            data_type,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
