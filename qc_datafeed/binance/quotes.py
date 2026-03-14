import io
import logging
import os
import zipfile

import numpy as np
import pandas as pd

from qc_datafeed.binance.config import DATA_ROOT, PROXIES  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


def _format_symbol(symbol: str) -> str:
    """
    Normalize symbols like BTCUSDT or BTC/USDT:USDT to btcusdt.

    We try to stay consistent with ccxt_data_fetch.utils.format_symbol if available,
    but fall back to a local implementation if that package is not installed.
    """
    try:
        from ccxt_data_fetch.utils import format_symbol as _ext_format_symbol  # type: ignore[import]

        return _ext_format_symbol(symbol)
    except Exception:
        return symbol.split(":")[0].replace("/", "").lower()


def save_quote_bars(
    symbol: str,
    date_str: str,
    quote_bars_df: pd.DataFrame,
    asset_class: str = "cryptofuture",
    resolution: str = "minute",
) -> None:
    """
    Save QuoteBars to LEAN layout under qc_datafeed's DATA_ROOT.

    Output path (follows LEAN structure and ccxt_data_fetch conventions):
        data/<asset_class>/binance/<resolution>/<symbol>/<YYYYMMDD>_quote.zip

    CSV inside ZIP (no header):
        Time(ms since midnight), BidOpen, BidHigh, BidLow, BidClose, BidSize,
        AskOpen, AskHigh, AskLow, AskClose, AskSize.
    """
    if quote_bars_df is None or quote_bars_df.empty:
        logger.debug("save_quote_bars: empty dataframe for %s on %s, skipping", symbol, date_str)
        return

    if resolution not in ("minute", "hour", "daily"):
        logger.warning("save_quote_bars: unsupported resolution %s, defaulting to 'minute'", resolution)
        resolution = "minute"

    formatted_symbol = _format_symbol(symbol)
    base_dir = os.path.join(DATA_ROOT, asset_class, "binance", resolution)
    symbol_dir = os.path.join(base_dir, formatted_symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    df = quote_bars_df.copy()
    time_col = "time_val" if "time_val" in df.columns else "ms_midnight"

    # Ensure all required columns exist in the expected order.
    required_cols = [
        time_col,
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "bid_size",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
        "ask_size",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning("save_quote_bars: missing columns %s for %s on %s", missing, symbol, date_str)
        return

    lean_df = df[required_cols]
    zip_path = os.path.join(symbol_dir, f"{date_str}_quote.zip")
    csv_name = f"{date_str}_quote.csv"

    csv_content = lean_df.to_csv(index=False, header=False, float_format="%.8f")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, csv_content)

    logger.info(
        "save_quote_bars: wrote %s quote for %s on %s to %s",
        resolution,
        symbol,
        date_str,
        zip_path,
    )


def _normalize_aggtrades_for_quotes(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Binance aggTrades CSV to columns suitable for quote reconstruction.

    Expected Binance futures aggTrades layout (no header, 7+ columns):
        0: agg_trade_id
        1: price
        2: quantity
        3: first_trade_id
        4: last_trade_id
        5: transact_time (Unix ms or ms since midnight)
        6: is_buyer_maker (True if maker was seller, 0/1 or boolean)
        [7: is_best_match]  # often dropped
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    # If columns are positional (0, 1, 2, ...), assign standard Binance aggTrades semantics.
    cols = list(df.columns)
    col_map: dict[object, str] = {}
    if all(isinstance(c, int) for c in cols):
        if len(cols) >= 2:
            col_map[cols[1]] = "price"
        if len(cols) >= 3:
            col_map[cols[2]] = "qty"
        if len(cols) >= 6:
            col_map[cols[5]] = "timestamp"
        if len(cols) >= 7:
            col_map[cols[6]] = "is_buyer_maker"
    else:
        # Fallback: infer from string column names if present.
        for c in cols:
            c_lower = str(c).strip().lower()
            if c_lower in ("price", "p"):
                col_map[c] = "price"
            elif c_lower in ("qty", "quantity", "q"):
                col_map[c] = "qty"
            elif c_lower in ("timestamp", "time", "t", "transact_time"):
                col_map[c] = "timestamp"
            elif c_lower in ("isbuyermaker", "is_buyer_maker", "buyer_maker"):
                col_map[c] = "is_buyer_maker"

    df = df.rename(columns=col_map)

    required = {"price", "qty", "timestamp"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning("normalize_aggtrades_for_quotes: missing columns %s", missing)
        return pd.DataFrame()

    # Price and quantity as numeric.
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")

    # Timestamp → ms since midnight UTC.
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    one_day_ms = 24 * 60 * 60 * 1000
    # Heuristic: if max < 1 day, assume it is already "ms since midnight"; otherwise treat as Unix ms.
    if ts.max(skipna=True) < one_day_ms:
        ms_midnight = ts
    else:
        ms_midnight = ts % one_day_ms
    df["ms_midnight"] = ms_midnight.astype("Int64")

    # is_buyer_maker: True = seller aggressor → trade executes at bid.
    if "is_buyer_maker" in df.columns:
        s = df["is_buyer_maker"].astype(str).str.strip().str.lower()
        df["is_buyer_maker"] = s.isin(("true", "1", "t", "yes", "y"))
    else:
        # Unknown aggressor side: mark as NA; quotes will treat such trades as neutral.
        df["is_buyer_maker"] = pd.NA

    df = df.dropna(subset=["ms_midnight", "price", "qty"])
    return df[["ms_midnight", "price", "qty", "is_buyer_maker"]]


def build_minute_quote_bars_from_aggtrades(
    aggtrades_df: pd.DataFrame,
    static_spread: float = 0.5,
) -> pd.DataFrame:
    """
    Build minute-level QuoteBars from Binance aggTrades.

    Logic:
    - Classify each trade as bid-side or ask-side using is_buyer_maker:
        * is_buyer_maker = True  -> seller aggressor -> trade occurs at best bid.
        * is_buyer_maker = False -> buyer aggressor -> trade occurs at best ask.
    - Within each minute bucket:
        * Collect bid and ask price sequences.
        * If one side is missing, infer it from the other side plus/minus a static spread.
        * Aggregate each side to OHLC and sum quantities as BidSize / AskSize.

    Returns DataFrame with LEAN QuoteBar columns:
        time_val, bid_open, bid_high, bid_low, bid_close, bid_size,
        ask_open, ask_high, ask_low, ask_close, ask_size.
    """
    norm = _normalize_aggtrades_for_quotes(aggtrades_df)
    if norm.empty:
        return pd.DataFrame()

    # Sort by time and build a per-trade quote stream, carrying forward the most recent
    # bid/ask and spread so that bid <= ask always holds.
    df = norm.sort_values("ms_midnight").reset_index(drop=True)

    last_bid: float | None = None
    last_ask: float | None = None
    last_spread: float = max(static_spread, 0.0)

    snap_rows: list[dict[str, float]] = []
    for row in df.itertuples(index=False):
        ms_midnight = float(row.ms_midnight)
        price = float(row.price)
        qty = float(row.qty)
        side = row.is_buyer_maker

        bid_price: float | None = None
        ask_price: float | None = None
        bid_qty = 0.0
        ask_qty = 0.0

        if side is True:
            # Seller aggressor -> trade at bid.
            bid_price = price
            bid_qty = qty
            if last_ask is not None:
                # Keep or widen spread so ask >= bid.
                spread = max(last_ask - bid_price, static_spread, 0.0)
                ask_price = bid_price + spread
            else:
                spread = last_spread or static_spread
                ask_price = bid_price + max(spread, static_spread, 0.0)
        elif side is False:
            # Buyer aggressor -> trade at ask.
            ask_price = price
            ask_qty = qty
            if last_bid is not None:
                spread = max(ask_price - last_bid, static_spread, 0.0)
                bid_price = ask_price - spread
            else:
                spread = last_spread or static_spread
                bid_price = ask_price - max(spread, static_spread, 0.0)
        else:
            # Unknown side: fall back to last known bid/ask.
            if last_bid is None or last_ask is None:
                continue
            bid_price = last_bid
            ask_price = last_ask

        # Ensure we have a consistent quote.
        if bid_price is None or ask_price is None:
            continue
        if ask_price < bid_price:
            # Enforce non-crossing market by snapping ask to at least bid + static_spread.
            ask_price = bid_price + max(static_spread, 0.0)

        last_bid = bid_price
        last_ask = ask_price
        last_spread = max(ask_price - bid_price, static_spread, 0.0)

        snap_rows.append(
            {
                "ms_midnight": ms_midnight,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "bid_qty": bid_qty,
                "ask_qty": ask_qty,
            }
        )

    if not snap_rows:
        return pd.DataFrame()

    snap_df = pd.DataFrame(snap_rows)
    snap_df["minute_bucket"] = (snap_df["ms_midnight"] // 60_000) * 60_000

    rows: list[dict[str, float]] = []
    for bucket_ms, g in snap_df.groupby("minute_bucket"):
        if g.empty:
            continue

        bid_prices = g["bid_price"]
        ask_prices = g["ask_price"]

        bid_open = float(bid_prices.iloc[0])
        bid_high = float(bid_prices.max())
        bid_low = float(bid_prices.min())
        bid_close = float(bid_prices.iloc[-1])

        ask_open = float(ask_prices.iloc[0])
        ask_high = float(ask_prices.max())
        ask_low = float(ask_prices.min())
        ask_close = float(ask_prices.iloc[-1])

        # As an extra safety, enforce bid <= ask at bar level.
        if ask_low < bid_low:
            mid = 0.5 * (bid_low + ask_low)
            bid_low = mid - 0.5 * static_spread
            ask_low = mid + 0.5 * static_spread
        if ask_close < bid_close:
            mid = 0.5 * (bid_close + ask_close)
            bid_close = mid - 0.5 * static_spread
            ask_close = mid + 0.5 * static_spread

        bid_size = float(g["bid_qty"].sum())
        ask_size = float(g["ask_qty"].sum())
        total_qty = bid_size + ask_size

        if bid_size == 0.0 and ask_size == 0.0 and total_qty > 0.0:
            bid_size = total_qty / 2.0
            ask_size = total_qty / 2.0

        rows.append(
            {
                "time_val": int(bucket_ms),
                "bid_open": bid_open,
                "bid_high": bid_high,
                "bid_low": bid_low,
                "bid_close": bid_close,
                "bid_size": bid_size,
                "ask_open": ask_open,
                "ask_high": ask_high,
                "ask_low": ask_low,
                "ask_close": ask_close,
                "ask_size": ask_size,
            }
        )

    if not rows:
        return pd.DataFrame()

    out_df = pd.DataFrame(rows).sort_values("time_val").reset_index(drop=True)
    logger.info(
        "build_minute_quote_bars_from_aggtrades: built %d bars from %d aggTrades rows",
        len(out_df),
        len(df),
    )
    return out_df


def build_minute_quote_from_aggtrades_files(
    symbol: str,
    date_str: str,
    asset_class: str = "cryptofuture",
    static_spread: float = 0.5,
    redownload: bool = False,
) -> None:
    """
    Rebuild LEAN-compatible minute QuoteBars for one symbol/day from local aggTrades files.

    Input:
        data/<asset_class>/binance/aggtrades/<symbol>/<YYYYMMDD>_aggtrades.zip

    Output:
        data/<asset_class>/binance/minute/<symbol>/<YYYYMMDD>_quote.zip
        with CSV columns:
            Time(ms since midnight), BidOpen, BidHigh, BidLow, BidClose, BidSize,
            AskOpen, AskHigh, AskLow, AskClose, AskSize.
    """
    # Normalize date to YYYYMMDD for file naming.
    date_yyyymmdd = date_str.replace("-", "")

    formatted_symbol = _format_symbol(symbol)

    # Output quote path (skip if it already exists and redownload is False).
    quote_dir = os.path.join(
        DATA_ROOT,
        asset_class,
        "binance",
        "minute",
        formatted_symbol,
    )
    quote_zip = os.path.join(quote_dir, f"{date_yyyymmdd}_quote.zip")
    if os.path.isfile(quote_zip) and not redownload:
        logger.info(
            "build_minute_quote_from_aggtrades_files: quote already exists for %s on %s at %s; skipping",
            symbol,
            date_yyyymmdd,
            quote_zip,
        )
        return

    agg_dir = os.path.join(
        DATA_ROOT,
        asset_class,
        "binance",
        "aggtrades",
        formatted_symbol,
    )
    zip_name = f"{date_yyyymmdd}_aggtrades.zip"
    zip_path = os.path.join(agg_dir, zip_name)

    if not os.path.isfile(zip_path):
        logger.warning(
            "build_minute_quote_from_aggtrades_files: no aggTrades file for %s on %s at %s",
            symbol,
            date_str,
            zip_path,
        )
        return

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            if not names:
                logger.warning(
                    "build_minute_quote_from_aggtrades_files: empty zip for %s on %s at %s",
                    symbol,
                    date_str,
                    zip_path,
                )
                return
            entry = names[0]
            with zf.open(entry) as f:
                raw = pd.read_csv(f, header=None, low_memory=False, dtype=str)
    except Exception as e:
        logger.error(
            "build_minute_quote_from_aggtrades_files: failed to read %s: %s",
            zip_path,
            e,
        )
        return

    bars = build_minute_quote_bars_from_aggtrades(raw, static_spread=static_spread)
    if bars is None or bars.empty:
        logger.warning(
            "build_minute_quote_from_aggtrades_files: no quote bars built for %s on %s",
            symbol,
            date_str,
        )
        return

    save_quote_bars(
        symbol=symbol,
        date_str=date_yyyymmdd,
        quote_bars_df=bars,
        asset_class=asset_class,
        resolution="minute",
    )
    logger.info(
        "build_minute_quote_from_aggtrades_files: wrote minute quote for %s on %s from %s",
        symbol,
        date_yyyymmdd,
        zip_path,
    )

