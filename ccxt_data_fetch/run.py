import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from ccxt_data_fetch.config import START_DATE, END_DATE, DATA_LOCATION, TOP_N_SYMBOL
from ccxt_data_fetch.utils import get_top_200_symbols, format_symbol
from ccxt_data_fetch.fetcher import (
    fetch_ohlcv_range, 
    fetch_funding_rates,
    save_hour_data, 
    save_minute_data, 
    save_daily_data, 
    save_margin_interest
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_fetch(asset_class, resolution, tick_type="trade"):
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    # 1. Get Symbols
    symbols = get_top_200_symbols(asset_class)
    symbols = symbols[:TOP_N_SYMBOL]
    logger.info(f"Starting fetch: {asset_class} | {resolution} | {tick_type} for {len(symbols)} symbols...")

    for symbol in symbols:
        try:
            current_since = since_ms
            formatted_symbol = format_symbol(symbol)
            
            # Special Case: Margin Interest
            if resolution == "margin_interest":
                if asset_class != "cryptofuture":
                    logger.warning("Margin interest only valid for cryptofuture.")
                    return
                rates = fetch_funding_rates(symbol, since_ms, until_ms)
                save_margin_interest(symbol, rates)
                continue

            # Standard OHLCV Fetching
            timeframe_map = {"minute": "1m", "hour": "1h", "daily": "1d"}
            if resolution not in timeframe_map:
                logger.error(f"Unknown resolution: {resolution}")
                return
            ccxt_timeframe = timeframe_map[resolution]

            # Resume Logic (Optimized for minute data)
            if resolution == "minute":
                symbol_dir = os.path.join(DATA_LOCATION, asset_class, "binance", "minute", formatted_symbol)
                if os.path.exists(symbol_dir):
                    existing = [f for f in os.listdir(symbol_dir) if f.endswith(f"_{tick_type}.zip")]
                    if existing:
                        latest_zip = sorted(existing)[-1]
                        latest_date = latest_zip.split("_")[0]
                        try:
                            latest_dt = datetime.strptime(latest_date, "%Y%m%d").replace(tzinfo=timezone.utc)
                            resume_ms = int((latest_dt + timedelta(days=1)).timestamp() * 1000)
                            if resume_ms >= until_ms:
                                logger.info(f"Skipping {symbol}, already up to date.")
                                continue
                            current_since = max(current_since, resume_ms)
                        except ValueError:
                            pass # If filename parsing fails, restart/continue normal flow

            # Fetch
            data = fetch_ohlcv_range(symbol, ccxt_timeframe, current_since, until_ms, asset_class)

            # Save
            if resolution == "minute":
                save_minute_data(symbol, data, asset_class, tick_type)
            elif resolution == "hour":
                save_hour_data(symbol, data, asset_class, tick_type)
            elif resolution == "daily":
                save_daily_data(symbol, data, asset_class, tick_type)

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")

if __name__ == "__main__":
    # Usage: python -m ccxt_data_fetch.run [asset_class] [resolution] [tick_type]
    # Examples:
    # python -m ccxt_data_fetch.run cryptofuture    minute  trade
    # python -m ccxt_data_fetch.run crypto          daily   quote
    # python -m ccxt_data_fetch.run cryptofuture    margin_interest
    
    args = sys.argv[1:]
    
    # Defaults
    asset_class = "cryptofuture"
    resolution = "hour"
    tick_type = "trade"
    
    if len(args) >= 1:
        # Heuristic: if first arg is 'crypto' or 'cryptofuture', it's asset_class
        if args[0] in ["crypto", "cryptofuture"]:
            asset_class = args[0]
            if len(args) >= 2:
                resolution = args[1]
            if len(args) >= 3:
                tick_type = args[2]
        else:
            # Backward compatibility: user might have just passed 'minute' or 'hour'
            resolution = args[0]
            if len(args) >= 2:
                tick_type = args[1]
            # default asset_class 'cryptofuture' remains
    
    print(f"Downloading {asset_class} {resolution} {tick_type} to {DATA_LOCATION}")
    run_fetch(asset_class, resolution, tick_type)