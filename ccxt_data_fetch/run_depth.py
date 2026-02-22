import logging
import os
import sys
from datetime import datetime, time, timezone, timedelta
from ccxt_data_fetch.config import START_DATE, END_DATE, DATA_LOCATION, TOP_N_SYMBOL
from ccxt_data_fetch.utils import get_top_200_symbols, format_symbol
from ccxt_data_fetch.fetcher import (
    fetch_depth_range_cryptofuture,
    save_depth_data
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# datetime set time zone to UTC
os.environ['TZ'] = 'UTC'
time.tzset()

def run_fetch_depth(asset_class="cryptofuture"):
    """
    Fetches and saves book depth data for top symbols.
    Adheres to 'only minute data' by using step_ms=60000.
    """
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    # 1. Get Symbols
    symbols = get_top_200_symbols(asset_class)
    symbols = symbols[:TOP_N_SYMBOL]
    logger.info(f"Starting DEPTH fetch: {asset_class} from {START_DATE} to {END_DATE} for {len(symbols)} symbols...")

    for symbol in symbols:
        try:
            formatted_symbol = format_symbol(symbol)
            
            # Resume Logic: Check existing files to avoid re-downloading
            symbol_dir = os.path.join(DATA_LOCATION, asset_class, "binance", "depth", formatted_symbol)
            current_since = since_ms
            if os.path.exists(symbol_dir):
                existing = [f for f in os.listdir(symbol_dir) if f.endswith("_depth.zip")]
                if existing:
                    latest_zip = sorted(existing)[-1]
                    latest_date = latest_zip.split("_")[0]
                    try:
                        latest_dt = datetime.strptime(latest_date, "%Y%m%d").replace(tzinfo=timezone.utc)
                        # Start from the day after the latest downloaded file
                        resume_ms = int((latest_dt + timedelta(days=1)).timestamp() * 1000)
                        if resume_ms >= until_ms:
                            logger.info(f"Skipping {symbol}, depth already up to date up to {latest_date}.")
                            continue
                        current_since = max(current_since, resume_ms)
                    except ValueError:
                        pass

            # Fetch depth snapshots
            # step_ms=60000 ensures minute-level alignment and triggers 'only minute' filtering in fetcher
            data = fetch_depth_range_cryptofuture(symbol, current_since, until_ms, step_ms=60000)

            # Save to LEAN format
            if data:
                save_depth_data(symbol, data, asset_class)
                logger.info(f"Successfully saved depth data for {symbol}.")
            else:
                logger.info(f"No new depth data fetched for {symbol}.")

        except Exception as e:
            logger.error(f"Failed to process depth for {symbol}: {e}")

if __name__ == "__main__":
    # Usage: python -m ccxt_data_fetch.run_depth [asset_class]
    asset_class = sys.argv[1] if len(sys.argv) > 1 else "cryptofuture"
    
    print(f"Downloading depth data for {asset_class} (Minute resolution) to {DATA_LOCATION}")
    run_fetch_depth(asset_class)
