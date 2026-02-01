import logging
import os
from datetime import datetime, timezone, timedelta
from ccxt_data_fetch.config import START_DATE, END_DATE, DATA_LOCATION, ASSET_CLASS
from ccxt_data_fetch.utils import get_top_200_symbols, format_symbol
from ccxt_data_fetch.fetcher import fetch_ohlcv_range, save_hour_data, save_minute_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_fetch(resolution="hour"):
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    ) + timedelta(days=1)

    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    symbols = get_top_200_symbols()
    logger.info(f"Starting {resolution} fetch for {len(symbols)} symbols...")

    for symbol in symbols:
        try:
            current_since = since_ms
            formatted_symbol = format_symbol(symbol)

            # Resume logic for minute data
            if resolution == "minute":
                symbol_dir = os.path.join(
                    DATA_LOCATION, ASSET_CLASS, "binance", "minute", formatted_symbol
                )
                if os.path.exists(symbol_dir):
                    existing = [
                        f for f in os.listdir(symbol_dir) if f.endswith("_trade.zip")
                    ]
                    if existing:
                        latest_zip = sorted(existing)[-1]
                        latest_date = latest_zip.split("_")[0]
                        latest_dt = datetime.strptime(latest_date, "%Y%m%d").replace(
                            tzinfo=timezone.utc
                        )
                        resume_ms = int(
                            (latest_dt + timedelta(days=1)).timestamp() * 1000
                        )
                        if resume_ms >= until_ms:
                            logger.info(f"Skipping {symbol}, already up to date.")
                            continue
                        current_since = max(current_since, resume_ms)

            data = fetch_ohlcv_range(
                symbol, "1h" if resolution == "hour" else "1m", current_since, until_ms
            )

            if resolution == "hour":
                save_hour_data(symbol, data)
            else:
                save_minute_data(symbol, data)

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")


if __name__ == "__main__":
    import sys

    res = sys.argv[1] if len(sys.argv) > 1 else "hour"
    run_fetch(res)
