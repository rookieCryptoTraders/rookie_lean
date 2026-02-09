import ccxt
import pandas as pd
import os
import zipfile
import time
import logging
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from ccxt_data_fetch.config import DATA_LOCATION, PROXIES
from ccxt_data_fetch.utils import format_symbol, get_ms_from_midnight

logger = logging.getLogger(__name__)


def fetch_ohlcv_range(symbol, timeframe, since_ms, until_ms, asset_class="cryptofuture"):
    options = {}
    if asset_class == "cryptofuture":
        options["defaultType"] = "future"
    
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "proxies": PROXIES,
            "options": options,
        }
    )

    # Map timeframe to ms
    if timeframe == "1m":
        ms_per_step = 60000
    elif timeframe == "1h":
        ms_per_step = 3600000
    elif timeframe == "1d":
        ms_per_step = 86400000
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    all_ohlcv = []
    current_since = since_ms
    pbar = tqdm(total=until_ms - since_ms, desc=f"Fetching {timeframe} {symbol}")

    while current_since < until_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=current_since, limit=1000
            )
            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            # Ensure we move forward
            next_step = last_ts + ms_per_step
            
            # Update progress
            progress = min(next_step, until_ms) - current_since
            if progress > 0:
                pbar.update(progress)
            
            current_since = max(next_step, current_since + 1) # Prevent infinite loop if exchange returns same data

            if current_since >= until_ms:
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            logger.error(f"Error for {symbol}: {e}")
            time.sleep(10)
            continue

    pbar.close()
    return all_ohlcv


def fetch_funding_rates(symbol, since_ms, until_ms):
    """
    Fetch funding rate history for futures.
    Returns list of [timestamp, fundingRate, ...]
    """
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "proxies": PROXIES,
            "options": {"defaultType": "future"},
        }
    )
    
    all_rates = []
    current_since = since_ms
    pbar = tqdm(total=until_ms - since_ms, desc=f"Fetching Funding Rates {symbol}")
    
    while current_since < until_ms:
        try:
            rates = exchange.fetch_funding_rate_history(
                symbol, since=current_since, limit=1000
            )
            if not rates:
                break
                
            all_rates.extend(rates)
            last_ts = rates[-1]['timestamp']
            
            # Progress
            progress = min(last_ts, until_ms) - current_since
            if progress > 0:
                pbar.update(progress)
                
            current_since = last_ts + 1 # ccxt usually returns inclusive, need to step forward
            
            if current_since >= until_ms:
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            logger.error(f"Error fetching funding rates for {symbol}: {e}")
            time.sleep(10)
            break
            
    pbar.close()
    return all_rates


def get_lean_df(df, asset_class, tick_type, time_col):
    """
    Returns a dataframe with columns in the order LEAN expects for the given tick_type.
    """
    if tick_type == "trade":
        # Standard Trade format: Time, Open, High, Low, Close, Volume
        return df[[time_col, "open", "high", "low", "close", "volume"]]

    elif tick_type == "quote":
        # Standard Quote format: Time, BidOpen, BidHigh, BidLow, BidClose, BidSize, AskOpen, AskHigh, AskLow, AskClose, AskSize
        # Fallback to OHLC if Bid/Ask is missing
        for side in ["bid", "ask"]:
            for price in ["open", "high", "low", "close"]:
                col = f"{side}_{price}"
                if col not in df.columns:
                    df[col] = df[price]
        
        if asset_class in ["forex", "cfd"]:
            # Forex/CFD Quote usually omits sizes in some formats, but LEAN often expects Bid/Ask OHLC
            return df[[time_col, "bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close"]]
        else:
            if "bid_size" not in df.columns: df["bid_size"] = pd.NA
            if "ask_size" not in df.columns: df["ask_size"] = pd.NA
            return df[[time_col, "bid_open", "bid_high", "bid_low", "bid_close", "bid_size", "ask_open", "ask_high", "ask_low", "ask_close", "ask_size"]]

    elif tick_type == "open_interest":
        # Standard Open Interest format: Time, Open Interest
        if "open_interest" not in df.columns:
            df["open_interest"] = df["volume"]
        return df[[time_col, "open_interest"]]

    return df[[time_col, "open", "high", "low", "close", "volume"]]


def save_daily_data(symbol, ohlcv_data, asset_class="cryptofuture", tick_type="trade"):
    if not ohlcv_data:
        return
    df = pd.DataFrame(
        ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    # LEAN Daily format: YYYYMMDD 00:00
    df["formatted_time"] = df["dt"].dt.strftime("%Y%m%d 00:00")

    formatted_symbol = format_symbol(symbol)
    base_dir = os.path.join(DATA_LOCATION, asset_class, "binance", "daily")
    os.makedirs(base_dir, exist_ok=True)

    # For daily, we often use the tick_type suffix for clarity
    zip_path = os.path.join(base_dir, f"{formatted_symbol}_{tick_type}.zip")
    csv_filename = f"{formatted_symbol}.csv"

    lean_df = get_lean_df(df, asset_class, tick_type, "formatted_time")
    csv_content = lean_df.to_csv(index=False, header=False)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_filename, csv_content)
    logger.debug(f"Saved {zip_path}")


def save_hour_data(symbol, ohlcv_data, asset_class="cryptofuture", tick_type="trade"):
    if not ohlcv_data:
        return
    df = pd.DataFrame(
        ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["formatted_time"] = df["dt"].dt.strftime("%Y%m%d %H:%M")

    formatted_symbol = format_symbol(symbol)
    base_dir = os.path.join(DATA_LOCATION, asset_class, "binance", "hour")
    os.makedirs(base_dir, exist_ok=True)

    # For hour, primary tick type usually has no suffix
    primary_tick = "quote" if asset_class in ["forex", "cfd"] else "trade"
    suffix = f"_{tick_type}" if tick_type != primary_tick else ""
    zip_path = os.path.join(base_dir, f"{formatted_symbol}{suffix}.zip")
    csv_filename = f"{formatted_symbol}.csv"

    lean_df = get_lean_df(df, asset_class, tick_type, "formatted_time")
    csv_content = lean_df.to_csv(index=False, header=False)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_filename, csv_content)
    logger.debug(f"Saved {zip_path}")


def save_minute_data(symbol, ohlcv_data, asset_class="cryptofuture", tick_type="trade"):
    if not ohlcv_data:
        return
    df = pd.DataFrame(
        ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["date_str"] = df["dt"].dt.strftime("%Y%m%d")

    formatted_symbol = format_symbol(symbol)
    symbol_dir = os.path.join(
        DATA_LOCATION, asset_class, "binance", "minute", formatted_symbol
    )
    os.makedirs(symbol_dir, exist_ok=True)

    for date_str, group in df.groupby("date_str"):
        day_start = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        mask = (group["dt"] >= day_start) & (group["dt"] < day_end)
        day_group = group.loc[mask].copy()
        if day_group.empty:
            continue

        day_group["ms_midnight"] = day_group["dt"].apply(get_ms_from_midnight)
        lean_df = get_lean_df(day_group, asset_class, tick_type, "ms_midnight")

        # Filename: YYYYMMDD_trade.zip
        zip_path = os.path.join(symbol_dir, f"{date_str}_{tick_type}.zip")
        zf_name = f"{date_str}_{tick_type}.csv"
        csv_content = lean_df.to_csv(index=False, header=False)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(zf_name, csv_content)


def save_margin_interest(symbol, rates_data):
    """
    Save funding rates as margin interest.
    LEAN format for margin_rate: Time (YYYYMMDD HH:mm), Interest Rate
    """
    if not rates_data:
        return
        
    # CCXT fetchFundingRateHistory structure:
    # {'timestamp': 1600000000000, 'fundingRate': 0.0001, ...}
    
    data_list = []
    for r in rates_data:
        data_list.append([r['timestamp'], r['fundingRate']])
        
    df = pd.DataFrame(data_list, columns=["timestamp", "rate"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    # Format: YYYYMMDD HH:mm
    df["formatted_time"] = df["dt"].dt.strftime("%Y%m%d %H:%M")
    
    formatted_symbol = format_symbol(symbol)
    # Per instructions: data/cryptofuture/binance/margin_interest/
    base_dir = os.path.join(DATA_LOCATION, "cryptofuture", "binance", "margin_interest")
    os.makedirs(base_dir, exist_ok=True)
    
    zip_path = os.path.join(base_dir, f"{formatted_symbol}.zip")
    csv_filename = f"{formatted_symbol}.csv"
    
    # LEAN Margin Rate file: Time, Interest Rate
    lean_df = df[["formatted_time", "rate"]]
    csv_content = lean_df.to_csv(index=False, header=False)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_filename, csv_content)
    logger.debug(f"Saved margin interest {zip_path}")