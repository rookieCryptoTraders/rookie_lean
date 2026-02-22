import ccxt
import pandas as pd
import numpy as np
import os
import zipfile
import time
import logging
import requests
import io
from datetime import date, datetime, timezone, timedelta
from tqdm import tqdm
from config import DATA_LOCATION, PROXIES
from utils import format_symbol, get_ms_from_midnight

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

def fetch_depth_range_cryptofuture(symbol, since_ms, until_ms, margin_type="um", data_source="binancevision"):
    """fetch order book snapshots for crypto futures from [BinanceVision](https://data.binance.vision/?prefix=data/futures/margin_type/daily/bookDepth/)
    depth data has no resolution parameter, and is always about 30s interval snapshots. you should align the timestamps to the nearest next minute boundary (e.g., 60s) to ensure consistency, and fill missing snapshots at the start of the day if needed to ensure continuity at midnight.
    
    :return: (pd.DataFrame)
    """

    if data_source != "binancevision":
        raise NotImplementedError("Only binancevision data source is supported for depth data.")

    all_dfs_dict = {}
    processed_dates = set()
    current_since = since_ms - 24*60*60*1000 # start from the previous day to fill the 0:00 snapshot of since_ms
        
    pbar = tqdm(total=until_ms - since_ms, desc=f"Fetching Depth {symbol}")

    while current_since < until_ms:
        dt = datetime.fromtimestamp(current_since / 1000, tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
        date_str_yyyymmdd = dt.strftime("%Y%m%d")
        
        if date_str in processed_dates:
            next_day = (dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
            current_since = int(next_day.timestamp() * 1000)
            continue
            
        formatted_symbol = format_symbol(symbol).upper()
        url = f"https://data.binance.vision/data/futures/{margin_type}/daily/bookDepth/{formatted_symbol}/{formatted_symbol}-bookDepth-{date_str}.zip"
        
        try:
            response = requests.get(url, proxies=PROXIES, timeout=30)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as f:
                        df = pd.read_csv(f, low_memory=False)
                        all_dfs_dict[date_str_yyyymmdd] = df
                        
            elif response.status_code == 404:
                logger.debug(f"No depth data for {symbol} on {date_str} (404)")
            else:
                logger.warning(f"Failed to download depth data for {symbol} on {date_str}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching depth data for {symbol} on {date_str}: {e}")
            
        processed_dates.add(date_str)
        next_day = (dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
        new_since = int(next_day.timestamp() * 1000)
        
        progress = new_since - current_since
        if progress > 0:
            pbar.update(min(progress, until_ms - current_since))
        current_since = new_since

    pbar.close()
    if not all_dfs_dict:
        return {}
    return all_dfs_dict


def save_depth_data(symbol,date_str:str, depth_data:pd.DataFrame, asset_class="cryptofuture"):
    """
    Save depth data to LEAN format.
    Format: ms_midnight, percentage, depth, notional
    """
    if depth_data.empty:
        return
    df = pd.DataFrame(depth_data)

    formatted_symbol = format_symbol(symbol)
    symbol_dir = os.path.join(
        DATA_LOCATION, asset_class, "binance", "minute", formatted_symbol
    )
    os.makedirs(symbol_dir, exist_ok=True)

    # LEAN depth format: ms_midnight, percentage, depth, notional
    df["timestamp"]=pd.to_datetime(df["timestamp"], utc=True)
    df["ms_midnight"] = df["timestamp"].apply(get_ms_from_midnight)

    # Filename: YYYYMMDD_depth.zip
    zip_path = os.path.join(symbol_dir, f"{date_str}_depth.zip")
    zf_name = f"{date_str}_depth.csv"
    
    lean_df = df[["ms_midnight", "percentage", "depth", "notional"]]
    csv_content = lean_df.to_csv(index=False, header=False)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zf_name, csv_content)

def load_depth_data(symbol: str, date_str: str, pivot=False, ffill=False, align_ms=None, fill_first=True, asset_class="cryptofuture", exchange='binance', resolution="minute"):
    """
    Load depth data for a single day.
    
    Args:
        symbol (str): Symbol name (e.g., BTCUSDT)
        date_str (str): Date string in YYYYMMDD format.
        pivot (bool): Whether to pivot the data. Defaults to False.
        ffill (bool): Whether to forward fill missing data. Defaults to False.
        align_ms (int): Alignment interval in milliseconds. Defaults to None.
        fill_first (bool): Whether to fill the first snapshot with previous day's last. Defaults to True.
        asset_class (str): Asset class directory. Defaults to "cryptofuture".
        exchange (str): Exchange directory. Defaults to 'binance'.
    """
    assert resolution == "minute", "Currently only minute resolution is supported for depth data loading."
    def _load_day_file(d_str):
        formatted_symbol = format_symbol(symbol)
        symbol_dir = os.path.join(DATA_LOCATION, asset_class, exchange, "minute", formatted_symbol)
        zip_path = os.path.join(symbol_dir, f"{d_str}_depth.zip")
        
        if not os.path.exists(zip_path):
            # Attempt to download
            if exchange == 'binance' and asset_class == 'cryptofuture':
                try:
                    dt_obj = datetime.strptime(d_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                    since_ms = int(dt_obj.timestamp() * 1000)
                    until_ms = since_ms + 86400000
                    depth_data_dict = fetch_depth_range_cryptofuture(symbol, since_ms, until_ms)
                    if depth_data_dict:
                        for day_str, day_df in depth_data_dict.items():
                            save_depth_data(symbol, day_str, day_df, asset_class=asset_class)
                    
                    if d_str not in (depth_data_dict or {}):
                        return pd.DataFrame()
                except Exception as e:
                    logger.error(f"Error downloading depth data for {symbol} on {d_str}: {e}")
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
                
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                csv_name = f"{d_str}_depth.csv"
                if csv_name in zf.namelist():
                    with zf.open(csv_name) as f:
                        df=pd.read_csv(f, names=["ms_midnight", "percentage", "depth", "notional"])
                        # df['ms_original'] = df['ms_midnight']+datetime.strptime(d_str, "%Y%m%d").timestamp()*1000
                        return df
        except Exception as e:
            logger.error(f"Error reading depth data file {zip_path}: {e}")
            return pd.DataFrame()
        return pd.DataFrame()

    df = _load_day_file(date_str)
    
    if fill_first:
        prev_date = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        prev_df = _load_day_file(prev_date)
        if not prev_df.empty:
            last_ms = prev_df['ms_midnight'].max()
            last_snapshot = prev_df[prev_df['ms_midnight'] == last_ms].copy()
            last_snapshot['ms_midnight'] = 0
            df = pd.concat([last_snapshot, df], ignore_index=True)

    if df.empty:
        return pd.DataFrame()

    # Initial sort and drop duplicates
    df = df.sort_values(['ms_midnight', 'percentage']).drop_duplicates(['ms_midnight', 'percentage'], keep='last')

    if align_ms is not None:
        # Align to the nearest NEXT boundary (ceil)
        df['ms_midnight'] = (np.ceil(df['ms_midnight'] / align_ms) * align_ms).astype(int)
        df = df[df['ms_midnight'] < 86400000]
        df = df.sort_values(['ms_midnight', 'percentage']).drop_duplicates(['ms_midnight', 'percentage'], keep='last')
        
        ms_range = np.arange(0, 86400000, align_ms)
        if pivot:
            df_depth = df.pivot(index='ms_midnight', columns='percentage', values='depth')
            df_depth = df_depth.reindex(ms_range)
            df_depth.columns = [f"depth_{p}%" for p in df_depth.columns]
            df_notional = df.pivot(index='ms_midnight', columns='percentage', values='notional')
            df_notional = df_notional.reindex(ms_range)
            df_notional.columns = [f"notional_{p}%" for p in df_notional.columns]
            df = pd.concat([df_depth,df_notional],axis=1,join='outer')
        else:
            percentages = df['percentage'].unique()
            new_index = pd.MultiIndex.from_product([ms_range, percentages], names=['ms_midnight', 'percentage'])
            df = df.set_index(['ms_midnight', 'percentage']).reindex(new_index).reset_index()
    elif pivot:
        df_depth = df.pivot(index='ms_midnight', columns='percentage', values='depth')
        df_depth.columns = [f"depth_{p}%" for p in df_depth.columns]
        df_notional = df.pivot(index='ms_midnight', columns='percentage', values='notional')
        df_notional.columns = [f"notional_{p}%" for p in df_notional.columns]
        df = pd.concat([df_depth,df_notional],axis=1,join='outer')

    if ffill:
        if pivot:
            df = df.ffill()
        else:
            df = df.sort_values(['percentage', 'ms_midnight'])
            df[['depth', 'notional']] = df.groupby('percentage')[['depth', 'notional']].ffill()
            df = df.sort_values(['ms_midnight', 'percentage'])


    df = df.reset_index().set_index('ms_midnight')
        
    return df

def load_depth_data_range(symbol:str,resolution:str, start:datetime | int, end:datetime | int, pivot=False, ffill=False, align_ms=None, fill_first=True, asset_class="cryptofuture", exchange='binance'):
    """load depth data from zip wrapped csv data

    Args:
        symbol (str): Symbol name (e.g., BTCUSDT)
        resolution (str): Data resolution ('minute', 'hour', 'daily')
        start (datetime | int): Start timestamp or datetime
        end (datetime | int): End timestamp or datetime
        pivot (bool, optional): Whether to pivot the data. Defaults to False.
        ffill (bool, optional): Whether to forward fill missing data. Defaults to False.
        align_ms (int, optional): Alignment interval in milliseconds. Defaults to None.
        fill_first (bool, optional): Whether to fill the first snapshot with previous day's last. Defaults to True.
        asset_class (str, optional): Asset class directory. Defaults to "cryptofuture".
        exchange (str, optional): Exchange directory. Defaults to 'binance'.
    """
    if isinstance(start, (int, np.integer)):
        start = datetime.fromtimestamp(start / 1000, tz=timezone.utc)
    if isinstance(end, (int, np.integer)):
        end = datetime.fromtimestamp(end / 1000, tz=timezone.utc)

    formatted_symbol = format_symbol(symbol)
    start_date = start.date()
    end_date = end.date()
    
    all_dfs = []
    # Load from previous day if fill_first is True to get the last snapshot
    load_start = start_date - timedelta(days=1) if fill_first else start_date
    
    curr = load_start
    counter=0
    max_tries=3
    while curr <= end_date and counter<max_tries:
        date_str = curr.strftime("%Y%m%d")
        symbol_dir = os.path.join(DATA_LOCATION, asset_class, exchange, resolution, formatted_symbol)
        zip_path = os.path.join(symbol_dir, f"{date_str}_depth.zip")
        
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zf:
                csv_name = f"{date_str}_depth.csv"
                if csv_name in zf.namelist():
                    with zf.open(csv_name) as f:
                        df = pd.read_csv(f, names=["ms_midnight", "percentage", "depth", "notional"])
                        df['date'] = pd.to_datetime(curr)
                        all_dfs.append(df)
            curr += timedelta(days=1)
            
        else:
            logger.debug(f"Depth data file not found: {zip_path}, trying to download {(counter+1)}/{max_tries}")
            # download data
            if exchange == 'binance' and asset_class == 'cryptofuture':
                since_ms = int(datetime.combine(curr, datetime.min.time()).timestamp() * 1000)
                until_ms = int(datetime.combine(curr + timedelta(days=1), datetime.min.time()).timestamp() * 1000)
                depth_data_dict = fetch_depth_range_cryptofuture(symbol, since_ms, until_ms, margin_type="um", data_source="binancevision")
                if depth_data_dict and curr.strftime("%Y%m%d") in depth_data_dict:
                    save_depth_data(symbol, curr.strftime("%Y%m%d"), depth_data_dict[curr.strftime("%Y%m%d")], asset_class=asset_class)
            counter+=1
        
    if not all_dfs:
        return pd.DataFrame()
        
    df:pd.DataFrame = pd.concat(all_dfs, ignore_index=True)
    
    # 0. if fill_first, fill the first datetime with the end of last day's last snapshot
    if fill_first:
        prev_day_mask = df['date'].dt.date < start_date
        if prev_day_mask.any():
            prev_day_data = df[prev_day_mask]
            last_ms = prev_day_data['ms_midnight'].max()
            last_snapshot = prev_day_data[prev_day_data['ms_midnight'] == last_ms].copy()
            last_snapshot['date'] = pd.to_datetime(start_date)
            last_snapshot['ms_midnight'] = 0
            df = pd.concat([last_snapshot, df[df['date'].dt.date >= start_date]], ignore_index=True)
        else:
            df = df[df['date'].dt.date >= start_date]
    else:
        df = df[df['date'].dt.date >= start_date]

    # 1. if align_ms is provided, align data with an index dataframe with align_ms as index
    if align_ms is not None:
        dates = df['date'].unique()
        ms_range = np.arange(0, 86400000, align_ms)
        if pivot:
            df = df.pivot_table(index=['date', 'ms_midnight'], columns='percentage', values='depth')
            new_index = pd.MultiIndex.from_product([dates, ms_range], names=['date', 'ms_midnight'])
            df = df.reindex(new_index)
        else:
            percentages = df['percentage'].unique()
            new_index = pd.MultiIndex.from_product([dates, ms_range, percentages], names=['date', 'ms_midnight', 'percentage'])
            df = df.set_index(['date', 'ms_midnight', 'percentage']).reindex(new_index).reset_index()

    # 2. if pivot, pivot with index=ms_midnight, columns=percentage, values=depth
    if pivot and not isinstance(df.index, pd.MultiIndex):
        df = df.pivot_table(index=['date', 'ms_midnight'], columns='percentage', values='depth')

    # 3. if ffill and not pivot, ffill data with groupby('percentage'); if ffill and pivot, then ffill with groupby(level=0)
    if ffill:
        if pivot:
            df = df.groupby(level=0).ffill()
        else:
            df = df.sort_values(['percentage', 'date', 'ms_midnight'])
            df[['depth', 'notional']] = df.groupby('percentage')[['depth', 'notional']].ffill()

    # 4. only keep resolution aligned data (e.g., if resolution is minute, only keep data with ms_midnight aligned to minute boundary)
    res_map = {"minute": 60000, "hour": 3600000, "daily": 86400000}
    target_ms = res_map.get(resolution)
    if target_ms:
        if pivot:
            df = df[df.index.get_level_values('ms_midnight') % target_ms == 0]
        else:
            df = df[df['ms_midnight'] % target_ms == 0]
            
    return df


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


def save_ohlcv_data(symbol, ohlcv_data, resolution, asset_class="cryptofuture", tick_type="trade"):
    """
    Unified function to save OHLCV data to LEAN format for any resolution.
    """
    assert resolution in ["minute", "hour", "daily"], "Unsupported resolution for OHLCV saving"
    if not ohlcv_data:
        return
        
    df = pd.DataFrame(
        ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    formatted_symbol = format_symbol(symbol)
    base_dir = os.path.join(DATA_LOCATION, asset_class, "binance", resolution)
    
    if resolution in ["minute", "second", "tick"]:
        # Daily partitioned files in symbol subdirectory
        symbol_dir = os.path.join(base_dir, formatted_symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        df["date_str"] = df["dt"].dt.strftime("%Y%m%d")
        for date_str, group in df.groupby("date_str"):
            group = group.copy()
            group["time_val"] = group["dt"].apply(get_ms_from_midnight)
            lean_df = get_lean_df(group, asset_class, tick_type, "time_val")

            zip_path = os.path.join(symbol_dir, f"{date_str}_{tick_type}.zip")
            zf_name = f"{date_str}_{tick_type}.csv"
            csv_content = lean_df.to_csv(index=False, header=False)

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(zf_name, csv_content)
    else:
        # Consolidated files for Hour and Daily
        os.makedirs(base_dir, exist_ok=True)
        
        if resolution == "daily":
            df["time_val"] = df["dt"].dt.strftime("%Y%m%d 00:00")
            zip_path = os.path.join(base_dir, f"{formatted_symbol}_{tick_type}.zip")
        else: # hour
            df["time_val"] = df["dt"].dt.strftime("%Y%m%d %H:%M")
            primary_tick = "quote" if asset_class in ["forex", "cfd"] else "trade"
            suffix = f"_{tick_type}" if tick_type != primary_tick else ""
            zip_path = os.path.join(base_dir, f"{formatted_symbol}{suffix}.zip")
            
        csv_filename = f"{formatted_symbol}.csv"
        lean_df = get_lean_df(df, asset_class, tick_type, "time_val")
        csv_content = lean_df.to_csv(index=False, header=False)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(csv_filename, csv_content)
    
    logger.debug(f"Saved {resolution} data for {symbol}")



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