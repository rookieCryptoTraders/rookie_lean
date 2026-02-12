import os
import pandas as pd
from datetime import datetime
from settings import DATA_DIR

def load_trade_data(ticker, start_date, end_date, interval=None):
    """Efficiently load and concatenate minute data for a ticker, with optional resampling."""
    ticker_dir = os.path.join(DATA_DIR, ticker.lower())
    if not os.path.exists(ticker_dir):
        print(f"Data directory for {ticker_dir} does not exist.")
        return None
    else:
        print(f"Loading data for {ticker} from {ticker_dir}...")

    all_dfs = []
    files = sorted([f for f in os.listdir(ticker_dir) if f.endswith("_trade.zip")])
    for f in files:
        date_str = f.split("_")[0]
        try:
            file_date = datetime.strptime(date_str, "%Y%m%d")
            if start_date <= file_date <= end_date:
                df = pd.read_csv(
                    os.path.join(ticker_dir, f), header=None, compression="zip"
                )
                df.columns = ["ms", "open", "high", "low", "close", "volume"]
                df["time"] = file_date + pd.to_timedelta(df["ms"], unit="ms")
                df.set_index("time", inplace=True)
                all_dfs.append(df[["open", "high", "low", "close", "volume"]])
        except Exception as e:
            raise e

    if not all_dfs:
        return None

    df = pd.concat(all_dfs).sort_index().drop_duplicates()
    
    if interval:
        df = df.resample(interval).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        
    return df


def load_quote_data(ticker, start_date, end_date, interval=None):
    """Efficiently load and concatenate minute quote data for a ticker, with optional resampling."""
    ticker_dir = os.path.join(DATA_DIR, ticker.lower())
    if not os.path.exists(ticker_dir):
        print(f"Data directory for {ticker_dir} does not exist.")
        return None
    else:
        print(f"Loading quote data for {ticker} from {ticker_dir}...")

    all_dfs = []
    files = sorted([f for f in os.listdir(ticker_dir) if f.endswith("_quote.zip")])
    print(f"Found {len(files)} quote files.")
    for f in files:
        date_str = f.split("_")[0]
        file_date = datetime.strptime(date_str, "%Y%m%d")
        if start_date <= file_date <= end_date:
            
            df = pd.read_csv(
                os.path.join(ticker_dir, f), header=None, compression="zip"
            )
            df.columns = ["ms","bid_open", "bid_high", "bid_low", "bid_close", "bid_size", "ask_open", "ask_high", "ask_low", "ask_close", "ask_size"]
            df["time"] = file_date + pd.to_timedelta(df["ms"], unit="ms")
            df.set_index("time", inplace=True)
            all_dfs.append(
                df[["bid_open", "bid_high", "bid_low", "bid_close", "bid_size", "ask_open", "ask_high", "ask_low", "ask_close", "ask_size"]]
            )
    
    if not all_dfs:
        return None

    df = pd.concat(all_dfs).sort_index().drop_duplicates()

    if interval:
        df = df.resample(interval).agg({
            "bid_open": "first",
            "bid_high": "max",
            "bid_low": "min",
            "bid_close": "last",
            "bid_size": "sum",
            "ask_open": "first",
            "ask_high": "max",
            "ask_low": "min",
            "ask_close": "last",
            "ask_size": "sum"
        }).dropna()

    return df