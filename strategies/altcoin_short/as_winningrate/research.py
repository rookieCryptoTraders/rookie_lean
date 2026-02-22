# %% [markdown]
# ![QuantConnect Logo](https://cdn.quantconnect.com/web/i/icon.png)
# <hr>

# %%
# autoload newest code from files
# %load_ext autoreload
# %autoreload 2


# %%

import numpy as np
import pandas as pd
from datetime import timedelta,datetime
from utils import load_trade_data, load_quote_data
from settings import DATA_DIR, START_DATE, END_DATE, TICKERS


# %%
from datetime import datetime, timedelta

# --- Configuration ---
DATA_DIR = "../../../data/cryptofuture/binance/minute"
START_DATE = datetime(2026, 2, 3)  # Training: 2025 Full Year
END_DATE = datetime(2026, 2, 6)  # OOT Test: 2026 Jan

TICKERS = [
    "btcusdt",
    "ethusdt",
    # "bnbusdt",
    # "solusdt",
    # "xrpusdt",
    # "dogeusdt",
    # "adausdt",
    # "avaxusdt",
    # "dotusdt",
    # "linkusdt",
    # "maticusdt",
    # "ltcusdt",
    # "uniusdt",
    # "atomusdt",
    # "etcusdt",
    # "filusdt",
    # "aptusdt",
    # "nearusdt",
    # "arbusdt",
    # "opusdt",
    # "injusdt",
    # "suiusdt",
    # "tiausdt",
    # "seiusdt",
    # "stxusdt",
    # "imxusdt",
    # "runeusdt",
    # "aaveusdt",
    # "mkrusdt",
    # "ldousdt",
]


# %%
load_trade_data("ETHUSDT", START_DATE, END_DATE)
"""
                 open     high      low    close    volume
time                                                             
2026-02-03 00:00:00  2345.56  2346.90  2344.00  2344.00  2253.409
2026-02-03 00:01:00  2344.01  2344.36  2339.00  2340.61  9033.385
"""

load_quote_data("btcusdt", START_DATE, END_DATE)
"""
             bid_open  bid_high  bid_low  bid_close  bid_size  \
time                                                                    
2026-02-03 00:00:00   78692.5   78752.7  78668.9    78668.9         0   
2026-02-03 00:01:00   78668.8   78688.8  78630.0    78650.1         0   

      ask_open  ask_high  ask_low  ask_close  ask_size  
time                                                                   
2026-02-03 00:00:00   78692.5   78752.7  78668.9    78668.9         0  
2026-02-03 00:01:00   78668.8   78688.8  78630.0    78650.1         0  
"""

# %%
import pandas as pd
import numpy as np

symbol="btcusdt"

def calculate_features(symbol):
    # 1. Load Minute Data
    df_trade = load_trade_data(symbol, START_DATE, END_DATE, interval='1min')
    df_quote = load_quote_data(symbol, START_DATE, END_DATE, interval='1min')
    
    if df_trade is None or df_quote is None:
        return None
        
    # Merge trade and quote data
    df = pd.concat([df_trade, df_quote], axis=1).dropna()

    # 2. Minute-level Basic Calculations
    df['return'] = df['close'].pct_change()
    df['spread'] = (df['ask_close'] - df['bid_close']) / (df['bid_close'] + 1e-9)
    df['imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-9)
    
    # ---------------------------------------------------------
    # 3. Aggregate into Hourly Features (Encoding Information)
    # ---------------------------------------------------------
    # We split the hour into two 30-min windows to encode "shape" (Trend/Reversal)
    def aggregate_subwindows(group):
        n = len(group)
        mid = n // 2
        first_half = group.iloc[:mid]
        second_half = group.iloc[mid:]
        
        res = {
            'ret_total': group['close'].iloc[-1] / group['open'].iloc[0] - 1,
            'ret_1st_half': first_half['close'].iloc[-1] / first_half['open'].iloc[0] - 1 if not first_half.empty else 0,
            'ret_2nd_half': second_half['close'].iloc[-1] / second_half['open'].iloc[0] - 1 if not second_half.empty else 0,
            'vol_total': group['volume'].sum(),
            'vol_ratio': group['volume'].sum() / (group['volume'].mean() * n + 1e-9),
            'std_total': group['return'].std(),
            'skew_total': group['return'].skew(),
            'spread_avg': group['spread'].mean(),
            'imbalance_avg': group['imbalance'].mean(),
            'high_low_range': (group['high'].max() - group['low'].min()) / group['close'].iloc[-1]
        }
        return pd.Series(res)

    hourly_df = df.resample('1H').apply(aggregate_subwindows)

    # 4. Memory Features (Lags)
    # Give the model context about what happened in the previous few hours
    for lag in [1, 2, 3]:
        hourly_df[f'ret_lag_{lag}'] = hourly_df['ret_total'].shift(lag)
        hourly_df[f'vol_lag_{lag}'] = hourly_df['vol_total'].shift(lag)

    # Hourly Technical Indicators
    hourly_df['rsi_14'] = calculate_rsi(hourly_df['ret_total']) # Using hourly returns for RSI context

    # ---------------------------------------------------------
    # 5. Target Variables (Next Hour Return)
    # ---------------------------------------------------------
    hourly_df['next_hour_return'] = hourly_df['ret_total'].shift(-1)
    
    # Calculate rolling volatility of hourly returns for dynamic labeling
    # 48-hour window provides a stable estimate of current market regime
    hourly_df['hourly_vol'] = hourly_df['ret_total'].rolling(window=48).std()

    # Define 5 classes using dynamic standard deviation thresholds
    def generate_dynamic_label(row:pd.DataFrame, n1=1.5, n2=0.5, n3=0.5, n4=1.5):
        """

        Args:
            row (pd.DataFrame): _description_
            n1 (float, optional): down-- threshold. Defaults to 1.5.
            n2 (float, optional): down- threshold. Defaults to 0.5.
            n3 (float, optional): up+ threshold. Defaults to 0.5.
            n4 (float, optional): up++ threshold. Defaults to 1.5.

        Returns:
            _type_: _description_
        """
        ret = row['next_hour_return']
        std = row['hourly_vol']
        
        if pd.isna(ret) or pd.isna(std) or std == 0: 
            return np.nan
            
        # Threshold multipliers
        n1, n2, n3, n4 = abs(n1), abs(n2), abs(n3), abs(n4)
        
        if ret > n4 * std: return 2
        elif ret > n3 * std: return 1
        elif ret < -n1 * std: return -2
        elif ret < -n2 * std: return -1
        else: return 0

    hourly_df['target_class'] = hourly_df.apply(generate_dynamic_label, axis=1)

    # ---------------------------------------------------------
    # 6. Cleanup
    # ---------------------------------------------------------
    df_clean = hourly_df.dropna()
    
    return df_clean

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# --- Main Research Workflow ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

final_df = calculate_features(symbol)

if final_df is not None:
    # Print Class Distribution to diagnose imbalance
    print("\nClass Distribution:")
    print(final_df['target_class'].value_counts(normalize=True).sort_index())

    # Select features for training
    feature_cols = [
        'ret_total', 'ret_1st_half', 'ret_2nd_half', 
        'vol_total', 'vol_ratio', 'std_total', 'skew_total',
        'spread_avg', 'imbalance_avg', 'high_low_range',
        'ret_lag_1', 'vol_lag_1', 'ret_lag_2', 'rsi_14'
    ]
    
    X = final_df[feature_cols]
    y = final_df['target_class']
    
    # Split data (Time series split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"\nTraining on {len(X_train)} hourly samples, testing on {len(X_test)} samples...")
    
    # class_weight='balanced_subsample' is generally better for Random Forest with imbalance
    model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=10, 
        random_state=42, 
        class_weight='balanced_subsample',
        min_samples_leaf=5
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\nModel Evaluation (Hourly Prediction):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report (Balanced Subsample):")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(importances.head(10))
else:
    print("Failed to load or process data.")

