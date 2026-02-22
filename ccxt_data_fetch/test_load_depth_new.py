import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import tempfile
import shutil
import os
import sys

# Add current directory to path so we can import fetcher
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the package
from fetcher import load_depth_data, save_depth_data

@pytest.fixture
def temp_data_dir():
    temp_dir = tempfile.mkdtemp()
    # Mock DATA_LOCATION in the fetcher module
    with patch("fetcher.DATA_LOCATION", temp_dir):
        yield temp_dir
    shutil.rmtree(temp_dir)

def test_load_depth_data_single_day(temp_data_dir):
    symbol = "BTCUSDT"
    date_str = "20260101"
    dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
    
    # Create sample data
    df = pd.DataFrame([
        {"timestamp": dt + timedelta(seconds=10), "percentage": -0.1, "depth": 100, "notional": 1000},
        {"timestamp": dt + timedelta(seconds=10), "percentage": 0.1, "depth": 110, "notional": 1100},
        {"timestamp": dt + timedelta(seconds=40), "percentage": -0.1, "depth": 105, "notional": 1050},
        {"timestamp": dt + timedelta(seconds=40), "percentage": 0.1, "depth": 115, "notional": 1150},
    ])
    save_depth_data(symbol, date_str, df)
    
    # Test basic load
    loaded = load_depth_data(symbol, date_str, pivot=False, fill_first=False)
    assert not loaded.empty
    assert loaded.index.name == "ms_midnight"
    assert len(loaded) == 4
    
    # Test pivot
    pivoted = load_depth_data(symbol, date_str, pivot=True, fill_first=False)
    assert "depth_-0.1%" in pivoted.columns
    assert "depth_0.1%" in pivoted.columns
    assert len(pivoted) == 2 # two unique ms_midnight
    
    # Test alignment (align_ms=60000)
    # Snapshots at 10s and 40s should both align to 60s boundary
    # We should keep the last one (40s)
    aligned = load_depth_data(symbol, date_str, pivot=True, align_ms=60000, fill_first=False)
    # ms_range for 60000 is [0, 60000, 120000, ...]
    assert 60000 in aligned.index
    assert aligned.loc[60000, "depth_-0.1%"] == 105
    assert np.isnan(aligned.loc[0, "depth_-0.1%"])
    
    # Test ffill
    ffilled = load_depth_data(symbol, date_str, pivot=True, align_ms=60000, ffill=True, fill_first=False)
    assert ffilled.loc[120000, "depth_-0.1%"] == 105
    assert not np.isnan(ffilled.loc[120000, "depth_-0.1%"])

def test_load_depth_data_fill_first_new(temp_data_dir):
    symbol = "BTCUSDT"
    prev_date = "20251231"
    curr_date = "20260101"
    prev_dt = datetime.strptime(prev_date, "%Y%m%d").replace(tzinfo=timezone.utc)
    curr_dt = datetime.strptime(curr_date, "%Y%m%d").replace(tzinfo=timezone.utc)
    
    # Prev day last snapshot at 23:59:50
    prev_df = pd.DataFrame([
        {"timestamp": prev_dt + timedelta(hours=23, minutes=59, seconds=50), "percentage": 0.1, "depth": 99, "notional": 990}
    ])
    save_depth_data(symbol, prev_date, prev_df)
    
    # Curr day first snapshot at 00:00:10
    curr_df = pd.DataFrame([
        {"timestamp": curr_dt + timedelta(seconds=10), "percentage": 0.1, "depth": 101, "notional": 1010}
    ])
    save_depth_data(symbol, curr_date, curr_df)
    
    # Load with fill_first=True
    loaded = load_depth_data(symbol, curr_date, pivot=False, fill_first=True)
    assert 0 in loaded.index
    # If there are multiple entries for index 0, check the first one
    val = loaded.loc[0, "depth"]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    assert val == 99

if __name__ == "__main__":
    pytest.main([__file__])
