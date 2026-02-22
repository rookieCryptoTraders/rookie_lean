import os
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import tempfile
import shutil

# Import from the package
from fetcher import load_depth_data_range, save_depth_data,load_depth_data
# import fetcher as fetcher


@pytest.fixture
def temp_data_dir():
    temp_dir = tempfile.mkdtemp()
    # Mock DATA_LOCATION in the fetcher module
    with patch("ccxt_data_fetch.fetcher.DATA_LOCATION", temp_dir):
        yield temp_dir
    shutil.rmtree(temp_dir)


def create_sample_depth_df(date_obj, ms_list, percentages):
    rows = []
    for ms in ms_list:
        for p in percentages:
            rows.append(
                {
                    "timestamp": date_obj + timedelta(milliseconds=ms),
                    "percentage": p,
                    "depth": np.random.rand() * 100,
                    "notional": np.random.rand() * 1000,
                }
            )
    return pd.DataFrame(rows)


def test_load_depth_data_basic(temp_data_dir):
    symbol = "BTCUSDT".lower()
    date_str = "20260101"
    dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
    
    # Load data with minute resolution
    # Resolution "minute" (60000ms) should filter ms_midnight % 60000 == 0
    # Expected ms: 0, 60000. Each has 2 percentages. Total 4 rows.
    # loaded_df = load_depth_data(symbol, "minute", dt, dt, pivot=False,ffill=False,align_ms= None,fill_first=True,asset_class="x",exchange='binance')
    # loaded_df = load_depth_data(symbol, "minute", dt, dt, pivot=False,ffill=False,align_ms= None,fill_first=True,asset_class="cryptofuture",exchange='x')
    
    # loaded_df = load_depth_data(symbol, "minute", dt, dt, pivot=False,ffill=False,align_ms= None,fill_first=True,asset_class="cryptofuture",exchange='binance')
    loaded_df = load_depth_data_range(symbol, "minute", dt, dt, pivot=True,ffill=False,align_ms= None,fill_first=True,asset_class="cryptofuture",exchange='binance')
    loaded_df = load_depth_data_range(symbol, "minute", dt, dt, pivot=False,ffill=True,align_ms= None,fill_first=True,asset_class="cryptofuture",exchange='binance')
    loaded_df = load_depth_data_range(symbol, "minute", dt, dt, pivot=False,ffill=False,align_ms= None,fill_first=False,asset_class="cryptofuture",exchange='binance')






def test_load_depth_data_fill_first(temp_data_dir):
    symbol = "BTCUSDT".lower()
    prev_date_str = "20260209"
    curr_date_str = "20260101"
    prev_dt = datetime.strptime(prev_date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
    curr_dt = datetime.strptime(curr_date_str, "%Y%m%d").replace(tzinfo=timezone.utc)

    # Save previous day's data with a last snapshot at 23:59:30
    last_ms = 86400000 - 30000
    prev_df = create_sample_depth_df(prev_dt, [last_ms], [0.1])
    save_depth_data(symbol, prev_date_str, prev_df)

    # Save current day's data starting from 00:00:30 (missing 00:00:00)
    curr_df = create_sample_depth_df(curr_dt, [30000], [0.1])
    save_depth_data(symbol, curr_date_str, curr_df)

    # Load with fill_first=True, resolution minute
    loaded_df = load_depth_data_range(symbol, "minute", curr_dt, curr_dt, fill_first=True)

    # Resolution minute will look for ms=0.
    # ms=0 should be filled from prev_dt's last snapshot (ms=last_ms).
    assert any(loaded_df["ms_midnight"] == 0)
    val_at_0 = loaded_df[loaded_df["ms_midnight"] == 0]["depth"].iloc[0]
    expected_val = prev_df["depth"].iloc[0]
    assert val_at_0 == pytest.approx(expected_val)


def test_load_depth_data_align_pivot_ffill(temp_data_dir):
    symbol = "BTCUSDT".lower()
    date_str = "20260101"
    dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)

    # Create data with snapshots at 0 and 120000, missing 60000
    df = create_sample_depth_df(dt, [0, 120000], [0.1, 0.2])
    save_depth_data(symbol, date_str, df)

    # Load with align_ms=60000, pivot=True, ffill=True
    loaded_df = load_depth_data_range(
        symbol,
        "minute",
        dt,
        dt,
        align_ms=60000,
        pivot=True,
        ffill=True,
        fill_first=False,
    )

    # Expected index: (date, ms_midnight)
    # 86400000 / 60000 = 1440 points
    assert len(loaded_df) == 1440

    # Snapshot at 60000 should be forward filled from 0
    # The index is (Timestamp, ms_midnight), usually naive as created by pd.to_datetime(date)
    dt_index = pd.to_datetime(date_str)
    assert (dt_index, 60000) in loaded_df.index
    assert not np.isnan(loaded_df.loc[(dt_index, 60000), 0.1])
    assert loaded_df.loc[(dt_index, 60000), 0.1] == loaded_df.loc[(dt_index, 0), 0.1]


def test_load_depth_data_no_data(temp_data_dir):
    symbol = "NONEXISTENT"
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    loaded_df = load_depth_data_range(symbol, "minute", dt, dt)
    assert loaded_df.empty
    
def test_load_depth_data():
    symbol = "BTCUSDT".lower()
    date_str = "20260101"

    # Load with align_ms=60000, pivot=True, ffill=True
    # loaded_df = load_depth_data(symbol,date_str,align_ms=None,pivot=False,ffill=False,fill_first=False)
    # loaded_df = load_depth_data(symbol,date_str,align_ms=None,pivot=False,ffill=True,fill_first=False)
    # loaded_df = load_depth_data(symbol,date_str,align_ms=60*1000,pivot=False,ffill=False,fill_first=True)
    # loaded_df = load_depth_data(symbol,date_str,align_ms=60*1000,pivot=False,ffill=True,fill_first=False)
    # loaded_df = load_depth_data(symbol,date_str,align_ms=60*1000,pivot=False,ffill=True,fill_first=True)
    loaded_df = load_depth_data(symbol,date_str,align_ms=60*1000,pivot=True,ffill=True,fill_first=True)
    print(loaded_df)

if __name__ == "__main__":
    # test_load_depth_data_basic(temp_data_dir=temp_data_dir)
    # test_load_depth_data_fill_first(temp_data_dir=temp_data_dir)
    # test_load_depth_data_align_pivot_ffill(temp_data_dir=temp_data_dir)
    # test_load_depth_data_no_data(temp_data_dir=temp_data_dir)
    test_load_depth_data()