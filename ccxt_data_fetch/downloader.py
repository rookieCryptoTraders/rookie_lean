"""
ccxt_data_fetch Downloader Core
===============================
Unified downloader for all Binance Futures data types.

Supports:
- OHLCV Klines (via CCXT API)
- Metrics (OI, LS Ratio, Taker Ratio) from Binance Vision
- Index Price Klines from Binance Vision
- Mark Price Klines from Binance Vision
- Premium Index Klines from Binance Vision
"""

import os
import io
import time
import zipfile
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests
import pandas as pd
import ccxt

from .config import (
    BASE_DATA_PATH,
    PROXIES,
    REQUEST_TIMEOUT,
    MAX_WORKERS_FILES,
    MAX_WORKERS_TICKERS,
    START_DATE,
    END_DATE,
    DEFAULT_TIMEFRAME,
    TICKER_ALIASES,
    VALID_TIMEFRAMES,
)

logger = logging.getLogger(__name__)

# ============================================================================
# DATA TYPE DEFINITIONS
# ============================================================================

BINANCE_VISION_BASE = "https://data.binance.vision/data/futures/um/daily"


@dataclass
class DataTypeConfig:
    """Configuration for a data type."""

    name: str
    url_template: str
    raw_subdir: str
    final_subdir: str
    suffix: str
    is_kline: bool
    columns_map: Optional[Dict[str, str]] = None
    resample: bool = False


def get_data_types(timeframe: str) -> Dict[str, DataTypeConfig]:
    """Get data type configurations for a given timeframe."""
    return {
        "metrics": DataTypeConfig(
            name="metrics",
            url_template=f"{BINANCE_VISION_BASE}/metrics/{{ticker}}/{{ticker}}-metrics-{{date}}.zip",
            raw_subdir="raw_metrics",
            final_subdir="extra",
            suffix="_extra.csv",
            is_kline=False,
            columns_map={
                "create_time": "time",
                "sum_open_interest": "openInterest",
                "sum_toptrader_long_short_ratio": "ls_ratio",
                "sum_taker_long_short_vol_ratio": "taker_ratio",
            },
            resample=True,
        ),
        "indexPriceKlines": DataTypeConfig(
            name="indexPriceKlines",
            url_template=f"{BINANCE_VISION_BASE}/indexPriceKlines/{{ticker}}/{timeframe}/{{ticker}}-{timeframe}-{{date}}.zip",
            raw_subdir="raw_index",
            final_subdir="extra_index",
            suffix="_index.csv",
            is_kline=True,
        ),
        "markPriceKlines": DataTypeConfig(
            name="markPriceKlines",
            url_template=f"{BINANCE_VISION_BASE}/markPriceKlines/{{ticker}}/{timeframe}/{{ticker}}-{timeframe}-{{date}}.zip",
            raw_subdir="raw_mark",
            final_subdir="extra_mark",
            suffix="_mark.csv",
            is_kline=True,
        ),
        "premiumIndexKlines": DataTypeConfig(
            name="premiumIndexKlines",
            url_template=f"{BINANCE_VISION_BASE}/premiumIndexKlines/{{ticker}}/{timeframe}/{{ticker}}-{timeframe}-{{date}}.zip",
            raw_subdir="raw_premium",
            final_subdir="extra_premium",
            suffix="_premium.csv",
            is_kline=True,
        ),
    }


# ============================================================================
# DOWNLOADER CLASS
# ============================================================================


class BinanceDataDownloader:
    """
    Unified downloader for Binance Futures data.

    Usage:
        downloader = BinanceDataDownloader(timeframe="1h")
        downloader.download(["BTCUSDT", "ETHUSDT"])
    """

    def __init__(
        self,
        timeframe: str = DEFAULT_TIMEFRAME,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        include_ohlcv: bool = True,
        include_metrics: bool = True,
        include_index: bool = True,
        include_mark: bool = True,
        include_premium: bool = True,
    ):
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. Must be one of {VALID_TIMEFRAMES}"
            )

        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date

        # Data types to download
        self.include_ohlcv = include_ohlcv
        self.include_metrics = include_metrics
        self.include_index = include_index
        self.include_mark = include_mark
        self.include_premium = include_premium

        # Build date list
        self.dates = self._build_date_list()

        # Data type configs
        self.data_types = get_data_types(timeframe)

        # CCXT exchange (lazy init)
        self._exchange = None

    @property
    def exchange(self) -> ccxt.binance:
        """Lazy-initialized CCXT exchange."""
        if self._exchange is None:
            self._exchange = ccxt.binance(
                {
                    "enableRateLimit": True,
                    "proxies": PROXIES,
                    "options": {"defaultType": "future"},
                }
            )
        return self._exchange

    def _build_date_list(self) -> List[str]:
        """Build list of dates in the range."""
        dates = []
        curr = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        while curr <= end:
            dates.append(curr.strftime("%Y-%m-%d"))
            curr += timedelta(days=1)
        return dates

    def _get_internal_name(self, ticker: str) -> str:
        """Get internal (file) name for a ticker."""
        return TICKER_ALIASES.get(ticker, ticker.lower())

    def _download_file(self, url: str, save_dir: str) -> str:
        """Download and extract a single ZIP file."""
        try:
            r = requests.get(url, proxies=PROXIES, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    z.extractall(save_dir)
                return "OK"
            elif r.status_code == 404:
                return "404"
            return f"ERR_{r.status_code}"
        except Exception as e:
            return f"EXC_{type(e).__name__}"

    def _sync_vision_data(self, ticker: str, dtype_key: str) -> Dict[str, int]:
        """Sync a single data type from Binance Vision."""
        config = self.data_types[dtype_key]
        raw_dir = os.path.join(BASE_DATA_PATH, config.raw_subdir, ticker)
        os.makedirs(raw_dir, exist_ok=True)

        # Determine expected filename pattern
        def get_filename(date_str):
            if config.is_kline:
                return f"{ticker}-{self.timeframe}-{date_str}.csv"
            return f"{ticker}-metrics-{date_str}.csv"

        # Find dates that need downloading
        existing = set(os.listdir(raw_dir)) if os.path.exists(raw_dir) else set()
        dates_to_download = [d for d in self.dates if get_filename(d) not in existing]

        results = {
            "OK": 0,
            "EXISTS": len(self.dates) - len(dates_to_download),
            "404": 0,
            "ERR": 0,
        }

        if not dates_to_download:
            return results

        def do_download(date_str):
            url = config.url_template.format(ticker=ticker, date=date_str)
            return self._download_file(url, raw_dir)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS_FILES) as executor:
            futures = {executor.submit(do_download, d): d for d in dates_to_download}
            for future in as_completed(futures):
                res = future.result()
                if res == "OK":
                    results["OK"] += 1
                elif res == "404":
                    results["404"] += 1
                else:
                    results["ERR"] += 1

        return results

    def _merge_vision_data(self, ticker: str, dtype_key: str) -> Optional[str]:
        """Merge daily CSVs into a single file."""
        config = self.data_types[dtype_key]
        raw_dir = os.path.join(BASE_DATA_PATH, config.raw_subdir, ticker)
        final_dir = os.path.join(BASE_DATA_PATH, config.final_subdir)
        os.makedirs(final_dir, exist_ok=True)

        if not os.path.exists(raw_dir):
            return None

        csvs = sorted([f for f in os.listdir(raw_dir) if f.endswith(".csv")])
        if not csvs:
            return None

        dfs = []
        for csv_file in csvs:
            try:
                df = pd.read_csv(os.path.join(raw_dir, csv_file))

                if config.is_kline:
                    tmp = pd.DataFrame()
                    tmp["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                    tmp["open"] = pd.to_numeric(df["open"], errors="coerce")
                    tmp["high"] = pd.to_numeric(df["high"], errors="coerce")
                    tmp["low"] = pd.to_numeric(df["low"], errors="coerce")
                    tmp["close"] = pd.to_numeric(df["close"], errors="coerce")
                    dfs.append(tmp)
                else:
                    cols = config.columns_map
                    if not all(c in df.columns for c in cols.keys()):
                        continue
                    tmp = pd.DataFrame()
                    for src, dst in cols.items():
                        if src == "create_time":
                            tmp[dst] = pd.to_datetime(df[src]).dt.tz_localize("UTC")
                        else:
                            tmp[dst] = df[src]
                    dfs.append(tmp)
            except Exception:
                continue

        if not dfs:
            return None

        combined = pd.concat(dfs).sort_values("time").drop_duplicates(subset=["time"])

        # Resample if needed
        if config.resample:
            combined = combined.set_index("time")
            combined = (
                combined.resample(self.timeframe).last().dropna(how="all").reset_index()
            )

        # Format time
        combined["time"] = combined["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Save
        internal_name = self._get_internal_name(ticker)
        out_path = os.path.join(final_dir, f"{internal_name}{config.suffix}")
        combined.to_csv(out_path, index=False)

        return out_path

    def _download_ohlcv(self, ticker: str) -> Optional[str]:
        """Download OHLCV klines via CCXT."""
        ccxt_symbol = ticker[:-4] + "/USDT:USDT"
        internal_name = self._get_internal_name(ticker)

        output_dir = os.path.join(BASE_DATA_PATH, self.timeframe)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{internal_name}.csv")

        # Skip if already exists and reasonably complete
        if os.path.exists(output_path):
            existing = pd.read_csv(output_path)
            if len(existing) > len(self.dates) * 20:  # Rough check
                return output_path

        since_ms = int(
            datetime.strptime(self.start_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )
        until_ms = int(
            datetime.strptime(self.end_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )

        all_ohlcv = []
        current = since_ms

        try:
            while current < until_ms:
                ohlcv = self.exchange.fetch_ohlcv(
                    ccxt_symbol, self.timeframe, since=current, limit=1500
                )
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                current = ohlcv[-1][0] + 1
                time.sleep(self.exchange.rateLimit / 1000)
        except Exception as e:
            logger.warning(f"OHLCV error for {ticker}: {e}")
            if not all_ohlcv:
                return None

        if not all_ohlcv:
            return None

        df = pd.DataFrame(
            all_ohlcv, columns=["time", "open", "high", "low", "close", "volume"]
        )
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df = df.sort_values("time").drop_duplicates(subset=["time"])
        df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        df.to_csv(output_path, index=False)
        return output_path

    def download_ticker(self, ticker: str) -> Dict[str, Any]:
        """Download all enabled data types for a single ticker."""
        results = {"ticker": ticker}

        # OHLCV
        if self.include_ohlcv:
            logger.info(f"  [{ticker}] OHLCV downloading...")
            path = self._download_ohlcv(ticker)
            results["ohlcv"] = path
            logger.info(
                f"  [{ticker}] OHLCV -> {os.path.basename(path) if path else 'FAILED'}"
            )

        # Vision data types
        vision_types = []
        if self.include_metrics:
            vision_types.append("metrics")
        if self.include_index:
            vision_types.append("indexPriceKlines")
        if self.include_mark:
            vision_types.append("markPriceKlines")
        if self.include_premium:
            vision_types.append("premiumIndexKlines")

        for dtype in vision_types:
            logger.info(f"  [{ticker}] {dtype} downloading...")
            sync_result = self._sync_vision_data(ticker, dtype)
            logger.info(
                f"  [{ticker}] {dtype} sync: OK={sync_result['OK']} EXISTS={sync_result['EXISTS']} 404={sync_result['404']}"
            )

            logger.info(f"  [{ticker}] {dtype} merging...")
            path = self._merge_vision_data(ticker, dtype)
            results[dtype] = path
            logger.info(
                f"  [{ticker}] {dtype} -> {os.path.basename(path) if path else 'FAILED'}"
            )

        return results

    def download(self, tickers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Download all data for multiple tickers.

        Args:
            tickers: List of tickers. If None, uses DEFAULT_TICKERS from config.

        Returns:
            List of result dicts, one per ticker.
        """
        from .config import DEFAULT_TICKERS

        if tickers is None:
            tickers = DEFAULT_TICKERS

        logger.info("=" * 60)
        logger.info(" Binance Data Downloader")
        logger.info("=" * 60)
        logger.info(
            f" Date range : {self.start_date} to {self.end_date} ({len(self.dates)} days)"
        )
        logger.info(f" Timeframe  : {self.timeframe}")
        logger.info(f" Tickers    : {len(tickers)}")
        logger.info(f" OHLCV      : {'Yes' if self.include_ohlcv else 'No'}")
        logger.info(f" Metrics    : {'Yes' if self.include_metrics else 'No'}")
        logger.info(f" Index      : {'Yes' if self.include_index else 'No'}")
        logger.info(f" Mark       : {'Yes' if self.include_mark else 'No'}")
        logger.info(f" Premium    : {'Yes' if self.include_premium else 'No'}")
        logger.info("=" * 60)

        all_results = []

        # Process tickers with limited parallelism
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_TICKERS) as executor:
            futures = {executor.submit(self.download_ticker, t): t for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {ticker}: {e}")
                    all_results.append({"ticker": ticker, "error": str(e)})

        logger.info("=" * 60)
        logger.info(" Download complete!")
        logger.info("=" * 60)

        return all_results
