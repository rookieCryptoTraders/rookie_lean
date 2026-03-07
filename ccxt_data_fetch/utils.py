import logging
import time
import requests
import ccxt
from ccxt_data_fetch.config import PROXIES

logger = logging.getLogger(__name__)

# Retry settings for Binance API (helps with proxy/SSL transient errors)
_BINANCE_LOAD_MARKETS_RETRIES = 5
_BINANCE_LOAD_MARKETS_RETRY_DELAY_SEC = 3
_BINANCE_REQUEST_TIMEOUT_MS = 30_000


def format_symbol(symbol):
    """Convert symbols like BTC/USDT or BTC/USDT:USDT to btcusdt."""
    return symbol.split(":")[0].replace("/", "").lower()


def get_ms_from_midnight(dt):
    """Calculate milliseconds since midnight UTC"""
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int((dt - midnight).total_seconds() * 1000)


def get_top_200_symbols(asset_class="cryptofuture"):
    logger.info("Fetching top 200 coins from CoinGecko...")
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1,
        "sparkline": False,
    }

    try:
        response = requests.get(url, params=params, proxies=PROXIES, timeout=15)
        response.raise_for_status()
        top_coins = response.json()
        top_symbols = [coin["symbol"].upper() for coin in top_coins]
    except Exception as e:
        logger.error(f"Failed to fetch from CoinGecko: {e}")
        return ["BTC/USDT", "ETH/USDT"]

    logger.info(f"Fetching available {asset_class} symbols from Binance...")

    options = {}
    if asset_class == "cryptofuture":
        options["defaultType"] = "future"

    exchange = ccxt.binance({
        "proxies": PROXIES,
        "options": options,
        "timeout": _BINANCE_REQUEST_TIMEOUT_MS,
    })
    last_error = None
    for attempt in range(1, _BINANCE_LOAD_MARKETS_RETRIES + 1):
        try:
            markets = exchange.load_markets()
            break
        except (requests.exceptions.SSLError, ccxt.NetworkError, ConnectionError) as e:
            last_error = e
            if attempt < _BINANCE_LOAD_MARKETS_RETRIES:
                logger.warning(
                    f"Binance load_markets attempt {attempt}/{_BINANCE_LOAD_MARKETS_RETRIES} failed ({e}). "
                    f"Retrying in {_BINANCE_LOAD_MARKETS_RETRY_DELAY_SEC}s..."
                )
                time.sleep(_BINANCE_LOAD_MARKETS_RETRY_DELAY_SEC)
            else:
                logger.error(f"Binance load_markets failed after {_BINANCE_LOAD_MARKETS_RETRIES} attempts.")
                raise last_error

    matched = []
    for coin_sym in top_symbols:
        for m in markets.values():
            if m["quote"] == "USDT" and m["base"] == coin_sym:
                if asset_class == "cryptofuture" and m["type"] == "swap":
                     matched.append(m["symbol"])
                     break
                elif asset_class == "crypto" and m["type"] == "spot":
                     matched.append(m["symbol"])
                     break
        if len(matched) >= 200:
            break

    return matched
