import logging
import requests
import ccxt
from ccxt_data_fetch.config import PROXIES

logger = logging.getLogger(__name__)


def format_symbol(symbol):
    """Convert symbols like BTC/USDT or BTC/USDT:USDT to btcusdt."""
    return symbol.split(":")[0].replace("/", "").lower()


def get_ms_from_midnight(dt):
    """Calculate milliseconds since midnight UTC"""
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int((dt - midnight).total_seconds() * 1000)


def get_top_200_symbols():
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

    logger.info("Fetching available Futures symbols from Binance...")
    exchange = ccxt.binance({"proxies": PROXIES, "options": {"defaultType": "future"}})
    markets = exchange.load_markets()

    matched = []
    for coin_sym in top_symbols:
        for m in markets.values():
            if m["quote"] == "USDT" and m["type"] == "swap" and m["base"] == coin_sym:
                matched.append(m["symbol"])
                break
        if len(matched) >= 200:
            break

    return matched
