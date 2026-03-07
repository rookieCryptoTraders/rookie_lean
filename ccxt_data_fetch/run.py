"""
Entrypoint: run OHLCV fetch or margin interest fetch.

  OHLCV:       python -m ccxt_data_fetch.run <asset_class> <resolution> [tick_type]
  Margin rate: python -m ccxt_data_fetch.run cryptofuture margin_interest

For OHLCV-only or margin-interest-only, use:
  python -m ccxt_data_fetch.run_ohlcv [asset_class] [resolution] [tick_type]
  python -m ccxt_data_fetch.run_margin_interest [asset_class]
  
example:
python -m ccxt_data_fetch.run cryptofuture minute trade
python -m ccxt_data_fetch.run cryptofuture minute quote
python -m ccxt_data_fetch.run cryptofuture minute depth
python -m ccxt_data_fetch.run cryptofuture minute margin_interest
python -m ccxt_data_fetch.run_margin_interest cryptofuture
"""
import sys

from ccxt_data_fetch.run_ohlcv import run_fetch_ohlcv
from ccxt_data_fetch.run_margin_interest import run_fetch_margin_interest


def main() -> None:
    args = sys.argv[1:]
    asset_class = args[0] if len(args) >= 1 else "cryptofuture"
    resolution = args[1] if len(args) >= 2 else "minute"
    tick_type = args[2] if len(args) >= 3 else "trade"

    if resolution == "margin_interest":
        run_fetch_margin_interest(asset_class)
    else:
        run_fetch_ohlcv(asset_class, resolution, tick_type)


if __name__ == "__main__":
    main()
