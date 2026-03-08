"""
Entrypoint: run OHLCV fetch or margin interest fetch.

  OHLCV:       python -m ccxt_data_fetch.run <asset_class> <resolution> [tick_type] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
  Margin rate: python -m ccxt_data_fetch.run cryptofuture margin_interest [--start YYYY-MM-DD] [--end YYYY-MM-DD]

For OHLCV-only or margin-interest-only, use:
  python -m ccxt_data_fetch.run_ohlcv [asset_class] [resolution] [tick_type] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
  python -m ccxt_data_fetch.run_margin_interest [asset_class] [--start YYYY-MM-DD] [--end YYYY-MM-DD]

example:
python -m ccxt_data_fetch.run cryptofuture minute trade
python -m ccxt_data_fetch.run cryptofuture minute quote
python -m ccxt_data_fetch.run cryptofuture minute margin_interest
python -m ccxt_data_fetch.run_margin_interest cryptofuture
"""
import sys

from ccxt_data_fetch.run_ohlcv import run_fetch_ohlcv
from ccxt_data_fetch.run_margin_interest import run_fetch_margin_interest


def main() -> None:
    # Parse --start/--end so they are not treated as positionals
    argv = sys.argv[1:]
    start = None
    end = None
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--start" and i + 1 < len(argv):
            start = argv[i + 1]
            i += 2
            continue
        if argv[i] == "--end" and i + 1 < len(argv):
            end = argv[i + 1]
            i += 2
            continue
        rest.append(argv[i])
        i += 1

    asset_class = rest[0] if len(rest) >= 1 else "cryptofuture"
    resolution = rest[1] if len(rest) >= 2 else "minute"
    tick_type = rest[2] if len(rest) >= 3 else "trade"

    if resolution == "margin_interest":
        run_fetch_margin_interest(asset_class, start_date=start, end_date=end)
    else:
        run_fetch_ohlcv(asset_class, resolution, tick_type, start_date=start, end_date=end)


if __name__ == "__main__":
    main()
