"""
Backtest entry point for the Crypto Funding Reversal strategy.

Usage:
  python run_backtest.py                          # real data, all symbols
  python run_backtest.py --symbol BTCUSDT         # single symbol
  python run_backtest.py --synthetic              # synthetic AR(1) data, no API
  python run_backtest.py --use-cache              # load parquet cache if available
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from strategy import FundingReversalStrategy


def main():
    parser = argparse.ArgumentParser(description="Run Crypto Funding Reversal backtest")
    parser.add_argument("--symbol",    default=None,        help="Single symbol (default: all in config)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (no API)")
    parser.add_argument("--use-cache", action="store_true", help="Load parquet cache if available")
    args = parser.parse_args()

    strategy = FundingReversalStrategy()
    strategy.run_backtest(
        symbol    = args.symbol,
        use_cache = args.use_cache,
        synthetic = args.synthetic,
    )


if __name__ == "__main__":
    main()
