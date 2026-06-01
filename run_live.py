"""
Live execution entry point for the Crypto Funding Reversal strategy.

Usage:
  python run_live.py

Requires:
  .env file with BINANCE_API_KEY and BINANCE_API_SECRET
  Binance account with USDT-margined futures trading permissions
"""
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "live.log"),
    ],
)
logger = logging.getLogger("run_live")

from config import CONFIG
from strategy import FundingReversalStrategy
from data.websocket_feed import BinanceLiveFeed


async def main():
    logger.info("Crypto Funding Reversal — Live Execution starting")
    logger.info("Symbols: %s", CONFIG.symbols)

    if not os.environ.get("BINANCE_API_KEY") or not os.environ.get("BINANCE_API_SECRET"):
        logger.error("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env")
        sys.exit(1)

    strategy = FundingReversalStrategy()
    trader   = strategy.create_live_trader()

    feed = BinanceLiveFeed(
        symbols            = CONFIG.symbols,
        cfg                = CONFIG,
        on_settlement      = trader.on_settlement,
        on_mark_price_tick = trader.on_mark_price_tick,
    )

    loop = asyncio.get_running_loop()

    def _graceful_shutdown(signum, frame):
        logger.info("Shutdown signal received — closing all positions")
        for sym in CONFIG.symbols:
            try:
                trader._exit(sym, reason="manual_shutdown")
            except Exception as exc:
                logger.error("Error closing %s: %s", sym, exc)
        feed.stop()
        loop.stop()

    signal.signal(signal.SIGINT,  _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    logger.info("Connecting to Binance WebSocket streams…")
    await feed.start_async()


if __name__ == "__main__":
    asyncio.run(main())
