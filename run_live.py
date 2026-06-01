"""
Live execution entry point for Strategy 3 — Funding Rate Mean Reversion.

Usage:
  python run_live.py

Requires:
  .env file with BINANCE_API_KEY and BINANCE_API_SECRET
  API keys with USDT-margined futures trading permissions (not spot)
"""
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Load environment variables from .env
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
from data.websocket_feed import BinanceLiveFeed
from live.order_manager import OrderManager
from live.signal_monitor import SignalMonitor


async def main():
    logger.info("Strategy 3 Live Execution starting")
    logger.info("Symbols: %s", CONFIG.symbols)

    api_key    = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        logger.error("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env")
        sys.exit(1)

    order_manager  = OrderManager(cfg=CONFIG)
    signal_monitor = SignalMonitor(order_manager=order_manager, cfg=CONFIG)

    feed = BinanceLiveFeed(
        symbols             = CONFIG.symbols,
        cfg                 = CONFIG,
        on_settlement       = signal_monitor.on_settlement,
        on_mark_price_tick  = signal_monitor.on_mark_price_tick,
    )

    loop = asyncio.get_running_loop()

    def _graceful_shutdown(signum, frame):
        logger.info("Shutdown signal received — closing all positions")
        for symbol in CONFIG.symbols:
            try:
                order_manager.exit(symbol, reason="manual_shutdown")
            except Exception as exc:
                logger.error("Error closing %s: %s", symbol, exc)
        feed.stop()
        loop.stop()

    signal.signal(signal.SIGINT,  _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    logger.info("Connecting to Binance WebSocket streams…")
    await feed.start_async()


if __name__ == "__main__":
    asyncio.run(main())
