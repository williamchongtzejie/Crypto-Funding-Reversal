"""
Dashboard entry point for Strategy 3.
Loads backtest results from results/ and starts the Dash app.

Usage:
  python run_dashboard.py
  Open http://localhost:8050
"""
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("run_dashboard")

from dashboard.app import create_app

if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Strategy 3 dashboard at http://localhost:8050")
    logger.info("Run `python run_backtest.py` first to generate results.")
    app.run(host="127.0.0.1", port=8050, debug=False)
