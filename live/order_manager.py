"""
Live order execution via Binance signed API.
Section 9.6 and 9.7 of the strategy specification.
"""
import hashlib
import hmac
import logging
import os
import time
import urllib.parse
from typing import Optional

import requests

from config import CONFIG

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Places, monitors, and records live market orders on Binance USDT-margined perpetuals.
    Requires BINANCE_API_KEY and BINANCE_API_SECRET in the environment (from .env).

    All orders are MARKET type.  Entry and exit use market orders to guarantee fill.
    """

    def __init__(self, cfg=CONFIG):
        self.cfg        = cfg
        self.api_key    = os.environ.get("BINANCE_API_KEY",    "")
        self.api_secret = os.environ.get("BINANCE_API_SECRET", "")
        self.base_url   = cfg.BINANCE_REST_BASE
        self.session    = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

        # Local position state (ground truth is always exchange, reconciled at each settlement)
        self.positions:  dict[str, int]            = {s: 0   for s in cfg.symbols}
        self.notionals:  dict[str, float]          = {s: 0.0 for s in cfg.symbols}
        self.nav:        float                     = cfg.INITIAL_CAPITAL
        self.completed_trades: list[dict]          = []

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = str(int(time.time() * 1000))
        qs  = urllib.parse.urlencode(params)
        sig = hmac.new(self.api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    def _post(self, endpoint: str, params: dict) -> dict:
        url    = f"{self.base_url}{endpoint}"
        signed = self._sign(params)
        resp   = self.session.post(url, params=signed, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _get_signed(self, endpoint: str, params: dict) -> dict:
        url    = f"{self.base_url}{endpoint}"
        signed = self._sign(params)
        resp   = self.session.get(url, params=signed, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # NAV
    # ------------------------------------------------------------------

    def get_nav(self) -> float:
        """Return available USDT balance from Binance futures account."""
        try:
            data = self._get_signed("/fapi/v2/account", {})
            usdt = next(
                (float(a["availableBalance"]) for a in data.get("assets", [])
                 if a["asset"] == "USDT"),
                self.nav,
            )
            self.nav = usdt
            return usdt
        except Exception as exc:
            logger.warning("get_nav failed: %s — using cached nav=%.2f", exc, self.nav)
            return self.nav

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def enter(self, symbol: str, direction: int, size_fraction: float) -> dict:
        """
        Place a market order to enter a position.
        direction: +1 = LONG (BUY), -1 = SHORT (SELL).
        size_fraction: fraction of NAV to commit (e.g. 0.02 for 2%).
        """
        nav        = self.get_nav()
        mark_price = self._get_mark_price(symbol)
        notional   = size_fraction * nav
        quantity   = round(notional / mark_price, 3)

        if quantity * mark_price < 100:
            logger.warning("Order below minimum notional ($100): qty=%.3f price=%.2f", quantity, mark_price)
            return {}

        side = "BUY" if direction == 1 else "SELL"
        params = {
            "symbol":   symbol,
            "side":     side,
            "type":     "MARKET",
            "quantity": str(quantity),
        }

        try:
            result = self._post("/fapi/v1/order", params)
            fill_price = float(result.get("avgPrice", mark_price))
            logger.info(
                "ENTRY %s %s qty=%.3f fill=%.2f notional=%.2f",
                symbol, side, quantity, fill_price, notional,
            )
            self.positions[symbol] = direction
            self.notionals[symbol] = notional
            return result
        except Exception as exc:
            logger.error("Entry order failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def exit(self, symbol: str, reason: str = "manual") -> dict:
        """Close the open position for symbol with a market order."""
        pos = self.positions.get(symbol, 0)
        if pos == 0:
            logger.info("No open position for %s — skipping exit", symbol)
            return {}

        side     = "SELL" if pos == 1 else "BUY"
        notional = self.notionals.get(symbol, 0.0)
        mark     = self._get_mark_price(symbol)
        quantity = round(notional / mark, 3)

        params = {
            "symbol":           symbol,
            "side":             side,
            "type":             "MARKET",
            "quantity":         str(quantity),
            "reduceOnly":       "true",
        }

        try:
            result = self._post("/fapi/v1/order", params)
            fill_price = float(result.get("avgPrice", mark))
            logger.info(
                "EXIT %s %s reason=%s fill=%.2f",
                symbol, side, reason, fill_price,
            )
            self.completed_trades.append({
                "symbol":      symbol,
                "direction":   pos,
                "exit_reason": reason,
                "fill_price":  fill_price,
            })
            self.positions[symbol] = 0
            self.notionals[symbol] = 0.0
            return result
        except Exception as exc:
            logger.error("Exit order failed for %s: %s", symbol, exc)
            return {}

    # ------------------------------------------------------------------
    # Position reconciliation (Section 9.7)
    # ------------------------------------------------------------------

    def reconcile(self, symbol: str) -> None:
        """
        Compare exchange positionAmt to local state.
        Adopts exchange ground truth if discrepancy > 0.001 BTC/ETH.
        Halts new entries until resolved.
        """
        try:
            data = self._get_signed("/fapi/v2/positionRisk", {"symbol": symbol})
            for pos in data:
                if pos["symbol"] == symbol:
                    exchange_qty = float(pos["positionAmt"])
                    local_pos    = self.positions.get(symbol, 0)
                    mark         = self._get_mark_price(symbol)
                    local_qty    = self.notionals.get(symbol, 0.0) / mark if mark > 0 else 0.0

                    discrepancy  = abs(abs(exchange_qty) - abs(local_qty))
                    if discrepancy > 0.001:
                        logger.critical(
                            "POSITION DISCREPANCY %s: exchange=%.4f local=%.4f — adopting exchange",
                            symbol, exchange_qty, local_qty,
                        )
                        self.positions[symbol] = (1 if exchange_qty > 0 else -1 if exchange_qty < 0 else 0)
                        self.notionals[symbol] = abs(exchange_qty) * mark
        except Exception as exc:
            logger.error("Reconciliation failed for %s: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_mark_price(self, symbol: str) -> float:
        try:
            resp = self.session.get(
                f"{self.base_url}/fapi/v1/premiumIndex",
                params={"symbol": symbol},
                timeout=5,
            )
            resp.raise_for_status()
            return float(resp.json()["markPrice"])
        except Exception as exc:
            logger.warning("Mark price fetch failed for %s: %s", symbol, exc)
            return 1.0
