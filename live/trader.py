"""
Unified live trading layer: settlement-driven signal execution and order management.
Merges the former SignalMonitor and OrderManager into one class.
"""
import hashlib
import hmac
import logging
import os
import threading
import time
import urllib.parse
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import CONFIG
from signals.pipeline import SignalPipeline
from risk.manager import RiskManager

logger = logging.getLogger(__name__)

# Module-level live state — read by dashboard via get_live_state()
_LIVE_STATE: dict = {
    "funding_zscore": None,
    "mark_price":     None,
    "funding_rate":   None,
    "basis":          None,
    "ls_ratio":       None,
    "position":       0,
    "entry_price":    None,
    "bars_held":      0,
    "settlements":    [],
}
_STATE_LOCK = threading.Lock()


def get_live_state() -> dict:
    """Thread-safe snapshot of the current live state (used by dashboard callbacks)."""
    with _STATE_LOCK:
        return dict(_LIVE_STATE)


class LiveTrader:
    """
    Drives the live trading loop from WebSocket callbacks.

    Callbacks wired to BinanceLiveFeed:
        on_settlement(symbol, funding_rate, mark_price)  — fires at each 8h settlement
        on_mark_price_tick(symbol, mark_price)           — fires on every 1s markPrice tick

    Order execution (private):
        _enter(symbol, direction, size_fraction)
        _exit(symbol, reason)
        _reconcile(symbol)
        _get_nav() -> float
        _get_mark_price(symbol) -> float

    API signing (private):
        _sign(params) -> dict
        _post(endpoint, params) -> dict
        _get_signed(endpoint, params) -> dict
    """

    def __init__(self, cfg=CONFIG):
        self.cfg            = cfg
        self.signal_engine  = SignalPipeline(cfg)
        self.risk_manager   = RiskManager(cfg)

        # Binance signed-API credentials
        self.api_key    = os.environ.get("BINANCE_API_KEY",    "")
        self.api_secret = os.environ.get("BINANCE_API_SECRET", "")
        self.session    = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

        # Per-symbol rolling funding buffers for live z-score computation
        self.funding_buffers: dict[str, deque] = {
            s: deque(maxlen=cfg.FUNDING_Z_WINDOW * 2) for s in cfg.symbols
        }

        # Position state
        self.positions:       dict[str, int]            = {s: 0   for s in cfg.symbols}
        self.entry_prices:    dict[str, Optional[float]] = {s: None for s in cfg.symbols}
        self.atr_at_entry:    dict[str, Optional[float]] = {s: None for s in cfg.symbols}
        self.entry_notionals: dict[str, Optional[float]] = {s: None for s in cfg.symbols}
        self.bars_held:       dict[str, int]             = {s: 0   for s in cfg.symbols}
        self.nav:             float                      = cfg.INITIAL_CAPITAL
        self.completed_trades: list[dict]               = []

    # ------------------------------------------------------------------
    # WebSocket callbacks
    # ------------------------------------------------------------------

    def on_settlement(self, symbol: str, funding_rate: float, mark_price: float) -> None:
        """Called by BinanceLiveFeed at each confirmed 8-hour settlement."""
        buf = self.funding_buffers[symbol]
        buf.append(funding_rate)

        ls_global, ls_top = self._fetch_ls_ratio(symbol)

        df = self._buffer_to_df(symbol, funding_rate, mark_price, ls_global, ls_top)
        if df is None or len(df) < self.cfg.FUNDING_Z_WINDOW // 2:
            logger.info("Insufficient buffer for %s (%d bars)", symbol, len(buf))
            return

        df      = self.signal_engine.run(df)
        df      = self.risk_manager.compute_sizes(df)
        latest  = df.iloc[-1]
        z_score = float(latest["funding_zscore"])
        confirmed = int(latest["confirmed_signal"])
        size    = float(latest["final_size"])

        with _STATE_LOCK:
            _LIVE_STATE.update({
                "funding_zscore": z_score,
                "mark_price":     mark_price,
                "funding_rate":   funding_rate,
                "ls_ratio":       ls_global,
                "position":       self.positions.get(symbol, 0),
                "entry_price":    self.entry_prices.get(symbol),
                "bars_held":      self.bars_held.get(symbol, 0),
            })
            _LIVE_STATE["settlements"].append({
                "time":        datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "symbol":      symbol,
                "funding_rate": f"{funding_rate * 100:.4f}%",
                "mark_price":  f"${mark_price:,.2f}",
                "z_score":     f"{z_score:.3f}",
                "signal":      {-1: "SHORT", 0: "FLAT", 1: "LONG"}.get(confirmed, "FLAT"),
            })
            _LIVE_STATE["settlements"] = _LIVE_STATE["settlements"][-20:]

        logger.info(
            "Settlement %s: fr=%.6f z=%.3f signal=%d size=%.4f",
            symbol, funding_rate, z_score, confirmed, size,
        )

        current_pos = self.positions.get(symbol, 0)
        if current_pos != 0:
            if abs(z_score) < self.cfg.Z_EXIT_BAND:
                logger.info("Z-revert exit: %s z=%.3f", symbol, z_score)
                self._exit(symbol, "z_revert")
                return

            self.bars_held[symbol] = self.bars_held.get(symbol, 0) + 1
            if self.bars_held[symbol] >= self.cfg.TIME_STOP_BARS:
                logger.info("Time stop: %s bars_held=%d", symbol, self.bars_held[symbol])
                self._exit(symbol, "time_stop")
                return

        if self.positions.get(symbol, 0) == 0 and confirmed != 0 and size > 0:
            nav = self._get_nav()
            logger.info("Entry: %s dir=%d size=%.4f nav=%.2f", symbol, confirmed, size, nav)
            self._enter(symbol, confirmed, size)

    def on_mark_price_tick(self, symbol: str, mark_price: float) -> None:
        """Called on every 1-second markPrice tick. Checks the ATR hard stop."""
        pos   = self.positions.get(symbol, 0)
        entry = self.entry_prices.get(symbol)
        atr   = self.atr_at_entry.get(symbol)

        if pos == 0 or entry is None or atr is None or atr <= 0:
            return

        adverse = (entry - mark_price) * pos
        if adverse > self.cfg.ATR_STOP_MULT * atr:
            logger.warning(
                "ATR stop: %s adverse=%.2f > %.2f × ATR=%.2f",
                symbol, adverse, self.cfg.ATR_STOP_MULT, atr,
            )
            self._exit(symbol, "atr_stop")

    # ------------------------------------------------------------------
    # Order execution (private)
    # ------------------------------------------------------------------

    def _enter(self, symbol: str, direction: int, size_fraction: float) -> dict:
        """Place a market order to open a position."""
        nav        = self._get_nav()
        mark_price = self._get_mark_price(symbol)
        notional   = size_fraction * nav
        quantity   = round(notional / mark_price, 3)

        if quantity * mark_price < 100:
            logger.warning("Order below minimum notional: qty=%.3f price=%.2f", quantity, mark_price)
            return {}

        side   = "BUY" if direction == 1 else "SELL"
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": str(quantity)}

        try:
            result     = self._post("/fapi/v1/order", params)
            fill_price = float(result.get("avgPrice", mark_price))
            logger.info("ENTRY %s %s qty=%.3f fill=%.2f notional=%.2f", symbol, side, quantity, fill_price, notional)
            self.positions[symbol]       = direction
            self.entry_prices[symbol]    = fill_price
            self.entry_notionals[symbol] = notional
            self.bars_held[symbol]       = 0
            with _STATE_LOCK:
                _LIVE_STATE["position"]    = direction
                _LIVE_STATE["entry_price"] = fill_price
                _LIVE_STATE["bars_held"]   = 0
            return result
        except Exception as exc:
            logger.error("Entry failed for %s: %s", symbol, exc)
            return {}

    def _exit(self, symbol: str, reason: str = "manual") -> dict:
        """Close the open position for symbol with a market order."""
        pos = self.positions.get(symbol, 0)
        if pos == 0:
            logger.info("No open position for %s — skipping exit", symbol)
            return {}

        side     = "SELL" if pos == 1 else "BUY"
        notional = self.entry_notionals.get(symbol, 0.0) or 0.0
        mark     = self._get_mark_price(symbol)
        quantity = round(notional / mark, 3)

        params = {"symbol": symbol, "side": side, "type": "MARKET",
                  "quantity": str(quantity), "reduceOnly": "true"}

        try:
            result     = self._post("/fapi/v1/order", params)
            fill_price = float(result.get("avgPrice", mark))
            logger.info("EXIT %s %s reason=%s fill=%.2f", symbol, side, reason, fill_price)
            self.completed_trades.append({
                "symbol": symbol, "direction": pos,
                "exit_reason": reason, "fill_price": fill_price,
            })
            self._reset_position(symbol)
            return result
        except Exception as exc:
            logger.error("Exit failed for %s: %s", symbol, exc)
            return {}

    def _reconcile(self, symbol: str) -> None:
        """Compare exchange positionAmt to local state; adopt exchange as ground truth."""
        try:
            data = self._get_signed("/fapi/v2/positionRisk", {"symbol": symbol})
            for pos in data:
                if pos["symbol"] == symbol:
                    exchange_qty = float(pos["positionAmt"])
                    mark         = self._get_mark_price(symbol)
                    local_qty    = (self.entry_notionals.get(symbol) or 0.0) / mark if mark > 0 else 0.0
                    discrepancy  = abs(abs(exchange_qty) - abs(local_qty))
                    if discrepancy > 0.001:
                        logger.critical(
                            "POSITION DISCREPANCY %s: exchange=%.4f local=%.4f — adopting exchange",
                            symbol, exchange_qty, local_qty,
                        )
                        self.positions[symbol] = (1 if exchange_qty > 0 else -1 if exchange_qty < 0 else 0)
                        self.entry_notionals[symbol] = abs(exchange_qty) * mark
        except Exception as exc:
            logger.error("Reconciliation failed for %s: %s", symbol, exc)

    # ------------------------------------------------------------------
    # API helpers (private)
    # ------------------------------------------------------------------

    def _get_nav(self) -> float:
        try:
            data = self._get_signed("/fapi/v2/account", {})
            usdt = next(
                (float(a["availableBalance"]) for a in data.get("assets", []) if a["asset"] == "USDT"),
                self.nav,
            )
            self.nav = usdt
            return usdt
        except Exception as exc:
            logger.warning("get_nav failed: %s — using cached nav=%.2f", exc, self.nav)
            return self.nav

    def _get_mark_price(self, symbol: str) -> float:
        try:
            resp = self.session.get(
                f"{self.cfg.BINANCE_REST_BASE}/fapi/v1/premiumIndex",
                params={"symbol": symbol}, timeout=5,
            )
            resp.raise_for_status()
            return float(resp.json()["markPrice"])
        except Exception as exc:
            logger.warning("Mark price fetch failed for %s: %s", symbol, exc)
            return 1.0

    def _fetch_ls_ratio(self, symbol: str) -> tuple[Optional[float], Optional[float]]:
        try:
            r1 = self.session.get(
                f"{self.cfg.BINANCE_REST_BASE}/futures/data/globalLongShortAccountRatio",
                params={"symbol": symbol, "period": "1h", "limit": 1}, timeout=5,
            )
            r2 = self.session.get(
                f"{self.cfg.BINANCE_REST_BASE}/futures/data/topLongShortPositionRatio",
                params={"symbol": symbol, "period": "1h", "limit": 1}, timeout=5,
            )
            ls_global = float(r1.json()[0]["longShortRatio"]) if r1.ok and r1.json() else None
            ls_top    = float(r2.json()[0]["longShortRatio"]) if r2.ok and r2.json() else None
            return ls_global, ls_top
        except Exception as exc:
            logger.warning("L/S ratio fetch failed: %s", exc)
            return None, None

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = str(int(time.time() * 1000))
        qs  = urllib.parse.urlencode(params)
        sig = hmac.new(self.api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    def _post(self, endpoint: str, params: dict) -> dict:
        resp = self.session.post(
            f"{self.cfg.BINANCE_REST_BASE}{endpoint}",
            params=self._sign(params), timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _get_signed(self, endpoint: str, params: dict) -> dict:
        resp = self.session.get(
            f"{self.cfg.BINANCE_REST_BASE}{endpoint}",
            params=self._sign(params), timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _reset_position(self, symbol: str) -> None:
        self.positions[symbol]       = 0
        self.entry_prices[symbol]    = None
        self.atr_at_entry[symbol]    = None
        self.entry_notionals[symbol] = None
        self.bars_held[symbol]       = 0
        with _STATE_LOCK:
            _LIVE_STATE.update({"position": 0, "entry_price": None, "bars_held": 0})

    def _buffer_to_df(
        self,
        symbol: str,
        latest_rate: float,
        mark_price: float,
        ls_global: Optional[float],
        ls_top: Optional[float],
    ) -> Optional[pd.DataFrame]:
        buf = list(self.funding_buffers[symbol])
        if len(buf) < 2:
            return None
        now = pd.Timestamp.utcnow().floor("8h")
        idx = pd.date_range(end=now, periods=len(buf), freq="8h", tz="UTC")
        return pd.DataFrame({
            "funding_rate":     buf,
            "close":            [mark_price] * len(buf),
            "high":             [mark_price * 1.005] * len(buf),
            "low":              [mark_price * 0.995] * len(buf),
            "mark_close":       [mark_price] * len(buf),
            "index_close":      [mark_price] * len(buf),
            "basis":            [0.0] * len(buf),
            "ls_ratio":         [ls_global if ls_global is not None else float("nan")] * len(buf),
            "top_ls_ratio":     [ls_top    if ls_top    is not None else float("nan")] * len(buf),
            "atr_8h":           [mark_price * 0.02] * len(buf),
            "ret_8h":           [0.0] * len(buf),
            "rvol_ann":         [0.5] * len(buf),
            "confirmed_signal": [0] * len(buf),
        }, index=idx)
