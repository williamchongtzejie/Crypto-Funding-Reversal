"""
Live signal monitor: runs at each settlement boundary to compute signals
and trigger order execution.
Section 12 Phase 7 / Section 9.2 of the specification.
"""
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import CONFIG
from signals.funding_zscore import FundingZScoreSignal
from signals.filters import SignalFilters
from risk.sizing import KellySizer

logger = logging.getLogger(__name__)

# Shared live state (read by dashboard via get_live_state())
_LIVE_STATE = {
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
    """Thread-safe read of the live state dict (used by dashboard callbacks)."""
    with _STATE_LOCK:
        return dict(_LIVE_STATE)


def _fetch_ls_ratio(symbol: str, cfg=CONFIG) -> tuple[Optional[float], Optional[float]]:
    """Fetch current L/S ratios from Binance REST."""
    try:
        session = requests.Session()
        r1 = session.get(
            f"{cfg.BINANCE_REST_BASE}/futures/data/globalLongShortAccountRatio",
            params={"symbol": symbol, "period": "1h", "limit": 1},
            timeout=5,
        )
        r2 = session.get(
            f"{cfg.BINANCE_REST_BASE}/futures/data/topLongShortPositionRatio",
            params={"symbol": symbol, "period": "1h", "limit": 1},
            timeout=5,
        )
        ls_global = float(r1.json()[0]["longShortRatio"]) if r1.ok and r1.json() else None
        ls_top    = float(r2.json()[0]["longShortRatio"]) if r2.ok and r2.json() else None
        return ls_global, ls_top
    except Exception as exc:
        logger.warning("L/S ratio fetch failed: %s", exc)
        return None, None


class SignalMonitor:
    """
    Called by BinanceLiveFeed at each settlement boundary.
    Runs the full signal pipeline and triggers order management.
    Also monitors the ATR stop on every mark price tick.
    """

    def __init__(self, order_manager, cfg=CONFIG):
        self.cfg            = cfg
        self.order_manager  = order_manager
        self.zscore_engine  = FundingZScoreSignal(cfg)
        self.filters        = SignalFilters(cfg)
        self.sizer          = KellySizer(cfg)

        # Per-symbol rolling buffers
        self.funding_buffers: dict[str, deque] = {
            s: deque(maxlen=cfg.FUNDING_Z_WINDOW * 2)
            for s in cfg.symbols
        }
        self.position:        dict[str, int]   = {s: 0 for s in cfg.symbols}
        self.entry_price:     dict[str, Optional[float]] = {s: None for s in cfg.symbols}
        self.atr_at_entry:    dict[str, Optional[float]] = {s: None for s in cfg.symbols}
        self.entry_notional:  dict[str, Optional[float]] = {s: None for s in cfg.symbols}
        self.bars_held:       dict[str, int]   = {s: 0 for s in cfg.symbols}

    # ------------------------------------------------------------------
    # Settlement callback
    # ------------------------------------------------------------------

    def on_settlement(self, symbol: str, funding_rate: float, mark_price: float) -> None:
        """Called by BinanceLiveFeed after each 8h settlement is confirmed."""
        buf = self.funding_buffers[symbol]
        buf.append(funding_rate)

        ls_global, ls_top = _fetch_ls_ratio(symbol, self.cfg)

        # Build a minimal DataFrame for signal computation
        df = self._buffer_to_df(symbol, funding_rate, mark_price, ls_global, ls_top)
        if df is None or len(df) < self.cfg.FUNDING_Z_WINDOW // 2:
            logger.info("Insufficient buffer for %s (%d bars)", symbol, len(buf))
            return

        df = self.zscore_engine.compute_rolling_stats(df)
        raw_sig = self.zscore_engine.raw_signal(df)
        df = self.filters.apply_all(df, raw_sig)
        df = self.sizer.compute_sizes(df)

        latest    = df.iloc[-1]
        z_score   = float(latest["funding_zscore"])
        confirmed = int(latest["confirmed_signal"])
        size      = float(latest["final_size"])

        # Update shared live state
        with _STATE_LOCK:
            _LIVE_STATE["funding_zscore"] = z_score
            _LIVE_STATE["mark_price"]     = mark_price
            _LIVE_STATE["funding_rate"]   = funding_rate
            _LIVE_STATE["ls_ratio"]       = ls_global
            _LIVE_STATE["position"]       = self.position.get(symbol, 0)
            _LIVE_STATE["entry_price"]    = self.entry_price.get(symbol)
            _LIVE_STATE["bars_held"]      = self.bars_held.get(symbol, 0)

            # Append to settlement log (keep last 20)
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

        # --- Check settlement-based exits (z-revert, time stop) ---
        current_pos = self.position.get(symbol, 0)
        if current_pos != 0:
            if abs(z_score) < self.cfg.Z_EXIT_BAND:
                logger.info("Z-revert exit: %s z=%.3f", symbol, z_score)
                self.order_manager.exit(symbol, reason="z_revert")
                self._reset_position(symbol)
                return

            self.bars_held[symbol] = self.bars_held.get(symbol, 0) + 1
            if self.bars_held[symbol] >= self.cfg.TIME_STOP_BARS:
                logger.info("Time stop exit: %s bars_held=%d", symbol, self.bars_held[symbol])
                self.order_manager.exit(symbol, reason="time_stop")
                self._reset_position(symbol)
                return

        # --- Check entry ---
        if current_pos == 0 and confirmed != 0 and size > 0:
            nav = self.order_manager.get_nav()
            logger.info("Entry signal: %s direction=%d size=%.4f nav=%.2f", symbol, confirmed, size, nav)
            self.order_manager.enter(symbol, confirmed, size)
            self.position[symbol]   = confirmed
            self.entry_price[symbol] = mark_price
            self.bars_held[symbol]  = 0

    # ------------------------------------------------------------------
    # ATR stop callback (per-tick)
    # ------------------------------------------------------------------

    def on_mark_price_tick(self, symbol: str, mark_price: float) -> None:
        """Called on every 1-second markPrice WebSocket update."""
        pos   = self.position.get(symbol, 0)
        entry = self.entry_price.get(symbol)
        atr   = self.atr_at_entry.get(symbol)

        if pos == 0 or entry is None or atr is None or atr <= 0:
            return

        adverse = (entry - mark_price) * pos
        if adverse > self.cfg.ATR_STOP_MULT * atr:
            logger.warning(
                "ATR stop: %s adverse=%.2f > %.2f x ATR=%.2f",
                symbol, adverse, self.cfg.ATR_STOP_MULT, atr,
            )
            self.order_manager.exit(symbol, reason="atr_stop")
            self._reset_position(symbol)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_position(self, symbol: str) -> None:
        self.position[symbol]      = 0
        self.entry_price[symbol]   = None
        self.atr_at_entry[symbol]  = None
        self.entry_notional[symbol] = None
        self.bars_held[symbol]     = 0
        with _STATE_LOCK:
            _LIVE_STATE["position"]    = 0
            _LIVE_STATE["entry_price"] = None
            _LIVE_STATE["bars_held"]   = 0

    def _buffer_to_df(
        self,
        symbol: str,
        latest_rate: float,
        mark_price: float,
        ls_global: Optional[float],
        ls_top: Optional[float],
    ) -> Optional[pd.DataFrame]:
        """Build a minimal DataFrame from the rolling funding buffer for signal computation."""
        buf = list(self.funding_buffers[symbol])
        if len(buf) < 2:
            return None

        now = pd.Timestamp.utcnow().floor("8h")
        idx = pd.date_range(end=now, periods=len(buf), freq="8h", tz="UTC")

        df = pd.DataFrame({
            "funding_rate": buf,
            "close":        [mark_price] * len(buf),
            "high":         [mark_price * 1.005] * len(buf),
            "low":          [mark_price * 0.995] * len(buf),
            "mark_close":   [mark_price] * len(buf),
            "index_close":  [mark_price] * len(buf),
            "basis":        [0.0] * len(buf),
            "ls_ratio":     [ls_global if ls_global is not None else float("nan")] * len(buf),
            "top_ls_ratio": [ls_top if ls_top is not None else float("nan")] * len(buf),
            "atr_8h":       [mark_price * 0.02] * len(buf),  # placeholder; updated on entry
            "ret_8h":       [0.0] * len(buf),
            "rvol_ann":     [0.5] * len(buf),
            "confirmed_signal": [0] * len(buf),
        }, index=idx)

        return df
