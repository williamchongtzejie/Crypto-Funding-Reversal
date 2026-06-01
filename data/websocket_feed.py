"""
Live WebSocket feed for mark price and funding rate updates.
Maintains per-symbol state buffers and fires callbacks at settlement boundaries.
"""
import asyncio
import json
import logging
import time
from collections import deque
from typing import Callable, Dict, Optional

import websockets

from config import CONFIG

logger = logging.getLogger(__name__)

_PING_INTERVAL = 600        # send ping every 10 minutes
_BACKOFF_BASE  = 1.0        # seconds
_BACKOFF_MAX   = 60.0       # seconds


class SymbolState:
    """Per-symbol live state buffer."""

    def __init__(self, symbol: str, maxlen: int = 90):
        self.symbol = symbol
        self.funding_rate_buffer: deque = deque(maxlen=maxlen)
        self.mark_price_current: Optional[float] = None
        self.next_funding_rate:   Optional[float] = None
        self.next_settlement_ms:  Optional[int]   = None
        self.prev_settlement_ms:  Optional[int]   = None
        self.position:            int   = 0
        self.entry_price:         Optional[float] = None
        self.atr_at_entry:        Optional[float] = None
        self.entry_notional:      Optional[float] = None
        self.bars_held:           int   = 0
        self.atr_buffer:          deque = deque(maxlen=30)


class BinanceLiveFeed:
    """
    Manages asyncio WebSocket connection to Binance combined stream.
    Fires on_settlement and on_mark_price_tick callbacks when events occur.

    Streams subscribed (one per symbol):
      {symbol}@markPrice@1s   — mark price + next funding rate (1-second)
      {symbol}@kline_1h       — 1h bars for ATR rolling update
    """

    def __init__(
        self,
        symbols=None,
        cfg=CONFIG,
        on_settlement:       Optional[Callable] = None,
        on_mark_price_tick:  Optional[Callable] = None,
    ):
        self.cfg = cfg
        self.symbols = symbols or cfg.symbols
        self.on_settlement = on_settlement
        self.on_mark_price_tick = on_mark_price_tick
        self.states: Dict[str, SymbolState] = {
            s: SymbolState(s) for s in self.symbols
        }
        self._running = False

    def _build_ws_url(self) -> str:
        streams = []
        for sym in self.symbols:
            s = sym.lower()
            streams.append(f"{s}@markPrice@1s")
            streams.append(f"{s}@kline_1h")
        combined = "/".join(streams)
        return f"{self.cfg.BINANCE_WS_BASE}/stream?streams={combined}"

    async def _restore_funding_buffer(self, symbol: str) -> None:
        """On reconnect, re-fetch last 90 funding rates via REST to restore buffer."""
        import requests
        state = self.states[symbol]
        url = f"{self.cfg.BINANCE_REST_BASE}/fapi/v1/fundingRate"
        try:
            resp = requests.get(url, params={"symbol": symbol, "limit": 90}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            state.funding_rate_buffer.clear()
            for item in sorted(data, key=lambda x: x["fundingTime"]):
                state.funding_rate_buffer.append(float(item["fundingRate"]))
            logger.info("Restored %d funding rates for %s", len(state.funding_rate_buffer), symbol)
        except Exception as exc:
            logger.warning("Failed to restore funding buffer for %s: %s", symbol, exc)

    def _handle_mark_price(self, msg: dict) -> None:
        symbol = msg["s"]
        if symbol not in self.states:
            return
        state = self.states[symbol]

        mark_price      = float(msg["p"])
        next_rate       = float(msg["r"])
        next_settle_ms  = int(msg["T"])

        state.mark_price_current  = mark_price
        state.next_funding_rate   = next_rate

        if self.on_mark_price_tick:
            self.on_mark_price_tick(symbol, mark_price)

        # Settlement detection: settlement_ms has advanced since last tick
        prev_settle = state.prev_settlement_ms
        if prev_settle is not None and next_settle_ms > prev_settle:
            # A new settlement has occurred; the confirmed rate is in "r" at that time
            confirmed_rate = next_rate
            state.funding_rate_buffer.append(confirmed_rate)
            logger.info(
                "Settlement: %s funding=%.6f mark=%.2f buffer_len=%d",
                symbol, confirmed_rate, mark_price, len(state.funding_rate_buffer),
            )
            if self.on_settlement:
                self.on_settlement(symbol, confirmed_rate, mark_price)

        state.next_settlement_ms = next_settle_ms
        state.prev_settlement_ms = next_settle_ms

    async def _listen(self) -> None:
        url = self._build_ws_url()
        backoff = _BACKOFF_BASE

        while self._running:
            try:
                logger.info("Connecting WebSocket: %s", url)
                async with websockets.connect(url, ping_interval=_PING_INTERVAL) as ws:
                    backoff = _BACKOFF_BASE
                    for symbol in self.symbols:
                        await self._restore_funding_buffer(symbol)

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            envelope = json.loads(raw_msg)
                            data = envelope.get("data", envelope)
                            event = data.get("e", "")

                            if event == "markPriceUpdate":
                                self._handle_mark_price(data)
                        except Exception as exc:
                            logger.warning("Error processing WS message: %s", exc)

            except Exception as exc:
                logger.error("WebSocket error: %s — reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX)

    async def start_async(self) -> None:
        self._running = True
        await self._listen()

    def start(self) -> None:
        """Run the feed synchronously (blocking)."""
        self._running = True
        asyncio.run(self.start_async())

    def stop(self) -> None:
        self._running = False
