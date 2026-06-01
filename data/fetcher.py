"""
Binance REST API fetcher for all Strategy 3 data requirements.
All historical endpoints are public — no API key needed for data pulls.
API keys are only required for live order execution (live/order_manager.py).
"""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import CONFIG

logger = logging.getLogger(__name__)

_RETRY_STRATEGY = Retry(
    total=5,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
)


def _make_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY_STRATEGY)
    session.mount("https://", adapter)
    return session


def _to_ms(date_str: str) -> int:
    dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


class BinanceFetcher:
    """
    Fetches all data required for Strategy 3 from Binance public REST API.
    Caches results as .parquet files; reloads from cache if present.
    """

    def __init__(self, base_url: str = CONFIG.BINANCE_REST_BASE):
        self.base_url = base_url
        self.session = _make_session()

    # ------------------------------------------------------------------
    # Dataset 1 — Funding Rate History
    # ------------------------------------------------------------------

    def fetch_funding_rates(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Pull funding rate history from /fapi/v1/fundingRate.
        Returns DataFrame with columns: funding_rate, mark_price.
        Index: pd.DatetimeIndex (UTC), named 'funding_time'.
        """
        url = f"{self.base_url}/fapi/v1/fundingRate"
        start_ms = _to_ms(start_date)
        end_ms = _to_ms(end_date)
        limit = 1000
        rows = []

        while start_ms < end_ms:
            params = {
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            }
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break

            for item in batch:
                mp = item.get("markPrice", "") or ""
                rows.append({
                    "funding_time": pd.Timestamp(item["fundingTime"], unit="ms", tz="UTC"),
                    "funding_rate": float(item["fundingRate"]),
                    "mark_price":   float(mp) if mp else float("nan"),
                })

            start_ms = batch[-1]["fundingTime"] + 1
            if len(batch) < limit:
                break
            time.sleep(0.2)

        if not rows:
            logger.warning("No funding rate data for %s %s→%s", symbol, start_date, end_date)
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("funding_time").sort_index()
        df = df[~df.index.duplicated()]
        logger.info("Fetched %d funding rate rows for %s", len(df), symbol)
        return df

    # ------------------------------------------------------------------
    # Dataset 2/3/4 — OHLCV Klines (perp, mark price, index price)
    # ------------------------------------------------------------------

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        endpoint: str = "/fapi/v1/klines",
    ) -> pd.DataFrame:
        """
        Pull OHLCV bars from the given klines endpoint.
        Returns DataFrame with columns: open, high, low, close, volume.
        Index: pd.DatetimeIndex (UTC), named 'open_time'.
        """
        url = f"{self.base_url}{endpoint}"
        start_ms = _to_ms(start_date)
        end_ms = _to_ms(end_date)
        limit = 1000
        rows = []

        # indexPriceKlines uses "pair" instead of "symbol"
        sym_key = "pair" if endpoint == "/fapi/v1/indexPriceKlines" else "symbol"

        while start_ms < end_ms:
            params = {
                sym_key:     symbol,
                "interval":  interval,
                "startTime": start_ms,
                "endTime":   end_ms,
                "limit":     limit,
            }
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break

            for k in batch:
                rows.append({
                    "open_time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
                    "open":      float(k[1]),
                    "high":      float(k[2]),
                    "low":       float(k[3]),
                    "close":     float(k[4]),
                    "volume":    float(k[5]),
                })

            start_ms = batch[-1][0] + 1
            if len(batch) < limit:
                break
            time.sleep(0.2)

        if not rows:
            logger.warning("No kline data from %s for %s %s", endpoint, symbol, interval)
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("open_time").sort_index()
        df = df[~df.index.duplicated()]
        logger.info("Fetched %d kline rows (%s %s) from %s", len(df), symbol, interval, endpoint)
        return df

    # ------------------------------------------------------------------
    # Dataset 5 — Long/Short Ratio (30-day rolling window only)
    # ------------------------------------------------------------------

    def fetch_ls_ratio(
        self,
        symbol: str,
        endpoint: str,
        period: str = "1h",
    ) -> Optional[pd.DataFrame]:
        """
        Pull Long/Short ratio from globalLongShortAccountRatio or
        topLongShortPositionRatio. Returns at most 30 days of history.
        Returns None on failure (treated as NaN in processor).
        """
        url = f"{self.base_url}{endpoint}"
        limit = 500
        rows = []
        params = {"symbol": symbol, "period": period, "limit": limit}

        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as exc:
            logger.warning("L/S ratio fetch failed: %s", exc)
            return None

        for item in batch:
            rows.append({
                "timestamp":       pd.Timestamp(item["timestamp"], unit="ms", tz="UTC"),
                "long_short_ratio": float(item["longShortRatio"]),
            })

        if not rows:
            return None

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df = df[~df.index.duplicated()]
        logger.info("Fetched %d L/S ratio rows from %s", len(df), endpoint)
        return df

    # ------------------------------------------------------------------
    # Composite fetch — all datasets for one symbol
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        save_dir: Optional[str] = None,
    ) -> dict:
        """
        Fetch all datasets for `symbol` between start_date and end_date.
        If save_dir is provided, caches each dataset as a .parquet file and
        loads from cache on subsequent calls.

        Returns dict with keys:
          funding, klines_8h, klines_1h, mark_8h, index_8h, ls_global, ls_top
        """
        cache_dir = Path(save_dir) / symbol if save_dir else None

        def _load_or_fetch(name: str, fetch_fn, *args, **kwargs) -> Optional[pd.DataFrame]:
            if cache_dir is not None:
                cache_path = cache_dir / f"{name}.parquet"
                if cache_path.exists():
                    logger.info("Loading %s from cache: %s", name, cache_path)
                    return pd.read_parquet(cache_path)

            df = fetch_fn(*args, **kwargs)
            if cache_dir is not None and df is not None and not df.empty:
                cache_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path)
                logger.info("Cached %s → %s", name, cache_path)
            return df

        funding = _load_or_fetch(
            f"{symbol}_funding", self.fetch_funding_rates,
            symbol, start_date, end_date,
        )
        klines_8h = _load_or_fetch(
            f"{symbol}_klines_8h", self.fetch_klines,
            symbol, "8h", start_date, end_date,
        )
        klines_1h = _load_or_fetch(
            f"{symbol}_klines_1h", self.fetch_klines,
            symbol, "1h", start_date, end_date,
        )
        mark_8h = _load_or_fetch(
            f"{symbol}_mark_8h", self.fetch_klines,
            symbol, "8h", start_date, end_date,
            endpoint="/fapi/v1/markPriceKlines",
        )
        index_8h = _load_or_fetch(
            f"{symbol}_index_8h", self.fetch_klines,
            symbol, "8h", start_date, end_date,
            endpoint="/fapi/v1/indexPriceKlines",
        )

        # L/S data: always fresh (max 30-day history, don't cache)
        ls_global = self.fetch_ls_ratio(
            symbol,
            "/futures/data/globalLongShortAccountRatio",
        )
        ls_top = self.fetch_ls_ratio(
            symbol,
            "/futures/data/topLongShortPositionRatio",
        )

        return {
            "funding":   funding,
            "klines_8h": klines_8h,
            "klines_1h": klines_1h,
            "mark_8h":   mark_8h,
            "index_8h":  index_8h,
            "ls_global": ls_global,
            "ls_top":    ls_top,
        }
