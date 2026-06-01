"""
Unified data layer: fetches from Binance REST API and builds the master DataFrame.
Merges the former BinanceFetcher and DataProcessor into one cohesive class.
"""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
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

_REQUIRED_COLS = [
    "funding_rate", "open", "high", "low", "close", "volume",
    "mark_close", "index_close", "basis", "atr_8h", "ret_8h", "rvol_ann",
]


def _make_session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=_RETRY_STRATEGY))
    return s


def _to_ms(date_str: str) -> int:
    dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


class DataPipeline:
    """
    Unified data layer combining REST fetching and feature engineering.

    Public interface:
        fetch(symbol, start_date, end_date, use_cache)  -> raw dict
        build(raw)                                       -> master DataFrame
        fetch_and_build(symbol, start_date, end_date,
                        use_cache)                       -> master DataFrame
        generate_synthetic(symbol, n_bars)               -> raw dict (no API)
    """

    def __init__(self, cfg=CONFIG):
        self.cfg     = cfg
        self.session = _make_session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = False,
    ) -> dict:
        """
        Fetch all datasets for symbol between start_date and end_date.
        When use_cache=True, loads parquet files from data/raw/{symbol}/
        and writes new fetches back to the same directory.

        Returns dict with keys:
            funding, klines_8h, klines_1h, mark_8h, index_8h, ls_global, ls_top
        """
        cache_dir = (Path(self.cfg.DATA_DIR) / symbol) if use_cache else None

        def _load_or_fetch(name, fetch_fn, *args, **kwargs) -> Optional[pd.DataFrame]:
            if cache_dir is not None:
                path = cache_dir / f"{name}.parquet"
                if path.exists():
                    logger.info("Loading %s from cache: %s", name, path)
                    return pd.read_parquet(path)
            df = fetch_fn(*args, **kwargs)
            if cache_dir is not None and df is not None and not df.empty:
                cache_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_dir / f"{name}.parquet")
                logger.info("Cached %s → %s", name, cache_dir / f"{name}.parquet")
            return df

        funding   = _load_or_fetch(f"{symbol}_funding",   self._fetch_funding_rates, symbol, start_date, end_date)
        klines_8h = _load_or_fetch(f"{symbol}_klines_8h", self._fetch_klines, symbol, "8h",  start_date, end_date)
        klines_1h = _load_or_fetch(f"{symbol}_klines_1h", self._fetch_klines, symbol, "1h",  start_date, end_date)
        mark_8h   = _load_or_fetch(f"{symbol}_mark_8h",   self._fetch_klines, symbol, "8h",  start_date, end_date, endpoint="/fapi/v1/markPriceKlines")
        index_8h  = _load_or_fetch(f"{symbol}_index_8h",  self._fetch_klines, symbol, "8h",  start_date, end_date, endpoint="/fapi/v1/indexPriceKlines")

        # L/S data: always fresh (max 30-day history)
        ls_global = self._fetch_ls_ratio(symbol, "/futures/data/globalLongShortAccountRatio")
        ls_top    = self._fetch_ls_ratio(symbol, "/futures/data/topLongShortPositionRatio")

        return {
            "funding":   funding,
            "klines_8h": klines_8h,
            "klines_1h": klines_1h,
            "mark_8h":   mark_8h,
            "index_8h":  index_8h,
            "ls_global": ls_global,
            "ls_top":    ls_top,
        }

    def build(self, raw: dict) -> pd.DataFrame:
        """
        Build the master DataFrame from the raw dict returned by fetch().

        Steps:
          1-2.  Index on funding timestamps (floor to second to absorb ms offsets)
          3.    Join 8h OHLCV
          4.    Join mark_close
          5.    Join index_close (fallback to close)
          6.    Compute perp-spot basis
          7.    Join ls_ratio
          8.    Join top_ls_ratio
          9.    Forward-fill price columns (max 3 bars) before computing derivatives
          10.   Drop rows missing close or funding_rate
          11.   Compute ATR (EWM span-14)
          12.   Compute ret_8h (log return)
          13.   Compute rvol_ann (rolling annualised vol)
        """
        funding = raw.get("funding")
        if funding is None or funding.empty:
            raise ValueError("Funding rate data is required and missing.")

        df = funding[["funding_rate"]].copy()
        df.index = df.index.floor("s")
        df = df[~df.index.duplicated(keep="last")]
        df.index.name = "time"

        klines_8h = raw.get("klines_8h")
        if klines_8h is not None and not klines_8h.empty:
            df = df.join(klines_8h[["open", "high", "low", "close", "volume"]], how="left")
        else:
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = np.nan

        mark_8h = raw.get("mark_8h")
        if mark_8h is not None and not mark_8h.empty:
            df = df.join(mark_8h[["close"]].rename(columns={"close": "mark_close"}), how="left")
        else:
            df["mark_close"] = df["close"]

        index_8h = raw.get("index_8h")
        if index_8h is not None and not index_8h.empty:
            df = df.join(index_8h[["close"]].rename(columns={"close": "index_close"}), how="left")
        else:
            logger.warning("No index price data; using perpetual close as fallback.")
            df["index_close"] = df["close"]

        df["basis"] = (df["mark_close"] - df["index_close"]) / df["index_close"]

        ls_global = raw.get("ls_global")
        if ls_global is not None and not ls_global.empty:
            df = df.join(
                ls_global[["long_short_ratio"]].rename(columns={"long_short_ratio": "ls_ratio"}),
                how="left",
            )
        else:
            df["ls_ratio"] = np.nan

        ls_top = raw.get("ls_top")
        if ls_top is not None and not ls_top.empty:
            df = df.join(
                ls_top[["long_short_ratio"]].rename(columns={"long_short_ratio": "top_ls_ratio"}),
                how="left",
            )
        else:
            df["top_ls_ratio"] = np.nan

        # Forward-fill before computing ret_8h and ATR
        for col in ["open", "high", "low", "close", "volume", "mark_close", "index_close"]:
            if col in df.columns:
                df[col] = df[col].ffill(limit=3)

        df.dropna(subset=["close", "funding_rate"], inplace=True)

        df["tr"] = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"]  - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_8h"] = df["tr"].ewm(span=self.cfg.ATR_PERIOD, adjust=False).mean()
        df.drop(columns=["tr"], inplace=True)

        df["ret_8h"] = np.log(df["close"] / df["close"].shift(1))

        W_vol = self.cfg.FUNDING_Z_WINDOW
        df["rvol_ann"] = (
            df["ret_8h"]
            .rolling(window=W_vol, min_periods=W_vol // 2)
            .std(ddof=1)
            * np.sqrt(self.cfg.BARS_PER_YEAR)
        )

        missing = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Master DataFrame missing columns: {missing}")

        logger.info("Master DataFrame built: %d bars, %d columns", len(df), len(df.columns))
        return df

    def fetch_and_build(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = False,
    ) -> pd.DataFrame:
        """Convenience method: fetch then build in one call."""
        raw = self.fetch(symbol, start_date, end_date, use_cache=use_cache)
        return self.build(raw)

    @staticmethod
    def generate_synthetic(symbol: str, n_bars: int = 4000) -> dict:
        """
        Generate synthetic 8h data for testing without any API access.
        Funding rate: AR(1) with mean reversion.
        Price: geometric random walk.
        """
        rng   = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=n_bars, freq="8h", tz="UTC")

        phi, sigma = 0.85, 0.0003
        f = np.zeros(n_bars)
        f[0] = 0.0001
        for i in range(1, n_bars):
            f[i] = 0.0001 + phi * (f[i - 1] - 0.0001) + rng.normal(0, sigma)

        returns = rng.normal(0.0002, 0.01, n_bars)
        close   = 40000.0 * np.exp(np.cumsum(returns))
        high    = close * (1 + np.abs(rng.normal(0, 0.005, n_bars)))
        low     = close * (1 - np.abs(rng.normal(0, 0.005, n_bars)))
        open_   = close * (1 + rng.normal(0, 0.003, n_bars))

        funding_df = pd.DataFrame(
            {"funding_rate": f, "mark_price": close},
            index=pd.DatetimeIndex(dates, name="funding_time"),
        )
        klines_df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close,
             "volume": rng.uniform(100, 1000, n_bars)},
            index=pd.DatetimeIndex(dates, name="open_time"),
        )
        return {
            "funding":   funding_df,
            "klines_8h": klines_df,
            "klines_1h": None,
            "mark_8h":   klines_df.copy(),
            "index_8h":  klines_df.copy(),
            "ls_global": None,
            "ls_top":    None,
        }

    # ------------------------------------------------------------------
    # Private REST methods
    # ------------------------------------------------------------------

    def _fetch_funding_rates(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        url      = f"{self.cfg.BINANCE_REST_BASE}/fapi/v1/fundingRate"
        start_ms = _to_ms(start_date)
        end_ms   = _to_ms(end_date)
        rows     = []

        while start_ms < end_ms:
            resp = self.session.get(url, params={
                "symbol": symbol, "startTime": start_ms,
                "endTime": end_ms, "limit": 1000,
            }, timeout=30)
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
            if len(batch) < 1000:
                break
            time.sleep(0.2)

        if not rows:
            logger.warning("No funding rate data for %s %s→%s", symbol, start_date, end_date)
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("funding_time").sort_index()
        df = df[~df.index.duplicated()]
        logger.info("Fetched %d funding rate rows for %s", len(df), symbol)
        return df

    def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        endpoint: str = "/fapi/v1/klines",
    ) -> pd.DataFrame:
        url      = f"{self.cfg.BINANCE_REST_BASE}{endpoint}"
        start_ms = _to_ms(start_date)
        end_ms   = _to_ms(end_date)
        sym_key  = "pair" if endpoint == "/fapi/v1/indexPriceKlines" else "symbol"
        rows     = []

        while start_ms < end_ms:
            resp = self.session.get(url, params={
                sym_key: symbol, "interval": interval,
                "startTime": start_ms, "endTime": end_ms, "limit": 1000,
            }, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for k in batch:
                rows.append({
                    "open_time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
                    "open": float(k[1]), "high": float(k[2]),
                    "low":  float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]),
                })
            start_ms = batch[-1][0] + 1
            if len(batch) < 1000:
                break
            time.sleep(0.2)

        if not rows:
            logger.warning("No kline data from %s for %s %s", endpoint, symbol, interval)
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("open_time").sort_index()
        df = df[~df.index.duplicated()]
        logger.info("Fetched %d klines (%s %s) from %s", len(df), symbol, interval, endpoint)
        return df

    def _fetch_ls_ratio(self, symbol: str, endpoint: str, period: str = "1h") -> Optional[pd.DataFrame]:
        url  = f"{self.cfg.BINANCE_REST_BASE}{endpoint}"
        rows = []
        try:
            resp = self.session.get(url, params={"symbol": symbol, "period": period, "limit": 500}, timeout=30)
            resp.raise_for_status()
            for item in resp.json():
                rows.append({
                    "timestamp":        pd.Timestamp(item["timestamp"], unit="ms", tz="UTC"),
                    "long_short_ratio": float(item["longShortRatio"]),
                })
        except Exception as exc:
            logger.warning("L/S ratio fetch failed (%s): %s", endpoint, exc)
            return None

        if not rows:
            return None

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df = df[~df.index.duplicated()]
        logger.info("Fetched %d L/S ratio rows from %s", len(df), endpoint)
        return df
