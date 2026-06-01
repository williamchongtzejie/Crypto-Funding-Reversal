"""
Data processing pipeline: aligns raw datasets, computes derived features,
and returns a single master DataFrame aligned to the 8h funding schedule.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)

_REQUIRED_COLS = [
    "funding_rate", "open", "high", "low", "close", "volume",
    "mark_close", "index_close", "basis", "atr_8h", "ret_8h", "rvol_ann",
]


class DataProcessor:
    """
    Builds the master DataFrame from raw fetched datasets.
    All features are computed here. No feature computation elsewhere.
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    def build(self, raw: dict) -> pd.DataFrame:
        """
        Accepts the dict returned by BinanceFetcher.fetch_all() and produces
        a clean master DataFrame indexed on the 8h funding settlement schedule.

        Steps (in spec order):
          1-2.  Rename + index funding rates
          3.    Join 8h OHLCV
          4.    Join mark_close
          5.    Join index_close (fallback to close if unavailable)
          6.    Compute basis
          7.    Join ls_ratio
          8.    Join top_ls_ratio
          9.    Compute ATR
          10.   Compute ret_8h
          11.   Compute rvol_ann
          12.   Forward-fill gaps (max 3 bars)
          13.   Drop rows without price or funding
        """
        funding = raw.get("funding")
        if funding is None or funding.empty:
            raise ValueError("Funding rate data is required and is missing.")

        # 1-2. Funding rate as the master index
        # Round to the nearest second to absorb the millisecond offsets Binance
        # sometimes embeds in fundingTime (e.g. 00:00:00.002 instead of 00:00:00).
        df = funding[["funding_rate"]].copy()
        df.index = df.index.floor("s")
        df = df[~df.index.duplicated(keep="last")]
        df.index.name = "time"

        # 3. Join 8h klines: open, high, low, close, volume
        klines_8h = raw.get("klines_8h")
        if klines_8h is not None and not klines_8h.empty:
            df = df.join(klines_8h[["open", "high", "low", "close", "volume"]], how="left")
        else:
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = np.nan

        # 4. Mark price close
        mark_8h = raw.get("mark_8h")
        if mark_8h is not None and not mark_8h.empty:
            df = df.join(mark_8h[["close"]].rename(columns={"close": "mark_close"}), how="left")
        else:
            df["mark_close"] = df["close"]

        # 5. Index price close (fallback to close)
        index_8h = raw.get("index_8h")
        if index_8h is not None and not index_8h.empty:
            df = df.join(index_8h[["close"]].rename(columns={"close": "index_close"}), how="left")
        else:
            logger.warning("No index price data; using perpetual close as fallback.")
            df["index_close"] = df["close"]

        # 6. Perp-spot basis (Section 5.4)
        df["basis"] = (df["mark_close"] - df["index_close"]) / df["index_close"]

        # 7. Global Long/Short account ratio (NaN for historical)
        ls_global = raw.get("ls_global")
        if ls_global is not None and not ls_global.empty:
            df = df.join(
                ls_global[["long_short_ratio"]].rename(columns={"long_short_ratio": "ls_ratio"}),
                how="left",
            )
        else:
            df["ls_ratio"] = np.nan

        # 8. Top trader L/S position ratio (NaN for historical)
        ls_top = raw.get("ls_top")
        if ls_top is not None and not ls_top.empty:
            df = df.join(
                ls_top[["long_short_ratio"]].rename(columns={"long_short_ratio": "top_ls_ratio"}),
                how="left",
            )
        else:
            df["top_ls_ratio"] = np.nan

        # 12. Forward-fill price columns up to 3 consecutive NaN
        # Done BEFORE computing ret_8h and ATR so derived features use clean prices.
        price_cols = ["open", "high", "low", "close", "volume", "mark_close", "index_close"]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill(limit=3)

        # 13. Drop rows without essential data
        df.dropna(subset=["close", "funding_rate"], inplace=True)

        # 9. ATR (EWM span=14 on 8h bars, Section 5.3)
        df["tr"] = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"]  - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_8h"] = df["tr"].ewm(span=self.cfg.ATR_PERIOD, adjust=False).mean()
        df.drop(columns=["tr"], inplace=True)

        # 10. Log return per 8h bar (Section 5.5 Step 1)
        df["ret_8h"] = np.log(df["close"] / df["close"].shift(1))

        # 11. Annualised realised volatility (Section 5.5)
        W_vol = self.cfg.FUNDING_Z_WINDOW  # 90 bars = 30 days
        df["rvol_ann"] = (
            df["ret_8h"]
            .rolling(window=W_vol, min_periods=W_vol // 2)
            .std(ddof=1)
            * np.sqrt(self.cfg.BARS_PER_YEAR)
        )

        # Validate required columns are present
        missing = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Master DataFrame missing required columns: {missing}")

        logger.info("Master DataFrame built: %d bars, %d columns", len(df), len(df.columns))
        return df
