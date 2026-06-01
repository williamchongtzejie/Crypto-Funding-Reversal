"""
Funding rate z-score computation and raw directional signal.
Section 5.2 of the strategy specification.
"""
import logging

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)


class FundingZScoreSignal:
    """
    Computes the rolling z-score of the funding rate and converts it to
    a raw directional signal before any filters are applied.
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    def compute_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling mean, std, and z-score of funding_rate.

        Uses rolling(window=W, min_periods=W//2) per spec to handle warmup.
        Returns a copy of df with added columns:
          funding_mu     — rolling mean (W=90)
          funding_sigma  — rolling sample std (ddof=1)
          funding_zscore — (funding_rate - mu) / max(sigma, epsilon)
        """
        W = self.cfg.FUNDING_Z_WINDOW
        epsilon = self.cfg.Z_SCORE_EPSILON

        out = df.copy()
        roll = out["funding_rate"].rolling(window=W, min_periods=W // 2)
        out["funding_mu"]    = roll.mean()
        out["funding_sigma"] = roll.std(ddof=1)

        sigma_star = out["funding_sigma"].clip(lower=epsilon)
        out["funding_zscore"] = (out["funding_rate"] - out["funding_mu"]) / sigma_star

        logger.debug(
            "Z-score computed: non-null=%d, range=[%.3f, %.3f]",
            out["funding_zscore"].notna().sum(),
            out["funding_zscore"].min(),
            out["funding_zscore"].max(),
        )
        return out

    def raw_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Convert funding_zscore to a raw directional signal.

        z_t >  Z_SHORT_ENTRY (+2.0)  ->  -1  (SHORT)
        z_t <  Z_LONG_ENTRY  (-1.5)  ->  +1  (LONG)
        otherwise                    ->   0  (FLAT)

        Returns: pd.Series[int8] aligned to df.index.
        """
        z = df["funding_zscore"]
        signal = pd.Series(0, index=df.index, dtype="int8")
        signal = signal.where(z <= self.cfg.Z_SHORT_ENTRY, other=-1)
        signal = signal.where(z >= self.cfg.Z_LONG_ENTRY,  other=1)

        n_short = (signal == -1).sum()
        n_long  = (signal ==  1).sum()
        n_total = len(signal.dropna())
        logger.info(
            "Raw signal: SHORT=%d (%.1f%%), LONG=%d (%.1f%%), FLAT=%d",
            n_short, 100 * n_short / max(n_total, 1),
            n_long,  100 * n_long  / max(n_total, 1),
            n_total - n_short - n_long,
        )
        return signal
