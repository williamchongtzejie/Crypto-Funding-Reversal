"""
Half-Kelly position sizing with volatility regime scaling and NAV cap.
Section 5.8 of the strategy specification.
"""
import logging

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)


class KellySizer:
    """
    Computes the final position size fraction for each bar using:
      - Half-Kelly criterion estimated from signal returns
      - Binary volatility regime scalar
      - Hard NAV cap of 2%

    All computations use lag-1 confirmed_signal to prevent look-ahead.
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    def compute_sizes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds sizing columns to df and returns the enriched DataFrame.

        Added columns:
          signal_ret   — lag-1 confirmed_signal x ret_8h (signal return series)
          kelly_full   — full Kelly fraction (clipped 0 to KELLY_UPPER_CLIP)
          kelly_half   — 0.5 x kelly_full
          vol_scaled   — kelly_half x vol_scalar (0.5 in high-vol, else 1.0)
          final_size   — min(vol_scaled, NAV_CAP)
        """
        W        = self.cfg.KELLY_EST_WINDOW   # 180 bars
        clip_max = self.cfg.KELLY_UPPER_CLIP   # 0.20
        kf       = self.cfg.KELLY_FRACTION     # 0.50
        vol_thr  = self.cfg.VOL_REGIME_THRESH  # 1.20
        hi_vol   = self.cfg.HIGH_VOL_SCALAR    # 0.50
        nav_cap  = self.cfg.NAV_CAP            # 0.02

        out = df.copy()

        # Step 1 — signal return (lag-1 signal, no look-ahead)
        out["signal_ret"] = out["confirmed_signal"].shift(1) * out["ret_8h"]

        # Steps 2-4 — rolling mean and variance of signal returns
        roll = out["signal_ret"].rolling(window=W, min_periods=W // 2)
        mu_signal  = roll.mean()
        var_signal = roll.var(ddof=1).clip(lower=1e-8)

        # Step 5 — full Kelly, clipped to [0, KELLY_UPPER_CLIP]
        # Direction is handled by signal sign; size is always non-negative
        f_star = (mu_signal / var_signal).clip(lower=0, upper=clip_max)
        out["kelly_full"] = f_star

        # Step 6 — half-Kelly
        out["kelly_half"] = kf * f_star

        # Step 7 — binary vol scalar
        vol_scalar = out["rvol_ann"].apply(lambda v: hi_vol if v > vol_thr else 1.0)
        out["vol_scaled"] = out["kelly_half"] * vol_scalar

        # Step 8 — NAV hard cap
        out["final_size"] = out["vol_scaled"].clip(upper=nav_cap)

        # Zeroise sizing wherever confirmed_signal is flat
        out.loc[out["confirmed_signal"] == 0, "final_size"] = 0.0

        logger.info(
            "Sizing computed: mean_size=%.4f, max_size=%.4f, non-zero=%d bars",
            out["final_size"][out["final_size"] > 0].mean() if (out["final_size"] > 0).any() else 0,
            out["final_size"].max(),
            (out["final_size"] > 0).sum(),
        )
        return out
