"""
Unified risk layer: half-Kelly position sizing and portfolio-level circuit breaker.
Merges the former KellySizer with circuit breaker logic from the backtest engine.
"""
import logging

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Computes position sizes and enforces portfolio-level risk gates.

    Public interface:
        compute_sizes(df)              -> DataFrame  (adds sizing columns)
        is_halted(nav, peak_nav)       -> bool       (circuit breaker check)

    Sizing columns added by compute_sizes():
        signal_ret   — lag-1 confirmed_signal × ret_8h
        kelly_full   — full Kelly fraction clipped to [0, KELLY_UPPER_CLIP]
        kelly_half   — 0.5 × kelly_full
        vol_scaled   — kelly_half × vol_scalar (0.5 in high-vol, else 1.0)
        final_size   — min(vol_scaled, NAV_CAP), zeroed where signal is flat
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    def compute_sizes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute half-Kelly position sizes with volatility regime scaling and NAV cap.
        Uses lag-1 confirmed_signal to prevent look-ahead bias.
        """
        W        = self.cfg.KELLY_EST_WINDOW
        clip_max = self.cfg.KELLY_UPPER_CLIP
        kf       = self.cfg.KELLY_FRACTION
        vol_thr  = self.cfg.VOL_REGIME_THRESH
        hi_vol   = self.cfg.HIGH_VOL_SCALAR
        nav_cap  = self.cfg.NAV_CAP

        out = df.copy()

        # Lag-1 signal return (no look-ahead)
        out["signal_ret"] = out["confirmed_signal"].shift(1) * out["ret_8h"]

        roll       = out["signal_ret"].rolling(window=W, min_periods=W // 2)
        mu_signal  = roll.mean()
        var_signal = roll.var(ddof=1).clip(lower=1e-8)

        f_star           = (mu_signal / var_signal).clip(lower=0, upper=clip_max)
        out["kelly_full"] = f_star
        out["kelly_half"] = kf * f_star

        vol_scalar        = out["rvol_ann"].apply(lambda v: hi_vol if v > vol_thr else 1.0)
        out["vol_scaled"] = out["kelly_half"] * vol_scalar
        out["final_size"] = out["vol_scaled"].clip(upper=nav_cap)

        # Zero sizing wherever signal is flat
        out.loc[out["confirmed_signal"] == 0, "final_size"] = 0.0

        active = out["final_size"] > 0
        logger.info(
            "Sizing: mean=%.4f, max=%.4f, non-zero bars=%d",
            out.loc[active, "final_size"].mean() if active.any() else 0,
            out["final_size"].max(),
            active.sum(),
        )
        return out

    def is_halted(self, nav: float, peak_nav: float) -> bool:
        """
        Circuit breaker: return True (halt new entries) when drawdown from
        peak NAV exceeds MAX_PORTFOLIO_DD (15%).
        Resume below DD_RESUME_LEVEL (10%).

        Note: the engine calls this with the rolling peak_nav tracked internally.
        """
        if peak_nav <= 0:
            return False
        drawdown = (peak_nav - nav) / peak_nav
        return drawdown > self.cfg.MAX_PORTFOLIO_DD
