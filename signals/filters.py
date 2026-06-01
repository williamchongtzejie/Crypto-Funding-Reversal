"""
Signal filters: L/S ratio, basis, and regime filters.
Each filter can only suppress signals, never generate them.
Sections 5.6 and 5.7 of the strategy specification.
"""
import logging

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)


class SignalFilters:
    """
    Applies the three sequential signal filters to the raw z-score signal.
    All filters return boolean Series (True = pass, False = block).
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Filter A: Long/Short Ratio (Section 5.7)
    # ------------------------------------------------------------------

    def ls_filter(self, df: pd.DataFrame, raw_signal: pd.Series) -> pd.Series:
        """
        Pass when ls_ratio is NaN (historical data has no L/S data).
        For SHORT signal: pass when ls_ratio > LS_SHORT_MIN (1.2) — crowd is net long.
        For LONG signal:  pass when ls_ratio < LS_LONG_MAX  (0.85) — crowd is net short.

        Secondary confirmation via top_ls_ratio (when available):
          For SHORT: block if top_ls_ratio < 1.0 (smart money is net short)
          For LONG:  block if top_ls_ratio > 1.0 (smart money is net long)
        """
        ls  = df.get("ls_ratio",     pd.Series(np.nan, index=df.index))
        tls = df.get("top_ls_ratio", pd.Series(np.nan, index=df.index))

        ok = pd.Series(True, index=df.index)

        # --- Global L/S ratio ---
        is_short = raw_signal == -1
        is_long  = raw_signal ==  1

        # SHORT filter
        short_ls_ok = ls.isna() | (ls > self.cfg.LS_SHORT_MIN)
        ok = ok & (~is_short | short_ls_ok)

        # LONG filter
        long_ls_ok = ls.isna() | (ls < self.cfg.LS_LONG_MAX)
        ok = ok & (~is_long | long_ls_ok)

        # --- Top trader secondary confirmation ---
        short_top_ok = tls.isna() | (tls >= 1.0)   # block if smart money net short
        ok = ok & (~is_short | short_top_ok)

        long_top_ok = tls.isna() | (tls <= 1.0)    # block if smart money net long
        ok = ok & (~is_long | long_top_ok)

        blocked = (~ok & (raw_signal != 0)).sum()
        logger.debug("L/S filter blocked %d signals", blocked)
        return ok

    # ------------------------------------------------------------------
    # Filter B: Perp-Spot Basis (Section 6.1 Stage 3B)
    # ------------------------------------------------------------------

    def basis_filter(self, df: pd.DataFrame, raw_signal: pd.Series) -> pd.Series:
        """
        SHORT signal only: basis must be > BASIS_SHORT_MIN (0.002 = +0.2%) or NaN.
        LONG signal: always passes.
        NaN basis: always passes.
        """
        basis = df.get("basis", pd.Series(np.nan, index=df.index))
        is_short = raw_signal == -1

        basis_ok = basis.isna() | (basis > self.cfg.BASIS_SHORT_MIN)
        ok = ~is_short | basis_ok

        blocked = (~ok & (raw_signal != 0)).sum()
        logger.debug("Basis filter blocked %d signals", blocked)
        return ok

    # ------------------------------------------------------------------
    # Filter C: Regime Filter (Section 5.6)
    # ------------------------------------------------------------------

    def regime_filter(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes a z-score of the 30-day rolling return and blocks new entries
        when abs(z_reg) > REGIME_Z_THRESH (2.5) — indicates parabolic trend.

        Returns True (allow) when NOT trending.
        """
        W_reg  = self.cfg.REGIME_WINDOW       # 90 bars = 30 days
        W_hist = self.cfg.REGIME_HIST_WINDOW  # 180 bars = 60 days
        epsilon = 1e-6

        # 30-day log return series
        ret30d = np.log(df["close"] / df["close"].shift(W_reg))

        # Rolling mean and std of the 30-day return series
        roll = ret30d.rolling(window=W_hist, min_periods=W_hist // 2)
        mu_reg    = roll.mean()
        sigma_reg = roll.std(ddof=1).clip(lower=epsilon)

        z_reg = (ret30d - mu_reg) / sigma_reg
        df_out = df.copy()
        df_out["regime_z"] = z_reg

        regime_ok = z_reg.abs() <= self.cfg.REGIME_Z_THRESH
        # NaN regime z (insufficient history) → allow entries
        regime_ok = regime_ok.fillna(True)

        blocked = (~regime_ok).sum()
        logger.debug("Regime filter blocked %d bars", blocked)
        return regime_ok

    # ------------------------------------------------------------------
    # Apply all filters and produce confirmed signal
    # ------------------------------------------------------------------

    def apply_all(self, df: pd.DataFrame, raw_signal: pd.Series) -> pd.DataFrame:
        """
        Applies all three filters in sequence and returns a DataFrame with columns:
          raw_signal, ls_filter_ok, basis_filter_ok, regime_ok, confirmed_signal

        confirmed_signal = raw_signal where all three filters pass, else 0.
        """
        ls_ok     = self.ls_filter(df, raw_signal)
        basis_ok  = self.basis_filter(df, raw_signal)
        regime_ok = self.regime_filter(df)

        all_pass = ls_ok & basis_ok & regime_ok

        confirmed = raw_signal.copy()
        confirmed[~all_pass] = 0

        result = df.copy()
        result["raw_signal"]      = raw_signal.astype("int8")
        result["ls_filter_ok"]    = ls_ok
        result["basis_filter_ok"] = basis_ok
        result["regime_ok"]       = regime_ok
        result["confirmed_signal"] = confirmed.astype("int8")

        n_raw  = (raw_signal != 0).sum()
        n_conf = (confirmed  != 0).sum()
        logger.info(
            "Filter pipeline: raw=%d signals → confirmed=%d (%.0f%% survival rate)",
            n_raw, n_conf, 100 * n_conf / max(n_raw, 1),
        )
        return result
