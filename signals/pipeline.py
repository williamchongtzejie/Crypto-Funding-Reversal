"""
Unified signal layer: funding z-score computation, raw signal generation,
and all three filters (L/S ratio, basis, regime) in one class.
Merges the former FundingZScoreSignal and SignalFilters.
"""
import logging

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)


class SignalPipeline:
    """
    Computes the full signal stack from a raw master DataFrame.

    Public interface:
        run(df)                          -> DataFrame  (all-in-one)
        compute_zscore(df)               -> DataFrame  (adds funding_mu/sigma/zscore)
        raw_signal(df)                   -> Series     (int8: -1, 0, +1)
        apply_filters(df, raw_signal)    -> DataFrame  (adds filter flags + confirmed_signal)

    Individual filter methods (each returns a bool Series — True = pass):
        ls_filter(df, raw_signal)
        basis_filter(df, raw_signal)
        regime_filter(df)
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # All-in-one
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete signal pipeline:
          1. Compute rolling z-score stats
          2. Generate raw directional signal
          3. Apply all three filters → confirmed_signal
        Returns enriched DataFrame with all signal columns added.
        """
        df = self.compute_zscore(df)
        raw = self.raw_signal(df)
        df = self.apply_filters(df, raw)
        return df

    # ------------------------------------------------------------------
    # Z-Score (formerly FundingZScoreSignal)
    # ------------------------------------------------------------------

    def compute_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling mean, std, and z-score of funding_rate.

        Adds columns: funding_mu, funding_sigma, funding_zscore.
        Window: W=90 bars (30 days), min_periods=W//2.
        Sigma is clipped to Z_SCORE_EPSILON to prevent division by zero.
        """
        W       = self.cfg.FUNDING_Z_WINDOW
        epsilon = self.cfg.Z_SCORE_EPSILON

        out  = df.copy()
        roll = out["funding_rate"].rolling(window=W, min_periods=W // 2)
        out["funding_mu"]    = roll.mean()
        out["funding_sigma"] = roll.std(ddof=1)

        sigma_star = out["funding_sigma"].clip(lower=epsilon)
        out["funding_zscore"] = (out["funding_rate"] - out["funding_mu"]) / sigma_star

        logger.debug(
            "Z-score: non-null=%d, range=[%.3f, %.3f]",
            out["funding_zscore"].notna().sum(),
            out["funding_zscore"].min(),
            out["funding_zscore"].max(),
        )
        return out

    def raw_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Convert funding_zscore to a raw directional signal (int8).

        z > +Z_SHORT_ENTRY  ->  -1  (SHORT)
        z < +Z_LONG_ENTRY   ->  +1  (LONG)
        otherwise           ->   0  (FLAT)
        """
        z      = df["funding_zscore"]
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

    # ------------------------------------------------------------------
    # Filters (formerly SignalFilters)
    # ------------------------------------------------------------------

    def ls_filter(self, df: pd.DataFrame, raw_signal: pd.Series) -> pd.Series:
        """
        Filter A — Long/Short Ratio.
        SHORT: pass when ls_ratio > LS_SHORT_MIN (1.2) or NaN.
        LONG:  pass when ls_ratio < LS_LONG_MAX  (0.85) or NaN.
        Secondary: blocks if top_ls_ratio contradicts direction.
        NaN always passes (historical data has no L/S).
        """
        ls  = df.get("ls_ratio",     pd.Series(np.nan, index=df.index))
        tls = df.get("top_ls_ratio", pd.Series(np.nan, index=df.index))

        ok       = pd.Series(True, index=df.index)
        is_short = raw_signal == -1
        is_long  = raw_signal ==  1

        ok = ok & (~is_short | ls.isna() | (ls > self.cfg.LS_SHORT_MIN))
        ok = ok & (~is_long  | ls.isna() | (ls < self.cfg.LS_LONG_MAX))
        ok = ok & (~is_short | tls.isna() | (tls >= 1.0))
        ok = ok & (~is_long  | tls.isna() | (tls <= 1.0))

        logger.debug("L/S filter blocked %d signals", (~ok & (raw_signal != 0)).sum())
        return ok

    def basis_filter(self, df: pd.DataFrame, raw_signal: pd.Series) -> pd.Series:
        """
        Filter B — Perp-Spot Basis.
        SHORT only: requires basis > BASIS_SHORT_MIN (+0.2%) or NaN.
        LONG: always passes.
        """
        basis    = df.get("basis", pd.Series(np.nan, index=df.index))
        is_short = raw_signal == -1
        ok       = ~is_short | basis.isna() | (basis > self.cfg.BASIS_SHORT_MIN)

        logger.debug("Basis filter blocked %d signals", (~ok & (raw_signal != 0)).sum())
        return ok

    def regime_filter(self, df: pd.DataFrame) -> pd.Series:
        """
        Filter C — Regime (Trend Guard).
        Computes z-score of the 30-day rolling log return.
        Blocks entries when |z_reg| > REGIME_Z_THRESH (2.5) — parabolic trend.
        NaN (insufficient history) passes through.
        """
        W_reg  = self.cfg.REGIME_WINDOW
        W_hist = self.cfg.REGIME_HIST_WINDOW
        eps    = 1e-6

        ret30d    = np.log(df["close"] / df["close"].shift(W_reg))
        roll      = ret30d.rolling(window=W_hist, min_periods=W_hist // 2)
        mu_reg    = roll.mean()
        sigma_reg = roll.std(ddof=1).clip(lower=eps)
        z_reg     = (ret30d - mu_reg) / sigma_reg

        regime_ok = (z_reg.abs() <= self.cfg.REGIME_Z_THRESH).fillna(True)
        logger.debug("Regime filter blocked %d bars", (~regime_ok).sum())
        return regime_ok

    def apply_filters(self, df: pd.DataFrame, raw_signal: pd.Series) -> pd.DataFrame:
        """
        Apply all three filters sequentially and produce the confirmed signal.

        Adds columns: raw_signal, ls_filter_ok, basis_filter_ok, regime_ok,
                      confirmed_signal (int8).
        confirmed_signal = raw_signal where all filters pass, else 0.
        """
        ls_ok     = self.ls_filter(df, raw_signal)
        basis_ok  = self.basis_filter(df, raw_signal)
        regime_ok = self.regime_filter(df)
        all_pass  = ls_ok & basis_ok & regime_ok

        confirmed = raw_signal.copy()
        confirmed[~all_pass] = 0

        result = df.copy()
        result["raw_signal"]       = raw_signal.astype("int8")
        result["ls_filter_ok"]     = ls_ok
        result["basis_filter_ok"]  = basis_ok
        result["regime_ok"]        = regime_ok
        result["confirmed_signal"] = confirmed.astype("int8")

        n_raw  = (raw_signal != 0).sum()
        n_conf = (confirmed  != 0).sum()
        logger.info(
            "Filter pipeline: raw=%d → confirmed=%d (%.0f%% survival)",
            n_raw, n_conf, 100 * n_conf / max(n_raw, 1),
        )
        return result
