"""
Phase 3 gate: sizing engine tests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from signals.funding_zscore import FundingZScoreSignal
from signals.filters import SignalFilters
from risk.sizing import KellySizer


def _make_enriched_df(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=n, freq="8h", tz="UTC")
    phi, sigma = 0.85, 0.0003
    f = np.zeros(n)
    f[0] = 0.0001
    for i in range(1, n):
        f[i] = 0.0001 + phi * (f[i - 1] - 0.0001) + rng.normal(0, sigma)
    close = 40000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    rvol  = np.where(rng.random(n) < 0.2, 1.5, 0.6)  # 20% high-vol bars

    df = pd.DataFrame({
        "funding_rate": f,
        "close":        close,
        "high":         close * 1.005,
        "low":          close * 0.995,
        "mark_close":   close,
        "index_close":  close * 0.999,
        "basis":        close * 0.001 / close,
        "ls_ratio":     np.nan,
        "top_ls_ratio": np.nan,
        "atr_8h":       close * 0.02,
        "ret_8h":       np.concatenate([[0.0], np.log(close[1:] / close[:-1])]),
        "rvol_ann":     rvol,
    }, index=idx)

    engine = FundingZScoreSignal()
    df = engine.compute_rolling_stats(df)
    raw = engine.raw_signal(df)
    sf = SignalFilters()
    df = sf.apply_all(df, raw)
    return df


def test_final_size_never_exceeds_nav_cap():
    df = _make_enriched_df()
    sizer = KellySizer()
    out = sizer.compute_sizes(df)
    assert (out["final_size"] <= CONFIG.NAV_CAP + 1e-9).all(), \
        f"final_size exceeds NAV_CAP: max={out['final_size'].max():.6f}"


def test_final_size_zero_when_signal_flat():
    df = _make_enriched_df()
    sizer = KellySizer()
    out = sizer.compute_sizes(df)
    flat_mask = out["confirmed_signal"] == 0
    assert (out.loc[flat_mask, "final_size"] == 0.0).all(), \
        "final_size must be 0 wherever confirmed_signal is 0"


def test_vol_scaled_less_than_kelly_half_in_high_vol():
    df = _make_enriched_df()
    sizer = KellySizer()
    out = sizer.compute_sizes(df)
    high_vol = out["rvol_ann"] > CONFIG.VOL_REGIME_THRESH
    pos_signal = out["confirmed_signal"] != 0
    mask = high_vol & pos_signal & (out["kelly_half"] > 0)
    if mask.any():
        assert (out.loc[mask, "vol_scaled"] < out.loc[mask, "kelly_half"] + 1e-9).all(), \
            "vol_scaled must be <= kelly_half in high-vol regime"


def test_final_size_non_negative():
    df = _make_enriched_df()
    sizer = KellySizer()
    out = sizer.compute_sizes(df)
    assert (out["final_size"] >= 0.0).all(), "final_size must always be non-negative"
