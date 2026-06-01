"""
Phase 2 gate: signal engine tests (no API, synthetic data).
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


def _make_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="8h", tz="UTC")
    phi, sigma = 0.85, 0.0003
    f = np.zeros(n)
    f[0] = 0.0001
    for i in range(1, n):
        f[i] = 0.0001 + phi * (f[i - 1] - 0.0001) + rng.normal(0, sigma)
    close = 40000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
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
        "ret_8h":       np.log(close / np.roll(close, 1)),
        "rvol_ann":     np.full(n, 0.6),
    }, index=idx)


def test_zscore_fires_within_expected_range():
    df = _make_df()
    engine = FundingZScoreSignal()
    df = engine.compute_rolling_stats(df)
    raw = engine.raw_signal(df)

    short_pct = 100 * (raw == -1).sum() / len(raw.dropna())
    long_pct  = 100 * (raw ==  1).sum() / len(raw.dropna())

    assert short_pct >= 1.0,  f"SHORT fires {short_pct:.1f}% — too rare (expect >= 1%)"
    assert short_pct <= 15.0, f"SHORT fires {short_pct:.1f}% — too frequent (expect <= 15%)"
    assert long_pct  >= 1.0,  f"LONG fires {long_pct:.1f}% — too rare"


def test_raw_signal_dtype():
    df = _make_df()
    engine = FundingZScoreSignal()
    df = engine.compute_rolling_stats(df)
    raw = engine.raw_signal(df)
    assert raw.dtype == np.int8, f"Expected int8, got {raw.dtype}"


def test_filter_outputs_are_bool():
    df = _make_df()
    engine = FundingZScoreSignal()
    df = engine.compute_rolling_stats(df)
    raw = engine.raw_signal(df)

    sf = SignalFilters()
    result = sf.apply_all(df, raw)

    assert result["ls_filter_ok"].dtype == bool, "ls_filter_ok must be bool"
    assert result["basis_filter_ok"].dtype == bool, "basis_filter_ok must be bool"
    assert result["regime_ok"].dtype == bool, "regime_ok must be bool"


def test_confirmed_signal_dtype():
    df = _make_df()
    engine = FundingZScoreSignal()
    df = engine.compute_rolling_stats(df)
    raw = engine.raw_signal(df)
    sf = SignalFilters()
    result = sf.apply_all(df, raw)
    assert result["confirmed_signal"].dtype == np.int8


def test_no_lookahead():
    """
    Shift confirmed_signal back by 1 and verify it has no correlation
    to the CURRENT bar's return (only to the NEXT bar's return matters).
    This is a necessary (not sufficient) no-lookahead check.
    """
    df = _make_df(1000)
    engine = FundingZScoreSignal()
    df = engine.compute_rolling_stats(df)
    raw = engine.raw_signal(df)
    sf = SignalFilters()
    result = sf.apply_all(df, raw)

    sig_shifted_back = result["confirmed_signal"].shift(-1)
    ret = df["ret_8h"]
    corr = sig_shifted_back.corr(ret)
    assert abs(corr) < 0.25, f"Potential lookahead: signal(t+1) corr with ret(t) = {corr:.3f}"


def test_filters_never_generate_signals():
    """Confirmed signal must always be a strict subset of raw signal."""
    df = _make_df()
    engine = FundingZScoreSignal()
    df = engine.compute_rolling_stats(df)
    raw = engine.raw_signal(df)
    sf = SignalFilters()
    result = sf.apply_all(df, raw)

    raw_flat  = result["raw_signal"] == 0
    conf_nonflat = result["confirmed_signal"] != 0
    assert not (raw_flat & conf_nonflat).any(), "Filters generated a signal where raw was 0"
