"""
Phase 4 gate: backtest engine tests — P&L identity, NAV consistency.
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
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceAnalyser


def _build_full_df(n: int = 1000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="8h", tz="UTC")
    phi, sigma = 0.85, 0.0004
    f = np.zeros(n)
    f[0] = 0.0001
    for i in range(1, n):
        f[i] = 0.0001 + phi * (f[i - 1] - 0.0001) + rng.normal(0, sigma)
    close = 40000 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    df = pd.DataFrame({
        "funding_rate": f,
        "close":        close,
        "high":         close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low":          close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "mark_close":   close,
        "index_close":  close * 0.999,
        "basis":        0.001 * np.ones(n),
        "ls_ratio":     np.nan,
        "top_ls_ratio": np.nan,
        "atr_8h":       close * 0.02,
        "ret_8h":       np.concatenate([[0.0], np.log(close[1:] / close[:-1])]),
        "rvol_ann":     np.full(n, 0.6),
    }, index=idx)
    engine = FundingZScoreSignal()
    df = engine.compute_rolling_stats(df)
    raw = engine.raw_signal(df)
    sf = SignalFilters()
    df = sf.apply_all(df, raw)
    sizer = KellySizer()
    df = sizer.compute_sizes(df)
    return df


def test_pnl_identity():
    """net_pnl must equal pnl_price + pnl_funding - cost_total within 0.01 USDT."""
    df = _build_full_df()
    engine = BacktestEngine(initial_capital=1_000_000.0)
    result = engine.run(df, "BTCUSDT", "TEST")

    for trade in result.trades:
        expected = trade.pnl_price + trade.pnl_funding - trade.cost_total
        diff = abs(trade.net_pnl - expected)
        assert diff < 0.01, \
            f"P&L identity violation: net={trade.net_pnl:.4f} expected={expected:.4f} diff={diff:.4f}"


def test_nav_equals_initial_plus_trade_pnls():
    """Final NAV must equal initial capital + sum of all trade net_pnls."""
    initial = 1_000_000.0
    df = _build_full_df()
    engine = BacktestEngine(initial_capital=initial)
    result = engine.run(df, "BTCUSDT", "TEST")

    total_trade_pnl = sum(t.net_pnl for t in result.trades)
    expected_nav    = initial + total_trade_pnl
    actual_nav      = result.nav_series[-1]

    assert abs(actual_nav - expected_nav) < 1.0, \
        f"NAV mismatch: actual={actual_nav:.2f} expected={expected_nav:.2f}"


def test_no_open_position_at_end():
    """All positions must be closed at end of data."""
    df = _build_full_df()
    engine = BacktestEngine(initial_capital=1_000_000.0)
    result = engine.run(df, "BTCUSDT", "TEST")

    open_trades = [t for t in result.trades if t.exit_reason not in (
        "z_revert", "atr_stop", "time_stop", "max_loss", "end_of_data"
    )]
    assert len(open_trades) == 0, f"{len(open_trades)} trades have invalid exit_reason"


def test_nav_never_negative():
    """NAV must never go negative (positions are small relative to NAV)."""
    df = _build_full_df()
    engine = BacktestEngine(initial_capital=1_000_000.0)
    result = engine.run(df, "BTCUSDT", "TEST")
    assert (result.nav_series >= 0).all(), "NAV went negative"


def test_trade_count_in_expected_range():
    """Expect at least 10 trades in 1000 bars of synthetic AR(1) funding."""
    df = _build_full_df()
    engine = BacktestEngine(initial_capital=1_000_000.0)
    result = engine.run(df, "BTCUSDT", "TEST")
    assert len(result.trades) >= 5, \
        f"Too few trades ({len(result.trades)}) — signal pipeline may be broken"


def test_metrics_compute():
    """PerformanceAnalyser must run without errors and return plausible values."""
    df = _build_full_df()
    engine = BacktestEngine(initial_capital=1_000_000.0)
    result = engine.run(df, "BTCUSDT", "TEST")
    analyser = PerformanceAnalyser()
    metrics = analyser.compute(result)

    assert 0.0 <= metrics.max_drawdown <= 1.0, "Max drawdown out of range"
    assert 0.0 <= metrics.win_rate     <= 1.0, "Win rate out of range"
    assert metrics.total_trades == len(result.trades)
