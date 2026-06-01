"""
Backtester tests: P&L identity, NAV consistency, and metrics sanity.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from signals.pipeline import SignalPipeline
from risk.manager import RiskManager
from backtest.runner import Backtester


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

    sp   = SignalPipeline()
    risk = RiskManager()
    df   = sp.run(df)
    df   = risk.compute_sizes(df)
    return df


def test_pnl_identity():
    """net_pnl == pnl_price + pnl_funding - cost_total within $0.01."""
    bt     = Backtester(initial_capital=1_000_000.0)
    result = bt.run(_build_full_df(), "BTCUSDT", "TEST")
    for trade in result.trades:
        expected = trade.pnl_price + trade.pnl_funding - trade.cost_total
        diff     = abs(trade.net_pnl - expected)
        assert diff < 0.01, \
            f"P&L identity violation: net={trade.net_pnl:.4f} expected={expected:.4f} diff={diff:.4f}"


def test_nav_equals_initial_plus_trade_pnls():
    """Final NAV == initial capital + sum of all trade net_pnls."""
    initial = 1_000_000.0
    bt      = Backtester(initial_capital=initial)
    result  = bt.run(_build_full_df(), "BTCUSDT", "TEST")
    expected = initial + sum(t.net_pnl for t in result.trades)
    assert abs(result.nav_series[-1] - expected) < 1.0, \
        f"NAV mismatch: actual={result.nav_series[-1]:.2f} expected={expected:.2f}"


def test_no_open_position_at_end():
    """All positions must be closed at end of data."""
    bt     = Backtester(initial_capital=1_000_000.0)
    result = bt.run(_build_full_df(), "BTCUSDT", "TEST")
    valid_reasons = {"z_revert", "atr_stop", "time_stop", "max_loss", "end_of_data"}
    bad = [t for t in result.trades if t.exit_reason not in valid_reasons]
    assert len(bad) == 0, f"{len(bad)} trades have invalid exit_reason"


def test_nav_never_negative():
    """NAV must never go negative."""
    bt     = Backtester(initial_capital=1_000_000.0)
    result = bt.run(_build_full_df(), "BTCUSDT", "TEST")
    assert (result.nav_series >= 0).all(), "NAV went negative"


def test_trade_count_in_expected_range():
    """Expect at least 5 trades in 1000 bars of synthetic AR(1) funding."""
    bt     = Backtester(initial_capital=1_000_000.0)
    result = bt.run(_build_full_df(), "BTCUSDT", "TEST")
    assert len(result.trades) >= 5, \
        f"Too few trades ({len(result.trades)}) — signal pipeline may be broken"


def test_metrics_compute():
    """compute_metrics must return plausible values."""
    bt      = Backtester(initial_capital=1_000_000.0)
    result  = bt.run(_build_full_df(), "BTCUSDT", "TEST")
    metrics = bt.compute_metrics(result)
    assert 0.0 <= metrics.max_drawdown <= 1.0, "Max drawdown out of [0, 1]"
    assert 0.0 <= metrics.win_rate     <= 1.0, "Win rate out of [0, 1]"
    assert metrics.total_trades == len(result.trades)


def test_trades_to_dataframe_schema():
    """trades_to_dataframe must contain all required columns."""
    bt      = Backtester(initial_capital=1_000_000.0)
    result  = bt.run(_build_full_df(), "BTCUSDT", "TEST")
    df      = bt.trades_to_dataframe(result.trades)
    for col in ["symbol", "direction", "entry_time", "exit_time", "net_pnl", "exit_reason"]:
        assert col in df.columns, f"Missing column in trades DataFrame: {col}"
