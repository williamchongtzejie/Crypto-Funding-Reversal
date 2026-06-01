"""
Shared data containers: trade records, backtest results, and performance metrics.
Imported by backtest, dashboard, and live modules to avoid circular dependencies.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    """One completed round-trip trade."""

    symbol:          str
    direction:       int            # +1 = LONG, -1 = SHORT
    entry_bar:       int
    entry_time:      pd.Timestamp
    entry_price:     float
    entry_size:      float          # fraction of NAV at entry
    entry_notional:  float          # USDT
    atr_at_entry:    float
    exit_bar:        int
    exit_time:       pd.Timestamp
    exit_price:      float
    exit_reason:     str            # z_revert | atr_stop | time_stop | max_loss | end_of_data
    pnl_price:       float
    pnl_funding:     float
    cost_total:      float
    bars_held:       int
    funding_periods: int

    @property
    def net_pnl(self) -> float:
        return self.pnl_price + self.pnl_funding - self.cost_total

    @property
    def gross_pnl(self) -> float:
        return self.pnl_price + self.pnl_funding

    @property
    def return_pct(self) -> float:
        return self.net_pnl / self.entry_notional if self.entry_notional > 0 else 0.0


@dataclass
class BacktestResult:
    """Full output from one Backtester.run() call."""
    symbol:         str
    label:          str            # 'IS' or 'OOS'
    nav_series:     np.ndarray
    returns_series: np.ndarray
    trades:         List[TradeRecord]
    df_enriched:    pd.DataFrame


@dataclass
class StrategyMetrics:
    """All performance metrics returned by Backtester.compute_metrics()."""

    total_return:        float = 0.0
    ann_return:          float = 0.0
    ann_volatility:      float = 0.0
    sharpe_ratio:        float = 0.0
    sortino_ratio:       float = 0.0
    calmar_ratio:        float = 0.0
    max_drawdown:        float = 0.0
    max_dd_duration_days: float = 0.0
    beta_to_btc:         float = 0.0
    funding_fraction:    float = 0.0
    total_trades:        int   = 0
    win_rate:            float = 0.0
    profit_factor:       float = 0.0
    avg_bars_held:       float = 0.0
    avg_hours_held:      float = 0.0
    initial_nav:         float = 0.0
    final_nav:           float = 0.0
