"""
Performance metrics computation for Strategy 3.
All formulas exactly match Section 5.10 of the specification.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config import CONFIG
from backtest.engine import BacktestResult, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """All Section 5.10 performance metrics as typed fields."""

    # Returns
    total_return:       float = 0.0
    ann_return:         float = 0.0
    ann_volatility:     float = 0.0

    # Risk-adjusted
    sharpe_ratio:       float = 0.0
    sortino_ratio:      float = 0.0
    calmar_ratio:       float = 0.0

    # Drawdown
    max_drawdown:       float = 0.0
    max_dd_duration_days: float = 0.0

    # Market neutral
    beta_to_btc:        float = 0.0

    # Strategy-specific
    funding_fraction:   float = 0.0

    # Trade statistics
    total_trades:       int   = 0
    win_rate:           float = 0.0
    profit_factor:      float = 0.0
    avg_bars_held:      float = 0.0
    avg_hours_held:     float = 0.0

    # Capital
    initial_nav:        float = 0.0
    final_nav:          float = 0.0


class PerformanceAnalyser:
    """
    Computes all strategy metrics from a BacktestResult.
    Uses BARS_PER_YEAR = 1095 for all annualisation.
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    def compute(self, result: BacktestResult, btc_returns: Optional[pd.Series] = None) -> StrategyMetrics:
        """
        Compute all Section 5.10 metrics from a BacktestResult.

        btc_returns: optional pd.Series of BTC 8h log-returns for beta computation.
        """
        K   = self.cfg.BARS_PER_YEAR  # 1095
        nav = result.nav_series
        ret = result.returns_series
        initial = nav[0] if nav[0] > 0 else self.cfg.INITIAL_CAPITAL

        ret_clean = ret[~np.isnan(ret)]
        n = len(ret_clean)

        # Total and annualised return
        total_return = (nav[-1] / initial) - 1
        n_years = n / K
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        # Annualised volatility
        ann_vol = float(np.std(ret_clean, ddof=1)) * np.sqrt(K) if n > 1 else 0.0

        # Sharpe ratio (r_f = 0)
        sharpe = (float(np.mean(ret_clean)) / float(np.std(ret_clean, ddof=1)) * np.sqrt(K)
                  if np.std(ret_clean, ddof=1) > 0 else 0.0)

        # Sortino ratio
        neg_ret = ret_clean[ret_clean < 0]
        sigma_down = float(np.std(neg_ret, ddof=1)) * np.sqrt(K) if len(neg_ret) > 1 else 0.0
        sortino = ann_return / sigma_down if sigma_down > 0 else 0.0

        # Maximum drawdown
        peak = np.maximum.accumulate(nav)
        dd = np.where(peak > 0, (peak - nav) / peak, 0.0)
        max_dd = float(np.max(dd))

        # MDD duration (longest contiguous bars below peak)
        in_dd = dd > 0
        max_dd_bars = 0
        current_run = 0
        for flag in in_dd:
            if flag:
                current_run += 1
                max_dd_bars = max(max_dd_bars, current_run)
            else:
                current_run = 0
        max_dd_days = max_dd_bars / self.cfg.BARS_PER_DAY

        # Calmar ratio
        calmar = ann_return / max_dd if max_dd > 0 else 0.0

        # Beta to BTC
        beta = 0.0
        if btc_returns is not None and len(btc_returns) >= len(ret_clean):
            btc = btc_returns.values[-len(ret_clean):]
            cov = np.cov(ret_clean, btc, ddof=1)[0, 1]
            var_btc = np.var(btc, ddof=1)
            beta = cov / var_btc if var_btc > 0 else 0.0

        # Funding fraction
        total_price_pnl   = sum(t.pnl_price   for t in result.trades)
        total_funding_pnl = sum(t.pnl_funding  for t in result.trades)
        gross_pnl_sum     = total_price_pnl + total_funding_pnl
        funding_fraction  = total_funding_pnl / gross_pnl_sum if abs(gross_pnl_sum) > 0 else 0.0

        # Trade-level metrics
        trades = result.trades
        net_pnls  = [t.net_pnl for t in trades]
        winners   = [p for p in net_pnls if p > 0]
        losers    = [p for p in net_pnls if p <= 0]
        win_rate  = len(winners) / len(net_pnls) if net_pnls else 0.0
        gross_win = sum(winners)
        gross_loss = abs(sum(losers))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

        bars_held_list = [t.bars_held for t in trades]
        avg_bars  = float(np.mean(bars_held_list)) if bars_held_list else 0.0

        return StrategyMetrics(
            total_return        = total_return,
            ann_return          = ann_return,
            ann_volatility      = ann_vol,
            sharpe_ratio        = sharpe,
            sortino_ratio       = sortino,
            calmar_ratio        = calmar,
            max_drawdown        = max_dd,
            max_dd_duration_days = max_dd_days,
            beta_to_btc         = beta,
            funding_fraction    = funding_fraction,
            total_trades        = len(trades),
            win_rate            = win_rate,
            profit_factor       = pf,
            avg_bars_held       = avg_bars,
            avg_hours_held      = avg_bars * 8.0,
            initial_nav         = float(initial),
            final_nav           = float(nav[-1]),
        )

    def rolling_sharpe(self, result: BacktestResult, window_months: int = 6) -> pd.Series:
        """
        Compute rolling N-month Sharpe ratio for signal decay monitoring.
        window_bars = window_months x 30 x 3
        """
        K = self.cfg.BARS_PER_YEAR
        window_bars = window_months * 30 * self.cfg.BARS_PER_DAY
        ret = pd.Series(result.returns_series, index=result.df_enriched.index)
        roll = ret.rolling(window=window_bars, min_periods=window_bars // 2)
        rolling_sr = roll.mean() / roll.std(ddof=1) * np.sqrt(K)
        return rolling_sr

    def benchmark_comparison(self, result: BacktestResult, bh_returns: pd.Series) -> dict:
        """Compute beta, annualised alpha, and correlation vs buy-and-hold."""
        K = self.cfg.BARS_PER_YEAR
        strat = pd.Series(result.returns_series, index=result.df_enriched.index)
        aligned_strat, aligned_bh = strat.align(bh_returns, join="inner")

        s = aligned_strat.dropna().values
        b = aligned_bh.dropna().values
        min_len = min(len(s), len(b))
        s, b = s[:min_len], b[:min_len]

        if min_len < 2:
            return {"beta": 0.0, "alpha_ann": 0.0, "correlation": 0.0}

        cov_matrix = np.cov(s, b, ddof=1)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0
        alpha_ann = (np.mean(s) - beta * np.mean(b)) * K
        corr = float(np.corrcoef(s, b)[0, 1])

        return {"beta": beta, "alpha_ann": alpha_ann, "correlation": corr}

    def trades_to_dataframe(self, trades: List[TradeRecord]) -> pd.DataFrame:
        """Convert trades list to DataFrame with Section 10.4 column schema."""
        if not trades:
            return pd.DataFrame()

        rows = [{
            "symbol":       t.symbol,
            "direction":    t.direction,
            "entry_time":   t.entry_time,
            "exit_time":    t.exit_time,
            "entry_price":  t.entry_price,
            "exit_price":   t.exit_price,
            "bars_held":    t.bars_held,
            "hours_held":   t.bars_held * 8.0,
            "size_pct_nav": t.entry_size,
            "notional":     t.entry_notional,
            "pnl_price":    t.pnl_price,
            "pnl_funding":  t.pnl_funding,
            "cost_total":   t.cost_total,
            "net_pnl":      t.net_pnl,
            "gross_pnl":    t.gross_pnl,
            "return_pct":   t.return_pct,
            "exit_reason":  t.exit_reason,
        } for t in trades]

        return pd.DataFrame(rows)

    def print_report(self, m: StrategyMetrics, label: str = "") -> None:
        """Print a formatted performance report to stdout."""
        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  STRATEGY 3 PERFORMANCE REPORT  {label}")
        print(sep)
        print(f"  Total Return:          {m.total_return:>10.2%}")
        print(f"  Ann. Return:           {m.ann_return:>10.2%}")
        print(f"  Ann. Volatility:       {m.ann_volatility:>10.2%}")
        print(f"  Sharpe Ratio:          {m.sharpe_ratio:>10.3f}")
        print(f"  Sortino Ratio:         {m.sortino_ratio:>10.3f}")
        print(f"  Calmar Ratio:          {m.calmar_ratio:>10.3f}")
        print(f"  Max Drawdown:          {m.max_drawdown:>10.2%}")
        print(f"  Max DD Duration:       {m.max_dd_duration_days:>8.0f} days")
        print(f"  Beta to BTC:           {m.beta_to_btc:>10.3f}")
        print(f"  Funding Fraction:      {m.funding_fraction:>10.2%}")
        print(sep)
        print(f"  Total Trades:          {m.total_trades:>10d}")
        print(f"  Win Rate:              {m.win_rate:>10.2%}")
        print(f"  Profit Factor:         {m.profit_factor:>10.2f}")
        print(f"  Avg Hold (bars):       {m.avg_bars_held:>10.1f}")
        print(f"  Avg Hold (hours):      {m.avg_hours_held:>10.1f}")
        print(sep)
        print(f"  Initial NAV:           {m.initial_nav:>12,.2f}")
        print(f"  Final NAV:             {m.final_nav:>12,.2f}")
        print(f"{sep}\n")
