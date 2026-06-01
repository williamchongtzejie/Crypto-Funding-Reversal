"""
Unified backtest layer: bar-by-bar simulation and performance analytics in one class.
Merges the former BacktestEngine and PerformanceAnalyser.
"""
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from config import CONFIG
from models import TradeRecord, BacktestResult, StrategyMetrics

logger = logging.getLogger(__name__)


class Backtester:
    """
    Runs bar-by-bar backtests and computes all performance metrics.

    Public interface:
        run(df, symbol, label)                   -> BacktestResult
        compute_metrics(result, btc_returns)     -> StrategyMetrics
        rolling_sharpe(result, window_months)    -> pd.Series
        trades_to_dataframe(trades)              -> pd.DataFrame
        print_report(metrics, label)
        export(result, metrics, results_dir, symbol, label)

    Conservative assumptions enforced in run():
        1. Entry at bar t+1 (signal fires at t, fill at t+1)
        2. Exit at bar close
        3. ATR stop checked once per bar, at close
        4. Fixed 7 bps cost per side
        5. No funding income on the entry bar
        6. L/S filter passes when NaN (historical data)
        7. No partial fills
    """

    def __init__(self, initial_capital: Optional[float] = None, cfg=CONFIG):
        self.cfg             = cfg
        self.initial_capital = initial_capital or cfg.INITIAL_CAPITAL

    # ------------------------------------------------------------------
    # Simulation (formerly BacktestEngine)
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, symbol: str, label: str) -> BacktestResult:
        """
        Run the bar-by-bar simulation on an enriched master DataFrame.
        df must contain: confirmed_signal, final_size, funding_rate,
                         mark_close, funding_zscore, atr_8h.
        """
        n       = len(df)
        nav_arr = np.zeros(n)
        ret_arr = np.zeros(n)

        nav        = float(self.initial_capital)
        nav_arr[0] = nav

        position       = 0
        entry_bar      = None
        entry_price    = None
        entry_notional = None
        atr_at_entry   = None
        bars_held      = 0
        cum_pnl_price  = 0.0
        cum_pnl_fund   = 0.0
        trade_cost     = 0.0
        trades: List[TradeRecord] = []

        cost_per_side = self.cfg.COST_PER_SIDE

        def execute_exit(t: int, reason: str) -> None:
            nonlocal nav, position, entry_bar, entry_price, entry_notional
            nonlocal atr_at_entry, bars_held, cum_pnl_price, cum_pnl_fund, trade_cost

            exit_cost   = entry_notional * cost_per_side
            nav        -= exit_cost
            trade_cost += exit_cost

            trades.append(TradeRecord(
                symbol          = symbol,
                direction       = position,
                entry_bar       = entry_bar,
                entry_time      = df.index[entry_bar],
                entry_price     = entry_price,
                entry_size      = entry_notional / nav_arr[entry_bar] if nav_arr[entry_bar] > 0 else 0.0,
                entry_notional  = entry_notional,
                atr_at_entry    = atr_at_entry,
                exit_bar        = t,
                exit_time       = df.index[t],
                exit_price      = float(df.iloc[t]["mark_close"]),
                exit_reason     = reason,
                pnl_price       = cum_pnl_price,
                pnl_funding     = cum_pnl_fund,
                cost_total      = trade_cost,
                bars_held       = bars_held,
                funding_periods = bars_held,
            ))
            logger.debug(
                "EXIT %s %s t=%d reason=%s net_pnl=%.2f",
                symbol, "LONG" if position == 1 else "SHORT",
                t, reason, trades[-1].net_pnl,
            )

            position = 0
            entry_bar = entry_price = entry_notional = atr_at_entry = None
            bars_held = 0
            cum_pnl_price = cum_pnl_fund = trade_cost = 0.0

        for t in range(1, n):
            bar  = df.iloc[t]
            prev = df.iloc[t - 1]

            if position != 0:
                price_ret      = (bar["mark_close"] - prev["mark_close"]) / prev["mark_close"]
                pnl_p          = position * price_ret * entry_notional
                pnl_f          = -position * bar["funding_rate"] * entry_notional
                nav           += pnl_p + pnl_f
                cum_pnl_price += pnl_p
                cum_pnl_fund  += pnl_f
                bars_held     += 1

                # Priority 0 — max trade loss backstop
                if (cum_pnl_price + cum_pnl_fund) < -self.cfg.MAX_TRADE_LOSS_PCT * entry_notional:
                    execute_exit(t, "max_loss")

                # Priority 1 — ATR hard stop
                elif position != 0:
                    adverse = (entry_price - bar["mark_close"]) * position
                    if adverse > self.cfg.ATR_STOP_MULT * atr_at_entry:
                        execute_exit(t, "atr_stop")

                # Priority 2 — z-score reversion
                if position != 0 and abs(bar["funding_zscore"]) < self.cfg.Z_EXIT_BAND:
                    execute_exit(t, "z_revert")

                # Priority 3 — time stop
                if position != 0 and bars_held >= self.cfg.TIME_STOP_BARS:
                    execute_exit(t, "time_stop")

            if position == 0:
                peak_nav = np.max(nav_arr[:t]) if t > 0 else nav
                drawdown = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0.0

                if drawdown <= self.cfg.MAX_PORTFOLIO_DD:
                    sig  = int(bar["confirmed_signal"])
                    size = float(bar["final_size"])

                    if sig != 0 and size > 0:
                        notional   = size * nav
                        entry_cost = notional * cost_per_side
                        nav       -= entry_cost
                        trade_cost = entry_cost

                        position       = sig
                        entry_bar      = t
                        entry_price    = float(bar["mark_close"])
                        entry_notional = notional
                        atr_at_entry   = float(bar["atr_8h"]) if not np.isnan(bar["atr_8h"]) else 0.0
                        bars_held      = 0
                        cum_pnl_price  = 0.0
                        cum_pnl_fund   = 0.0
                        logger.debug(
                            "ENTRY %s %s t=%d price=%.2f notional=%.2f",
                            symbol, "LONG" if sig == 1 else "SHORT",
                            t, entry_price, notional,
                        )

            nav_arr[t] = nav
            ret_arr[t] = (nav - nav_arr[t - 1]) / nav_arr[t - 1] if nav_arr[t - 1] > 0 else 0.0

        if position != 0:
            execute_exit(n - 1, "end_of_data")

        logger.info(
            "%s %s: %d trades, final_nav=%.2f (%.2f%% return)",
            symbol, label, len(trades), nav_arr[-1],
            100 * (nav_arr[-1] / self.initial_capital - 1),
        )

        return BacktestResult(
            symbol         = symbol,
            label          = label,
            nav_series     = nav_arr,
            returns_series = ret_arr,
            trades         = trades,
            df_enriched    = df,
        )

    # ------------------------------------------------------------------
    # Analytics (formerly PerformanceAnalyser)
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        result: BacktestResult,
        btc_returns: Optional[pd.Series] = None,
    ) -> StrategyMetrics:
        """Compute all performance metrics from a BacktestResult."""
        K       = self.cfg.BARS_PER_YEAR
        nav     = result.nav_series
        ret     = result.returns_series
        initial = nav[0] if nav[0] > 0 else self.initial_capital

        ret_clean = ret[~np.isnan(ret)]
        n         = len(ret_clean)
        n_years   = n / K

        total_return = (nav[-1] / initial) - 1
        ann_return   = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        ann_vol      = float(np.std(ret_clean, ddof=1)) * np.sqrt(K) if n > 1 else 0.0
        sharpe       = (float(np.mean(ret_clean)) / float(np.std(ret_clean, ddof=1)) * np.sqrt(K)
                        if np.std(ret_clean, ddof=1) > 0 else 0.0)

        neg_ret     = ret_clean[ret_clean < 0]
        sigma_down  = float(np.std(neg_ret, ddof=1)) * np.sqrt(K) if len(neg_ret) > 1 else 0.0
        sortino     = ann_return / sigma_down if sigma_down > 0 else 0.0

        peak    = np.maximum.accumulate(nav)
        dd      = np.where(peak > 0, (peak - nav) / peak, 0.0)
        max_dd  = float(np.max(dd))
        calmar  = ann_return / max_dd if max_dd > 0 else 0.0

        max_dd_bars, run = 0, 0
        for flag in (dd > 0):
            run = run + 1 if flag else 0
            max_dd_bars = max(max_dd_bars, run)
        max_dd_days = max_dd_bars / self.cfg.BARS_PER_DAY

        beta = 0.0
        if btc_returns is not None and len(btc_returns) >= len(ret_clean):
            btc = btc_returns.values[-len(ret_clean):]
            cov = np.cov(ret_clean, btc, ddof=1)[0, 1]
            var = np.var(btc, ddof=1)
            beta = cov / var if var > 0 else 0.0

        total_price_pnl   = sum(t.pnl_price   for t in result.trades)
        total_funding_pnl = sum(t.pnl_funding  for t in result.trades)
        gross_sum         = total_price_pnl + total_funding_pnl
        funding_fraction  = total_funding_pnl / gross_sum if abs(gross_sum) > 0 else 0.0

        net_pnls  = [t.net_pnl for t in result.trades]
        winners   = [p for p in net_pnls if p > 0]
        losers    = [p for p in net_pnls if p <= 0]
        win_rate  = len(winners) / len(net_pnls) if net_pnls else 0.0
        pf        = sum(winners) / abs(sum(losers)) if losers else float("inf")
        avg_bars  = float(np.mean([t.bars_held for t in result.trades])) if result.trades else 0.0

        return StrategyMetrics(
            total_return         = total_return,
            ann_return           = ann_return,
            ann_volatility       = ann_vol,
            sharpe_ratio         = sharpe,
            sortino_ratio        = sortino,
            calmar_ratio         = calmar,
            max_drawdown         = max_dd,
            max_dd_duration_days = max_dd_days,
            beta_to_btc          = beta,
            funding_fraction     = funding_fraction,
            total_trades         = len(result.trades),
            win_rate             = win_rate,
            profit_factor        = pf,
            avg_bars_held        = avg_bars,
            avg_hours_held       = avg_bars * 8.0,
            initial_nav          = float(initial),
            final_nav            = float(nav[-1]),
        )

    def rolling_sharpe(self, result: BacktestResult, window_months: int = 6) -> pd.Series:
        """Compute rolling N-month Sharpe ratio for signal decay monitoring."""
        K           = self.cfg.BARS_PER_YEAR
        window_bars = window_months * 30 * self.cfg.BARS_PER_DAY
        ret         = pd.Series(result.returns_series, index=result.df_enriched.index)
        roll        = ret.rolling(window=window_bars, min_periods=window_bars // 2)
        return roll.mean() / roll.std(ddof=1) * np.sqrt(K)

    def trades_to_dataframe(self, trades: List[TradeRecord]) -> pd.DataFrame:
        """Convert trade list to a flat DataFrame with all columns."""
        if not trades:
            return pd.DataFrame()
        return pd.DataFrame([{
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
        } for t in trades])

    def print_report(self, m: StrategyMetrics, label: str = "") -> None:
        """Print a formatted performance report to stdout."""
        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  PERFORMANCE REPORT  {label}")
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

    def export(
        self,
        result: BacktestResult,
        metrics: StrategyMetrics,
        results_dir: Path,
        symbol: str,
        label: str,
    ) -> None:
        """
        Write trades CSV and enriched master parquet to results_dir.
        Attaches nav and period_return columns from result before saving.
        """
        results_dir = Path(results_dir)
        results_dir.mkdir(exist_ok=True)

        trades_df = self.trades_to_dataframe(result.trades)
        if not trades_df.empty:
            path = results_dir / f"{symbol}_{label}_trades.csv"
            trades_df.to_csv(path, index=False)
            logger.info("Saved %d trades → %s", len(trades_df), path)

        df_export = result.df_enriched.copy()
        df_export["nav"]           = result.nav_series
        df_export["period_return"] = result.returns_series
        path = results_dir / f"{symbol}_{label}_master.parquet"
        df_export.to_parquet(path)
        logger.info("Saved master parquet → %s", path)
