"""
Top-level orchestrator: wires all pipeline components together.
Entry points (run_backtest.py, run_live.py) delegate to this class.
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import CONFIG
from data.pipeline import DataPipeline
from signals.pipeline import SignalPipeline
from risk.manager import RiskManager
from backtest.runner import Backtester
from models import StrategyMetrics

logger = logging.getLogger(__name__)


class FundingReversalStrategy:
    """
    Orchestrates the full pipeline for the Crypto Funding Reversal strategy.

    Components:
        self.data     — DataPipeline   (fetch + build master DataFrame)
        self.signal   — SignalPipeline (z-score + filters → confirmed_signal)
        self.risk     — RiskManager    (half-Kelly sizing + circuit breaker)
        self.backtest — Backtester     (simulation + metrics + export)

    Public interface:
        run_backtest(symbol, use_cache, synthetic)  -> dict[str, StrategyMetrics]
        create_live_trader()                        -> LiveTrader
    """

    def __init__(self, cfg=CONFIG):
        self.cfg      = cfg
        self.data     = DataPipeline(cfg)
        self.signal   = SignalPipeline(cfg)
        self.risk     = RiskManager(cfg)
        self.backtest = Backtester(cfg=cfg)

    def run_backtest(
        self,
        symbol:     Optional[str]  = None,
        use_cache:  bool           = False,
        synthetic:  bool           = False,
    ) -> dict:
        """
        Run the full backtest pipeline for one or all configured symbols.

        Args:
            symbol:    Single symbol to run (e.g. 'BTCUSDT'). None = all in config.
            use_cache: Load / save Binance parquet cache in data/raw/.
            synthetic: Use AR(1) synthetic data instead of the Binance API.

        Returns:
            dict keyed by (symbol, label) -> StrategyMetrics
            e.g. {('BTCUSDT', 'IS'): <StrategyMetrics>, ('BTCUSDT', 'OOS'): ...}
        """
        symbols     = [symbol] if symbol else self.cfg.symbols
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        all_metrics: dict = {}
        summary_rows      = []

        for sym in symbols:
            logger.info("=" * 60)
            logger.info("Processing %s", sym)
            logger.info("=" * 60)

            # 1. Fetch or generate data
            if synthetic:
                logger.info("Using synthetic data for %s", sym)
                raw = DataPipeline.generate_synthetic(sym)
            else:
                raw = self.data.fetch(sym, self.cfg.IS_START, self.cfg.OOS_END, use_cache=use_cache)

            # 2. Build master DataFrame + full signal stack + sizing
            df_master = self.data.build(raw)
            df_master = self.signal.run(df_master)
            df_master = self.risk.compute_sizes(df_master)

            # Signal frequency diagnostic
            n_short  = (df_master["raw_signal"] == -1).sum()
            pct_short = 100 * n_short / len(df_master)
            logger.info("Signal frequency: SHORT=%.1f%% (expect 2–5%%)", pct_short)
            if pct_short < 1.0:
                logger.warning("Z_SHORT_ENTRY may be too tight (SHORT < 1%% of bars)")
            elif pct_short > 10.0:
                logger.warning("Z_SHORT_ENTRY may be too loose (SHORT > 10%% of bars)")

            # 3. IS / OOS split
            is_mask  = (df_master.index >= self.cfg.IS_START)  & (df_master.index <= self.cfg.IS_END)
            oos_mask = (df_master.index >= self.cfg.OOS_START) & (df_master.index <= self.cfg.OOS_END)

            for df_period, label in [(df_master[is_mask].copy(), "IS"),
                                     (df_master[oos_mask].copy(), "OOS")]:
                if len(df_period) == 0:
                    logger.warning("No data for %s %s", sym, label)
                    continue

                # 4. Simulate + compute metrics
                result  = self.backtest.run(df_period, sym, label)
                metrics = self.backtest.compute_metrics(result)
                self.backtest.print_report(metrics, label=f"{sym} {label}")

                # 5. P&L identity gate
                for trade in result.trades:
                    expected = trade.pnl_price + trade.pnl_funding - trade.cost_total
                    diff     = abs(trade.net_pnl - expected)
                    if diff > 0.01:
                        logger.error(
                            "P&L identity violation: net=%.4f expected=%.4f diff=%.4f",
                            trade.net_pnl, expected, diff,
                        )

                # 6. Export
                self.backtest.export(result, metrics, results_dir, sym, label)
                all_metrics[(sym, label)] = metrics

                summary_rows.append({
                    "symbol":          sym,
                    "label":           label,
                    "total_return":    metrics.total_return,
                    "ann_return":      metrics.ann_return,
                    "sharpe_ratio":    metrics.sharpe_ratio,
                    "sortino_ratio":   metrics.sortino_ratio,
                    "calmar_ratio":    metrics.calmar_ratio,
                    "max_drawdown":    metrics.max_drawdown,
                    "beta_to_btc":     metrics.beta_to_btc,
                    "funding_fraction": metrics.funding_fraction,
                    "total_trades":    metrics.total_trades,
                    "win_rate":        metrics.win_rate,
                    "profit_factor":   metrics.profit_factor,
                })

        if summary_rows:
            summary_path = results_dir / "performance_summary.csv"
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            logger.info("Saved performance summary → %s", summary_path)

        return all_metrics

    def create_live_trader(self):
        """
        Construct a LiveTrader wired to this strategy's signal and risk components.
        Returns a LiveTrader ready to be passed to BinanceLiveFeed as callbacks.
        """
        from live.trader import LiveTrader
        return LiveTrader(cfg=self.cfg)
