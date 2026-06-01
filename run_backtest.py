"""
Backtest entry point for Strategy 3 — Funding Rate Mean Reversion.

Usage:
  python run_backtest.py                          # real data, both symbols
  python run_backtest.py --symbol BTCUSDT         # single symbol
  python run_backtest.py --synthetic              # synthetic data, no API call
  python run_backtest.py --use-cache              # load parquet cache if available
  python run_backtest.py --synthetic --symbol BTCUSDT
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_backtest")

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config import CONFIG
from data.fetcher import BinanceFetcher
from data.processor import DataProcessor
from signals.funding_zscore import FundingZScoreSignal
from signals.filters import SignalFilters
from risk.sizing import KellySizer
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceAnalyser


# ------------------------------------------------------------------
# Synthetic data generator (Phase 4 gate — no API needed)
# ------------------------------------------------------------------

def _generate_synthetic(symbol: str, n_bars: int = 4000) -> dict:
    """
    Generate synthetic 8h data for testing without API access.
    Funding rate: AR(1) process with mean reversion.
    Price: geometric random walk.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n_bars, freq="8h", tz="UTC")

    # AR(1) funding rate: mean=0.0001, phi=0.85
    phi   = 0.85
    sigma = 0.0003
    f = np.zeros(n_bars)
    f[0] = 0.0001
    for i in range(1, n_bars):
        f[i] = 0.0001 + phi * (f[i - 1] - 0.0001) + rng.normal(0, sigma)

    # Random walk price
    returns = rng.normal(0.0002, 0.01, n_bars)
    close   = 40000.0 * np.exp(np.cumsum(returns))
    high    = close * (1 + np.abs(rng.normal(0, 0.005, n_bars)))
    low     = close * (1 - np.abs(rng.normal(0, 0.005, n_bars)))
    open_   = close * (1 + rng.normal(0, 0.003, n_bars))

    funding_df = pd.DataFrame(
        {"funding_rate": f, "mark_price": close},
        index=pd.DatetimeIndex(dates, name="funding_time"),
    )
    klines_df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": rng.uniform(100, 1000, n_bars)},
        index=pd.DatetimeIndex(dates, name="open_time"),
    )

    return {
        "funding":   funding_df,
        "klines_8h": klines_df,
        "klines_1h": None,
        "mark_8h":   klines_df.copy(),
        "index_8h":  klines_df.copy(),
        "ls_global": None,
        "ls_top":    None,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Strategy 3 backtest")
    parser.add_argument("--symbol",    default=None,  help="Single symbol (default: all in config)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (no API)")
    parser.add_argument("--use-cache", action="store_true", help="Load parquet cache if available")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else CONFIG.symbols
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    fetcher   = BinanceFetcher()
    processor = DataProcessor()
    zscore    = FundingZScoreSignal()
    filters   = SignalFilters()
    sizer     = KellySizer()

    all_metrics = []

    for symbol in symbols:
        logger.info("=" * 60)
        logger.info("Processing %s", symbol)
        logger.info("=" * 60)

        # 1. Fetch or generate data
        if args.synthetic:
            logger.info("Using synthetic data for %s", symbol)
            raw = _generate_synthetic(symbol)
        else:
            save_dir = str(BASE_DIR / CONFIG.DATA_DIR) if args.use_cache else None
            raw = fetcher.fetch_all(
                symbol    = symbol,
                start_date = CONFIG.IS_START,
                end_date   = CONFIG.OOS_END,
                save_dir   = save_dir,
            )

        # 2. Build master DataFrame
        df_master = processor.build(raw)

        # 3. Compute z-score + rolling stats
        df_master = zscore.compute_rolling_stats(df_master)
        raw_sig   = zscore.raw_signal(df_master)

        # 4. Apply filters
        df_master = filters.apply_all(df_master, raw_sig)

        # 5. Compute position sizes
        df_master = sizer.compute_sizes(df_master)

        # Verify signal frequency (Section 7.4)
        n_short = (df_master["raw_signal"] == -1).sum()
        n_total = len(df_master)
        pct_short = 100 * n_short / n_total
        logger.info(
            "Signal frequency check: SHORT=%.1f%% of bars (expect 2-5%%)",
            pct_short,
        )
        if pct_short < 1.0:
            logger.warning("Z_SHORT_ENTRY may be too tight (SHORT < 1%% of bars)")
        elif pct_short > 10.0:
            logger.warning("Z_SHORT_ENTRY may be too loose or has a bug (SHORT > 10%%)")

        # 6. Split IS / OOS
        is_mask  = (df_master.index >= CONFIG.IS_START)  & (df_master.index <= CONFIG.IS_END)
        oos_mask = (df_master.index >= CONFIG.OOS_START) & (df_master.index <= CONFIG.OOS_END)
        df_is    = df_master[is_mask].copy()
        df_oos   = df_master[oos_mask].copy()

        engine   = BacktestEngine()
        analyser = PerformanceAnalyser()

        for df_period, label in [(df_is, "IS"), (df_oos, "OOS")]:
            if len(df_period) == 0:
                logger.warning("No data for %s %s", symbol, label)
                continue

            result  = engine.run(df_period, symbol, label)
            metrics = analyser.compute(result)
            analyser.print_report(metrics, label=f"{symbol} {label}")

            # P&L identity verification (Phase 4 gate)
            for trade in result.trades:
                expected = trade.pnl_price + trade.pnl_funding - trade.cost_total
                diff = abs(trade.net_pnl - expected)
                if diff > 0.01:
                    logger.error(
                        "P&L identity violation: trade net_pnl=%.4f, expected=%.4f, diff=%.4f",
                        trade.net_pnl, expected, diff,
                    )

            # Export trades CSV
            trades_df = analyser.trades_to_dataframe(result.trades)
            if not trades_df.empty:
                trades_path = results_dir / f"{symbol}_{label}_trades.csv"
                trades_df.to_csv(trades_path, index=False)
                logger.info("Saved %d trades → %s", len(trades_df), trades_path)

            # Attach nav and period_return from backtest result before saving
            df_export = df_period.copy()
            df_export["nav"]           = result.nav_series
            df_export["period_return"] = result.returns_series

            # Export enriched master parquet
            master_path = results_dir / f"{symbol}_{label}_master.parquet"
            df_export.to_parquet(master_path)
            logger.info("Saved master parquet → %s", master_path)

            all_metrics.append({
                "symbol": symbol,
                "label":  label,
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

    # Save combined performance summary
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_path = results_dir / "performance_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved performance summary → %s", summary_path)


if __name__ == "__main__":
    main()
