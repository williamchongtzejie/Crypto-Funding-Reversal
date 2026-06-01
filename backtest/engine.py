"""
Bar-by-bar backtest engine implementing the state machine from Section 7.3.
All seven conservative assumptions from Section 7.2 are honoured.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class TradeRecord:
    """One completed trade — exactly as specified in Section 10.3."""

    # Identity
    symbol:         str
    direction:      int            # +1 = LONG, -1 = SHORT

    # Entry
    entry_bar:      int
    entry_time:     pd.Timestamp
    entry_price:    float
    entry_size:     float          # fraction of NAV
    entry_notional: float          # USDT
    atr_at_entry:   float

    # Exit
    exit_bar:       int
    exit_time:      pd.Timestamp
    exit_price:     float
    exit_reason:    str            # z_revert | atr_stop | time_stop | max_loss | end_of_data

    # P&L
    pnl_price:      float
    pnl_funding:    float
    cost_total:     float

    # Hold
    bars_held:      int
    funding_periods: int

    @property
    def net_pnl(self) -> float:
        return self.pnl_price + self.pnl_funding - self.cost_total

    @property
    def gross_pnl(self) -> float:
        return self.pnl_price + self.pnl_funding

    @property
    def return_pct(self) -> float:
        if self.entry_notional <= 0:
            return 0.0
        return self.net_pnl / self.entry_notional


@dataclass
class BacktestResult:
    """Full output from one BacktestEngine.run() call."""
    symbol:          str
    label:           str           # 'IS' or 'OOS'
    nav_series:      np.ndarray
    returns_series:  np.ndarray
    trades:          List[TradeRecord]
    df_enriched:     pd.DataFrame


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class BacktestEngine:
    """
    Implements the exact pseudocode from Section 7.3.

    Conservative assumptions enforced:
      1. Entry at bar t+1 (signal fires at t, fills at t+1 open/close)
      2. Exit at bar close
      3. ATR stop checked once per bar (at bar close)
      4. Fixed 7bps cost
      5. No funding income on entry bar (funding starts bar t+2)
      6. L/S filter inactive (NaN treated as pass-through)
      7. No partial fills
    """

    def __init__(self, initial_capital: Optional[float] = None, cfg=CONFIG):
        self.cfg = cfg
        self.initial_capital = initial_capital or cfg.INITIAL_CAPITAL

    def run(self, df: pd.DataFrame, symbol: str, label: str) -> BacktestResult:
        """
        Run the backtest on df and return a BacktestResult.

        df must have all enriched columns (confirmed_signal, final_size,
        funding_rate, mark_close, funding_zscore, atr_8h).
        """
        n = len(df)
        nav_arr = np.zeros(n)
        ret_arr = np.zeros(n)
        pos_arr = np.zeros(n, dtype=int)

        nav         = float(self.initial_capital)
        nav_arr[0]  = nav

        # Trade state
        position      = 0
        entry_bar     = None
        entry_price   = None
        entry_notional = None
        atr_at_entry  = None
        bars_held     = 0
        cum_pnl_price = 0.0
        cum_pnl_fund  = 0.0
        trade_cost    = 0.0

        trades: List[TradeRecord] = []

        cost_per_side = self.cfg.COST_PER_SIDE   # 0.0007

        def execute_exit(t: int, reason: str) -> None:
            nonlocal nav, position, entry_bar, entry_price, entry_notional
            nonlocal atr_at_entry, bars_held, cum_pnl_price, cum_pnl_fund, trade_cost

            exit_cost = entry_notional * cost_per_side
            nav -= exit_cost
            trade_cost += exit_cost

            bar = df.iloc[t]
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
                exit_price      = float(bar["mark_close"]),
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

        # ------------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------------
        for t in range(1, n):
            bar  = df.iloc[t]
            prev = df.iloc[t - 1]

            # --- Accrue P&L while in position ---
            if position != 0:
                price_ret = (bar["mark_close"] - prev["mark_close"]) / prev["mark_close"]
                pnl_p = position * price_ret * entry_notional
                pnl_f = -position * bar["funding_rate"] * entry_notional
                nav += pnl_p + pnl_f
                cum_pnl_price += pnl_p
                cum_pnl_fund  += pnl_f
                bars_held += 1

                # Exit 4 — max trade loss backstop (priority 0, highest)
                if (cum_pnl_price + cum_pnl_fund) < -self.cfg.MAX_TRADE_LOSS_PCT * entry_notional:
                    execute_exit(t, "max_loss")

                # Exit 2 — ATR hard stop (priority 1)
                elif position != 0:
                    adverse = (entry_price - bar["mark_close"]) * position
                    if adverse > self.cfg.ATR_STOP_MULT * atr_at_entry:
                        execute_exit(t, "atr_stop")

                # Exit 1 — z-score reversion: thesis fulfilled (priority 2)
                if position != 0 and abs(bar["funding_zscore"]) < self.cfg.Z_EXIT_BAND:
                    execute_exit(t, "z_revert")

                # Exit 3 — time stop: thesis decay (priority 3)
                if position != 0 and bars_held >= self.cfg.TIME_STOP_BARS:
                    execute_exit(t, "time_stop")

            # --- Try to enter a new position ---
            if position == 0:
                # Circuit breaker check
                peak_nav = np.max(nav_arr[:t]) if t > 0 else nav
                drawdown = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0.0

                if drawdown <= self.cfg.MAX_PORTFOLIO_DD:
                    sig  = int(bar["confirmed_signal"])
                    size = float(bar["final_size"])

                    if sig != 0 and size > 0:
                        notional   = size * nav
                        entry_cost = notional * cost_per_side
                        nav -= entry_cost
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
            pos_arr[t] = position
            ret_arr[t] = (nav - nav_arr[t - 1]) / nav_arr[t - 1] if nav_arr[t - 1] > 0 else 0.0

        # Close any open position at end of data
        if position != 0:
            execute_exit(n - 1, "end_of_data")

        logger.info(
            "%s %s: %d trades, final_nav=%.2f (%.1f%% return)",
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
