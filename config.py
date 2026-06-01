"""
Single source of truth for all Strategy 3 parameters.
All values are set here before any backtest is run. No post-hoc tuning.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class StrategyConfig:

    # Universe
    symbols:              List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    primary_symbol:       str       = "BTCUSDT"

    # Data
    BINANCE_REST_BASE:    str       = "https://fapi.binance.com"
    BINANCE_WS_BASE:      str       = "wss://fstream.binance.com"
    DATA_DIR:             str       = "data/raw"

    # Z-Score Signal
    FUNDING_Z_WINDOW:     int       = 90       # W in Section 5.2 (30 days x 3 settlements)
    Z_SHORT_ENTRY:        float     = 2.0      # SHORT if z_t > this
    Z_LONG_ENTRY:         float     = -1.5     # LONG if z_t < this
    Z_EXIT_BAND:          float     = 0.5      # Exit if abs(z_t) < this
    Z_SCORE_EPSILON:      float     = 1e-7     # Floor for sigma to prevent division by zero

    # Long/Short Ratio Filter
    LS_SHORT_MIN:         float     = 1.2      # L/S ratio required to pass SHORT signal
    LS_LONG_MAX:          float     = 0.85     # L/S ratio required to pass LONG signal

    # Basis Filter
    BASIS_SHORT_MIN:      float     = 0.002    # Minimum perp-spot basis for SHORT (+0.2%)

    # Regime Filter
    REGIME_WINDOW:        int       = 90       # W_reg: 30-day log return window
    REGIME_HIST_WINDOW:   int       = 180      # W_hist: 60-day rolling stats window
    REGIME_Z_THRESH:      float     = 2.5      # Block new entries above this regime z-score

    # Exit Parameters
    ATR_PERIOD:           int       = 14       # EWM span for ATR
    ATR_STOP_MULT:        float     = 2.0      # Stop at N x ATR_at_entry
    TIME_STOP_BARS:       int       = 6        # Maximum bars held (6 x 8h = 48h)
    MAX_TRADE_LOSS_PCT:   float     = 0.04     # Backstop at 4% of notional

    # Position Sizing (Half-Kelly)
    KELLY_FRACTION:       float     = 0.5      # Half-Kelly multiplier
    KELLY_EST_WINDOW:     int       = 180      # W_kelly: 60-day estimation window
    KELLY_UPPER_CLIP:     float     = 0.20     # Max full-Kelly fraction before halving
    VOL_REGIME_THRESH:    float     = 1.20     # rvol_ann threshold for high-vol regime
    HIGH_VOL_SCALAR:      float     = 0.50     # Size multiplier in high-vol regime
    NAV_CAP:              float     = 0.02     # 2% NAV hard cap per position

    # Transaction Costs
    TAKER_FEE_BPS:        float     = 5.0      # Binance VIP 0 USDT-margined taker fee
    SLIPPAGE_BPS:         float     = 2.0      # Conservative slippage (valid <$100k notional)

    @property
    def COST_PER_SIDE(self) -> float:
        return (self.TAKER_FEE_BPS + self.SLIPPAGE_BPS) / 10_000

    # Portfolio Risk
    MAX_PORTFOLIO_DD:     float     = 0.15     # Circuit breaker threshold (15%)
    DD_RESUME_LEVEL:      float     = 0.10     # Resume entries below this (10%)
    MAX_DIRECTIONAL_NAV:  float     = 0.04     # Max total notional in one direction (4%)

    # Backtest
    INITIAL_CAPITAL:      float     = 1_000_000.0
    IS_START:             str       = "2020-01-01"
    IS_END:               str       = "2022-12-31"
    OOS_START:            str       = "2023-01-01"
    OOS_END:              str       = "2024-12-31"

    # Derived Constants
    BARS_PER_DAY:         int       = 3        # 24h / 8h
    BARS_PER_YEAR:        int       = 1095     # 3 x 365


CONFIG = StrategyConfig()
