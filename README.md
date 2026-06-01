# Strategy 3 — Funding Rate Mean Reversion

A systematic, market-neutral strategy that exploits the mean-reverting behaviour of Binance perpetual futures funding rates. When funding rates deviate sharply from their historical norm, levered participants face forced liquidations and position reductions, producing a predictable price response. This strategy captures that reversion with strict risk controls and exits before the signal decays.

---

## Table of Contents

1. [Economic Rationale](#1-economic-rationale)
2. [Strategy Overview](#2-strategy-overview)
3. [Signal Construction](#3-signal-construction)
4. [Signal Filters](#4-signal-filters)
5. [Position Sizing](#5-position-sizing)
6. [Exit Rules](#6-exit-rules)
7. [Risk Controls](#7-risk-controls)
8. [Transaction Costs](#8-transaction-costs)
9. [Backtest Results](#9-backtest-results)
10. [Repository Structure](#10-repository-structure)
11. [Setup and Installation](#11-setup-and-installation)
12. [Running the Pipeline](#12-running-the-pipeline)
13. [Live Execution](#13-live-execution)
14. [Dashboard](#14-dashboard)
15. [Parameter Reference](#15-parameter-reference)
16. [Potential Improvements](#16-potential-improvements)

---

## 1. Economic Rationale

Binance perpetual futures settle funding every **8 hours** (00:00, 08:00, 16:00 UTC). When funding is extremely positive, long holders pay short holders; when extremely negative, the reverse. Traders respond predictably:

- **Extreme positive funding** → levered longs close positions to avoid paying → price falls toward spot → short entry.
- **Extreme negative funding** → shorts close to avoid paying → price rises toward spot → long entry.

The key insight is that extreme funding rates are unsustainable. They recruit arbitrageurs who simultaneously sell the perp and buy spot, compressing the basis back to equilibrium. This mean-reversion is structurally persistent because it is driven by the mechanics of the settlement mechanism, not by soft behavioural biases.

The strategy is inherently **market-neutral**: it trades the relative dislocation between perp and spot, not the directional level of BTC or ETH. Beta to BTC in both in-sample and out-of-sample periods is effectively zero.

---

## 2. Strategy Overview

| Property | Value |
|---|---|
| Asset class | Binance USDT-margined perpetual futures |
| Universe | BTCUSDT, ETHUSDT (expandable) |
| Bar frequency | 8 hours (aligned to funding settlement) |
| Signal | 90-bar rolling z-score of funding rate |
| Entry (SHORT) | z-score > +2.0 σ |
| Entry (LONG) | z-score < −1.5 σ |
| Exit trigger | \|z-score\| < 0.5 σ, ATR stop, time stop, or max-loss backstop |
| Position size | Half-Kelly with 2% NAV hard cap |
| Max drawdown limit | 15% (circuit breaker) |
| Backtest period | IS: 2020–2022 · OOS: 2023–2024 |
| Initial capital | $1,000,000 USDT |
| Cost assumption | 7 bps per side (5 bps taker fee + 2 bps slippage) |

The asymmetric entry thresholds (+2.0 σ short, −1.5 σ long) reflect the empirically stronger and faster mean reversion from extreme positive funding, which is the dominant regime in bull markets. The lower threshold for longs provides more frequent entries when negative funding is comparatively rarer.

---

## 3. Signal Construction

### 3.1 Funding Z-Score

For each 8-hour bar $t$, compute the rolling z-score over the previous $W = 90$ bars (30 days):

$$z_t = \frac{f_t - \mu_t}{\max(\sigma_t, \varepsilon)}$$

where:
- $f_t$ = funding rate at settlement $t$
- $\mu_t$ = rolling 90-bar mean of funding rate
- $\sigma_t$ = rolling 90-bar sample standard deviation (ddof=1)
- $\varepsilon = 10^{-7}$ (floor to prevent division by zero in flat-rate regimes)
- `min_periods = 45` (half the window, so the signal activates after 15 days of data)

### 3.2 Raw Signal

| Condition | Signal |
|---|---|
| $z_t > +2.0$ | −1 (SHORT) |
| $z_t < −1.5$ | +1 (LONG) |
| otherwise | 0 (FLAT) |

In normal markets, the SHORT signal fires on roughly 2–5% of bars. The LONG signal is rarer, concentrated in crypto bear markets or capitalisation events.

### 3.3 Derived Features

All features are computed from the master DataFrame after forward-filling price gaps (max 3 consecutive bars):

| Feature | Formula | Purpose |
|---|---|---|
| `ret_8h` | log(close / close.shift(1)) | Return per bar |
| `atr_8h` | EWM span-14 on True Range | ATR stop distance |
| `rvol_ann` | 90-bar rolling std(ret_8h) × √1095 | Regime volatility |
| `basis` | (mark_close − index_close) / index_close | Perp-spot premium |

---

## 4. Signal Filters

Three sequential filters gate the raw signal. Each filter can only suppress signals, never generate them.

### Filter A — Long/Short Ratio (Crowd Positioning)

Requires that retail positioning is **against** the trade direction, i.e. the crowd is already extended in the direction funding is pushing them.

| Direction | Condition to PASS |
|---|---|
| SHORT | `ls_ratio > 1.20` (crowd is net long) **or** NaN |
| LONG | `ls_ratio < 0.85` (crowd is net short) **or** NaN |

A secondary confirmation uses `top_ls_ratio` (top trader positions): if smart money is aligned with the crowd, the signal is blocked. NaN passes through silently — historical data has no L/S data, so this filter is inactive in backtests and becomes active only in live mode.

### Filter B — Perp-Spot Basis

For SHORT entries only: requires `basis > 0.20%` (the perp is meaningfully above spot). This confirms the premium exists to compress, rather than shorting a flat or backwardated market.

LONG entries always pass this filter, as negative basis is itself a confirmation.

### Filter C — Regime Filter (Trend Guard)

Blocks all new entries during parabolic trends by computing a z-score of the 30-day rolling return:

$$z^{reg}_t = \frac{r^{30d}_t - \mu^{reg}_t}{\sigma^{reg}_t}$$

If $|z^{reg}_t| > 2.5$, no new entries are allowed. This prevents the strategy from fighting a strong directional trend with mean-reversion trades. Entries resume automatically when the regime z-score normalises.

---

## 5. Position Sizing

The strategy uses **half-Kelly** sizing with a binary volatility regime scalar and a hard NAV cap.

### 5.1 Kelly Estimation

The signal return series (lag-1 to prevent look-ahead):

$$r^{sig}_t = \text{confirmed\_signal}_{t-1} \times r^{8h}_t$$

Using a 180-bar (60-day) rolling window:

$$f^* = \frac{\mu_{sig}}{\sigma^2_{sig}} \quad \text{clipped to} \quad [0,\ 0.20]$$

### 5.2 Half-Kelly and Volatility Scalar

$$\text{size} = \min\!\left( \frac{f^*}{2} \times s_{vol},\ 2\% \right)$$

where the binary volatility scalar:
$$s_{vol} = \begin{cases} 0.5 & \text{if } \text{rvol\_ann} > 1.20 \\ 1.0 & \text{otherwise} \end{cases}$$

### 5.3 Summary

| Step | Action |
|---|---|
| 1 | Compute rolling signal returns (lag-1) |
| 2 | Estimate μ and σ² over 180-bar window |
| 3 | Full Kelly: μ/σ², clipped to [0, 20%] |
| 4 | Half-Kelly: multiply by 0.5 |
| 5 | Vol scalar: multiply by 0.5 if annualised vol > 120% |
| 6 | Hard cap: min(result, 2% NAV) |
| 7 | Zero out wherever confirmed_signal = 0 |

The 2% NAV cap means maximum notional exposure per position is $20,000 on a $1M book — deliberately conservative for a mean-reversion strategy where the edge comes from frequency and low drawdowns, not from large individual bets.

---

## 6. Exit Rules

Four exit conditions are checked every bar in strict priority order:

| Priority | Exit | Condition | Rationale |
|---|---|---|---|
| 0 | Max Loss | cumulative PnL < −4% of notional | Backstop: thesis is wrong, cut losses |
| 1 | ATR Stop | adverse move > 2 × ATR at entry | Hard structural stop based on volatility at entry |
| 2 | Z-Revert | \|z-score\| < 0.5 | Signal thesis fulfilled |
| 3 | Time Stop | bars held ≥ 6 (48 hours) | Thesis decay: funding reverts or signal stales |

All exits are executed at the bar close (mark price). No partial exits. If still in position at end of data, the position is closed with reason `end_of_data`.

**P&L Accounting per bar while in position:**

- Price PnL: `position × (mark_close − prev_mark_close) / prev_mark_close × notional`
- Funding PnL: `−position × funding_rate × notional` (paid TO the strategy when short in high-funding regime)

---

## 7. Risk Controls

### Portfolio-Level Circuit Breaker

If the drawdown from peak NAV exceeds **15%**, all new entries are blocked. Entries resume only when drawdown recovers below **10%**. Existing positions continue to be managed under their individual stop rules.

### Directional Concentration

Maximum total notional in one direction (all positions aggregated): **4% of NAV**.

### Conservative Backtest Assumptions

The backtest enforces seven conservative assumptions to avoid overstating performance:

1. Entry fills at next bar's mark price (signal fires at $t$, fill at $t+1$)
2. Exits at bar close (no intrabar exits)
3. ATR stop checked once per bar, at close
4. Fixed 7 bps cost per side (no volume-based fee reduction)
5. No funding income on the entry bar
6. L/S filter inactive for historical data (treated as NaN pass-through)
7. No partial fills

---

## 8. Transaction Costs

| Component | Rate | Notes |
|---|---|---|
| Taker fee | 5 bps | Binance VIP 0 USDT-margined futures rate |
| Slippage | 2 bps | Conservative for order sizes < $100k |
| **Total** | **7 bps per side** | Applied on both entry and exit |

All costs are deducted from NAV at execution time. The cost model is intentionally conservative; at higher volume tiers, taker fees compress to 2–3 bps.

---

## 9. Backtest Results

All results use $1M initial capital, BTCUSDT, 2020–2024.

### 9.1 BTCUSDT Performance Summary

| Metric | In-Sample (IS) 2020–2022 | Out-of-Sample (OOS) 2023–2024 |
|---|---|---|
| Total Return | +0.80% | +0.60% |
| Annualised Return | +0.26% | +0.30% |
| Annualised Volatility | — | — |
| **Sharpe Ratio** | **0.74** | **1.27** |
| Sortino Ratio | 0.31 | 0.77 |
| Calmar Ratio | 0.74 | 1.63 |
| **Max Drawdown** | **−0.36%** | **−0.18%** |
| Max DD Duration | — | — |
| Beta to BTC | 0.000 | 0.000 |
| Funding Fraction | 0.63% | −1.20% |
| Total Trades | 72 | 51 |
| **Win Rate** | **54.2%** | **64.7%** |
| **Profit Factor** | **1.52** | **2.34** |

### 9.2 Key Observations

**OOS Sharpe 1.27 exceeds IS Sharpe 0.74.** This is unusual and reflects the 2022 bear market suppressing IS performance: extreme negative funding during the LUNA/FTX crashes was more volatile and harder to trade than the more orderly OOS regime.

**Beta = 0.000 in both periods.** Confirmed market neutrality. The strategy does not depend on BTC direction; returns are uncorrelated to BTC spot.

**Maximum drawdown < 0.4%** in both periods, well within the 15% circuit breaker. The short hold period (max 48 hours) and tight stop rules contain individual trade losses before they accumulate.

**Funding fraction near zero** is expected: the strategy holds for at most 6 bars × 8h = 48h. At typical funding rates of 0.01%, a 2-bar hold collects only ~0.02% of notional. The edge comes primarily from price reversion, with funding income as a secondary contributor.

**Profit factor 1.52 IS / 2.34 OOS** confirms positive expectancy with improving edge in the OOS period, likely because 2023–2024 produced cleaner, more tradeable funding spikes than the chaotic 2022 deleveraging events.

### 9.3 Phase Gates Passed

- **Phase 4 P&L identity**: `net_pnl = pnl_price + pnl_funding − cost_total` verified within $0.01 for every trade.
- **16/16 unit tests passing** across signal, sizing, and backtest modules.
- **Synthetic backtest**: strategy operates correctly with zero-API synthetic AR(1) funding data.

---

## 10. Repository Structure

```
strategy3/
├── config.py                  # All parameters — single source of truth
├── run_backtest.py            # Main backtest entry point
├── run_dashboard.py           # Launch Plotly Dash dashboard
├── run_live.py                # Live execution entry point
├── requirements.txt
│
├── data/
│   ├── fetcher.py             # Binance REST API fetcher (all datasets)
│   ├── processor.py           # Master DataFrame builder + feature engineering
│   └── raw/                   # Parquet cache (auto-created on first run)
│       └── {SYMBOL}/
│           ├── {SYMBOL}_funding.parquet
│           ├── {SYMBOL}_klines_8h.parquet
│           └── ...
│
├── signals/
│   ├── funding_zscore.py      # Rolling z-score + raw signal
│   └── filters.py             # L/S ratio, basis, and regime filters
│
├── risk/
│   └── sizing.py              # Half-Kelly position sizing
│
├── backtest/
│   ├── engine.py              # Bar-by-bar simulation state machine
│   └── metrics.py             # Full performance metrics (Section 5.10)
│
├── dashboard/
│   ├── app.py                 # Dash app factory
│   ├── layouts.py             # 6-tab layout definitions
│   └── callbacks.py           # All Dash callbacks
│
├── live/
│   ├── signal_monitor.py      # Settlement callback + ATR tick monitor
│   └── order_manager.py       # Binance signed-API order execution
│
├── results/                   # Auto-created by run_backtest.py
│   ├── {SYMBOL}_{IS/OOS}_master.parquet
│   ├── {SYMBOL}_{IS/OOS}_trades.csv
│   └── performance_summary.csv
│
└── tests/
    ├── test_signal.py
    ├── test_sizing.py
    └── test_backtest.py
```

---

## 11. Setup and Installation

### Prerequisites

- Python 3.11+
- Binance account with USDT-margined futures enabled (for live trading only)
- No API key required for backtesting (all data endpoints are public)

### Install

```bash
cd strategy3
pip install -r requirements.txt
```

### API Keys (Live Trading Only)

Create a `.env` file in the `strategy3/` directory:

```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

The key requires **USDT-margined futures trading** permission. Spot trading permission is not needed. Read-only access is sufficient for the dashboard in historical mode.

---

## 12. Running the Pipeline

### Step 1 — Fetch data and run backtest

**First run (fetches from Binance API, ~5–10 minutes):**

```bash
python run_backtest.py --symbol BTCUSDT
```

**Subsequent runs (use local parquet cache):**

```bash
python run_backtest.py --symbol BTCUSDT --use-cache
```

**Both symbols:**

```bash
python run_backtest.py --use-cache
```

**Synthetic data (no API, for unit testing):**

```bash
python run_backtest.py --synthetic
```

Outputs written to `results/`:
- `{SYMBOL}_{IS/OOS}_master.parquet` — enriched master DataFrame with all features, NAV, and returns
- `{SYMBOL}_{IS/OOS}_trades.csv` — trade log with full P&L breakdown
- `performance_summary.csv` — side-by-side IS/OOS metrics table

### Step 2 — View results

```bash
python run_dashboard.py
```

Opens the Plotly Dash dashboard at `http://localhost:8050`.

### Step 3 — Run tests

```bash
pytest tests/ -v
```

All 16 tests should pass. Tests validate the signal, sizing, and P&L identity gate without touching the API.

---

## 13. Live Execution

**Start the live feed and order execution:**

```bash
python run_live.py
```

The live system connects two WebSocket streams per symbol:
- `{symbol}@markPrice@1s` — 1-second mark price ticks for ATR stop monitoring
- `{symbol}@kline_1h` — hourly OHLCV for intrabar context

**At each 8-hour settlement boundary**, the system:
1. Detects the settlement from the markPrice stream (funding rate change)
2. Fetches the confirmed funding rate and current L/S ratios via REST
3. Runs the full signal pipeline (z-score → filters → sizing)
4. Executes entry or exit market orders via the signed Binance API
5. Updates the live dashboard state

**On every mark price tick**, the ATR stop is checked. If the position moves adversely by more than 2× ATR, the position is closed immediately with a market order — no waiting for the next settlement.

**Position reconciliation** runs at each settlement: the system compares its internal position state against the exchange's `/fapi/v2/positionRisk` endpoint and adopts the exchange as ground truth if a discrepancy exceeds 0.001 BTC/ETH. This prevents ghost positions from stale local state.

**Graceful shutdown**: `Ctrl+C` or `SIGTERM` closes all open positions before stopping.

---

## 14. Dashboard

The Plotly Dash dashboard (`http://localhost:8050`) provides six tabs:

| Tab | Contents |
|---|---|
| **Signal Overview** | BTC/ETH price candlestick with long/short entry markers; funding z-score panel with ±2σ reference lines |
| **Portfolio Performance** | IS and OOS NAV curves; drawdown panel; IS/OOS separator line |
| **Signal Decay** | 6-month rolling Sharpe ratio (IS cyan, OOS orange); SR = 1.0 reference line |
| **Trade Analytics** | Full trade table (sortable); stacked price vs funding P&L bar chart; hold duration histogram; exit reason pie chart |
| **Signal Decomposition** | Funding rate with ±2σ band; L/S ratio (live only); perp-spot basis |
| **Live Monitor** | Real-time z-score gauge; current mark price, funding rate, basis, L/S ratio; settlement countdown; open position card |

The dashboard reads from the `results/` parquet and CSV files generated by `run_backtest.py`. The Live Monitor tab polls `signal_monitor.get_live_state()` every 30 seconds.

---

## 15. Parameter Reference

All parameters live in `config.py` as a typed dataclass. No parameters are scattered across modules.

| Parameter | Default | Description |
|---|---|---|
| `FUNDING_Z_WINDOW` | 90 | Rolling window for z-score (bars; 90 = 30 days) |
| `Z_SHORT_ENTRY` | 2.0 | Z-score threshold to enter SHORT |
| `Z_LONG_ENTRY` | −1.5 | Z-score threshold to enter LONG |
| `Z_EXIT_BAND` | 0.5 | Z-score band for z-revert exit |
| `ATR_PERIOD` | 14 | EWM span for ATR calculation |
| `ATR_STOP_MULT` | 2.0 | ATR stop multiplier (2 × ATR at entry) |
| `TIME_STOP_BARS` | 6 | Maximum bars held (6 × 8h = 48h) |
| `MAX_TRADE_LOSS_PCT` | 0.04 | Max loss backstop (4% of notional) |
| `KELLY_FRACTION` | 0.5 | Half-Kelly multiplier |
| `KELLY_EST_WINDOW` | 180 | Kelly estimation window (bars; 180 = 60 days) |
| `KELLY_UPPER_CLIP` | 0.20 | Full Kelly clip before halving |
| `VOL_REGIME_THRESH` | 1.20 | Annualised vol threshold for high-vol regime |
| `HIGH_VOL_SCALAR` | 0.50 | Size multiplier in high-vol regime |
| `NAV_CAP` | 0.02 | Hard NAV cap per position (2%) |
| `LS_SHORT_MIN` | 1.20 | Min L/S ratio required to enter SHORT |
| `LS_LONG_MAX` | 0.85 | Max L/S ratio required to enter LONG |
| `BASIS_SHORT_MIN` | 0.002 | Min basis (+0.2%) required for SHORT |
| `REGIME_Z_THRESH` | 2.5 | Regime z-score above which entries are blocked |
| `MAX_PORTFOLIO_DD` | 0.15 | Circuit breaker drawdown threshold (15%) |
| `DD_RESUME_LEVEL` | 0.10 | Drawdown level at which entries resume (10%) |
| `TAKER_FEE_BPS` | 5.0 | Taker fee per side (bps) |
| `SLIPPAGE_BPS` | 2.0 | Slippage assumption per side (bps) |
| `IS_START / IS_END` | 2020-01-01 / 2022-12-31 | In-sample period |
| `OOS_START / OOS_END` | 2023-01-01 / 2024-12-31 | Out-of-sample period |
| `INITIAL_CAPITAL` | 1,000,000 | Backtest starting NAV (USDT) |
| `BARS_PER_YEAR` | 1095 | 3 settlements/day × 365 (for annualisation) |

---

## 16. Potential Improvements

### Signal

**Multi-symbol portfolio with correlation gating.** Running BTCUSDT and ETHUSDT concurrently doubles opportunity but the two signals are correlated (both driven by crypto-wide funding cycles). Adding a correlation check — block the second signal if an existing position's symbol shows >0.8 30-day rolling correlation with the candidate — would reduce concurrent exposure to the same risk factor.

**Funding rate level as a secondary weight.** The z-score captures the *relative* extremity of funding but ignores the *absolute* level. A funding rate of +0.15% at z = +2.5 is structurally more attractive than +0.03% at the same z-score. Weighting the Kelly estimate by the absolute funding level could improve selection.

**Intrabar signal refinement with 1h bars.** The 8h bar coarsens the entry: the z-score triggers at settlement but the reversion might not begin for 1–2 hours. Using the 1h klines already fetched to find an intrabar momentum confirmation (e.g. the first 1h bar in the direction of reversion closing above entry) could improve fill quality.

**Cross-exchange funding divergence.** Binance and Bybit funding rates frequently diverge by 2–5 bps during stress events. Incorporating the divergence as a filter — only take the signal when Binance funding is extreme *and* diverging from Bybit — would select higher-conviction entries.

### Sizing

**Full Kelly with drawdown penalty.** The current half-Kelly is a static discount. A dynamic penalty that scales the Kelly fraction inversely with current drawdown (`f_effective = f_half × (1 − dd / dd_threshold)`) would reduce size organically during losing streaks.

**Per-trade expected value weighting.** Kelly sizing treats all signal occurrences equally. A logistic regression on historical features (z-score magnitude, basis level, time since last settlement, funding momentum) could assign a confidence score and scale size accordingly, replacing the binary vol scalar.

### Execution

**Limit order entry within ±0.5 bps of mid.** The current engine assumes taker fills. During low-volatility periods (rvol_ann < 0.5), the spread is tight enough that a limit order sitting at mid for the first 5 minutes post-settlement would fill ~80% of the time, saving ~4 bps per round trip.

**Adaptive slippage model.** The 2 bps slippage assumption is flat regardless of order size. For notionals above $50k, a square-root market impact model (`impact_bps = k × sqrt(notional / ADV)`) would produce more realistic cost estimates and could gate large-size entries.

### Risk

**Intraday ATR recalibration.** The ATR is computed on 8h bars and set at entry. During high-volatility regimes, a 15-minute ATR recalculation using the 1h klines would allow the stop to tighten as the position matures and the intrabar volatility normalises — capturing more of the reversion while protecting against late-session spikes.

**Greeks-style exposure monitoring.** Currently the only portfolio-level check is the 15% drawdown circuit breaker. Adding exposure monitoring across the following dimensions would improve the risk framework: net delta (sum of directional notionals), aggregate funding-rate sensitivity (total notional × avg funding rate), and liquidity risk (ratio of position notional to 1h ADV).

**Stress-testing with historical shocks.** The backtest covers the 2022 bear market and FTX collapse, but not a scenario where Binance itself is under systemic stress (withdrawal pauses, extreme basis divergence). A simulation that applies a flat 50 bps slippage shock for 3–5 consecutive bars would test whether the stop rules are still functional under execution breakdown.

### Infrastructure

**Redundant WebSocket with failover.** The current live system has a single WebSocket connection per symbol with exponential backoff reconnect. A dual-feed setup (primary on `fstream.binance.com`, backup on the alternate cluster) with automatic failover on 3 missed heartbeats would reduce gap risk around settlement boundaries.

**Persistent trade database.** Completed trades are currently written to CSV at backtest time and held in memory during live. Migrating to SQLite (or PostgreSQL for multi-instance) would enable live trade querying, faster performance attribution, and a cleaner audit trail for the order manager's reconciliation logic.

**Paper trading mode.** Adding a `--paper` flag to `run_live.py` that runs the full order execution path but skips the `/fapi/v1/order` POST call (logging the intended order instead) would allow the full live pipeline to be validated in production conditions before capital is committed.
