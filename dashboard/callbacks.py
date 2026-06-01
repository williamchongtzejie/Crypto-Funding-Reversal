"""
Dash callback functions for all Strategy 3 dashboard tabs.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Input, Output, callback, html
from plotly.subplots import make_subplots

from config import CONFIG

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Z-score reference lines
Z_REFS = [
    dict(y=CONFIG.Z_SHORT_ENTRY, color="red",    name=f"Short Entry (+{CONFIG.Z_SHORT_ENTRY})"),
    dict(y=CONFIG.Z_LONG_ENTRY,  color="green",  name=f"Long Entry ({CONFIG.Z_LONG_ENTRY})"),
    dict(y= CONFIG.Z_EXIT_BAND,  color="gray",   name=f"Exit Band (+{CONFIG.Z_EXIT_BAND})"),
    dict(y=-CONFIG.Z_EXIT_BAND,  color="gray",   name=f"Exit Band (-{CONFIG.Z_EXIT_BAND})"),
]


def _load_master(symbol: str, label: str) -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / f"{symbol}_{label}_master.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _load_trades(symbol: str, label: str) -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / f"{symbol}_{label}_trades.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    return df


def register(app):
    """Register all callbacks with the Dash app."""

    # ------------------------------------------------------------------
    # Tab 1: Signal Overview — price chart + z-score overlay
    # ------------------------------------------------------------------

    @app.callback(
        Output("signal-price-chart", "figure"),
        [
            Input("signal-symbol",       "value"),
            Input("signal-period",       "value"),
            Input("signal-type-toggle",  "value"),
        ],
    )
    def update_signal_chart(symbol, period, sig_type):
        df = _load_master(symbol, period)
        if df is None:
            return go.Figure()

        trades = _load_trades(symbol, period)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35],
            subplot_titles=["Mark Price + Trade Entries", "Funding Z-Score"],
            vertical_spacing=0.08,
        )

        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Price",
        ), row=1, col=1)

        if trades is not None:
            long_entries  = trades[trades["direction"] ==  1]
            short_entries = trades[trades["direction"] == -1]
            fig.add_trace(go.Scatter(
                x=long_entries["entry_time"], y=long_entries["entry_price"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="lime"),
                name="Long Entry",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=short_entries["entry_time"], y=short_entries["entry_price"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="red"),
                name="Short Entry",
            ), row=1, col=1)

        sig_col = "confirmed_signal" if sig_type == "confirmed" else "raw_signal"
        if sig_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["funding_zscore"],
                mode="lines", name="Z-Score", line=dict(color="cyan", width=1),
            ), row=2, col=1)

            for ref in Z_REFS:
                fig.add_hline(y=ref["y"], line_color=ref["color"],
                              line_dash="dash", row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=500,
            legend=dict(orientation="h", y=1.02),
        )
        return fig

    # ------------------------------------------------------------------
    # Tab 2: Portfolio Performance — NAV curve + drawdown
    # ------------------------------------------------------------------

    @app.callback(
        [Output("nav-curve-chart", "figure"), Output("drawdown-chart", "figure")],
        Input("signal-symbol", "value"),
    )
    def update_performance_charts(symbol):
        nav_fig = go.Figure()
        dd_fig  = go.Figure()

        for label, color in [("IS", "cyan"), ("OOS", "orange")]:
            df = _load_master(symbol, label)
            if df is None or "nav" not in df.columns:
                continue

            nav = df["nav"]
            nav_fig.add_trace(go.Scatter(
                x=nav.index, y=nav.values,
                mode="lines", name=f"Strategy {label}", line=dict(color=color),
            ))

            # Drawdown
            peak = nav.cummax()
            dd   = (nav - peak) / peak
            dd_fig.add_trace(go.Scatter(
                x=dd.index, y=dd.values * 100,
                fill="tozeroy", name=f"Drawdown {label}",
                line=dict(color=color), opacity=0.7,
            ))

        # add_vline with a date string requires at least one trace in Plotly 6+;
        # use add_shape instead to avoid the annotation mean-calculation bug.
        if nav_fig.data:
            nav_fig.add_shape(
                type="line",
                x0="2023-01-01", x1="2023-01-01", y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="white", dash="dot", width=1),
            )
            nav_fig.add_annotation(
                x="2023-01-01", y=1, xref="x", yref="paper",
                text="IS/OOS", showarrow=False,
                font=dict(color="white", size=11), yshift=8,
            )

        nav_fig.update_layout(template="plotly_dark", height=400,
                               yaxis_title="NAV (USDT)")
        dd_fig.update_layout(template="plotly_dark", height=250,
                              yaxis_title="Drawdown (%)")
        return nav_fig, dd_fig

    # ------------------------------------------------------------------
    # Tab 3: Signal Decay — rolling 6-month Sharpe
    # ------------------------------------------------------------------

    @app.callback(
        Output("rolling-sharpe-chart", "figure"),
        Input("signal-symbol", "value"),
    )
    def update_rolling_sharpe(symbol):
        fig = go.Figure()
        K = CONFIG.BARS_PER_YEAR
        window = 6 * 30 * CONFIG.BARS_PER_DAY

        for label, color in [("IS", "cyan"), ("OOS", "orange")]:
            df = _load_master(symbol, label)
            if df is None or "period_return" not in df.columns:
                continue
            ret = df["period_return"].fillna(0)
            roll = ret.rolling(window=window, min_periods=window // 2)
            rs = roll.mean() / roll.std(ddof=1) * np.sqrt(K)
            fig.add_trace(go.Scatter(
                x=rs.index, y=rs.values,
                mode="lines", name=f"Rolling SR {label}", line=dict(color=color),
            ))

        if fig.data:
            fig.add_hline(y=1.0, line_color="white", line_dash="dash",
                          annotation_text="SR = 1.0")
        fig.update_layout(template="plotly_dark", height=400,
                          yaxis_title="6-Month Rolling Sharpe")
        return fig

    # ------------------------------------------------------------------
    # Tab 4: Trade Analytics
    # ------------------------------------------------------------------

    @app.callback(
        [
            Output("trade-table",      "data"),
            Output("pnl-bar-chart",    "figure"),
            Output("hold-duration-hist", "figure"),
            Output("exit-reason-pie",  "figure"),
        ],
        [Input("signal-symbol", "value"), Input("signal-period", "value")],
    )
    def update_trade_analytics(symbol, period):
        trades = _load_trades(symbol, period)
        if trades is None or trades.empty:
            empty = go.Figure()
            return [], empty, empty, empty

        table_data = trades.to_dict("records")

        # Stacked bar: price PnL vs funding PnL
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=list(range(len(trades))),
            y=trades["pnl_price"],
            name="Price PnL", marker_color="cyan",
        ))
        bar_fig.add_trace(go.Bar(
            x=list(range(len(trades))),
            y=trades["pnl_funding"],
            name="Funding PnL", marker_color="orange",
        ))
        bar_fig.update_layout(
            template="plotly_dark", barmode="stack",
            title="Price vs Funding PnL per Trade", height=300,
        )

        # Hold duration histogram
        hist_fig = px.histogram(
            trades, x="hours_held",
            title="Trade Duration (hours)",
            template="plotly_dark", height=300,
        )

        # Exit reason pie
        exit_counts = trades["exit_reason"].value_counts()
        pie_fig = go.Figure(go.Pie(
            labels=exit_counts.index, values=exit_counts.values,
            hole=0.3,
        ))
        pie_fig.update_layout(
            template="plotly_dark", title="Exit Reasons", height=300,
        )

        return table_data, bar_fig, hist_fig, pie_fig

    # ------------------------------------------------------------------
    # Tab 5: Signal Decomposition
    # ------------------------------------------------------------------

    @app.callback(
        [
            Output("funding-rate-chart", "figure"),
            Output("ls-ratio-chart",     "figure"),
            Output("basis-chart",        "figure"),
        ],
        [Input("signal-symbol", "value"), Input("signal-period", "value")],
    )
    def update_decomposition(symbol, period):
        df = _load_master(symbol, period)

        def empty():
            return go.Figure()

        if df is None:
            return empty(), empty(), empty()

        # Funding rate with rolling mean ± 2σ
        fr_fig = go.Figure()
        if "funding_rate" in df.columns:
            fr_fig.add_trace(go.Scatter(
                x=df.index, y=df["funding_rate"] * 100,
                mode="lines", name="Funding Rate (%)", line=dict(color="cyan", width=1),
            ))
        if "funding_mu" in df.columns and "funding_sigma" in df.columns:
            mu  = df["funding_mu"] * 100
            sig = df["funding_sigma"] * 100
            fr_fig.add_trace(go.Scatter(
                x=df.index, y=(mu + 2 * sig).values,
                mode="lines", line=dict(color="red", dash="dot", width=1),
                name="+2σ",
            ))
            fr_fig.add_trace(go.Scatter(
                x=df.index, y=(mu - 2 * sig).values,
                mode="lines", line=dict(color="green", dash="dot", width=1),
                name="-2σ", fill="tonexty", fillcolor="rgba(0,255,0,0.05)",
            ))
        fr_fig.update_layout(template="plotly_dark", height=300,
                              title="Funding Rate with Rolling ±2σ Band")

        # L/S ratio
        ls_fig = go.Figure()
        if "ls_ratio" in df.columns:
            ls_fig.add_trace(go.Scatter(
                x=df.index, y=df["ls_ratio"],
                mode="lines", name="L/S Ratio", line=dict(color="yellow"),
            ))
        ls_fig.update_layout(template="plotly_dark", height=200,
                              title="Global Long/Short Account Ratio (live only)")

        # Basis
        basis_fig = go.Figure()
        if "basis" in df.columns:
            basis_fig.add_trace(go.Scatter(
                x=df.index, y=df["basis"] * 100,
                mode="lines", name="Basis (%)", line=dict(color="orange"),
            ))
        basis_fig.update_layout(template="plotly_dark", height=200,
                                 title="Perp-Spot Basis (%)")

        return fr_fig, ls_fig, basis_fig

    # ------------------------------------------------------------------
    # Tab 6: Live Monitor — interval-driven refresh
    # ------------------------------------------------------------------

    @app.callback(
        [
            Output("zscore-gauge",              "figure"),
            Output("live-mark-price",           "children"),
            Output("live-funding-rate",         "children"),
            Output("live-basis",                "children"),
            Output("live-ls-ratio",             "children"),
            Output("live-settlement-countdown", "children"),
            Output("live-position-card",        "children"),
            Output("settlements-table",         "data"),
        ],
        Input("live-interval", "n_intervals"),
    )
    def update_live_monitor(n):
        """
        Reads live state from a shared module (set by BinanceLiveFeed callbacks).
        Falls back to placeholder values when not connected.
        """
        try:
            from live.trader import get_live_state
            live_state = get_live_state()
        except Exception:
            live_state = {}

        z_score = live_state.get("funding_zscore", 0.0) or 0.0
        mark    = live_state.get("mark_price",     0.0) or 0.0
        fr      = live_state.get("funding_rate",   0.0) or 0.0
        basis   = live_state.get("basis",          0.0) or 0.0
        ls      = live_state.get("ls_ratio",       None)
        position = live_state.get("position",      0)
        settlements = live_state.get("settlements", [])

        # Z-score gauge
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=z_score,
            title={"text": "Funding Z-Score"},
            gauge=dict(
                axis=dict(range=[-3, 3], tickwidth=1),
                bar=dict(color="cyan"),
                steps=[
                    dict(range=[-3,   CONFIG.Z_LONG_ENTRY],  color="green"),
                    dict(range=[CONFIG.Z_LONG_ENTRY, CONFIG.Z_SHORT_ENTRY], color="gray"),
                    dict(range=[CONFIG.Z_SHORT_ENTRY, 3],    color="red"),
                ],
                threshold=dict(
                    line=dict(color="white", width=2),
                    thickness=0.75,
                    value=z_score,
                ),
            ),
        ))
        gauge_fig.update_layout(template="plotly_dark", height=280, margin=dict(t=40, b=20))

        mark_text  = f"Mark Price: ${mark:,.2f}"
        fr_text    = f"Funding Rate: {fr * 100:.4f}%"
        basis_text = f"Basis: {basis * 100:.4f}%"
        ls_text    = f"L/S Ratio: {ls:.3f}" if ls is not None else "L/S Ratio: N/A"
        countdown  = "Settlement: checking..."

        if position != 0:
            pos_card = [
                html.P(f"Direction: {'LONG' if position == 1 else 'SHORT'}", className="fw-bold"),
                html.P(f"Entry Price: ${live_state.get('entry_price', 0):,.2f}"),
                html.P(f"Bars Held: {live_state.get('bars_held', 0)}"),
            ]
        else:
            pos_card = [html.P("No open position", className="text-muted")]

        return (
            gauge_fig, mark_text, fr_text, basis_text, ls_text, countdown,
            pos_card, settlements,
        )
