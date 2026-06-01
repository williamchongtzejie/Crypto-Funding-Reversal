"""
Dashboard entry point for Strategy 3 — Funding Rate Mean Reversion.
Run: python run_dashboard.py
Opens at http://localhost:8050
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

from dashboard.layouts import (
    tab_signal_overview,
    tab_portfolio_performance,
    tab_signal_decay,
    tab_trade_analytics,
    tab_signal_decomposition,
    tab_live_monitor,
)
import dashboard.callbacks as cbs


def create_app() -> Dash:
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
    )
    app.title = "Crypto Funding Reversal"

    app.layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Row([
                dbc.Col(
                    html.H4(
                        "Crypto Funding Reversal",
                        className="my-3 text-white",
                    ),
                    width=10,
                ),
                dbc.Col(
                    html.Div(id="header-status", className="my-3 text-end text-secondary"),
                    width=2,
                ),
            ], className="border-bottom border-secondary mb-3"),

            dbc.Tabs(
                id="main-tabs",
                active_tab="tab-signal",
                children=[
                    tab_signal_overview(),
                    tab_portfolio_performance(),
                    tab_signal_decay(),
                    tab_trade_analytics(),
                    tab_signal_decomposition(),
                    tab_live_monitor(),
                ],
            ),

            # 30-second refresh for live tab
            dcc.Interval(id="live-interval", interval=30_000, n_intervals=0),

            # Store selected symbol + period across tab switches
            dcc.Store(id="selected-symbol", data="BTCUSDT"),
            dcc.Store(id="selected-period", data="IS"),
        ],
        style={"minHeight": "100vh"},
    )

    cbs.register(app)
    return app
