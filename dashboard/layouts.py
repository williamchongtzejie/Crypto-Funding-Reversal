"""
Dashboard layout definitions for all six tabs.
Section 12 Phase 6 of the strategy specification.
"""
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table


def tab_signal_overview() -> dbc.Tab:
    return dbc.Tab(
        label="Signal Overview",
        tab_id="tab-signal",
        children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Symbol"),
                                    dcc.Dropdown(
                                        id="signal-symbol",
                                        options=[
                                            {"label": "BTCUSDT", "value": "BTCUSDT"},
                                            {"label": "ETHUSDT", "value": "ETHUSDT"},
                                        ],
                                        value="BTCUSDT",
                                        clearable=False,
                                    ),
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Period"),
                                    dcc.Dropdown(
                                        id="signal-period",
                                        options=[
                                            {"label": "IS (2020-2022)", "value": "IS"},
                                            {"label": "OOS (2023-2024)", "value": "OOS"},
                                            {"label": "Full", "value": "FULL"},
                                        ],
                                        value="IS",
                                        clearable=False,
                                    ),
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Signal Type"),
                                    dbc.RadioItems(
                                        id="signal-type-toggle",
                                        options=[
                                            {"label": "Raw", "value": "raw"},
                                            {"label": "Confirmed", "value": "confirmed"},
                                        ],
                                        value="confirmed",
                                        inline=True,
                                        className="mt-1",
                                    ),
                                ], width=3),
                            ]),
                            dcc.DatePickerRange(
                                id="signal-date-range",
                                className="mt-3",
                            ),
                        ]),
                    ], className="mb-3"),
                ]),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="signal-price-chart", style={"height": "500px"})),
            ]),
        ],
    )


def tab_portfolio_performance() -> dbc.Tab:
    return dbc.Tab(
        label="Portfolio Performance",
        tab_id="tab-portfolio",
        children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("NAV Curve — Strategy vs BTC Buy & Hold"),
                        dbc.CardBody(dcc.Graph(id="nav-curve-chart", style={"height": "400px"})),
                    ], className="mb-3"),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Drawdown"),
                        dbc.CardBody(dcc.Graph(id="drawdown-chart", style={"height": "250px"})),
                    ]),
                ]),
            ]),
        ],
    )


def tab_signal_decay() -> dbc.Tab:
    return dbc.Tab(
        label="Signal Decay",
        tab_id="tab-decay",
        children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Rolling 6-Month Sharpe Ratio"),
                        dbc.CardBody(dcc.Graph(id="rolling-sharpe-chart", style={"height": "400px"})),
                    ]),
                ]),
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Alert(
                        "A sustained declining trend in rolling Sharpe signals alpha compression from crowding.",
                        color="warning",
                        className="mt-3",
                    ),
                ),
            ]),
        ],
    )


def tab_trade_analytics() -> dbc.Tab:
    return dbc.Tab(
        label="Trade Analytics",
        tab_id="tab-trades",
        children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Trade Log"),
                        dbc.CardBody(
                            dash_table.DataTable(
                                id="trade-table",
                                columns=[
                                    {"name": c, "id": c}
                                    for c in [
                                        "symbol", "direction", "entry_time", "exit_time",
                                        "entry_price", "exit_price", "bars_held", "hours_held",
                                        "size_pct_nav", "notional", "pnl_price", "pnl_funding",
                                        "cost_total", "net_pnl", "return_pct", "exit_reason",
                                    ]
                                ],
                                page_size=20,
                                sort_action="native",
                                filter_action="native",
                                style_table={"overflowX": "auto"},
                                style_cell={"fontSize": 12},
                                style_header={"backgroundColor": "#2c2f33", "color": "white"},
                                style_data={"backgroundColor": "#1e2124", "color": "white"},
                            ),
                        ),
                    ], className="mb-3"),
                ]),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="pnl-bar-chart",        style={"height": "300px"}), width=6),
                dbc.Col(dcc.Graph(id="hold-duration-hist",   style={"height": "300px"}), width=3),
                dbc.Col(dcc.Graph(id="exit-reason-pie",      style={"height": "300px"}), width=3),
            ]),
        ],
    )


def tab_signal_decomposition() -> dbc.Tab:
    return dbc.Tab(
        label="Signal Decomposition",
        tab_id="tab-decomp",
        children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="funding-rate-chart",  style={"height": "300px"})),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="ls-ratio-chart",      style={"height": "200px"}), width=6),
                dbc.Col(dcc.Graph(id="basis-chart",         style={"height": "200px"}), width=6),
            ]),
        ],
    )


def tab_live_monitor() -> dbc.Tab:
    return dbc.Tab(
        label="Live Monitor",
        tab_id="tab-live",
        children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Z-Score Gauge"),
                        dbc.CardBody(dcc.Graph(id="zscore-gauge", style={"height": "280px"})),
                    ]),
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Market Snapshot"),
                        dbc.CardBody([
                            html.Div(id="live-mark-price",    className="mb-2 fs-5"),
                            html.Div(id="live-funding-rate",  className="mb-2"),
                            html.Div(id="live-basis",         className="mb-2"),
                            html.Div(id="live-ls-ratio",      className="mb-2"),
                            html.Div(id="live-settlement-countdown", className="mb-2 fw-bold"),
                        ]),
                    ]),
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Open Position"),
                        dbc.CardBody(html.Div(id="live-position-card")),
                    ]),
                ], width=4),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Last 20 Settlements"),
                        dbc.CardBody(
                            dash_table.DataTable(
                                id="settlements-table",
                                columns=[
                                    {"name": c, "id": c}
                                    for c in ["time", "symbol", "funding_rate", "mark_price", "z_score", "signal"]
                                ],
                                page_size=20,
                                style_cell={"fontSize": 12},
                                style_header={"backgroundColor": "#2c2f33", "color": "white"},
                                style_data={"backgroundColor": "#1e2124", "color": "white"},
                            ),
                        ),
                    ]),
                ]),
            ]),
        ],
    )
