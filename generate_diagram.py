"""
Generates a framework architecture diagram and saves it as framework_diagram.png
Run: python3 generate_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Palette ────────────────────────────────────────────────────────────────
BG        = "#0d1117"
C_ENTRY   = "#30363d"   # dark grey  — entry points
C_ORCH    = "#6e40c9"   # purple     — orchestrator
C_DATA    = "#1f6feb"   # blue       — data layer
C_SIG     = "#0e7490"   # teal       — signal layer
C_FILT    = "#164e63"   # dark teal  — filters (sub)
C_RISK    = "#b45309"   # amber      — risk layer
C_BT      = "#166534"   # green      — backtest
C_LIVE    = "#991b1b"   # red        — live
C_DASH    = "#4c1d95"   # indigo     — dashboard
C_MODEL   = "#374151"   # slate      — shared models
C_EXT     = "#1c1c2e"   # very dark  — external systems
ARROW     = "#8b949e"
WHITE     = "#e6edf3"
GREY      = "#8b949e"
LTGREY    = "#c9d1d9"

FIG_W, FIG_H = 18, 26
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")


# ── Helpers ─────────────────────────────────────────────────────────────────

def box(x, y, w, h, color, title, lines=(), title_size=11, line_size=9,
        edge_color=None, lw=1.5, alpha=1.0, radius=0.25):
    ec = edge_color or _lighten(color)
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.08,rounding_size={radius}",
        facecolor=color, edgecolor=ec, linewidth=lw,
        alpha=alpha, zorder=2, clip_on=False,
    )
    ax.add_patch(patch)

    # title
    title_y = y + h - 0.38 if lines else y + h / 2
    ax.text(x + w / 2, title_y, title,
            ha="center", va="center", fontsize=title_size,
            fontweight="bold", color=WHITE, zorder=3)

    # subtitle lines
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, title_y - 0.42 - i * 0.38, ln,
                ha="center", va="center", fontsize=line_size,
                color=LTGREY, zorder=3)


def _lighten(hex_color, amount=0.4):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def arrow(x1, y1, x2, y2, label="", color=ARROW, lw=1.8,
          label_dx=0.15, label_dy=0.0, style="->"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                        connectionstyle="arc3,rad=0.0"),
        zorder=5,
    )
    if label:
        mx = (x1 + x2) / 2 + label_dx
        my = (y1 + y2) / 2 + label_dy
        ax.text(mx, my, label, ha="left", va="center",
                fontsize=8, color=GREY, zorder=6)


def section_label(x, y, text):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=8, color=GREY, style="italic", zorder=6)


def divider(y, alpha=0.15):
    ax.axhline(y, color=WHITE, lw=0.5, alpha=alpha, zorder=1)


# ══════════════════════════════════════════════════════════════════════════
# Layout constants
# ══════════════════════════════════════════════════════════════════════════
CX   = 9.0    # horizontal centre
BW   = 7.0    # standard box width
HBW  = 3.3    # half box width (split columns)
GAP  = 0.35   # gap between split boxes

# y-positions (top of each major block, counting down from top)
Y_TITLE = 25.0
Y_ENTRY = 23.0
Y_ORCH  = 21.2
Y_DATA  = 19.0
Y_SIG   = 16.6
Y_FILT  = 14.6
Y_RISK  = 12.5
Y_SPLIT = 10.3   # backtester | live trader
Y_OUT   = 7.8    # results / websocket
Y_DASH  = 5.5
Y_MODEL = 3.0


# ══════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════
ax.text(CX, Y_TITLE, "Crypto Funding Reversal — Framework Architecture",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color=WHITE, zorder=6)
ax.text(CX, Y_TITLE - 0.55, "Full pipeline from data ingestion to live order execution",
        ha="center", va="center", fontsize=10, color=GREY, zorder=6)


# ══════════════════════════════════════════════════════════════════════════
# Entry Points
# ══════════════════════════════════════════════════════════════════════════
EW = 3.3
box(CX - EW - 0.2, Y_ENTRY, EW, 0.9, C_ENTRY,
    "run_backtest.py", ["Backtest entry point"], line_size=8)
box(CX + 0.2,       Y_ENTRY, EW, 0.9, C_ENTRY,
    "run_live.py", ["Live execution entry point"], line_size=8)

section_label(0.4, Y_ENTRY + 0.45, "Entry\nPoints")

# arrows entry → orchestrator
arrow(CX - EW / 2 - 0.2 + EW, Y_ENTRY, CX - 0.5, Y_ORCH + 1.1)
arrow(CX + 0.2 + EW / 2,      Y_ENTRY, CX + 0.5, Y_ORCH + 1.1)


# ══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════
box(CX - BW / 2, Y_ORCH, BW, 1.0, C_ORCH,
    "FundingReversalStrategy   ·   strategy.py",
    ["run_backtest()  ·  create_live_trader()"])

section_label(0.4, Y_ORCH + 0.5, "Orchestrator")

arrow(CX, Y_ORCH, CX, Y_DATA + 1.85, label="raw dict")


# ══════════════════════════════════════════════════════════════════════════
# Config — side panel
# ══════════════════════════════════════════════════════════════════════════
box(0.3, Y_DATA - 0.2, 2.5, 2.3, C_MODEL,
    "config.py",
    ["StrategyConfig", "30+ typed params", "Single source", "of truth"],
    title_size=10, line_size=8)

# dashed config arrows to each layer
for y_tgt in [Y_DATA + 0.9, Y_SIG + 0.8, Y_RISK + 0.55]:
    ax.annotate(
        "", xy=(CX - BW / 2, y_tgt), xytext=(0.3 + 2.5, y_tgt),
        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0,
                        linestyle="dashed"),
        zorder=4,
    )


# ══════════════════════════════════════════════════════════════════════════
# Data Layer
# ══════════════════════════════════════════════════════════════════════════
DH = 1.8
box(CX - BW / 2, Y_DATA, BW, DH, C_DATA,
    "DataPipeline   ·   data/pipeline.py",
    ["fetch()  ·  build()  ·  fetch_and_build()",
     "generate_synthetic()  ·  Parquet cache"])

section_label(0.4, Y_DATA + DH / 2, "Data\nLayer")

# external: Binance REST
box(14.8 - 2.8, Y_DATA + 0.4, 3.2, 1.0, C_EXT,
    "Binance REST API",
    ["Funding rates · OHLCV", "Mark/Index · L/S ratio"],
    title_size=9, line_size=8, edge_color="#30363d")
arrow(14.8 - 2.8, Y_DATA + 0.9, CX + BW / 2, Y_DATA + 0.9,
      color="#30363d", style="<-")

arrow(CX, Y_DATA, CX, Y_SIG + 1.7, label="Master DataFrame")


# ══════════════════════════════════════════════════════════════════════════
# Signal Layer
# ══════════════════════════════════════════════════════════════════════════
SH = 1.7
box(CX - BW / 2, Y_SIG, BW, SH, C_SIG,
    "SignalPipeline   ·   signals/pipeline.py",
    ["compute_zscore()  ·  raw_signal()  ·  apply_filters()  ·  run()",
     "z > +2.0σ → SHORT  ·  z < -1.5σ → LONG  ·  exit |z| < 0.5σ"])

section_label(0.4, Y_SIG + SH / 2, "Signal\nLayer")

arrow(CX, Y_SIG, CX, Y_FILT + 0.85, label="raw signal")


# ══════════════════════════════════════════════════════════════════════════
# Filter sub-boxes
# ══════════════════════════════════════════════════════════════════════════
FW  = 2.1
FH  = 0.82
FY  = Y_FILT
FX1 = CX - BW / 2
FX2 = FX1 + FW + 0.15
FX3 = FX2 + FW + 0.15

box(FX1, FY, FW, FH, C_FILT,
    "Filter A — L/S Ratio",
    ["Crowd positioning gate"], title_size=9, line_size=8)
box(FX2, FY, FW, FH, C_FILT,
    "Filter B — Basis",
    ["Perp-spot premium gate"], title_size=9, line_size=8)
box(FX3, FY, FW, FH, C_FILT,
    "Filter C — Regime",
    ["Trend guard (z-reg > 2.5)"], title_size=9, line_size=8)

# arrows filter → risk
arrow(FX1 + FW / 2, FY, CX, Y_RISK + 1.05, color=_lighten(C_FILT, 0.3))
arrow(FX2 + FW / 2, FY, CX, Y_RISK + 1.05, color=_lighten(C_FILT, 0.3))
arrow(FX3 + FW / 2, FY, CX, Y_RISK + 1.05, color=_lighten(C_FILT, 0.3))

ax.text(CX, Y_FILT - 0.25, "confirmed_signal",
        ha="center", va="center", fontsize=8, color=GREY)


# ══════════════════════════════════════════════════════════════════════════
# Risk Layer
# ══════════════════════════════════════════════════════════════════════════
RH = 1.0
box(CX - BW / 2, Y_RISK, BW, RH, C_RISK,
    "RiskManager   ·   risk/manager.py",
    ["compute_sizes()  Half-Kelly · vol scalar · 2% NAV cap",
     "is_halted()  Circuit breaker — halt at 15% drawdown, resume at 10%"])

section_label(0.4, Y_RISK + RH / 2, "Risk\nLayer")

# fork arrows
LX = CX - HBW - GAP / 2
RX = CX + GAP / 2
arrow(CX - 1.2, Y_RISK, LX + HBW / 2, Y_SPLIT + 1.6,
      label="final_size", color=ARROW)
arrow(CX + 1.2, Y_RISK, RX + HBW / 2, Y_SPLIT + 1.6,
      color=ARROW)


# ══════════════════════════════════════════════════════════════════════════
# Backtester + LiveTrader (split)
# ══════════════════════════════════════════════════════════════════════════
SPH = 1.65

box(LX, Y_SPLIT, HBW, SPH, C_BT,
    "Backtester   ·   backtest/runner.py",
    ["run()  Bar-by-bar simulation",
     "compute_metrics()  Sharpe, MDD, beta…",
     "export()  CSV + Parquet"])

box(RX, Y_SPLIT, HBW, SPH, C_LIVE,
    "LiveTrader   ·   live/trader.py",
    ["on_settlement()  Signal pipeline",
     "on_mark_price_tick()  ATR stop",
     "_enter() / _exit()  Market orders"])

section_label(0.4, Y_SPLIT + SPH / 2, "Execution\nLayer")

arrow(LX + HBW / 2, Y_SPLIT, LX + HBW / 2, Y_OUT + 1.15, label="results/")
arrow(RX + HBW / 2, Y_SPLIT, RX + HBW / 2, Y_OUT + 1.15)


# ══════════════════════════════════════════════════════════════════════════
# Outputs
# ══════════════════════════════════════════════════════════════════════════
OH = 1.1
box(LX, Y_OUT, HBW, OH, C_MODEL,
    "results/",
    ["*_trades.csv  ·  *_master.parquet",
     "performance_summary.csv"],
    title_size=10, line_size=8)

# WebSocket feed
box(RX, Y_OUT, HBW, OH, C_EXT,
    "BinanceLiveFeed   ·   websocket_feed.py",
    ["markPrice@1s  ·  kline_1h",
     "Settlement detection  ·  Reconnect"],
    title_size=9, line_size=8, edge_color="#30363d")

# arrow from results → dashboard
arrow(LX + HBW / 2, Y_OUT, LX + HBW / 2, Y_DASH + 0.85, label="reads parquet/CSV")

# arrow from Binance REST → LiveTrader (orders)
box(RX, Y_DASH, HBW, 0.85, C_EXT,
    "Binance Futures API",
    ["/fapi/v1/order  ·  MARKET fills"],
    title_size=9, line_size=8, edge_color="#30363d")
arrow(RX + HBW / 2, Y_OUT, RX + HBW / 2, Y_DASH + 0.85,
      label="signed orders", color="#30363d")


# ══════════════════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════════════════
DH2 = 0.85
box(LX, Y_DASH, HBW, DH2, C_DASH,
    "Plotly Dash Dashboard   ·   dashboard/",
    ["Signal · Performance · Decay · Trades · Decomposition · Live"],
    title_size=10, line_size=8)


# ══════════════════════════════════════════════════════════════════════════
# Shared models
# ══════════════════════════════════════════════════════════════════════════
box(CX - BW / 2, Y_MODEL, BW, 0.85, C_MODEL,
    "models.py   ·   Shared Data Containers",
    ["TradeRecord  ·  BacktestResult  ·  StrategyMetrics"],
    title_size=10, line_size=9)

section_label(0.4, Y_MODEL + 0.42, "Shared\nModels")

# dotted arrow from models to backtester and live
for tx, ty in [(LX + HBW / 2, Y_SPLIT),
               (RX + HBW / 2, Y_SPLIT)]:
    ax.annotate(
        "", xy=(tx, ty), xytext=(CX, Y_MODEL + 0.85),
        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.9,
                        linestyle="dotted"),
        zorder=4,
    )


# ══════════════════════════════════════════════════════════════════════════
# Legend
# ══════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_ORCH,  "Orchestrator"),
    (C_DATA,  "Data Layer"),
    (C_SIG,   "Signal Layer"),
    (C_RISK,  "Risk Layer"),
    (C_BT,    "Backtest Layer"),
    (C_LIVE,  "Live Layer"),
    (C_DASH,  "Dashboard"),
    (C_MODEL, "Shared / Config"),
]
lx0, ly0 = 0.4, 1.6
for i, (color, label) in enumerate(legend_items):
    lx = lx0 + i * 2.15
    patch = FancyBboxPatch((lx, ly0), 0.35, 0.28,
                            boxstyle="round,pad=0.04",
                            facecolor=color, edgecolor=_lighten(color),
                            linewidth=1, zorder=6)
    ax.add_patch(patch)
    ax.text(lx + 0.45, ly0 + 0.14, label,
            ha="left", va="center", fontsize=8, color=LTGREY, zorder=7)

ax.text(CX, 0.85, "Solid arrows = data flow   ·   Dashed arrows = config injection   ·   Dotted arrows = type dependency",
        ha="center", va="center", fontsize=8, color=GREY)


# ══════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════
out_path = "framework_diagram.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print(f"Saved → {out_path}")
