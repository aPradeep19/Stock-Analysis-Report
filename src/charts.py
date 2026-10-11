"""
Chart rendering for the PDF newsletter.

One figure per stock: price with its 50- and 200-day averages on top, volume underneath.
They are two panels sharing the date axis, never one plot with two y-axes, because a second
y-scale invents correlations that aren't in the data.

Design notes:
- Three line colors in a fixed order (price = blue, 50-day = orange, 200-day = aqua), checked
  for color-blind separation. Each line is also labeled at its right end, so color is never
  the only way to tell them apart.
- Gridlines are thin and pale so the data stands out.
- The pre-market price is drawn as a separate dot to the right of the last close, so the gap
  you read about in the email is visible on the chart.
"""

import io

import matplotlib
matplotlib.use("Agg")  # No display on GitHub's servers: render straight to an image
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

PRICE = "#2a78d6"
SMA_50 = "#eb6834"
SMA_200 = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
VOLUME = "#c3c2b7"
VOLUME_LAST = "#52514e"
UP_TEXT = "#006300"
DOWN_TEXT = "#d03b3b"


def _style_axis(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def price_volume_chart(candles: pd.DataFrame, metrics: dict, sessions: int = 126,
                       width_in: float = 7.3, height_in: float = 3.3) -> io.BytesIO:
    """
    Returns a PNG (in memory) of the last `sessions` trading days (126 = about six months).
    The moving averages are computed on the FULL history first, then trimmed, so the
    200-day line exists from the first bar shown.
    """
    df = candles.copy()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    df["vol_avg_20"] = df["volume"].rolling(20).mean()
    df = df.tail(sessions)

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(width_in, height_in), dpi=200, sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08})
    fig.patch.set_facecolor("white")

    # --- Price panel --------------------------------------------------------
    x = df.index
    lines = [("Price", df["close"], PRICE, 1.6),
             ("50-day avg", df["sma_50"], SMA_50, 1.1),
             ("200-day avg", df["sma_200"], SMA_200, 1.1)]
    for label, series, color, width in lines:
        if series.notna().any():
            ax_price.plot(x, series, color=color, linewidth=width, label=label, solid_capstyle="round")

    # Pre-market dot one day to the right of the last close, joined by a thin connector
    last_x, last_close = x[-1], df["close"].iloc[-1]
    premarket = metrics.get("premarket_price")
    gap = metrics.get("gap_pct")
    next_x = last_x + pd.Timedelta(days=3 if last_x.weekday() == 4 else 1)
    gap_color = UP_TEXT if (gap or 0) > 0 else DOWN_TEXT if (gap or 0) < 0 else INK_2
    if premarket is not None:
        ax_price.plot([last_x, next_x], [last_close, premarket], color=gap_color, linewidth=0.9)
        ax_price.scatter([next_x], [premarket], s=22, color=gap_color, zorder=5,
                         edgecolors="white", linewidths=1.2)

    # Right-edge labels: the pre-market price plus each average's name (the legend is the backup).
    # When lines end close together, the labels are nudged apart so they never overprint.
    label_x = next_x if premarket is not None else last_x
    labels = []
    if premarket is not None:
        word = {"regular": "Now", "after-hours": "After-hrs"}.get(metrics.get("premarket_session"), "Pre-mkt")
        labels.append([premarket, f"{word} ${premarket:,.2f} ({gap:+.1f}%)", gap_color, "bold", 7.5])
    for label, series, color, _ in lines[1:]:
        if series.notna().any():
            labels.append([series.iloc[-1], label, INK_2, "normal", 7])
    low, high = ax_price.get_ylim()
    min_gap = (high - low) * 0.075
    labels.sort(key=lambda item: item[0])
    for i in range(1, len(labels)):  # push overlapping labels up...
        if labels[i][0] - labels[i - 1][0] < min_gap:
            labels[i][0] = labels[i - 1][0] + min_gap
    if labels and labels[-1][0] > high - min_gap / 2:  # ...but keep the top one inside the plot
        labels[-1][0] = high - min_gap / 2
        for i in range(len(labels) - 2, -1, -1):
            if labels[i + 1][0] - labels[i][0] < min_gap:
                labels[i][0] = labels[i + 1][0] - min_gap
    for y, text, color, weight, size in labels:
        ax_price.annotate(text, (label_x, y), xytext=(6, 0), textcoords="offset points", va="center",
                          fontsize=size, color=color, fontweight=weight, annotation_clip=False)

    _style_axis(ax_price)
    ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    # Legend sits above the plot so it can never cover a line
    ax_price.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7, frameon=False, ncol=3,
                    labelcolor=INK_2, handlelength=1.6, borderaxespad=0.1)
    # Leave room on the right for the end-of-line labels
    span = (label_x - x[0])
    ax_price.set_xlim(x[0], label_x + span * 0.16)

    # --- Volume panel -------------------------------------------------------
    colors = [VOLUME] * (len(df) - 1) + [VOLUME_LAST]
    ax_vol.bar(x, df["volume"], width=0.8, color=colors, linewidth=0)
    ax_vol.plot(x, df["vol_avg_20"], color=INK_2, linewidth=0.9)
    ax_vol.annotate("20-day avg volume", (x[-1], df["vol_avg_20"].iloc[-1]), xytext=(6, 4),
                    textcoords="offset points", fontsize=6.5, color=INK_2)
    _style_axis(ax_vol)
    ax_vol.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: "0" if v == 0 else (f"{v / 1e6:.0f}M" if v >= 1e6 else f"{v / 1e3:.0f}K")))
    ax_vol.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax_vol.xaxis.set_major_locator(mdates.MonthLocator())
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)  # free memory: one figure per stock adds up
    buffer.seek(0)
    return buffer
