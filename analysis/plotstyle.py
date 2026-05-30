"""IEEE-friendly matplotlib styling: vector PDF output, embedded TrueType fonts,
small font sizes matching a two-column body, and a colorblind-safe palette."""
from __future__ import annotations

# IEEE single-column width is ~3.5 in; full width ~7.16 in.
COLUMN_WIDTH = 3.5
FULL_WIDTH = 7.16

# Okabe-Ito colorblind-safe categorical palette.
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Stable color per config code.
CONFIG_COLOR = {
    "zeroshot": "#999999",
    "frozen_head": "#E69F00",
    "lora_r8": "#0072B2",
    "lora_r16": "#56B4E9",
    "full_ft": "#D55E00",
}

_RC = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],  # covers Greek + most IPA glyphs
    "pdf.fonttype": 42,  # embed TrueType (no Type-3)
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "lines.markersize": 5,
}


def apply_ieee_style():
    import matplotlib

    matplotlib.use("Agg")  # headless / no display needed
    import matplotlib.pyplot as plt

    plt.rcParams.update(_RC)
    return plt
