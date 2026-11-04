"""Estilo común de las figuras del artículo (paleta validada con el validador CVD de dataviz)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Orden categórico fijo (nunca ciclado). MDD/HC = slots 2/1.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GROUP_COLORS = {"HC": SERIES[0], "MDD": SERIES[1]}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#104281", "#3987e5", NEUTRAL, "#e66767", "#9e1f1f"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 8.5,
        "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 9,
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": INK2,
        "ytick.color": INK2, "legend.frameon": False, "lines.linewidth": 2.0, "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
    })


def save(fig, path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path) + ".pdf")
    fig.savefig(str(path) + ".png")
    plt.close(fig)
