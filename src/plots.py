"""Estilo comum das figuras (matplotlib) e utilitários de gravação."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .utils import CLASSES, out, register  # noqa: E402

# 10 classes: 8 tons categóricos em ordem fixa + cinzento + castanho. Como 10 cores nunca são todas
# distinguíveis entre si (sobretudo com daltonismo), os gráficos de dispersão levam também o nome
# da classe escrito no centróide de cada grupo.
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                "#008300", "#6250d6", "#e34948", "#7a7a7a", "#8c5a2b"]
SERIES = CLASS_COLORS[:8]
SEQ = LinearSegmentedColormap.from_list(
    "seq_azul", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#c2410c", "#f2b8a2", "#e9e8e4", "#9ec5f4", "#1c5cab"])
SHORT = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#bdbcb6", "axes.linewidth": 0.8, "axes.grid": True,
        "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "text.color": "#0b0b0b", "lines.linewidth": 2.0, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 9, "font.family": "DejaVu Sans",
    })


setup()


def savefig(fig, name: str, description: str, config: dict | None = None):
    p = out("figures", name)
    fig.savefig(p)
    plt.close(fig)
    register(p, description, config, "png")
    return p


def scatter_classes(ax, Z, y, s=2, alpha=0.6, labels=True, fontsize=7):
    """Dispersão 2D colorida por classe com o nome da classe no centróide (mediana) de cada grupo."""
    Z = np.asarray(Z)
    for c in range(10):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], s=s, c=CLASS_COLORS[c], alpha=alpha, linewidths=0, rasterized=True)
    if labels:
        for c in range(10):
            mx, my = np.median(Z[y == c], axis=0)
            ax.text(mx, my, SHORT[c], fontsize=fontsize, ha="center", va="center", color="#0b0b0b",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def class_legend(fig, loc="lower center", ncol=10, **kw):
    handles = [plt.Line2D([], [], marker="o", ls="", color=CLASS_COLORS[c], markersize=6) for c in range(10)]
    fig.legend(handles, CLASSES, loc=loc, ncol=ncol, **kw)


def show_images(axes, imgs, titles=None, fontsize=7):
    for i, ax in enumerate(np.ravel(axes)):
        ax.axis("off")
        if i < len(imgs):
            ax.imshow(np.asarray(imgs[i]).reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
            if titles is not None:
                ax.set_title(titles[i], fontsize=fontsize)


def heatmap(ax, M, xlabels, ylabels, fmt="{:.2f}", cmap=None, vmin=None, vmax=None, fontsize=7):
    im = ax.imshow(M, cmap=cmap or SEQ, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.grid(False)
    lo, hi = np.nanmin(M), np.nanmax(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            dark = (v - lo) / (hi - lo + 1e-12) > 0.6
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                    color="white" if dark else "#0b0b0b")
    return im
