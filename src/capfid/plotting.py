"""Capacity–fidelity scatter plot (requires matplotlib)."""
from __future__ import annotations

import numpy as np

__all__ = ["capacity_fidelity_plot"]


def capacity_fidelity_plot(result, highlight=None, ax=None, capacity="information_bits",
                           xlim=(1e-4, None), heldout_min=-8.0, s=12, colorbar=True,
                           highlight_label="highlighted", cmap="viridis"):
    """Scatter capacity (x, log scale) against fidelity log2 p(y|x,θ*) (y).

    Points are colored by held-out fidelity log2 p(y|x,θ*_{-i}) (clipped at ``heldout_min``);
    ``highlight`` is an optional boolean mask drawn as red-edged diamonds (e.g. suspected
    label errors). ``result`` is the dict returned by ``LinearizedLaplace.pointwise``.
    Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.6))
    x = np.asarray(result[capacity])
    lo = xlim[0]
    hi = xlim[1] if xlim[1] is not None else max(10 ** np.ceil(np.log10(x.max() + 1e-12)), lo * 10)
    x = np.clip(x, lo, hi)
    y = np.asarray(result["fidelity_bits"])
    c = np.clip(np.asarray(result["heldout_fidelity_bits"]), heldout_min, 0)
    hl = np.zeros(len(x), bool) if highlight is None else np.asarray(highlight, bool)
    kw = dict(cmap=cmap, vmin=heldout_min, vmax=0)
    sc = ax.scatter(x[~hl], y[~hl], s=s, c=c[~hl], lw=0, alpha=0.85, **kw)
    if hl.any():
        ax.scatter(x[hl], y[hl], s=s * 2.4, c=c[hl], marker="D", edgecolors="#D62728",
                   linewidths=1.1, **kw)
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color="0.5", label="other"),
                           Line2D([], [], marker="D", ls="", mfc="0.85", mec="#D62728",
                                  label=highlight_label)], fontsize=7, frameon=False,
                  loc="lower left")
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(min(-3.0, y.min() - 0.3), 0.3)
    name = {"information_bits": "KL(q_D ‖ q_D∖i)", "volume_bits": "h(θ|D∖i) − h(θ|D)"}.get(capacity, capacity)
    ax.set_xlabel(f"capacity: {name}  [bits]")
    ax.set_ylabel("fidelity: log₂ p(yᵢ | xᵢ, θ*)  [bits]")
    if colorbar:
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label("held-out log₂ p(yᵢ | xᵢ, θ*₋ᵢ)  [bits]", fontsize=8)
    return ax
