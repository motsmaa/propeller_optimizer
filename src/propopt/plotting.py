"""Phase 3 plotting — OPTIONAL (matplotlib is an optional extra, not a runtime
dependency). Every entry point imports matplotlib lazily and raises a clear
PlottingUnavailable if it is missing; callers degrade gracefully. Uses the
non-interactive Agg backend (no display required) and saves to files.

Four objectives don't fit one scatter, so:
  - a pairwise scatter matrix (every objective pair),
  - a parallel-coordinates plot (all four axes at once),
with constraint-binding designs visually distinguished in both.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import DesignRecord
from .config import OBJECTIVES

_LABELS = {"thrust_N": "thrust [N]", "figure_of_merit": "FoM",
           "mass_kg": "mass [kg]", "safety_factor": "safety factor"}


class PlottingUnavailable(RuntimeError):
    """Raised when matplotlib (the optional 'plot' extra) is not installed."""


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")           # non-interactive; no display needed
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:          # pragma: no cover - exercised via skip
        raise PlottingUnavailable(
            "matplotlib not installed; install the optional 'plot' extra: "
            "pip install -e '.[plot]'") from exc


def _objective_matrix(records):
    f = np.array([[r.objectives[o] for o in OBJECTIVES] for r in records])
    binding = np.array([any(v == "binding" for v in r.activity.values())
                        for r in records])
    return f, binding


def scatter_matrix(records: list[DesignRecord], path: str | Path) -> Path:
    """Pairwise objective scatter matrix; binding designs marked distinctly."""
    plt = _mpl()
    f, binding = _objective_matrix(records)
    m = len(OBJECTIVES)
    fig, axes = plt.subplots(m, m, figsize=(2.6 * m, 2.6 * m))
    for i in range(m):
        for j in range(m):
            ax = axes[i, j]
            if i == j:
                ax.hist(f[:, i], bins=12, color="#bbbbbb")
            else:
                ax.scatter(f[~binding, j], f[~binding, i], s=14,
                           c="#4477aa", label="interior")
                ax.scatter(f[binding, j], f[binding, i], s=34, c="#cc3311",
                           marker="D", label="binding")
            if i == m - 1:
                ax.set_xlabel(_LABELS[OBJECTIVES[j]], fontsize=8)
            if j == 0:
                ax.set_ylabel(_LABELS[OBJECTIVES[i]], fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle("Pareto front — objective scatter matrix "
                 "(red diamonds = constraint-binding)")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def parallel_coordinates(records: list[DesignRecord], path: str | Path) -> Path:
    """Parallel-coordinates plot of the 4 objectives (each min-max normalized),
    binding designs highlighted."""
    plt = _mpl()
    f, binding = _objective_matrix(records)
    lo, hi = f.min(axis=0), f.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    norm = (f - lo) / span
    m = len(OBJECTIVES)
    fig, ax = plt.subplots(figsize=(1.9 * m, 4.5))
    xs = np.arange(m)
    for row, b in zip(norm, binding):
        ax.plot(xs, row, color="#cc3311" if b else "#4477aa",
                lw=1.6 if b else 0.7, alpha=0.9 if b else 0.45,
                zorder=3 if b else 1)
    ax.set_xticks(xs)
    ax.set_xticklabels([_LABELS[o] for o in OBJECTIVES], fontsize=8)
    ax.set_ylabel("min-max normalized")
    ax.set_title("Pareto front — parallel coordinates\n"
                 "(red = constraint-binding, blue = interior)", fontsize=10)
    # annotate real min/max at each axis
    for k in range(m):
        ax.text(k, -0.06, f"{lo[k]:.3g}", ha="center", va="top", fontsize=6)
        ax.text(k, 1.04, f"{hi[k]:.3g}", ha="center", va="bottom", fontsize=6)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def hv_convergence(convergence: dict, path: str | Path) -> Path:
    """HV-vs-generation curves, one line per seed."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for seed, hist in convergence["histories"].items():
        ax.plot(np.arange(1, len(hist) + 1), hist, label=f"seed {seed}", lw=1.4)
    ax.set_xlabel("generation")
    ax.set_ylabel("feasible-archive hypervolume (normalized)")
    ax.set_title("Convergence: HV vs generation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
