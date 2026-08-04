"""Plotting helpers.

Every comparison in the report consists of two figures built from the same
runs: suboptimality against the iteration count, and suboptimality against
wall-clock time. `plot_comparison` produces both and saves them under a
common stem.

All text on the figures is in English. Colors are assigned per method and
kept fixed across the whole report; inside a single-method comparison the
runs are separated by line style and shade instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

FIGURE_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

# One fixed color per method, used everywhere in the report.
METHOD_COLORS: dict[str, str] = {
    "gradient_descent": "#1f77b4",
    "sgd": "#ff7f0e",
    "accelerated_gradient": "#2ca02c",
    "newton": "#d62728",
    "newton_cg": "#9467bd",
    "lbfgs": "#8c564b",
    "heavy_ball": "#e377c2",
    "adam": "#7f7f7f",
    "sklearn": "#17becf",
    "coordinate_descent": "#bcbd22",
}

# Shades used when several runs of the same method are compared.
SHADE_CYCLE = plt.get_cmap("viridis")

LINESTYLES = ["-", "--", "-.", ":"]

DEFAULT_FIGSIZE = (6.0, 4.0)


def set_style() -> None:
    """Global matplotlib settings, applied once per notebook."""
    mpl.rcParams.update(
        {
            "figure.figsize": DEFAULT_FIGSIZE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.6,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
        }
    )


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#333333")


def _series_style(results: Sequence, color_by: str) -> list[dict]:
    """Color and line style for each run in a comparison."""
    styles = []
    if color_by == "method":
        seen: dict[str, int] = {}
        for res in results:
            index = seen.get(res.method, 0)
            seen[res.method] = index + 1
            styles.append(
                {"color": method_color(res.method), "linestyle": LINESTYLES[index % len(LINESTYLES)]}
            )
    else:
        n = max(len(results), 2)
        for i, _ in enumerate(results):
            styles.append({"color": SHADE_CYCLE(i / (n - 1) * 0.85), "linestyle": "-"})
    return styles


def _x_values(result, xaxis: str, n_samples: int | None) -> np.ndarray:
    if xaxis == "iter":
        return np.asarray(result.iter_hist, dtype=np.float64)
    if xaxis == "time":
        return np.asarray(result.time_hist, dtype=np.float64)
    if xaxis == "epoch":
        if n_samples is None:
            raise ValueError("n_samples is required for the epoch axis")
        return result.epochs(n_samples)
    if xaxis == "access":
        return np.asarray(result.access_hist, dtype=np.float64)
    raise ValueError(f"unknown x axis: {xaxis}")


X_LABELS = {
    "iter": "Iteration",
    "time": "Wall-clock time (s)",
    "epoch": "Effective passes over the data",
    "access": "Cumulative sample gradient evaluations",
}


def plot_convergence(
    results: Sequence,
    f_star: float,
    xaxis: str = "iter",
    ax=None,
    title: str | None = None,
    labels: Sequence[str] | None = None,
    color_by: str = "index",
    n_samples: int | None = None,
    floor: float = 1e-16,
    ylabel: str = r"$f(w_k) - f^*$",
):
    """Draw one convergence panel on a logarithmic vertical axis."""
    if ax is None:
        _, ax = plt.subplots()

    styles = _series_style(results, color_by)
    for result, style, index in zip(results, styles, range(len(results))):
        x = _x_values(result, xaxis, n_samples)
        y = result.suboptimality(f_star, floor=floor)
        label = labels[index] if labels is not None else result.params.get("label", result.method)
        if result.status == "diverged":
            label = f"{label} [diverged]"
        ax.semilogy(x, y, label=label, **style)

    ax.set_xlabel(X_LABELS[xaxis])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    return ax


def plot_comparison(
    results: Sequence,
    f_star: float,
    stem: str,
    title: str = "",
    labels: Sequence[str] | None = None,
    color_by: str = "index",
    n_samples: int | None = None,
    first_axis: str = "iter",
    save: bool = True,
    figure_dir: Path | None = None,
):
    """Build the iteration panel and the time panel for one comparison.

    Returns the two figures. Files are written as `<stem>_iter.{pdf,png}` and
    `<stem>_time.{pdf,png}`.
    """
    figures = []
    for xaxis, suffix in ((first_axis, first_axis), ("time", "time")):
        fig, ax = plt.subplots()
        plot_convergence(
            results,
            f_star,
            xaxis=xaxis,
            ax=ax,
            title=title,
            labels=labels,
            color_by=color_by,
            n_samples=n_samples,
        )
        fig.tight_layout()
        if save:
            save_figure(fig, f"{stem}_{suffix}", figure_dir=figure_dir)
        figures.append(fig)
    return tuple(figures)


def save_figure(fig, name: str, figure_dir: Path | None = None) -> list[Path]:
    """Write a figure as PDF (for LaTeX) and PNG (for quick viewing)."""
    directory = Path(figure_dir) if figure_dir is not None else FIGURE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("pdf", "png"):
        path = directory / f"{name}.{extension}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    return paths


def plot_theoretical_rate(ax, f0_gap: float, kappa: float, n_iter: int, kind: str = "gd") -> None:
    """Overlay the textbook linear rate for a strongly convex quadratic.

    kind = "gd"  -> ((kappa - 1) / (kappa + 1))^(2k)
    kind = "agd" -> (1 - 1 / sqrt(kappa))^k
    """
    k = np.arange(n_iter + 1, dtype=np.float64)
    if kind == "gd":
        rate = ((kappa - 1.0) / (kappa + 1.0)) ** (2.0 * k)
        label = r"GD bound $\left(\frac{\kappa-1}{\kappa+1}\right)^{2k}$"
    elif kind == "agd":
        rate = (1.0 - 1.0 / np.sqrt(kappa)) ** k
        label = r"AGD bound $\left(1-\kappa^{-1/2}\right)^{k}$"
    else:
        raise ValueError(f"unknown rate: {kind}")
    ax.semilogy(k, f0_gap * rate, color="black", linestyle=":", linewidth=1.2, label=label)
    ax.legend(loc="best")


def observed_rate(result, f_star: float, skip: float = 0.2) -> float | None:
    """Fit a linear rate to the log of the suboptimality.

    Returns the per-iteration contraction factor estimated on the tail of the
    run, or None when the history is too short or already at machine
    precision. `skip` is the fraction of the history dropped from the start.
    """
    gap = result.suboptimality(f_star, floor=0.0)
    iters = np.asarray(result.iter_hist, dtype=np.float64)
    mask = gap > 1e-14
    if mask.sum() < 5:
        return None
    start = int(skip * mask.sum())
    x = iters[mask][start:]
    y = np.log(gap[mask][start:])
    if x.size < 5:
        return None
    slope = np.polyfit(x, y, 1)[0]
    return float(np.exp(slope))


def summary_table(results: Iterable, f_star: float, threshold: float = 1e-6):
    """Rows for the quantitative comparison table.

    Requires pandas, which is imported lazily so that plotting alone does not
    depend on it.
    """
    import pandas as pd

    rows = []
    for result in results:
        rows.append(
            {
                "method": result.method,
                "setup": result.params.get("label", ""),
                "status": result.status,
                f"iters_to_{threshold:g}": result.iters_to_reach(f_star, threshold),
                f"time_to_{threshold:g}_s": result.time_to_reach(f_star, threshold),
                "final_f": result.f_final,
                "final_gap": result.f_final - f_star,
                "final_grad_norm": result.gnorm_hist[-1],
                "total_time_s": result.total_time,
                "n_f_evals": result.n_f_evals,
                "n_grad_evals": result.n_grad_evals,
                # Stochastic methods never evaluate the full objective or the
                # full gradient, so the two counters above are zero for them.
                # The comparable measure of work is how many individual sample
                # gradients were touched.
                "n_sample_grads": result.access_hist[-1] if result.access_hist else 0,
            }
        )
    return pd.DataFrame(rows)
