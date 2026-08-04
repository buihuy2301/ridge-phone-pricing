"""Wide versions of the comparison figures, for the Beamer slides.

The report figures are 6 by 4 inches, which suits a portrait A4 page. A 16:9
slide is much wider than it is tall, so the same file has to be shrunk to
about half the slide width before it fits vertically, and the axis labels stop
being readable from the back of a room. This module re-renders the same runs
at a wider aspect ratio and with larger text, reading the cached results in
results/raw rather than recomputing anything.

    python -m src.slide_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .data import load_problem
from .plotting import plot_convergence, save_figure, set_style
from .runner import best_result, load_results

SLIDE_DIR = Path(__file__).resolve().parents[1] / "results" / "figures" / "slides"

SLIDE_FIGSIZE = (9.0, 4.0)

# group name -> (stem, title, x axis of the first panel, how to colour series)
SLIDE_FIGURES = {
    "gd_fixed": ("gd_fixed", "Gradient descent: fixed step sizes", "iter", "index"),
    "gd_backtracking": ("gd_backtracking", "Gradient descent: Armijo backtracking", "iter", "index"),
    "sgd_batch": ("sgd_batch", "SGD: batch size, step matched to each batch", "epoch", "index"),
    "sgd_common_step": ("sgd_common_step", "SGD: one common step size for every batch size", "epoch", "index"),
    "sgd_schedule": ("sgd_schedule", "SGD: step size schedules", "epoch", "index"),
    "agd": ("agd", "Accelerated gradient descent", "iter", "index"),
    "newton": ("newton", "Newton variants", "iter", "index"),
    "lbfgs": ("lbfgs", "L-BFGS: memory size", "iter", "index"),
}


def _style_for_slides() -> None:
    set_style()
    plt.rcParams.update(
        {
            "figure.figsize": SLIDE_FIGSIZE,
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.0,
        }
    )


def render_group(problem, name: str, stem: str, title: str, xaxis: str, color_by: str) -> None:
    runs, _ = load_results(name)
    for axis, suffix in ((xaxis, xaxis), ("time", "time")):
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        plot_convergence(
            runs,
            problem.f_star,
            xaxis=axis,
            ax=ax,
            title=title,
            color_by=color_by,
            n_samples=problem.n,
        )
        ax.legend(loc="best", ncol=2 if len(runs) > 5 else 1)
        fig.tight_layout()
        save_figure(fig, f"{stem}_{suffix}", figure_dir=SLIDE_DIR)
        plt.close(fig)


def render_all_methods(problem) -> None:
    """The summary figure, one best configuration per method."""
    best = []
    for name in ("gd_fixed", "gd_backtracking", "sgd_schedule", "agd", "newton", "lbfgs"):
        try:
            runs, _ = load_results(name)
        except FileNotFoundError:
            continue
        best.append(best_result(runs, problem.f_star, threshold=1e-6))

    for axis, suffix in (("iter", "iter"), ("time", "time")):
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        plot_convergence(
            best,
            problem.f_star,
            xaxis=axis,
            ax=ax,
            title="All methods at their best setup",
            color_by="method",
            n_samples=problem.n,
        )
        fig.tight_layout()
        save_figure(fig, f"all_methods_{suffix}", figure_dir=SLIDE_DIR)
        plt.close(fig)


def main() -> None:
    _style_for_slides()
    problem, _ = load_problem()

    for name, (stem, title, xaxis, color_by) in SLIDE_FIGURES.items():
        try:
            render_group(problem, name, stem, title, xaxis, color_by)
            print(f"[ok]   {stem}")
        except FileNotFoundError:
            print(f"[skip] {name}: no cached results")

    render_all_methods(problem)
    print(f"[ok]   all_methods")
    print(f"\nwritten to {SLIDE_DIR}")


if __name__ == "__main__":
    main()
