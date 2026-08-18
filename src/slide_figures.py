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

import json
from pathlib import Path

import matplotlib

# The backend has to be selected before pyplot is imported, which is why the
# imports below sit past the top of the module.
matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt

from .data import load_problem
from .plotting import (
    plot_convergence,
    plot_rmse_vs_gap,
    plot_rmse_vs_time,
    plot_theoretical_rate,
    save_figure,
    set_style,
)
from .runner import best_result, load_results

ROOT = Path(__file__).resolve().parents[1]
SLIDE_DIR = ROOT / "results" / "figures" / "slides"
RAW_DIR = ROOT / "results" / "raw"

SLIDE_FIGSIZE = (9.0, 4.0)

# group name -> (stem, title, x axis of the first panel, how to colour series)
SLIDE_FIGURES = {
    "gd_fixed": ("gd_fixed", "Gradient descent: fixed step sizes", "iter", "index"),
    "gd_backtracking": (
        "gd_backtracking",
        "Gradient descent: Armijo backtracking",
        "iter",
        "index",
    ),
    "sgd_batch": ("sgd_batch", "SGD: batch size, step matched to each batch", "epoch", "index"),
    "sgd_common_step": (
        "sgd_common_step",
        "SGD: one common step size for every batch size",
        "epoch",
        "index",
    ),
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
    """Redraw one comparison group at slide size, from its cached runs."""
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


def _to_slide_size(fig, stem: str) -> None:
    """Save a figure that was laid out at report size as a slide-width copy.

    The two RMSE panels build their own figure rather than drawing into an axis
    handed to them, so widening happens after the fact. Tick and axis labels
    already carry the slide font sizes from the rcParams above; only the legend
    sets its own size inside the plotting helper, so it is bumped here.
    """
    fig.set_size_inches(*SLIDE_FIGSIZE)
    legend = fig.axes[0].get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(10)
    fig.tight_layout()
    save_figure(fig, stem, figure_dir=SLIDE_DIR)
    plt.close(fig)


def render_rmse_panels(problem) -> None:
    """The two panels that read the runs in price units instead of gap units."""
    runs, _ = load_results("rmse_tracking")
    derived = json.loads((RAW_DIR / "rmse_threshold.json").read_text())
    reference = derived["rmse_at_closed_form"]

    fig = plot_rmse_vs_gap(
        runs,
        problem.f_star,
        title="Test RMSE against the optimization gap",
        reference=reference,
        save=False,
    )
    _to_slide_size(fig, "rmse_vs_gap")

    fig = plot_rmse_vs_time(
        runs,
        title="Test RMSE against wall-clock time",
        reference=reference,
        # Same window as the report figure. The first history point lands a few
        # microseconds in, so the full log axis would spend five decades on the
        # gap before any run starts.
        xlim=(1e-2, 40.0),
        save=False,
    )
    _to_slide_size(fig, "rmse_vs_time")


def render_theory(problem) -> None:
    """Observed convergence against the two textbook bounds.

    Same construction as the report figure: the best fixed-step run and the
    best accelerated run, with both bounds started from the same initial gap.
    """
    best_gd = best_result(load_results("gd_fixed")[0], problem.f_star)
    best_agd = best_result(load_results("agd")[0], problem.f_star)

    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    for result in (best_gd, best_agd):
        ax.semilogy(
            result.iter_hist,
            result.suboptimality(problem.f_star),
            label=result.params["label"],
        )

    gap0 = best_gd.f_hist[0] - problem.f_star
    n_show = min(best_gd.n_iter, 1200)
    plot_theoretical_rate(ax, gap0, problem.kappa, n_show, kind="gd")
    plot_theoretical_rate(ax, gap0, problem.kappa, n_show, kind="agd")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f(w_k) - f^*$")
    ax.set_title("Observed convergence against the theoretical bounds")
    ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, "theory_vs_practice", figure_dir=SLIDE_DIR)
    plt.close(fig)


def main() -> None:
    """Render every slide figure for which cached results exist."""
    _style_for_slides()
    problem, _ = load_problem()

    for name, (stem, title, xaxis, color_by) in SLIDE_FIGURES.items():
        try:
            render_group(problem, name, stem, title, xaxis, color_by)
            print(f"[ok]   {stem}")
        except FileNotFoundError:
            print(f"[skip] {name}: no cached results")

    render_all_methods(problem)
    print("[ok]   all_methods")

    for label, render in (("rmse panels", render_rmse_panels), ("theory_vs_practice", render_theory)):
        try:
            render(problem)
            print(f"[ok]   {label}")
        except FileNotFoundError:
            print(f"[skip] {label}: no cached results")
    print(f"\nwritten to {SLIDE_DIR}")


if __name__ == "__main__":
    main()
