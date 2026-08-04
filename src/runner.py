"""Experiment runner.

A run is described by a plain dictionary so that a whole parameter grid can
be written down as data, stored next to its results, and replayed later:

    {"method": "gradient_descent",
     "kwargs": {"max_iter": 2000, "step_rule": {"kind": "fixed", "multiple": 1.0}}}

Each configuration is executed `repeats` times and the run with the median
total running time is kept, which removes most of the noise coming from other
processes on the machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .first_order import accelerated_gradient, adam, gradient_descent, heavy_ball, sgd
from .history import OptimizeResult, warm_up
from .second_order import lbfgs, newton, newton_cg

RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "raw"

METHODS: dict[str, Callable] = {
    "gradient_descent": gradient_descent,
    "sgd": sgd,
    "accelerated_gradient": accelerated_gradient,
    "heavy_ball": heavy_ball,
    "adam": adam,
    "newton": newton,
    "newton_cg": newton_cg,
    "lbfgs": lbfgs,
}


def run_one(problem, spec: dict, repeats: int = 3, warm: bool = True) -> OptimizeResult:
    """Execute one configuration and return the median-time run."""
    method_name = spec["method"]
    method = METHODS[method_name]
    kwargs = dict(spec.get("kwargs", {}))

    if warm:
        warm_up(problem)

    runs = [method(problem, **kwargs) for _ in range(max(1, repeats))]
    times = [run.total_time for run in runs]
    median_index = int(np.argsort(times)[len(times) // 2])
    chosen = runs[median_index]

    chosen.params["repeats"] = repeats
    chosen.params["time_all_runs"] = times
    if "label" in spec:
        chosen.params["label"] = spec["label"]
    return chosen


def run_grid(
    problem,
    specs: Sequence[dict],
    repeats: int = 3,
    warm: bool = True,
    verbose: bool = True,
) -> list[OptimizeResult]:
    """Execute every configuration in a grid."""
    results = []
    for i, spec in enumerate(specs, start=1):
        result = run_one(problem, spec, repeats=repeats, warm=warm)
        results.append(result)
        if verbose:
            gap = result.f_final - problem.f_star
            print(
                f"[{i}/{len(specs)}] {result.params.get('label', result.method):<45s} "
                f"status={result.status:<10s} gap={gap:.3e} time={result.total_time:.4f}s"
            )
    return results


def best_result(results: Sequence[OptimizeResult], f_star: float, threshold: float = 1e-6):
    """Pick the configuration that reaches the threshold in the least time.

    Runs that never reach the threshold are ranked afterwards by final gap.
    """
    reached = [r for r in results if r.time_to_reach(f_star, threshold) is not None]
    if reached:
        return min(reached, key=lambda r: r.time_to_reach(f_star, threshold))
    return min(results, key=lambda r: r.f_final)


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def save_results(results: Sequence[OptimizeResult], name: str, extra: dict | None = None) -> Path:
    """Write a set of runs to results/raw/<name>.json."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"{name}.json"
    payload = {
        "name": name,
        "extra": extra or {},
        "runs": [result.to_dict() for result in results],
    }
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_results(name: str) -> tuple[list[OptimizeResult], dict]:
    """Read back a set of runs saved by `save_results`."""
    path = RESULT_DIR / f"{name}.json"
    payload = json.loads(path.read_text())
    runs = [OptimizeResult.from_dict(item) for item in payload["runs"]]
    return runs, payload.get("extra", {})


def run_or_load(
    problem,
    name: str,
    specs: Sequence[dict],
    repeats: int = 3,
    extra: dict | None = None,
    force: bool = False,
    verbose: bool = True,
) -> list[OptimizeResult]:
    """Run a group of configurations, or reuse the results already on disk.

    A full sweep takes long enough that an interruption in the middle is a
    real possibility. Each group is written to its own file as soon as it
    finishes, and this helper skips any group whose file is already present,
    so restarting the notebook picks up where it stopped instead of starting
    over. Pass force=True to recompute a group after changing its grid.
    """
    path = RESULT_DIR / f"{name}.json"
    if path.exists() and not force:
        runs, _ = load_results(name)
        if verbose:
            print(f"[skip] {name}: reusing {len(runs)} runs from {path.name}")
        return runs

    runs = run_grid(problem, specs, repeats=repeats, verbose=verbose)
    save_results(runs, name, extra=extra)
    if verbose:
        print(f"[done] {name}: {len(runs)} runs written to {path.name}")
    return runs


# ----------------------------------------------------------------------
# Grid builders
# ----------------------------------------------------------------------


def gd_fixed_step_grid(problem, max_iter: int = 1200, record_every: int = 10) -> list[dict]:
    """Section 5.2: fixed step sizes around the stability limit 2 / L."""
    multiples = [2.1, 2.0, 1.9, 1.0, 0.5, 0.1, 0.01]
    specs = [
        {
            "method": "gradient_descent",
            "kwargs": {"max_iter": max_iter, "record_every": record_every,
                       "step_rule": {"kind": "fixed", "t": m / problem.L}},
            "label": f"GD (t = {m:g}/L)",
        }
        for m in multiples
    ]
    specs.append(
        {
            "method": "gradient_descent",
            "kwargs": {
                "max_iter": max_iter,
                "record_every": record_every,
                "step_rule": {"kind": "fixed", "multiple": 1.0, "reference": "L+mu"},
            },
            "label": "GD (t = 2/(L+mu))",
        }
    )
    return specs


def gd_backtracking_grid(problem, max_iter: int = 250, record_every: int = 5) -> list[dict]:
    """Section 5.2: Armijo parameters.

    This is the most expensive group in the sweep. With t0 = 1 and beta = 0.9
    the search has to shrink the step about ten times before it drops below
    the acceptance threshold near 1 / L, and every trial costs a full pass
    over the data. The iteration budget is therefore smaller than for the
    fixed-step grid; gradient descent needs only a few hundred iterations on
    this problem anyway.
    """
    del problem
    specs = []
    for alpha in (0.1, 0.3, 0.5):
        for beta in (0.5, 0.8, 0.9):
            specs.append(
                {
                    "method": "gradient_descent",
                    "kwargs": {
                        "max_iter": max_iter,
                        "record_every": record_every,
                        "step_rule": {
                            "kind": "backtracking",
                            "alpha": alpha,
                            "beta": beta,
                            "t0": 1.0,
                        },
                    },
                    "label": f"GD bt (alpha={alpha:g}, beta={beta:g})",
                }
            )
    return specs


def gd_backtracking_cost_grid(problem, max_iter: int = 100, record_every: int = 5) -> list[dict]:
    """The same Armijo grid, but stopped inside the productive phase.

    The quantity the assignment asks for is how many objective evaluations one
    backtracking iteration costs. Measuring it over a full run overstates it:
    once the objective is at machine precision the Armijo test can no longer be
    satisfied by any step, so the search burns its whole budget every
    iteration. On this problem the productive phase lasts about 115
    iterations, so the count is taken over the first 100.
    """
    specs = gd_backtracking_grid(problem, max_iter=max_iter, record_every=record_every)
    for spec in specs:
        spec["label"] = spec["label"].replace("GD bt", "GD bt cost")
    return specs


BATCH_SIZES = (1, 16, 64, 256, 1024)


def sgd_batch_size_grid(problem, epochs: int = 30, batch_sizes=BATCH_SIZES) -> list[dict]:
    """Section 5.2: batch sizes, each with a step size matched to its own
    smoothness constant.

    Using one common step size for every batch size is not a fair comparison:
    the stability limit of a stochastic step is set by the curvature of the
    samples in the batch, which is far larger than L for a single sample. Each
    batch size therefore gets eta0 = 1 / L_B, with L_B the mini-batch
    smoothness constant. The unfair version is available separately in
    `sgd_common_step_grid`, and the divergence it produces is itself a result.
    """
    specs = []
    for B in batch_sizes:
        eta0 = 1.0 / problem.minibatch_smoothness(B)
        specs.append(
            {
                "method": "sgd",
                "kwargs": {
                    "max_iter": epochs,
                    "batch_size": B,
                    "step_rule": {"kind": "schedule", "schedule": "constant", "eta0": eta0},
                    "seed": 0,
                },
                "label": f"SGD (B = {B}, eta = 1/L_B = {eta0:.3g})",
            }
        )
    return specs


def sgd_common_step_grid(problem, epochs: int = 30, batch_sizes=BATCH_SIZES) -> list[dict]:
    """The same batch sizes at one common step size eta0 = 0.1 / L.

    Small batches are expected to diverge here, which is the point: L governs
    the full-batch gradient, not a single stochastic step.
    """
    eta0 = 0.1 / problem.L
    return [
        {
            "method": "sgd",
            "kwargs": {
                "max_iter": epochs,
                "batch_size": B,
                "step_rule": {"kind": "schedule", "schedule": "constant", "eta0": eta0},
                "seed": 0,
            },
            "label": f"SGD (B = {B}, common eta = 0.1/L = {eta0:.3g})",
        }
        for B in batch_sizes
    ]


def sgd_schedule_grid(problem, epochs: int = 30, batch_size: int = 64) -> list[dict]:
    """Section 5.2: step size schedules at a common batch size."""
    eta0 = 1.0 / problem.minibatch_smoothness(batch_size)
    schedules = [
        ({"kind": "schedule", "schedule": "constant", "eta0": eta0}, "constant"),
        ({"kind": "schedule", "schedule": "inverse", "eta0": eta0, "gamma": 0.5}, "eta0/(1+0.5k)"),
        ({"kind": "schedule", "schedule": "sqrt", "eta0": eta0}, "eta0/sqrt(k)"),
        (
            {"kind": "schedule", "schedule": "staircase", "eta0": eta0, "period": 5},
            "halving every 5 epochs",
        ),
    ]
    return [
        {
            "method": "sgd",
            "kwargs": {"max_iter": epochs, "batch_size": batch_size, "step_rule": rule, "seed": 0},
            "label": f"SGD (B = {batch_size}, {name})",
        }
        for rule, name in schedules
    ]


def agd_grid(problem, max_iter: int = 400, record_every: int = 5) -> list[dict]:
    """Section 5.2: momentum rules, restart, and the backtracking variants."""
    step = {"kind": "fixed", "multiple": 1.0, "reference": "L"}
    specs = []
    for momentum in ("strongly_convex", "sequence"):
        for restart in (False, True):
            specs.append(
                {
                    "method": "accelerated_gradient",
                    "kwargs": {
                        "max_iter": max_iter,
                        "record_every": record_every,
                        "step_rule": step,
                        "momentum": momentum,
                        "restart": restart,
                    },
                    "label": f"AGD [{momentum}]" + (" + restart" if restart else ""),
                }
            )
    # Backtracking needs more care here than it does for plain gradient
    # descent. The search runs at the extrapolated point y, so it certifies a
    # decrease relative to f(y), not to f(w); the iteration is therefore not
    # monotone and an over-long step is not corrected by the next one. Armijo
    # with alpha = 1/2 is exactly the descent-lemma condition
    #     f(y - t g) <= f(y) - (t/2) ||g||^2
    # that the convergence proof of the accelerated method relies on, and it
    # is stable. Starting the search at t0 = 1 / L is the other safe option,
    # since the search only ever shrinks from there. The third entry below is
    # the configuration that fails, kept for the comparison.
    backtracking_variants = [
        ({"alpha": 0.5, "beta": 0.5, "t0": 1.0}, "bt alpha=0.5 (descent lemma), t0=1"),
        ({"alpha": 0.3, "beta": 0.8, "t0": 1.0 / problem.L}, "bt alpha=0.3, t0=1/L"),
        ({"alpha": 0.3, "beta": 0.8, "t0": 1.0}, "bt alpha=0.3, t0=1 (unstable)"),
    ]
    for params, name in backtracking_variants:
        specs.append(
            {
                "method": "accelerated_gradient",
                "kwargs": {
                    "max_iter": max_iter,
                    "record_every": record_every,
                    "step_rule": {"kind": "backtracking", **params},
                    "momentum": "strongly_convex",
                },
                "label": f"AGD [strongly_convex] {name}",
            }
        )
    return specs


def newton_grid(problem, max_iter: int = 10) -> list[dict]:
    """Section 5.2: full versus damped step, and the two Hessian policies."""
    del problem
    return [
        {
            "method": "newton",
            "kwargs": {"max_iter": max_iter, "refactor": True},
            "label": "Newton (t = 1, refactor each iteration)",
        },
        {
            "method": "newton",
            "kwargs": {"max_iter": max_iter, "refactor": False},
            "label": "Newton (t = 1, cached factor)",
        },
        {
            "method": "newton",
            "kwargs": {
                "max_iter": max_iter,
                "refactor": True,
                "step_rule": {"kind": "backtracking", "alpha": 0.3, "beta": 0.8, "t0": 1.0},
            },
            "label": "Newton (damped, backtracking)",
        },
        {
            "method": "newton_cg",
            "kwargs": {"max_iter": max_iter, "cg_tol": 1e-2},
            "label": "Newton-CG (cg_tol = 1e-2)",
        },
        {
            "method": "newton_cg",
            "kwargs": {"max_iter": max_iter, "cg_tol": 1e-6},
            "label": "Newton-CG (cg_tol = 1e-6)",
        },
    ]
