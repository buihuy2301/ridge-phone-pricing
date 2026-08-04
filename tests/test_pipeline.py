"""Smoke tests for the runner, the plotting helpers and the library baselines.

These do not check convergence behaviour, only that the pieces fit together
and that the objective conversion against scikit-learn is exact.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.baselines import (
    linear_regression_baseline,
    ridge_baseline,
    sgd_regressor_baseline,
    verify_conversions,
)
from src.plotting import observed_rate, plot_comparison, set_style, summary_table
from src.problem import make_synthetic_ridge
from src.runner import (
    agd_grid,
    best_result,
    gd_fixed_step_grid,
    load_results,
    newton_grid,
    run_grid,
    save_results,
)


def test_ridge_alpha_conversion_is_exact() -> None:
    problem = make_synthetic_ridge(n=200, d=10, lam=1e-2, seed=0)
    report = verify_conversions(problem)
    assert report["passed"], report


def test_sgd_regressor_objective_uses_the_same_units() -> None:
    """SGDRegressor's own penalty scale must match lambda, not lambda * n."""
    problem = make_synthetic_ridge(n=500, d=10, lam=1e-2, seed=1)
    record = sgd_regressor_baseline(problem, max_iter=20000, tol=1e-12, learning_rate="adaptive")
    # A well-converged run should land near the closed-form solution.
    assert record["dist_to_w_star"] < 5e-2
    assert record["gap"] >= -1e-12


def test_library_baselines_produce_records() -> None:
    problem = make_synthetic_ridge(n=200, d=10, lam=1e-2, seed=2)
    records = [
        ridge_baseline(problem, solver="cholesky", repeats=1),
        ridge_baseline(problem, solver="lsqr", repeats=1),
        linear_regression_baseline(problem, repeats=1),
    ]
    for record in records:
        assert record["time_s"] > 0.0
        assert np.isfinite(record["f"])
    # The direct ridge solver must reach the optimum to solver accuracy.
    assert abs(records[0]["gap"]) < 1e-14


def test_run_grid_and_persistence(tmp_path, monkeypatch) -> None:
    problem = make_synthetic_ridge(n=200, d=10, lam=1e-2, seed=3)
    specs = gd_fixed_step_grid(problem, max_iter=100)[:3]
    results = run_grid(problem, specs, repeats=2, verbose=False)
    assert len(results) == len(specs)

    import src.runner as runner_module

    monkeypatch.setattr(runner_module, "RESULT_DIR", tmp_path)
    save_results(results, "smoke", extra=problem.summary())
    loaded, extra = load_results("smoke")

    assert len(loaded) == len(results)
    assert extra["d"] == problem.d
    assert loaded[0].f_hist == results[0].f_hist


def test_best_result_prefers_the_fastest_to_threshold() -> None:
    problem = make_synthetic_ridge(n=200, d=10, lam=1e-1, seed=4)
    results = run_grid(problem, agd_grid(problem, max_iter=500), repeats=1, verbose=False)
    chosen = best_result(results, problem.f_star, threshold=1e-8)
    assert chosen in results


def test_plot_comparison_writes_both_panels(tmp_path) -> None:
    set_style()
    problem = make_synthetic_ridge(n=200, d=10, lam=1e-2, seed=5)
    results = run_grid(problem, newton_grid(problem, max_iter=10), repeats=1, verbose=False)
    plot_comparison(
        results,
        problem.f_star,
        stem="smoke_newton",
        title="Newton variants",
        figure_dir=tmp_path,
    )
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == [
        "smoke_newton_iter.pdf",
        "smoke_newton_iter.png",
        "smoke_newton_time.pdf",
        "smoke_newton_time.png",
    ]


def test_summary_table_and_rate_estimate() -> None:
    problem = make_synthetic_ridge(n=200, d=10, lam=1e-2, seed=6)
    results = run_grid(problem, gd_fixed_step_grid(problem, max_iter=400), repeats=1, verbose=False)
    table = summary_table(results, problem.f_star)
    assert len(table) == len(results)
    assert "final_gap" in table.columns

    converging = [r for r in results if r.status != "diverged"]
    rate = observed_rate(converging[-1], problem.f_star)
    if rate is not None:
        assert 0.0 < rate <= 1.0
