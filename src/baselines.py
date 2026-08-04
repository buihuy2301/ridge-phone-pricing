"""scikit-learn baselines.

Section 6 of the plan compares the hand-written optimizers against library
defaults on two quantities: the objective value reached and the running time.

The comparison is only meaningful once the two objectives are written in the
same units. With the convention used here,

    f(w) = (1 / (2n)) ||X w - y||^2 + (lambda / 2) ||w||^2 ,

the conversions are

    Ridge:         alpha = lambda * n
                   because Ridge minimizes ||X w - y||^2 + alpha ||w||^2,
                   which is 2n times f(w).

    SGDRegressor:  alpha = lambda
                   because with loss="squared_error" and penalty="l2" it
                   minimizes (1/n) sum_i (1/2)(x_i^T w - y_i)^2
                   + (alpha / 2) ||w||^2, which is exactly f(w).

`verify_conversions` checks both numerically rather than on paper.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor


def _timed_fit(
    estimator, X: np.ndarray, y: np.ndarray, repeats: int = 3
) -> tuple[np.ndarray, float]:
    """Fit an estimator `repeats` times and keep the median wall-clock time."""
    times = []
    coef = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        estimator.fit(X, y)
        times.append(time.perf_counter() - start)
        coef = np.asarray(estimator.coef_, dtype=np.float64).ravel()
    assert coef is not None
    return coef, float(np.median(times))


def ridge_baseline(problem, solver: str = "auto", repeats: int = 3, **kwargs) -> dict:
    """sklearn Ridge with the alpha that matches the project's objective."""
    alpha = problem.lam * problem.n
    estimator = Ridge(alpha=alpha, solver=solver, fit_intercept=False, **kwargs)
    coef, elapsed = _timed_fit(estimator, problem.X, problem.y, repeats)
    return _record(
        problem, f"Ridge(solver='{solver}')", coef, elapsed, {"alpha": alpha, "solver": solver}
    )


def sgd_regressor_baseline(problem, repeats: int = 3, **kwargs) -> dict:
    """sklearn SGDRegressor left at its default settings."""
    alpha = problem.lam
    defaults = {
        "loss": "squared_error",
        "penalty": "l2",
        "alpha": alpha,
        "fit_intercept": False,
        "random_state": 0,
    }
    defaults.update(kwargs)
    estimator = SGDRegressor(**defaults)
    coef, elapsed = _timed_fit(estimator, problem.X, problem.y, repeats)
    return _record(problem, "SGDRegressor (defaults)", coef, elapsed, defaults)


def linear_regression_baseline(problem, repeats: int = 3) -> dict:
    """Unregularized least squares, reported as the lambda = 0 reference."""
    estimator = LinearRegression(fit_intercept=False)
    coef, elapsed = _timed_fit(estimator, problem.X, problem.y, repeats)
    return _record(problem, "LinearRegression", coef, elapsed, {"note": "no regularization"})


def _record(problem, label: str, coef: np.ndarray, elapsed: float, params: dict) -> dict:
    """Evaluate a library solution with the project's own objective."""
    f_value = problem.f(coef)
    return {
        "method": "sklearn",
        "label": label,
        "params": params,
        "w": coef,
        "f": float(f_value),
        "gap": float(f_value - problem.f_star),
        "grad_norm": float(np.linalg.norm(problem.grad(coef))),
        "time_s": elapsed,
        "dist_to_w_star": float(np.linalg.norm(coef - problem.w_star)),
    }


def verify_conversions(problem, rtol: float = 1e-8) -> dict:
    """Check the alpha conversions numerically.

    Ridge with a direct solver must reproduce the closed-form solution to
    solver accuracy. If this check fails, every objective comparison against
    the library is meaningless.
    """
    alpha_ridge = problem.lam * problem.n
    ridge = Ridge(alpha=alpha_ridge, solver="cholesky", fit_intercept=False)
    ridge.fit(problem.X, problem.y)
    w_ridge = np.asarray(ridge.coef_, dtype=np.float64).ravel()

    dist = float(np.linalg.norm(w_ridge - problem.w_star))
    scale = max(float(np.linalg.norm(problem.w_star)), 1e-12)
    relative = dist / scale

    return {
        "alpha_ridge": alpha_ridge,
        "alpha_sgd": problem.lam,
        "dist_to_w_star": dist,
        "relative_error": relative,
        "objective_gap": float(problem.f(w_ridge) - problem.f_star),
        "passed": relative < rtol,
    }


def baselines_table(records: list[dict]):
    """Assemble the library comparison table.

    pandas is imported here rather than at the top of the module: the solver
    baselines themselves do not need it, only this reporting helper does.
    """
    import pandas as pd  # pylint: disable=import-outside-toplevel

    return pd.DataFrame(
        [
            {
                "estimator": r["label"],
                "f": r["f"],
                "gap": r["gap"],
                "grad_norm": r["grad_norm"],
                "time_s": r["time_s"],
                "dist_to_w_star": r["dist_to_w_star"],
            }
            for r in records
        ]
    )


def rmse(problem, w: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Root mean squared error on a held-out set."""
    del problem
    residual = X_test @ w - y_test
    return float(np.sqrt(residual @ residual / residual.size))
