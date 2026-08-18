"""Correctness checks required before running the full experiment grid.

Covers the four checks listed in section 11 of the plan:

1. analytic gradient against a central finite difference;
2. analytic Hessian against a central finite difference of the gradient;
3. the closed-form solution is a stationary point;
4. every optimizer reaches the same minimizer on a small synthetic instance.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.first_order import accelerated_gradient, adam, gradient_descent, heavy_ball, sgd
from src.problem import RidgeProblem, make_synthetic_ridge
from src.second_order import lbfgs, newton, newton_cg

# The pytest fixtures below are named after the argument they fill in, so every
# test that requests one shadows the fixture function.
# pylint: disable=redefined-outer-name

EPS = 1e-6


@pytest.fixture
def problem() -> RidgeProblem:
    """Small well-conditioned instance, cheap enough to solve many times."""
    return make_synthetic_ridge(n=100, d=5, lam=1e-2, seed=0)


# ----------------------------------------------------------------------
# Derivatives
# ----------------------------------------------------------------------


def test_gradient_matches_finite_differences(problem: RidgeProblem) -> None:
    """Central differences of f must reproduce grad f to a relative 1e-6."""
    rng = np.random.default_rng(1)
    w = rng.standard_normal(problem.d)
    analytic = problem.grad(w)

    numeric = np.empty_like(analytic)
    for i in range(problem.d):
        e = np.zeros(problem.d)
        e[i] = EPS
        numeric[i] = (problem.f(w + e) - problem.f(w - e)) / (2.0 * EPS)

    rel_error = np.linalg.norm(analytic - numeric) / np.linalg.norm(analytic)
    assert rel_error < 1e-6


def test_hessian_matches_finite_differences(problem: RidgeProblem) -> None:
    """Central differences of the gradient must reproduce the Hessian."""
    rng = np.random.default_rng(2)
    w = rng.standard_normal(problem.d)
    analytic = problem.hess(w)

    numeric = np.empty_like(analytic)
    for i in range(problem.d):
        e = np.zeros(problem.d)
        e[i] = EPS
        numeric[:, i] = (problem.grad(w + e) - problem.grad(w - e)) / (2.0 * EPS)

    rel_error = np.linalg.norm(analytic - numeric) / np.linalg.norm(analytic)
    assert rel_error < 1e-6


def test_hessian_vector_product(problem: RidgeProblem) -> None:
    """`hess_vec` must agree with forming the Hessian and multiplying by it."""
    rng = np.random.default_rng(3)
    w = rng.standard_normal(problem.d)
    v = rng.standard_normal(problem.d)
    assert np.allclose(problem.hess_vec(w, v), problem.hess(w) @ v)


def test_stochastic_gradient_is_unbiased_on_full_batch(problem: RidgeProblem) -> None:
    """On the whole index set the mini-batch estimate is the full gradient."""
    rng = np.random.default_rng(4)
    w = rng.standard_normal(problem.d)
    full_idx = np.arange(problem.n)
    assert np.allclose(problem.stochastic_grad(w, full_idx), problem.grad(w))


# ----------------------------------------------------------------------
# Closed form
# ----------------------------------------------------------------------


def test_closed_form_is_stationary(problem: RidgeProblem) -> None:
    """The gradient vanishes at w*, which is what makes f* a valid reference."""
    assert np.linalg.norm(problem.grad(problem.w_star)) < 1e-12


def test_spectral_constants_are_consistent(problem: RidgeProblem) -> None:
    """L and mu must be the extreme eigenvalues of the Hessian, and kappa >= 1."""
    hessian = problem.hess()
    eigvals = np.linalg.eigvalsh(hessian)
    assert np.isclose(problem.L, eigvals[-1])
    assert np.isclose(problem.mu, eigvals[0])
    assert problem.kappa >= 1.0


def test_f_star_is_the_minimum(problem: RidgeProblem) -> None:
    """No point in a small neighbourhood of w* has an objective below f*."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        w = problem.w_star + 1e-2 * rng.standard_normal(problem.d)
        assert problem.f(w) >= problem.f_star


# ----------------------------------------------------------------------
# Optimizers
# ----------------------------------------------------------------------


DETERMINISTIC_RUNS = {
    "gd_fixed": (
        gradient_descent,
        {"max_iter": 5000, "step_rule": {"kind": "fixed", "multiple": 1.0}},
    ),
    "gd_optimal": (
        gradient_descent,
        {"max_iter": 5000, "step_rule": {"kind": "fixed", "multiple": 1.0, "reference": "L+mu"}},
    ),
    "gd_backtracking": (
        gradient_descent,
        {
            "max_iter": 5000,
            "step_rule": {"kind": "backtracking", "alpha": 0.3, "beta": 0.8, "t0": 1.0},
        },
    ),
    "agd_strongly_convex": (
        accelerated_gradient,
        {
            "max_iter": 5000,
            "step_rule": {"kind": "fixed", "multiple": 1.0},
            "momentum": "strongly_convex",
        },
    ),
    "agd_sequence_restart": (
        accelerated_gradient,
        {
            "max_iter": 5000,
            "step_rule": {"kind": "fixed", "multiple": 1.0},
            "momentum": "sequence",
            "restart": True,
        },
    ),
    "heavy_ball": (heavy_ball, {"max_iter": 5000, "step_rule": {"kind": "optimal"}}),
    "newton_full": (newton, {"max_iter": 20}),
    "newton_damped": (
        newton,
        {
            "max_iter": 20,
            "step_rule": {"kind": "backtracking", "alpha": 0.3, "beta": 0.8, "t0": 1.0},
        },
    ),
    "newton_cg": (newton_cg, {"max_iter": 50, "cg_tol": 1e-8}),
    "lbfgs": (lbfgs, {"max_iter": 500}),
}


@pytest.mark.parametrize("name", sorted(DETERMINISTIC_RUNS))
def test_deterministic_optimizers_reach_the_minimizer(problem: RidgeProblem, name: str) -> None:
    """Every deterministic method lands on the same w*, to within 1e-6."""
    method, kwargs = DETERMINISTIC_RUNS[name]
    result = method(problem, **kwargs)

    assert result.status == "converged", f"{name} did not converge: {result.status}"
    assert np.linalg.norm(result.w_final - problem.w_star) < 1e-6
    assert result.f_final - problem.f_star < 1e-12


def test_newton_solves_a_quadratic_in_one_step(problem: RidgeProblem) -> None:
    """The Newton step from any point lands exactly on w* for a quadratic."""
    result = newton(problem, max_iter=1, tol=0.0)
    assert np.linalg.norm(result.w_final - problem.w_star) < 1e-10


def test_accelerated_gradient_momentum_follows_the_step_size() -> None:
    """Momentum must be derived from the step actually taken.

    The constant beta = (sqrt(kappa)-1)/(sqrt(kappa)+1) is the special case
    step = 1/L. A line search can accept a much larger step, and pairing that
    step with the 1/L momentum makes the iteration blow up.
    """
    ill = make_synthetic_ridge(n=400, d=30, lam=1e-3, seed=11)
    result = accelerated_gradient(
        ill,
        max_iter=400,
        step_rule={"kind": "backtracking", "alpha": 0.5, "beta": 0.5, "t0": 1.0},
        momentum="strongly_convex",
    )
    assert result.status != "diverged"
    assert result.f_final - ill.f_star < 1e-8


def test_gradient_descent_diverges_above_the_stability_limit(problem: RidgeProblem) -> None:
    """t > 2 / L must break the method, which is the point of the experiment."""
    result = gradient_descent(
        problem, max_iter=500, step_rule={"kind": "fixed", "t": 2.1 / problem.L}
    )
    assert result.status == "diverged"


def test_sgd_approaches_the_minimizer(problem: RidgeProblem) -> None:
    """A decreasing schedule drives SGD to the optimum, but only slowly.

    The tolerance is loose on purpose: after 300 epochs the remaining gap is
    of order 1e-6, which is the accuracy a decaying step size buys here.
    """
    result = sgd(
        problem,
        max_iter=300,
        batch_size=16,
        step_rule={"kind": "schedule", "schedule": "inverse", "eta0": 0.5, "gamma": 0.05},
        seed=0,
    )
    assert result.f_final - problem.f_star < 1e-4


def test_sgd_with_constant_step_stalls_above_the_optimum(problem: RidgeProblem) -> None:
    """A constant step size leaves SGD oscillating in a neighbourhood of w*."""
    result = sgd(
        problem,
        max_iter=300,
        batch_size=1,
        step_rule={"kind": "schedule", "schedule": "constant", "eta0": 0.2},
        seed=0,
    )
    gap = result.f_final - problem.f_star
    assert gap > 1e-9
    assert gap < 1e-1


def test_adam_reduces_the_objective(problem: RidgeProblem) -> None:
    """Adam closes the gap to below 1e-6 on this small instance."""
    result = adam(problem, max_iter=2000, step_rule={"eta0": 1e-2}, seed=0)
    assert result.f_final < result.f_hist[0]
    assert result.f_final - problem.f_star < 1e-6


# ----------------------------------------------------------------------
# Bookkeeping
# ----------------------------------------------------------------------


def test_history_lengths_agree(problem: RidgeProblem) -> None:
    """The five history arrays are filled at the same recorded points."""
    result = gradient_descent(problem, max_iter=50, record_every=5)
    lengths = {
        len(result.iter_hist),
        len(result.f_hist),
        len(result.gnorm_hist),
        len(result.time_hist),
        len(result.access_hist),
    }
    assert len(lengths) == 1


def test_recorded_time_is_monotone(problem: RidgeProblem) -> None:
    """Pausing the clock for the recording must not make it run backwards."""
    result = gradient_descent(problem, max_iter=100)
    times = np.asarray(result.time_hist)
    assert np.all(np.diff(times) >= 0.0)


def test_monitoring_does_not_inflate_evaluation_counts(problem: RidgeProblem) -> None:
    """Recording every iteration must cost the same as recording rarely.

    The stall test is switched off here: its budget is counted in recorded
    points, so it would fire at different iterations for the two cadences and
    the runs would no longer be comparable.
    """
    dense = gradient_descent(problem, max_iter=100, record_every=1, tol=0.0, patience=None)
    sparse = gradient_descent(problem, max_iter=100, record_every=50, tol=0.0, patience=None)
    assert dense.n_f_evals == sparse.n_f_evals
    assert dense.n_grad_evals == sparse.n_grad_evals


def test_result_is_json_serializable(problem: RidgeProblem) -> None:
    """`to_dict` must survive json.dumps, since results are cached on disk."""
    result = gradient_descent(problem, max_iter=10)
    json.dumps(result.to_dict())


def test_recorder_collects_monitor_values(problem: RidgeProblem) -> None:
    """An optional monitor runs once per recorded point.

    Chapter 3 of the report plots the test RMSE against the optimization gap,
    which needs a per-iteration quantity that the problem itself does not know
    about. The recorder evaluates it inside the window where the clock is
    already paused, so the reported running time is unaffected.
    """
    problem.monitor = lambda w: float(np.linalg.norm(w))

    result = gradient_descent(
        problem,
        w0=np.zeros(problem.d),
        max_iter=10,
        record_every=1,
        patience=None,
    )

    assert len(result.metric_hist) == len(result.f_hist)
    assert result.metric_hist[0] == 0.0
    assert result.metric_hist[-1] > 0.0


def test_monitor_is_optional(problem: RidgeProblem) -> None:
    """Without a monitor the history is empty rather than absent."""
    result = gradient_descent(problem, max_iter=10)
    assert result.metric_hist == []
