"""Step size rules.

Two families are used in the experiments:

* fixed step sizes, expressed as a multiple of 1 / L or of 2 / (L + mu);
* backtracking line search with the Armijo sufficient decrease condition.

Every routine returns the number of extra objective evaluations it spent, so
that the cost of the line search can be reported next to the iteration count.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def backtracking_armijo(
    f: Callable[[np.ndarray], float],
    w: np.ndarray,
    f_w: float,
    grad: np.ndarray,
    direction: np.ndarray,
    t0: float = 1.0,
    alpha: float = 0.3,
    beta: float = 0.8,
    max_backtracks: int = 100,
) -> tuple[float, int]:
    """Armijo backtracking along an arbitrary descent direction.

    Shrinks t by the factor beta until

        f(w + t * direction) <= f(w) + alpha * t * <grad, direction>.

    With direction = -grad this reduces to the textbook condition
    f(w - t * grad) <= f(w) - alpha * t * ||grad||^2.

    Returns the accepted step size and the number of objective evaluations.
    """
    slope = float(grad @ direction)
    if slope >= 0.0:
        raise ValueError("direction is not a descent direction")

    t = float(t0)
    n_evals = 0
    for _ in range(max_backtracks):
        f_new = f(w + t * direction)
        n_evals += 1
        if f_new <= f_w + alpha * t * slope:
            return t, n_evals
        t *= beta

    # The loop exhausted its budget. Return the smallest step tried; the
    # caller sees the stall through the objective history.
    return t, n_evals


def exact_step_quadratic(
    hess_vec: Callable[[np.ndarray], np.ndarray],
    grad: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Exact minimizer of t -> f(w + t * direction) for a quadratic f.

    Available in closed form because the objective is quadratic; used only as
    a reference point against which the practical rules are compared.
    """
    Hd = hess_vec(direction)
    denominator = float(direction @ Hd)
    if denominator <= 0.0:
        raise ValueError("direction has non-positive curvature")
    return -float(grad @ direction) / denominator


def fixed_step(problem, multiple: float = 1.0, reference: str = "L") -> float:
    """Fixed step size expressed relative to a spectral constant.

    reference = "L"      -> t = multiple / L
    reference = "L+mu"   -> t = multiple * 2 / (L + mu)
    """
    if reference == "L":
        return multiple / problem.L
    if reference == "L+mu":
        return multiple * 2.0 / (problem.L + problem.mu)
    raise ValueError(f"unknown reference: {reference}")


def make_step_schedule(kind: str, eta0: float, gamma: float = 1.0, period: int = 10):
    """Step size schedules for the stochastic methods.

    kind = "constant"  -> eta_k = eta0
    kind = "inverse"   -> eta_k = eta0 / (1 + gamma * k)
    kind = "sqrt"      -> eta_k = eta0 / sqrt(k + 1)
    kind = "staircase" -> eta_k = eta0 * 2^(-floor(k / period))

    For the staircase rule, k is expected to be the epoch index and period is
    the number of epochs between halvings.
    """
    if kind == "constant":
        return lambda k: eta0
    if kind == "inverse":
        return lambda k: eta0 / (1.0 + gamma * k)
    if kind == "sqrt":
        return lambda k: eta0 / np.sqrt(k + 1.0)
    if kind == "staircase":
        return lambda k: eta0 * 2.0 ** (-(k // period))
    raise ValueError(f"unknown schedule: {kind}")
