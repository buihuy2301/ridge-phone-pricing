"""Second-order and quasi-Newton methods.

The same signature and result structure as the first-order methods, so that
every run can go through the same runner and the same plotting code.

For the ridge objective the Hessian is constant, which makes Newton's method
converge in a single full step. Two variants are therefore provided:

* `refactor=True` recomputes the Cholesky factor at every iteration, which is
  what a general implementation has to do and what the timing comparison
  should use;
* `refactor=False` factors once and reuses it, which is the cheapest possible
  version for this particular objective.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .history import OptimizeResult, Recorder
from .line_search import backtracking_armijo


# ----------------------------------------------------------------------
# Newton
# ----------------------------------------------------------------------


def newton(
    problem,
    w0: np.ndarray | None = None,
    max_iter: int = 50,
    step_rule: dict | None = None,
    tol: float = 1e-10,
    record_every: int = 1,
    seed: int | None = None,
    refactor: bool = True,
    patience: int | None = 10,
) -> OptimizeResult:
    """Newton's method with a Cholesky solve of the Newton system.

    The system H p = -g is solved through `cho_factor` / `cho_solve` rather
    than by forming an explicit inverse, which is both cheaper and more
    stable.
    """
    del seed
    rule = dict(step_rule) if step_rule else {"kind": "fixed", "t": 1.0}
    rule.setdefault("kind", "fixed")

    w = w0.copy() if w0 is not None else np.zeros(problem.d)

    rec = Recorder(problem, record_every, patience=patience)
    accesses = 0
    n_ls_evals = 0
    n_factorizations = 0
    status = "max_iter"

    rec.start()
    rec.record(0, w, accesses, force=True)

    f_w, g = problem.f_and_grad(w)
    accesses += problem.n
    gnorm0 = float(np.linalg.norm(g))

    factor = None
    for k in range(1, max_iter + 1):
        if factor is None or refactor:
            factor = cho_factor(problem.hess(w), lower=True)
            n_factorizations += 1
            # Forming and factoring the Hessian touches the whole data set.
            accesses += problem.n

        p = cho_solve(factor, -g)

        if rule["kind"] == "backtracking":
            t, evals = backtracking_armijo(
                problem.f,
                w,
                f_w,
                g,
                p,
                t0=float(rule.get("t0", 1.0)),
                alpha=float(rule.get("alpha", 0.3)),
                beta=float(rule.get("beta", 0.8)),
            )
            n_ls_evals += evals
            accesses += evals * problem.n
        else:
            t = float(rule.get("t", 1.0))

        w = w + t * p
        f_w, g = problem.f_and_grad(w)
        accesses += problem.n

        rec.record(k, w, accesses)
        if rec.diverged:
            status = "diverged"
            break
        if rec.stalled:
            status = "stalled"
            break
        if np.linalg.norm(g) <= tol * gnorm0:
            status = "converged"
            rec.record(k, w, accesses, force=True)
            break

    params = dict(rule)
    params["refactor"] = refactor
    params["n_factorizations"] = n_factorizations
    params["line_search_evals"] = n_ls_evals
    damped = rule["kind"] == "backtracking"
    params["label"] = "Newton (damped)" if damped else "Newton (t = 1)"
    if not refactor:
        params["label"] += ", cached factor"
    return rec.finish(w, "newton", params, status)


# ----------------------------------------------------------------------
# Inexact Newton
# ----------------------------------------------------------------------


def _conjugate_gradient(hess_vec, b: np.ndarray, tol: float, max_iter: int) -> tuple[np.ndarray, int]:
    """Solve H p = b approximately, using Hessian-vector products only."""
    p = np.zeros_like(b)
    r = b.copy()
    d = r.copy()
    rs = float(r @ r)
    target = tol * tol * rs
    n_products = 0

    for _ in range(max_iter):
        if rs <= target:
            break
        Hd = hess_vec(d)
        n_products += 1
        curvature = float(d @ Hd)
        if curvature <= 0.0:
            break
        alpha = rs / curvature
        p += alpha * d
        r -= alpha * Hd
        rs_new = float(r @ r)
        d = r + (rs_new / rs) * d
        rs = rs_new

    return p, n_products


def newton_cg(
    problem,
    w0: np.ndarray | None = None,
    max_iter: int = 50,
    step_rule: dict | None = None,
    tol: float = 1e-10,
    record_every: int = 1,
    seed: int | None = None,
    cg_tol: float = 1e-2,
    cg_max_iter: int | None = None,
    patience: int | None = 10,
) -> OptimizeResult:
    """Newton's method with the linear system solved by conjugate gradients.

    Avoids the O(d^3) factorization: each CG iteration costs one
    Hessian-vector product, which is O(n d) for this objective.
    """
    del seed
    rule = dict(step_rule) if step_rule else {"kind": "fixed", "t": 1.0}
    rule.setdefault("kind", "fixed")
    cg_max_iter = cg_max_iter or problem.d

    w = w0.copy() if w0 is not None else np.zeros(problem.d)

    rec = Recorder(problem, record_every, patience=patience)
    accesses = 0
    n_ls_evals = 0
    n_cg_products = 0
    status = "max_iter"

    rec.start()
    rec.record(0, w, accesses, force=True)

    f_w, g = problem.f_and_grad(w)
    accesses += problem.n
    gnorm0 = float(np.linalg.norm(g))

    for k in range(1, max_iter + 1):
        p, products = _conjugate_gradient(
            lambda v: problem.hess_vec(w, v), -g, cg_tol, cg_max_iter
        )
        n_cg_products += products
        accesses += products * problem.n

        if float(g @ p) >= 0.0:
            # CG returned a non-descent direction; fall back on the gradient.
            p = -g

        if rule["kind"] == "backtracking":
            t, evals = backtracking_armijo(
                problem.f,
                w,
                f_w,
                g,
                p,
                t0=float(rule.get("t0", 1.0)),
                alpha=float(rule.get("alpha", 0.3)),
                beta=float(rule.get("beta", 0.8)),
            )
            n_ls_evals += evals
            accesses += evals * problem.n
        else:
            t = float(rule.get("t", 1.0))

        w = w + t * p
        f_w, g = problem.f_and_grad(w)
        accesses += problem.n

        rec.record(k, w, accesses)
        if rec.diverged:
            status = "diverged"
            break
        if rec.stalled:
            status = "stalled"
            break
        if np.linalg.norm(g) <= tol * gnorm0:
            status = "converged"
            rec.record(k, w, accesses, force=True)
            break

    params = dict(rule)
    params["cg_tol"] = cg_tol
    params["n_cg_products"] = n_cg_products
    params["line_search_evals"] = n_ls_evals
    params["label"] = f"Newton-CG (cg_tol = {cg_tol:g})"
    return rec.finish(w, "newton_cg", params, status)


# ----------------------------------------------------------------------
# L-BFGS
# ----------------------------------------------------------------------


def _two_loop_recursion(g: np.ndarray, s_list, y_list, rho_list) -> np.ndarray:
    """Standard L-BFGS two-loop recursion returning the search direction."""
    q = g.copy()
    alphas = []
    for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
        alpha = rho * float(s @ q)
        alphas.append(alpha)
        q -= alpha * y

    if s_list:
        s_last, y_last = s_list[-1], y_list[-1]
        gamma = float(s_last @ y_last) / float(y_last @ y_last)
        q *= gamma

    for (s, y, rho), alpha in zip(zip(s_list, y_list, rho_list), reversed(alphas)):
        beta = rho * float(y @ q)
        q += (alpha - beta) * s

    return -q


def lbfgs(
    problem,
    w0: np.ndarray | None = None,
    max_iter: int = 500,
    step_rule: dict | None = None,
    tol: float = 1e-10,
    record_every: int = 1,
    seed: int | None = None,
    memory: int = 10,
    patience: int | None = 10,
) -> OptimizeResult:
    """Limited-memory BFGS with Armijo backtracking.

    Costs O(memory * d) per iteration on top of one gradient evaluation, so it
    sits between the first-order methods and Newton in both cost and speed.
    """
    del seed
    rule = dict(step_rule) if step_rule else {"kind": "backtracking", "alpha": 1e-4, "beta": 0.5, "t0": 1.0}
    rule.setdefault("kind", "backtracking")

    w = w0.copy() if w0 is not None else np.zeros(problem.d)

    s_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    rho_list: list[float] = []

    rec = Recorder(problem, record_every, patience=patience)
    accesses = 0
    n_ls_evals = 0
    status = "max_iter"

    rec.start()
    rec.record(0, w, accesses, force=True)

    f_w, g = problem.f_and_grad(w)
    accesses += problem.n
    gnorm0 = float(np.linalg.norm(g))

    for k in range(1, max_iter + 1):
        p = _two_loop_recursion(g, s_list, y_list, rho_list)
        if float(g @ p) >= 0.0:
            p = -g
            s_list.clear()
            y_list.clear()
            rho_list.clear()

        if rule["kind"] == "backtracking":
            t, evals = backtracking_armijo(
                problem.f,
                w,
                f_w,
                g,
                p,
                t0=float(rule.get("t0", 1.0)),
                alpha=float(rule.get("alpha", 1e-4)),
                beta=float(rule.get("beta", 0.5)),
            )
            n_ls_evals += evals
            accesses += evals * problem.n
        else:
            t = float(rule.get("t", 1.0))

        w_new = w + t * p
        f_new, g_new = problem.f_and_grad(w_new)
        accesses += problem.n

        s = w_new - w
        y = g_new - g
        sy = float(s @ y)
        if sy > 1e-12:
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / sy)
            if len(s_list) > memory:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)

        w, f_w, g = w_new, f_new, g_new

        rec.record(k, w, accesses)
        if rec.diverged:
            status = "diverged"
            break
        if rec.stalled:
            status = "stalled"
            break
        if np.linalg.norm(g) <= tol * gnorm0:
            status = "converged"
            rec.record(k, w, accesses, force=True)
            break

    params = dict(rule)
    params["memory"] = memory
    params["line_search_evals"] = n_ls_evals
    params["label"] = f"L-BFGS (m = {memory})"
    return rec.finish(w, "lbfgs", params, status)
