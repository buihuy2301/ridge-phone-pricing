"""First-order methods.

Every routine shares the signature

    method(problem, w0=None, max_iter=..., step_rule=None, tol=...,
           record_every=1, seed=None)

and returns an OptimizeResult. Step sizes are described by a `step_rule`
dictionary so that a whole parameter grid can be stored as plain data:

    {"kind": "fixed", "t": 0.01}
    {"kind": "fixed", "multiple": 1.0, "reference": "L"}
    {"kind": "fixed", "multiple": 1.0, "reference": "L+mu"}
    {"kind": "backtracking", "alpha": 0.3, "beta": 0.8, "t0": 1.0}
    {"kind": "schedule", "schedule": "inverse", "eta0": 0.1, "gamma": 0.01}
"""

from __future__ import annotations

import numpy as np

from .history import OptimizeResult, Recorder
from .line_search import backtracking_armijo, fixed_step, make_step_schedule


# ----------------------------------------------------------------------
# Step rule helpers
# ----------------------------------------------------------------------


def resolve_fixed_step(problem, step_rule: dict) -> float:
    """Turn a fixed step rule into a number."""
    if "t" in step_rule:
        return float(step_rule["t"])
    return fixed_step(
        problem,
        multiple=float(step_rule.get("multiple", 1.0)),
        reference=str(step_rule.get("reference", "L")),
    )


def _default_step_rule(step_rule: dict | None) -> dict:
    if step_rule is None:
        return {"kind": "fixed", "multiple": 1.0, "reference": "L"}
    rule = dict(step_rule)
    rule.setdefault("kind", "fixed")
    return rule


def _initial_step(problem, rule: dict) -> float:
    """Step size before the first iteration.

    For a fixed rule this is the step used throughout. For backtracking it is
    only the starting trial value, which the line search then shrinks.
    """
    kind = rule["kind"]
    if kind == "fixed":
        return resolve_fixed_step(problem, rule)
    if kind == "backtracking":
        return float(rule.get("t0", 1.0))
    raise ValueError(f"unsupported step rule kind: {kind}")


def _init_w(problem, w0: np.ndarray | None) -> np.ndarray:
    if w0 is None:
        return np.zeros(problem.d)
    return np.asarray(w0, dtype=np.float64).copy()


def _label(prefix: str, step_rule: dict) -> str:
    """Short human-readable description of a step rule, used in legends."""
    kind = step_rule.get("kind")
    if kind == "fixed":
        if "t" in step_rule:
            return f"{prefix} (t = {step_rule['t']:.3g})"
        ref = step_rule.get("reference", "L")
        mult = step_rule.get("multiple", 1.0)
        if ref == "L+mu":
            return f"{prefix} (t = {mult:g}*2/(L+mu))"
        return f"{prefix} (t = {mult:g}/L)"
    if kind == "backtracking":
        return (
            f"{prefix} (bt, alpha={step_rule.get('alpha', 0.3):g}, "
            f"beta={step_rule.get('beta', 0.8):g})"
        )
    if kind == "schedule":
        return (
            f"{prefix} ({step_rule.get('schedule', 'constant')}, "
            f"eta0={step_rule.get('eta0', 0.1):.3g})"
        )
    return prefix


# ----------------------------------------------------------------------
# Gradient descent
# ----------------------------------------------------------------------


def gradient_descent(
    problem,
    w0: np.ndarray | None = None,
    max_iter: int = 1000,
    step_rule: dict | None = None,
    tol: float = 1e-10,
    record_every: int = 1,
    seed: int | None = None,
    patience: int | None = 10,
) -> OptimizeResult:
    """Gradient descent with a fixed step or Armijo backtracking."""
    del seed  # deterministic method
    rule = _default_step_rule(step_rule)
    w = _init_w(problem, w0)

    rec = Recorder(problem, record_every, patience=patience)
    accesses = 0
    n_ls_evals = 0

    rec.start()
    rec.record(0, w, accesses, force=True)

    f_w, g = problem.f_and_grad(w)
    accesses += problem.n
    gnorm0 = float(np.linalg.norm(g))
    status = "max_iter"

    t = _initial_step(problem, rule)

    for k in range(1, max_iter + 1):
        if rule["kind"] == "backtracking":
            t, evals = backtracking_armijo(
                problem.f,
                w,
                f_w,
                g,
                -g,
                t0=float(rule.get("t0", 1.0)),
                alpha=float(rule.get("alpha", 0.3)),
                beta=float(rule.get("beta", 0.8)),
            )
            n_ls_evals += evals
            accesses += evals * problem.n

        w = w - t * g
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
    params["label"] = _label("GD", rule)
    params["line_search_evals"] = n_ls_evals
    return rec.finish(w, "gradient_descent", params, status)


# ----------------------------------------------------------------------
# Stochastic gradient descent
# ----------------------------------------------------------------------


def sgd(
    problem,
    w0: np.ndarray | None = None,
    max_iter: int = 100,
    step_rule: dict | None = None,
    tol: float = 0.0,
    record_every: int = 1,
    seed: int | None = 0,
    batch_size: int = 32,
) -> OptimizeResult:
    """Mini-batch stochastic gradient descent.

    `max_iter` counts epochs, not individual updates, so that runs with
    different batch sizes see the same amount of data. The history is
    recorded once every `record_every` epochs.
    """
    rule = _default_step_rule(step_rule)
    if rule.get("kind") != "schedule":
        rule = {
            "kind": "schedule",
            "schedule": "constant",
            "eta0": resolve_fixed_step(problem, rule),
        }

    schedule = make_step_schedule(
        kind=str(rule.get("schedule", "constant")),
        eta0=float(rule.get("eta0", 0.1)),
        gamma=float(rule.get("gamma", 1.0)),
        period=int(rule.get("period", 10)),
    )
    unit = str(rule.get("unit", "epoch"))

    n = problem.n
    batch_size = max(1, min(int(batch_size), n))
    n_batches = int(np.ceil(n / batch_size))

    rng = np.random.default_rng(seed)
    w = _init_w(problem, w0)

    rec = Recorder(problem, record_every)
    accesses = 0
    step_index = 0
    status = "max_iter"

    rec.start()
    rec.record(0, w, accesses, force=True)

    for epoch in range(1, max_iter + 1):
        order = rng.permutation(n)
        # A step size above the stability limit makes the iterate blow up
        # mid-epoch. The divergence is detected and reported at the end of the
        # epoch; the overflow warnings it produces on the way there carry no
        # extra information.
        with np.errstate(over="ignore", invalid="ignore"):
            for b in range(n_batches):
                idx = order[b * batch_size : (b + 1) * batch_size]
                if idx.size == 0:
                    continue
                eta = schedule(epoch - 1 if unit == "epoch" else step_index)
                g = problem.stochastic_grad(w, idx)
                w = w - eta * g
                accesses += idx.size
                step_index += 1

        rec.record(epoch, w, accesses)
        if rec.diverged:
            status = "diverged"
            break
        if tol > 0.0 and rec.gnorm_hist and rec.gnorm_hist[-1] <= tol * rec.gnorm_hist[0]:
            status = "converged"
            break

    params = dict(rule)
    params["batch_size"] = batch_size
    params["seed"] = seed
    params["label"] = _label(f"SGD (B={batch_size})", rule)
    params["n_updates"] = step_index
    return rec.finish(w, "sgd", params, status)


# ----------------------------------------------------------------------
# Accelerated gradient descent
# ----------------------------------------------------------------------


def accelerated_gradient(
    problem,
    w0: np.ndarray | None = None,
    max_iter: int = 1000,
    step_rule: dict | None = None,
    tol: float = 1e-10,
    record_every: int = 1,
    seed: int | None = None,
    momentum: str = "strongly_convex",
    restart: bool = False,
    patience: int | None = 10,
) -> OptimizeResult:
    """Nesterov's accelerated gradient method.

    momentum = "strongly_convex" uses the constant beta = (sqrt(k)-1)/(sqrt(k)+1)
    with k the condition number, which requires knowing mu.
    momentum = "sequence" uses beta_k = (k-1)/(k+2), which does not.

    With restart=True the momentum counter is reset whenever the gradient at
    the extrapolated point makes an obtuse angle with the last step, which
    removes the oscillations produced by the "sequence" rule.
    """
    del seed
    rule = _default_step_rule(step_rule)
    w = _init_w(problem, w0)
    w_prev = w.copy()

    if momentum not in ("strongly_convex", "sequence"):
        raise ValueError(f"unknown momentum rule: {momentum}")

    def strongly_convex_momentum(step: float) -> float:
        """Momentum consistent with the step size actually taken.

        The usual constant beta = (sqrt(kappa) - 1) / (sqrt(kappa) + 1) is the
        special case step = 1 / L of

            beta = (1 - sqrt(step * mu)) / (1 + sqrt(step * mu)).

        Writing it in terms of the step matters as soon as the step comes from
        a line search: Armijo only limits the step along the current gradient,
        so it can accept values far above 1 / L when the gradient points along
        a flat direction. Pairing such a step with a momentum computed for
        1 / L makes the iteration unstable, and the objective blows up.
        """
        product = max(step * problem.mu, 0.0)
        if product >= 1.0:
            return 0.0
        root = np.sqrt(product)
        return float((1.0 - root) / (1.0 + root))

    rec = Recorder(problem, record_every, patience=patience)
    accesses = 0
    n_ls_evals = 0
    n_restarts = 0
    inner = 0
    status = "max_iter"

    rec.start()
    rec.record(0, w, accesses, force=True)

    g0 = problem.grad(w)
    accesses += problem.n
    gnorm0 = float(np.linalg.norm(g0))

    t = _initial_step(problem, rule)

    for k in range(1, max_iter + 1):
        if momentum == "strongly_convex":
            # With backtracking, t still holds the step accepted at the
            # previous iteration, which is the best estimate available before
            # the current line search runs.
            beta = strongly_convex_momentum(t)
        else:
            beta = (inner - 1.0) / (inner + 2.0)
        beta = max(beta, 0.0)
        y = w + beta * (w - w_prev)

        f_y, g_y = problem.f_and_grad(y)
        accesses += problem.n

        if rule["kind"] == "backtracking":
            t, evals = backtracking_armijo(
                problem.f,
                y,
                f_y,
                g_y,
                -g_y,
                t0=float(rule.get("t0", 1.0)),
                alpha=float(rule.get("alpha", 0.3)),
                beta=float(rule.get("beta", 0.8)),
            )
            n_ls_evals += evals
            accesses += evals * problem.n

        w_next = y - t * g_y

        if restart and float(g_y @ (w_next - w)) > 0.0:
            inner = 0
            n_restarts += 1
        else:
            inner += 1

        w_prev, w = w, w_next

        # The stopping test needs the gradient at w, but the iteration only
        # computes the gradient at the extrapolated point y. Evaluating an
        # extra full gradient every iteration would double the cost of this
        # method relative to gradient descent, which reuses the gradient it
        # already needs for its update. The recorder already evaluates
        # ||grad f(w)|| at every recorded point, outside the timed region, so
        # the test reuses that value and is applied on the recording grid.
        recorded = rec.should_record(k)
        rec.record(k, w, accesses)
        if rec.diverged:
            status = "diverged"
            break
        if rec.stalled:
            status = "stalled"
            break
        if recorded and rec.gnorm_hist[-1] <= tol * gnorm0:
            status = "converged"
            break

    params = dict(rule)
    params["momentum"] = momentum
    params["restart"] = restart
    params["n_restarts"] = n_restarts
    params["line_search_evals"] = n_ls_evals
    suffix = "AGD" + ("+restart" if restart else "")
    params["label"] = _label(f"{suffix} [{momentum}]", rule)
    return rec.finish(w, "accelerated_gradient", params, status)


# ----------------------------------------------------------------------
# Heavy ball
# ----------------------------------------------------------------------


def heavy_ball(
    problem,
    w0: np.ndarray | None = None,
    max_iter: int = 1000,
    step_rule: dict | None = None,
    tol: float = 1e-10,
    record_every: int = 1,
    seed: int | None = None,
    beta: float | None = None,
    patience: int | None = 10,
) -> OptimizeResult:
    """Polyak's heavy ball method.

    With the optimal quadratic tuning the step size and momentum are

        t = 4 / (sqrt(L) + sqrt(mu))^2,
        beta = ((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^2.
    """
    del seed
    rule = _default_step_rule(step_rule)
    w = _init_w(problem, w0)
    w_prev = w.copy()

    kind = rule.get("kind")
    step_left_open = "t" not in rule and "multiple" not in rule
    if kind == "optimal" or (kind == "fixed" and step_left_open):
        t = 4.0 / (np.sqrt(problem.L) + np.sqrt(problem.mu)) ** 2
    else:
        t = resolve_fixed_step(problem, rule)

    if beta is None:
        sqrt_kappa = np.sqrt(problem.kappa)
        beta = float(((sqrt_kappa - 1.0) / (sqrt_kappa + 1.0)) ** 2)

    rec = Recorder(problem, record_every, patience=patience)
    accesses = 0
    status = "max_iter"

    rec.start()
    rec.record(0, w, accesses, force=True)

    g = problem.grad(w)
    accesses += problem.n
    gnorm0 = float(np.linalg.norm(g))

    for k in range(1, max_iter + 1):
        w_next = w - t * g + beta * (w - w_prev)
        w_prev, w = w, w_next
        g = problem.grad(w)
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
    params["beta"] = beta
    params["t"] = t
    params["label"] = f"Heavy ball (t = {t:.3g}, beta = {beta:.3g})"
    return rec.finish(w, "heavy_ball", params, status)


# ----------------------------------------------------------------------
# Adam
# ----------------------------------------------------------------------


def adam(
    problem,
    w0: np.ndarray | None = None,
    max_iter: int = 100,
    step_rule: dict | None = None,
    tol: float = 0.0,
    record_every: int = 1,
    seed: int | None = 0,
    batch_size: int | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> OptimizeResult:
    """Adam, in a full-batch or mini-batch variant.

    `max_iter` counts epochs when batch_size is set, and iterations otherwise,
    which keeps the comparison with SGD on the same footing.
    """
    del tol  # Adam has no gradient-norm stopping rule in these experiments
    rule = _default_step_rule(step_rule)
    eta = float(rule.get("eta0", rule.get("t", 1e-2)))

    n = problem.n
    stochastic = batch_size is not None
    if stochastic:
        batch_size = max(1, min(int(batch_size), n))
        n_batches = int(np.ceil(n / batch_size))
    else:
        n_batches = 1

    rng = np.random.default_rng(seed)
    w = _init_w(problem, w0)
    m = np.zeros_like(w)
    v = np.zeros_like(w)

    rec = Recorder(problem, record_every)
    accesses = 0
    step_index = 0
    status = "max_iter"

    rec.start()
    rec.record(0, w, accesses, force=True)

    for outer in range(1, max_iter + 1):
        order = rng.permutation(n) if stochastic else None
        for b in range(n_batches):
            if stochastic and order is not None and batch_size is not None:
                idx = order[b * batch_size : (b + 1) * batch_size]
                if idx.size == 0:
                    continue
                g = problem.stochastic_grad(w, idx)
                accesses += idx.size
            else:
                g = problem.grad(w)
                accesses += n

            step_index += 1
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)
            m_hat = m / (1.0 - beta1**step_index)
            v_hat = v / (1.0 - beta2**step_index)
            w = w - eta * m_hat / (np.sqrt(v_hat) + eps)

        rec.record(outer, w, accesses)
        if rec.diverged:
            status = "diverged"
            break

    params = dict(rule)
    params["eta"] = eta
    params["beta1"] = beta1
    params["beta2"] = beta2
    params["batch_size"] = batch_size
    params["seed"] = seed
    tag = f"Adam (eta = {eta:.3g}"
    tag += f", B={batch_size})" if stochastic else ", full batch)"
    params["label"] = tag
    return rec.finish(w, "adam", params, status)
