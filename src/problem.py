"""Optimization problems used in the project.

The main objective is the ridge-regularized least squares problem

    f(w) = (1 / (2n)) * ||X w - y||^2 + (lambda / 2) * ||w||^2

with gradient and Hessian

    grad f(w)  = (1/n) X^T (X w - y) + lambda w
    hess f(w)  = (1/n) X^T X + lambda I .

The intercept is handled by centering the data, so the problem has a single
variable block w and no unpenalized coordinate.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigvalsh


class RidgeProblem:
    """Ridge-regularized least squares objective.

    Function and gradient evaluations use the factored form X^T (X w - y),
    which costs O(n d) per call. The Gram matrix is formed only for the
    quantities that are computed once outside any timed loop: the Hessian,
    the closed-form solution, and the spectral constants L and mu.
    """

    name = "ridge"

    def __init__(self, X: np.ndarray, y: np.ndarray, lam: float) -> None:
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1-D with the same number of rows as X")
        if lam < 0.0:
            raise ValueError("lam must be non-negative")

        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.lam = float(lam)

        self.n, self.d = self.X.shape

        # Evaluation counters. Reset them before a run if they are reported.
        self.n_f_evals = 0
        self.n_grad_evals = 0

        # Optional per-iteration measurement, evaluated by the Recorder inside
        # the window where the clock is already paused, so its cost stays out
        # of the reported running time. Chapter 3 of the report uses it for the
        # test RMSE.
        self.monitor: Callable[[np.ndarray], float] | None = None

        # Lazily computed quantities.
        self._gram: np.ndarray | None = None
        self._Xty: np.ndarray | None = None
        self._eigvals: np.ndarray | None = None
        self._L_max_sample: float | None = None
        self._w_star: np.ndarray | None = None
        self._f_star: float | None = None

    # ------------------------------------------------------------------
    # Objective, gradient, Hessian
    # ------------------------------------------------------------------

    def f(self, w: np.ndarray) -> float:
        """Objective value at w."""
        self.n_f_evals += 1
        residual = self.X @ w - self.y
        return float(residual @ residual) / (2.0 * self.n) + 0.5 * self.lam * float(w @ w)

    def grad(self, w: np.ndarray) -> np.ndarray:
        """Full gradient at w."""
        self.n_grad_evals += 1
        residual = self.X @ w - self.y
        return self.X.T @ residual / self.n + self.lam * w

    def f_and_grad(self, w: np.ndarray) -> tuple[float, np.ndarray]:
        """Objective and gradient sharing a single residual computation."""
        self.n_f_evals += 1
        self.n_grad_evals += 1
        residual = self.X @ w - self.y
        value = float(residual @ residual) / (2.0 * self.n) + 0.5 * self.lam * float(w @ w)
        gradient = self.X.T @ residual / self.n + self.lam * w
        return value, gradient

    def hess(self, w: np.ndarray | None = None) -> np.ndarray:
        """Hessian. Constant for this problem, so w is ignored."""
        del w
        return self.gram + self.lam * np.eye(self.d)

    def hess_vec(self, w: np.ndarray | None, v: np.ndarray) -> np.ndarray:
        """Hessian-vector product without forming the Hessian.

        The Hessian does not depend on w either, but the argument is kept so
        that the signature matches the general case the callers assume.
        """
        del w
        return self.X.T @ (self.X @ v) / self.n + self.lam * v

    # ------------------------------------------------------------------
    # Stochastic gradient
    # ------------------------------------------------------------------

    def stochastic_grad(self, w: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Mini-batch gradient estimate on the sample indices in idx."""
        Xb = self.X[idx]
        residual = Xb @ w - self.y[idx]
        return Xb.T @ residual / idx.size + self.lam * w

    # ------------------------------------------------------------------
    # Precomputed quantities
    # ------------------------------------------------------------------

    @property
    def gram(self) -> np.ndarray:
        """(1/n) X^T X, computed once and cached."""
        if self._gram is None:
            self._gram = self.X.T @ self.X / self.n
        return self._gram

    @property
    def Xty(self) -> np.ndarray:
        """(1/n) X^T y, computed once and cached."""
        if self._Xty is None:
            self._Xty = self.X.T @ self.y / self.n
        return self._Xty

    @property
    def eigvals(self) -> np.ndarray:
        """Ascending eigenvalues of (1/n) X^T X."""
        if self._eigvals is None:
            self._eigvals = eigvalsh(self.gram)
        return self._eigvals

    @property
    def L(self) -> float:
        """Lipschitz constant of the gradient."""
        return float(self.eigvals[-1]) + self.lam

    @property
    def mu(self) -> float:
        """Strong convexity constant."""
        return float(self.eigvals[0]) + self.lam

    @property
    def kappa(self) -> float:
        """Condition number L / mu."""
        return self.L / self.mu

    @property
    def L_max_sample(self) -> float:
        """Largest Lipschitz constant among the individual sample losses.

        The loss of one sample is (1/2)(x_i^T w - y_i)^2 + (lam/2)||w||^2,
        whose gradient is Lipschitz with constant ||x_i||^2 + lam. A single
        stochastic step is stable up to 2 / this constant, not up to 2 / L,
        and the two differ by orders of magnitude for a wide design matrix.
        This is why a step size that is safe for full-batch gradient descent
        can make single-sample SGD diverge.
        """
        if self._L_max_sample is None:
            row_norms = np.einsum("ij,ij->i", self.X, self.X)
            self._L_max_sample = float(row_norms.max()) + self.lam
        return self._L_max_sample

    def minibatch_smoothness(self, batch_size: int) -> float:
        """Smoothness constant of a mini-batch gradient of the given size.

        Interpolates between the single-sample constant at batch size 1 and
        the full-batch constant L at batch size n:

            L_B = L + (L_max - L) / B .

        Used to give each batch size a step size on a comparable footing.
        """
        batch_size = max(1, min(int(batch_size), self.n))
        return self.L + (self.L_max_sample - self.L) / batch_size

    @property
    def w_star(self) -> np.ndarray:
        """Closed-form minimizer."""
        if self._w_star is None:
            factor = cho_factor(self.hess(), lower=True)
            self._w_star = cho_solve(factor, self.Xty)
        return self._w_star

    @property
    def f_star(self) -> float:
        """Optimal objective value.

        Computed without touching the evaluation counters, since it is not
        part of any algorithm's work.
        """
        cached = self._f_star
        if cached is not None:
            return cached
        w = self.w_star
        residual = self.X @ w - self.y
        value = float(residual @ residual) / (2.0 * self.n) + 0.5 * self.lam * float(w @ w)
        self._f_star = value
        return value

    def suboptimality(self, w: np.ndarray) -> float:
        """f(w) - f*, clipped at zero to stay plottable on a log scale."""
        return max(self.f(w) - self.f_star, 0.0)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset_counters(self) -> None:
        """Zero the objective and gradient evaluation counters."""
        self.n_f_evals = 0
        self.n_grad_evals = 0

    def summary(self) -> dict:
        """Constants that describe the problem instance."""
        return {
            "name": self.name,
            "n": self.n,
            "d": self.d,
            "lam": self.lam,
            "L": self.L,
            "mu": self.mu,
            "kappa": self.kappa,
            "f_star": self.f_star,
            "w_star_norm": float(np.linalg.norm(self.w_star)),
        }

    def __repr__(self) -> str:
        return f"RidgeProblem(n={self.n}, d={self.d}, lam={self.lam:.6g})"


def make_synthetic_ridge(
    n: int = 100,
    d: int = 5,
    lam: float = 1e-2,
    noise: float = 0.1,
    seed: int = 0,
) -> RidgeProblem:
    """Small synthetic instance used by the correctness tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    w_true = rng.standard_normal(d)
    y = X @ w_true + noise * rng.standard_normal(n)
    y = y - y.mean()
    return RidgeProblem(X, y, lam)
