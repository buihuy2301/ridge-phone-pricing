"""Shared bookkeeping for every optimizer in this project.

All algorithms return the same structure, which keeps the plotting code and
the result files uniform. The recorder is also responsible for keeping the
logging cost out of the reported running time: the clock is paused while the
objective value and the gradient norm are evaluated, and the problem's own
evaluation counters are restored afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptimizeResult:
    """Outcome of a single optimizer run."""

    method: str
    params: dict
    w_final: np.ndarray
    iter_hist: list[int] = field(default_factory=list)
    f_hist: list[float] = field(default_factory=list)
    gnorm_hist: list[float] = field(default_factory=list)
    time_hist: list[float] = field(default_factory=list)
    access_hist: list[int] = field(default_factory=list)
    status: str = "max_iter"
    n_f_evals: int = 0
    n_grad_evals: int = 0
    n_iter: int = 0
    total_time: float = 0.0

    @property
    def f_final(self) -> float:
        """Last recorded objective value."""
        return self.f_hist[-1]

    def suboptimality(self, f_star: float, floor: float = 1e-18) -> np.ndarray:
        """f(w_k) - f*, floored so the values stay usable on a log axis."""
        gap = np.asarray(self.f_hist, dtype=np.float64) - f_star
        return np.maximum(gap, floor)

    def epochs(self, n_samples: int) -> np.ndarray:
        """Cumulative number of passes over the data."""
        return np.asarray(self.access_hist, dtype=np.float64) / n_samples

    def iters_to_reach(self, f_star: float, threshold: float) -> int | None:
        """First recorded iteration with f(w_k) - f* below the threshold."""
        for k, value in zip(self.iter_hist, self.f_hist):
            if value - f_star < threshold:
                return k
        return None

    def time_to_reach(self, f_star: float, threshold: float) -> float | None:
        """Elapsed time at the first recorded point below the threshold."""
        for t, value in zip(self.time_hist, self.f_hist):
            if value - f_star < threshold:
                return t
        return None

    def to_dict(self) -> dict:
        """JSON-serializable view, used by the runner."""
        return {
            "method": self.method,
            "params": self.params,
            "status": self.status,
            "n_iter": self.n_iter,
            "n_f_evals": self.n_f_evals,
            "n_grad_evals": self.n_grad_evals,
            "total_time": self.total_time,
            "iter_hist": self.iter_hist,
            "f_hist": self.f_hist,
            "gnorm_hist": self.gnorm_hist,
            "time_hist": self.time_hist,
            "access_hist": self.access_hist,
            "w_final_norm": float(np.linalg.norm(self.w_final)),
            "w_final": self.w_final.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OptimizeResult":
        """Rebuild a result from its JSON form, so figures can be redrawn
        without re-running the experiments."""
        return cls(
            method=payload["method"],
            params=payload["params"],
            w_final=np.asarray(payload.get("w_final", []), dtype=np.float64),
            iter_hist=list(payload["iter_hist"]),
            f_hist=list(payload["f_hist"]),
            gnorm_hist=list(payload["gnorm_hist"]),
            time_hist=list(payload["time_hist"]),
            access_hist=list(payload["access_hist"]),
            status=payload["status"],
            n_f_evals=payload["n_f_evals"],
            n_grad_evals=payload["n_grad_evals"],
            n_iter=payload["n_iter"],
            total_time=payload["total_time"],
        )


class Recorder:
    """Collects the convergence history while excluding its own cost.

    Usage:

        rec = Recorder(problem, record_every=1)
        rec.start()
        rec.record(0, w, accesses=0)
        for k in range(max_iter):
            ...                       # algorithm work, clock running
            rec.record(k + 1, w, accesses=cumulative_accesses)
        result = rec.finish(w, method="gd", params={...})
    """

    DIVERGENCE_FACTOR = 1e12
    STALL_REL_TOL = 1e-13

    def __init__(self, problem, record_every: int = 1, patience: int | None = None) -> None:
        """
        `patience` stops a run once the objective has not improved over that
        many consecutive recorded points. Without it, a deterministic method
        that has already reached machine precision keeps iterating to its
        budget, and on this problem that dead time is the bulk of the measured
        running time: a backtracking run reached 1e-15 at iteration 115 and
        then spent 98% of its 72 seconds going nowhere, with the line search
        failing every trial because the Armijo test cannot be met once the
        decrease is below the resolution of f. Leave it at None for the
        stochastic methods, whose objective is not monotone and whose plateau
        is itself the result being shown.
        """
        self.problem = problem
        self.record_every = max(1, int(record_every))
        self.patience = patience

        # Counters are per run, so a problem instance can be reused across the
        # whole parameter grid without the counts accumulating.
        self.problem.reset_counters()

        self.iter_hist: list[int] = []
        self.f_hist: list[float] = []
        self.gnorm_hist: list[float] = []
        self.time_hist: list[float] = []
        self.access_hist: list[int] = []

        self._elapsed = 0.0
        self._clock_start: float | None = None
        self._f_first: float | None = None
        self._best_f: float | None = None
        self._records_without_progress = 0
        self.diverged = False
        self.stalled = False

    # ------------------------------------------------------------------
    # Clock control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the clock, just before the first iteration."""
        self._clock_start = time.perf_counter()

    def _pause(self) -> None:
        if self._clock_start is not None:
            self._elapsed += time.perf_counter() - self._clock_start
            self._clock_start = None

    def _resume(self) -> None:
        self._clock_start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Running time so far, excluding the paused recording intervals."""
        if self._clock_start is None:
            return self._elapsed
        return self._elapsed + (time.perf_counter() - self._clock_start)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def should_record(self, k: int) -> bool:
        """True when iteration k falls on the recording grid."""
        return k % self.record_every == 0

    def record(self, k: int, w: np.ndarray, accesses: int, force: bool = False) -> None:
        """Store one point of the history. Does nothing off the record grid."""
        if not force and not self.should_record(k):
            return

        self._pause()

        # Monitoring must not pollute the algorithm's own evaluation counts.
        saved_f = self.problem.n_f_evals
        saved_g = self.problem.n_grad_evals
        f_value = self.problem.f(w)
        gnorm = float(np.linalg.norm(self.problem.grad(w)))
        self.problem.n_f_evals = saved_f
        self.problem.n_grad_evals = saved_g

        self.iter_hist.append(int(k))
        self.f_hist.append(float(f_value))
        self.gnorm_hist.append(gnorm)
        self.time_hist.append(self._elapsed)
        self.access_hist.append(int(accesses))

        f_first = self._f_first
        if f_first is None:
            f_first = float(f_value)
            self._f_first = f_first
        if not np.isfinite(f_value) or f_value > self.DIVERGENCE_FACTOR * abs(f_first):
            self.diverged = True

        if self.patience is not None and f_value <= f_first:
            best = self._best_f
            if best is None or f_value < best - abs(best) * self.STALL_REL_TOL:
                self._best_f = float(f_value)
                self._records_without_progress = 0
            else:
                self._records_without_progress += 1
                if self._records_without_progress >= self.patience:
                    self.stalled = True

        self._resume()

    def finish(self, w: np.ndarray, method: str, params: dict, status: str) -> OptimizeResult:
        """Stop the clock and package the history into an OptimizeResult."""
        self._pause()
        return OptimizeResult(
            method=method,
            params=params,
            w_final=np.asarray(w, dtype=np.float64).copy(),
            iter_hist=self.iter_hist,
            f_hist=self.f_hist,
            gnorm_hist=self.gnorm_hist,
            time_hist=self.time_hist,
            access_hist=self.access_hist,
            status="diverged" if self.diverged else status,
            n_f_evals=self.problem.n_f_evals,
            n_grad_evals=self.problem.n_grad_evals,
            n_iter=self.iter_hist[-1] if self.iter_hist else 0,
            total_time=self._elapsed,
        )


def warm_up(problem, repeats: int = 3) -> None:
    """Touch the BLAS paths once so the first timed run is not penalized."""
    w = np.zeros(problem.d)
    for _ in range(repeats):
        problem.f_and_grad(w)
    problem.reset_counters()
