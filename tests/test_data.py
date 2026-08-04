"""Tests for the data preparation pipeline.

The real Kaggle file is not in the repository, so the pipeline is exercised on
a synthetic table with the same shape of problem: numeric columns written with
units attached, high-cardinality categorical columns, missing values, and a
positive skewed target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    build_design_matrix,
    coerce_numeric_like,
    extract_number,
    group_rare_levels,
    guess_target,
    load_problem,
    ridge_solution,
    save_processed,
    select_lambda,
)
from src.problem import RidgeProblem

# The pytest fixtures below are named after the argument they fill in, so every
# test that requests one shadows the fixture function.
# pylint: disable=redefined-outer-name


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Synthetic table with the same awkward features as the Kaggle file."""
    rng = np.random.default_rng(0)
    n = 800
    brands = rng.choice(["Apple", "Samsung", "Xiaomi", "Oppo", "Vivo"], size=n)
    models = rng.choice([f"model_{i}" for i in range(40)], size=n)
    ram = rng.choice([4, 6, 8, 12], size=n)
    storage = rng.choice([64, 128, 256, 512], size=n)
    screen = np.round(rng.uniform(5.5, 6.9, size=n), 1)
    battery = rng.integers(3000, 5500, size=n)
    age = rng.integers(0, 6, size=n)
    condition = rng.choice(["new", "good", "fair", "poor"], size=n)

    base = 200.0 + 30.0 * ram + 0.5 * storage + 20.0 * screen - 60.0 * age
    price = np.exp(np.log(np.maximum(base, 20.0)) + 0.15 * rng.standard_normal(n))

    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "brand": brands,
            "model": models,
            "ram": [f"{v}GB" for v in ram],
            "storage": [f"{v} GB" for v in storage],
            "screen_size": [f"{v} inch" for v in screen],
            "battery_mah": battery,
            "device_age": age,
            "condition": condition,
            "used_price": price,
        }
    )
    # Inject missing values in both kinds of column.
    frame.loc[frame.index[:20], "battery_mah"] = np.nan
    frame.loc[frame.index[20:40], "condition"] = np.nan
    return frame


# ----------------------------------------------------------------------
# Cleaning helpers
# ----------------------------------------------------------------------


def test_extract_number_handles_units() -> None:
    """A leading number is recovered from '128GB'; plain text yields NaN."""
    assert extract_number("128GB") == 128.0
    assert extract_number("6.1 inch") == 6.1
    assert extract_number(42) == 42.0
    assert np.isnan(extract_number(None))
    assert np.isnan(extract_number("unknown"))


def test_coerce_numeric_like_converts_unit_columns(raw_frame: pd.DataFrame) -> None:
    """Columns written with a unit become numeric; brand names stay categorical."""
    converted = coerce_numeric_like(raw_frame)
    assert pd.api.types.is_numeric_dtype(converted["ram"])
    assert pd.api.types.is_numeric_dtype(converted["storage"])
    assert pd.api.types.is_numeric_dtype(converted["screen_size"])
    assert not pd.api.types.is_numeric_dtype(converted["brand"])


def test_group_rare_levels() -> None:
    """Levels seen fewer than `threshold` times collapse into a single 'rare'."""
    series = pd.Series(["a"] * 20 + ["b"] * 3 + ["c"] * 2)
    grouped = group_rare_levels(series, threshold=10)
    assert set(grouped.unique()) == {"a", "rare"}


def test_guess_target(raw_frame: pd.DataFrame) -> None:
    """The resale price column is picked without being named explicitly."""
    assert guess_target(raw_frame) == "used_price"


# ----------------------------------------------------------------------
# Design matrix
# ----------------------------------------------------------------------


def test_design_matrix_is_standardized_and_centered(raw_frame: pd.DataFrame) -> None:
    """Training columns have zero mean and unit variance, and y is centered.

    Centering y is what removes the intercept from the objective.
    """
    design = build_design_matrix(raw_frame, seed=0)

    assert design.target == "used_price"
    assert design.X_train.shape[0] + design.X_test.shape[0] == len(raw_frame)
    assert design.X_train.shape[1] == len(design.feature_names)

    assert np.allclose(design.X_train.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(design.X_train.std(axis=0), 1.0, atol=1e-10)
    assert abs(design.y_train.mean()) < 1e-10


def test_design_matrix_drops_identifier_columns(raw_frame: pd.DataFrame) -> None:
    """Row identifiers carry no signal and must not reach the design matrix."""
    design = build_design_matrix(raw_frame, seed=0)
    assert not any(name == "id" for name in design.feature_names)


def test_one_hot_encoding_widens_the_problem(raw_frame: pd.DataFrame) -> None:
    """The high-cardinality model column is what makes d large enough."""
    design = build_design_matrix(raw_frame, rare_threshold=1, seed=0)
    assert design.X_train.shape[1] > 40


def test_no_constant_columns_survive(raw_frame: pd.DataFrame) -> None:
    """A constant column would make the Gram matrix singular after centering."""
    design = build_design_matrix(raw_frame, seed=0)
    assert np.all(design.X_train.std(axis=0) > 1e-12)


# ----------------------------------------------------------------------
# Lambda selection and persistence
# ----------------------------------------------------------------------


def test_ridge_solution_matches_the_normal_equations(raw_frame: pd.DataFrame) -> None:
    """The closed-form solution leaves a gradient of norm below 1e-8."""
    design = build_design_matrix(raw_frame, seed=0)
    X, y, lam = design.X_train, design.y_train, 1e-2
    w = ridge_solution(X, y, lam)
    residual = X.T @ (X @ w - y) / X.shape[0] + lam * w
    assert np.linalg.norm(residual) < 1e-8


def test_select_lambda_returns_a_grid_point(raw_frame: pd.DataFrame) -> None:
    """Cross-validation returns one of the candidate values, not an average."""
    design = build_design_matrix(raw_frame, seed=0)
    grid = np.logspace(-4, 1, 8)
    lam, records = select_lambda(design.X_train, design.y_train, grid=grid, seed=0)
    assert lam in set(float(g) for g in grid)
    assert len(records) == len(grid)
    assert all(np.isfinite(r["cv_mse"]) for r in records)


def test_save_and_reload_round_trip(raw_frame: pd.DataFrame, tmp_path) -> None:
    """A problem written to disk reloads with the same constants and optimum."""
    design = build_design_matrix(raw_frame, seed=0)
    config = save_processed(design, lam=1e-2, out_dir=tmp_path)

    assert config["d"] == design.X_train.shape[1]
    assert config["kappa"] >= 1.0
    assert np.isfinite(config["f_star"])

    problem, reloaded = load_problem(directory=tmp_path)
    assert problem.d == design.X_train.shape[1]
    assert np.isclose(problem.f_star, reloaded["f_star"])
    assert np.linalg.norm(problem.grad(problem.w_star)) < 1e-10


def test_larger_lambda_lowers_the_condition_number(raw_frame: pd.DataFrame) -> None:
    """The link between regularization and conditioning, used in section 5.6."""
    design = build_design_matrix(raw_frame, seed=0)
    small = RidgeProblem(design.X_train, design.y_train, 1e-4)
    large = RidgeProblem(design.X_train, design.y_train, 1e0)
    assert large.kappa < small.kappa
    assert large.mu > small.mu
