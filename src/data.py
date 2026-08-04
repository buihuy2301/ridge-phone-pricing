"""Data preparation.

Turns the raw Kaggle file into the fixed pair (X, y) that every experiment
shares. The processing is deliberately minimal, as the assignment is about
optimization rather than feature engineering; the only step that matters for
the optimizer is the column standardization, which changes the condition
number of the problem.

Command line usage:

    python -m src.data --csv data/raw/used_phone.csv --target normalized_used_price

If --target is omitted the script guesses it from the column names and prints
what it picked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .problem import RidgeProblem

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Column names that typically hold the resale price in this kind of data set.
TARGET_HINTS = (
    "used_price",
    "used price",
    "resale",
    "selling_price",
    "sale_price",
    "price",
)

# Columns that carry no signal and only inflate the design matrix.
ID_PATTERNS = (r"^id$", r"^unnamed", r"^index$", r"_id$")

NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")

# A value counts as "a number with a unit" only when the number comes first
# and is followed by a short unit token: "128GB", "6.1 inch", "4500 mAh".
# This deliberately rejects entries such as "model_36" or "iPhone 13", whose
# digits are part of a name and whose column belongs in the one-hot block.
NUMBER_WITH_UNIT_PATTERN = re.compile(r'^\s*[-+]?\d*\.?\d+\s*[a-zA-Z"\']{0,6}\s*$')


@dataclass
class DesignMatrix:
    """Everything the experiments need, plus what the report has to cite."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target: str
    log_target: bool
    n_raw_rows: int
    n_raw_cols: int


# ----------------------------------------------------------------------
# Reading and cleaning
# ----------------------------------------------------------------------


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read the raw file and report its shape."""
    df = pd.read_csv(csv_path)
    return df


def guess_target(df: pd.DataFrame) -> str:
    """Pick the most likely resale-price column."""
    lowered = {col.lower(): col for col in df.columns}
    for hint in TARGET_HINTS:
        for lower, original in lowered.items():
            if hint in lower and pd.api.types.is_numeric_dtype(df[original]):
                return original
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise ValueError("no numeric column found to use as the target")
    return numeric[-1]


def _is_id_column(name: str) -> bool:
    lowered = name.lower()
    return any(re.search(pattern, lowered) for pattern in ID_PATTERNS)


def extract_number(value) -> float:
    """Pull the leading number out of strings such as '128GB' or '6.1 inch'."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = NUMBER_PATTERN.search(str(value))
    return float(match.group()) if match else np.nan


def looks_numeric_with_unit(value) -> bool:
    """True when the value is a number optionally followed by a short unit."""
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    return bool(NUMBER_WITH_UNIT_PATTERN.match(str(value)))


def coerce_numeric_like(df: pd.DataFrame, min_match_ratio: float = 0.9) -> pd.DataFrame:
    """Convert string columns that are really numbers with a unit attached.

    A column is converted when at least `min_match_ratio` of its non-missing
    entries look like a number followed by a unit. Columns whose digits are
    part of a name, such as a model identifier, stay categorical and end up in
    the one-hot block, which is what makes the design matrix wide.
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        non_null = out[col].dropna()
        if non_null.empty:
            continue
        if non_null.map(looks_numeric_with_unit).mean() >= min_match_ratio:
            out[col] = out[col].map(extract_number)
    return out


def split_column_types(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Separate numeric predictors from categorical ones."""
    numeric, categorical = [], []
    for col in df.columns:
        if col == target or _is_id_column(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def group_rare_levels(series: pd.Series, threshold: int = 10) -> pd.Series:
    """Collapse levels seen fewer than `threshold` times into 'rare'."""
    counts = series.value_counts()
    frequent = set(counts[counts >= threshold].index)
    return series.where(series.isin(frequent), other="rare")


# ----------------------------------------------------------------------
# Design matrix
# ----------------------------------------------------------------------


def drop_duplicate_columns(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Remove columns that are bitwise identical to an earlier one.

    Squaring a binary column reproduces it exactly, and duplicated columns make
    the Gram matrix singular for no benefit. Only exact duplicates are caught;
    near-collinear columns are left in place on purpose, since they are what
    makes the condition number interesting.
    """
    seen: dict[bytes, int] = {}
    keep: list[int] = []
    for j in range(X.shape[1]):
        column = np.ascontiguousarray(X[:, j])
        key = hashlib.blake2b(column.tobytes(), digest_size=16).digest()
        if key not in seen:
            seen[key] = j
            keep.append(j)
    return X[:, keep], [names[j] for j in keep]


def add_interaction_terms(numeric_block: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Append pairwise products and squares of the given numeric columns.

    The products are formed from standardized columns, not from the raw ones.
    Multiplying columns whose scales differ by orders of magnitude (a price
    near 1e5 against a screen size near 6) produces a design matrix with a much
    larger leading eigenvalue and a far worse condition number, even though the
    column span is the same. The scaling applied here is part of the feature
    definition and is shared by the train and test halves; the standardization
    in `build_design_matrix` is still fitted on the training rows only.
    """
    scaled = {}
    for col in columns:
        values = numeric_block[col].to_numpy(dtype=np.float64)
        spread = values.std()
        scaled[col] = (values - values.mean()) / (spread if spread > 1e-12 else 1.0)

    products = {}
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            products[f"{a}_x_{b}"] = scaled[a] * scaled[b]
    for col in columns:
        # Squaring a binary column reproduces it, so skip those.
        if numeric_block[col].nunique() > 2:
            products[f"{col}_sq"] = scaled[col] ** 2

    return pd.concat([numeric_block, pd.DataFrame(products, index=numeric_block.index)], axis=1)


def split_and_standardize(
    X: np.ndarray, y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train and test, then center and scale using training rows.

    Centering y removes the intercept from the problem, which is why the
    objective in `RidgeProblem` has a single variable block.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[0])
    n_test = int(round(test_size * X.shape[0]))
    test_idx, train_idx = order[:n_test], order[n_test:]

    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train_raw, y_test_raw = y[train_idx], y[test_idx]

    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std < 1e-12] = 1.0

    y_mean = y_train_raw.mean()
    return (
        (X_train_raw - mean) / std,
        y_train_raw - y_mean,
        (X_test_raw - mean) / std,
        y_test_raw - y_mean,
    )


def build_design_matrix(
    df: pd.DataFrame,
    target: str | None = None,
    rare_threshold: int = 10,
    add_interactions: bool = True,
    interaction_base: int | None = None,
    log_target: bool = True,
    test_size: float = 0.2,
    seed: int = 0,
    sample_rows: int | None = None,
) -> DesignMatrix:
    """Build the standardized, centered (X, y) pair and split it.

    `interaction_base` limits how many numeric columns take part in the
    pairwise products; None uses all of them. `sample_rows` draws a random
    subset of the rows before anything else, which is how the problem size is
    kept at a scale where the whole parameter grid can be run.
    """
    n_raw_rows, n_raw_cols = df.shape
    target = target or guess_target(df)

    if sample_rows is not None and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=seed)

    df = coerce_numeric_like(df)
    df = df[df[target].notna()]
    df = df[df[target] > 0] if log_target else df

    numeric, categorical = split_column_types(df, target)

    numeric_block = df[numeric].copy()
    for col in numeric:
        numeric_block[col] = numeric_block[col].fillna(numeric_block[col].median())

    categorical_block = df[categorical].copy()
    for col in categorical:
        filled = categorical_block[col].astype("object").fillna("unknown").astype(str)
        categorical_block[col] = group_rare_levels(filled, rare_threshold)

    if add_interactions and len(numeric) >= 2:
        # Pairwise products and squares, only to widen the design matrix.
        base = numeric if interaction_base is None else numeric[: max(2, interaction_base)]
        numeric_block = add_interaction_terms(numeric_block, base)

    if categorical:
        dummies = pd.get_dummies(categorical_block, drop_first=True, dtype=float)
        features = pd.concat(
            [numeric_block.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1
        )
    else:
        features = numeric_block.reset_index(drop=True)

    y_raw = df[target].to_numpy(dtype=np.float64)
    y = np.log(y_raw) if log_target else y_raw

    X = features.to_numpy(dtype=np.float64)
    feature_names = [str(c) for c in features.columns]

    # Drop constant columns: they carry no information and make the Gram
    # matrix singular once the intercept is removed by centering.
    keep = X.std(axis=0) > 1e-12
    X = X[:, keep]
    feature_names = [name for name, k in zip(feature_names, keep) if k]
    X, feature_names = drop_duplicate_columns(X, feature_names)

    X_train, y_train, X_test, y_test = split_and_standardize(X, y, test_size, seed)

    return DesignMatrix(
        X_train=np.ascontiguousarray(X_train),
        y_train=np.ascontiguousarray(y_train),
        X_test=np.ascontiguousarray(X_test),
        y_test=np.ascontiguousarray(y_test),
        feature_names=feature_names,
        target=target,
        log_target=log_target,
        n_raw_rows=n_raw_rows,
        n_raw_cols=n_raw_cols,
    )


# ----------------------------------------------------------------------
# Choosing lambda
# ----------------------------------------------------------------------


def ridge_solution(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge solution for the objective used in this project."""
    n = X.shape[0]
    A = X.T @ X / n + lam * np.eye(X.shape[1])
    b = X.T @ y / n
    return np.linalg.solve(A, b)


def select_lambda(
    X: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray | None = None,
    n_folds: int = 5,
    seed: int = 0,
    rule: str = "one_se",
) -> tuple[float, list[dict]]:
    """Pick lambda by k-fold cross-validation on the training set.

    This is a model selection step, not an optimization one: the value chosen
    here is fixed for every experiment that follows.

    Two rules are available.

    rule = "best"    picks the grid point with the smallest mean CV error.
    rule = "one_se"  picks the largest lambda whose mean CV error is still
                     within one standard error of the best. This is the usual
                     conservative choice when the CV curve is flat over
                     several orders of magnitude, which is exactly the
                     situation here: the curve does not distinguish between
                     the small values of lambda, while the condition number
                     kappa = L / mu is inversely proportional to lambda. The
                     rule therefore returns a statistically equivalent model
                     on a far better conditioned problem.
    """
    lam_grid = np.logspace(-6, 2, 25) if grid is None else np.asarray(grid, dtype=np.float64)

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)

    records = []
    for lam in lam_grid:
        errors = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            w = ridge_solution(X[train_idx], y[train_idx], float(lam))
            residual = X[val_idx] @ w - y[val_idx]
            errors.append(float(residual @ residual / residual.size))
        errors_array = np.asarray(errors, dtype=np.float64)
        records.append(
            {
                "lam": float(lam),
                "cv_mse": float(errors_array.mean()),
                "cv_mse_std": float(errors_array.std(ddof=1)),
                "cv_mse_se": float(errors_array.std(ddof=1) / np.sqrt(n_folds)),
            }
        )

    best = min(records, key=lambda r: r["cv_mse"])
    if rule == "best":
        return best["lam"], records
    if rule != "one_se":
        raise ValueError(f"unknown selection rule: {rule}")

    ceiling = best["cv_mse"] + best["cv_mse_se"]
    within = [r for r in records if r["cv_mse"] <= ceiling]
    chosen = max(within, key=lambda r: r["lam"])
    return chosen["lam"], records


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def save_processed(design: DesignMatrix, lam: float, out_dir: Path | None = None) -> dict:
    """Write the fixed problem instance to data/processed and return its config."""
    directory = Path(out_dir) if out_dir is not None else PROCESSED_DIR
    directory.mkdir(parents=True, exist_ok=True)

    np.save(directory / "X_train.npy", design.X_train)
    np.save(directory / "y_train.npy", design.y_train)
    np.save(directory / "X_test.npy", design.X_test)
    np.save(directory / "y_test.npy", design.y_test)
    (directory / "feature_names.json").write_text(json.dumps(design.feature_names, indent=1))

    problem = RidgeProblem(design.X_train, design.y_train, lam)
    config = problem.summary()
    config.update(
        {
            "target": design.target,
            "log_target": design.log_target,
            "n_test": int(design.X_test.shape[0]),
            "n_raw_rows": design.n_raw_rows,
            "n_raw_cols": design.n_raw_cols,
        }
    )
    (directory / "problem_config.json").write_text(json.dumps(config, indent=1))
    return config


def load_problem(lam: float | None = None, directory: Path | None = None):
    """Rebuild the fixed problem instance from data/processed."""
    path = Path(directory) if directory is not None else PROCESSED_DIR
    X_train = np.load(path / "X_train.npy")
    y_train = np.load(path / "y_train.npy")
    config = json.loads((path / "problem_config.json").read_text())
    return RidgeProblem(X_train, y_train, lam if lam is not None else config["lam"]), config


def load_test_set(directory: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read back the held-out split written by `save_processed`."""
    path = Path(directory) if directory is not None else PROCESSED_DIR
    return np.load(path / "X_test.npy"), np.load(path / "y_test.npy")


# ----------------------------------------------------------------------
# Command line entry point
# ----------------------------------------------------------------------


def main() -> None:
    """Build the problem instance from a raw csv file and write it to disk."""
    parser = argparse.ArgumentParser(description="Prepare the fixed optimization problem.")
    parser.add_argument("--csv", required=True, help="path to the raw Kaggle csv file")
    parser.add_argument("--target", default=None, help="name of the target column")
    parser.add_argument("--rare-threshold", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-interactions", action="store_true")
    parser.add_argument("--no-log-target", action="store_true")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="draw this many rows at random before building the design matrix",
    )
    parser.add_argument(
        "--interaction-base",
        type=int,
        default=None,
        help="number of numeric columns entering the pairwise products (default: all)",
    )
    args = parser.parse_args()

    df = load_raw(args.csv)
    print(f"raw shape: {df.shape[0]} rows, {df.shape[1]} columns")

    design = build_design_matrix(
        df,
        target=args.target,
        rare_threshold=args.rare_threshold,
        add_interactions=not args.no_interactions,
        interaction_base=args.interaction_base,
        log_target=not args.no_log_target,
        test_size=args.test_size,
        seed=args.seed,
        sample_rows=args.sample_rows,
    )
    print(f"target column: {design.target} (log transform: {design.log_target})")
    print(f"design matrix: n = {design.X_train.shape[0]}, d = {design.X_train.shape[1]}")

    lam, records = select_lambda(design.X_train, design.y_train, seed=args.seed)
    print(f"lambda selected by 5-fold CV: {lam:.6g}")

    config = save_processed(design, lam)
    print("problem constants:")
    for key in ("n", "d", "lam", "L", "mu", "kappa", "f_star"):
        print(f"  {key:>7s} = {config[key]:.6g}")

    (PROCESSED_DIR / "lambda_cv.json").write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
