"""Run baseline evaluation for categorical regression autoresearch.

Usage:
    uv run --extra research evolve_categorical_regression/run_baselines.py

Outputs under evolve_categorical_regression/results/:
    openml_categorical_regression_manifest.csv
    interpretability_results.csv
    performance_results.csv
    overall_results.csv
    interpretability_vs_performance.png
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from evolve_categorical_regression.interpretable_categorical_regressor import (
    CategoricalEffectRegressor,
)
from evolve_categorical_regression.src.interp_eval import CATEGORY_TESTS, run_all_interp_tests
from evolve_categorical_regression.src.performance_eval import (
    RESULTS_DIR,
    TARGET_DATASET_COUNT,
    compute_rank_scores,
    evaluate_all_regressors,
    upsert_overall_results,
)
from evolve_categorical_regression.src.visualize import plot_interp_vs_performance


def _onehot_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=20,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, make_column_selector(dtype_exclude=["object", "category", "bool"])),
            ("cat", categorical_pipe, make_column_selector(dtype_include=["object", "category", "bool"])),
        ],
        sparse_threshold=0.0,
    )


def _baseline(estimator, scale_numeric: bool = False) -> Pipeline:
    return Pipeline([("prep", _onehot_preprocessor(scale_numeric=scale_numeric)), ("model", estimator)])


REGRESSOR_DEFS = [
    ("Mean", _baseline(DummyRegressor(strategy="mean"))),
    ("OLS", _baseline(LinearRegression(), scale_numeric=True)),
    ("RidgeCV", _baseline(RidgeCV(cv=3), scale_numeric=True)),
    ("LassoCV", _baseline(LassoCV(cv=3, max_iter=5000), scale_numeric=True)),
    ("DT_mini", _baseline(DecisionTreeRegressor(max_leaf_nodes=8, random_state=42))),
    ("DT_large", _baseline(DecisionTreeRegressor(max_leaf_nodes=24, random_state=42))),
    ("RF", _baseline(RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1))),
    ("GBM", _baseline(GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=42))),
    ("CategoricalEffect", CategoricalEffectRegressor()),
]

MODEL_DESCRIPTIONS = {
    "Mean": "mean-only baseline",
    "OLS": "one-hot categorical encoding plus ordinary least squares",
    "RidgeCV": "one-hot categorical encoding plus ridge regression with CV",
    "LassoCV": "one-hot categorical encoding plus sparse lasso regression with CV",
    "DT_mini": "one-hot categorical encoding plus a small decision tree",
    "DT_large": "one-hot categorical encoding plus a larger decision tree",
    "RF": "one-hot categorical encoding plus random forest",
    "GBM": "one-hot categorical encoding plus gradient boosting",
    "CategoricalEffect": "additive categorical-level effects plus numeric ridge terms",
}


def _suite(test_name: str) -> str:
    for label, tests in CATEGORY_TESTS:
        if test_name in {fn.__name__ for fn in tests}:
            return label
    return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="gpt-4o")
    parser.add_argument("--target-count", type=int, default=TARGET_DATASET_COUNT)
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  CATEGORICAL INTERPRETABILITY TESTS")
    print("=" * 60)
    interp_results = run_all_interp_tests(REGRESSOR_DEFS, checkpoint=args.checkpoint)
    model_names = list(dict.fromkeys(row["model"] for row in interp_results))
    interp_scores = {
        model: sum(row["passed"] for row in interp_results if row["model"] == model)
        / len([row for row in interp_results if row["model"] == model])
        for model in model_names
    }

    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "test", "suite", "passed", "ground_truth", "response"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in interp_results:
            writer.writerow({**row, "suite": _suite(row["test"])})

    print("\n" + "=" * 60)
    print("  CATEGORICAL PERFORMANCE EVALUATION")
    print("=" * 60)
    dataset_rmses = evaluate_all_regressors(REGRESSOR_DEFS, target_count=args.target_count)
    avg_rank, avg_rmse = compute_rank_scores(dataset_rmses)

    performance_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    with open(performance_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model", "rmse", "rank"])
        for ds_name, model_rmses in dataset_rmses.items():
            valid = [(name, rmse) for name, rmse in model_rmses.items() if not np.isnan(rmse)]
            rank_map = {name: rank + 1 for rank, (name, _) in enumerate(sorted(valid, key=lambda item: item[1]))}
            for name, rmse in model_rmses.items():
                writer.writerow([ds_name, name, "" if np.isnan(rmse) else f"{rmse:.6f}", rank_map.get(name, "")])

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    overall_rows = [
        {
            "commit": git_hash or "baseline",
            "mean_rank": f"{avg_rank[name]:.2f}" if name in avg_rank else "nan",
            "frac_interpretability_tests_passed": f"{interp_scores[name]:.4f}" if name in interp_scores else "nan",
            "status": "baseline",
            "model_name": name,
            "description": MODEL_DESCRIPTIONS.get(name, name),
        }
        for name in model_names
    ]
    upsert_overall_results(overall_rows, RESULTS_DIR)
    plot_interp_vs_performance(
        os.path.join(RESULTS_DIR, "overall_results.csv"),
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print("\nPerformance summary:")
    for name, rank in sorted(avg_rank.items(), key=lambda item: item[1]):
        print(f"  {name:<20}: avg_rank={rank:.2f} mean_rmse={avg_rmse.get(name, float('nan')):.4f}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")
