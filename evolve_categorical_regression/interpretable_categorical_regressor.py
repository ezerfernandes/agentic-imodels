"""Agent-editable categorical regressor experiment.

Run with:
    uv run --extra research evolve_categorical_regression/interpretable_categorical_regressor.py

Agents should edit only the estimator class and the metadata variables near the
top of this file. The evaluation harness below mirrors `evolve/` but preserves
categorical feature values instead of ordinal-encoding them away.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from evolve_categorical_regression.src.feature_metadata import (
    coerce_categorical_frame,
    infer_feature_metadata,
)
from evolve_categorical_regression.src.interp_eval import CATEGORY_TESTS, run_all_interp_tests
from evolve_categorical_regression.src.performance_eval import (
    RESULTS_DIR,
    compute_rank_scores,
    evaluate_all_regressors,
    recompute_all_mean_ranks,
    upsert_overall_results,
)
from evolve_categorical_regression.src.visualize import plot_interp_vs_performance


class CategoricalEffectRegressor(BaseEstimator, RegressorMixin):
    """Additive baseline with numeric linear terms and smoothed category effects."""

    def __init__(self, smoothing=3.0, numeric_l2=1e-3, max_levels_to_print=8):
        self.smoothing = smoothing
        self.numeric_l2 = numeric_l2
        self.max_levels_to_print = max_levels_to_print

    def fit(self, X, y):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        y = np.asarray(y, dtype=float)
        self.metadata_ = infer_feature_metadata(df)
        df = coerce_categorical_frame(df, self.metadata_)

        self.intercept_ = float(np.nanmean(y))
        residual = y - self.intercept_

        self.numeric_medians_ = {}
        self.numeric_means_ = {}
        self.numeric_scales_ = {}
        self.numeric_coefs_ = {}

        if self.metadata_.numeric_features:
            X_num = []
            for col in self.metadata_.numeric_features:
                values = df[col].astype(float)
                median = float(values.median()) if values.notna().any() else 0.0
                values = values.fillna(median).to_numpy(dtype=float)
                mean = float(values.mean())
                scale = float(values.std()) or 1.0
                self.numeric_medians_[col] = median
                self.numeric_means_[col] = mean
                self.numeric_scales_[col] = scale
                X_num.append((values - mean) / scale)

            X_design = np.column_stack(X_num)
            ridge = self.numeric_l2 * np.eye(X_design.shape[1])
            coefs = np.linalg.solve(X_design.T @ X_design + ridge, X_design.T @ residual)
            for col, coef in zip(self.metadata_.numeric_features, coefs):
                self.numeric_coefs_[col] = float(coef)
            residual = residual - X_design @ coefs

        self.category_effects_ = {}
        self.category_counts_ = {}
        for col in self.metadata_.categorical_features:
            effects = {}
            counts = {}
            grouped = pd.DataFrame({"level": df[col].astype(str), "residual": residual}).groupby("level")
            for level, group in grouped:
                count = int(len(group))
                effect = float(group["residual"].sum() / (count + self.smoothing))
                effects[str(level)] = effect
                counts[str(level)] = count
            self.category_effects_[col] = effects
            self.category_counts_[col] = counts

        self.n_features_in_ = len(self.metadata_.feature_names)
        return self

    def predict(self, X):
        check_is_fitted(self, "metadata_")
        df = coerce_categorical_frame(X, self.metadata_)
        pred = np.full(len(df), self.intercept_, dtype=float)

        for col, coef in self.numeric_coefs_.items():
            values = df[col].fillna(self.numeric_medians_[col]).to_numpy(dtype=float)
            pred += coef * ((values - self.numeric_means_[col]) / self.numeric_scales_[col])

        for col, effects in self.category_effects_.items():
            pred += df[col].astype(str).map(effects).fillna(0.0).to_numpy(dtype=float)

        return pred

    def __str__(self):
        check_is_fitted(self, "metadata_")
        lines = [
            "CategoricalEffectRegressor",
            "prediction = intercept + numeric linear terms + smoothed categorical level effects",
            f"intercept: {self.intercept_:+.4f}",
        ]

        if self.numeric_coefs_:
            lines.append("\nNumeric terms (standardized):")
            for col, coef in sorted(self.numeric_coefs_.items(), key=lambda item: -abs(item[1])):
                lines.append(f"  {coef:+.4f} * z({col})")

        if self.category_effects_:
            lines.append("\nCategorical effects (unseen levels use +0.0000):")
            for col in self.metadata_.categorical_features:
                lines.append(f"  {col}:")
                effects = self.category_effects_[col]
                counts = self.category_counts_[col]
                shown = sorted(effects.items(), key=lambda item: -abs(item[1]))[: self.max_levels_to_print]
                for level, effect in shown:
                    lines.append(f"    {level!r}: {effect:+.4f}  (n={counts[level]})")
                remaining = len(effects) - len(shown)
                if remaining > 0:
                    lines.append(f"    ... {remaining} more levels omitted")

        return "\n".join(lines)


import sys as _sys

_sys.modules.setdefault("interpretable_categorical_regressor", _sys.modules[__name__])
CategoricalEffectRegressor.__module__ = "interpretable_categorical_regressor"

model_shorthand_name = "CategoricalEffect"
model_description = "Baseline additive categorical-level effects plus numeric ridge terms"
model_defs = [(model_shorthand_name, CategoricalEffectRegressor())]


def _suite(test_name):
    for label, tests in CATEGORY_TESTS:
        if test_name in {fn.__name__ for fn in tests}:
            return label
    return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="gpt-4o")
    args = parser.parse_args()

    t0 = time.time()
    interp_results = run_all_interp_tests(model_defs, checkpoint=args.checkpoint)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    dataset_rmses = evaluate_all_regressors(model_defs)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    model_name = model_defs[0][0]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]
    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            existing_interp = [row for row in csv.DictReader(f) if row.get("model") != model_name]
    new_interp = [
        {
            "model": r["model"],
            "test": r["test"],
            "suite": _suite(r["test"]),
            "passed": r["passed"],
            "ground_truth": r.get("ground_truth", ""),
            "response": r.get("response", ""),
        }
        for r in interp_results
    ]
    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)

    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            existing_perf = [row for row in csv.DictReader(f) if row.get("model") != model_name]

    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append(
            {
                "dataset": ds_name,
                "model": model_name,
                "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
                "rank": "",
            }
        )

    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)
    for rows in by_dataset.values():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda item: item[1])
        for rank_idx, (row, _) in enumerate(valid, 1):
            row["rank"] = rank_idx

    with open(perf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=perf_fields)
        writer.writeheader()
        for rows in by_dataset.values():
            writer.writerows(rows)

    all_dataset_rmses = defaultdict(dict)
    for row in existing_perf:
        rmse_str = row.get("rmse", "")
        all_dataset_rmses[row["dataset"]][row["model"]] = (
            float(rmse_str) if rmse_str not in ("", None) else float("nan")
        )
    avg_rank, _ = compute_rank_scores(dict(all_dataset_rmses))
    mean_rank = avg_rank.get(model_shorthand_name, float("nan"))

    upsert_overall_results(
        [
            {
                "commit": git_hash,
                "mean_rank": f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
                "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total else "nan",
                "status": "",
                "model_name": model_shorthand_name,
                "description": model_description,
            }
        ],
        RESULTS_DIR,
    )
    recompute_all_mean_ranks(RESULTS_DIR)
    plot_interp_vs_performance(
        os.path.join(RESULTS_DIR, "overall_results.csv"),
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed / total:.2%})" if total else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
