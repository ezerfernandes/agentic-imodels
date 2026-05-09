from __future__ import annotations

import numpy as np
import pandas as pd

from evolve_categorical_regression.interpretable_categorical_regressor import (
    CategoricalEffectRegressor,
)
from evolve_categorical_regression.src.feature_metadata import infer_feature_metadata
from evolve_categorical_regression.src.performance_eval import (
    CategoricalDataset,
    compute_rank_scores,
    evaluate_dataset_collection,
)


def _toy_dataset() -> CategoricalDataset:
    cities = ["a", "b", "a", "c", "b", "a"] * 10
    segments = ["low", "high", "low", "mid", "high", "mid"] * 10
    incomes = np.tile(np.array([1.0, 2.0, 1.5, 3.0, 2.5, 1.2]), 10)
    X_train = pd.DataFrame(
        {
            "city": cities,
            "segment": segments,
            "income": incomes,
        }
    )
    y_train = (
        np.array([{"a": 0.0, "b": 1.5, "c": 3.0}[city] for city in cities])
        + np.array([{"low": -0.2, "mid": 0.4, "high": 0.9}[segment] for segment in segments])
        + incomes
    )
    X_test = pd.DataFrame(
        {
            "city": ["a", "new"],
            "segment": ["mid", "low"],
            "income": [1.1, 2.2],
        }
    )
    y_test = np.array([1.7, 2.1])
    return CategoricalDataset(
        name="toy",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        categorical_features=("city", "segment"),
        numeric_features=("income",),
    )


def test_infer_feature_metadata_finds_categorical_and_numeric_columns() -> None:
    metadata = infer_feature_metadata(_toy_dataset().X_train)

    assert metadata.categorical_features == ("city", "segment")
    assert metadata.numeric_features == ("income",)
    assert metadata.feature_names == ("city", "segment", "income")


def test_default_categorical_regressor_predicts_unseen_categories_and_prints_labels() -> None:
    dataset = _toy_dataset()
    model = CategoricalEffectRegressor().fit(dataset.X_train, dataset.y_train)

    preds = model.predict(dataset.X_test)
    text = str(model)

    assert preds.shape == (2,)
    assert np.isfinite(preds).all()
    assert "city" in text
    assert "segment" in text
    assert "a" in text
    assert "income" in text


def test_evaluate_dataset_collection_preserves_dataframe_contract() -> None:
    dataset = _toy_dataset()
    results = evaluate_dataset_collection(
        [dataset],
        [("CategoricalEffect", CategoricalEffectRegressor())],
        n_jobs=1,
    )

    assert set(results) == {"toy"}
    assert set(results["toy"]) == {"CategoricalEffect"}
    assert np.isfinite(results["toy"]["CategoricalEffect"])


def test_compute_rank_scores_requires_all_datasets_for_mean_rank() -> None:
    ranks, rmses = compute_rank_scores(
        {
            "a": {"m1": 0.2, "m2": 0.4},
            "b": {"m1": 0.3, "m2": 0.1},
        }
    )

    assert ranks == {"m1": 1.5, "m2": 1.5}
    assert rmses == {"m1": 0.25, "m2": 0.25}
