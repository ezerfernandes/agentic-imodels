"""Performance evaluation for categorical OpenML regression tasks.

The harness preserves pandas DataFrames and categorical values so candidate
models can reason over original category labels rather than ordinal-coded
surrogates.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .feature_metadata import infer_feature_metadata


MAX_SAMPLES = 1000
MAX_FEATURES = 50
MIN_SAMPLES = 50
MIN_FEATURES = 2
MIN_CATEGORICAL_FEATURES = 1
SUBSAMPLE_SEED = 42
TARGET_DATASET_COUNT = 150
MIN_DISCOVERED_DATASETS = 120

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = str(ROOT / "results")
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "imodels-evolve", "categorical_regression")
_OPENML_CACHE_DIR = os.path.join(_CACHE_DIR, "openml")
_DATA_CACHE_DIR = os.path.join(_CACHE_DIR, "datasets")
_MANIFEST_PATH = os.path.join(RESULTS_DIR, "openml_categorical_regression_manifest.csv")
_memory = Memory(location=os.path.join(RESULTS_DIR, "cache"), verbose=0)


@dataclass(frozen=True)
class OpenMLDatasetSpec:
    task_id: int
    dataset_id: int
    name: str
    target_name: str
    n_instances: int
    n_features: int
    n_symbolic_features: int


@dataclass(frozen=True)
class CategoricalDataset:
    name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]


def _first_existing_column(df: pd.DataFrame, names: tuple[str, ...], default=None):
    for name in names:
        if name in df.columns:
            return name
    return default


def _list_regression_tasks() -> pd.DataFrame:
    import openml

    try:
        return openml.tasks.list_tasks(task_type_id=2, output_format="dataframe")
    except TypeError:
        return openml.tasks.list_tasks(task_type=2, output_format="dataframe")


def discover_openml_categorical_regression_specs(
    target_count: int = TARGET_DATASET_COUNT,
    min_count: int = MIN_DISCOVERED_DATASETS,
    manifest_path: str = _MANIFEST_PATH,
) -> list[OpenMLDatasetSpec]:
    """Discover active OpenML regression tasks with at least one categorical feature."""

    import openml

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    openml.config.cache_directory = _OPENML_CACHE_DIR

    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
    else:
        tasks = _list_regression_tasks()
        datasets = openml.datasets.list_datasets(output_format="dataframe")

        did_col = _first_existing_column(tasks, ("did", "dataset_id"))
        task_col = _first_existing_column(tasks, ("tid", "task_id"))
        target_col = _first_existing_column(tasks, ("target_feature", "target_name", "target"))
        if did_col is None or task_col is None or target_col is None:
            raise RuntimeError(f"Unexpected OpenML task columns: {list(tasks.columns)}")

        merged = tasks.merge(datasets, left_on=did_col, right_on="did", suffixes=("_task", ""))
        symbolic_col = _first_existing_column(
            merged,
            ("NumberOfSymbolicFeatures", "number_of_symbolic_features"),
            default="NumberOfSymbolicFeatures",
        )
        n_instances_col = _first_existing_column(
            merged,
            ("NumberOfInstances", "number_of_instances"),
            default="NumberOfInstances",
        )
        n_features_col = _first_existing_column(
            merged,
            ("NumberOfFeatures", "number_of_features"),
            default="NumberOfFeatures",
        )

        merged[symbolic_col] = pd.to_numeric(merged[symbolic_col], errors="coerce").fillna(0)
        merged[n_instances_col] = pd.to_numeric(merged[n_instances_col], errors="coerce").fillna(0)
        merged[n_features_col] = pd.to_numeric(merged[n_features_col], errors="coerce").fillna(0)

        status_col = _first_existing_column(merged, ("status",), default=None)
        if status_col is not None:
            merged = merged[merged[status_col].astype(str).str.lower().eq("active")]

        filtered = merged[
            (merged[symbolic_col] >= MIN_CATEGORICAL_FEATURES)
            & (merged[n_instances_col] >= MIN_SAMPLES)
            & (merged[n_features_col] >= MIN_FEATURES)
        ].copy()
        filtered = filtered.drop_duplicates(subset=[did_col]).sort_values(
            [n_instances_col, n_features_col, did_col],
            ascending=[True, True, True],
        )
        filtered = filtered.head(target_count)

        name_col = _first_existing_column(filtered, ("name", "name_task"), default="name")
        manifest = pd.DataFrame(
            {
                "task_id": filtered[task_col].astype(int),
                "dataset_id": filtered[did_col].astype(int),
                "name": filtered[name_col].astype(str),
                "target_name": filtered[target_col].astype(str),
                "n_instances": filtered[n_instances_col].astype(int),
                "n_features": filtered[n_features_col].astype(int),
                "n_symbolic_features": filtered[symbolic_col].astype(int),
            }
        )
        manifest.to_csv(manifest_path, index=False)

    if len(manifest) < min_count:
        raise RuntimeError(
            f"Discovered only {len(manifest)} categorical regression datasets; "
            f"expected at least {min_count}. Delete {manifest_path} and retry if the cache is stale."
        )

    return [
        OpenMLDatasetSpec(
            task_id=int(row.task_id),
            dataset_id=int(row.dataset_id),
            name=str(row.name),
            target_name=str(row.target_name),
            n_instances=int(row.n_instances),
            n_features=int(row.n_features),
            n_symbolic_features=int(row.n_symbolic_features),
        )
        for row in manifest.head(target_count).itertuples(index=False)
    ]


def _load_openml_dataset(spec: OpenMLDatasetSpec) -> CategoricalDataset:
    import openml

    os.makedirs(_DATA_CACHE_DIR, exist_ok=True)
    openml.config.cache_directory = _OPENML_CACHE_DIR
    cache_path = os.path.join(_DATA_CACHE_DIR, f"{spec.dataset_id}_{spec.task_id}.pkl")

    if os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)
    else:
        dataset = openml.datasets.get_dataset(spec.dataset_id, download_data=True)
        X, y, _, attribute_names = dataset.get_data(target=spec.target_name)
        df = pd.DataFrame(X, columns=attribute_names)
        df["__target__"] = y
        df.to_pickle(cache_path)

    y = pd.to_numeric(df["__target__"], errors="coerce").to_numpy(dtype=float)
    X = df.drop(columns=["__target__"]).copy()
    valid = np.isfinite(y)
    X = X.loc[valid].reset_index(drop=True)
    y = y[valid]

    metadata = infer_feature_metadata(X)
    if len(metadata.categorical_features) < MIN_CATEGORICAL_FEATURES:
        raise ValueError(f"{spec.name} has no categorical features after loading")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return CategoricalDataset(
        name=f"openml/{spec.dataset_id}:{spec.name}",
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=np.asarray(y_train, dtype=float),
        y_test=np.asarray(y_test, dtype=float),
        categorical_features=metadata.categorical_features,
        numeric_features=metadata.numeric_features,
    )


def get_all_datasets(target_count: int = TARGET_DATASET_COUNT):
    specs = discover_openml_categorical_regression_specs(target_count=target_count)
    for spec in specs:
        try:
            yield _load_openml_dataset(spec)
        except Exception as exc:
            print(f"  WARNING: skipping openml/{spec.dataset_id}:{spec.name}: {exc}")


def subsample_dataset(dataset: CategoricalDataset, seed: int = SUBSAMPLE_SEED) -> CategoricalDataset:
    rng = np.random.RandomState(seed)
    X_train = dataset.X_train.copy()
    X_test = dataset.X_test.copy()
    y_train = np.asarray(dataset.y_train, dtype=float)
    y_test = np.asarray(dataset.y_test, dtype=float)

    columns = list(X_train.columns)
    if len(columns) > MAX_FEATURES:
        categorical = [c for c in dataset.categorical_features if c in columns]
        numeric = [c for c in dataset.numeric_features if c in columns]
        keep = categorical[:MAX_FEATURES]
        remaining = MAX_FEATURES - len(keep)
        if remaining > 0 and numeric:
            chosen = rng.choice(numeric, min(remaining, len(numeric)), replace=False).tolist()
            keep.extend(chosen)
        keep = keep[:MAX_FEATURES]
        X_train = X_train.loc[:, keep]
        X_test = X_test.loc[:, keep]

    if len(X_train) > MAX_SAMPLES:
        idx = rng.choice(len(X_train), MAX_SAMPLES, replace=False)
        X_train = X_train.iloc[idx].reset_index(drop=True)
        y_train = y_train[idx]

    metadata = infer_feature_metadata(X_train)
    return CategoricalDataset(
        name=dataset.name,
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train,
        y_test=y_test,
        categorical_features=metadata.categorical_features,
        numeric_features=metadata.numeric_features,
    )


@_memory.cache
def _run_one_regressor(model_name, ds_name, reg, X_train, X_test, y_train, y_test):
    try:
        model = deepcopy(reg)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return float(np.sqrt(mean_squared_error(y_test, preds)))
    except Exception as exc:
        return str(exc)


def _eval_one_dataset(dataset: CategoricalDataset, model_defs):
    dataset = subsample_dataset(dataset)
    if len(dataset.X_train) < MIN_SAMPLES:
        return dataset.name, {}
    if dataset.X_train.shape[1] < MIN_FEATURES:
        return dataset.name, {}
    if len(dataset.categorical_features) < MIN_CATEGORICAL_FEATURES:
        return dataset.name, {}

    y_mean = float(np.mean(dataset.y_train))
    y_std = float(np.std(dataset.y_train))
    y_train = (dataset.y_train - y_mean) / y_std if y_std > 0 else dataset.y_train
    y_test = (dataset.y_test - y_mean) / y_std if y_std > 0 else dataset.y_test

    print(
        f"\n  Dataset: {dataset.name} — {dataset.X_train.shape[1]} features "
        f"({len(dataset.categorical_features)} categorical), {len(dataset.X_train)} train samples"
    )
    model_rmses = {}
    for name, reg in model_defs:
        result = _run_one_regressor(name, dataset.name, reg, dataset.X_train, dataset.X_test, y_train, y_test)
        if isinstance(result, float):
            model_rmses[name] = result
            print(f"    {name:<20}: {result:.4f}")
        else:
            print(f"    {name:<20}: ERROR — {result}")
            model_rmses[name] = float("nan")
    return dataset.name, model_rmses


def evaluate_dataset_collection(datasets, model_defs, n_jobs: int = -1):
    from joblib import Parallel, delayed

    datasets = list(datasets)
    if n_jobs == 1:
        return dict(_eval_one_dataset(dataset, model_defs) for dataset in datasets)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_one_dataset)(dataset, model_defs) for dataset in datasets
    )
    return dict(results)


def evaluate_all_regressors(model_defs, target_count: int = TARGET_DATASET_COUNT):
    return evaluate_dataset_collection(get_all_datasets(target_count=target_count), model_defs)


def compute_rank_scores(dataset_rmses):
    all_model_names = set()
    for model_rmses in dataset_rmses.values():
        all_model_names.update(model_rmses.keys())

    ranks_per_model = {name: [] for name in all_model_names}
    mean_rmse_per_model = {name: [] for name in all_model_names}

    for model_rmses in dataset_rmses.values():
        valid = [(name, value) for name, value in model_rmses.items() if not np.isnan(value)]
        rank_map = {name: rank + 1 for rank, (name, _) in enumerate(sorted(valid, key=lambda item: item[1]))}
        for name in all_model_names:
            if name in model_rmses and not np.isnan(model_rmses[name]):
                ranks_per_model[name].append(rank_map[name])
                mean_rmse_per_model[name].append(model_rmses[name])

    n_datasets = len(dataset_rmses)
    avg_rank = {
        name: float(np.mean(values))
        for name, values in ranks_per_model.items()
        if values and len(values) == n_datasets
    }
    avg_rmse = {
        name: float(np.mean(values))
        for name, values in mean_rmse_per_model.items()
        if values and len(values) == n_datasets
    }
    return avg_rank, avg_rmse


OVERALL_CSV_COLS = [
    "commit",
    "mean_rank",
    "frac_interpretability_tests_passed",
    "status",
    "model_name",
    "description",
]


def upsert_overall_results(rows, results_dir):
    path = os.path.join(results_dir, "overall_results.csv")
    existing = []
    new_keys = {(row["model_name"], row.get("description", "")) for row in rows}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("model_name"), row.get("description", "")) not in new_keys:
                    existing.append(row)

    all_rows = existing + [{key: row.get(key, "") for key in OVERALL_CSV_COLS} for row in rows]
    os.makedirs(results_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Overall results saved -> {path}")


def recompute_all_mean_ranks(results_dir):
    perf_path = os.path.join(results_dir, "performance_results.csv")
    overall_path = os.path.join(results_dir, "overall_results.csv")
    if not (os.path.exists(perf_path) and os.path.exists(overall_path)):
        return {}

    dataset_rmses = defaultdict(dict)
    with open(perf_path, newline="") as f:
        for row in csv.DictReader(f):
            rmse_str = row.get("rmse", "")
            dataset_rmses[row["dataset"]][row["model"]] = (
                float(rmse_str) if rmse_str not in ("", None) else float("nan")
            )

    avg_rank, _ = compute_rank_scores(dict(dataset_rmses))
    with open(overall_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        rank = avg_rank.get(row.get("model_name"), float("nan"))
        row["mean_rank"] = "nan" if np.isnan(rank) else f"{rank:.2f}"
    with open(overall_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(rows)
    return avg_rank
