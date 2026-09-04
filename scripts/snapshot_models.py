"""Capture and verify public estimator behavior across representative datasets."""

from __future__ import annotations

import argparse
import pickle
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing, make_regression

from agentic_imodels import MODEL_REGISTRY


DEFAULT_PATH = Path(__file__).resolve().parents[1] / ".snapshots" / "model_behavior.pkl"


def _datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    california_x, california_y = fetch_california_housing(return_X_y=True)
    regression_x, regression_y = make_regression(
        n_samples=300, n_features=6, random_state=0
    )
    return {
        "make_regression": (regression_x, regression_y),
        "california_housing_first_2000": (california_x[:2000], california_y[:2000]),
    }


def _estimator_classes():
    for model_name, info in MODEL_REGISTRY.items():
        yield model_name, getattr(import_module(info.module), model_name)


def _capture() -> dict[str, dict[str, dict[str, object]]]:
    snapshots: dict[str, dict[str, dict[str, object]]] = {}
    for dataset_name, (X, y) in _datasets().items():
        dataset_snapshot = snapshots.setdefault(dataset_name, {})
        for model_name, estimator_class in _estimator_classes():
            model = estimator_class().fit(X, y)
            predictions = np.asarray(model.predict(X))
            dataset_snapshot[model_name] = {
                "predictions": predictions,
                "display": str(model),
            }
            print(f"captured {dataset_name}/{model_name} ({predictions.size} predictions)")
    return snapshots


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(_capture(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote snapshot: {path}")


def _check(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"snapshot not found: {path}; run with --write first")
    with path.open("rb") as handle:
        expected = pickle.load(handle)
    actual = _capture()
    mismatches = []
    for dataset_name, model_snapshots in expected.items():
        for model_name, expected_model in model_snapshots.items():
            actual_model = actual.get(dataset_name, {}).get(model_name)
            if actual_model is None:
                mismatches.append(f"missing {dataset_name}/{model_name}")
                continue
            # Parallel tree reductions can vary by a few ulps across runs;
            # displays remain byte-for-byte compared below.
            if not np.allclose(
                expected_model["predictions"], actual_model["predictions"], rtol=0.0, atol=1e-12
            ):
                mismatches.append(f"predictions changed for {dataset_name}/{model_name}")
            if expected_model["display"] != actual_model["display"]:
                mismatches.append(f"display changed for {dataset_name}/{model_name}")
    if mismatches:
        raise SystemExit("snapshot mismatch:\n" + "\n".join(mismatches))
    print(f"snapshot matches: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write a new behavior snapshot")
    mode.add_argument("--check", action="store_true", help="compare behavior with a snapshot")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH, help="snapshot pickle path")
    args = parser.parse_args()
    if args.write:
        _write(args.path)
    else:
        _check(args.path)


if __name__ == "__main__":
    main()
