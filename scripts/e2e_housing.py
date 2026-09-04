"""Run the public estimators end to end on a named California housing split."""

from __future__ import annotations

import argparse
import json
import time
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from agentic_imodels import MODEL_REGISTRY

N_ROWS = 3000
RANDOM_STATE = 0
BATCH_TOLERANCE = 1e-8


def _load_data() -> tuple[object, object, object, object]:
    housing = fetch_california_housing(as_frame=True)
    X = housing.data.sample(n=N_ROWS, random_state=RANDOM_STATE)
    y = housing.target.loc[X.index]
    return train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)


def _estimator_classes():
    for model_name, info in MODEL_REGISTRY.items():
        yield model_name, getattr(import_module(info.module), model_name)


def _evaluate(model_name: str, estimator_class, X_train, X_test, y_train, y_test) -> dict:
    started = time.perf_counter()
    result = {
        "model": model_name,
        "fit_s": None,
        "rmse": None,
        "r2": None,
        "display_chars": 0,
        "names_ok": False,
        "batch_ok": False,
        "batch_max_diff": None,
        "error": None,
    }
    print(f"=== {model_name} ===")
    try:
        model = estimator_class().fit(X_train, y_train)
        result["fit_s"] = time.perf_counter() - started
        display = str(model)
        batch_predictions = np.asarray(model.predict(X_test), dtype=float)
        row_predictions = np.concatenate(
            [
                np.asarray(model.predict(X_test.iloc[[index]]), dtype=float)
                for index in range(len(X_test))
            ]
        )
        max_diff = float(np.max(np.abs(batch_predictions - row_predictions)))
        result.update(
            {
                "rmse": float(mean_squared_error(y_test, batch_predictions) ** 0.5),
                "r2": float(r2_score(y_test, batch_predictions)),
                "display_chars": len(display),
                "names_ok": any(name in display for name in X_test.columns),
                "batch_ok": max_diff <= BATCH_TOLERANCE,
                "batch_max_diff": max_diff,
            }
        )
        print(display)
        print(f"batch_max_diff: {max_diff:.3e}")
    except Exception as exc:
        result["fit_s"] = time.perf_counter() - started
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"ERROR: {result['error']}")
    return result


def has_failures(results: list[dict]) -> bool:
    return any(
        result["error"] is not None or not result["names_ok"] or not result["batch_ok"]
        for result in results
    )


def run(json_path: Path | None = None) -> list[dict]:
    X_train, X_test, y_train, y_test = _load_data()
    results = [
        _evaluate(model_name, estimator_class, X_train, X_test, y_train, y_test)
        for model_name, estimator_class in _estimator_classes()
    ]

    print("Summary:")
    print("model\tfit_s\trmse\tr2\tdisplay_chars\tnames_ok\tbatch_ok")
    for result in results:
        fit_s = "-" if result["fit_s"] is None else f"{result['fit_s']:.2f}"
        rmse = "-" if result["rmse"] is None else f"{result['rmse']:.4f}"
        r2 = "-" if result["r2"] is None else f"{result['r2']:.4f}"
        print(
            f"{result['model']}\t{fit_s}\t{rmse}\t{r2}\t"
            f"{result['display_chars']}\t{result['names_ok']}\t{result['batch_ok']}"
        )

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON summary: {json_path}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the summary table as JSON")
    args = parser.parse_args()
    return 1 if has_failures(run(args.json)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
