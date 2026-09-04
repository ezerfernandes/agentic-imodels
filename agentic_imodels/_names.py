"""Feature-name handling shared by the public estimators."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from sklearn.utils.validation import check_array, check_X_y

try:
    from sklearn.utils.validation import validate_data as _sklearn_validate_data
except ImportError:  # scikit-learn < 1.6
    _sklearn_validate_data = None


def _column_names(X) -> list[str] | None:
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    return [str(column) for column in columns]


def resolve_feature_names(X, n_features: int, feature_names=None) -> list[str]:
    """Resolve display names, preferring explicit names and then columns."""

    if feature_names is not None:
        names = [str(name) for name in feature_names]
    else:
        names = _column_names(X)
        if names is None:
            names = [f"x{i}" for i in range(n_features)]
    if len(names) != n_features:
        raise ValueError(f"feature_names must contain {n_features} names; got {len(names)}")
    return names


def input_feature_names(X) -> list[str] | None:
    """Return DataFrame column names before input validation/conversion."""

    return _column_names(X)


def to_array(X) -> tuple[np.ndarray, list[str]]:
    """Convert X to an array and return its positional/column names."""

    names = _column_names(X)
    array = np.asarray(X)
    if array.ndim == 1:
        n_features = array.shape[0]
    elif array.ndim >= 2:
        n_features = array.shape[1]
    else:
        n_features = 0
    if names is None:
        names = [f"x{i}" for i in range(n_features)]
    return array, names


def validate_fit_data(estimator, X, y):
    """Validate dense regression inputs and set scikit-learn fit metadata."""

    if _sklearn_validate_data is not None:
        return _sklearn_validate_data(
            estimator, X, y, reset=True, accept_sparse=False, ensure_2d=True, y_numeric=True
        )
    return check_X_y(X, y, accept_sparse=False, ensure_2d=True, y_numeric=True)


def validate_predict_data(estimator, X):
    """Validate dense prediction inputs against fitted scikit-learn metadata."""

    if _sklearn_validate_data is not None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but .* was fitted with feature names",
                category=UserWarning,
            )
            return _sklearn_validate_data(
                estimator, X, reset=False, accept_sparse=False, ensure_2d=True
            )
    return check_array(X, accept_sparse=False, ensure_2d=True)


def align_feature_names(
    X, expected_names: Sequence[str], fitted_input_names: Sequence[str] | None = None
) -> np.ndarray:
    """Align DataFrame-like input to fitted columns; leave arrays positional."""

    array, _ = to_array(X)
    expected = [str(name) for name in expected_names]
    actual = _column_names(X)
    if actual is None:
        return array
    if fitted_input_names is not None:
        fitted = [str(name) for name in fitted_input_names]
        if all(name in actual for name in expected):
            target = expected
        elif any(name in actual for name in fitted) or not any(name in actual for name in expected):
            target = fitted
        else:
            target = expected
    else:
        target = expected
    missing = [name for name in target if name not in actual]
    if missing:
        raise ValueError(f"Input is missing fitted feature columns: {', '.join(missing)}")
    order = [actual.index(name) for name in target]
    return array[:, order]


def format_feature_name(name: str) -> str:
    """Format names safely inside equations while preserving simple xN output."""

    name = str(name)
    return name if name.isidentifier() else f"`{name}`"
