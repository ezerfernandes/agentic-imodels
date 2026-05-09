"""Feature metadata helpers for categorical regression experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


@dataclass(frozen=True)
class FeatureMetadata:
    """Column names, inferred feature types, and observed categorical levels."""

    feature_names: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_levels: dict[str, tuple[str, ...]]


def ensure_dataframe(X, feature_names: Iterable[str] | None = None) -> pd.DataFrame:
    """Return `X` as a DataFrame while preserving existing column names."""

    if isinstance(X, pd.DataFrame):
        return X.copy()
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D tabular features, got shape {arr.shape}")
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=list(feature_names))


def _is_categorical(series: pd.Series) -> bool:
    return (
        series.dtype.name in {"object", "category", "string"}
        or is_bool_dtype(series)
        or not is_numeric_dtype(series)
    )


def infer_feature_metadata(X) -> FeatureMetadata:
    """Infer categorical and numeric columns from a DataFrame-like object."""

    df = ensure_dataframe(X)
    categorical = []
    numeric = []
    levels: dict[str, tuple[str, ...]] = {}

    for col in df.columns:
        name = str(col)
        series = df[col]
        if _is_categorical(series):
            categorical.append(name)
            non_missing = series.dropna().astype(str)
            levels[name] = tuple(sorted(non_missing.unique().tolist()))
        else:
            numeric.append(name)

    return FeatureMetadata(
        feature_names=tuple(str(c) for c in df.columns),
        categorical_features=tuple(categorical),
        numeric_features=tuple(numeric),
        categorical_levels=levels,
    )


def coerce_categorical_frame(X, metadata: FeatureMetadata) -> pd.DataFrame:
    """Coerce columns to the metadata contract used at fit time."""

    df = ensure_dataframe(X, metadata.feature_names)
    missing = [c for c in metadata.feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.loc[:, list(metadata.feature_names)].copy()
    for col in metadata.categorical_features:
        df[col] = df[col].astype("object").where(df[col].notna(), "__MISSING__").astype(str)
    for col in metadata.numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

