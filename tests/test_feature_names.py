from __future__ import annotations

import re

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing

pd = pytest.importorskip("pandas")

import agentic_imodels as ai  # noqa: E402
from agentic_imodels._names import resolve_feature_names, to_array  # noqa: E402

FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


@pytest.fixture(scope="module")
def named_data() -> tuple[object, np.ndarray, np.ndarray]:
    try:
        housing = fetch_california_housing()
    except Exception as exc:
        pytest.skip(f"California housing unavailable: {exc}")
    X = housing.data[:1500]
    y = housing.target[:1500]
    return pd.DataFrame(X, columns=FEATURE_NAMES), X, y


@pytest.fixture(scope="module")
def named_models(named_data):
    frame, _, y = named_data
    return {name: getattr(ai, name)().fit(frame, y) for name in ai.__all__}


def test_name_helpers_support_dataframes_arrays_and_explicit_names() -> None:
    frame = pd.DataFrame([[1.0, 2.0]], columns=["left", "right"])
    frame_array, frame_names = to_array(frame)
    np.testing.assert_allclose(frame_array, [[1.0, 2.0]])
    assert frame_names == ["left", "right"]

    array, array_names = to_array(np.ones((3, 2)))
    assert array.shape == (3, 2)
    assert array_names == ["x0", "x1"]

    list_array, list_names = to_array([[1, 2], [3, 4]])
    assert list_array.shape == (2, 2)
    assert list_names == ["x0", "x1"]

    assert resolve_feature_names(frame, 2) == ["left", "right"]
    assert resolve_feature_names(frame, 2, ["first", "second"]) == ["first", "second"]
    with pytest.raises(ValueError, match="2 names"):
        resolve_feature_names(frame, 2, ["only"])


@pytest.mark.parametrize("model_name", ai.__all__)
def test_named_dataframe_displays_use_columns(model_name: str, named_models) -> None:
    model = named_models[model_name]
    text = str(model)
    assert any(name in text for name in FEATURE_NAMES)
    assert not re.search(r"\bx\d+\b", text)
    assert list(model.feature_names_in_) == FEATURE_NAMES
    assert model.n_features_in_ == len(FEATURE_NAMES)


@pytest.mark.parametrize("model_name", ai.__all__)
def test_ndarray_displays_keep_x_fallback_and_predictions(
    model_name: str, named_data, named_models
) -> None:
    _, X, y = named_data
    ndarray_model = getattr(ai, model_name)().fit(X, y)
    named_model = named_models[model_name]

    assert "x0" in str(ndarray_model)
    np.testing.assert_allclose(ndarray_model.predict(X[:40]), named_model.predict(X[:40]))


@pytest.mark.parametrize("model_name", ai.__all__)
def test_dataframe_column_reordering_is_aligned(model_name: str, named_data, named_models) -> None:
    frame, _, _ = named_data
    model = named_models[model_name]
    reversed_frame = frame.loc[:, list(reversed(FEATURE_NAMES))]

    np.testing.assert_allclose(model.predict(frame), model.predict(reversed_frame))
    missing = frame.drop(columns=[FEATURE_NAMES[-1]])
    with pytest.raises(ValueError, match=FEATURE_NAMES[-1]):
        model.predict(missing)


@pytest.mark.parametrize("model_name", ai.__all__)
def test_explicit_display_names_preserve_dataframe_alignment(model_name: str, named_data) -> None:
    frame, _, y = named_data
    frame = frame.iloc[:120]
    y = y[:120]
    explicit_names = [f"display {i}" for i in range(frame.shape[1])]
    model = getattr(ai, model_name)(feature_names=explicit_names).fit(frame, y)
    reversed_frame = frame.loc[:, list(reversed(FEATURE_NAMES))]

    np.testing.assert_allclose(model.predict(frame), model.predict(reversed_frame))
    assert any(name in str(model) for name in explicit_names)
    missing = frame.drop(columns=[FEATURE_NAMES[-1]])
    with pytest.raises(ValueError, match=FEATURE_NAMES[-1]):
        model.predict(missing)
