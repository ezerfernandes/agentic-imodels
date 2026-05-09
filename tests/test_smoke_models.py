from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression

import agentic_imodels as ai


@pytest.fixture(scope="module")
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    return make_regression(
        n_samples=80,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42,
    )


@pytest.mark.parametrize("model_name", ai.__all__)
def test_public_model_fit_predict_and_stringify(
    model_name: str, regression_data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = regression_data
    cls = getattr(ai, model_name)

    model = cls()
    model.fit(X, y)

    y_hat = model.predict(X[:7])
    text = str(model)

    assert y_hat.shape == (7,)
    assert np.isfinite(y_hat).all()
    assert isinstance(text, str)
    assert len(text.strip()) > 20
