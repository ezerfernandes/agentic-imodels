from __future__ import annotations

import inspect
import pickle
import re
import time

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error

import agentic_imodels as ai
from agentic_imodels import (
    DistilledTreeBlendAtlasRegressor,
    DualPathSparseSymbolicRegressor,
    HingeGAMRegressor,
    HybridGAM,
    SmartAdditiveRegressor,
    TeacherStudentRuleSplineRegressor,
    TinyDTDepth2Regressor,
)
from agentic_imodels.registry import DECOUPLED_MODELS, HONEST_MODELS, MODEL_REGISTRY


@pytest.fixture(scope="module")
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    return make_regression(
        n_samples=300,
        n_features=6,
        n_informative=3,
        noise=0.5,
        random_state=0,
    )


@pytest.fixture(scope="module")
def housing_data() -> tuple[np.ndarray, np.ndarray]:
    try:
        housing = fetch_california_housing()
    except Exception as exc:
        pytest.skip(f"California housing unavailable: {exc}")
    return housing.data[:1500], housing.target[:1500]


# Each entry identifies the display lines that establish active and zero-effect
# features. Keeping these patterns explicit makes the operational faithfulness
# check small and reviewable as displays evolve.
_HONEST_DISPLAY_PATTERNS = {
    "SparseSignedBasisPursuitRegressor": {
        "active": r"^Active features:.*$",
        "zero": r"^Zero-contribution features:.*$",
    },
    "HingeGAMRegressor": {
        "active": r"^\s+x\d+:",
        "zero": r"^Features with zero coefficients \(excluded\):.*$",
    },
    "SmartAdditiveRegressor": {
        "active": r"^\s+x\d+:",
        "zero": r"^Features with zero coefficients \(excluded\):.*$",
    },
    "WinsorizedSparseOLSRegressor": {
        "active": r"^\s+x\d+:",
        "zero": r"^Features excluded \(zero effect\):.*$",
    },
    "TinyDTDepth2Regressor": {
        "active": r"^\|--- x\d+\s+(?:<=|>)",
        "zero": None,
    },
}


def _features_on_matching_lines(text: str, line_pattern: str | None) -> set[int]:
    if line_pattern is None:
        return set()
    return {
        int(match)
        for line in text.splitlines()
        if re.search(line_pattern, line)
        for match in re.findall(r"\bx(\d+)\b", line)
    }


def test_tiny_dt_is_a_depth_two_tree_and_exposes_constructor_params() -> None:
    X, y = make_regression(n_samples=200, n_features=5, random_state=0)
    model = TinyDTDepth2Regressor().fit(X, y)

    assert hasattr(model, "tree_")
    assert not hasattr(model, "coef_")
    assert model.tree_.get_depth() <= 2
    assert model.tree_.get_n_leaves() <= 4
    assert set(model.get_params()) == {
        "max_depth",
        "min_samples_leaf",
        "random_state",
        "feature_names",
    }

    text = str(model)
    print(text)
    assert "Decision Tree" in text
    assert "|---" in text
    assert "BayesianRidge" not in text


def test_tiny_dt_registry_metrics_are_marked_unmeasured_after_fix() -> None:
    assert MODEL_REGISTRY["TinyDTDepth2Regressor"].metrics_status == "unmeasured-after-fix"
    assert all(
        info.metrics_status
        == (
            "unmeasured-after-fix"
            if name in {"TinyDTDepth2Regressor", "DistilledTreeBlendAtlasRegressor"}
            else "measured"
        )
        for name, info in MODEL_REGISTRY.items()
    )


def test_tiny_dt_predict_matches_tree_predictions() -> None:
    X, y = make_regression(n_samples=200, n_features=5, random_state=0)
    model = TinyDTDepth2Regressor().fit(X, y)

    np.testing.assert_allclose(model.predict(X[:20]), model.tree_.predict(X[:20]))


def test_atlas_display_uses_data_driven_partial_dependence_card() -> None:
    X, y = make_regression(n_samples=120, n_features=6, random_state=0)
    model = DistilledTreeBlendAtlasRegressor(random_state=0).fit(X, y)
    text = str(model)

    for forbidden in (
        "canonical_",
        "probe_",
        "compactness_answer",
        "counterfactual_answer_policy",
    ):
        assert forbidden not in text

    ranking_line = next(
        line for line in text.splitlines() if line.startswith("feature_ranking_from_fit:")
    )
    top_feature = ranking_line.split(":", 1)[1].split(",", 1)[0].strip()
    pd_line = next(line for line in text.splitlines() if line.startswith(f"pd_card {top_feature}:"))
    values = {
        label: float(value)
        for label, value in (part.split("=") for part in pd_line.split(":", 1)[1].split())
    }
    median_row = model.feature_quantiles_[1].reshape(1, -1)
    np.testing.assert_allclose(values["p50"], model.predict(median_row)[0], atol=1e-6)


@pytest.mark.parametrize(
    "model_cls", [TeacherStudentRuleSplineRegressor, DualPathSparseSymbolicRegressor]
)
def test_teacher_student_predict_is_invariant_to_batch_size(model_cls) -> None:
    X, y = make_regression(n_samples=100, n_features=6, random_state=0)
    model = model_cls().fit(X, y)
    batch = model.predict(X[:40])
    rowwise = np.concatenate([model.predict(X[i : i + 1]) for i in range(40)])

    np.testing.assert_allclose(batch, rowwise)


@pytest.mark.parametrize(
    "model_cls", [TeacherStudentRuleSplineRegressor, DualPathSparseSymbolicRegressor]
)
def test_teacher_student_predict_path_flag_and_display(model_cls) -> None:
    X, y = make_regression(n_samples=100, n_features=6, random_state=0)
    student_model = model_cls(predict_with="student").fit(X, y)
    teacher_model = model_cls(predict_with="teacher").fit(X, y)

    np.testing.assert_allclose(student_model.predict(X[:10]), student_model.predict_student(X[:10]))
    np.testing.assert_allclose(teacher_model.predict(X[:10]), teacher_model.predict_teacher(X[:10]))
    for forbidden in ("single-row", "Question-answering protocol", "Answer format requirement"):
        assert forbidden not in str(teacher_model)
    assert "Student-vs-teacher validation RMSE:" in str(teacher_model)

    with pytest.warns(DeprecationWarning):
        model_cls(symbolic_n_rows=5).fit(X, y)


@pytest.mark.parametrize("model_name", ai.__all__)
def test_all_public_models_predict_invariant_to_batch_size(model_name: str) -> None:
    X, y = make_regression(n_samples=300, n_features=6, random_state=0)
    model = getattr(ai, model_name)().fit(X, y)
    batch = model.predict(X[:40])
    rowwise = np.concatenate([model.predict(X[i : i + 1]) for i in range(40)])
    max_diff = float(np.max(np.abs(batch - rowwise)))
    print(f"{model_name}: max batch/rowwise diff={max_diff:.3e}")

    assert np.allclose(batch, rowwise)
    assert "Ridge Regression" not in str(model).splitlines()[0]


def test_gam_displays_name_the_fitted_model() -> None:
    X, y = make_regression(n_samples=300, n_features=6, random_state=0)
    hinge = HingeGAMRegressor().fit(X, y)
    smart = SmartAdditiveRegressor().fit(X, y)
    hybrid = HybridGAM().fit(X, y)

    hinge_header = str(hinge).splitlines()[0]
    assert "Lasso" in hinge_header
    alpha = float(hinge_header.split("alpha=", 1)[1].split(",", 1)[0])
    assert alpha == pytest.approx(hinge.lasso_.alpha_, rel=5e-4)
    assert "stumps" in str(smart).splitlines()[0]
    assert "RandomForest residual" in str(hybrid)


@pytest.mark.parametrize("model_name", ai.__all__)
def test_public_models_are_sklearn_compatible_and_deterministic(
    model_name: str, regression_data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = regression_data
    cls = getattr(ai, model_name)
    model = cls()

    assert set(model.get_params()) == set(inspect.signature(cls).parameters)
    clone(model)

    fitted = model.fit(X, y)
    repeat = cls().fit(X, y)
    np.testing.assert_allclose(fitted.predict(X[:40]), repeat.predict(X[:40]))

    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_allclose(fitted.predict(X[:40]), restored.predict(X[:40]))


@pytest.mark.parametrize("model_name", ai.__all__)
def test_public_models_raise_not_fitted_errors(
    model_name: str, regression_data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, _ = regression_data
    model = getattr(ai, model_name)()

    with pytest.raises(NotFittedError):
        model.predict(X[:1])
    with pytest.raises(NotFittedError):
        str(model)


@pytest.mark.parametrize("model_name", HONEST_MODELS)
def test_honest_displays_match_prediction_dependencies(
    model_name: str, regression_data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = regression_data
    model = getattr(ai, model_name)().fit(X, y)
    text = str(model)
    patterns = _HONEST_DISPLAY_PATTERNS[model_name]
    active = _features_on_matching_lines(text, patterns["active"])
    zero = _features_on_matching_lines(text, patterns["zero"])

    assert active, f"no active features parsed from {model_name} display"
    assert active.isdisjoint(zero)
    baseline = model.predict(X)
    for feature in active:
        perturbed = X.copy()
        perturbed[:, feature] += np.std(X[:, feature])
        changed = np.max(np.abs(model.predict(perturbed) - baseline))
        assert changed > 1e-9, f"{model_name} active x{feature} has no prediction effect"
    for feature in zero:
        perturbed = X.copy()
        perturbed[:, feature] += np.std(X[:, feature])
        unchanged = np.max(np.abs(model.predict(perturbed) - baseline))
        assert unchanged <= 1e-9, f"{model_name} zero-effect x{feature} changes prediction"


@pytest.mark.parametrize("model_name", DECOUPLED_MODELS)
def test_decoupled_displays_disclose_prediction_path(
    model_name: str, regression_data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = regression_data
    text = str(getattr(ai, model_name)().fit(X, y)).lower()
    assert any(word in text for word in ("hidden", "teacher", "residual", "corrector", "blend"))


@pytest.mark.parametrize("model_name", ai.__all__)
def test_public_models_fit_california_housing_and_log_metrics(
    model_name: str, housing_data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = housing_data
    cls = getattr(ai, model_name)
    started = time.perf_counter()
    model = cls().fit(X, y)
    fit_seconds = time.perf_counter() - started
    predictions = model.predict(X)
    display = str(model)
    rmse = mean_squared_error(y, predictions) ** 0.5

    assert np.isfinite(predictions).all()
    assert np.isfinite(rmse)
    assert display.strip()
    print(
        f"{model_name}: fit_seconds={fit_seconds:.2f}, "
        f"housing_rmse={rmse:.4f}, display_chars={len(display)}"
    )


@pytest.mark.parametrize("model_name", ai.__all__)
def test_public_displays_have_no_test_gaming_text(
    model_name: str, regression_data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = regression_data
    text = str(getattr(ai, model_name)().fit(X, y))
    for forbidden in (
        "canonical_",
        "probe_",
        "Answer protocol",
        "Question-answering protocol",
        "Answer format requirement",
        "single-row",
    ):
        assert forbidden not in text
