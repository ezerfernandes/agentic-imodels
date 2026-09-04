from __future__ import annotations

import importlib
import importlib.metadata

import numpy as np
import pytest

import agentic_imodels
import agentic_imodels._names as names_module
import agentic_imodels.distilled_tree_blend_atlas as atlas_module
import agentic_imodels.dual_path_sparse_symbolic as dual_module
import agentic_imodels.sparse_signed_basis_pursuit as sparse_module
import agentic_imodels.teacher_student_rule_spline as teacher_module
from agentic_imodels.distilled_tree_blend_atlas import DistilledTreeBlendAtlasRegressor
from agentic_imodels.dual_path_sparse_symbolic import DualPathSparseSymbolicRegressor
from agentic_imodels.hinge_ebm import HingeEBMRegressor
from agentic_imodels.hinge_gam import HingeGAMRegressor
from agentic_imodels.hybrid_gam import HybridGAM, SmartAdditiveGAM
from agentic_imodels.smooth_additive_gam import SmartAdditiveRegressor
from agentic_imodels.sparse_signed_basis_pursuit import SparseSignedBasisPursuitRegressor
from agentic_imodels.teacher_student_rule_spline import TeacherStudentRuleSplineRegressor


class ColumnArray:
    """Small DataFrame-like input for exercising name alignment without pandas."""

    def __init__(self, values, columns):
        self.values = np.asarray(values)
        self.columns = list(columns)

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)


def _raise_linalg_error(*args, **kwargs):
    raise np.linalg.LinAlgError("forced solver failure")


def test_shared_name_helpers_cover_scalar_alignment_and_validation_fallback(monkeypatch):
    scalar, scalar_names = names_module.to_array(np.asarray(3.0))
    assert scalar.shape == ()
    assert scalar_names == []

    frame = ColumnArray([[1.0, 2.0]], ["left", "right"])
    np.testing.assert_allclose(
        names_module.align_feature_names(frame, ["right", "left"]), [[2.0, 1.0]]
    )

    with pytest.raises(ValueError, match="missing fitted feature columns: right"):
        names_module.align_feature_names(
            ColumnArray([[1.0, 3.0]], ["left", "extra"]),
            ["left", "right"],
            ["old_left", "old_right"],
        )
    with pytest.raises(ValueError, match="missing fitted feature columns: right"):
        names_module.align_feature_names(
            ColumnArray([[1.0, 3.0]], ["left", "extra"]),
            ["left", "right"],
            ["other_left", "other_right"],
        )

    monkeypatch.setattr(names_module, "_sklearn_validate_data", None)
    X_fit, y_fit = names_module.validate_fit_data(object(), [[1.0], [2.0]], [1.0, 2.0])
    assert X_fit.shape == (2, 1)
    np.testing.assert_allclose(y_fit, [1.0, 2.0])
    X_pred = names_module.validate_predict_data(object(), [[3.0]])
    assert X_pred.shape == (1, 1)


def test_package_version_fallback_is_safe(monkeypatch):
    original_version = importlib.metadata.version

    def missing_distribution(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing_distribution)
    reloaded = importlib.reload(agentic_imodels)
    assert reloaded.__version__ == "0.0.0"
    monkeypatch.setattr(importlib.metadata, "version", original_version)
    importlib.reload(agentic_imodels)


def test_atlas_helpers_fallbacks_and_zero_feature_model(monkeypatch):
    y = np.array([1.0, 2.0, 4.0])
    model = DistilledTreeBlendAtlasRegressor(student_alpha_grid=(np.nan, -1.0))

    intercept, coef = model._ridge_with_intercept(np.empty((3, 0)), y, 0.1)
    assert intercept == pytest.approx(np.mean(y))
    assert coef.shape == (0,)
    np.testing.assert_allclose(model._ridge_predict(np.empty((2, 0)), 4.0, np.empty(0)), [4.0, 4.0])
    alpha, fallback_intercept, fallback_coef = model._fit_student(np.ones((3, 1)), y, None, None)
    assert alpha == pytest.approx(0.1)
    assert fallback_intercept == pytest.approx(np.mean(y))
    assert fallback_coef.shape == (1,)
    assert model._solve_nonnegative_weights(np.empty((0, 0)), y).tolist() == [1.0]
    assert model._term_to_str(("unknown", 0)) == "('unknown', 0)"

    monkeypatch.setattr(atlas_module.np.linalg, "solve", _raise_linalg_error)
    fallback_intercept, fallback_coef = model._ridge_with_intercept(
        np.array([[0.0], [1.0], [2.0]]), y, 0.1
    )
    assert np.isfinite(fallback_intercept)
    assert fallback_coef.shape == (1,)
    weights = model._solve_nonnegative_weights(np.zeros((3, 2)), np.zeros(3))
    np.testing.assert_allclose(weights, [0.5, 0.5])

    def fake_fit_data(estimator, X, target):
        return np.empty((len(target), 0)), np.asarray(target, dtype=float)

    monkeypatch.setattr(atlas_module, "validate_fit_data", fake_fit_data)
    monkeypatch.setattr(atlas_module, "validate_predict_data", lambda estimator, X: X)
    zero_model = DistilledTreeBlendAtlasRegressor().fit(np.empty((3, 0)), y)
    np.testing.assert_allclose(zero_model.predict(np.asarray([])), np.mean(y))
    with pytest.raises(ValueError, match="Expected 0 features, got 1"):
        zero_model.predict(np.ones((1, 1)))

    linear_only = DistilledTreeBlendAtlasRegressor()
    linear_only.n_features_in_ = 1
    linear_only.feature_names_in_ = np.asarray(["x0"], dtype=object)
    linear_only.input_feature_names_in_ = None
    linear_only.x_mean_ = np.zeros(1)
    linear_only.x_scale_ = np.ones(1)
    linear_only.student_intercept_ = 1.0
    linear_only.student_coef_ = np.asarray([2.0])
    linear_only.blend_weights_ = np.asarray([0.0, 0.0, 1.0])
    linear_only.gbm_ = None
    linear_only.rf_ = None
    linear_only.intercept_ = 0.0
    linear_only.coef_ = np.asarray([2.0])
    linear_only.terms_ = [("linear", 0)]
    linear_only.feature_importance_ = np.asarray([1.0])
    linear_only.training_rmse_ = 0.0
    linear_only.calibration_intercept_ = 0.0
    linear_only.calibration_slope_ = 1.0
    np.testing.assert_allclose(linear_only.predict(np.asarray([[3.0]])), [7.0])


def test_atlas_fit_guards_and_constant_calibration(monkeypatch):
    def fake_fit_data(estimator, X, target):
        return np.asarray([1.0, 2.0]), np.asarray([3.0])

    monkeypatch.setattr(atlas_module, "validate_fit_data", fake_fit_data)
    with pytest.raises(ValueError, match="same number of rows"):
        DistilledTreeBlendAtlasRegressor().fit(np.asarray([1.0, 2.0]), [3.0])
    monkeypatch.undo()

    X = np.ones((10, 1))
    target = np.full(10, 3.0)
    constant = DistilledTreeBlendAtlasRegressor(
        gbm_estimators_base=60,
        gbm_estimators_cap=60,
        rf_estimators_base=60,
        rf_estimators_cap=60,
        random_state=0,
    ).fit(X, target)
    assert constant.calibration_slope_ == pytest.approx(1.0)
    assert constant.terms_ == [("linear", 0)]
    np.testing.assert_allclose(constant.feature_importance_, [0.0])


def test_hinge_models_select_features_and_handle_constant_display():
    X = np.column_stack([np.linspace(-1.0, 1.0, 24), np.linspace(2.0, 5.0, 24), np.ones(24)])
    y = 2.0 * X[:, 0] - X[:, 1]

    ebm = HingeEBMRegressor(n_knots=1, max_input_features=2, ebm_outer_bags=1).fit(X, y)
    assert len(ebm.selected_) == 2
    assert ebm.predict(X[:3]).shape == (3,)

    hinge = HingeGAMRegressor(n_knots=1, max_input_features=2).fit(X, y)
    assert len(hinge.selected_) == 2
    assert hinge.predict(X[:3]).shape == (3,)

    mostly_constant = np.r_[np.zeros(19), 1.0].reshape(-1, 1)
    degenerate = HingeGAMRegressor(n_knots=2).fit(mostly_constant, mostly_constant[:, 0])
    assert degenerate.linear_approx_[0][0] == pytest.approx(0.0)

    constant = HingeGAMRegressor()
    constant.feature_names_in_ = np.asarray(["x0"], dtype=object)
    constant.shape_functions_ = {}
    constant.feature_importances_ = np.asarray([0.0])
    constant.intercept_ = 2.5
    assert str(constant) == "Constant model: y = 2.5000"


def test_hybrid_residual_types_and_smart_gam_edge_paths():
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = X[:, 0] ** 2
    gbm_hybrid = HybridGAM(
        gam_n_rounds=1,
        gam_min_leaf=2,
        n_residual_trees=2,
        residual_type="gbm",
    ).fit(X, y)
    assert type(gbm_hybrid.residual_gbm_).__name__ == "GradientBoostingRegressor"
    assert gbm_hybrid.predict(X[:2]).shape == (2,)

    with pytest.raises(ValueError, match="Unknown residual_type"):
        HybridGAM(
            gam_n_rounds=1,
            gam_min_leaf=2,
            n_residual_trees=2,
            residual_type="unknown",
        ).fit(X, y)

    constant = SmartAdditiveGAM(n_rounds=1, min_samples_leaf=4).fit(np.ones((6, 1)), np.arange(6.0))
    assert constant.shape_functions_ == {}
    assert "Smart Additive GAM" in str(constant)

    capped = SmartAdditiveGAM(
        n_rounds=3, min_samples_leaf=2, max_thresholds_per_feature=1, n_smooth_passes=1
    ).fit(X, y)
    assert capped.n_rounds_ >= 1
    assert all(len(thresholds) <= 1 for thresholds, _ in capped.shape_functions_.values())

    piecewise = SmartAdditiveGAM()
    piecewise.intercept_ = 0.0
    piecewise.feature_importances_ = np.asarray([1.0])
    piecewise.shape_functions_ = {0: ([0.0], [1.0, 3.0])}
    piecewise.linear_approx_ = {0: (0.0, 0.0, 0.0)}
    np.testing.assert_allclose(piecewise.predict(np.asarray([[-1.0], [1.0]])), [1.0, 3.0])


def test_smooth_additive_degenerate_and_early_stop_paths():
    constant = SmartAdditiveRegressor(n_rounds=1, min_samples_leaf=3).fit(
        np.ones((8, 1)), np.arange(8.0)
    )
    assert constant.shape_functions_ == {}
    assert "Smart Additive GAM" in str(constant)
    assert constant._predict_from_stumps(np.ones((2, 1)), {0: []}).tolist() == [
        constant.intercept_,
        constant.intercept_,
    ]

    mixed_X = np.column_stack([np.linspace(-1.0, 1.0, 50), np.ones(50)])
    mixed = SmartAdditiveRegressor(n_rounds=1, min_samples_leaf=2).fit(mixed_X, mixed_X[:, 0] ** 2)
    assert mixed.n_rounds_ > 0

    X = np.linspace(-1.0, 1.0, 50).reshape(-1, 1)
    early = SmartAdditiveRegressor(n_rounds=210, learning_rate=0.1, min_samples_leaf=2).fit(
        X, np.ones(50)
    )
    assert early.n_rounds_ > 0


def test_sparse_basis_helpers_fallbacks_empty_model_and_all_specs(monkeypatch):
    y = np.array([1.0, 2.0, 4.0])
    model = SparseSignedBasisPursuitRegressor()
    intercept, coef = model._ridge_with_intercept(np.empty((3, 0)), y, 0.1)
    assert intercept == pytest.approx(np.mean(y))
    assert coef.shape == (0,)
    intercept, coef = model._ridge_with_intercept(np.ones((3, 1)), y, 0.1)
    assert np.isfinite(intercept)
    assert coef.shape == (1,)
    with pytest.raises(ValueError, match="Unknown basis spec"):
        model._eval_spec(np.ones((2, 1)), ("bad", 0))

    monkeypatch.setattr(sparse_module.np.linalg, "solve", _raise_linalg_error)
    intercept, coef = model._ridge_with_intercept(np.arange(3.0).reshape(-1, 1), y, 0.1)
    assert np.isfinite(intercept)
    assert coef.shape == (1,)

    X = np.ones((10, 1))
    constant = SparseSignedBasisPursuitRegressor().fit(X, np.arange(10.0))
    assert constant.term_specs_ == []
    assert "  (none)" in str(constant)

    varying = np.column_stack([np.linspace(-1.0, 1.0, 12), np.linspace(1.0, 2.0, 12)])
    no_steps = SparseSignedBasisPursuitRegressor(max_terms=0).fit(varying, varying[:, 0])
    assert no_steps.term_specs_


def test_dual_path_helpers_terms_and_empty_student(monkeypatch):
    y = np.array([1.0, 2.0, 4.0])
    model = DualPathSparseSymbolicRegressor()
    intercept, coef = model._solve_ridge_with_intercept(np.empty((3, 0)), y, 0.1)
    assert intercept == pytest.approx(np.mean(y))
    assert coef.shape == (0,)
    monkeypatch.setattr(dual_module.np.linalg, "solve", _raise_linalg_error)
    intercept, coef = model._solve_ridge_with_intercept(np.arange(3.0).reshape(-1, 1), y, 0.1)
    assert np.isfinite(intercept)
    assert coef.shape == (1,)

    X = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    for term, expected in [
        ({"type": "linear", "feature": 0}, [1.0, 3.0]),
        ({"type": "square", "feature": 1}, [4.0, 16.0]),
        ({"type": "hinge_pos", "feature": 0, "knot": 2.0}, [0.0, 1.0]),
        ({"type": "hinge_neg", "feature": 1, "knot": 3.0}, [1.0, 0.0]),
        ({"type": "interaction", "feature_a": 0, "feature_b": 1}, [2.0, 12.0]),
    ]:
        np.testing.assert_allclose(model._eval_term(X, term), expected)
    with pytest.raises(ValueError, match="Unknown term type"):
        model._eval_term(X, {"type": "bad"})
    assert model._design_matrix(X, []).shape == (2, 0)
    assert model._term_key({"type": "bad"}) == ("bad",)
    assert model._term_features({"type": "bad"}) == set()

    model.feature_names_in_ = np.asarray(["feature 0", "x1"], dtype=object)
    for term, expected in [
        ({"type": "square", "feature": 0}, "(`feature 0`^2)"),
        ({"type": "interaction", "feature_a": 0, "feature_b": 1}, "(`feature 0`*x1)"),
        ({"type": "bad"}, "0"),
    ]:
        assert model._term_text(term, 2) == expected

    empty_student = DualPathSparseSymbolicRegressor()._fit_student(np.empty((4, 0)), np.arange(4.0))
    assert empty_student["terms"] == []
    no_selection = DualPathSparseSymbolicRegressor(student_max_terms=0)._fit_student(
        np.arange(12.0).reshape(-1, 1), np.arange(12.0)
    )
    assert no_selection["terms"]


def test_dual_fit_guards_and_unknown_term_importance(monkeypatch):
    with pytest.raises(ValueError, match="predict_with"):
        DualPathSparseSymbolicRegressor(predict_with="bad").fit(np.ones((4, 1)), np.arange(4.0))

    model = DualPathSparseSymbolicRegressor()
    monkeypatch.setattr(model, "_fit_teacher", lambda X, y: None)
    monkeypatch.setattr(
        model,
        "_fit_student",
        lambda X, y: {
            "intercept": 1.0,
            "terms": [{"type": "unknown", "coef": 2.0}],
            "validation_rmse": 0.0,
        },
    )
    fitted = model.fit(np.ones((4, 1)), np.arange(4.0))
    assert fitted.selected_features_ == []


def test_teacher_student_helpers_terms_and_empty_selection(monkeypatch):
    y = np.array([1.0, 2.0, 4.0])
    model = TeacherStudentRuleSplineRegressor()
    intercept, coef = model._solve_ridge_with_intercept(np.empty((3, 0)), y, 0.1)
    assert intercept == pytest.approx(np.mean(y))
    assert coef.shape == (0,)
    monkeypatch.setattr(teacher_module.np.linalg, "solve", _raise_linalg_error)
    intercept, coef = model._solve_ridge_with_intercept(np.arange(3.0).reshape(-1, 1), y, 0.1)
    assert np.isfinite(intercept)
    assert coef.shape == (1,)

    assert model._screen_features(np.empty((4, 0)), y=[1.0, 2.0, 3.0, 4.0]) == []
    assert model._screen_features(np.ones((4, 2)), np.array([1.0, 2.0, 3.0, 4.0])) == [0, 1]
    selected = TeacherStudentRuleSplineRegressor(
        max_student_features=1, corr_screen_rel=1e9
    )._screen_features(np.arange(8.0).reshape(4, 2), np.array([0.0, 1.0, 2.0, 3.0]))
    assert len(selected) == 1

    X = np.asarray([[-2.0, 1.0], [2.0, 3.0]])
    terms = [
        {"type": "lin", "feature": 0},
        {"type": "sq", "feature": 0},
        {"type": "abs", "feature": 0},
        {"type": "hinge", "feature": 0, "knot": 0.0, "direction": 1},
        {"type": "hinge", "feature": 0, "knot": 0.0, "direction": -1},
        {"type": "step", "feature": 0, "knot": 0.0},
        {"type": "int", "a": 0, "b": 1},
        {"type": "gate", "gate": 0, "target": 1, "knot": 0.0},
    ]
    expected = [
        [-2.0, 2.0],
        [4.0, 4.0],
        [2.0, 2.0],
        [0.0, 2.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [-2.0, 6.0],
        [0.0, 3.0],
    ]
    for term, values in zip(terms, expected, strict=True):
        np.testing.assert_allclose(model._eval_term(X, term), values)
    with pytest.raises(ValueError, match="Unknown term type"):
        model._eval_term(X, {"type": "bad"})

    model.feature_names_in_ = np.asarray(["feature 0", "x1"], dtype=object)
    for term in terms + [{"type": "bad"}]:
        assert model._term_text(term, 2)
    deduped = model._dedupe_terms(
        terms + [terms[0], {"type": "other", "value": 1}, {"type": "other", "value": 1}]
    )
    assert len(deduped) == len(terms) + 1
    assert model._design_matrix(X, []).shape == (2, 0)

    candidates = model._build_candidates(np.asarray([[np.nan, 1.0], [1.0, 2.0]]), [0])
    assert all(np.isfinite(float(term.get("knot", 0.0))) for term in candidates)


def test_teacher_student_empty_and_pruned_student_paths(monkeypatch):
    empty = TeacherStudentRuleSplineRegressor()._fit_student(np.empty((4, 0)), np.arange(4.0))
    assert empty["terms"] == []

    model = TeacherStudentRuleSplineRegressor(max_student_terms=2, min_rel_gain=-1.0)
    monkeypatch.setattr(
        model, "_build_candidates", lambda X, screened: [{"type": "lin", "feature": 0}]
    )
    exhausted = model._fit_student(np.arange(8.0).reshape(-1, 1), np.arange(8.0))
    assert exhausted["terms"]

    pruned = TeacherStudentRuleSplineRegressor(max_student_terms=1, min_rel_gain=-1.0, coef_tol=1e9)
    monkeypatch.setattr(
        pruned, "_build_candidates", lambda X, screened: [{"type": "lin", "feature": 0}]
    )
    pruned_result = pruned._fit_student(np.arange(8.0).reshape(-1, 1), np.arange(8.0))
    assert pruned_result["terms"] == []

    with pytest.raises(ValueError, match="predict_with"):
        TeacherStudentRuleSplineRegressor(predict_with="bad").fit(np.ones((4, 1)), np.arange(4.0))

    model = TeacherStudentRuleSplineRegressor()
    monkeypatch.setattr(model, "_fit_teacher", lambda X, y: None)
    monkeypatch.setattr(
        model,
        "_fit_student",
        lambda X, y: {
            "intercept": 1.0,
            "terms": [{"type": "unknown", "coef": 2.0}],
            "validation_rmse": 0.0,
        },
    )
    fitted = model.fit(np.ones((4, 1)), np.arange(4.0))
    assert fitted.selected_features_ == []
