"""Structured metadata for the public agentic-imodels estimators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a public estimator."""

    name: str
    module: str
    shorthand: str
    rank: float
    dev_interpretability: float
    test_interpretability: float
    category: str
    summary: str
    provenance: str
    metrics_status: str
    predict_notes: str


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "HingeEBMRegressor": ModelInfo(
        name="HingeEBMRegressor",
        module="agentic_imodels.hinge_ebm",
        shorthand="HingeEBM_5bag",
        rank=108.2,
        dev_interpretability=0.651,
        test_interpretability=0.707,
        category="display-predict decoupled",
        summary="Lasso on hinge basis plus hidden EBM residual corrector.",
        provenance="success @ apr9-claude-effort=medium-main-result",
        metrics_status="measured",
        predict_notes=(
            "predict adds a hidden EBM residual corrector to the displayed hinge formula."
        ),
    ),
    "DistilledTreeBlendAtlasRegressor": ModelInfo(
        name="DistilledTreeBlendAtlasRegressor",
        module="agentic_imodels.distilled_tree_blend_atlas",
        shorthand="DistilledTreeBlendAtlas_v1",
        rank=139.7,
        dev_interpretability=1.000,
        test_interpretability=0.707,
        category="display-predict decoupled",
        summary="Ridge student distilled from GBM and RF teachers, displayed as an atlas card.",
        provenance="success @ apr19-codex-5.3-effort=xhigh",
        metrics_status="unmeasured-after-fix",
        predict_notes=(
            "predict returns the calibrated GBM/RF/student blend, not the displayed "
            "sparse equation."
        ),
    ),
    "DualPathSparseSymbolicRegressor": ModelInfo(
        name="DualPathSparseSymbolicRegressor",
        module="agentic_imodels.dual_path_sparse_symbolic",
        shorthand="DualPathSparseSymbolic_v2",
        rank=163.5,
        dev_interpretability=0.698,
        test_interpretability=0.713,
        category="display-predict decoupled",
        summary="Sparse symbolic display with a blended GBM/RF/Ridge predictor.",
        provenance="failure @ apr17-codex-5.3-effort=high",
        metrics_status="measured",
        predict_notes=(
            "predict uses the teacher ensemble by default; pass predict_with='student' to "
            "predict with the displayed equation."
        ),
    ),
    "HybridGAM": ModelInfo(
        name="HybridGAM",
        module="agentic_imodels.hybrid_gam",
        shorthand="HybridGAM_v9",
        rank=163.8,
        dev_interpretability=0.721,
        test_interpretability=0.675,
        category="display-predict decoupled",
        summary="Smart additive GAM display plus hidden random-forest residual corrector.",
        provenance="failure @ apr20-claude-4.7-effort=medium-rerun4",
        metrics_status="measured",
        predict_notes=(
            "predict adds a shrunk random-forest residual correction to the displayed "
            "additive model."
        ),
    ),
    "TeacherStudentRuleSplineRegressor": ModelInfo(
        name="TeacherStudentRuleSplineRegressor",
        module="agentic_imodels.teacher_student_rule_spline",
        shorthand="TeacherStudentRuleSpline_v1",
        rank=204.0,
        dev_interpretability=0.605,
        test_interpretability=0.803,
        category="display-predict decoupled",
        summary="GBM teacher with sparse symbolic student over rules, splines, and interactions.",
        provenance="failure @ apr17-codex-5.3-effort=high",
        metrics_status="measured",
        predict_notes=(
            "predict uses the teacher ensemble by default; pass predict_with='student' to "
            "predict with the displayed equation."
        ),
    ),
    "SparseSignedBasisPursuitRegressor": ModelInfo(
        name="SparseSignedBasisPursuitRegressor",
        module="agentic_imodels.sparse_signed_basis_pursuit",
        shorthand="SparseSignedBasisPursuit_v1",
        rank=272.7,
        dev_interpretability=0.674,
        test_interpretability=0.758,
        category="honest",
        summary="Forward-selected signed basis with ridge refit and rounded coefficients.",
        provenance="success @ apr17-codex-5.3-effort=high",
        metrics_status="measured",
        predict_notes="predict computes exactly the displayed form.",
    ),
    "HingeGAMRegressor": ModelInfo(
        name="HingeGAMRegressor",
        module="agentic_imodels.hinge_gam",
        shorthand="HingeGAM_10bp",
        rank=280.2,
        dev_interpretability=0.558,
        test_interpretability=0.783,
        category="honest",
        summary="Pure Lasso on hinge features with ten breakpoints.",
        provenance="failure @ apr9-claude-effort=medium-main-result",
        metrics_status="unmeasured-after-fix",
        predict_notes="predict computes exactly the displayed form.",
    ),
    "WinsorizedSparseOLSRegressor": ModelInfo(
        name="WinsorizedSparseOLSRegressor",
        module="agentic_imodels.winsorized_sparse_ols",
        shorthand="WinsorizedSparseOLS",
        rank=326.9,
        dev_interpretability=0.651,
        test_interpretability=0.726,
        category="honest",
        summary="Winsorized features, LassoCV selection, and OLS refit.",
        provenance="failure @ apr19-claude-4.7-effort=medium-rerun2",
        metrics_status="measured",
        predict_notes="predict computes exactly the displayed form.",
    ),
    "TinyDTDepth2Regressor": ModelInfo(
        name="TinyDTDepth2Regressor",
        module="agentic_imodels.tiny_dt",
        shorthand="TinyDTDepth2_v1",
        rank=334.0,
        dev_interpretability=0.674,
        test_interpretability=0.713,
        category="honest",
        summary="Depth-2 decision tree with four leaves.",
        provenance="failure @ apr19-claude-4.7-effort=medium-rerun3",
        metrics_status="unmeasured-after-fix",
        predict_notes="predict computes exactly the displayed form.",
    ),
    "SmartAdditiveRegressor": ModelInfo(
        name="SmartAdditiveRegressor",
        module="agentic_imodels.smooth_additive_gam",
        shorthand="SmoothGAM_msl3",
        rank=354.3,
        dev_interpretability=0.744,
        test_interpretability=0.733,
        category="honest",
        summary="Laplacian-smoothed boosted stumps rendered as linear or short piecewise terms.",
        provenance="failure @ apr9-claude-effort=medium-main-result",
        metrics_status="measured",
        predict_notes="predict computes exactly the displayed form.",
    ),
}

HONEST_MODELS = tuple(name for name, info in MODEL_REGISTRY.items() if info.category == "honest")

DECOUPLED_MODELS = tuple(
    name for name, info in MODEL_REGISTRY.items() if info.category == "display-predict decoupled"
)


def get_model_info(name: str) -> ModelInfo:
    """Return metadata for a public estimator by class name."""

    return MODEL_REGISTRY[name]
