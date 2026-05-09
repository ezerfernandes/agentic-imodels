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
    ),
}

HONEST_MODELS = tuple(
    name for name, info in MODEL_REGISTRY.items() if info.category == "honest"
)

DECOUPLED_MODELS = tuple(
    name
    for name, info in MODEL_REGISTRY.items()
    if info.category == "display-predict decoupled"
)


def get_model_info(name: str) -> ModelInfo:
    """Return metadata for a public estimator by class name."""

    return MODEL_REGISTRY[name]
