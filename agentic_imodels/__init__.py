"""agentic-imodels — interpretable tabular regressors discovered via an agentic loop."""

from importlib.metadata import version as _pkg_version

__all__ = [
    "HingeEBMRegressor",
    "HybridGAM",
    "SmartAdditiveRegressor",
    "HingeGAMRegressor",
    "TeacherStudentRuleSplineRegressor",
    "DualPathSparseSymbolicRegressor",
    "SparseSignedBasisPursuitRegressor",
    "DistilledTreeBlendAtlasRegressor",
    "WinsorizedSparseOLSRegressor",
    "TinyDTDepth2Regressor",
]

from .distilled_tree_blend_atlas import DistilledTreeBlendAtlasRegressor
from .dual_path_sparse_symbolic import DualPathSparseSymbolicRegressor
from .hinge_ebm import HingeEBMRegressor
from .hinge_gam import HingeGAMRegressor
from .hybrid_gam import HybridGAM
from .registry import (
    DECOUPLED_MODELS,  # noqa: F401
    HONEST_MODELS,  # noqa: F401
    MODEL_REGISTRY,  # noqa: F401
    ModelInfo,  # noqa: F401
    get_model_info,  # noqa: F401
)
from .smooth_additive_gam import SmartAdditiveRegressor
from .sparse_signed_basis_pursuit import SparseSignedBasisPursuitRegressor
from .teacher_student_rule_spline import TeacherStudentRuleSplineRegressor
from .tiny_dt import TinyDTDepth2Regressor
from .winsorized_sparse_ols import WinsorizedSparseOLSRegressor

try:
    __version__ = _pkg_version("agentic-imodels")
except Exception:
    __version__ = "0.0.0"
