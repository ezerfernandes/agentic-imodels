#!/usr/bin/env python3
"""Render registry literals from the archived combined-results CSV."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


MODEL_CONFIG = {
    "HingeEBMRegressor": ("HingeEBM_5bag", "agentic_imodels.hinge_ebm", "display-predict decoupled", "HingeEBM_5bag", "Lasso on hinge basis plus hidden EBM residual corrector.", "measured", "predict adds a hidden EBM residual corrector to the displayed hinge formula."),
    "DistilledTreeBlendAtlasRegressor": ("DistilledTreeBlendAtlasApr18aa", "agentic_imodels.distilled_tree_blend_atlas", "display-predict decoupled", "DistilledTreeBlendAtlas_v1", "Ridge student distilled from GBM and RF teachers, displayed as an atlas card.", "unmeasured-after-fix", "predict returns the calibrated GBM/RF/student blend, not the displayed sparse equation."),
    "DualPathSparseSymbolicRegressor": ("DualPathSparseSymbolic_v2", "agentic_imodels.dual_path_sparse_symbolic", "display-predict decoupled", "DualPathSparseSymbolic_v2", "Sparse symbolic display with a blended GBM/RF/Ridge predictor.", "measured", "predict uses the teacher ensemble by default; pass predict_with='student' to predict with the displayed equation."),
    "HybridGAM": ("HybridGAM_v9", "agentic_imodels.hybrid_gam", "display-predict decoupled", "HybridGAM_v9", "Smart additive GAM display plus hidden random-forest residual corrector.", "measured", "predict adds a shrunk random-forest residual correction to the displayed additive model."),
    "TeacherStudentRuleSplineRegressor": ("TeacherStudentRuleSpline_v1", "agentic_imodels.teacher_student_rule_spline", "display-predict decoupled", "TeacherStudentRuleSpline_v1", "GBM teacher with sparse symbolic student over rules, splines, and interactions.", "measured", "predict uses the teacher ensemble by default; pass predict_with='student' to predict with the displayed equation."),
    "SparseSignedBasisPursuitRegressor": ("SparseSignedBasisPursuit_v1", "agentic_imodels.sparse_signed_basis_pursuit", "honest", "SparseSignedBasisPursuit_v1", "Forward-selected signed basis with ridge refit and rounded coefficients.", "measured", "predict computes exactly the displayed form."),
    "HingeGAMRegressor": ("HingeGAM_10bp", "agentic_imodels.hinge_gam", "honest", "HingeGAM_10bp", "Pure Lasso on hinge features with ten breakpoints.", "measured", "predict computes exactly the displayed form."),
    "WinsorizedSparseOLSRegressor": ("WinsorizedSparseOLS", "agentic_imodels.winsorized_sparse_ols", "honest", "WinsorizedSparseOLS", "Winsorized features, LassoCV selection, and OLS refit.", "measured", "predict computes exactly the displayed form."),
    "TinyDTDepth2Regressor": ("TinyDTDepth2_v1", "agentic_imodels.tiny_dt", "honest", "TinyDTDepth2_v1", "Depth-2 decision tree with four leaves.", "unmeasured-after-fix", "predict computes exactly the displayed form."),
    "SmartAdditiveRegressor": ("SmoothGAM_msl3", "agentic_imodels.smooth_additive_gam", "honest", "SmoothGAM_msl3", "Laplacian-smoothed boosted stumps rendered as linear or short piecewise terms.", "measured", "predict computes exactly the displayed form."),
}


def rounded(value: str, places: int) -> str:
    """Round decimal CSV values predictably instead of using binary floats."""

    quantum = Decimal(1).scaleb(-places)
    return format(Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP), f".{places}f")


def load_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open(newline="") as handle:
        rows = {row["model_name"]: row for row in csv.DictReader(handle)}
    missing = [config[0] for config in MODEL_CONFIG.values() if config[0] not in rows]
    if missing:
        raise SystemExit(f"Missing model rows in {csv_path}: {', '.join(missing)}")
    return rows


def render(csv_path: Path) -> str:
    rows = load_rows(csv_path)
    output = ["MODEL_REGISTRY: dict[str, ModelInfo] = {"]
    for name, config in MODEL_CONFIG.items():
        shorthand_key, module, category, shorthand, summary, metrics_status, predict_notes = config
        row = rows[shorthand_key]
        model_file = row["model_file"]
        provenance_kind = "success" if "/success/" in model_file else "failure"
        provenance = f"{provenance_kind} @ {row['experiment']}"
        output.extend(
            [
                f'    "{name}": ModelInfo(',
                f'        name="{name}",',
                f'        module="{module}",',
                f'        shorthand="{shorthand}",',
                f'        rank={rounded(row["mean_rank_global"], 1)},',
                f'        dev_interpretability={rounded(row["dev_interp_score"], 3)},',
                f'        test_interpretability={rounded(row["test_interp_score"], 3)},',
                f'        category="{category}",',
                f'        summary="{summary}",',
                f'        provenance="{provenance}",',
                f'        metrics_status="{metrics_status}",',
                f'        predict_notes="{predict_notes}",',
                "    ),",
            ]
        )
    output.append("}")
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "result_libs" / "combined_results.csv",
        help="combined results CSV (default: %(default)s)",
    )
    args = parser.parse_args()
    print(render(args.csv))


if __name__ == "__main__":
    main()
