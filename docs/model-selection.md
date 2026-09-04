# Model Selection Guide

The package includes ten public regressors. Pick based on whether the printed model must be the actual computation or whether a readable display plus stronger hidden predictor is acceptable.

## Default Choices

| User need | Pick | Reason |
| --- | --- | --- |
| Best general default | `HingeEBMRegressor` | Best predictive rank in the curated set while keeping a compact display. |
| Printed formula must match prediction | `HingeGAMRegressor` or `SmartAdditiveRegressor` | Honest models with readable additive/threshold structure. |
| Maximum LLM-readability | `TeacherStudentRuleSplineRegressor` | Highest measured held-out interpretability score. |
| Smallest tree explanation | `TinyDTDepth2Regressor` | Depth-2 tree with four leaves. |
| Sparse linear story | `WinsorizedSparseOLSRegressor` | Outlier clipping, feature selection, OLS refit. |

## Full Table

Rank is mean global RMSE rank across development regression datasets. Lower is better. Test interp is the held-out LLM-graded interpretability pass rate. Higher is better.

| Class | Rank ↓ | Dev interp ↑ | Test interp ↑ | Category | Summary | Provenance | Metrics |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `HingeEBMRegressor` | 108.2 | 0.651 | 0.707 | display-predict decoupled | Lasso on hinge basis plus hidden EBM residual corrector. | success @ apr9-claude-effort=medium-main-result | measured |
| `DistilledTreeBlendAtlasRegressor` | 139.7 | 1.000 | 0.707 | display-predict decoupled | Ridge student distilled from GBM and RF teachers, displayed as an atlas card. | success @ apr19-codex-5.3-effort=xhigh | unmeasured-after-fix |
| `DualPathSparseSymbolicRegressor` | 163.5 | 0.698 | 0.713 | display-predict decoupled | Sparse symbolic display with a blended GBM/RF/Ridge predictor. | failure @ apr17-codex-5.3-effort=high | measured |
| `HybridGAM` | 163.8 | 0.721 | 0.675 | display-predict decoupled | Smart additive GAM display plus hidden random-forest residual corrector. | failure @ apr20-claude-4.7-effort=medium-rerun4 | measured |
| `TeacherStudentRuleSplineRegressor` | 204.0 | 0.605 | 0.803 | display-predict decoupled | GBM teacher with sparse symbolic student over rules, splines, and interactions. | failure @ apr17-codex-5.3-effort=high | measured |
| `SparseSignedBasisPursuitRegressor` | 272.7 | 0.674 | 0.758 | honest | Forward-selected signed basis with ridge refit and rounded coefficients. | success @ apr17-codex-5.3-effort=high | measured |
| `HingeGAMRegressor` | 280.2 | 0.558 | 0.783 | honest | Pure Lasso on hinge features with ten breakpoints. | failure @ apr9-claude-effort=medium-main-result | measured |
| `WinsorizedSparseOLSRegressor` | 326.9 | 0.651 | 0.726 | honest | Winsorized features, LassoCV selection, and OLS refit. | failure @ apr19-claude-4.7-effort=medium-rerun2 | measured |
| `TinyDTDepth2Regressor` | 334.0 | 0.674 | 0.713 | honest | Depth-2 decision tree with four leaves. | failure @ apr19-claude-4.7-effort=medium-rerun3 | unmeasured-after-fix |
| `SmartAdditiveRegressor` | 354.3 | 0.744 | 0.733 | honest | Laplacian-smoothed boosted stumps rendered as linear or short piecewise terms. | failure @ apr9-claude-effort=medium-main-result | measured |

Rank and interpretability were measured by the evolution harness on 65 development datasets (rank) and 157 held-out LLM-graded tests (interp), on ndarray input, before the fixes listed in CHANGELOG. 'failure' provenance means the model did not improve on its predecessor within its own run but was selected for architectural diversity.

## Honest vs. Display-Predict Decoupled

Honest models are the right default for regulated, audited, medical, financial, or stakeholder-facing settings where the displayed formula must be a faithful description of `predict`.

Display-predict decoupled models are useful when you want a compact explanation but still want stronger numeric predictions. Their string output is still useful, but it understates the full predictor because residual correctors or teacher ensembles can contribute to `predict`.

When reporting a decoupled model, say that the displayed form is a readable summary, not the complete computational graph.

## Practical Workflow

1. Fit one honest model and one high-rank decoupled model.
2. Print both fitted forms.
3. Check whether the story agrees on feature direction, thresholds, and dominant predictors.
4. If they agree, report the shared pattern and the performance/readability tradeoff.
5. If they disagree, prefer the honest model for interpretation and the decoupled model for prediction.

```python
from sklearn.metrics import r2_score
from agentic_imodels import HingeEBMRegressor, SmartAdditiveRegressor

for cls in (SmartAdditiveRegressor, HingeEBMRegressor):
    model = cls().fit(X_train, y_train)
    print(f"=== {cls.__name__}: R2={r2_score(y_test, model.predict(X_test)):.3f} ===")
    print(model)
```
