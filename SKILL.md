---
name: agentic-imodels
description: Use when the user wants an interpretable, explainable or transparent tabular regression model, a printed equation, rules, GAM or small tree, or asks to explain a numeric relationship without SHAP. Provides sklearn-compatible regressors whose str(model) is the explanation.
---

# agentic-imodels

Use this skill when the user has a **tabular regression** task and wants a model whose fitted form can be printed, inspected, and reasoned about. The package is useful when the downstream reader is a human analyst, a stakeholder, or another LLM.

Typical triggers:

- "fit an interpretable regressor"
- "show me the equation / rules / tree"
- "I need an explanation without SHAP"
- "pick a transparent model for this dataset"
- "answer this data question and show the fitted model"

Do not use this skill for classification unless you explicitly explain that `agentic_imodels` is a regressor library and the model is only being used for shape discovery on a numeric target.

## Install

```bash
uv add git+https://github.com/ezerfernandes/agentic-imodels
pip install git+https://github.com/ezerfernandes/agentic-imodels
```

For a local checkout:

```bash
uv sync --extra dev
uv run --extra dev python -m pytest
```

Runtime dependencies are `numpy`, `scikit-learn`, and `interpret`.

## Basic API

Every public class is a scikit-learn-compatible regressor:

```python
from agentic_imodels import HingeEBMRegressor

model = HingeEBMRegressor()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
print(model)
```

Always print the fitted model. The printed representation is the core artifact this package provides.

## Feature names

Keep `X` as a pandas DataFrame when it has meaningful column names; fitted displays then use names such as `MedInc` and `Latitude`. You can also pass `feature_names=[...]` to any public estimator when fitting an array. An ndarray or list without explicit names uses positional names `x0`, `x1`, and so on. DataFrame predictions may use a different column order, but missing fitted columns raise `ValueError`.

## Choosing A Model

| Model | Category | Summary | Predict note | Provenance | Metrics |
| --- | --- | --- | --- | --- | --- |
| `HingeEBMRegressor` | display-predict decoupled | Lasso on hinge basis plus hidden EBM residual corrector. | predict adds a hidden EBM residual corrector to the displayed hinge formula. | success @ apr9-claude-effort=medium-main-result | measured |
| `DistilledTreeBlendAtlasRegressor` | display-predict decoupled | Ridge student distilled from GBM and RF teachers, displayed as an atlas card. | predict returns the calibrated GBM/RF/student blend, not the displayed sparse equation. | success @ apr19-codex-5.3-effort=xhigh | unmeasured-after-fix |
| `DualPathSparseSymbolicRegressor` | display-predict decoupled | Sparse symbolic display with a blended GBM/RF/Ridge predictor. | predict uses the teacher ensemble by default; pass predict_with='student' to predict with the displayed equation. | failure @ apr17-codex-5.3-effort=high | measured |
| `HybridGAM` | display-predict decoupled | Smart additive GAM display plus hidden random-forest residual corrector. | predict adds a shrunk random-forest residual correction to the displayed additive model. | failure @ apr20-claude-4.7-effort=medium-rerun4 | measured |
| `TeacherStudentRuleSplineRegressor` | display-predict decoupled | GBM teacher with sparse symbolic student over rules, splines, and interactions. | predict uses the teacher ensemble by default; pass predict_with='student' to predict with the displayed equation. | failure @ apr17-codex-5.3-effort=high | measured |
| `SparseSignedBasisPursuitRegressor` | honest | Forward-selected signed basis with ridge refit and rounded coefficients. | predict computes exactly the displayed form. | success @ apr17-codex-5.3-effort=high | measured |
| `HingeGAMRegressor` | honest | Pure Lasso on hinge features with ten breakpoints. | predict computes exactly the displayed form. | failure @ apr9-claude-effort=medium-main-result | unmeasured-after-fix |
| `WinsorizedSparseOLSRegressor` | honest | Winsorized features, LassoCV selection, and OLS refit. | predict computes exactly the displayed form. | failure @ apr19-claude-4.7-effort=medium-rerun2 | measured |
| `TinyDTDepth2Regressor` | honest | Depth-2 decision tree with four leaves. | predict computes exactly the displayed form. | failure @ apr19-claude-4.7-effort=medium-rerun3 | unmeasured-after-fix |
| `SmartAdditiveRegressor` | honest | Laplacian-smoothed boosted stumps rendered as linear or short piecewise terms. | predict computes exactly the displayed form. | failure @ apr9-claude-effort=medium-main-result | measured |

Rank and interpretability were measured by the evolution harness on 65 development datasets (rank) and 157 held-out LLM-graded tests (interp), on ndarray input, before the fixes listed in CHANGELOG. 'failure' provenance means the model did not improve on its predecessor within its own run but was selected for architectural diversity.

Model categories:

- **Honest:** `SparseSignedBasisPursuitRegressor`, `HingeGAMRegressor`, `WinsorizedSparseOLSRegressor`, `TinyDTDepth2Regressor`, `SmartAdditiveRegressor`. The display closely matches what `predict` computes.
  - `predict computes exactly the displayed form.`
- **Display-predict decoupled:** `HingeEBMRegressor`, `DistilledTreeBlendAtlasRegressor`, `DualPathSparseSymbolicRegressor`, `HybridGAM`, `TeacherStudentRuleSplineRegressor`. The display is a readable summary, while `predict` may include a hidden residual corrector or teacher ensemble.
  - `HingeEBMRegressor`: `predict adds a hidden EBM residual corrector to the displayed hinge formula.`
  - `DistilledTreeBlendAtlasRegressor`: `predict returns the calibrated GBM/RF/student blend, not the displayed sparse equation.`
  - `DualPathSparseSymbolicRegressor`: `predict uses the teacher ensemble by default; pass predict_with='student' to predict with the displayed equation.`
  - `HybridGAM`: `predict adds a shrunk random-forest residual correction to the displayed additive model.`
  - `TeacherStudentRuleSplineRegressor`: `predict uses the teacher ensemble by default; pass predict_with='student' to predict with the displayed equation.`

If you use a decoupled model, disclose that the printed text is not the complete computational graph.

## Recommended Analysis Workflow

For data-analysis questions:

1. Identify the dependent variable, focal predictor, and plausible controls.
2. Run classical statistical checks when the user needs p-values or confidence intervals.
3. Fit at least one honest model and one high-rank decoupled model for shape and robustness.
4. Print each fitted model.
5. Interpret only what the model text supports: direction, thresholds, feature importance, and disagreements.
6. State limitations: regression-only, numeric features, no built-in missing-value or categorical handling.

Example:

```python
from sklearn.metrics import r2_score
from agentic_imodels import HingeEBMRegressor, SmartAdditiveRegressor

for cls in (SmartAdditiveRegressor, HingeEBMRegressor):
    model = cls().fit(X_train, y_train)
    print(f"=== {cls.__name__}: R2={r2_score(y_test, model.predict(X_test)):.3f} ===")
    print(model)
```

## Reporting Guidance

Return the fitted model text verbatim when practical. Then summarize:

- model class and category
- evaluation metric used
- strongest positive and negative features
- thresholds or nonlinear shapes
- whether honest and decoupled models agree
- any preprocessing assumptions
- Never report `x3`-style positional names when real DataFrame column names exist

Good phrasing for decoupled models:

> `HingeEBMRegressor` uses a readable hinge formula plus a hidden EBM residual corrector, so the printed formula is a compact explanation rather than the full prediction function.

## Registry

Use the registry for structured metadata:

```python
from agentic_imodels import HONEST_MODELS, MODEL_REGISTRY, get_model_info

print(HONEST_MODELS)
print(get_model_info("HingeEBMRegressor"))
```

Package docs live in `docs/`:

- `docs/model-selection.md`
- `docs/api-reference.md`
- `docs/agent-skill.md`
- `docs/development.md`
