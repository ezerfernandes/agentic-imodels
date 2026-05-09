---
name: agentic-imodels
description: Use for tabular regression tasks where the fitted model must be readable by humans or LLM agents. Provides sklearn-compatible interpretable regressors and model-selection guidance.
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

## Choosing A Model

| Situation | Model | Why |
| --- | --- | --- |
| User asks for a general interpretable regressor | `HingeEBMRegressor` | Best predictive rank in the curated set with a readable display. |
| Printed model must match prediction | `HingeGAMRegressor` or `SmartAdditiveRegressor` | Honest models with direct additive/threshold displays. |
| Maximum LLM-readability | `TeacherStudentRuleSplineRegressor` | Highest held-out interpretability score. |
| Small tree explanation | `TinyDTDepth2Regressor` | Four-leaf depth-2 decision tree. |
| Sparse linear explanation | `WinsorizedSparseOLSRegressor` | Outlier clipping plus sparse OLS-style display. |

Model categories:

- **Honest:** `SmartAdditiveRegressor`, `HingeGAMRegressor`, `SparseSignedBasisPursuitRegressor`, `WinsorizedSparseOLSRegressor`, `TinyDTDepth2Regressor`. The display closely matches what `predict` computes.
- **Display-predict decoupled:** `HingeEBMRegressor`, `DistilledTreeBlendAtlasRegressor`, `DualPathSparseSymbolicRegressor`, `HybridGAM`, `TeacherStudentRuleSplineRegressor`. The display is a readable summary, while `predict` may include a hidden residual corrector or teacher ensemble.

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
