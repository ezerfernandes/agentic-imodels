# agentic-imodels Documentation

This documentation covers the installable package at the repository root. The research folders are retained for provenance, but users should start with the package and skill files in the root.

## Quick Map

- [Model selection guide](model-selection.md): which estimator to pick and why.
- [API reference](api-reference.md): imports, registry metadata, estimator contract, and examples.
- [Agent skill guide](agent-skill.md): how Codex, Claude Code, and other agents should use `SKILL.md`.
- [Development and release guide](development.md): adding models, running checks, and preparing versions.

## Core Concepts

`agentic-imodels` treats the printed fitted model as a product surface. After `.fit(X, y)`, `print(model)` should produce text that a human or LLM can inspect without separate SHAP plots or notebook-only artifacts.

The package exposes two categories:

- **Honest models:** the displayed form closely matches what `predict` computes.
- **Display-predict decoupled models:** the display is intentionally simpler than the full predictor, which may include hidden residual correctors or teacher ensembles.

That distinction is the most important choice a user or agent needs to make.

## Minimal Example

```python
from sklearn.datasets import make_regression
from agentic_imodels import SmartAdditiveRegressor

X, y = make_regression(n_samples=200, n_features=6, random_state=0)
model = SmartAdditiveRegressor().fit(X, y)

print(model)
y_hat = model.predict(X[:5])
```

## Local Checks

```bash
uv run --extra dev python -m pytest
uv run --extra dev python -m build
```

The tests fit, predict, and stringify every public estimator.
