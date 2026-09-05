# agentic-imodels

[![CI](https://github.com/ezerfernandes/agentic-imodels/actions/workflows/ci.yml/badge.svg)](https://github.com/ezerfernandes/agentic-imodels/actions/workflows/ci.yml)

`agentic-imodels` is a Python package of ten scikit-learn-compatible regressors for numeric tabular data. Each estimator has the usual `fit` and `predict` methods. The unusual part is `str(model)`: after fitting, the model prints an equation, rule set, small tree, or model card that a person or coding agent can inspect.

The models came from autonomous research loops that wrote regressors, measured them, and tried again. The installable package is the small, hardened result of that work. The repository also retains the research machinery and raw experiment outputs for provenance.

## Install

```bash
pip install git+https://github.com/ezerfernandes/agentic-imodels
uv add git+https://github.com/ezerfernandes/agentic-imodels
```

Runtime dependencies are `numpy`, `scikit-learn`, and `interpret`. The larger research stack is available through the optional `research` extra.

## Quick Start

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from agentic_imodels import HingeEBMRegressor

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = HingeEBMRegressor().fit(X_train, y_train)
print(model)
predictions = model.predict(X_test)
```

Keep `X` as a pandas DataFrame when column names matter. Fitted displays then use names such as `MedInc` and `Latitude`. Array input uses `x0`, `x1`, and so on unless you pass `feature_names=[...]`.

A depth-two tree fitted on the first 1,500 California housing rows prints text like this:

```text
Decision Tree Regressor (max_depth=2):
|--- MedInc <= 4.72
|   |--- Latitude <= 38.00
|   |   |--- value: [1.74]
|   |--- Latitude >  38.00
|   |   |--- value: [1.03]
```

## The Main Design Choice

The package treats the printed fitted model as part of its public API. There is still a tradeoff. A model small enough to print may lose predictive accuracy, while a stronger model may need more machinery than a clean explanation can show.

The registry makes that choice explicit.

### Honest Models

For an honest model, the displayed form closely matches what `predict` computes. If the display says that a feature has zero effect, changing that feature should not change the prediction. The test suite checks this behavior.

Choose an honest model when the printed explanation must describe the prediction path used in production.

- `SparseSignedBasisPursuitRegressor`
- `HingeGAMRegressor`
- `WinsorizedSparseOLSRegressor`
- `TinyDTDepth2Regressor`
- `SmartAdditiveRegressor`

### Display-Predict Decoupled Models

A decoupled model prints a readable summary, but `predict` also uses a residual corrector or teacher ensemble. These models often have better predictive rank. Their displays disclose the hidden path, and reports should do the same.

Choose a decoupled model when predictive performance matters more than reconstructing every prediction from the printed text.

- `HingeEBMRegressor` adds an EBM residual corrector to its hinge formula.
- `DistilledTreeBlendAtlasRegressor` predicts with a calibrated GBM, random forest, and ridge-student blend.
- `DualPathSparseSymbolicRegressor` uses its teacher ensemble by default. Set `predict_with="student"` to use the displayed equation.
- `HybridGAM` adds a shrunken forest or gradient-boosting residual correction to its additive model.
- `TeacherStudentRuleSplineRegressor` uses its teacher ensemble by default. Set `predict_with="student"` to use the displayed equation.

Do not present a decoupled model's text as the complete computational graph.

## Model Guide

Lower rank is better. Test interpretability is the fraction of held-out LLM-graded questions answered correctly from fitted model text.

| Class | Rank | Test interp | Category | What it does | Provenance | Metrics |
| --- | ---: | ---: | --- | --- | --- | --- |
| `HingeEBMRegressor` | 108.2 | 0.71 | decoupled | Fits a sparse hinge formula, then models its residuals with an EBM. | success @ apr9-claude-effort=medium-main-result | measured |
| `DistilledTreeBlendAtlasRegressor` | 139.7 | 0.71 | decoupled | Blends GBM, random forest, and ridge predictions, then prints a sparse equation and partial-dependence card. | success @ apr19-codex-5.3-effort=xhigh | unmeasured-after-fix |
| `DualPathSparseSymbolicRegressor` | 163.5 | 0.71 | decoupled | Builds a symbolic student from linear, square, hinge, and interaction terms beside a teacher ensemble. | failure @ apr17-codex-5.3-effort=high | measured |
| `HybridGAM` | 163.8 | 0.68 | decoupled | Combines an additive shape model with a hidden residual correction. | failure @ apr20-claude-4.7-effort=medium-rerun4 | measured |
| `TeacherStudentRuleSplineRegressor` | 204.0 | 0.80 | decoupled | Distills a teacher ensemble into sparse rules, splines, and interactions. | failure @ apr17-codex-5.3-effort=high | measured |
| `SparseSignedBasisPursuitRegressor` | 272.7 | 0.76 | honest | Forward-selects signed basis terms, refits them, and prints the exact equation. | success @ apr17-codex-5.3-effort=high | measured |
| `HingeGAMRegressor` | 280.2 | 0.78 | honest | Uses Lasso to select a sparse piecewise-linear additive model. | failure @ apr9-claude-effort=medium-main-result | unmeasured-after-fix |
| `WinsorizedSparseOLSRegressor` | 326.9 | 0.73 | honest | Clips outliers, selects at most eight features with Lasso, and refits ordinary least squares. | failure @ apr19-claude-4.7-effort=medium-rerun2 | measured |
| `TinyDTDepth2Regressor` | 334.0 | 0.71 | honest | Fits a decision tree with depth at most two and no more than four leaves. | failure @ apr19-claude-effort=medium-rerun3 | unmeasured-after-fix |
| `SmartAdditiveRegressor` | 354.3 | 0.73 | honest | Converts boosted stumps into per-feature shapes and prints near-linear shapes as coefficients. | failure @ apr9-claude-effort=medium-main-result | measured |

The evolution harness measured rank across 65 development datasets and interpretability across 157 held-out tests. Those measurements used array input and predate the fixes recorded by `metrics_status`. A `failure` provenance label means that the model did not improve on its predecessor within that research run. It does not mean that the packaged estimator crashes or fails the current test suite. Some such models were kept because they add a useful model shape to the final set.

For most analysis work, fit one honest model and one strong decoupled model. Compare the feature directions, thresholds, and dominant predictors. If they disagree, use the honest model for interpretation and the decoupled model for prediction.

The same metadata is available in Python:

```python
from agentic_imodels import HONEST_MODELS, MODEL_REGISTRY, get_model_info

print(HONEST_MODELS)
print(get_model_info("HingeEBMRegressor"))
```

See the [model selection guide](docs/model-selection.md) for recommendations by use case.

## How the Package Works

Every public estimator follows the scikit-learn regressor contract:

1. The constructor stores each hyperparameter so cloning and parameter search work.
2. `fit(X, y)` validates dense numeric inputs, records feature names, and learns fitted state.
3. `predict(X)` aligns DataFrame columns to their fitted order, rejects missing columns, and returns a one-dimensional numeric array.
4. `str(model)` checks that the estimator is fitted, then renders its readable form.

DataFrame prediction columns may arrive in a different order. The package aligns them by name. Array and list inputs remain positional and must have the same feature count used during fitting.

The estimators expect numeric features and a continuous target. Encode categorical variables and impute missing values with normal scikit-learn preprocessing.

## From Research Loop to Package

The research workflow has a narrow editing boundary. An agent changes one regressor file while fixed harnesses measure two objectives:

- RMSE rank across tabular regression datasets
- Whether another LLM can answer questions from the fitted model text

Each experiment is committed, evaluated, and saved with its metrics. Processing scripts pool results across runs and identify models on or near the performance and interpretability Pareto front. The final package then removes research-harness imports and adds stable input handling, metadata, documentation, and production tests.

The current tests check more than importability. They check cloning, deterministic fitting, pickling, not-fitted errors, batch invariance, DataFrame column alignment, display disclosures, and the dependency claims made by honest displays.

## Repository Map

The checkout contains a small product core and a large research archive.

### Product Core

| Path | Purpose |
| --- | --- |
| `agentic_imodels/` | Canonical installable package with ten estimator modules, shared input helpers, and the registry. |
| `tests/` | Public API, sklearn compatibility, display faithfulness, packaging, and edge-case tests. |
| `docs/` | Model selection, API, skill, development, and release documentation. |
| `SKILL.md` | Instructions for coding agents that use the package in data-analysis work. |
| `pyproject.toml` | Package metadata, dependencies, build rules, and test configuration. |

### Research Archive

| Path | Purpose |
| --- | --- |
| `evolve/` | Claude-oriented model evolution loop and fixed evaluators. |
| `evolve_codex/` | Codex-oriented version of the evolution loop. |
| `result_libs/` | Raw generated models, metrics, logs, and plots. |
| `result_libs_processed/` | Aggregation and selection scripts, plus a historical package snapshot. |
| `generalization_experiments/` | Held-out checks across evaluators and random seeds. |
| `e2e_experiments/` | BLADE benchmark runs that use the package and skill. |

Applications should import from the root `agentic_imodels/` package. Do not depend on `result_libs_processed/agentic-imodels/`; it is a historical snapshot.

## Development

```bash
git clone https://github.com/ezerfernandes/agentic-imodels
cd agentic-imodels
uv sync --extra dev
uv run --extra dev python -m pytest
```

The wheel contains the runtime package and distribution metadata. The source distribution also contains the root README, skill, license, and package documentation. Research dependencies and output archives do not enter the wheel.

Before publishing a change, run:

```bash
uv run --extra dev ruff check agentic_imodels tests
uv run --extra dev ruff format --check agentic_imodels tests
uv run --extra dev python -m pytest -q
uv run --extra dev python -m build
```

See the [development and release guide](docs/development.md) before adding a model or changing the package boundary.

## Use as an Agent Skill

The root [`SKILL.md`](SKILL.md) teaches Codex, Claude Code, and similar coding agents when to choose these regressors, how to compare an honest model with a decoupled model, and how to report fitted text without overstating it.

Point an agent at the repository root when you want it to choose, fit, print, and interpret a regressor in a data-science workflow. See the [skill guide](docs/agent-skill.md) for installation and usage.

## Documentation

- [Documentation index](docs/README.md)
- [Model selection guide](docs/model-selection.md)
- [API reference](docs/api-reference.md)
- [Agent skill guide](docs/agent-skill.md)
- [Development and release guide](docs/development.md)

## Citation

```bibtex
@misc{singh2026agenticimodels,
      title={Agentic-imodels: Evolving agentic interpretability tools via autoresearch},
      author={Chandan Singh and Yan Shuo Tan and Weijia Xu and Zelalem Gero and Weiwei Yang and Michel Galley and Jianfeng Gao},
      year={2026},
      eprint={2605.03808},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.03808},
}
```

## License

MIT.
