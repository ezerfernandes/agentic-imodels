# agentic-imodels

[![CI](https://github.com/ezerfernandes/agentic-imodels/actions/workflows/ci.yml/badge.svg)](https://github.com/ezerfernandes/agentic-imodels/actions/workflows/ci.yml)

`agentic-imodels` is a small Python package of scikit-learn-compatible tabular regressors whose fitted forms are designed to be read by humans and coding agents. The repository root is the canonical install target and the canonical skill entrypoint.

```bash
pip install git+https://github.com/ezerfernandes/agentic-imodels
uv add git+https://github.com/ezerfernandes/agentic-imodels
```

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from agentic_imodels import HingeEBMRegressor

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = HingeEBMRegressor().fit(X_train, y_train)
print(model)              # compact equation / rule set / model card
predictions = model.predict(X_test)
```

Because `X` is a DataFrame, fitted displays use its column names. For example, a depth-2 tree fitted on the first 1,500 California-housing rows prints this excerpt:

```text
Decision Tree Regressor (max_depth=2):
|--- MedInc <= 4.72
|   |--- Latitude <= 38.00
|   |   |--- value: [1.74]
|   |--- Latitude >  38.00
|   |   |--- value: [1.03]
```

## Why This Exists

The models in this package came from agentic research loops that repeatedly wrote, evaluated, and selected interpretable regressors. They optimize for two things that are usually in tension:

- **Predictive performance:** RMSE rank across tabular regression datasets.
- **Agent readability:** whether another LLM can answer questions from the fitted model text.

The result is a curated set of ten estimators that plug into normal scikit-learn workflows while making `str(model)` a first-class artifact.

## Install And Develop

For local development:

```bash
git clone https://github.com/ezerfernandes/agentic-imodels
cd agentic-imodels
uv sync --extra dev
uv run --extra dev python -m pytest
```

Runtime dependencies are intentionally lean: `numpy`, `scikit-learn`, and `interpret`. Research dependencies are kept behind the optional `research` extra.

## Model Guide

| Class | Rank ↓ | Test interp ↑ | Category | Best use | Provenance | Metrics |
| --- | ---: | ---: | --- | --- | --- | --- |
| `HingeEBMRegressor` | 108.2 | 0.71 | display-predict decoupled | predict adds a hidden EBM residual corrector to the displayed hinge formula. | success @ apr9-claude-effort=medium-main-result | measured |
| `DistilledTreeBlendAtlasRegressor` | 139.7 | 0.71 | display-predict decoupled | predict returns the calibrated GBM/RF/student blend, not the displayed sparse equation. | success @ apr19-codex-5.3-effort=xhigh | unmeasured-after-fix |
| `DualPathSparseSymbolicRegressor` | 163.5 | 0.71 | display-predict decoupled | predict uses the teacher ensemble by default; pass predict_with='student' to predict with the displayed equation. | failure @ apr17-codex-5.3-effort=high | measured |
| `HybridGAM` | 163.8 | 0.68 | display-predict decoupled | predict adds a shrunk random-forest residual correction to the displayed additive model. | failure @ apr20-claude-4.7-effort=medium-rerun4 | measured |
| `TeacherStudentRuleSplineRegressor` | 204.0 | 0.80 | display-predict decoupled | predict uses the teacher ensemble by default; pass predict_with='student' to predict with the displayed equation. | failure @ apr17-codex-5.3-effort=high | measured |
| `SparseSignedBasisPursuitRegressor` | 272.7 | 0.76 | honest | predict computes exactly the displayed form. | success @ apr17-codex-5.3-effort=high | measured |
| `HingeGAMRegressor` | 280.2 | 0.78 | honest | predict computes exactly the displayed form. | failure @ apr9-claude-effort=medium-main-result | measured |
| `WinsorizedSparseOLSRegressor` | 326.9 | 0.73 | honest | predict computes exactly the displayed form. | failure @ apr19-claude-4.7-effort=medium-rerun2 | measured |
| `TinyDTDepth2Regressor` | 334.0 | 0.71 | honest | predict computes exactly the displayed form. | failure @ apr19-claude-4.7-effort=medium-rerun3 | unmeasured-after-fix |
| `SmartAdditiveRegressor` | 354.3 | 0.73 | honest | predict computes exactly the displayed form. | failure @ apr9-claude-effort=medium-main-result | measured |

Rank and interpretability were measured by the evolution harness on 65 development datasets (rank) and 157 held-out LLM-graded tests (interp), on ndarray input, before the fixes listed in CHANGELOG. 'failure' provenance means the model did not improve on its predecessor within its own run but was selected for architectural diversity.

Use a **display-predict decoupled** model when predictive rank matters most. Use an **honest** model when the printed explanation must closely match the computation used by `predict`.

The same metadata is available programmatically:

```python
from agentic_imodels import HONEST_MODELS, MODEL_REGISTRY, get_model_info

print(HONEST_MODELS)
print(get_model_info("HingeEBMRegressor"))
```

## Documentation

- [Docs index](docs/README.md)
- [Model selection guide](docs/model-selection.md)
- [API reference](docs/api-reference.md)
- [Agent skill guide](docs/agent-skill.md)
- [Development and release guide](docs/development.md)

## Use As A Skill

The root [`SKILL.md`](SKILL.md) is written for Codex, Claude Code, and similar coding agents. Point an agent at this repository root when you want it to choose, fit, print, and interpret these regressors in a data-science workflow.

See the [skill installation guide](docs/agent-skill.md#install-the-skill) for Claude Code user/project installs and the current Codex CLI skill locations.

## Research Provenance

The repository still carries the research machinery that produced the package:

| Path | Purpose |
| --- | --- |
| `agentic_imodels/` | Canonical installable package. |
| `SKILL.md` | Canonical agent skill entrypoint. |
| `evolve/` | Claude-oriented model evolution loop. |
| `evolve_codex/` | Codex-oriented model evolution loop. |
| `result_libs/` | Raw generated models and evaluation outputs. |
| `result_libs_processed/` | Historical processed package snapshot and extraction scripts. |
| `generalization_experiments/` | Held-out regression and interpretability checks. |
| `e2e_experiments/` | BLADE benchmark experiments using the package/skill. |

Wheels contain only the runtime package. Source distributions include the package, root skill, license, README, and package docs.

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
