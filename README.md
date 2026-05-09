# agentic-imodels

`agentic-imodels` is a small Python package of scikit-learn-compatible tabular regressors whose fitted forms are designed to be read by humans and coding agents. The repository root is the canonical install target and the canonical skill entrypoint.

```bash
pip install git+https://github.com/csinva/agentic-imodels
uv add git+https://github.com/csinva/agentic-imodels
```

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from agentic_imodels import HingeEBMRegressor

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = HingeEBMRegressor().fit(X_train, y_train)
print(model)              # compact equation / rule set / model card
predictions = model.predict(X_test)
```

## Why This Exists

The models in this package came from agentic research loops that repeatedly wrote, evaluated, and selected interpretable regressors. They optimize for two things that are usually in tension:

- **Predictive performance:** RMSE rank across tabular regression datasets.
- **Agent readability:** whether another LLM can answer questions from the fitted model text.

The result is a curated set of ten estimators that plug into normal scikit-learn workflows while making `str(model)` a first-class artifact.

## Install And Develop

For local development:

```bash
git clone https://github.com/csinva/agentic-imodels
cd agentic-imodels
uv sync --extra dev
uv run --extra dev python -m pytest
```

Runtime dependencies are intentionally lean: `numpy`, `scikit-learn`, and `interpret`. Research dependencies are kept behind the optional `research` extra.

## Model Guide

| Class | Rank ↓ | Test interp ↑ | Category | Best use |
| --- | ---: | ---: | --- | --- |
| `HingeEBMRegressor` | 108.2 | 0.71 | display-predict decoupled | Default when accuracy matters and a readable display is enough. |
| `DistilledTreeBlendAtlasRegressor` | 139.7 | 0.71 | display-predict decoupled | Strong prediction with a probe-answer atlas summary. |
| `DualPathSparseSymbolicRegressor` | 163.5 | 0.71 | display-predict decoupled | Sparse symbolic display with a stronger hidden ensemble predictor. |
| `HybridGAM` | 163.8 | 0.68 | display-predict decoupled | Additive display plus residual random-forest correction. |
| `TeacherStudentRuleSplineRegressor` | 204.0 | 0.80 | display-predict decoupled | Highest held-out LLM interpretability score. |
| `SparseSignedBasisPursuitRegressor` | 272.7 | 0.76 | honest | Sparse basis model where display and prediction are aligned. |
| `HingeGAMRegressor` | 280.2 | 0.78 | honest | Pure hinge GAM with readable threshold effects. |
| `WinsorizedSparseOLSRegressor` | 326.9 | 0.73 | honest | Sparse linear model after outlier clipping. |
| `TinyDTDepth2Regressor` | 334.0 | 0.71 | honest | Smallest tree-style explanation. |
| `SmartAdditiveRegressor` | 354.3 | 0.73 | honest | Honest additive shapes with linear or short piecewise terms. |

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
