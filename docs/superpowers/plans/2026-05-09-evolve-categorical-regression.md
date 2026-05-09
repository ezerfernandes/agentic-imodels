# Evolve Categorical Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `evolve_categorical_regression`, a research loop for discovering interpretable regressors that preserve and reason over categorical feature values.

**Architecture:** Mirror the original `evolve/` loop while changing the data contract from numeric arrays to pandas DataFrames plus feature metadata. The performance harness dynamically discovers a large OpenML supervised-regression task pool with categorical predictors, targets roughly 150 datasets, caches the manifest/results, and evaluates baseline pipelines plus the agent-editable candidate.

**Tech Stack:** Python 3.10+, uv, pandas, numpy, scikit-learn, openml, joblib, imodels/imodelsx optional research dependencies.

---

## File Map

- Create `evolve_categorical_regression/readme.md`: human-facing setup and run instructions.
- Create `evolve_categorical_regression/program.md`: agent loop instructions.
- Create `evolve_categorical_regression/run_baselines.py`: baseline model evaluation entrypoint.
- Create `evolve_categorical_regression/interpretable_categorical_regressor.py`: only file the research agent edits.
- Create `evolve_categorical_regression/src/feature_metadata.py`: feature type inference and categorical formatting helpers.
- Create `evolve_categorical_regression/src/performance_eval.py`: OpenML categorical regression discovery, loading, preprocessing, ranking, and CSV helpers.
- Create `evolve_categorical_regression/src/interp_eval.py`: categorical-specific LLM interpretability tests.
- Create `evolve_categorical_regression/src/visualize.py`: plot helper copied/adapted from the regression loop.
- Create `tests/test_categorical_regression_harness.py`: fast unit tests that do not use network or LLMs.

## Tasks

1. Write tests that fail because `evolve_categorical_regression` does not exist.
2. Add feature metadata helpers and make tests pass for categorical/numeric inference.
3. Add a default categorical-aware regressor that handles unseen categories and prints category labels.
4. Add a performance harness with injectable datasets so tests avoid network.
5. Add scripts/docs for the full OpenML experiment loop.
6. Run targeted tests and import checks.

## Dataset Requirement

The harness should default to `TARGET_DATASET_COUNT = 150`, close to the original loop's broad dataset scale. It should discover OpenML supervised-regression tasks with at least one symbolic/categorical feature, cache a manifest in `results/openml_categorical_regression_manifest.csv`, and fail loudly if discovery returns too few datasets unless the caller opts into a lower count for quick tests.

## Verification

Run:

```bash
rtk uv run --extra dev python -m pytest tests/test_categorical_regression_harness.py -q
rtk uv run --extra dev python -m pytest tests/test_public_api.py tests/test_smoke_models.py -q
rtk uv run --extra dev python -m py_compile evolve_categorical_regression/run_baselines.py evolve_categorical_regression/interpretable_categorical_regressor.py evolve_categorical_regression/src/performance_eval.py evolve_categorical_regression/src/interp_eval.py
```
