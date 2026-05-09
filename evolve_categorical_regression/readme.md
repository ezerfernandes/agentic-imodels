# evolve_categorical_regression

This folder is the categorical-regression counterpart to `evolve/`. It keeps the same autoresearch loop: a coding agent edits one model file, the fixed harness evaluates predictive performance and LLM-readability, and promising candidates are saved for later packaging.

The key difference is the data contract. Models receive pandas DataFrames with original categorical labels preserved. They should not treat category codes as ordered numeric quantities unless they explicitly create such a representation and explain it.

## Setup

Use the research extra because this loop depends on OpenML, imodels, imodelsx, plotting, and other experiment-only packages:

```bash
uv sync --extra dev --extra research
```

## Dataset Scale

The harness targets `150` OpenML supervised-regression tasks with at least one symbolic/categorical feature. This is intentionally close to the original regression loop's broad dataset scale. The discovered manifest is cached at:

```text
evolve_categorical_regression/results/openml_categorical_regression_manifest.csv
```

Delete that file to force a fresh OpenML discovery pass.

## Run Baselines

```bash
uv run --extra research evolve_categorical_regression/run_baselines.py
```

Useful quick check with fewer datasets:

```bash
uv run --extra research evolve_categorical_regression/run_baselines.py --target-count 10
```

Outputs:

- `results/interpretability_results.csv`
- `results/performance_results.csv`
- `results/overall_results.csv`
- `results/interpretability_vs_performance.png`

## Run One Candidate

```bash
uv run --extra research evolve_categorical_regression/interpretable_categorical_regressor.py
```

The default candidate is `CategoricalEffectRegressor`, a simple additive model with numeric ridge terms and smoothed categorical level effects.

## What To Evolve

Edit only:

```text
evolve_categorical_regression/interpretable_categorical_regressor.py
```

Promising model families:

- grouped one-hot sparse linear models that print original levels
- smoothed target/effect encoded GAMs with leakage-safe fitting
- categorical rule lists with numeric thresholds
- category-by-numeric interaction displays
- readable categorical display plus hidden residual corrector

Every candidate should handle:

- unseen categories at prediction time
- missing categorical values
- high-cardinality categorical features through grouping or regularization
- readable `__str__` output using original category labels
