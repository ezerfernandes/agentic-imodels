# autoresearch — categorical interpretable regressors

This experiment asks an agent to discover scikit-learn-compatible regressors that work well on tabular datasets with categorical predictors and whose fitted text can be read by humans or LLMs.

## Setup

1. Create a fresh branch named `autoresearch/categorical-<tag>`.
2. Run baselines first:

   ```bash
   uv run --extra research evolve_categorical_regression/run_baselines.py
   ```

3. Read these files:
   - `evolve_categorical_regression/readme.md`
   - `evolve_categorical_regression/run_baselines.py`
   - `evolve_categorical_regression/interpretable_categorical_regressor.py`
   - `evolve_categorical_regression/src/performance_eval.py`
   - `evolve_categorical_regression/src/interp_eval.py`

## What You May Edit

Edit only:

```text
evolve_categorical_regression/interpretable_categorical_regressor.py
```

Everything inside the estimator class is fair game:

- category encoding or grouping
- numeric transformations
- sparsity and feature selection
- rule lists
- category-by-numeric interactions
- hidden residual correctors, if disclosed by `__str__`

Do not edit:

- `run_baselines.py`
- files under `src/`
- cached datasets or cached LLM results
- package files outside this experiment loop

## Goal

Optimize both columns in `results/overall_results.csv`:

- `mean_rank`: average RMSE rank across categorical OpenML regression datasets; lower is better.
- `frac_interpretability_tests_passed`: fraction of categorical LLM-readability tests passed; higher is better.

Both metrics matter. Prefer Pareto improvements over baselines.

## Categorical Requirements

Your model receives pandas DataFrames with original category labels. A good categorical model must:

- preserve category labels in `__str__`
- handle unseen levels at prediction time
- handle missing categorical values
- avoid target leakage in any target/effect encoding
- avoid pretending arbitrary ordinal codes are meaningful orderings
- make clear whether printed text is the exact predictor or a readable display with hidden correction

## Loop

Run one experiment with:

```bash
uv run --extra research evolve_categorical_regression/interpretable_categorical_regressor.py
```

Then:

1. Inspect the printed summary and CSV rows.
2. Commit the candidate code.
3. Save promising candidates under a success archive you create for this run.
4. Save failed but informative attempts under a failure archive.
5. Continue until manually stopped.

Never stop the loop just because one model fails. Diagnose crashes, fix the candidate, and continue.

## Ideas

- Sparse categorical effects with hierarchical shrinkage.
- Rule lists like `if state in {CA, NY} and income > 60k`.
- Numeric hinge terms plus categorical level offsets.
- Rare-level grouping with an explicit `OTHER` term.
- Category interactions only when they pass a compactness budget.
- Distilled teacher models whose printed student is categorical and sparse.
