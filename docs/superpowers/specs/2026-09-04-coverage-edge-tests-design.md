# Production Coverage Edge Tests

## Goal

Raise statement coverage for the production package (`agentic_imodels`) above
99% while preserving the existing behavioral and integration coverage. The
coverage measurement is the full pytest suite with `--cov=agentic_imodels`;
tests, scripts, and research directories are outside the target.

## Approach

Add a focused `tests/test_coverage_edges.py` module. Tests use small,
deterministic arrays and public estimators where an input can naturally reach
the branch. Pure private helpers are tested directly only when a branch is a
defensive guard or a formatting/evaluation utility that normal model fitting
does not expose. Solver fallback branches are exercised by monkeypatching the
module-local `numpy.linalg.solve` to raise `LinAlgError`, then asserting the
least-squares result. No production code changes are planned.

## Test groups

- Shared feature-name helpers: empty/one-dimensional conversion, fitted
  feature-name alignment with explicit and fitted DataFrame columns, missing
  columns, and the sklearn validation fallback path.
- Atlas: zero-feature fit/predict, empty ridge/blend helpers, singular ridge
  and blend fallback paths, invalid row/feature shapes, constant calibration,
  inactive-feature fallback, and linear/equation formatting.
- DualPath and TeacherStudent: zero-column ridge helpers, singular solver
  fallbacks, empty design/candidate paths, all term evaluation and formatting
  variants, no-correlation screening, invalid `predict_with`, and empty student
  selections.
- GAM estimators: feature-selection truncation, constant-feature handling,
  no-split/early-stop/refit guards, constant-model displays, residual-type
  selection, and linear versus nonlinear prediction/display paths.
- Sparse basis pursuit: empty ridge/design paths, no-candidate selection,
  solver fallback, all basis/formatting variants, and empty-equation display.
- Package exports: the optional import failure branch in `__init__.py` is
  covered with an isolated import simulation if it remains reachable under the
  installed dependency set.

## Quality constraints

Tests must assert observable values, errors, or rendered text—not merely call
lines for coverage. They must be deterministic, avoid network data, and use
small datasets so the added suite remains practical. Existing slow tests stay
marked slow; new edge tests should remain in the fast suite unless a test
requires an expensive estimator fit.

## Acceptance

1. The new tests pass under the project’s supported Python environment.
2. The full suite passes with no new failures.
3. `uv run --extra dev --with coverage --with pytest-cov python -m pytest -q
   --cov=agentic_imodels --cov-report=term-missing` reports `TOTAL` coverage
   strictly greater than 99%.
4. Ruff and `git diff --check` pass.
