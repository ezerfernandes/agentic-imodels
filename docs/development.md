# Development And Release Guide

This guide covers changes to the installable package. The research/evolution folders are useful provenance, but package users should depend on the root package.

## Environment

Use uv for package management:

```bash
uv sync --extra dev
uv run --extra dev python -m pytest
```

Research dependencies are optional:

```bash
uv sync --extra dev --extra research
```

## Adding A Model

Adding a public model should touch a small, predictable set of files.

1. Add `agentic_imodels/<module_name>.py`.
2. Implement a scikit-learn-compatible estimator class with `fit`, `predict`, and `__str__`.
3. Export the class from `agentic_imodels/__init__.py`.
4. Add a `ModelInfo` entry in `agentic_imodels/registry.py`.
5. Update `tests/test_public_api.py` if the registry contract changes.
6. Rely on `tests/test_smoke_models.py` to fit, predict, and stringify the new class through `agentic_imodels.__all__`.
7. Update `README.md`, `docs/model-selection.md`, and `SKILL.md` if the model is part of the recommended public set.

Keep the estimator module self-contained. Do not add research harness imports to package modules.

## Public API Rules

- `agentic_imodels.__all__` contains estimator class names only.
- Registry helpers are importable from `agentic_imodels`, but they do not belong in `__all__`.
- `MODEL_REGISTRY` must have one entry for every public estimator.
- `HONEST_MODELS` and `DECOUPLED_MODELS` are derived from the registry.

## Verification

Run these before publishing or handing off package changes:

```bash
uv run --extra dev ruff check agentic_imodels tests
uv run --extra dev ruff format --check agentic_imodels tests
uv run --extra dev python -m pytest -q
uv run --extra dev python -m build
uv run --extra dev python -c "from agentic_imodels import MODEL_REGISTRY; print(len(MODEL_REGISTRY))"
```

Inspect built artifacts when package boundaries change:

```bash
uv run --extra dev python -c "import zipfile; z=zipfile.ZipFile('dist/agentic_imodels-0.1.0-py3-none-any.whl'); print('\n'.join(z.namelist()))"
uv run --extra dev python -c "import tarfile; t=tarfile.open('dist/agentic_imodels-0.1.0.tar.gz'); print('\n'.join(t.getnames()))"
```

The wheel should contain runtime package files and dist metadata. The source distribution should contain the runtime package, root docs, skill file, license, README, and package docs.

## Continuous Integration

GitHub Actions mirrors the local Ruff, fast-test, build, and wheel-content checks on Python 3.10–3.12. A Python 3.12 slow job runs the California-housing test and end-to-end script, uploading the JSON summary and printed displays as reviewable artifacts.

## Release Checklist

1. Update `project.version` in `pyproject.toml`.
2. Update model tables if public model metadata changed.
3. Run the verification commands above.
4. Inspect wheel and sdist contents.
5. Tag the release in git.

## Documentation Checklist

When docs are regenerated, keep these files aligned:

- `README.md`: concise package landing page.
- `SKILL.md`: agent-facing operational instructions.
- `docs/model-selection.md`: model tradeoffs and defaults.
- `docs/api-reference.md`: import and registry API.
- `docs/agent-skill.md`: how external agents should use the skill.
- `docs/development.md`: extension and release workflow.
