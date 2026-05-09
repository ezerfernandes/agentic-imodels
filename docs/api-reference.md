# API Reference

All public estimators are available from the top-level package.

```python
from agentic_imodels import HingeEBMRegressor, SmartAdditiveRegressor
```

Each estimator follows the scikit-learn regressor contract:

- constructor has no required arguments
- `fit(X, y)` trains the model and returns `self`
- `predict(X)` returns a one-dimensional numeric prediction array
- `str(model)` or `print(model)` returns the readable fitted form

## Public Estimators

```python
from agentic_imodels import (
    HingeEBMRegressor,
    HybridGAM,
    SmartAdditiveRegressor,
    HingeGAMRegressor,
    TeacherStudentRuleSplineRegressor,
    DualPathSparseSymbolicRegressor,
    SparseSignedBasisPursuitRegressor,
    DistilledTreeBlendAtlasRegressor,
    WinsorizedSparseOLSRegressor,
    TinyDTDepth2Regressor,
)
```

The public estimator class names are also listed in `agentic_imodels.__all__`.

## Registry

The registry is the structured source for model metadata. Use it in docs, selection helpers, tests, and agent prompts instead of duplicating tradeoff data in code.

```python
from agentic_imodels import (
    DECOUPLED_MODELS,
    HONEST_MODELS,
    MODEL_REGISTRY,
    ModelInfo,
    get_model_info,
)

info = get_model_info("HingeEBMRegressor")
print(info.rank)
print(info.test_interpretability)
```

`ModelInfo` fields:

| Field | Meaning |
| --- | --- |
| `name` | Public estimator class name. |
| `module` | Import module path. |
| `shorthand` | Original experiment shorthand. |
| `rank` | Mean global RMSE rank across development datasets; lower is better. |
| `dev_interpretability` | Development interpretability pass fraction. |
| `test_interpretability` | Held-out interpretability pass fraction. |
| `category` | `honest` or `display-predict decoupled`. |
| `summary` | One-line model description. |

## Pipeline Example

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from agentic_imodels import HingeGAMRegressor

pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("model", HingeGAMRegressor()),
    ]
)

pipe.fit(X_train, y_train)
print(pipe.named_steps["model"])
```

## Cross-Validation Example

```python
from sklearn.model_selection import cross_val_score
from agentic_imodels import WinsorizedSparseOLSRegressor

scores = cross_val_score(
    WinsorizedSparseOLSRegressor(),
    X,
    y,
    scoring="neg_root_mean_squared_error",
    cv=5,
)
print(scores)
```

## Data Expectations

The models are intended for tabular regression:

- continuous target
- numeric features
- up to a few thousand rows for comfortable iteration
- up to roughly 50 features for readable output

Encode categorical variables and impute missing values upstream with normal scikit-learn preprocessing.

## Version

```python
import agentic_imodels

print(agentic_imodels.__version__)
```

Source checkouts that are not installed may report `0.0.0`; installed packages report the version from `pyproject.toml`.
