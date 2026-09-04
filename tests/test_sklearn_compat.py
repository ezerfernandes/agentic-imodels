"""Scikit-learn estimator checks for the public regressors."""

from importlib import import_module

import pytest
from sklearn.utils.estimator_checks import check_estimator

from agentic_imodels import MODEL_REGISTRY

# ai-sklearn-compat-vkf: one-sample fitting still cannot satisfy the internal
# cross-validation assumptions of the ensemble-style Atlas and TeacherStudent
# estimators. All other known baseline failures must stay out of this list.
ALLOWED_FAILURES = {"check_fit2d_1sample"}


@pytest.mark.slow
@pytest.mark.parametrize("model_name", tuple(MODEL_REGISTRY))
def test_public_estimators_pass_sklearn_checks(model_name):
    info = MODEL_REGISTRY[model_name]
    estimator_class = getattr(import_module(info.module), model_name)

    results = check_estimator(estimator_class(), on_fail=None)
    failed = {result["check_name"] for result in results if result["status"] == "failed"}
    print(f"{model_name}: failed checks = {sorted(failed)}")

    assert "check_methods_subset_invariance" not in failed
    assert failed <= ALLOWED_FAILURES
