"""Categorical interpretability tests for sklearn-style regressors."""

from __future__ import annotations

import os
import re
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import clone
from sklearn.metrics import r2_score


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
_memory = Memory(location=os.path.join(RESULTS_DIR, "cache"), verbose=0)
CHECKPOINT = "gpt-4o"


def _safe_clone(model):
    try:
        return clone(model)
    except Exception:
        return deepcopy(model)


def get_model_str(model) -> str:
    return str(model)


def ask_llm(llm, model_str, question, max_tokens=200):
    prompt = f"Here is a trained categorical regression model:\n\n{model_str}\n\n{question}"
    return llm(
        prompt,
        max_completion_tokens=max_tokens,
        stop=["cannot", "I do not have enough", "I'm sorry"],
    )


def _category_main_effect_data(n=360, seed=0):
    rng = np.random.RandomState(seed)
    city = rng.choice(["Austin", "Boston", "Chicago"], size=n, p=[0.35, 0.35, 0.30])
    income = rng.normal(0, 1, size=n)
    city_effect = {"Austin": 0.0, "Boston": 4.0, "Chicago": -2.0}
    y = np.array([city_effect[c] for c in city]) + 1.5 * income + rng.randn(n) * 0.2
    X = pd.DataFrame({"city": city, "income": income})
    return X, y


def _two_category_data(n=420, seed=1):
    rng = np.random.RandomState(seed)
    plan = rng.choice(["basic", "pro", "enterprise"], size=n)
    region = rng.choice(["north", "south"], size=n)
    age = rng.normal(0, 1, size=n)
    y = (
        np.where(plan == "enterprise", 5.0, np.where(plan == "pro", 2.0, 0.0))
        + np.where(region == "south", -1.5, 1.0)
        + age
        + rng.randn(n) * 0.2
    )
    X = pd.DataFrame({"plan": plan, "region": region, "age": age})
    return X, y


def _category_numeric_interaction_data(n=520, seed=2):
    rng = np.random.RandomState(seed)
    tier = rng.choice(["standard", "premium"], size=n)
    usage = rng.normal(0, 1, size=n)
    y = np.where(tier == "premium", 3.0 * usage + 2.0, 0.5 * usage - 1.0) + rng.randn(n) * 0.2
    X = pd.DataFrame({"tier": tier, "usage": usage})
    return X, y


def _contains_any(text: str | None, patterns: tuple[str, ...]) -> bool:
    lower = (text or "").lower()
    return any(pattern.lower() in lower for pattern in patterns)


def test_identify_positive_category(model, llm):
    X, y = _category_main_effect_data()
    m = _safe_clone(model)
    m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.35
    response = ask_llm(
        llm,
        get_model_str(m),
        "Which city category has the largest positive effect on the prediction? "
        "Answer with only the category label.",
        max_tokens=20,
    )
    return {
        "test": "category_positive_level",
        "passed": _contains_any(response, ("Boston",)),
        "ground_truth": "Boston",
        "response": response,
    }


def test_identify_negative_category(model, llm):
    X, y = _category_main_effect_data(seed=3)
    m = _safe_clone(model)
    m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.35
    response = ask_llm(
        llm,
        get_model_str(m),
        "Which city category lowers the prediction the most? Answer with only the category label.",
        max_tokens=20,
    )
    return {
        "test": "category_negative_level",
        "passed": _contains_any(response, ("Chicago",)),
        "ground_truth": "Chicago",
        "response": response,
    }


def test_numeric_plus_category_simulation(model, llm):
    X, y = _two_category_data()
    m = _safe_clone(model)
    m.fit(X, y)
    sample = pd.DataFrame({"plan": ["enterprise"], "region": ["north"], "age": [1.0]})
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm,
        get_model_str(m),
        "What does the model predict for plan=enterprise, region=north, age=1.0? "
        "Answer with just one number.",
    )
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    passed = False
    for value in nums:
        if abs(float(value) - true_pred) < max(abs(true_pred) * 0.2, 1.0):
            passed = True
            break
    return {
        "test": "category_numeric_simulation",
        "passed": passed,
        "ground_truth": round(true_pred, 3),
        "response": response,
    }


def test_category_swap_counterfactual(model, llm):
    X, y = _two_category_data(seed=4)
    m = _safe_clone(model)
    m.fit(X, y)
    base = pd.DataFrame({"plan": ["basic"], "region": ["north"], "age": [0.0]})
    changed = pd.DataFrame({"plan": ["enterprise"], "region": ["north"], "age": [0.0]})
    delta = float(m.predict(changed)[0] - m.predict(base)[0])
    response = ask_llm(
        llm,
        get_model_str(m),
        "Holding region=north and age=0 fixed, how much does the prediction change when "
        "plan changes from basic to enterprise? Answer with just one number.",
    )
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    passed = False
    for value in nums:
        if abs(float(value) - delta) < max(abs(delta) * 0.25, 1.0):
            passed = True
            break
    return {
        "test": "category_swap_counterfactual",
        "passed": passed,
        "ground_truth": round(delta, 3),
        "response": response,
    }


def test_unseen_category_policy(model, llm):
    X, y = _category_main_effect_data(seed=5)
    m = _safe_clone(model)
    m.fit(X, y)
    response = ask_llm(
        llm,
        get_model_str(m),
        "If the model sees a city category that was not present during training, what category "
        "effect should be used according to the printed model? Answer briefly.",
    )
    passed = _contains_any(response, ("0", "zero", "unknown", "unseen", "missing", "baseline"))
    return {
        "test": "unseen_category_policy",
        "passed": passed,
        "ground_truth": "unseen category uses zero/baseline effect",
        "response": response,
    }


def test_category_numeric_interaction(model, llm):
    X, y = _category_numeric_interaction_data()
    m = _safe_clone(model)
    m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.25
    response = ask_llm(
        llm,
        get_model_str(m),
        "Which tier has the stronger positive slope with respect to usage: standard or premium? "
        "Answer with only the tier label.",
        max_tokens=20,
    )
    return {
        "test": "category_numeric_interaction",
        "passed": _contains_any(response, ("premium",)),
        "ground_truth": "premium",
        "response": response,
    }


CATEGORY_TESTS = [
    ("category_effects", [test_identify_positive_category, test_identify_negative_category]),
    ("simulation", [test_numeric_plus_category_simulation]),
    ("counterfactual", [test_category_swap_counterfactual]),
    ("robustness", [test_unseen_category_policy]),
    ("interactions", [test_category_numeric_interaction]),
]

ALL_TESTS = [fn for _, tests in CATEGORY_TESTS for fn in tests]
_ALL_TEST_FNS = {fn.__name__: fn for fn in ALL_TESTS}


@_memory.cache
def _run_one_test(model_name, test_fn_name, model, checkpoint=None):
    import imodelsx.llm

    llm = imodelsx.llm.get_llm(checkpoint or CHECKPOINT)
    test_fn = _ALL_TEST_FNS[test_fn_name]
    try:
        result = test_fn(model, llm)
    except AssertionError as exc:
        result = {"test": test_fn_name, "passed": False, "error": f"Assertion: {exc}", "response": None}
    except Exception as exc:
        result = {"test": test_fn_name, "passed": False, "error": str(exc), "response": None}
    result["model"] = model_name
    result.setdefault("test", test_fn_name)
    return result


def run_all_interp_tests(model_defs, checkpoint=None):
    from joblib import Parallel, delayed

    tasks = [(name, reg, test_fn) for name, reg in model_defs for test_fn in ALL_TESTS]
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_run_one_test)(name, test_fn.__name__, reg, checkpoint=checkpoint)
        for name, reg, test_fn in tasks
    )

    for name, _ in model_defs:
        print(f"\n{'=' * 60}\n  Model: {name}\n{'=' * 60}")
        for label, tests in CATEGORY_TESTS:
            suite_results = [r for r in results if r["model"] == name and r["test"] in {t.__name__ for t in tests}]
            print(f"\n  [{label}]")
            for result in suite_results:
                status = "PASS" if result["passed"] else "FAIL"
                response = (result.get("response") or "")[:80].replace("\n", " ")
                print(f"  [{status}] {result['test']}")
                print(f"         ground_truth : {result.get('ground_truth', '')}")
                print(f"         llm_response : {response}")
            print(f"\n  -> {sum(r['passed'] for r in suite_results)}/{len(tests)} passed")

    return results

