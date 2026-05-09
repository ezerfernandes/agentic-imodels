from __future__ import annotations

import agentic_imodels as ai
from agentic_imodels.registry import HONEST_MODELS, MODEL_REGISTRY, get_model_info


def test_registry_matches_public_exports() -> None:
    assert set(MODEL_REGISTRY) == set(ai.__all__)


def test_each_registry_entry_has_required_metadata() -> None:
    for name in ai.__all__:
        info = get_model_info(name)
        assert info.name == name
        assert info.category in {"honest", "display-predict decoupled"}
        assert info.rank > 0
        assert 0 <= info.dev_interpretability <= 1
        assert 0 <= info.test_interpretability <= 1
        assert info.module.startswith("agentic_imodels.")
        assert info.summary


def test_honest_models_subset_registry() -> None:
    assert HONEST_MODELS
    assert set(HONEST_MODELS).issubset(MODEL_REGISTRY)
    assert "HingeGAMRegressor" in HONEST_MODELS
    assert "HingeEBMRegressor" not in HONEST_MODELS
