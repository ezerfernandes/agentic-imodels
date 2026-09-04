from __future__ import annotations

import pytest
from sklearn.datasets import fetch_california_housing

from scripts.e2e_housing import has_failures, run


@pytest.mark.slow
def test_all_public_models_pass_california_housing_end_to_end() -> None:
    try:
        fetch_california_housing(as_frame=True)
    except Exception as exc:
        pytest.skip(f"California housing unavailable: {exc}")

    results = run()

    assert len(results) == 10
    assert not has_failures(results)
    assert all(result["display_chars"] > 0 for result in results)
