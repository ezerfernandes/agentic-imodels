from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = [ROOT / "README.md", ROOT / "SKILL.md", *sorted((ROOT / "docs").glob("*.md"))]
PYTHON_BLOCK = re.compile(r"^```python\s*$([\s\S]*?)^```\s*$", re.MULTILINE)
PREAMBLE = "\n".join(
    [
        "X, y = make_regression(n_samples=300, n_features=6, random_state=0)",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)",
    ]
)


@dataclass(frozen=True)
class PythonBlock:
    path: Path
    index: int
    source: str


def _python_blocks() -> list[PythonBlock]:
    blocks = []
    for path in DOC_PATHS:
        text = path.read_text(encoding="utf-8")
        blocks.extend(
            PythonBlock(path=path, index=index, source=source)
            for index, source in enumerate(PYTHON_BLOCK.findall(text), start=1)
        )
    return blocks


def _block_id(block: PythonBlock) -> str:
    return f"{block.path.relative_to(ROOT)}-block-{block.index}"


@pytest.mark.parametrize(
    "block",
    [
        pytest.param(
            block,
            id=_block_id(block),
            marks=pytest.mark.slow if "fetch_california_housing" in block.source else (),
        )
        for block in _python_blocks()
    ],
)
def test_python_documentation_block(block: PythonBlock) -> None:
    source = block.source.strip()
    if source.startswith("# docs-test: skip"):
        pytest.skip("documentation block opted out with docs-test: skip")

    if "fetch_california_housing" in source:
        try:
            fetch_california_housing()
        except Exception as exc:
            pytest.skip(f"California housing dataset unavailable (likely offline): {exc}")

    namespace = {
        "fetch_california_housing": fetch_california_housing,
        "make_regression": make_regression,
        "train_test_split": train_test_split,
    }
    exec(PREAMBLE, namespace)
    try:
        exec(source, namespace)
    except Exception as exc:
        location = f"{block.path.relative_to(ROOT)} Python block {block.index}"
        raise AssertionError(f"{location} failed: {exc}") from exc
