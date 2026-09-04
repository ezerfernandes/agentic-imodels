#!/usr/bin/env python3
"""Render documentation tables from the canonical model registry."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_imodels.registry import MODEL_REGISTRY


FOOTNOTE = (
    "Rank and interpretability were measured by the evolution harness on 65 development datasets "
    "(rank) and 157 held-out LLM-graded tests (interp), on ndarray input, before the fixes listed "
    "in CHANGELOG. 'failure' provenance means the model did not improve on its predecessor within "
    "its own run but was selected for architectural diversity."
)


def cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def full_table() -> str:
    lines = [
        "| Class | Rank ↓ | Dev interp ↑ | Test interp ↑ | Category | Summary | Provenance | Metrics |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for info in MODEL_REGISTRY.values():
        lines.append(
            "| "
            + " | ".join(
                map(
                    cell,
                    (
                        f"`{info.name}`",
                        f"{info.rank:.1f}",
                        f"{info.dev_interpretability:.3f}",
                        f"{info.test_interpretability:.3f}",
                        info.category,
                        info.summary,
                        info.provenance,
                        info.metrics_status,
                    ),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def model_guide() -> str:
    lines = [
        "| Class | Rank ↓ | Test interp ↑ | Category | Best use | Provenance | Metrics |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for info in MODEL_REGISTRY.values():
        best_use = info.predict_notes
        lines.append(
            "| "
            + " | ".join(
                map(
                    cell,
                    (
                        f"`{info.name}`",
                        f"{info.rank:.1f}",
                        f"{info.test_interpretability:.2f}",
                        info.category,
                        best_use,
                        info.provenance,
                        info.metrics_status,
                    ),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def choosing_table() -> str:
    lines = [
        "| Model | Category | Summary | Predict note | Provenance | Metrics |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for info in MODEL_REGISTRY.values():
        lines.append(
            "| "
            + " | ".join(
                map(
                    cell,
                    (
                        f"`{info.name}`",
                        info.category,
                        info.summary,
                        info.predict_notes,
                        info.provenance,
                        info.metrics_status,
                    ),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def categories() -> str:
    honest = [info.name for info in MODEL_REGISTRY.values() if info.category == "honest"]
    decoupled = [info for info in MODEL_REGISTRY.values() if info.category == "display-predict decoupled"]
    lines = [
        "Model categories:",
        "",
        "- **Honest:** " + ", ".join(f"`{name}`" for name in honest) + ". The display closely matches what `predict` computes.",
        "  - `predict computes exactly the displayed form.`",
        "- **Display-predict decoupled:** "
        + ", ".join(f"`{info.name}`" for info in decoupled)
        + ". The display is a readable summary, while `predict` may include a hidden residual corrector or teacher ensemble.",
    ]
    lines.extend(f"  - `{info.name}`: `{info.predict_notes}`" for info in decoupled)
    return "\n".join(lines)


def main() -> None:
    print("## README: Model Guide")
    print()
    print(model_guide())
    print()
    print(FOOTNOTE)
    print()
    print("## docs/model-selection.md: Full Table")
    print()
    print(full_table())
    print()
    print(FOOTNOTE)
    print()
    print("## SKILL.md: Choosing A Model")
    print()
    print(choosing_table())
    print()
    print(FOOTNOTE)
    print()
    print("## SKILL.md: Model categories")
    print()
    print(categories())


if __name__ == "__main__":
    main()
