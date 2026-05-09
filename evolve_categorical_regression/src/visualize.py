"""Plot categorical-regression interpretability vs. performance."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_GROUPS = {
    "black-box": {"RF", "GBM", "TabPFN"},
    "linear": {"OLS", "RidgeCV", "LassoCV"},
    "tree": {"DT_mini", "DT_large"},
    "additive": {"EBM"},
}
GROUP_COLORS = {
    "black-box": "black",
    "linear": "tab:green",
    "tree": "tab:red",
    "additive": "tab:orange",
}


def _is_known(name: str) -> bool:
    return any(name in members for members in MODEL_GROUPS.values())


def _color_for(name: str):
    if _is_known(name):
        for group, members in MODEL_GROUPS.items():
            if name in members:
                return GROUP_COLORS[group], "X"
    return "steelblue", "o"


def plot_interp_vs_performance(csv_path: str | Path, out_path: str | Path | None = None) -> None:
    csv_path = Path(csv_path)
    if out_path is None:
        out_path = csv_path.parent / "interpretability_vs_performance.png"
    out_path = Path(out_path)

    df = pd.read_csv(csv_path)
    required = {"mean_rank", "frac_interpretability_tests_passed", "model_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.replace("", np.nan).replace("nan", np.nan).dropna(subset=list(required)).copy()
    df["mean_rank"] = df["mean_rank"].astype(float)
    df["frac_interpretability_tests_passed"] = df["frac_interpretability_tests_passed"].astype(float)

    fig, ax = plt.subplots(figsize=(10, 10))
    for _, row in df.iterrows():
        color, marker = _color_for(str(row["model_name"]))
        ax.scatter(
            row["frac_interpretability_tests_passed"],
            row["mean_rank"],
            color=color,
            marker=marker,
            s=60,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.text(
            row["frac_interpretability_tests_passed"],
            row["mean_rank"],
            str(row["model_name"]),
            fontsize=8,
            color=color,
        )

    ax.set_xlabel("Interpretability (fraction tests passed)")
    ax.set_ylabel("Prediction performance (rank, lower is better)")
    ax.set_title("Categorical Regression: Interpretability vs. Performance")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {out_path}")

