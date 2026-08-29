from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gp3mlpy as gp
from gp3mlpy.plotting import plot_engine_capabilities


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "plots"


plt.rcParams.update(
    {
        "figure.figsize": (7.2, 4.2),
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.size": 9.5,
        "svg.hashsalt": "gp3mlpy-docs",
    }
)


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    fig.savefig(
        target,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "gp3mlpy documentation build"},
    )
    plt.close(fig)
    print(f"wrote {target.relative_to(ROOT)}")


def threshold_plot() -> None:
    truth = [
        "control",
        "control",
        "control",
        "control",
        "target",
        "target",
        "target",
        "target",
        "target",
        "control",
        "target",
        "control",
    ]
    probability = [0.08, 0.18, 0.36, 0.58, 0.42, 0.55, 0.63, 0.74, 0.91, 0.47, 0.67, 0.29]
    evaluation = gp.evaluate_gazepoint_thresholds(
        truth=truth,
        probability=probability,
        positive="target",
        thresholds=np.linspace(0.20, 0.80, 7),
    )
    fig = evaluation.plot(metric="balanced_accuracy")
    _save(fig, "threshold-evaluation.svg")


def shift_plot() -> None:
    development = pd.DataFrame(
        {
            "tracking_ratio": [0.94, 0.91, 0.93, 0.89, 0.96, 0.92, 0.90, 0.95],
            "fixation_duration": [212, 225, 219, 231, 208, 223, 228, 216],
            "condition": ["A", "A", "A", "B", "B", "B", "A", "B"],
        }
    )
    external = pd.DataFrame(
        {
            "tracking_ratio": [0.86, 0.88, 0.84, 0.90, 0.87, 0.89, 0.85, 0.91],
            "fixation_duration": [238, 244, 251, 235, 247, 242, 255, 239],
            "condition": ["A", "B", "B", "B", "B", "B", "A", "B"],
        }
    )
    audit = gp.audit_gazepoint_dataset_shift(
        development,
        external,
        predictors=["tracking_ratio", "fixation_duration", "condition"],
    )
    fig = audit.plot()
    _save(fig, "dataset-shift.svg")


def engine_plot() -> None:
    capabilities = gp.gp3ml_engine_capabilities()
    fig = plot_engine_capabilities(capabilities)
    _save(fig, "engine-capabilities.svg")


def governance_plot() -> None:
    profile = gp.create_gp3ml_governance_profile(
        {
            "task": object(),
            "feature_manifest": object(),
            "folds": object(),
            "performance": object(),
            "decision_rule": object(),
        }
    )
    audit = gp.audit_gp3ml_governance_profile(profile)
    fig = audit.plot()
    _save(fig, "governance-profile.svg")


def main() -> None:
    threshold_plot()
    shift_plot()
    engine_plot()
    governance_plot()


if __name__ == "__main__":
    main()
