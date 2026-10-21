"""Regenerate an original two-panel diagnostic figure from the synthetic ledger.

This source-only helper is not part of the gp3mlpy installed public API.
Run participant_leakage_demo.py first, then this script from any directory.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def render_figure(ledger_path: str | Path, figure_path: str | Path) -> None:
    """Plot all repeat-level values without misleading uncertainty bands."""
    with Path(ledger_path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Audit ledger is empty")
    expected = {"Observation holdout", "Participant holdout"}
    if {row["strategy"] for row in rows} != expected:
        raise ValueError("Audit ledger must contain both validation strategies")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for label, color in (
        ("Observation holdout", "#1877ad"),
        ("Participant holdout", "#cd673a"),
    ):
        chosen = sorted((r for r in rows if r["strategy"] == label), key=lambda r: int(r["repeat"]))
        x = [int(row["repeat"]) + 1 for row in chosen]
        axs[0].scatter(x, [float(row["accuracy"]) for row in chosen], label=label, s=22, color=color)
        axs[1].scatter(
            x, [int(row["shared_participants"]) for row in chosen], label=label, s=22, color=color
        )
    axs[0].axhline(0.5, linestyle="--", linewidth=1.0, color="gray")
    axs[0].set_ylim(-0.03, 1.07)
    axs[0].set_ylabel("Held-out accuracy")
    axs[0].set_title("A. Split accuracy")
    axs[1].set_ylim(-1.5, 48)
    axs[1].set_ylabel("Participants shared across partitions")
    axs[1].set_title("B. Participant overlap")
    for ax in axs:
        ax.set_xlabel("Dependent partition repeat (one source)")
        ax.grid(axis="y", alpha=0.25)
    axs[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Synthetic participant leakage: record versus participant holdout")
    output = Path(figure_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg")
    plt.close(fig)


if __name__ == "__main__":
    base = Path(__file__).parent
    render_figure(base / "split_audit.csv", base / "participant_leakage_regenerated.svg")
