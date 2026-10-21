"""Source-only synthetic illustration of unit leakage in predictive validation.

Not part of the installed gp3mlpy public API. Outcomes are randomized at the
participant level and deliberately contain no transportable predictor signal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier


@dataclass(frozen=True)
class SplitResult:
    repeat: int
    strategy: str
    accuracy: float
    train_participants: int
    test_participants: int
    shared_participants: int


def make_synthetic_repeated_data(*, n_participants: int = 48, trials: int = 6, seed: int = 20261021):
    """Return participant-specific fingerprints and independently randomized labels.

    No generalizable association between a new participant's fingerprint and
    outcome is built into this source. This is an illustration, not a benchmark.
    """
    if n_participants < 8 or n_participants % 2:
        raise ValueError("n_participants must be even and >= 8")
    if trials < 2:
        raise ValueError("trials must be >= 2")
    rng = np.random.default_rng(seed)
    participant_ids = np.repeat(np.arange(n_participants), trials)
    fingerprint = rng.normal(size=(n_participants, 8))
    X = fingerprint[participant_ids] + 0.03 * rng.normal(size=(len(participant_ids), 8))
    y_person = rng.permutation(np.tile([0, 1], n_participants // 2))
    return X, y_person[participant_ids], participant_ids


def evaluate_split_strategies(*, repeats: int = 32, seed: int = 20261021) -> list[SplitResult]:
    """Contrast row-held-out with unit-held-out generalization on one frozen source.

    Repeated partitions of the SAME synthetic source are dependent and cannot
    be treated as independent experiments or used for coverage inference.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    X, y, groups = make_synthetic_repeated_data(seed=seed)
    results: list[SplitResult] = []
    for repeat in range(repeats):
        random_state = seed + repeat
        obs_train, obs_test = train_test_split(
            np.arange(len(groups)), test_size=0.25, random_state=random_state, stratify=y
        )
        grouped = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
        group_train, group_test = next(grouped.split(X, y, groups))
        for strategy, train, test in (
            ("Observation holdout", obs_train, obs_test),
            ("Participant holdout", group_train, group_test),
        ):
            train_units = set(groups[train].tolist())
            test_units = set(groups[test].tolist())
            overlap = len(train_units & test_units)
            if strategy == "Participant holdout" and overlap:
                raise AssertionError("Participant holdout leaked units")
            model = KNeighborsClassifier(n_neighbors=1)
            model.fit(X[train], y[train])
            score = accuracy_score(y[test], model.predict(X[test]))
            results.append(SplitResult(
                repeat=repeat,
                strategy=strategy,
                accuracy=float(score),
                train_participants=len(train_units),
                test_participants=len(test_units),
                shared_participants=overlap,
            ))
    return results


def write_audit_csv(rows: list[SplitResult], path: str | Path) -> None:
    """Write the complete repeat-level descriptive ledger, not just an average."""
    if not rows:
        raise ValueError("Cannot write an empty audit ledger")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


if __name__ == "__main__":
    result = evaluate_split_strategies()
    write_audit_csv(result, Path(__file__).with_name("split_audit.csv"))
    for strategy in ("Observation holdout", "Participant holdout"):
        selected = [r for r in result if r.strategy == strategy]
        mean = sum(r.accuracy for r in selected) / len(selected)
        overlap = sum(r.shared_participants for r in selected) / len(selected)
        print(f"{strategy}: {mean:.4f} mean accuracy; {overlap:.1f} mean overlapping units")
