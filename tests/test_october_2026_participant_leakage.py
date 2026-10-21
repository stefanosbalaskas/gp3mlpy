"""Regression contracts for the research-only synthetic split illustration."""
import csv
import importlib.util
from pathlib import Path
import sys

import pytest

MODULE = Path(__file__).resolve().parents[1] / "research/october_2026/participant_leakage_demo.py"
SPEC = importlib.util.spec_from_file_location("participant_leakage_demo", MODULE)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_generated_source_contract():
    X, y, groups = module.make_synthetic_repeated_data()
    assert X.shape == (288, 8)
    assert y.shape == groups.shape == (288,)
    assert len(set(groups)) == 48
    assert sum(y == 1) == 144
    assert all(len(set(y[groups == p])) == 1 for p in range(48))


def test_invalid_generators():
    with pytest.raises(ValueError, match="n_participants"):
        module.make_synthetic_repeated_data(n_participants=9)
    with pytest.raises(ValueError, match="trials"):
        module.make_synthetic_repeated_data(trials=1)
    with pytest.raises(ValueError, match="repeats"):
        module.evaluate_split_strategies(repeats=0)


def test_exact_repeat_and_unit_guard():
    a = module.evaluate_split_strategies(repeats=4)
    b = module.evaluate_split_strategies(repeats=4)
    assert a == b
    assert len(a) == 8
    observations = [r for r in a if r.strategy == "Observation holdout"]
    participants = [r for r in a if r.strategy == "Participant holdout"]
    assert len(observations) == len(participants) == 4
    assert all(r.shared_participants > 0 for r in observations)
    assert all(r.shared_participants == 0 for r in participants)
    assert all(0 <= r.accuracy <= 1 for r in a)


def test_csv_ledger(tmp_path):
    rows = module.evaluate_split_strategies(repeats=2)
    destination = tmp_path / "audit.csv"
    module.write_audit_csv(rows, destination)
    with destination.open(encoding="utf-8", newline="") as handle:
        data = list(csv.DictReader(handle))
    assert len(data) == 4
    assert {r["strategy"] for r in data} == {"Observation holdout", "Participant holdout"}
    with pytest.raises(ValueError, match="empty"):
        module.write_audit_csv([], destination)
