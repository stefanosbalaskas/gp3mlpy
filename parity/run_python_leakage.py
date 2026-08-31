from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import gp3mlpy


def _scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return str(value)


def _frame(data: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): _scalar(v) for k, v in row.items()} for _, row in data.iterrows()]


def _audit(value: Any) -> dict[str, Any]:
    return {
        "class": value.r_class,
        "status": value.status,
        "generalization_target": value.generalization_target,
        "outcome": value.outcome,
        "predictors": list(value.predictors),
        "roles": {
            "participant_id": value.roles["participant_id"],
            "trial_id": value.roles["trial_id"],
            "stimulus_id": value.roles["stimulus_id"],
            "target_derived": list(value.roles["target_derived"]),
            "post_outcome": list(value.roles["post_outcome"]),
        },
        "partition_summary": _frame(value.partition_summary),
        "checks": _frame(value.checks),
        "issues": _frame(value.issues),
    }


def _capture(call: Callable[[], Any], normalize: Callable[[Any], Any]) -> dict[str, Any]:
    try:
        return {"status": "success", "value": normalize(call())}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _base_partitions() -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = pd.DataFrame(
        {
            "participant_id": ["P01", "P01", "P02", "P02"],
            "trial_id": ["T01", "T02", "T03", "T04"],
            "stimulus_id": ["S01", "S02", "S03", "S04"],
            "outcome": [0, 1, 0, 1],
            "feature_a": [1.1, 1.2, 1.3, 1.4],
            "feature_b": [2.1, 2.2, 2.3, 2.4],
        }
    )
    assessment = pd.DataFrame(
        {
            "participant_id": ["P03", "P03", "P04", "P04"],
            "trial_id": ["T05", "T06", "T07", "T08"],
            "stimulus_id": ["S05", "S06", "S07", "S08"],
            "outcome": [1, 0, 1, 0],
            "feature_a": [1.5, 1.6, 1.7, 1.8],
            "feature_b": [2.5, 2.6, 2.7, 2.8],
        }
    )
    return analysis, assessment


def _case_call(kind: str, target: str) -> Any:
    analysis, assessment = _base_partitions()
    predictors: list[str] = ["feature_a", "feature_b"]
    participant_id: str | None = "participant_id"
    target_derived: list[str] = []
    post_outcome: list[str] = []

    if kind == "participant_overlap":
        assessment.loc[0, "participant_id"] = "P01"
    elif kind == "known_participant_trials":
        assessment["participant_id"] = ["P01", "P02", "P01", "P02"]
    elif kind == "participant_trial_overlap":
        assessment["participant_id"] = ["P01", "P02", "P01", "P02"]
        assessment.loc[0, "trial_id"] = "T01"
    elif kind == "reused_trial_labels":
        analysis["trial_id"] = ["T01", "T02", "T01", "T02"]
        assessment["trial_id"] = ["T01", "T02", "T01", "T02"]
    elif kind == "exact_row_overlap":
        assessment.iloc[0] = analysis.iloc[0]
    elif kind == "duplicate_within":
        analysis = pd.concat([analysis, analysis.iloc[[0]]], ignore_index=True)
    elif kind == "role_failures":
        analysis["target_proxy"] = [1, 0, 1, 0]
        assessment["target_proxy"] = [0, 1, 0, 1]
        analysis["post_metric"] = [4, 3, 2, 1]
        assessment["post_metric"] = [8, 7, 6, 5]
        predictors = ["outcome", "participant_id", "feature_a", "target_proxy", "post_metric"]
        target_derived = ["target_proxy"]
        post_outcome = ["post_metric"]
    elif kind == "identifier_like":
        analysis["record_index"] = [1, 2, 3, 4]
        assessment["record_index"] = [5, 6, 7, 8]
        predictors = ["feature_a", "feature_b", "record_index"]
    elif kind == "missing_participant":
        participant_id = None
    elif kind == "mismatched_columns_error":
        assessment["extra_column"] = 1
    elif kind == "missing_predictor_error":
        predictors = ["missing_predictor"]
    elif kind == "empty_partition_error":
        analysis = analysis.iloc[0:0].copy()
    elif kind != "clean":
        raise RuntimeError(f"Unknown leakage case kind: {kind}")

    return gp3mlpy.audit_gazepoint_ml_leakage(
        analysis=analysis,
        assessment=assessment,
        outcome="outcome",
        predictors=predictors,
        participant_id=participant_id,
        trial_id="trial_id",
        stimulus_id="stimulus_id",
        generalization_target=target,
        target_derived=target_derived,
        post_outcome=post_outcome,
    )


def _writer_case(kind: str) -> Any:
    audit = _case_call("participant_overlap", "new_participants")
    suffix = ".txt" if kind == "bad_extension" else ".csv"
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / f"audit{suffix}"
        table = "issues" if kind in {"issues", "bad_extension"} else "checks"
        result = gp3mlpy.write_gazepoint_ml_leakage_audit_csv(audit, path, table=table)
        exported = pd.read_csv(path, keep_default_na=False)
        return {
            "returned_extension": Path(result).suffix.lower(),
            "columns": list(exported.columns),
            "rows": _frame(exported),
        }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python parity/run_python_leakage.py <fixture.json> <output.json>")
    fixture = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(sys.argv[2])
    audits: dict[str, Any] = {}
    writers: dict[str, Any] = {}
    for case in fixture["audit_cases"]:
        audits[case["id"]] = _capture(
            lambda case=case: _case_call(case["kind"], case["generalization_target"]),
            _audit,
        )
    for case in fixture["writer_cases"]:
        writers[case["id"]] = _capture(lambda case=case: _writer_case(case["kind"]), lambda value: value)
    result = {
        "runtime": "Python",
        "package": "gp3mlpy",
        "package_version": gp3mlpy.__version__,
        "r_reference_version": gp3mlpy.r_reference_version,
        "audits": audits,
        "writers": writers,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
