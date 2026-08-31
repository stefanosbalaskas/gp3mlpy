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

PREDICTORS = ["fixation_duration", "pupil_change"]


def _scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return str(value)


def _frame(data: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _scalar(value) for key, value in row.items()}
        for _, row in data.iterrows()
    ]


def _capture(call: Callable[[], Any], normalize: Callable[[Any], Any]) -> dict[str, Any]:
    try:
        return {"status": "success", "value": normalize(call())}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _data() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    index = 0
    for participant in range(1, 7):
        for stimulus in range(1, 5):
            for repetition in range(1, 4):
                index += 1
                rows.append(
                    {
                        "participant_id": f"P{participant:02d}",
                        "stimulus_id": f"S{stimulus:02d}",
                        "trial_id": f"S{stimulus:02d}_T{repetition}",
                        "outcome": int(index % 2),
                        "fixation_duration": float(180 + index),
                        "pupil_change": float(index) / 1000.0,
                    }
                )
    return pd.DataFrame(rows)


def _manifest() -> pd.DataFrame:
    return gp3mlpy.create_gazepoint_feature_manifest(
        features=PREDICTORS,
        scientific_source=["Gazepoint fixation export", "Gazepoint all-gaze export"],
        source_table=["fixations", "all_gaze"],
        transformation=["Trial-level mean", "Baseline-adjusted change"],
        availability_stage=["during_exposure", "during_exposure"],
        prediction_time_available=[True, True],
        outcome_derived=[False, False],
        post_outcome=[False, False],
        identifier=[False, False],
        preprocessing_scope=["none", "none"],
        fold_local_required=[False, False],
        reviewer_notes=["", ""],
    )


def _plan(
    target: str = "new_participants",
    *,
    v: Any = 3,
    repeats: int = 2,
    seed: int = 77,
    data: pd.DataFrame | None = None,
    manifest: pd.DataFrame | None = None,
    **overrides: Any,
) -> Any:
    args: dict[str, Any] = {
        "data": _data() if data is None else data,
        "outcome": "outcome",
        "predictors": PREDICTORS,
        "feature_manifest": _manifest() if manifest is None else manifest,
        "generalization_target": target,
        "participant_id": "participant_id",
        "trial_id": "trial_id",
        "stimulus_id": "stimulus_id",
        "v": v,
        "repeats": repeats,
        "seed": seed,
    }
    args.update(overrides)
    return gp3mlpy.create_gazepoint_group_folds(**args)


def _assessment_coverage(plan: Any) -> pd.DataFrame:
    return (
        plan.assignments.assign(
            n_assessment=(plan.assignments["partition"] == "assessment").astype(int)
        )
        .groupby(["repeat", "source_row"], as_index=False)["n_assessment"]
        .sum()
    )


def _trial_units(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            frame["participant_id"].astype(str),
            frame["trial_id"].astype(str),
            strict=True,
        )
    )


def _success(plan: Any, case: dict[str, Any]) -> dict[str, Any]:
    repeated = _plan(
        case["target"],
        v=case["v"],
        repeats=int(case["repeats"]),
        seed=int(case["seed"]),
    )
    coverage = _assessment_coverage(plan)
    source_rows = list(range(1, int(plan.metadata["n_source_rows"]) + 1))

    all_accounted = True
    all_nonempty = True
    participant_overlap = 0
    stimulus_overlap = 0
    trial_overlap = 0
    excluded_positive = True
    assessment_has_all_participants = True
    for fold in plan.folds.values():
        assigned = plan.assignments[
            (plan.assignments["repeat"] == fold["repeat"])
            & (plan.assignments["fold"] == fold.fold)
        ]
        all_accounted &= sorted(assigned["source_row"].astype(int).tolist()) == source_rows
        all_nonempty &= len(fold.analysis) > 0 and len(fold.assessment) > 0
        participant_overlap += len(
            set(fold.analysis["participant_id"].astype(str))
            & set(fold.assessment["participant_id"].astype(str))
        )
        stimulus_overlap += len(
            set(fold.analysis["stimulus_id"].astype(str))
            & set(fold.assessment["stimulus_id"].astype(str))
        )
        trial_overlap += len(_trial_units(fold.analysis) & _trial_units(fold.assessment))
        if case["target"] == "new_participants_and_new_stimuli":
            excluded_positive &= len(fold.excluded) > 0
        else:
            excluded_positive &= len(fold.excluded) == 0
        if case["target"] == "new_trials_known_participants":
            assessment_has_all_participants &= set(
                fold.assessment["participant_id"].astype(str)
            ) == set(_data()["participant_id"].astype(str))

    target = case["target"]
    invariants = {
        "participant_overlap_zero": participant_overlap == 0
        if target in {"new_participants", "new_participants_and_new_stimuli"}
        else True,
        "stimulus_overlap_zero": stimulus_overlap == 0
        if target in {"new_stimuli", "new_participants_and_new_stimuli"}
        else True,
        "participant_trial_overlap_zero": trial_overlap == 0
        if target == "new_trials_known_participants"
        else True,
        "assessment_has_all_participants": assessment_has_all_participants,
        "excluded_behavior_valid": excluded_positive,
    }

    return {
        "class": plan.r_class,
        "target": target,
        "metadata": {
            "repeats": int(plan.metadata["repeats"]),
            "n_source_rows": int(plan.metadata["n_source_rows"]),
            "n_folds_per_repeat": int(plan.metadata["n_folds_per_repeat"]),
            "n_folds_total": int(plan.metadata["n_folds_total"]),
        },
        "fold_count": len(plan.folds),
        "fold_ids_unique": len({fold.fold_id for fold in plan.folds.values()}) == len(plan.folds),
        "same_seed_reproducible": plan.assignments.equals(repeated.assignments),
        "source_rows_accounted_per_fold": all_accounted,
        "all_analysis_assessment_nonempty": all_nonempty,
        "assessment_once_per_repeat": bool((coverage["n_assessment"] == 1).all()),
        "audit": {
            "class": plan.audit.r_class,
            "status": plan.audit.status,
            "n_summary": len(plan.audit.summary),
            "n_issues": len(plan.audit.issues),
        },
        "validation": {
            "class": plan.validation.r_class,
            "status": plan.validation.status,
            "summary": _frame(plan.validation.summary),
            "checks": _frame(plan.validation.checks),
            "issues": _frame(plan.validation.issues),
        },
        "invariants": invariants,
    }


def _error_case(kind: str) -> Any:
    if kind == "missing_manifest":
        return _plan(manifest=None, feature_manifest=None)  # type: ignore[arg-type]
    if kind == "missing_participant":
        return _plan(participant_id=None)
    if kind == "v_vector_single_target":
        return _plan(v=[2, 3])
    if kind == "v_too_many_participants":
        return _plan(v=7)
    if kind == "v_too_many_stimuli":
        return _plan("new_stimuli", v=5)
    if kind == "insufficient_trials":
        data = _data()
        data.loc[data["participant_id"] == "P01", "trial_id"] = "ONLY"
        return _plan("new_trials_known_participants", v=3, data=data)
    raise RuntimeError(f"Unknown resampling error case: {kind}")


def _audit(value: Any) -> dict[str, Any]:
    return {
        "class": value.r_class,
        "status": value.status,
        "summary": _frame(value.summary),
        "issues": _frame(value.issues),
    }


def _audit_case(kind: str) -> Any:
    if kind == "invalid_object":
        return gp3mlpy.audit_gazepoint_group_folds({})  # type: ignore[arg-type]
    plan = _plan(repeats=1)
    if kind == "clean":
        return gp3mlpy.audit_gazepoint_group_folds(plan)
    if kind == "empty_folds":
        plan.folds = {}
        return gp3mlpy.audit_gazepoint_group_folds(plan)
    raise RuntimeError(f"Unknown audit case: {kind}")


def _validation(value: Any) -> dict[str, Any]:
    return {
        "class": value.r_class,
        "status": value.status,
        "summary": _frame(value.summary),
        "checks": _frame(value.checks),
        "issues": _frame(value.issues),
        "assessment_coverage": _frame(value.assessment_coverage),
    }


def _validation_case(kind: str) -> Any:
    if kind == "invalid_object":
        return gp3mlpy.validate_gazepoint_group_folds({})  # type: ignore[arg-type]
    plan = _plan(repeats=1)
    if kind == "clean":
        return gp3mlpy.validate_gazepoint_group_folds(plan)
    if kind == "assignment_damage":
        position = plan.assignments.index[plan.assignments["partition"] == "assessment"][0]
        plan.assignments.loc[position, "partition"] = "analysis"
        return gp3mlpy.validate_gazepoint_group_folds(plan)
    if kind == "missing_component":
        del plan["audit"]
        return gp3mlpy.validate_gazepoint_group_folds(plan)
    raise RuntimeError(f"Unknown validation case: {kind}")


def _writer_case(kind: str) -> Any:
    plan = _plan(repeats=1)
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        if kind == "bad_prefix":
            return gp3mlpy.write_gazepoint_group_folds_csv(plan, root, prefix="bad/name")
        if kind == "bad_table":
            return gp3mlpy.write_gazepoint_group_folds_csv(plan, root, tables="unknown")
        if kind == "overwrite":
            gp3mlpy.write_gazepoint_group_folds_csv(
                plan, root, prefix="parity", tables="fold_summary"
            )
            try:
                gp3mlpy.write_gazepoint_group_folds_csv(
                    plan, root, prefix="parity", tables="fold_summary"
                )
            except Exception as exc:
                message = str(exc).replace(root.resolve().as_posix(), "<TMP>")
                message = message.replace(str(root.resolve()), "<TMP>")
                raise RuntimeError(message) from None
            raise RuntimeError("Expected overwrite protection error.")
        if kind == "summary_tables":
            paths = gp3mlpy.write_gazepoint_group_folds_csv(
                plan,
                root,
                prefix="parity",
                tables=["assignments", "fold_summary"],
            )
            result: dict[str, Any] = {}
            for name in ["assignments", "fold_summary"]:
                path = Path(paths[name])
                frame = pd.read_csv(path, keep_default_na=False)
                result[name] = {
                    "basename": path.name,
                    "columns": list(frame.columns),
                    "n_rows": len(frame),
                }
            return result
        if kind == "include_fold_data":
            paths = gp3mlpy.write_gazepoint_group_folds_csv(
                plan,
                root,
                prefix="parity",
                tables="fold_summary",
                include_fold_data=True,
            )
            basenames = sorted(Path(path).name for path in paths.values())
            return {
                "n_paths": len(paths),
                "expected_n_paths": 1 + len(plan.folds) * 3,
                "all_exist": all(Path(path).exists() for path in paths.values()),
                "has_fold_summary": "parity_fold_summary.csv" in basenames,
                "n_analysis_files": sum(name.endswith("_analysis.csv") for name in basenames),
                "n_assessment_files": sum(name.endswith("_assessment.csv") for name in basenames),
                "n_excluded_files": sum(name.endswith("_excluded.csv") for name in basenames),
            }
    raise RuntimeError(f"Unknown writer case: {kind}")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python parity/run_python_resampling.py <fixture.json> <output.json>"
        )
    fixture = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output_path = Path(sys.argv[2])
    successes = {
        case["id"]: _capture(
            lambda case=case: _plan(
                case["target"],
                v=case["v"],
                repeats=int(case["repeats"]),
                seed=int(case["seed"]),
            ),
            lambda value, case=case: _success(value, case),
        )
        for case in fixture["success_cases"]
    }
    errors = {
        case["id"]: _capture(
            lambda case=case: _error_case(case["kind"]), lambda _: True
        )
        for case in fixture["error_cases"]
    }
    audits = {
        case["id"]: _capture(
            lambda case=case: _audit_case(case["kind"]), _audit
        )
        for case in fixture["audit_cases"]
    }
    validations = {
        case["id"]: _capture(
            lambda case=case: _validation_case(case["kind"]), _validation
        )
        for case in fixture["validation_cases"]
    }
    writers = {
        case["id"]: _capture(
            lambda case=case: _writer_case(case["kind"]), lambda value: value
        )
        for case in fixture["writer_cases"]
    }
    result = {
        "runtime": "Python",
        "package": "gp3mlpy",
        "package_version": gp3mlpy.__version__,
        "r_reference_version": gp3mlpy.r_reference_version,
        "successes": successes,
        "errors": errors,
        "audits": audits,
        "validations": validations,
        "writers": writers,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
