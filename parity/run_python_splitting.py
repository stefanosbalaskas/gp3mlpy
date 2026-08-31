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
    return [{str(key): _scalar(value) for key, value in row.items()} for _, row in data.iterrows()]


def _capture(call: Callable[[], Any], normalize: Callable[[Any], Any]) -> dict[str, Any]:
    try:
        return {"status": "success", "value": normalize(call())}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _data() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source = 0
    for participant in range(1, 9):
        for stimulus in range(1, 5):
            for repetition in range(1, 3):
                source += 1
                rows.append(
                    {
                        "participant_id": f"P{participant:02d}",
                        "trial_id": f"S{stimulus:02d}_R{repetition}",
                        "stimulus_id": f"S{stimulus:02d}",
                        "fixation_duration": float(150 + source),
                        "pupil_change": float((source % 9) - 4) / 10.0,
                        "quality_status": "pass" if (participant + stimulus + repetition) % 2 == 0 else "review",
                    }
                )
    data = pd.DataFrame(rows)
    data["quality_status"] = pd.Categorical(data["quality_status"], categories=["pass", "review"])
    return data


def _manifest(clean: bool = True, extra_feature: bool = False) -> pd.DataFrame:
    features = PREDICTORS + (["extra_feature"] if extra_feature else [])
    if not clean:
        return gp3mlpy.create_gazepoint_feature_manifest(features=features)
    n = len(features)
    return gp3mlpy.create_gazepoint_feature_manifest(
        features=features,
        scientific_source=["Gazepoint export"] * n,
        source_table=["all_gaze"] * n,
        transformation=["Predefined trial-level feature"] * n,
        availability_stage=["during_exposure"] * n,
        prediction_time_available=[True] * n,
        outcome_derived=[False] * n,
        post_outcome=[False] * n,
        identifier=[False] * n,
        preprocessing_scope=["none"] * n,
        fold_local_required=[False] * n,
        reviewer_notes=[""] * n,
    )


def _split(target: str, data: pd.DataFrame | None = None, manifest: pd.DataFrame | None = None, **overrides: Any) -> Any:
    arguments: dict[str, Any] = {
        "data": _data() if data is None else data,
        "outcome": "quality_status",
        "predictors": PREDICTORS,
        "feature_manifest": _manifest() if manifest is None else manifest,
        "generalization_target": target,
        "participant_id": "participant_id",
        "trial_id": "trial_id",
        "stimulus_id": "stimulus_id",
        "assessment_prop": 0.25,
        "seed": 101,
    }
    arguments.update(overrides)
    return gp3mlpy.split_gazepoint_ml_data(**arguments)


def _trial_units(data: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(data["participant_id"].astype(str), data["trial_id"].astype(str), strict=True))


def _success(split: Any, target: str) -> dict[str, Any]:
    repeated = _split(target)
    analysis_source = split.analysis[split.metadata["source_row_id"]].astype(int).tolist()
    assessment_source = split.assessment[split.metadata["source_row_id"]].astype(int).tolist()
    excluded_source = split.excluded[split.metadata["source_row_id"]].astype(int).tolist()
    repeated_partition = repeated.assignment["partition"].astype(str).tolist()
    partition = split.assignment["partition"].astype(str).tolist()
    source_all = analysis_source + assessment_source + excluded_source

    analysis_participant = set(split.analysis["participant_id"].astype(str))
    assessment_participant = set(split.assessment["participant_id"].astype(str))
    analysis_stimulus = set(split.analysis["stimulus_id"].astype(str))
    assessment_stimulus = set(split.assessment["stimulus_id"].astype(str))

    target_invariants: dict[str, Any]
    if target == "new_trials_known_participants":
        target_invariants = {
            "analysis_participants": len(analysis_participant),
            "assessment_participants": len(assessment_participant),
            "participant_trial_overlap": len(_trial_units(split.analysis) & _trial_units(split.assessment)),
        }
    elif target == "new_participants":
        target_invariants = {
            "analysis_participants": len(analysis_participant),
            "assessment_participants": len(assessment_participant),
            "participant_overlap": len(analysis_participant & assessment_participant),
        }
    elif target == "new_stimuli":
        target_invariants = {
            "analysis_stimuli": len(analysis_stimulus),
            "assessment_stimuli": len(assessment_stimulus),
            "stimulus_overlap": len(analysis_stimulus & assessment_stimulus),
        }
    else:
        target_invariants = {
            "analysis_participants": len(analysis_participant),
            "assessment_participants": len(assessment_participant),
            "participant_overlap": len(analysis_participant & assessment_participant),
            "analysis_stimuli": len(analysis_stimulus),
            "assessment_stimuli": len(assessment_stimulus),
            "stimulus_overlap": len(analysis_stimulus & assessment_stimulus),
        }

    return {
        "class": split.r_class,
        "target": target,
        "summary": _frame(split.summary),
        "validation": {
            "class": split.validation.r_class,
            "status": split.validation.status,
            "summary": _frame(split.validation.summary),
            "checks": _frame(split.validation.checks),
            "issues": _frame(split.validation.issues),
        },
        "feature_manifest_status": split.feature_manifest_validation.status,
        "leakage_status": split.leakage_audit.status,
        "row_counts": {
            "analysis": len(split.analysis),
            "assessment": len(split.assessment),
            "excluded": len(split.excluded),
        },
        "source_rows_unique": len(source_all) == len(set(source_all)),
        "source_rows_accounted": sorted(source_all) == list(range(1, int(split.metadata["n_source_rows"]) + 1)),
        "same_seed_reproducible": partition == repeated_partition,
        "target_invariants": target_invariants,
    }


def _error_case(kind: str) -> Any:
    data = _data()
    if kind == "missing_manifest":
        return gp3mlpy.split_gazepoint_ml_data(
            data=data,
            outcome="quality_status",
            predictors=PREDICTORS,
            feature_manifest=None,  # type: ignore[arg-type]
            generalization_target="new_participants",
            participant_id="participant_id",
            trial_id="trial_id",
            stimulus_id="stimulus_id",
        )
    if kind == "review_manifest":
        return _split("new_participants", manifest=_manifest(clean=False))
    if kind == "missing_participant":
        return _split("new_participants", participant_id=None)
    if kind == "missing_stimulus":
        return _split("new_stimuli", stimulus_id=None)
    if kind == "bad_assessment_prop":
        return _split("new_participants", assessment_prop=1.0)
    if kind == "outcome_predictor":
        return _split("new_participants", predictors=["quality_status", "fixation_duration"])
    if kind == "identifier_predictor":
        return _split("new_participants", predictors=["participant_id", "fixation_duration"])
    if kind == "reserved_source_row":
        data[".gp3ml_source_row"] = np.arange(1, len(data) + 1)
        return _split("new_participants", data=data)
    if kind == "predictor_missing_manifest":
        manifest = gp3mlpy.create_gazepoint_feature_manifest(
            features="fixation_duration",
            scientific_source="Gazepoint export",
            source_table="all_gaze",
            transformation="Predefined trial-level feature",
            availability_stage="during_exposure",
            prediction_time_available=True,
            outcome_derived=False,
            post_outcome=False,
            identifier=False,
            preprocessing_scope="none",
            fold_local_required=False,
            reviewer_notes="",
        )
        return _split("new_participants", manifest=manifest)
    if kind == "too_few_trials":
        data.loc[data["participant_id"] == "P01", "trial_id"] = "ONLY"
        return _split("new_trials_known_participants", data=data)
    raise RuntimeError(f"Unknown split error case: {kind}")


def _validation(value: Any) -> dict[str, Any]:
    return {
        "class": value.r_class,
        "status": value.status,
        "summary": _frame(value.summary),
        "checks": _frame(value.checks),
        "issues": _frame(value.issues),
    }


def _validation_case(kind: str) -> Any:
    if kind == "invalid_object":
        return gp3mlpy.validate_gazepoint_ml_split({})  # type: ignore[arg-type]
    split = _split("new_participants")
    if kind == "clean":
        return gp3mlpy.validate_gazepoint_ml_split(split)
    if kind == "source_overlap":
        source = split.metadata["source_row_id"]
        split.assessment.loc[split.assessment.index[0], source] = split.analysis[source].iloc[0]
        return gp3mlpy.validate_gazepoint_ml_split(split)
    if kind == "missing_component":
        del split["assignment"]
        return gp3mlpy.validate_gazepoint_ml_split(split)
    raise RuntimeError(f"Unknown validation case: {kind}")


def _writer_case(kind: str) -> Any:
    split = _split("new_participants")
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        if kind == "bad_prefix":
            return gp3mlpy.write_gazepoint_ml_split_csv(split, root, prefix="bad/name", tables="summary")
        if kind == "bad_table":
            return gp3mlpy.write_gazepoint_ml_split_csv(split, root, tables="unknown")
        if kind == "overwrite":
            gp3mlpy.write_gazepoint_ml_split_csv(split, root, prefix="parity", tables="summary")
            try:
                gp3mlpy.write_gazepoint_ml_split_csv(split, root, prefix="parity", tables="summary")
            except Exception as exc:
                raise RuntimeError(str(exc).replace(root.resolve().as_posix(), "<TMP>").replace(str(root.resolve()), "<TMP>")) from None
            raise RuntimeError("Expected overwrite protection error.")
        table = "summary" if kind == "summary" else "checks"
        paths = gp3mlpy.write_gazepoint_ml_split_csv(split, root, prefix="parity", tables=table)
        path = Path(paths[table])
        exported = pd.read_csv(path, keep_default_na=False)
        return {"table": table, "basename": path.name, "columns": list(exported.columns), "rows": _frame(exported)}


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python parity/run_python_splitting.py <fixture.json> <output.json>")
    fixture = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output_path = Path(sys.argv[2])
    successes = {
        case["id"]: _capture(lambda case=case: _split(case["target"]), lambda value, case=case: _success(value, case["target"]))
        for case in fixture["success_cases"]
    }
    errors = {case["id"]: _capture(lambda case=case: _error_case(case["kind"]), lambda _: True) for case in fixture["error_cases"]}
    validations = {case["id"]: _capture(lambda case=case: _validation_case(case["kind"]), _validation) for case in fixture["validation_cases"]}
    writers = {case["id"]: _capture(lambda case=case: _writer_case(case["kind"]), lambda value: value) for case in fixture["writer_cases"]}
    result = {
        "runtime": "Python",
        "package": "gp3mlpy",
        "package_version": gp3mlpy.__version__,
        "r_reference_version": gp3mlpy.r_reference_version,
        "successes": successes,
        "errors": errors,
        "validations": validations,
        "writers": writers,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
