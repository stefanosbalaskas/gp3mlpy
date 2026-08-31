from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable

import numpy as np
import pandas as pd

import gp3mlpy
from gp3mlpy.objects import GP3MLResampleEvaluation


def _scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return str(value)


def _frame(data: pd.DataFrame, *, sort_by: list[str] | None = None) -> dict[str, Any]:
    frame = data.copy()
    if sort_by and len(frame):
        keys = [name for name in sort_by if name in frame.columns]
        if keys:
            frame = frame.sort_values(keys, kind="stable", na_position="last").reset_index(drop=True)
    rows = [
        {str(name): _scalar(value) for name, value in row.items()}
        for _, row in frame.iterrows()
    ]
    return {"columns": [str(name) for name in frame.columns], "rows": rows}


def _capture(call: Callable[[], Any], normalize: Callable[[Any], Any] | None = None) -> dict[str, Any]:
    try:
        value = call()
        return {"status": "success", "value": value if normalize is None else normalize(value)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _task(spec: dict[str, Any]):
    n = len(spec["truth"])
    data = pd.DataFrame(
        {
            "trial_id": [f"T{i:02d}" for i in range(1, n + 1)],
            "participant_id": [f"P{i:02d}" for i in range(1, n + 1)],
            "stimulus_id": [f"S{((i - 1) % 2) + 1:02d}" for i in range(1, n + 1)],
            "quality_status": pd.Categorical(spec["truth"], categories=["pass", "review"]),
        }
    )
    task = gp3mlpy.declare_gazepoint_task(
        data,
        outcome="quality_status",
        purpose="Predict observed recording quality.",
        task_type="classification",
        unit_id="trial_id",
        participant_id="participant_id",
        stimulus_id="stimulus_id",
        generalization_target="new_participants",
        positive="review",
    )
    return data, task


def _predictions(spec: dict[str, Any]) -> pd.DataFrame:
    n = len(spec["truth"])
    return pd.DataFrame(
        {
            ".gp3ml_source_row": list(range(1, n + 1)),
            "trial_id": [f"T{i:02d}" for i in range(1, n + 1)],
            "participant_id": [f"P{i:02d}" for i in range(1, n + 1)],
            "stimulus_id": [f"S{((i - 1) % 2) + 1:02d}" for i in range(1, n + 1)],
            "repeat": [1, 1, 1, 1],
            "fold": [1, 1, 2, 2],
            "fold_id": ["Repeat01_Fold01", "Repeat01_Fold01", "Repeat01_Fold02", "Repeat01_Fold02"],
            "candidate_id": [None] * n,
            "stage": ["assessment"] * n,
            "truth": spec["truth"],
            "prediction": spec["prediction"],
            "probability": spec["probability"],
            "prediction_missing": [False] * n,
        }
    )


def _fold_status(*, failed: bool = False) -> pd.DataFrame:
    rows = [
        {
            "repeat": 1, "fold": 1, "fold_id": "Repeat01_Fold01", "status": "pass",
            "leakage_status": "pass", "n_analysis": 2, "n_assessment": 2, "n_excluded": 0,
            "n_predictions": 2, "n_missing_predictions": 0, "warning_count": 0,
            "error": None, "warnings": "",
        },
        {
            "repeat": 1, "fold": 2, "fold_id": "Repeat01_Fold02", "status": "pass",
            "leakage_status": "pass", "n_analysis": 2, "n_assessment": 2, "n_excluded": 0,
            "n_predictions": 2, "n_missing_predictions": 0, "warning_count": 0,
            "error": None, "warnings": "",
        },
    ]
    if failed:
        rows.append(
            {
                "repeat": 1, "fold": 3, "fold_id": "Repeat01_Fold03", "status": "fail",
                "leakage_status": "pass", "n_analysis": 2, "n_assessment": 2, "n_excluded": 0,
                "n_predictions": 0, "n_missing_predictions": 2, "warning_count": 0,
                "error": "synthetic fold failure", "warnings": "",
            }
        )
    return pd.DataFrame(rows)


def _evaluation(spec: dict[str, Any], metrics_spec: list[dict[str, Any]], *, failed: bool = False) -> GP3MLResampleEvaluation:
    _, task = _task(spec)
    status = _fold_status(failed=failed)
    obj = GP3MLResampleEvaluation(
        fold_results=[],
        predictions=_predictions(spec),
        metrics=pd.DataFrame(metrics_spec),
        fold_status=status,
        excluded=pd.DataFrame(columns=["trial_id", "repeat", "fold", "fold_id"]),
        task=task,
        predictors=["tracking_ratio"],
        engine="glm",
        preprocessor_args={},
        engine_args={},
        threshold=float(spec["threshold"]),
        seed=101,
        generalization_target="new_participants",
        folds_metadata={"n_folds_total": int(len(status))},
        folds_audit=None,
        folds_validation=None,
        keep_models=False,
        call="parity_fixture",
    )
    obj.validation = gp3mlpy.validate_gazepoint_resample_evaluation(obj)
    return obj


def _validation(value: Any) -> dict[str, Any]:
    return {
        "class": value.r_class,
        "status": value.status,
        "checks": _frame(value.checks, sort_by=["check_id"]),
        "issues": _frame(value.issues, sort_by=["check_id"]),
    }


def _summary(value: Any) -> dict[str, Any]:
    return {
        "class": value.r_class,
        "aggregation": value.aggregation,
        "conf_level": float(value.conf_level),
        "generalization_target": value.generalization_target,
        "n_folds": int(value.n_folds),
        "n_failed_folds": int(value.n_failed_folds),
        "summary": _frame(value.summary, sort_by=["metric"]),
    }


def _read_csv(path: Path) -> dict[str, Any]:
    try:
        frame = pd.read_csv(path, keep_default_na=True)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame()
    return _frame(frame)


def _feature_write(call: Callable[[Path], str], filename: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        returned = call(path)
        return {
            "basename": Path(returned).name,
            "exists": path.exists(),
            "table": _read_csv(path),
        }


def _resample_write(value: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        returned = gp3mlpy.write_gazepoint_resample_evaluation(
            value, tmp, prefix="parity_eval", overwrite=False
        )
        tables = {
            key: {
                "basename": Path(path).name,
                "exists": Path(path).exists(),
                "table": _read_csv(Path(path)),
            }
            for key, path in returned.items()
        }
        return {"keys": list(returned.keys()), "tables": tables}


def _feature_overwrite(manifest: pd.DataFrame):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.csv"
        gp3mlpy.write_gazepoint_feature_manifest_csv(manifest, path)
        return gp3mlpy.write_gazepoint_feature_manifest_csv(manifest, path, overwrite=False)


def _feature_invalid_extension(manifest: pd.DataFrame):
    with tempfile.TemporaryDirectory() as tmp:
        return gp3mlpy.write_gazepoint_feature_manifest_csv(manifest, Path(tmp) / "manifest.txt")


def _feature_plain_checks(manifest: pd.DataFrame):
    with tempfile.TemporaryDirectory() as tmp:
        return gp3mlpy.write_gazepoint_feature_manifest_csv(
            manifest, Path(tmp) / "checks.csv", table="checks"
        )


def _resample_overwrite(value: Any):
    with tempfile.TemporaryDirectory() as tmp:
        gp3mlpy.write_gazepoint_resample_evaluation(
            value, tmp, prefix="parity_eval", overwrite=False
        )
        return gp3mlpy.write_gazepoint_resample_evaluation(
            value, tmp, prefix="parity_eval", overwrite=False
        )


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python parity/run_python_resample_outputs.py <fixture.json> <output.json>"
        )
    fixture_path, output_path = map(Path, sys.argv[1:])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    spec = fixture["classification"]

    clean = _evaluation(spec, fixture["metrics"], failed=False)
    failed = _evaluation(spec, fixture["metrics"], failed=True)

    bad_stage = _evaluation(spec, fixture["metrics"], failed=False)
    bad_stage.predictions = bad_stage.predictions.copy()
    bad_stage.predictions.loc[0, "stage"] = "analysis"

    incomplete = _evaluation(spec, fixture["metrics"], failed=False)
    incomplete.folds_metadata = dict(incomplete.folds_metadata)
    incomplete.folds_metadata["n_folds_total"] = 3

    target_mismatch = _evaluation(spec, fixture["metrics"], failed=False)
    target_mismatch.generalization_target = "new_stimuli"

    no_predictions = _evaluation(spec, fixture["metrics"], failed=False)
    no_predictions.predictions = no_predictions.predictions.iloc[0:0].copy()

    manifest_spec = fixture["manifest"]
    manifest = gp3mlpy.create_gazepoint_feature_manifest(
        manifest_spec["features"],
        scientific_source=manifest_spec["scientific_source"],
        source_table=manifest_spec["source_table"],
        transformation=manifest_spec["transformation"],
        availability_stage=manifest_spec["availability_stage"],
        prediction_time_available=manifest_spec["prediction_time_available"],
        preprocessing_scope=manifest_spec["preprocessing_scope"],
        fold_local_required=manifest_spec["fold_local_required"],
    )
    manifest_validation = gp3mlpy.validate_gazepoint_feature_manifest(manifest)

    feature_cases = {
        "manifest_success": _capture(
            lambda: _feature_write(
                lambda path: gp3mlpy.write_gazepoint_feature_manifest_csv(manifest, path),
                "manifest.csv",
            )
        ),
        "validation_checks_success": _capture(
            lambda: _feature_write(
                lambda path: gp3mlpy.write_gazepoint_feature_manifest_csv(
                    manifest_validation, path, table="checks"
                ),
                "checks.csv",
            )
        ),
        "validation_issues_success": _capture(
            lambda: _feature_write(
                lambda path: gp3mlpy.write_gazepoint_feature_manifest_csv(
                    manifest_validation, path, table="issues"
                ),
                "issues.csv",
            )
        ),
        "overwrite_refusal": _capture(lambda: _feature_overwrite(manifest)),
        "invalid_extension": _capture(lambda: _feature_invalid_extension(manifest)),
        "plain_checks_rejection": _capture(lambda: _feature_plain_checks(manifest)),
    }

    collector_cases = {
        "clean_predictions": _capture(
            lambda: gp3mlpy.collect_gazepoint_fold_predictions(clean),
            lambda value: _frame(value, sort_by=["repeat", "fold", ".gp3ml_source_row"]),
        ),
        "failed_include": _capture(
            lambda: gp3mlpy.collect_gazepoint_fold_predictions(failed, include_failed=True),
            lambda value: _frame(value, sort_by=["repeat", "fold", ".gp3ml_source_row"]),
        ),
        "failed_exclude": _capture(
            lambda: gp3mlpy.collect_gazepoint_fold_predictions(failed, include_failed=False),
            lambda value: _frame(value, sort_by=["repeat", "fold", ".gp3ml_source_row"]),
        ),
        "invalid_object": _capture(lambda: gp3mlpy.collect_gazepoint_fold_predictions({})),
    }

    summary_cases = {
        "fold_distribution": _capture(
            lambda: gp3mlpy.summarize_gazepoint_resample_performance(
                clean, aggregation="fold_distribution", conf_level=0.8
            ),
            _summary,
        ),
        "pooled_rows": _capture(
            lambda: gp3mlpy.summarize_gazepoint_resample_performance(
                clean, aggregation="pooled_rows", conf_level=0.8
            ),
            _summary,
        ),
        "no_predictions": _capture(
            lambda: gp3mlpy.summarize_gazepoint_resample_performance(
                no_predictions, aggregation="pooled_rows"
            )
        ),
        "invalid_aggregation": _capture(
            lambda: gp3mlpy.summarize_gazepoint_resample_performance(clean, aggregation="invalid")
        ),
        "invalid_object": _capture(lambda: gp3mlpy.summarize_gazepoint_resample_performance({})),
    }

    validation_cases = {
        "clean_validation": _capture(lambda: gp3mlpy.validate_gazepoint_resample_evaluation(clean), _validation),
        "failed_validation": _capture(lambda: gp3mlpy.validate_gazepoint_resample_evaluation(failed), _validation),
        "bad_stage_validation": _capture(lambda: gp3mlpy.validate_gazepoint_resample_evaluation(bad_stage), _validation),
        "incomplete_status_validation": _capture(lambda: gp3mlpy.validate_gazepoint_resample_evaluation(incomplete), _validation),
        "target_mismatch_validation": _capture(lambda: gp3mlpy.validate_gazepoint_resample_evaluation(target_mismatch), _validation),
        "invalid_object": _capture(lambda: gp3mlpy.validate_gazepoint_resample_evaluation({})),
    }

    writer_cases = {
        "clean_write": _capture(lambda: _resample_write(clean)),
        "overwrite_refusal": _capture(lambda: _resample_overwrite(clean)),
        "invalid_object": _capture(
            lambda: gp3mlpy.write_gazepoint_resample_evaluation({}, tempfile.gettempdir())
        ),
    }

    result = {
        "schema_version": 1,
        "runtime": "python",
        "r_reference_version": "0.3.0",
        "feature_writer_cases": feature_cases,
        "collector_cases": collector_cases,
        "summary_cases": summary_cases,
        "validation_cases": validation_cases,
        "writer_cases": writer_cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
