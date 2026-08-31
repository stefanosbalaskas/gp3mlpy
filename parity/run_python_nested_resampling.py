from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

import gp3mlpy as gp


PREDICTORS = ["tracking_ratio", "blink_rate"]


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        return [_jsonable(v) for v in list(value)]
    return str(value) if not isinstance(value, str) else value


def _capture(fn) -> dict[str, Any]:
    try:
        return {"status": "success", "value": _jsonable(fn())}
    except Exception as exc:
        return {"status": "error", "error_type": type(exc).__name__, "message": str(exc)}


def _data() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    trial = 0
    for participant_index in range(1, 13):
        for stimulus_index in range(1, 5):
            trial += 1
            tracking_ratio = 0.50 + ((participant_index * 3 + stimulus_index * 2) % 13) / 25.0
            blink_rate = 3.0 + ((participant_index * 7 + stimulus_index * 5) % 17) / 2.0
            quality = (
                "review"
                if ((participant_index + 2 * stimulus_index + (participant_index * stimulus_index) % 3) % 4) in {0, 1}
                else "pass"
            )
            rows.append(
                {
                    "participant_id": f"P{participant_index:02d}",
                    "trial_id": f"T{trial:03d}",
                    "stimulus_id": f"S{stimulus_index:02d}",
                    "tracking_ratio": tracking_ratio,
                    "blink_rate": blink_rate,
                    "quality_status": quality,
                }
            )
    out = pd.DataFrame(rows)
    out["quality_status"] = pd.Categorical(out["quality_status"], categories=["pass", "review"])
    return out


def _task(data: pd.DataFrame):
    return gp.create_gazepoint_synthetic_task(data, "recording_quality", "new_participants")


def _outer_folds(data: pd.DataFrame, seed: int):
    manifest = gp.create_gazepoint_synthetic_manifest("quality_status", PREDICTORS)
    return gp.create_gazepoint_group_folds(
        data,
        "quality_status",
        PREDICTORS,
        manifest,
        "new_participants",
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id="stimulus_id",
        v=3,
        repeats=1,
        seed=seed,
    )


def _max_numeric(frame: pd.DataFrame, columns: list[str]) -> float | None:
    values: list[float] = []
    for column in columns:
        if column not in frame:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        values.extend(float(v) for v in series)
    return max(values) if values else None


def _nested_value(value: Any) -> dict[str, Any]:
    audit = value.audit.checks
    validation = value.validation.checks
    inner_counts = sorted(
        0 if item["inner"] is None else len(item["inner"].folds)
        for item in value.folds
    )
    return {
        "class": value.r_class,
        "n_outer": len(value.folds),
        "inner_ready": sum(item["inner"] is not None for item in value.folds),
        "inner_fold_counts": inner_counts,
        "inner_v": int(value.inner_v),
        "inner_repeats": int(value.inner_repeats),
        "target": str(value.outer_metadata["generalization_target"]),
        "failed_outer": sum(item["status"] == "fail" for item in value.folds),
        "audit_rows": len(audit),
        "audit_failures": int((audit.status == "fail").sum()),
        "max_outer_assessment_overlap": _max_numeric(
            audit,
            [
                "outer_assessment_inner_analysis_overlap",
                "outer_assessment_inner_assessment_overlap",
                "outer_assessment_inner_excluded_overlap",
                "outer_assessment_overlap",
            ],
        ),
        "max_inner_partition_overlap": _max_numeric(
            audit,
            [
                "inner_analysis_assessment_overlap",
                "inner_analysis_excluded_overlap",
                "inner_assessment_excluded_overlap",
            ],
        ),
        "validation_checks": sorted(str(x) for x in validation.check_id),
        "validation_failures": int((validation.status == "fail").sum()),
    }


def _audit_value(value: Any) -> dict[str, Any]:
    checks = value.checks
    return {
        "class": value.r_class,
        "rows": len(checks),
        "outer_folds": int(checks.outer_fold_id.nunique(dropna=True)),
        "inner_folds": int(checks.inner_fold_id.nunique(dropna=True)),
        "failures": int((checks.status == "fail").sum()),
        "max_outer_assessment_overlap": _max_numeric(
            checks,
            [
                "outer_assessment_inner_analysis_overlap",
                "outer_assessment_inner_assessment_overlap",
                "outer_assessment_inner_excluded_overlap",
                "outer_assessment_overlap",
            ],
        ),
        "max_inner_partition_overlap": _max_numeric(
            checks,
            [
                "inner_analysis_assessment_overlap",
                "inner_analysis_excluded_overlap",
                "inner_assessment_excluded_overlap",
            ],
        ),
    }


def _validation_value(value: Any) -> dict[str, Any]:
    checks = value.checks
    return {
        "class": value.r_class,
        "n_checks": len(checks),
        "check_ids": [str(x) for x in checks.check_id],
        "failures": int((checks.status == "fail").sum()),
    }


def _status_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts().to_dict()
    return {str(k): int(v) for k, v in sorted(counts.items())}


def _evaluation_value(value: Any) -> dict[str, Any]:
    prediction_fold_counts = (
        sorted(int(v) for v in value.predictions.fold_id.value_counts().tolist())
        if len(value.predictions)
        else []
    )
    selected = sorted(
        str(result["selection"].candidate_id)
        for result in value.results
        if result["selection"] is not None
    )
    return {
        "class": value.r_class,
        "target": str(value.generalization_target),
        "predictors": [str(x) for x in value.predictors],
        "n_outer": len(value.fold_status),
        "status_counts": _status_counts(value.fold_status.status),
        "failed_outer": int((value.fold_status.status == "fail").sum()),
        "selection_count": len(selected),
        "selected_candidates": selected,
        "n_predictions": len(value.predictions),
        "prediction_fold_counts": prediction_fold_counts,
        "outer_assessment_only": bool(
            len(value.predictions) == 0 or (value.predictions.stage == "outer_assessment").all()
        ),
        "n_metrics": len(value.metrics),
        "metric_names": sorted(str(x) for x in pd.unique(value.metrics.metric)) if len(value.metrics) else [],
        "keep_models": bool(value.keep_models),
        "models_retained": sum(result["model"] is not None for result in value.results),
        "validation_failures": int((value.validation.checks.status == "fail").sum()),
        "partition_audit_failures": int((value.nested_folds_audit.checks.status == "fail").sum()),
    }


def _write_value(value: Any) -> dict[str, Any]:
    with TemporaryDirectory() as directory:
        paths = gp.write_gazepoint_nested_evaluation(
            value,
            directory,
            prefix="nested_parity",
            overwrite=False,
        )
        return {
            "table_names": sorted(str(name) for name in paths),
            "basenames": sorted(Path(path).name for path in paths.values()),
            "n_files": len(paths),
            "all_exist": all(Path(path).is_file() for path in paths.values()),
        }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python parity/run_python_nested_resampling.py <fixture.json> <output.json>"
        )
    fixture_path, output_path = map(Path, sys.argv[1:])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    seed = int(fixture["seed"])

    data = _data()
    task = _task(data)
    outer = _outer_folds(data, seed)
    nested = gp.create_gazepoint_nested_folds(
        outer,
        inner_v=2,
        inner_repeats=1,
        seed=seed,
    )
    failed_nested = gp.create_gazepoint_nested_folds(
        outer,
        inner_v=20,
        inner_repeats=1,
        seed=seed,
        continue_on_error=True,
    )
    grid = gp.create_gazepoint_tuning_grid("glm", thresholds=0.5)
    evaluation = gp.evaluate_gazepoint_nested_resampling(
        nested,
        task,
        grid,
        selection_metric="brier",
        direction="minimize",
        predictors=PREDICTORS,
        minimum_success_prop=0.5,
        seed=seed,
        keep_models=False,
        continue_on_error=True,
    )
    evaluation_keep = gp.evaluate_gazepoint_nested_resampling(
        nested,
        task,
        grid,
        selection_metric="brier",
        direction="minimize",
        predictors=PREDICTORS,
        minimum_success_prop=0.5,
        seed=seed,
        keep_models=True,
        continue_on_error=True,
    )
    failed_evaluation = gp.evaluate_gazepoint_nested_resampling(
        failed_nested,
        task,
        grid,
        selection_metric="brier",
        direction="minimize",
        predictors=PREDICTORS,
        minimum_success_prop=0.5,
        seed=seed,
        keep_models=False,
        continue_on_error=True,
    )

    cases: dict[str, dict[str, Any]] = {}

    cases["create_gazepoint_nested_folds::successful_nested_folds"] = _capture(
        lambda: _nested_value(nested)
    )
    cases["create_gazepoint_nested_folds::retain_inner_failures"] = _capture(
        lambda: _nested_value(failed_nested)
    )
    cases["create_gazepoint_nested_folds::invalid_outer"] = _capture(
        lambda: gp.create_gazepoint_nested_folds("not folds")
    )
    cases["create_gazepoint_nested_folds::invalid_inner_v"] = _capture(
        lambda: gp.create_gazepoint_nested_folds(outer, inner_v=1)
    )
    cases["create_gazepoint_nested_folds::invalid_inner_repeats"] = _capture(
        lambda: gp.create_gazepoint_nested_folds(outer, inner_repeats=0)
    )

    cases["audit_gazepoint_nested_resampling::successful_audit"] = _capture(
        lambda: _audit_value(gp.audit_gazepoint_nested_resampling(nested))
    )
    cases["audit_gazepoint_nested_resampling::failed_inner_audit"] = _capture(
        lambda: _audit_value(gp.audit_gazepoint_nested_resampling(failed_nested))
    )
    cases["audit_gazepoint_nested_resampling::invalid_object"] = _capture(
        lambda: gp.audit_gazepoint_nested_resampling("not nested")
    )

    cases["validate_gazepoint_nested_folds::successful_validation"] = _capture(
        lambda: _validation_value(gp.validate_gazepoint_nested_folds(nested))
    )
    cases["validate_gazepoint_nested_folds::failed_inner_validation"] = _capture(
        lambda: _validation_value(gp.validate_gazepoint_nested_folds(failed_nested))
    )
    cases["validate_gazepoint_nested_folds::invalid_object"] = _capture(
        lambda: gp.validate_gazepoint_nested_folds("not nested")
    )

    cases["evaluate_gazepoint_nested_resampling::successful_evaluation"] = _capture(
        lambda: _evaluation_value(evaluation)
    )
    cases["evaluate_gazepoint_nested_resampling::keep_models"] = _capture(
        lambda: _evaluation_value(evaluation_keep)
    )
    cases["evaluate_gazepoint_nested_resampling::retain_outer_failures"] = _capture(
        lambda: _evaluation_value(failed_evaluation)
    )
    cases["evaluate_gazepoint_nested_resampling::invalid_nested"] = _capture(
        lambda: gp.evaluate_gazepoint_nested_resampling(
            "not nested",
            task,
            grid,
            selection_metric="brier",
            direction="minimize",
        )
    )
    cases["evaluate_gazepoint_nested_resampling::invalid_grid"] = _capture(
        lambda: gp.evaluate_gazepoint_nested_resampling(
            nested,
            task,
            "not grid",
            selection_metric="brier",
            direction="minimize",
        )
    )
    cases["evaluate_gazepoint_nested_resampling::stop_on_outer_failure"] = _capture(
        lambda: gp.evaluate_gazepoint_nested_resampling(
            failed_nested,
            task,
            grid,
            selection_metric="brier",
            direction="minimize",
            predictors=PREDICTORS,
            seed=seed,
            continue_on_error=False,
        )
    )

    cases["validate_gazepoint_nested_evaluation::successful_validation"] = _capture(
        lambda: _validation_value(gp.validate_gazepoint_nested_evaluation(evaluation))
    )
    cases["validate_gazepoint_nested_evaluation::failed_validation"] = _capture(
        lambda: _validation_value(gp.validate_gazepoint_nested_evaluation(failed_evaluation))
    )
    cases["validate_gazepoint_nested_evaluation::invalid_object"] = _capture(
        lambda: gp.validate_gazepoint_nested_evaluation("not evaluation")
    )

    cases["write_gazepoint_nested_evaluation::successful_write"] = _capture(
        lambda: _write_value(evaluation)
    )

    def overwrite_case():
        with TemporaryDirectory() as directory:
            gp.write_gazepoint_nested_evaluation(
                evaluation, directory, prefix="nested_parity", overwrite=False
            )
            return gp.write_gazepoint_nested_evaluation(
                evaluation, directory, prefix="nested_parity", overwrite=False
            )

    cases["write_gazepoint_nested_evaluation::overwrite_rejected"] = _capture(overwrite_case)
    cases["write_gazepoint_nested_evaluation::invalid_object"] = _capture(
        lambda: gp.write_gazepoint_nested_evaluation("not evaluation", ".")
    )

    output = {
        "runtime": "python",
        "r_reference_version": fixture["r_reference"]["version"],
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
