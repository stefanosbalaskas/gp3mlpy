from __future__ import annotations

import copy
import json
import math
import tempfile
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd

import gp3mlpy
from gp3mlpy.objects import GP3MLResampleEvaluation


METRIC_EXCLUDE = {"n", "threshold", "replicate", "resample_n"}


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
            frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
    rows = [
        {str(name): _scalar(value) for name, value in row.items()}
        for _, row in frame.iterrows()
    ]
    return {"columns": [str(name) for name in frame.columns], "rows": rows}


def _rng_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _draw_summary(draws: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in draws.columns:
        if name in METRIC_EXCLUDE or not pd.api.types.is_numeric_dtype(draws[name]):
            continue
        values = pd.to_numeric(draws[name], errors="coerce")
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "metric": str(name),
                "n": int(len(finite)),
                "mean": _scalar(finite.mean()) if len(finite) else None,
                "sd": _scalar(finite.std(ddof=1)) if len(finite) > 1 else None,
                "min": _scalar(finite.min()) if len(finite) else None,
                "max": _scalar(finite.max()) if len(finite) else None,
            }
        )
    return rows


def _target_uncertainty(value: Any, global_rng_preserved: bool) -> dict[str, Any]:
    intervals = value.intervals.copy()
    if len(intervals):
        intervals = intervals.sort_values("metric", kind="stable").reset_index(drop=True)
    replicate_sizes = np.asarray(value.replicate_sizes, dtype=int)
    checks = {
        "global_rng_preserved": bool(global_rng_preserved),
        "successful_plus_failed_equals_bootstrap": bool(
            int(value.successful_replicates) + int(value.failed_replicates) == int(value.bootstrap)
        ),
        "replicate_sizes_recorded": bool(len(replicate_sizes) == int(value.bootstrap)),
        "replicate_sizes_positive": bool(np.all(replicate_sizes > 0)),
        "intervals_ordered": bool(
            all(
                row.lower <= row.upper
                for row in intervals.itertuples(index=False)
                if row.lower is not None
                and row.upper is not None
                and np.isfinite(row.lower)
                and np.isfinite(row.upper)
            )
        ),
    }
    return {
        "class": value.r_class,
        "unit": value.unit,
        "generalization_target": value.generalization_target,
        "bootstrap": int(value.bootstrap),
        "successful_replicates": int(value.successful_replicates),
        "failed_replicates": int(value.failed_replicates),
        "conf_level": float(value.conf_level),
        "seed": int(value.seed),
        "limitations": value.limitations,
        "point": _frame(value.point),
        "intervals": _frame(intervals),
        "draw_columns": [str(name) for name in value.draws.columns],
        "draw_nrow": int(len(value.draws)),
        "draw_replicates": sorted(int(x) for x in pd.unique(value.draws["replicate"])),
        "draw_resample_n": {
            "min": int(value.draws["resample_n"].min()),
            "max": int(value.draws["resample_n"].max()),
        },
        "draw_summary": _draw_summary(value.draws),
        "failure_columns": [str(name) for name in value.failures.columns],
        "failures": _frame(value.failures, sort_by=["replicate"]),
        "replicate_sizes": [int(x) for x in replicate_sizes.tolist()],
        "checks": checks,
    }


def _resample_uncertainty(value: Any) -> dict[str, Any]:
    return {
        "class": value.r_class,
        "unit": value.unit,
        "conf_level": float(value.conf_level),
        "generalization_target": value.generalization_target,
        "limitations": value.limitations,
        "summary": _frame(value.summary, sort_by=["metric"]),
        "distribution": _frame(
            value.distribution,
            sort_by=["metric", "repeat", "fold", "fold_id"],
        ),
    }


def _validation(value: Any) -> dict[str, Any]:
    checks = value.checks.loc[:, ["check_id", "status"]].copy()
    issues = value.issues.loc[:, ["check_id", "status"]].copy()
    return {
        "class": value.r_class,
        "status": value.status,
        "checks": _frame(checks),
        "issues": _frame(issues),
    }


def _capture(call: Callable[[], Any], normalize: Callable[[Any], Any] | None = None) -> dict[str, Any]:
    try:
        value = call()
        return {"status": "success", "value": value if normalize is None else normalize(value)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _classification_fixture(spec: dict[str, Any]) -> tuple[pd.DataFrame, Any]:
    n = len(spec["truth"])
    data = pd.DataFrame(
        {
            "trial_id": [f"T{i:02d}" for i in range(1, n + 1)],
            "participant_id": spec["participant_id"],
            "stimulus_id": spec["stimulus_id"],
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
        generalization_target="new_participants_and_new_stimuli",
        positive="review",
    )
    return data, task


def _regression_fixture(spec: dict[str, Any]) -> tuple[pd.DataFrame, Any]:
    n = len(spec["truth"])
    data = pd.DataFrame(
        {
            "trial_id": [f"R{i:02d}" for i in range(1, n + 1)],
            "observed_duration": np.asarray(spec["truth"], dtype=float),
        }
    )
    task = gp3mlpy.declare_gazepoint_task(
        data,
        outcome="observed_duration",
        purpose="Predict an observed response duration.",
        task_type="regression",
        unit_id="trial_id",
        generalization_target="new_trials_known_participants",
    )
    return data, task


def _run_bootstrap(task: Any, spec: dict[str, Any], unit: str, *, regression: bool = False) -> dict[str, Any]:
    np.random.seed(20260831)
    before = np.random.get_state()
    kwargs: dict[str, Any] = {
        "task": task,
        "truth": spec["truth"],
        "unit": unit,
        "bootstrap": spec["bootstrap"],
        "conf_level": spec["conf_level"],
        "seed": spec["seed"],
    }
    if regression:
        kwargs["prediction"] = spec["prediction"]
    else:
        kwargs.update(
            {
                "prediction": spec["prediction"],
                "probability": spec["probability"],
                "participant_id": spec["participant_id"],
                "stimulus_id": spec["stimulus_id"],
            }
        )
    value = gp3mlpy.bootstrap_gazepoint_metrics_by_unit(**kwargs)
    after = np.random.get_state()
    return _target_uncertainty(value, _rng_equal(before, after))


def _evaluation(metrics: list[dict[str, Any]]) -> GP3MLResampleEvaluation:
    return GP3MLResampleEvaluation(
        metrics=pd.DataFrame(metrics),
        generalization_target="new_participants",
    )


def _normalize_csv(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    return _frame(frame)


def _writer_evidence(value: Any, *, prefix: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gp3mlpy-parity-") as directory:
        paths = gp3mlpy.write_gazepoint_target_uncertainty(
            value, directory, prefix=prefix, overwrite=False
        )
        normalized: dict[str, Any] = {}
        for name in sorted(paths):
            path = Path(paths[name])
            normalized[name] = {
                "basename": path.name,
                "table": _normalize_csv(path),
            }
        return normalized


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python parity/run_python_target_uncertainty.py <fixture.json> <output.json>"
        )
    fixture_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected = fixture["r_reference"]["version"]
    if gp3mlpy.r_reference_version != expected:
        raise RuntimeError(
            f"gp3mlpy reference {gp3mlpy.r_reference_version!r} does not match fixture {expected!r}."
        )

    cls_spec = fixture["classification"]
    _, cls_task = _classification_fixture(cls_spec)
    reg_spec = fixture["regression"]
    _, reg_task = _regression_fixture(reg_spec)

    bootstrap_cases: dict[str, dict[str, Any]] = {}
    for unit in ("observation", "participant", "stimulus", "participant_and_stimulus"):
        case_id = f"classification_{unit}"
        bootstrap_cases[case_id] = _capture(
            lambda unit=unit: _run_bootstrap(cls_task, cls_spec, unit)
        )
    bootstrap_cases["regression_observation"] = _capture(
        lambda: _run_bootstrap(reg_task, reg_spec, "observation", regression=True)
    )
    bootstrap_cases["invalid_unit"] = _capture(
        lambda: gp3mlpy.bootstrap_gazepoint_metrics_by_unit(
            cls_task, cls_spec["truth"], probability=cls_spec["probability"], unit="rows"
        )
    )
    bootstrap_cases["bootstrap_zero"] = _capture(
        lambda: gp3mlpy.bootstrap_gazepoint_metrics_by_unit(
            cls_task,
            cls_spec["truth"],
            probability=cls_spec["probability"],
            bootstrap=0,
        )
    )
    bootstrap_cases["truth_too_short"] = _capture(
        lambda: gp3mlpy.bootstrap_gazepoint_metrics_by_unit(
            cls_task, ["pass"], probability=[0.2]
        )
    )
    bootstrap_cases["probability_length_mismatch"] = _capture(
        lambda: gp3mlpy.bootstrap_gazepoint_metrics_by_unit(
            cls_task, cls_spec["truth"], probability=cls_spec["probability"][:-1]
        )
    )
    bootstrap_cases["prediction_length_mismatch"] = _capture(
        lambda: gp3mlpy.bootstrap_gazepoint_metrics_by_unit(
            reg_task, reg_spec["truth"], prediction=reg_spec["prediction"][:-1]
        )
    )
    bootstrap_cases["participant_length_mismatch"] = _capture(
        lambda: gp3mlpy.bootstrap_gazepoint_metrics_by_unit(
            cls_task,
            cls_spec["truth"],
            probability=cls_spec["probability"],
            participant_id=cls_spec["participant_id"][:-1],
            unit="participant",
        )
    )
    bootstrap_cases["stimulus_length_mismatch"] = _capture(
        lambda: gp3mlpy.bootstrap_gazepoint_metrics_by_unit(
            cls_task,
            cls_spec["truth"],
            probability=cls_spec["probability"],
            stimulus_id=cls_spec["stimulus_id"][:-1],
            unit="stimulus",
        )
    )
    participant_missing = list(cls_spec["participant_id"])
    participant_missing[0] = None
    bootstrap_cases["participant_missing_identifier"] = _capture(
        lambda: gp3mlpy.bootstrap_gazepoint_metrics_by_unit(
            cls_task,
            cls_spec["truth"],
            probability=cls_spec["probability"],
            participant_id=participant_missing,
            unit="participant",
            bootstrap=2,
        )
    )

    evaluation = _evaluation(fixture["resample_metrics"])
    fold_uncertainty = gp3mlpy.summarize_gazepoint_resample_uncertainty(
        evaluation, unit="fold", conf_level=0.90
    )
    repeat_uncertainty = gp3mlpy.summarize_gazepoint_resample_uncertainty(
        evaluation, unit="repeat", conf_level=0.90
    )
    summarize_cases = {
        "fold_summary": {"status": "success", "value": _resample_uncertainty(fold_uncertainty)},
        "repeat_summary": {"status": "success", "value": _resample_uncertainty(repeat_uncertainty)},
        "invalid_unit": _capture(
            lambda: gp3mlpy.summarize_gazepoint_resample_uncertainty(evaluation, unit="participant")
        ),
        "invalid_object": _capture(
            lambda: gp3mlpy.summarize_gazepoint_resample_uncertainty(42)
        ),
        "empty_metrics": _capture(
            lambda: gp3mlpy.summarize_gazepoint_resample_uncertainty(
                GP3MLResampleEvaluation(
                    metrics=pd.DataFrame(),
                    generalization_target="new_participants",
                )
            )
        ),
        "repeat_missing_column": _capture(
            lambda: gp3mlpy.summarize_gazepoint_resample_uncertainty(
                GP3MLResampleEvaluation(
                    metrics=pd.DataFrame(fixture["resample_metrics"]).drop(columns=["repeat"]),
                    generalization_target="new_participants",
                ),
                unit="repeat",
            )
        ),
    }

    target_clean = gp3mlpy.bootstrap_gazepoint_metrics_by_unit(
        cls_task,
        cls_spec["truth"],
        prediction=cls_spec["prediction"],
        probability=cls_spec["probability"],
        participant_id=cls_spec["participant_id"],
        stimulus_id=cls_spec["stimulus_id"],
        unit="observation",
        bootstrap=10,
        seed=cls_spec["seed"],
    )
    damaged_target = copy.deepcopy(target_clean)
    damaged_target.limitations = ""
    damaged_resample = copy.deepcopy(fold_uncertainty)
    damaged_resample.generalization_target = ""
    validation_cases = {
        "target_clean": _capture(
            lambda: gp3mlpy.validate_gazepoint_target_uncertainty(target_clean), _validation
        ),
        "resample_clean": _capture(
            lambda: gp3mlpy.validate_gazepoint_target_uncertainty(fold_uncertainty), _validation
        ),
        "target_missing_limitations": _capture(
            lambda: gp3mlpy.validate_gazepoint_target_uncertainty(damaged_target), _validation
        ),
        "resample_missing_target": _capture(
            lambda: gp3mlpy.validate_gazepoint_target_uncertainty(damaged_resample), _validation
        ),
        "invalid_object": _capture(
            lambda: gp3mlpy.validate_gazepoint_target_uncertainty(42)
        ),
    }

    writer_cases = {
        "write_target": _capture(
            lambda: _writer_evidence(target_clean, prefix="uncertainty_target")
        ),
        "write_resample": _capture(
            lambda: _writer_evidence(fold_uncertainty, prefix="uncertainty_resample")
        ),
        "invalid_object": _capture(
            lambda: gp3mlpy.write_gazepoint_target_uncertainty(42, tempfile.mkdtemp())
        ),
    }
    with tempfile.TemporaryDirectory(prefix="gp3mlpy-parity-overwrite-") as directory:
        gp3mlpy.write_gazepoint_target_uncertainty(
            fold_uncertainty, directory, prefix="overwrite_case", overwrite=False
        )
        writer_cases["overwrite_refusal"] = _capture(
            lambda: gp3mlpy.write_gazepoint_target_uncertainty(
                fold_uncertainty, directory, prefix="overwrite_case", overwrite=False
            )
        )

    result = {
        "runtime": "Python",
        "package": "gp3mlpy",
        "package_version": gp3mlpy.__version__,
        "r_reference_version": gp3mlpy.r_reference_version,
        "bootstrap_cases": bootstrap_cases,
        "summarize_cases": summarize_cases,
        "validation_cases": validation_cases,
        "writer_cases": writer_cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
