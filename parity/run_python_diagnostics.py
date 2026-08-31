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
        value = call()
        return {"status": "success", "value": normalize(value)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _data(outcome_type: str = "categorical") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source = 0
    for participant in range(1, 7):
        for stimulus in range(1, 5):
            for repetition in range(1, 4):
                source += 1
                outcome: Any
                if outcome_type == "numeric":
                    outcome = 180.0 + source / 3.0
                else:
                    outcome = "yes" if (participant + stimulus + repetition) % 2 == 0 else "no"
                rows.append(
                    {
                        "participant_id": f"P{participant:02d}",
                        "stimulus_id": f"S{stimulus:02d}",
                        "trial_id": f"S{stimulus:02d}_T{repetition}",
                        "outcome": outcome,
                        "fixation_duration": float(180 + source),
                        "pupil_change": float(source) / 1000.0,
                    }
                )
    data = pd.DataFrame(rows)
    if outcome_type == "categorical":
        data["outcome"] = pd.Categorical(data["outcome"], categories=["no", "yes"])
    return data


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
    repeats: int = 1,
    seed: int = 101,
    outcome_type: str = "categorical",
) -> Any:
    return gp3mlpy.create_gazepoint_group_folds(
        data=_data(outcome_type),
        outcome="outcome",
        predictors=PREDICTORS,
        feature_manifest=_manifest(),
        generalization_target=target,
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id="stimulus_id",
        v=v,
        repeats=repeats,
        seed=seed,
    )


def _validation(value: Any) -> dict[str, Any]:
    return {
        "class": value.r_class,
        "status": value.status,
        "summary": _frame(value.summary),
        "checks": _frame(value.checks),
        "issues": _frame(value.issues),
    }


def _success(value: Any) -> dict[str, Any]:
    outcome = value.outcome_balance
    numeric = outcome.loc[outcome.metric_type == "numeric"]
    nonempty_numeric = numeric.loc[numeric.n > 0]
    categorical = outcome.loc[outcome.metric_type == "categorical"]
    categorical_levels = sorted(
        {
            str(level)
            for level in categorical.outcome_level.dropna().tolist()
        }
    )
    return {
        "class": value.r_class,
        "metadata": {
            "outcome": value.metadata["outcome"],
            "generalization_target": value.metadata["generalization_target"],
            "repeats": int(value.metadata["repeats"]),
            "n_source_rows": int(value.metadata["n_source_rows"]),
            "n_folds_total": int(value.metadata["n_folds_total"]),
            "imbalance_review": float(value.metadata["imbalance_review"]),
            "imbalance_fail": float(value.metadata["imbalance_fail"]),
            "outcome_type": value.metadata["outcome_type"],
            "source_validation_status": value.metadata["source_validation_status"],
            "source_audit_status": value.metadata["source_audit_status"],
        },
        "row_counts": {
            "fold_metrics": len(value.fold_metrics),
            "repeat_metrics": len(value.repeat_metrics),
            "outcome_balance": len(value.outcome_balance),
            "group_balance": len(value.group_balance),
            "assessment_coverage": len(value.assessment_coverage),
            "exclusion_summary": len(value.exclusion_summary),
        },
        "columns": {
            "fold_metrics": list(value.fold_metrics.columns),
            "repeat_metrics": list(value.repeat_metrics.columns),
            "outcome_balance": list(value.outcome_balance.columns),
            "group_balance": list(value.group_balance.columns),
            "assessment_coverage": list(value.assessment_coverage.columns),
            "exclusion_summary": list(value.exclusion_summary.columns),
        },
        "coverage_once_per_repeat": bool(
            len(value.assessment_coverage) > 0
            and (value.assessment_coverage.n_assessment == 1).all()
        ),
        "partition_accounting": bool(
            (
                value.fold_metrics.n_total
                == value.fold_metrics.n_analysis
                + value.fold_metrics.n_assessment
                + value.fold_metrics.n_excluded
            ).all()
        ),
        "nonempty_analysis_assessment": bool(
            (
                (value.fold_metrics.n_analysis > 0)
                & (value.fold_metrics.n_assessment > 0)
            ).all()
        ),
        "assessment_groups_present": bool(
            len(value.group_balance) > 0
            and (value.group_balance.n_assessment_groups > 0).all()
        ),
        "assessment_size_ratios": sorted(
            round(float(v), 12)
            for v in pd.to_numeric(
                value.repeat_metrics.assessment_size_ratio, errors="coerce"
            ).dropna()
        ),
        "total_excluded": int(value.fold_metrics.n_excluded.sum()),
        "categorical_levels": categorical_levels,
        "categorical_missing_assessment_level": bool(
            len(categorical) > 0
            and (
                categorical.loc[categorical.partition == "assessment", "n"] == 0
            ).any()
        ),
        "numeric_nonempty_means_finite": bool(
            len(nonempty_numeric) == 0
            or np.isfinite(pd.to_numeric(nonempty_numeric["mean"])).all()
        ),
        "validation": _validation(value.validation),
    }


def _success_case(case: dict[str, Any]) -> Any:
    value = gp3mlpy.diagnose_gazepoint_group_folds(
        _plan(
            case["target"],
            v=case["v"],
            repeats=int(case["repeats"]),
            outcome_type=case["outcome_type"],
        )
    )
    if case.get("sparse_first_assessment"):
        plan = _plan(
            case["target"],
            v=case["v"],
            repeats=int(case["repeats"]),
            outcome_type=case["outcome_type"],
        )
        first = next(iter(plan.folds.values()))
        first.assessment["outcome"] = pd.Categorical(
            ["no"] * len(first.assessment), categories=["no", "yes"]
        )
        value = gp3mlpy.diagnose_gazepoint_group_folds(plan)
    return value


def _error_case(kind: str) -> Any:
    if kind == "invalid_object":
        return gp3mlpy.diagnose_gazepoint_group_folds({})  # type: ignore[arg-type]
    plan = _plan()
    if kind == "bad_review_threshold":
        return gp3mlpy.diagnose_gazepoint_group_folds(plan, imbalance_review=0.9)
    if kind == "threshold_order":
        return gp3mlpy.diagnose_gazepoint_group_folds(
            plan, imbalance_review=2.0, imbalance_fail=1.5
        )
    if kind == "missing_component":
        del plan["audit"]
        return gp3mlpy.diagnose_gazepoint_group_folds(plan)
    if kind == "missing_summary_column":
        plan.fold_summary = plan.fold_summary.drop(columns=["n_total"])
        return gp3mlpy.diagnose_gazepoint_group_folds(plan)
    if kind == "missing_outcome_column":
        first = next(iter(plan.folds.values()))
        first.analysis = first.analysis.drop(columns=["outcome"])
        return gp3mlpy.diagnose_gazepoint_group_folds(plan)
    raise RuntimeError(f"Unknown diagnostics error case: {kind}")


def _validation_case(kind: str) -> Any:
    if kind == "invalid_object":
        return gp3mlpy.validate_gazepoint_fold_diagnostics({})  # type: ignore[arg-type]
    diagnostics = gp3mlpy.diagnose_gazepoint_group_folds(_plan())
    if kind == "clean":
        return gp3mlpy.validate_gazepoint_fold_diagnostics(diagnostics)
    if kind == "damaged_coverage":
        diagnostics.assessment_coverage.loc[0, "n_assessment"] = 0
    elif kind == "review_imbalance":
        diagnostics.repeat_metrics.loc[0, "assessment_size_ratio"] = 1.6
    elif kind == "fail_imbalance":
        diagnostics.repeat_metrics.loc[0, "assessment_size_ratio"] = 2.5
    elif kind == "missing_level":
        mask = (
            (diagnostics.outcome_balance.metric_type == "categorical")
            & (diagnostics.outcome_balance.partition == "assessment")
            & (diagnostics.outcome_balance.outcome_level.astype(str) == "yes")
        )
        diagnostics.outcome_balance.loc[diagnostics.outcome_balance.index[mask][0], "n"] = 0
    elif kind == "missing_component":
        del diagnostics["group_balance"]
    else:
        raise RuntimeError(f"Unknown diagnostics validation case: {kind}")
    return gp3mlpy.validate_gazepoint_fold_diagnostics(diagnostics)


def _writer_case(kind: str) -> Any:
    if kind == "invalid_object":
        with tempfile.TemporaryDirectory() as temp_dir:
            return gp3mlpy.write_gazepoint_fold_diagnostics_csv(  # type: ignore[arg-type]
                {}, temp_dir
            )
    diagnostics = gp3mlpy.diagnose_gazepoint_group_folds(_plan())
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        if kind == "unknown_table":
            return gp3mlpy.write_gazepoint_fold_diagnostics_csv(
                diagnostics, root, tables="unknown"
            )
        if kind == "bad_overwrite":
            return gp3mlpy.write_gazepoint_fold_diagnostics_csv(
                diagnostics, root, overwrite="yes"  # type: ignore[arg-type]
            )
        if kind == "overwrite":
            gp3mlpy.write_gazepoint_fold_diagnostics_csv(
                diagnostics, root, prefix="parity", tables="fold_metrics"
            )
            return gp3mlpy.write_gazepoint_fold_diagnostics_csv(
                diagnostics, root, prefix="parity", tables="fold_metrics"
            )
        if kind == "selected":
            tables = ["fold_metrics", "repeat_metrics", "validation_checks"]
            paths = gp3mlpy.write_gazepoint_fold_diagnostics_csv(
                diagnostics, root, prefix="parity", tables=tables
            )
            result: dict[str, Any] = {}
            for name in tables:
                path = Path(paths[name])
                exported = pd.read_csv(path, keep_default_na=False)
                result[name] = {
                    "basename": path.name,
                    "columns": list(exported.columns),
                    "n_rows": len(exported),
                }
            return result
    raise RuntimeError(f"Unknown diagnostics writer case: {kind}")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python parity/run_python_diagnostics.py <fixture.json> <output.json>"
        )
    fixture = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output_path = Path(sys.argv[2])
    successes = {
        case["id"]: _capture(
            lambda case=case: _success_case(case), _success
        )
        for case in fixture["success_cases"]
    }
    errors = {
        case["id"]: _capture(
            lambda case=case: _error_case(case["kind"]), lambda _: True
        )
        for case in fixture["error_cases"]
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
