from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._utils import assert_data, hash_jsonable, timestamp, worst_status
from .exceptions import GP3MLError
from .governance_reports import _markdown_table, evaluate_external_validation
from .objects import (
    GP3MLExternalDatasetDeclaration,
    GP3MLModel,
    GP3MLResampleEvaluation,
    GP3MLTransportabilityReport,
    GP3MLTransportabilityValidation,
)


def _safe_path(path: str | Path, overwrite: bool) -> Path:
    p = Path(path).expanduser()
    if p.exists() and not overwrite:
        raise GP3MLError(f"File exists: {p}.")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _data_hash(data: pd.DataFrame) -> str:
    # Python-native stable fingerprint. Cross-language RDS hash parity is tracked separately.
    return hash_jsonable(data, algorithm="md5")


def declare_gazepoint_external_dataset(
    data: pd.DataFrame,
    label: str,
    independent: bool,
    origin: str,
    collection_period: Any = None,
    participant_id: str | None = "participant_id",
    stimulus_id: str | None = "stimulus_id",
    notes: Any = (),
) -> GP3MLExternalDatasetDeclaration:
    """Declare a candidate external dataset and its independence status."""
    assert_data(data)
    if not isinstance(label, str) or not label.strip():
        raise GP3MLError("`label` is required.")
    if not isinstance(independent, (bool, np.bool_)):
        raise GP3MLError("`independent` must be explicitly TRUE or FALSE.")
    if not isinstance(origin, str) or not origin.strip():
        raise GP3MLError("`origin` is required.")
    for column in (participant_id, stimulus_id):
        if column is not None and str(column) and column not in data.columns:
            raise GP3MLError(f"Declared identifier `{column}` is not present.")
    if isinstance(notes, str):
        note_values = [notes]
    elif notes is None:
        note_values = []
    else:
        note_values = [str(x) for x in notes]
    return GP3MLExternalDatasetDeclaration(
        label=label.strip(),
        independent=bool(independent),
        origin=origin.strip(),
        collection_period=collection_period,
        participant_id=participant_id,
        stimulus_id=stimulus_id,
        n_rows=int(len(data)),
        n_participants=(int(data[participant_id].nunique(dropna=False)) if participant_id and participant_id in data else np.nan),
        n_stimuli=(int(data[stimulus_id].nunique(dropna=False)) if stimulus_id and stimulus_id in data else np.nan),
        data_hash=_data_hash(data),
        notes=note_values,
        declared_at=timestamp(),
    )


def _column_class(series: pd.Series) -> str:
    # Map pandas storage to a stable R-like semantic class for schema gates.
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return "factor"
    if pd.api.types.is_bool_dtype(dtype):
        return "logical"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "POSIXct/POSIXt"
    if pd.api.types.is_string_dtype(dtype) or dtype == object:
        return "character"
    return str(dtype)


def _schema_comparison(
    development_data: pd.DataFrame,
    external_data: pd.DataFrame,
    predictors: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    all_columns = list(dict.fromkeys([*development_data.columns.tolist(), *external_data.columns.tolist()]))
    rows: list[dict[str, Any]] = []
    predictor_set = set(predictors)
    for name in all_columns:
        development_present = name in development_data.columns
        external_present = name in external_data.columns
        development_class = _column_class(development_data[name]) if development_present else None
        external_class = _column_class(external_data[name]) if external_present else None
        rows.append(
            {
                "variable": name,
                "predictor": name in predictor_set,
                "development_present": development_present,
                "external_present": external_present,
                "development_class": development_class,
                "external_class": external_class,
                "class_match": bool(development_present and external_present and development_class == external_class),
                "development_missing_prop": float(development_data[name].isna().mean()) if development_present else np.nan,
                "external_missing_prop": float(external_data[name].isna().mean()) if external_present else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _group_transportability(
    development_data: pd.DataFrame,
    external_data: pd.DataFrame,
    column: str | None,
    unit: str,
) -> pd.DataFrame:
    if column is None or not str(column) or column not in development_data.columns or column not in external_data.columns:
        return pd.DataFrame(
            [{
                "unit": unit,
                "identifier": column,
                "development_groups": np.nan,
                "external_groups": np.nan,
                "overlapping_groups": np.nan,
                "external_novel_groups": np.nan,
                "external_coverage_prop": np.nan,
                "status": "not_available",
            }]
        )
    development_groups = list(dict.fromkeys(development_data[column].astype(str).tolist()))
    external_groups = list(dict.fromkeys(external_data[column].astype(str).tolist()))
    development_set = set(development_groups)
    overlap = [x for x in external_groups if x in development_set]
    novel = [x for x in external_groups if x not in development_set]
    return pd.DataFrame(
        [{
            "unit": unit,
            "identifier": column,
            "development_groups": len(development_groups),
            "external_groups": len(external_groups),
            "overlapping_groups": len(overlap),
            "external_novel_groups": len(novel),
            "external_coverage_prop": (len(novel) / len(external_groups)) if external_groups else np.nan,
            "status": "review" if overlap else "pass",
        }]
    )


def _prevalence_shift(task: Any, development_data: pd.DataFrame, external_data: pd.DataFrame) -> pd.DataFrame:
    if task.task_type != "classification":
        return pd.DataFrame()
    development_rate = float((development_data[task.outcome].astype(str) == str(task.positive)).mean())
    external_rate = float((external_data[task.outcome].astype(str) == str(task.positive)).mean())
    return pd.DataFrame(
        [{
            "positive": task.positive,
            "development_prevalence": development_rate,
            "external_prevalence": external_rate,
            "absolute_shift": external_rate - development_rate,
            "relative_shift": np.nan if development_rate == 0 else external_rate / development_rate,
        }]
    )


def _development_metric_summary(development_evaluation: Any) -> pd.DataFrame:
    if development_evaluation is None:
        return pd.DataFrame(columns=["metric", "development_estimate"])
    if getattr(development_evaluation, "r_class", None) in {"gp3ml_resample_evaluation", "gp3ml_nested_evaluation"}:
        metrics = development_evaluation.metrics
        if metrics is None or len(metrics) == 0:
            return pd.DataFrame(columns=["metric", "development_estimate"])
        return (
            metrics.groupby("metric", sort=False, dropna=False)["value"]
            .mean()
            .rename("development_estimate")
            .reset_index()
        )
    if isinstance(development_evaluation, pd.DataFrame):
        numeric_names = [c for c in development_evaluation.columns if pd.api.types.is_numeric_dtype(development_evaluation[c])]
        return pd.DataFrame(
            {
                "metric": numeric_names,
                "development_estimate": [float(development_evaluation[c].mean(skipna=True)) for c in numeric_names],
            }
        )
    raise GP3MLError("`development_evaluation` must be a grouped evaluation or metric data frame.")


def _metric_long(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or len(metrics) == 0:
        return pd.DataFrame(columns=["metric", "value"])
    excluded = {"n", "threshold"}
    rows = []
    # R's metric-long helper treats one-row metric frames as named scalar metrics.
    if len(metrics) == 1:
        for name in metrics.columns:
            if name not in excluded and pd.api.types.is_numeric_dtype(metrics[name]):
                rows.append({"metric": name, "value": float(metrics.iloc[0][name])})
    elif {"metric", "value"}.issubset(metrics.columns):
        return metrics.loc[:, ["metric", "value"]].copy()
    return pd.DataFrame(rows, columns=["metric", "value"])


def evaluate_gazepoint_external_transportability(
    model: GP3MLModel,
    development_data: pd.DataFrame,
    external_data: pd.DataFrame | None = None,
    declaration: GP3MLExternalDatasetDeclaration | None = None,
    development_evaluation: Any = None,
    threshold: float | None = None,
    bootstrap: int = 200,
    seed: int = 1,
) -> GP3MLTransportabilityReport:
    """Evaluate whether an explicitly declared external dataset supports external validation."""
    if not isinstance(model, GP3MLModel):
        raise GP3MLError("`model` must be a fitted gp3ml model.")
    assert_data(development_data, name="development_data")
    if threshold is None:
        threshold = model.threshold
    if external_data is None:
        report = GP3MLTransportabilityReport(
            status="not_externally_validated",
            reason="No independent external dataset was supplied.",
            declaration=declaration,
            declaration_hash_matches=np.nan,
            metrics=pd.DataFrame(),
            performance_comparison=pd.DataFrame(),
            calibration_drift=pd.DataFrame(),
            prevalence_shift=pd.DataFrame(),
            schema=pd.DataFrame(),
            group_coverage=pd.DataFrame(),
            predictor_shift=pd.DataFrame(),
            validation=None,
            task=model.task,
            model_engine=model.engine,
            limitations=["Internal holdout or grouped resampling does not establish external validation."],
        )
        report.validation_summary = validate_gazepoint_transportability(report)
        return report

    assert_data(external_data, name="external_data")
    if not isinstance(declaration, GP3MLExternalDatasetDeclaration):
        raise GP3MLError("Supply a `gp3ml_external_dataset_declaration` for external data.")

    schema = _schema_comparison(development_data, external_data, list(model.predictors))
    declaration_hash_matches = declaration.data_hash == _data_hash(external_data)
    outcome_row = schema.loc[schema["variable"] == model.task.outcome]
    outcome_available = len(outcome_row) == 1 and bool(outcome_row.iloc[0]["external_present"])
    outcome_type_compatible = outcome_available and bool(outcome_row.iloc[0]["class_match"])
    missing_predictors = schema.loc[schema["predictor"] & ~schema["external_present"], "variable"].tolist()
    type_mismatches = schema.loc[schema["predictor"] & schema["external_present"] & ~schema["class_match"], "variable"].tolist()
    group_coverage = pd.concat(
        [
            _group_transportability(development_data, external_data, model.task.participant_id, "participant"),
            _group_transportability(development_data, external_data, model.task.stimulus_id, "stimulus"),
        ],
        ignore_index=True,
    )
    independence_issues = pd.to_numeric(group_coverage["overlapping_groups"], errors="coerce").dropna()

    if not declaration_hash_matches:
        status = "external_declaration_mismatch"
        reason = "The external data no longer match the dataset fingerprint recorded in the declaration."
    elif not declaration.independent:
        status = "not_externally_validated"
        reason = "The supplied dataset was explicitly declared non-independent."
    elif (not outcome_available) or (not outcome_type_compatible) or missing_predictors or type_mismatches:
        pieces = []
        if not outcome_available:
            pieces.append(f"Missing outcome: {model.task.outcome}")
        if outcome_available and not outcome_type_compatible:
            pieces.append(f"Outcome type mismatch: {model.task.outcome}")
        if missing_predictors:
            pieces.append("Missing predictors: " + ", ".join(missing_predictors))
        if type_mismatches:
            pieces.append("Predictor type mismatches: " + ", ".join(type_mismatches))
        status = "incompatible_external_schema"
        reason = "; ".join(pieces)
    elif len(independence_issues) and bool((independence_issues > 0).any()):
        status = "external_independence_requires_review"
        reason = "Declared identifier values overlap between development and external data."
    else:
        status = "externally_validated"
        reason = "The explicitly independent dataset passed schema and identifier-overlap gates."

    validation = None
    metrics = pd.DataFrame()
    predictor_shift = pd.DataFrame()
    calibration_drift = pd.DataFrame()
    performance_comparison = pd.DataFrame()
    prevalence_shift = _prevalence_shift(model.task, development_data, external_data) if outcome_available and outcome_type_compatible else pd.DataFrame()

    if status in {"externally_validated", "external_independence_requires_review"}:
        validation = evaluate_external_validation(
            model,
            external_data,
            label=declaration.label,
            threshold=threshold,
            bootstrap=int(bootstrap),
            seed=int(seed),
        )
        metrics = validation.metrics
        predictor_shift = validation.shift
        development_metrics = _development_metric_summary(development_evaluation)
        external_long = _metric_long(metrics).rename(columns={"value": "external_estimate"})
        if len(external_long):
            performance_comparison = development_metrics.merge(
                external_long.loc[:, ["metric", "external_estimate"]],
                how="outer",
                on="metric",
                sort=False,
            )
            performance_comparison["difference"] = (
                performance_comparison["external_estimate"] - performance_comparison["development_estimate"]
            )
        if model.task.task_type == "classification" and validation.calibration is not None:
            external_calibration = validation.calibration.summary.copy()
            external_calibration.columns = [f"external_{c}" for c in external_calibration.columns]
            calibration_drift = external_calibration.reset_index(drop=True)
            if isinstance(development_evaluation, GP3MLResampleEvaluation):
                summaries = []
                for fold_result in development_evaluation.fold_results:
                    cal = getattr(fold_result, "calibration", None) if not isinstance(fold_result, dict) else fold_result.get("calibration")
                    if cal is not None:
                        summaries.append(cal.summary)
                if summaries:
                    development_calibration = pd.concat(summaries, ignore_index=True)
                    for name in development_calibration.columns:
                        if pd.api.types.is_numeric_dtype(development_calibration[name]):
                            dev_value = float(development_calibration[name].mean(skipna=True))
                            calibration_drift[f"development_{name}"] = dev_value
                            ext_name = f"external_{name}"
                            if ext_name in calibration_drift.columns:
                                calibration_drift[f"drift_{name}"] = calibration_drift[ext_name] - dev_value

    report = GP3MLTransportabilityReport(
        status=status,
        reason=reason,
        declaration=declaration,
        declaration_hash_matches=bool(declaration_hash_matches),
        metrics=metrics,
        performance_comparison=performance_comparison,
        calibration_drift=calibration_drift,
        prevalence_shift=prevalence_shift,
        schema=schema,
        group_coverage=group_coverage,
        predictor_shift=predictor_shift,
        validation=validation,
        task=model.task,
        model_engine=model.engine,
        development_hash=_data_hash(development_data.loc[:, [model.task.outcome, *model.predictors]]),
        external_hash=declaration.data_hash,
        limitations=[
            "External validation is specific to the declared dataset, outcome, predictors, and collection context.",
            "Transportability beyond the observed external context requires additional evidence.",
        ],
        call="evaluate_gazepoint_external_transportability",
    )
    report.validation_summary = validate_gazepoint_transportability(report)
    return report


def validate_gazepoint_transportability(x: GP3MLTransportabilityReport) -> GP3MLTransportabilityValidation:
    """Validate the gates and explicit status in a transportability report."""
    if not isinstance(x, GP3MLTransportabilityReport):
        raise GP3MLError("`x` must be a `gp3ml_transportability_report` object.")
    schema = x.schema
    group_coverage = x.group_coverage
    declaration = x.declaration
    has_schema = isinstance(schema, pd.DataFrame) and len(schema) > 0
    has_groups = isinstance(group_coverage, pd.DataFrame) and len(group_coverage) > 0
    statuses = [
        "review" if declaration is None else ("pass" if declaration.independent else "fail"),
        "review" if declaration is None else ("pass" if x.declaration_hash_matches is True else "fail"),
        "review" if not has_schema else ("pass" if bool(((schema["variable"] == x.task.outcome) & schema["external_present"]).any()) else "fail"),
        "review" if not has_schema else ("pass" if bool(((schema["variable"] == x.task.outcome) & schema["class_match"]).any()) else "fail"),
        "review" if not has_schema else ("fail" if bool((schema["predictor"] & ~schema["external_present"]).any()) else "pass"),
        "review" if not has_schema else ("fail" if bool((schema["predictor"] & schema["external_present"] & ~schema["class_match"]).any()) else "pass"),
        "review" if not has_groups else ("review" if bool((group_coverage["status"] == "review").any()) else "pass"),
        "pass" if isinstance(x.status, str) and bool(x.status) else "fail",
    ]
    messages = [
        "No external independence declaration is attached." if declaration is None else f"Independent: {declaration.independent}",
        "The supplied external data must match the declaration fingerprint.",
        "The observed outcome must be present in the external schema.",
        "The external outcome class must match the development schema.",
        "All model predictors must be available in the external schema.",
        "External predictor classes must match the development schema.",
        "Participant and stimulus identifier overlap is reviewed explicitly.",
        f"Status: {x.status} - {x.reason}",
    ]
    checks = pd.DataFrame(
        {
            "check_id": [
                "independence_declared",
                "declaration_matches_data",
                "outcome_available",
                "outcome_type_compatible",
                "predictors_available",
                "predictor_types_compatible",
                "identifier_overlap",
                "external_validation_status_explicit",
            ],
            "status": statuses,
            "message": messages,
        }
    )
    if x.status == "externally_validated":
        status = worst_status(checks["status"].tolist())
    elif x.status == "not_externally_validated":
        status = "review"
    else:
        status = worst_status(checks["status"].tolist())
    return GP3MLTransportabilityValidation(
        status=status,
        checks=checks,
        issues=checks.loc[checks["status"] != "pass"].reset_index(drop=True),
    )


def write_gazepoint_transportability_report(
    report: GP3MLTransportabilityReport,
    path: str | Path,
    overwrite: bool = False,
) -> str:
    """Write an expanded external-validation and transportability report."""
    if not isinstance(report, GP3MLTransportabilityReport):
        raise GP3MLError("`report` must be a `gp3ml_transportability_report` object.")
    p = _safe_path(path, overwrite)
    if report.declaration is None:
        declaration_lines = ["No independent external dataset was declared."]
    else:
        declaration_lines = [
            f"- Label: {report.declaration.label}",
            f"- Independent: {report.declaration.independent}",
            f"- Origin: {report.declaration.origin}",
            f"- Rows: {report.declaration.n_rows}",
        ]
    lines = [
        "# Gazepoint external validation and transportability", "",
        f"Status: **{report.status}**", "", report.reason, "",
        "## Declaration", "", *declaration_lines, "",
        "## Development versus external performance", "", *_markdown_table(report.performance_comparison), "",
        "## Calibration drift", "", *_markdown_table(report.calibration_drift), "",
        "## Class-prevalence shift", "", *_markdown_table(report.prevalence_shift), "",
        "## Predictor availability and schema", "", *_markdown_table(report.schema), "",
        "## Participant and stimulus coverage", "", *_markdown_table(report.group_coverage), "",
        "## Predictor shift", "", *_markdown_table(report.predictor_shift), "",
        "## Limitations", "", *[f"- {x}" for x in report.limitations], "",
        "## Interpretation boundary", "",
        "An internal holdout is not external validation. This report applies only to explicitly observed, non-sensitive outcomes and the declared external context.",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)
