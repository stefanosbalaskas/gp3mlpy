from __future__ import annotations

from typing import Any

import pandas as pd


def _nrow(value: Any) -> int:
    if isinstance(value, pd.DataFrame):
        return int(len(value.index))
    try:
        return int(len(value))
    except (TypeError, AttributeError):
        return 0


def _table_text(value: Any, *, columns: list[str] | None = None) -> str:
    if not isinstance(value, pd.DataFrame) or value.empty:
        return ""
    table = value
    if columns is not None:
        present = [column for column in columns if column in table.columns]
        if present:
            table = table.loc[:, present]
    return table.to_string(index=False)


def _status_counts(table: Any) -> tuple[int, int, int]:
    if not isinstance(table, pd.DataFrame) or "status" not in table.columns:
        return (0, 0, 0)
    values = table["status"].astype(str)
    return tuple(int((values == status).sum()) for status in ("pass", "review", "fail"))


def _component(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        return getattr(value, name)
    except (AttributeError, KeyError):
        if isinstance(value, dict):
            return value.get(name, default)
        return default


def render_r_print(obj: Any) -> str:
    """Render the information surfaced by gp3ml 0.3.0 S3 print methods."""
    data = object.__getattribute__(obj, "_data")
    r_class = obj.r_class
    get = data.get
    upper = lambda value: str(value).upper()
    lines: list[str]
    table = ""

    if r_class == "gazepoint_feature_manifest_validation":
        lines = [
            f"<{r_class}>",
            f"Overall status: {upper(get('status'))}",
            f"Features: {get('n_features')}",
            f"Non-passing checks: {_nrow(get('issues'))}",
        ]
        table = _table_text(get("summary"))
    elif r_class == "gazepoint_fold_diagnostics":
        meta = get("metadata", {})
        validation = get("validation")
        lines = [
            f"<{r_class}>",
            f"Target: {meta.get('generalization_target')}",
            f"Repeats: {meta.get('repeats')}",
            f"Folds: {_nrow(get('fold_metrics'))}",
            f"Outcome type: {meta.get('outcome_type')}",
            f"Diagnostic status: {upper(_component(validation, 'status'))}",
        ]
        repeat = get("repeat_metrics")
        if isinstance(repeat, pd.DataFrame) and "assessment_size_ratio" in repeat and not repeat.empty:
            maximum = pd.to_numeric(repeat["assessment_size_ratio"], errors="coerce").max()
            if pd.notna(maximum):
                lines.append(f"Maximum assessment-size ratio: {float(maximum):.3f}")
    elif r_class == "gazepoint_fold_diagnostics_validation":
        lines = [f"<{r_class}>", f"Overall status: {upper(get('status'))}"]
        summary = get("summary")
        for label in ("pass", "review", "fail"):
            n_checks: Any = 0
            if isinstance(summary, pd.DataFrame) and {"status", "n_checks"} <= set(summary.columns):
                values = summary.loc[summary["status"].astype(str) == label, "n_checks"]
                n_checks = values.iloc[0] if len(values) else 0
            lines.append(f"{label.title()}: {n_checks}")
    elif r_class == "gazepoint_group_folds":
        meta = get("metadata", {})
        validation = get("validation")
        lines = [
            f"<{r_class}>",
            f"Target: {meta.get('generalization_target')}",
            f"Repeats: {meta.get('repeats')}",
            f"Folds per repeat: {meta.get('n_folds_per_repeat')}",
            f"Total folds: {meta.get('n_folds_total')}",
            f"Status: {upper(_component(validation, 'status'))}",
        ]
    elif r_class == "gazepoint_group_folds_audit":
        lines = [
            f"<{r_class}>",
            f"Overall status: {upper(get('status'))}",
            f"Audited folds: {_nrow(get('summary'))}",
            f"Non-passing checks: {_nrow(get('issues'))}",
        ]
    elif r_class in {"gazepoint_group_folds_validation", "gazepoint_ml_split_validation"}:
        lines = [
            f"<{r_class}>",
            f"Overall status: {upper(get('status'))}",
            f"Non-passing checks: {_nrow(get('issues'))}",
        ]
        table = _table_text(get("summary"))
    elif r_class == "gazepoint_ml_leakage_audit":
        summary = get("partition_summary")
        analysis_rows: Any = 0
        assessment_rows: Any = 0
        if isinstance(summary, pd.DataFrame) and {"partition", "n_rows"} <= set(summary.columns):
            a = summary.loc[summary["partition"] == "analysis", "n_rows"]
            b = summary.loc[summary["partition"] == "assessment", "n_rows"]
            analysis_rows = a.iloc[0] if len(a) else 0
            assessment_rows = b.iloc[0] if len(b) else 0
        issues = get("issues")
        lines = [
            f"<{r_class}>",
            f"Overall status: {upper(get('status'))}",
            f"Generalization target: {get('generalization_target')}",
            f"Rows: {analysis_rows} analysis; {assessment_rows} assessment",
            f"Non-passing checks: {_nrow(issues)}",
        ]
        if _nrow(issues) == 0:
            lines.append("No leakage issues were detected by the implemented checks.")
        else:
            table = _table_text(issues, columns=["check_id", "status", "n_affected", "columns"])
    elif r_class == "gazepoint_ml_split":
        meta = get("metadata", {})
        validation = get("validation")
        lines = [
            f"<{r_class}>",
            f"Target: {meta.get('generalization_target')}",
            f"Status: {upper(_component(validation, 'status'))}",
            "Rows: "
            f"analysis={_nrow(get('analysis'))}, "
            f"assessment={_nrow(get('assessment'))}, "
            f"excluded={_nrow(get('excluded'))}",
            f"Seed: {meta.get('seed')}",
        ]
    elif r_class == "gp3ml_api_contract_registry":
        exports = get("exports")
        classes = get("classes")
        stable = experimental = 0
        if isinstance(exports, pd.DataFrame) and "stability" in exports.columns:
            stable = int((exports["stability"] == "stable").sum())
            experimental = int((exports["stability"] == "experimental").sum())
        lines = [
            "gp3ml API contract registry: "
            f"{stable} stable exports, {experimental} experimental exports, "
            f"{_nrow(classes)} stable public classes"
        ]
    elif r_class == "gp3ml_api_stability_audit":
        lines = [f"gp3ml API stability audit: {get('status')}"]
        table = _table_text(get("checks"))
    elif r_class == "gp3ml_calibration_assessment":
        lines = [f"<{r_class}>"]
        table = _table_text(get("summary"))
    elif r_class == "gp3ml_decision_rule":
        threshold = get("threshold")
        lines = [
            "gp3ml decision rule",
            f" metric: {get('metric')} ({get('direction')})",
            f" threshold: {'<not selected>' if threshold is None else threshold}",
            f" origin: {get('threshold_origin')}",
            f" target: {get('generalization_target')}",
            f" abstention: {'enabled' if get('abstention_allowed') is True else 'disabled'}",
        ]
    elif r_class == "gp3ml_engine_capabilities":
        lines = [f"<{r_class}>"]
        table = _table_text(get("table") if "table" in data else pd.DataFrame(data))
    elif r_class == "gp3ml_external_dataset_declaration":
        lines = [
            f"<{r_class}>",
            f"  Label: {get('label')}",
            f"  Independent: {get('independent')}",
            f"  Origin: {get('origin')}",
            f"  Rows: {get('n_rows')}",
        ]
    elif r_class == "gp3ml_external_validation":
        lines = [f"<{r_class}> {get('label')}"]
        table = _table_text(get("metrics"))
    elif r_class == "gp3ml_handoff":
        lines = [
            f"gp3ml handoff from {get('source_package')}: "
            f"{_nrow(get('data'))} rows, {len(get('predictors') or [])} predictors"
        ]
    elif r_class == "gp3ml_handoff_bundle":
        lines = [
            f"gp3ml handoff bundle: {len(get('handoffs') or [])} sources, "
            f"{_nrow(get('data'))} joined rows"
        ]
    elif r_class == "gp3ml_handoff_validation":
        lines = [f"gp3ml handoff validation: {get('status')}"]
        table = _table_text(get("checks"))
    elif r_class == "gp3ml_metric_uncertainty":
        lines = [f"<{r_class}> bootstrap={get('bootstrap')} confidence={get('conf_level')}"]
        table = _table_text(get("intervals"))
    elif r_class == "gp3ml_model":
        task = get("task")
        lines = [
            f"<{r_class}> engine={get('engine')} task={_component(task, 'task_type')} "
            f"n={get('training_n')} predictors={len(get('predictors') or [])}"
        ]
    elif r_class == "gp3ml_model_card":
        lines = [f"<{r_class}> {get('title')}"]
    elif r_class == "gp3ml_model_selection":
        lines = [
            f"<{r_class}>",
            f"  Candidate: {get('candidate_id')}",
            f"  Primary metric: {get('primary_metric')} ({get('direction')})",
            f"  Value: {get('primary_value')}",
            f"  Human rationale: {get('rationale')}",
            "  Refit performed: no",
        ]
    elif r_class == "gp3ml_model_tuning":
        grid = get("grid")
        candidates = _component(grid, "candidates")
        results = get("results") or []
        failures = sum(1 for item in results if _component(item, "status") == "fail")
        lines = [
            f"<{r_class}>",
            f"  Candidates: {_nrow(candidates)}",
            f"  Failed candidates: {failures}",
            "  Automatic winner: none",
        ]
    elif r_class in {
        "gp3ml_model_tuning_validation",
        "gp3ml_nested_evaluation_validation",
        "gp3ml_nested_folds_validation",
        "gp3ml_nested_resampling_audit",
        "gp3ml_resample_evaluation_validation",
        "gp3ml_transportability_validation",
        "gp3ml_uncertainty_validation",
    }:
        lines = [f"<{r_class}> status={get('status')}"]
        table = _table_text(get("checks"))
    elif r_class == "gp3ml_nested_evaluation":
        fold_status = get("fold_status")
        _, _, failed = _status_counts(fold_status)
        lines = [
            f"<{r_class}>",
            f"  Target: {get('generalization_target')}",
            f"  Outer folds: {_nrow(fold_status)}",
            f"  Failed outer folds: {failed}",
            f"  Outer assessment predictions: {_nrow(get('predictions'))}",
        ]
    elif r_class == "gp3ml_nested_folds":
        meta = get("outer_metadata", {})
        audit = get("audit")
        lines = [
            f"<{r_class}>",
            f"  Outer folds: {len(get('folds') or [])}",
            f"  Inner v/repeats: {get('inner_v')}/{get('inner_repeats')}",
            f"  Target: {meta.get('generalization_target')}",
            f"  Audit: {_component(audit, 'status')}",
        ]
    elif r_class == "gp3ml_object_contract_validation":
        lines = [f"gp3ml object contract: {get('status')}"]
        table = _table_text(get("checks"))
    elif r_class == "gp3ml_preprocessor":
        lines = [
            f"<{r_class}> {len(get('predictors') or [])} raw predictors -> "
            f"{len(get('columns') or [])} model columns"
        ]
    elif r_class == "gp3ml_release_evidence":
        lines = [
            f"<{r_class}> version={get('version')}",
            f"  Object hashes: {len(get('object_hashes') or {})}",
            f"  File checksums: {len(get('file_md5') or {})}",
        ]
    elif r_class == "gp3ml_release_model_card":
        task = get("task")
        lines = [
            f"<{r_class}>",
            f"  Outcome: {_component(task, 'outcome')}",
            f"  Target: {get('generalization_target')}",
            f"  Selection recorded: {get('selection_procedure_recorded')}",
            f"  Uncertainty unit: {get('uncertainty_unit')}",
            f"  External validation: {get('external_validation_status')}",
        ]
    elif r_class == "gp3ml_reproducibility_audit":
        lines = [
            f"gp3ml reproducibility audit: {get('status')} "
            f"({get('files_scanned')} files, {_nrow(get('findings'))} findings)"
        ]
    elif r_class == "gp3ml_reproducibility_report":
        lines = [f"<{r_class}> {get('created_at')}"]
    elif r_class == "gp3ml_resample_evaluation":
        fold_status = get("fold_status")
        passed, review, failed = _status_counts(fold_status)
        lines = [
            f"<{r_class}>",
            f"  Target: {get('generalization_target')}",
            f"  Engine: {get('engine')}",
            f"  Folds: {_nrow(fold_status)}",
            f"  Passed/review/failed: {passed}/{review}/{failed}",
            f"  Predictions: {_nrow(get('predictions'))}",
        ]
    elif r_class == "gp3ml_resample_performance_summary":
        lines = [
            f"<{r_class}>",
            f"  Aggregation: {get('aggregation')}",
            f"  Generalization target: {get('generalization_target')}",
        ]
        table = _table_text(get("summary"))
    elif r_class == "gp3ml_resample_uncertainty":
        lines = [f"<{r_class}> unit={get('unit')}"]
        table = _table_text(get("summary"))
    elif r_class == "gp3ml_research_bundle":
        lines = [
            f"gp3ml research bundle: {len(get('handoffs') or [])} sources; "
            f"outcome={get('outcome')}; target={get('generalization_target')}"
        ]
    elif r_class == "gp3ml_research_bundle_validation":
        lines = [f"gp3ml research bundle validation: {get('status')}"]
        table = _table_text(get("checks"))
    elif r_class == "gp3ml_role_validation":
        lines = [f"<{r_class}> {get('status')}"]
        table = _table_text(get("checks"))
    elif r_class == "gp3ml_target_uncertainty":
        lines = [
            f"<{r_class}>",
            f"  Unit: {get('unit')}",
            f"  Target: {get('generalization_target')}",
            f"  Successful/failed replicates: {get('successful_replicates')}/{get('failed_replicates')}",
        ]
        table = _table_text(get("intervals"))
    elif r_class == "gp3ml_task":
        lines = [
            f"<{r_class}>",
            f"  type: {get('task_type')}",
            f"  outcome: {get('outcome')}",
            f"  target: {get('generalization_target')}",
            f"  purpose: {get('purpose')}",
        ]
    elif r_class == "gp3ml_transportability_report":
        lines = [
            f"<{r_class}>",
            f"  Status: {get('status')}",
            f"  Reason: {get('reason')}",
        ]
        table = _table_text(get("performance_comparison"))
    elif r_class == "gp3ml_tuning_grid":
        lines = [f"<{r_class}> candidates={_nrow(get('candidates'))}"]
        table = _table_text(
            get("candidates"),
            columns=["candidate_id", "label", "engine", "threshold", "complexity", "interpretability"],
        )
    else:
        return f"<{r_class}> {data!r}"

    if table:
        lines.append(table)
    return "\n".join(lines)
