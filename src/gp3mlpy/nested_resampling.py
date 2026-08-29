from __future__ import annotations

from typing import Any
import warnings as pywarnings

import numpy as np
import pandas as pd

from ._utils import hash_jsonable, seed_from, worst_status, write_tables
from .exceptions import GP3MLError
from .metrics import gazepoint_performance_metrics
from .model_engines import fit_gazepoint_model
from .model_tuning import GP3MLTuningGrid, _validate_direction, select_gazepoint_model, tune_gazepoint_model
from .objects import (
    GP3MLNestedEvaluation,
    GP3MLNestedEvaluationValidation,
    GP3MLNestedFolds,
    GP3MLNestedFoldsValidation,
    GP3MLNestedResamplingAudit,
)
from .resample_evaluation import _metric_long, _prediction_table, _predictions_from_model, _redeclare_task
from .resampling import GazepointGroupFolds, create_gazepoint_group_folds, validate_gazepoint_group_folds
from .task_governance import assert_gp3ml_use_case


def create_gazepoint_nested_folds(
    outer_folds: GazepointGroupFolds,
    inner_v: int = 3,
    inner_repeats: int = 1,
    seed: int = 1,
    continue_on_error: bool = False,
) -> GP3MLNestedFolds:
    """Build inner grouped folds solely from each outer analysis partition."""
    if not isinstance(outer_folds, GazepointGroupFolds):
        raise GP3MLError("`outer_folds` must be a `gazepoint_group_folds` object.")
    outer_validation = validate_gazepoint_group_folds(outer_folds)
    if outer_validation.status != "pass":
        raise GP3MLError("Outer folds must pass validation before nesting.")
    try:
        inner_v = int(inner_v); inner_repeats = int(inner_repeats)
    except Exception as exc:
        raise GP3MLError("`inner_v` must be at least two and `inner_repeats` positive.") from exc
    if inner_v < 2 or inner_repeats < 1:
        raise GP3MLError("`inner_v` must be at least two and `inner_repeats` positive.")
    nested: list[dict[str, Any]] = []
    for outer in outer_folds.folds.values():
        inner_seed = seed_from(int(seed), "inner", outer.fold_id)
        error = None; caught: list[str] = []; inner = None
        try:
            with pywarnings.catch_warnings(record=True) as ws:
                pywarnings.simplefilter("always")
                inner = create_gazepoint_group_folds(
                    data=outer.analysis,
                    outcome=outer_folds.metadata["outcome"],
                    predictors=outer_folds.metadata["predictors"],
                    feature_manifest=outer_folds.feature_manifest,
                    generalization_target=outer_folds.metadata["generalization_target"],
                    participant_id=outer_folds.metadata["participant_id"],
                    trial_id=outer_folds.metadata["trial_id"],
                    stimulus_id=outer_folds.metadata["stimulus_id"],
                    v=inner_v,
                    repeats=inner_repeats,
                    seed=inner_seed,
                    source_row_id=".gp3ml_inner_source_row",
                )
                caught = list(dict.fromkeys(str(w.message) for w in ws))
        except Exception as exc:
            error = str(exc)
        item = {
            "outer": outer,
            "inner": inner,
            "outer_fold_id": outer.fold_id,
            "status": "fail" if error is not None else "pass",
            "error": error,
            "warnings": caught,
            "seed": inner_seed,
        }
        nested.append(item)
        if error is not None and not continue_on_error:
            raise GP3MLError(f"Could not create inner folds for `{outer.fold_id}`: {error}")
    obj = GP3MLNestedFolds(
        folds=nested,
        outer_metadata=outer_folds.metadata,
        outer_feature_manifest=outer_folds.feature_manifest,
        inner_v=inner_v,
        inner_repeats=inner_repeats,
        seed=int(seed),
        outer_validation=outer_validation,
        call="create_gazepoint_nested_folds",
    )
    obj.audit = audit_gazepoint_nested_resampling(obj)
    obj.validation = validate_gazepoint_nested_folds(obj)
    return obj


def audit_gazepoint_nested_resampling(x: GP3MLNestedFolds) -> GP3MLNestedResamplingAudit:
    if not isinstance(x, GP3MLNestedFolds):
        raise GP3MLError("`x` must be a `gp3ml_nested_folds` object.")
    source_row_id = x.outer_metadata["source_row_id"]
    rows: list[dict[str, Any]] = []
    for item in x.folds:
        if item["status"] == "fail" or item["inner"] is None:
            rows.append({
                "outer_fold_id": item["outer_fold_id"], "inner_fold_id": None, "status": "fail",
                "outer_assessment_overlap": np.nan, "inner_analysis_assessment_overlap": np.nan,
                "message": item["error"],
            })
            continue
        outer_assessment_rows = set(item["outer"].assessment[source_row_id].tolist())
        for inner in item["inner"].folds.values():
            ia = set(inner.analysis[source_row_id].tolist()); ie = set(inner.assessment[source_row_id].tolist())
            ix = set(inner.excluded[source_row_id].tolist()) if len(inner.excluded) else set()
            oaia = len(outer_assessment_rows & ia); oaie = len(outer_assessment_rows & ie); oaix = len(outer_assessment_rows & ix)
            iaie = len(ia & ie); iaix = len(ia & ix); ieix = len(ie & ix)
            overlaps = [oaia, oaie, oaix, iaie, iaix, ieix]
            status = "fail" if any(v > 0 for v in overlaps) else "review" if inner.leakage_audit.status == "review" else "pass"
            rows.append({
                "outer_fold_id": item["outer_fold_id"], "inner_fold_id": inner.fold_id, "status": status,
                "outer_assessment_inner_analysis_overlap": oaia,
                "outer_assessment_inner_assessment_overlap": oaie,
                "outer_assessment_inner_excluded_overlap": oaix,
                "inner_analysis_assessment_overlap": iaie,
                "inner_analysis_excluded_overlap": iaix,
                "inner_assessment_excluded_overlap": ieix,
                "outer_assessment_overlap": oaia + oaie + oaix,
                "message": "No outer-assessment or inner-partition row overlap detected." if status == "pass" else "Nested partition overlap requires review.",
            })
    checks = pd.DataFrame(rows)
    return GP3MLNestedResamplingAudit(status=worst_status(checks.status.tolist()), checks=checks, issues=checks.loc[checks.status != "pass"].reset_index(drop=True))


def validate_gazepoint_nested_folds(x: GP3MLNestedFolds) -> GP3MLNestedFoldsValidation:
    if not isinstance(x, GP3MLNestedFolds):
        raise GP3MLError("`x` must be a `gp3ml_nested_folds` object.")
    n_outer = len(x.folds); n_inner_ready = sum(z["inner"] is not None for z in x.folds)
    target_ok = all(z["inner"] is None or z["inner"].metadata["generalization_target"] == x.outer_metadata["generalization_target"] for z in x.folds)
    checks = pd.DataFrame({
        "check_id": ["outer_folds_retained", "inner_folds_created", "outer_assessment_isolation", "generalization_target_preserved"],
        "status": [
            "pass" if n_outer == x.outer_metadata["n_folds_total"] else "fail",
            "pass" if n_inner_ready == n_outer else "review",
            x.audit.status,
            "pass" if target_ok else "fail",
        ],
        "message": [
            f"Retained {n_outer} outer folds.",
            f"Created inner folds for {n_inner_ready} of {n_outer} outer folds.",
            "Inner resampling is audited against every outer assessment partition.",
            f"Target: {x.outer_metadata['generalization_target']}.",
        ],
    })
    return GP3MLNestedFoldsValidation(status=worst_status(checks.status), checks=checks, issues=checks.loc[checks.status != "pass"].reset_index(drop=True))


_DEFAULT_RATIONALE = "Candidate selected by the predeclared nested-resampling rule and retained for human review."


def evaluate_gazepoint_nested_resampling(
    nested_folds: GP3MLNestedFolds,
    task: Any,
    tuning_grid: GP3MLTuningGrid,
    selection_metric: str,
    direction: str,
    predictors: Any = None,
    minimum_success_prop: float = 0.8,
    tie_breakers: Any = None,
    selection_rationale: str = _DEFAULT_RATIONALE,
    seed: int = 1,
    keep_models: bool = False,
    continue_on_error: bool = True,
) -> GP3MLNestedEvaluation:
    """Run governed tuning on inner folds and score selected refits on outer assessment only."""
    if not isinstance(nested_folds, GP3MLNestedFolds):
        raise GP3MLError("`nested_folds` must be a `gp3ml_nested_folds` object.")
    if not isinstance(tuning_grid, GP3MLTuningGrid):
        raise GP3MLError("`tuning_grid` must be a `gp3ml_tuning_grid` object.")
    assert_gp3ml_use_case(task)
    direction = _validate_direction(selection_metric, direction)
    predictors = list(nested_folds.outer_metadata["predictors"] if predictors is None else predictors)
    source_row_id = nested_folds.outer_metadata["source_row_id"]
    results: list[dict[str, Any]] = []
    for item in nested_folds.folds:
        outer_seed = seed_from(int(seed), "outer", item["outer_fold_id"])
        error = None; caught: list[str] = []; value = None
        try:
            with pywarnings.catch_warnings(record=True) as ws:
                pywarnings.simplefilter("always")
                if item["inner"] is None:
                    raise GP3MLError(f"Inner folds are unavailable: {item['error']}")
                outer_task = _redeclare_task(item["outer"].analysis, task)
                tuned = tune_gazepoint_model(
                    item["inner"], outer_task, tuning_grid, predictors=predictors,
                    seed=seed_from(outer_seed, "tune"), continue_on_error=True, keep_evaluations=True,
                )
                selection = select_gazepoint_model(
                    tuned, metric=selection_metric, direction=direction,
                    minimum_success_prop=minimum_success_prop, tie_breakers=tie_breakers,
                    rationale=f"{selection_rationale} Outer fold: {item['outer_fold_id']}",
                )
                candidate = selection.candidate.iloc[0]
                outer_model = fit_gazepoint_model(
                    item["outer"].analysis, outer_task, predictors,
                    engine=str(candidate.engine), preprocessor_args=dict(candidate.preprocessor_args),
                    engine_args=dict(candidate.engine_args), seed=seed_from(outer_seed, "refit"),
                    threshold=float(candidate.threshold),
                )
                prediction, probability = _predictions_from_model(outer_model, item["outer"].assessment, task)
                prediction_table = _prediction_table(
                    item["outer"].assessment, task, item["outer"], prediction, probability, source_row_id,
                    candidate_id=selection.candidate_id, stage="outer_assessment",
                )
                metrics = gazepoint_performance_metrics(
                    task, item["outer"].assessment[task.outcome], prediction, probability, float(candidate.threshold)
                )
                metrics_long = _metric_long(metrics, {
                    "repeat": item["outer"]["repeat"], "fold": item["outer"].fold,
                    "fold_id": item["outer"].fold_id, "candidate_id": selection.candidate_id,
                })
                value = {"tuning": tuned, "selection": selection, "model": outer_model, "predictions": prediction_table, "metrics_long": metrics_long}
                caught = list(dict.fromkeys(str(w.message) for w in ws))
        except Exception as exc:
            error = str(exc)
        failed = error is not None
        result = {
            "outer_fold_id": item["outer_fold_id"], "repeat": item["outer"]["repeat"], "fold": item["outer"].fold,
            "status": "fail" if failed else "review" if caught else "pass", "error": error, "warnings": caught,
            "tuning": None if failed else value["tuning"], "selection": None if failed else value["selection"],
            "model": value["model"] if (not failed and keep_models) else None,
            "predictions": pd.DataFrame() if failed else value["predictions"],
            "metrics": pd.DataFrame() if failed else value["metrics_long"], "excluded": item["outer"].excluded.copy(),
            "outer_analysis_hash": hash_jsonable(item["outer"].analysis[source_row_id].tolist(), algorithm="md5"),
            "outer_assessment_hash": hash_jsonable(item["outer"].assessment[source_row_id].tolist(), algorithm="md5"),
            "seed": outer_seed,
        }
        results.append(result)
        if failed and not continue_on_error:
            raise GP3MLError(f"Outer fold `{item['outer_fold_id']}` failed: {error}")
    fold_status = pd.DataFrame([{
        "repeat": z["repeat"], "fold": z["fold"], "fold_id": z["outer_fold_id"], "status": z["status"],
        "selected_candidate": None if z["selection"] is None else z["selection"].candidate_id,
        "n_predictions": len(z["predictions"]), "warning_count": len(z["warnings"]), "error": z["error"],
    } for z in results])
    predictions = pd.concat([z["predictions"] for z in results if len(z["predictions"])], ignore_index=True) if any(len(z["predictions"]) for z in results) else pd.DataFrame()
    metrics = pd.concat([z["metrics"] for z in results if len(z["metrics"])], ignore_index=True) if any(len(z["metrics"]) for z in results) else pd.DataFrame()
    excluded_parts = []
    for z in results:
        if len(z["excluded"]):
            out = z["excluded"].copy(); out["repeat"] = z["repeat"]; out["fold"] = z["fold"]; out["fold_id"] = z["outer_fold_id"]; excluded_parts.append(out)
    excluded = pd.concat(excluded_parts, ignore_index=True) if excluded_parts else pd.DataFrame()
    obj = GP3MLNestedEvaluation(
        results=results, predictions=predictions, metrics=metrics, excluded=excluded, fold_status=fold_status,
        nested_folds_audit=nested_folds.audit, nested_folds_validation=nested_folds.validation,
        task=task, predictors=predictors, tuning_grid=tuning_grid, selection_metric=selection_metric,
        direction=direction, generalization_target=nested_folds.outer_metadata["generalization_target"],
        seed=int(seed), keep_models=bool(keep_models), call="evaluate_gazepoint_nested_resampling",
    )
    obj.validation = validate_gazepoint_nested_evaluation(obj)
    return obj


def validate_gazepoint_nested_evaluation(x: GP3MLNestedEvaluation) -> GP3MLNestedEvaluationValidation:
    if not isinstance(x, GP3MLNestedEvaluation):
        raise GP3MLError("`x` must be a `gp3ml_nested_evaluation` object.")
    selections = [z["selection"] is not None for z in x.results]
    outer_only = len(x.predictions) == 0 or bool((x.predictions.stage == "outer_assessment").all())
    checks = pd.DataFrame({
        "check_id": ["nested_partition_audit", "outer_assessment_only", "selection_recorded", "failures_retained", "generalization_target_preserved"],
        "status": [
            x.nested_folds_audit.status, "pass" if outer_only else "fail",
            "pass" if all(selections) else "review", "pass",
            "pass" if x.task.generalization_target == x.generalization_target else "fail",
        ],
        "message": [
            "Nested partitions retain the outer-assessment isolation audit.",
            "Reported predictions come only from outer assessment partitions.",
            f"Recorded governed selections for {sum(selections)} of {len(selections)} outer folds.",
            f"Retained {int((x.fold_status.status == 'fail').sum())} failed outer folds.",
            f"Target: {x.generalization_target}.",
        ],
    })
    return GP3MLNestedEvaluationValidation(status=worst_status(checks.status), checks=checks, issues=checks.loc[checks.status != "pass"].reset_index(drop=True))


def write_gazepoint_nested_evaluation(
    x: GP3MLNestedEvaluation,
    directory: str,
    prefix: str = "gazepoint_nested_evaluation",
    overwrite: bool = False,
) -> dict[str, str]:
    if not isinstance(x, GP3MLNestedEvaluation):
        raise GP3MLError("`x` must be a `gp3ml_nested_evaluation` object.")
    selections = pd.DataFrame([{
        "fold_id": z["outer_fold_id"], "candidate_id": z["selection"].candidate_id,
        "metric": z["selection"].primary_metric, "direction": z["selection"].direction,
        "value": z["selection"].primary_value, "rationale": z["selection"].rationale,
    } for z in x.results if z["selection"] is not None])
    return write_tables({
        "fold_status": x.fold_status, "selections": selections, "predictions": x.predictions, "metrics": x.metrics,
        "excluded": x.excluded, "validation": x.validation.checks, "partition_audit": x.nested_folds_audit.checks,
    }, directory, prefix, overwrite)


def _nested_repr(self: GP3MLNestedFolds) -> str:
    return f"<gp3ml_nested_folds>\n  Outer folds: {len(self.folds)}\n  Inner v/repeats: {self.inner_v}/{self.inner_repeats}\n  Target: {self.outer_metadata['generalization_target']}\n  Audit: {self.audit.status}"


def _eval_repr(self: GP3MLNestedEvaluation) -> str:
    return f"<gp3ml_nested_evaluation>\n  Target: {self.generalization_target}\n  Outer folds: {len(self.fold_status)}\n  Failed outer folds: {int((self.fold_status.status=='fail').sum())}\n  Outer assessment predictions: {len(self.predictions)}"

GP3MLNestedFolds.__repr__ = _nested_repr  # type: ignore[method-assign]
GP3MLNestedEvaluation.__repr__ = _eval_repr  # type: ignore[method-assign]
