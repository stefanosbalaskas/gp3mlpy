from __future__ import annotations

from itertools import product
from typing import Any
import warnings as pywarnings

import numpy as np
import pandas as pd

from ._utils import hash_jsonable, seed_from, timestamp, worst_status, write_tables
from .exceptions import GP3MLError
from .objects import GP3MLModelSelection, GP3MLModelTuning, GP3MLModelTuningValidation, GP3MLTuningGrid
from .resample_evaluation import evaluate_gazepoint_group_folds


def _as_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)) and not isinstance(value, str):
        return list(value)
    return [value]


def _r_arg_text(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if bool(value) else "FALSE"
    return str(value)


def _expand_grid_list(grid: dict[str, Any] | None) -> list[dict[str, Any]]:
    if grid is None or len(grid) == 0:
        return [{}]
    if not isinstance(grid, dict) or any(not isinstance(k, str) or not k for k in grid):
        raise GP3MLError("A tuning grid must be a named list of candidate values.")
    names = list(grid)
    values = [_as_values(grid[n]) for n in names]
    if any(len(v) < 1 for v in values):
        raise GP3MLError("Every tuning parameter needs at least one value.")
    # R expand.grid varies the first argument fastest.
    out: list[dict[str, Any]] = []
    for combo_rev in product(*reversed(values)):
        combo = list(reversed(combo_rev))
        out.append(dict(zip(names, combo, strict=True)))
    return out


def _collapse_args(x: dict[str, Any]) -> str:
    if not x:
        return "default"
    parts = []
    for name, value in x.items():
        values = _as_values(value)
        parts.append(f"{name}=" + "/".join(_r_arg_text(v) for v in values))
    return ",".join(parts)


def _candidate_label(engine: str, engine_args: dict[str, Any], prep_args: dict[str, Any], threshold: float) -> str:
    return f"{engine} [engine:{_collapse_args(engine_args)}; prep:{_collapse_args(prep_args)}; threshold={threshold:g}]"


def _recycle_metadata(value: Any, n: int, name: str) -> list[Any]:
    vals = _as_values(value)
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise GP3MLError(f"`{name}` must have length one or the candidate count.")
    return vals


def create_gazepoint_tuning_grid(
    engine: Any,
    engine_grid: dict[str, Any] | None = None,
    preprocessor_grid: dict[str, Any] | None = None,
    thresholds: Any = 0.5,
    complexity: Any = np.nan,
    interpretability: Any = np.nan,
    labels: Any = None,
) -> GP3MLTuningGrid:
    """Materialize an explicit governed candidate grid without selecting a winner."""
    engines = list(dict.fromkeys(str(x) for x in _as_values(engine)))
    if not engines or any(not e or e == "None" for e in engines):
        raise GP3MLError("`engine` must contain one or more non-empty engine names.")
    try:
        threshold_values = [float(x) for x in _as_values(thresholds)]
    except (TypeError, ValueError) as exc:
        raise GP3MLError("`thresholds` must contain finite values strictly between zero and one.") from exc
    if not threshold_values or any(not np.isfinite(x) or x <= 0 or x >= 1 for x in threshold_values):
        raise GP3MLError("`thresholds` must contain finite values strictly between zero and one.")
    engine_combinations = _expand_grid_list(engine_grid)
    prep_combinations = _expand_grid_list(preprocessor_grid)
    rows: list[dict[str, Any]] = []
    # R expand.grid(engine, engine_index, preprocessor_index, threshold): first varies fastest.
    for threshold in threshold_values:
        for prep in prep_combinations:
            for eng_args in engine_combinations:
                for eng in engines:
                    rows.append({"engine": eng, "threshold": threshold, "engine_args": dict(eng_args), "preprocessor_args": dict(prep)})
    n = len(rows)
    complexities = _recycle_metadata(complexity, n, "complexity")
    interpretabilities = _recycle_metadata(interpretability, n, "interpretability")
    generated = [_candidate_label(r["engine"], r["engine_args"], r["preprocessor_args"], r["threshold"]) for r in rows]
    if labels is not None:
        generated = [str(v) for v in _recycle_metadata(labels, n, "labels")]
    candidates = pd.DataFrame(
        {
            "candidate_id": [f"candidate_{i:03d}" for i in range(1, n + 1)],
            "label": generated,
            "engine": [r["engine"] for r in rows],
            "threshold": [r["threshold"] for r in rows],
            "complexity": complexities,
            "interpretability": interpretabilities,
            "engine_args": [r["engine_args"] for r in rows],
            "preprocessor_args": [r["preprocessor_args"] for r in rows],
        }
    )
    return GP3MLTuningGrid(candidates=candidates, created=timestamp(), call="create_gazepoint_tuning_grid")


def _metric_direction(metric: str) -> str:
    return "minimize" if metric in {"rmse", "mae", "log_loss", "brier", "ece", "calibration_intercept_abs", "calibration_slope_abs_error"} else "maximize"


def _comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        candidate = result["candidate"].iloc[0]
        base = {
            "candidate_id": result["candidate_id"],
            "label": candidate["label"],
            "engine": candidate["engine"],
            "threshold": float(candidate["threshold"]),
            "complexity": str(candidate["complexity"]),
            "interpretability": str(candidate["interpretability"]),
            "candidate_status": result["status"],
            "success_prop": float(result["success_prop"]),
            "failed_folds": int((result["fold_status"].status == "fail").sum()) if len(result["fold_status"]) else np.nan,
            "error": result["error"],
        }
        metrics = result["metrics"]
        if metrics is None or len(metrics) == 0:
            rows.append({**base, "metric": None, "mean": np.nan, "sd": np.nan, "n_folds": 0, "direction": None})
            continue
        for metric in pd.unique(metrics.metric):
            vals = pd.to_numeric(metrics.loc[metrics.metric == metric, "value"], errors="coerce")
            finite = vals[np.isfinite(vals)]
            rows.append({
                **base,
                "metric": metric,
                "mean": float(vals.mean(skipna=True)),
                "sd": float(vals.std(ddof=1, skipna=True)) if vals.notna().sum() > 1 else np.nan,
                "n_folds": int(len(finite)),
                "direction": _metric_direction(str(metric)),
            })
    return pd.DataFrame(rows)


def tune_gazepoint_model(
    folds: Any,
    task: Any,
    tuning_grid: GP3MLTuningGrid,
    predictors: Any = None,
    metrics: Any = None,
    seed: int = 1,
    continue_on_error: bool = True,
    keep_evaluations: bool = True,
) -> GP3MLModelTuning:
    """Evaluate every explicit candidate on the same grouped folds, retaining failures."""
    if not isinstance(tuning_grid, GP3MLTuningGrid):
        raise GP3MLError("`tuning_grid` must be a `gp3ml_tuning_grid` object.")
    requested_metrics = None if metrics is None else [str(x) for x in _as_values(metrics)]
    results: list[dict[str, Any]] = []
    for _, candidate_row in tuning_grid.candidates.iterrows():
        candidate = pd.DataFrame([candidate_row.to_dict()])
        candidate_id = str(candidate_row.candidate_id)
        candidate_seed = seed_from(int(seed), candidate_id)
        error = None
        captured_warnings: list[str] = []
        evaluation = None
        try:
            with pywarnings.catch_warnings(record=True) as ws:
                pywarnings.simplefilter("always")
                evaluation = evaluate_gazepoint_group_folds(
                    folds=folds,
                    task=task,
                    predictors=predictors,
                    engine=str(candidate_row.engine),
                    preprocessor_args=dict(candidate_row.preprocessor_args),
                    engine_args=dict(candidate_row.engine_args),
                    threshold=float(candidate_row.threshold),
                    seed=candidate_seed,
                    assess_calibration=task.task_type == "classification",
                    calibration_bootstrap=0,
                    keep_models=False,
                    continue_on_error=True,
                )
                captured_warnings = list(dict.fromkeys(str(w.message) for w in ws))
        except Exception as exc:
            error = str(exc)
        failed = error is not None
        fold_status = pd.DataFrame() if failed else evaluation.fold_status
        success_prop = 0.0 if failed or len(fold_status) == 0 else float((fold_status.status != "fail").mean())
        all_folds_failed = (not failed) and len(fold_status) > 0 and bool((fold_status.status == "fail").all())
        candidate_failed = failed or all_folds_failed
        if all_folds_failed:
            errors = [str(x) for x in fold_status.error.dropna().unique() if str(x)]
            error = " | ".join(errors) if errors else "Every grouped fold failed."
        fold_warnings = [] if failed or len(fold_status) == 0 else [str(x) for x in fold_status.warnings.dropna().unique() if str(x)]
        warnings_all = list(dict.fromkeys([*captured_warnings, *fold_warnings]))
        candidate_metrics = pd.DataFrame() if candidate_failed else evaluation.metrics.copy()
        if requested_metrics is not None and len(candidate_metrics):
            candidate_metrics = candidate_metrics.loc[candidate_metrics.metric.isin(requested_metrics)].reset_index(drop=True)
        result = {
            "candidate_id": candidate_id,
            "candidate": candidate,
            "status": "fail" if candidate_failed else evaluation.validation.status,
            "error": error,
            "warnings": warnings_all,
            "metrics": candidate_metrics,
            "fold_status": fold_status,
            "success_prop": success_prop,
            "evaluation": evaluation if (not failed and keep_evaluations) else None,
            "seed": candidate_seed,
        }
        results.append(result)
        if candidate_failed and not continue_on_error:
            raise GP3MLError(f"Candidate `{candidate_id}` failed: {error}")
    comparison = _comparison_table(results)
    obj = GP3MLModelTuning(
        grid=tuning_grid,
        results=results,
        comparison=comparison,
        task=task,
        predictors=list(folds.metadata["predictors"] if predictors is None else predictors),
        folds_metadata=folds.metadata,
        metrics_requested=requested_metrics,
        seed=int(seed),
        keep_evaluations=bool(keep_evaluations),
        selection=None,
        call="tune_gazepoint_model",
    )
    obj.validation = validate_gazepoint_model_tuning(obj)
    return obj


def compare_gazepoint_models(x: GP3MLModelTuning, metrics: Any = None) -> pd.DataFrame:
    if not isinstance(x, GP3MLModelTuning):
        raise GP3MLError("`x` must be a `gp3ml_model_tuning` object.")
    out = x.comparison.copy()
    if metrics is not None:
        requested = {str(v) for v in _as_values(metrics)}
        out = out.loc[out.metric.isna() | out.metric.isin(requested)].copy()
    return out.reset_index(drop=True)


def _validate_direction(metric: str, direction: str) -> str:
    if direction not in {"maximize", "minimize"}:
        raise GP3MLError("`direction` must be one of: maximize, minimize.")
    if metric == "accuracy":
        raise GP3MLError("`accuracy` cannot be the primary governed selection metric. Use a discrimination, calibration, or error metric and report accuracy only as a secondary measure.")
    return direction


def select_gazepoint_model(
    x: GP3MLModelTuning,
    metric: str,
    direction: str,
    minimum_success_prop: float = 0.8,
    tie_breakers: Any = None,
    rationale: str | None = None,
) -> GP3MLModelSelection:
    """Record an explicit human-governed candidate selection; no refit is performed."""
    if not isinstance(x, GP3MLModelTuning):
        raise GP3MLError("`x` must be a `gp3ml_model_tuning` object.")
    if not isinstance(metric, str) or not metric:
        raise GP3MLError("An explicit primary `metric` is required.")
    direction = _validate_direction(metric, direction)
    if not isinstance(rationale, str) or not rationale.strip():
        raise GP3MLError("A non-empty human review `rationale` is required.")
    if isinstance(minimum_success_prop, bool) or not isinstance(minimum_success_prop, (int, float, np.integer, np.floating)) or not 0 <= float(minimum_success_prop) <= 1:
        raise GP3MLError("`minimum_success_prop` must be between zero and one.")
    comparison = x.comparison
    eligible = comparison.loc[
        (comparison.metric == metric)
        & (comparison.candidate_status != "fail")
        & (comparison.success_prop >= float(minimum_success_prop))
        & np.isfinite(pd.to_numeric(comparison["mean"], errors="coerce"))
    ].copy()
    if len(eligible) == 0:
        raise GP3MLError("No eligible candidate has the requested metric.")
    eligible = eligible.sort_values("mean", ascending=direction == "minimize", kind="stable")
    best_value = float(eligible.iloc[0]["mean"])
    tied_ids = eligible.loc[eligible["mean"] == best_value, "candidate_id"].tolist()
    tie_rows = []
    if len(tied_ids) > 1 and tie_breakers is not None:
        for secondary in [str(v) for v in _as_values(tie_breakers)]:
            if secondary == "accuracy":
                continue
            secondary_rows = comparison.loc[(comparison.candidate_id.isin(tied_ids)) & (comparison.metric == secondary)].copy()
            secondary_rows = secondary_rows.loc[np.isfinite(pd.to_numeric(secondary_rows["mean"], errors="coerce"))]
            if len(secondary_rows) == 0:
                continue
            secondary_direction = _metric_direction(secondary)
            secondary_rows = secondary_rows.sort_values("mean", ascending=secondary_direction == "minimize", kind="stable")
            secondary_best = float(secondary_rows.iloc[0]["mean"])
            tied_ids = secondary_rows.loc[secondary_rows["mean"] == secondary_best, "candidate_id"].tolist()
            tie_rows.append({"metric": secondary, "direction": secondary_direction, "best_value": secondary_best, "remaining_candidates": ",".join(tied_ids)})
            if len(tied_ids) == 1:
                break
    if len(tied_ids) > 1:
        candidate_rows = x.grid.candidates.loc[x.grid.candidates.candidate_id.isin(tied_ids)].copy()
        complexity_numeric = pd.to_numeric(candidate_rows.complexity, errors="coerce")
        if complexity_numeric.notna().all() and np.isfinite(complexity_numeric).all():
            min_complexity = complexity_numeric.min()
            tied_ids = candidate_rows.loc[complexity_numeric == min_complexity, "candidate_id"].tolist()
    if len(tied_ids) != 1:
        raise GP3MLError("Selection remains tied among: " + ", ".join(tied_ids) + ". Supply defensible tie breakers or revise the candidate grid.")
    selected_id = tied_ids[0]
    selected_candidate = x.grid.candidates.loc[x.grid.candidates.candidate_id == selected_id].copy()
    primary = comparison.loc[(comparison.candidate_id == selected_id) & (comparison.metric == metric), "mean"].iloc[0]
    return GP3MLModelSelection(
        candidate_id=selected_id,
        candidate=selected_candidate,
        primary_metric=metric,
        direction=direction,
        primary_value=float(primary),
        minimum_success_prop=float(minimum_success_prop),
        tie_breakers=pd.DataFrame(tie_rows),
        rationale=rationale.strip(),
        eligible_candidates=eligible.candidate_id.tolist(),
        selection_time=timestamp(),
        tuning_hash=hash_jsonable(x.comparison, algorithm="md5"),
        refit_performed=False,
        autonomous_selection=False,
    )


def validate_gazepoint_model_tuning(x: GP3MLModelTuning) -> GP3MLModelTuningValidation:
    if not isinstance(x, GP3MLModelTuning):
        raise GP3MLError("`x` must be a `gp3ml_model_tuning` object.")
    expected = len(x.grid.candidates)
    observed = len(x.results)
    comparison_ids = set(x.comparison.candidate_id.dropna().astype(str))
    candidate_ids = set(x.grid.candidates.candidate_id.astype(str))
    failed = sum(1 for z in x.results if z["status"] == "fail")
    checks = pd.DataFrame(
        {
            "check_id": ["all_candidates_retained", "candidate_ids_complete", "failures_retained", "no_implicit_selection", "generalization_target_preserved"],
            "status": [
                "pass" if expected == observed else "fail",
                "pass" if candidate_ids == comparison_ids else "fail",
                "pass",
                "pass" if x.selection is None else "review",
                "pass" if x.task.generalization_target == x.folds_metadata["generalization_target"] else "fail",
            ],
            "message": [
                f"Retained {observed} of {expected} candidates.",
                "Comparison table retains every candidate identifier.",
                f"Explicitly retained {failed} failed candidates.",
                "Tuning does not automatically declare a winner.",
                f"Generalization target: {x.task.generalization_target}.",
            ],
        }
    )
    return GP3MLModelTuningValidation(status=worst_status(checks.status), checks=checks, issues=checks.loc[checks.status != "pass"].reset_index(drop=True))


def _flatten_args(x: dict[str, Any]) -> str:
    if not x:
        return ""
    return ";".join(f"{name}=" + "/".join(_r_arg_text(v) for v in _as_values(value)) for name, value in x.items())


def write_gazepoint_model_tuning(
    x: GP3MLModelTuning,
    directory: str,
    prefix: str = "gazepoint_model_tuning",
    selection: GP3MLModelSelection | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    if not isinstance(x, GP3MLModelTuning):
        raise GP3MLError("`x` must be a `gp3ml_model_tuning` object.")
    if selection is not None and not isinstance(selection, GP3MLModelSelection):
        raise GP3MLError("`selection` must be a `gp3ml_model_selection` object.")
    candidates = x.grid.candidates.copy()
    candidates["engine_args"] = candidates.engine_args.map(_flatten_args)
    candidates["preprocessor_args"] = candidates.preprocessor_args.map(_flatten_args)
    candidate_status = pd.DataFrame([
        {
            "candidate_id": r["candidate_id"], "status": r["status"], "success_prop": r["success_prop"],
            "warning_count": len(r["warnings"]), "warnings": " | ".join(r["warnings"]), "error": r["error"],
        }
        for r in x.results
    ])
    if selection is None:
        selection_table = pd.DataFrame()
    else:
        selection_table = pd.DataFrame([{
            "candidate_id": selection.candidate_id, "primary_metric": selection.primary_metric, "direction": selection.direction,
            "primary_value": selection.primary_value, "minimum_success_prop": selection.minimum_success_prop,
            "rationale": selection.rationale, "autonomous_selection": selection.autonomous_selection, "refit_performed": selection.refit_performed,
        }])
    return write_tables({"candidates": candidates, "candidate_status": candidate_status, "comparison": x.comparison, "selection": selection_table, "validation": x.validation.checks}, directory, prefix, overwrite)


def _grid_repr(self: GP3MLTuningGrid) -> str:
    return f"<gp3ml_tuning_grid> candidates={len(self.candidates)}\n" + self.candidates[["candidate_id", "label", "engine", "threshold", "complexity", "interpretability"]].to_string(index=False)


def _tuning_repr(self: GP3MLModelTuning) -> str:
    failed = sum(1 for z in self.results if z["status"] == "fail")
    return f"<gp3ml_model_tuning>\n  Candidates: {len(self.grid.candidates)}\n  Failed candidates: {failed}\n  Automatic winner: none"


def _selection_repr(self: GP3MLModelSelection) -> str:
    return f"<gp3ml_model_selection>\n  Candidate: {self.candidate_id}\n  Primary metric: {self.primary_metric} ({self.direction})\n  Value: {self.primary_value}\n  Human rationale: {self.rationale}\n  Refit performed: no"

GP3MLTuningGrid.__repr__ = _grid_repr  # type: ignore[method-assign]
GP3MLModelTuning.__repr__ = _tuning_repr  # type: ignore[method-assign]
GP3MLModelSelection.__repr__ = _selection_repr  # type: ignore[method-assign]
