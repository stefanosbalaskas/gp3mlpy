from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from gp3mlpy.model_tuning import (
    compare_gazepoint_models,
    create_gazepoint_tuning_grid,
    select_gazepoint_model,
    validate_gazepoint_model_tuning,
    write_gazepoint_model_tuning,
)
from gp3mlpy.objects import GP3MLModelSelection, GP3MLModelTuning, GP3MLTask, GP3MLTuningGrid


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Series):
        return [_jsonable(v) for v in value.tolist()]
    return str(value) if not isinstance(value, str) else value


def _frame(df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for _, row in df.iterrows():
        rows.append({str(c): _jsonable(row[c]) for c in df.columns})
    return {"columns": [str(c) for c in df.columns], "rows": rows}


def _grid_value(grid: GP3MLTuningGrid) -> dict[str, Any]:
    return {"class": grid.r_class, "candidates": _frame(grid.candidates)}


def _base_grid(*, textual_complexity: bool = False) -> GP3MLTuningGrid:
    complexity = ["low", "low", "high"] if textual_complexity else [2, 1, 3]
    candidates = pd.DataFrame(
        {
            "candidate_id": ["candidate_001", "candidate_002", "candidate_003"],
            "label": ["glm A", "glm B", "ranger C"],
            "engine": ["glm", "glm", "ranger"],
            "threshold": [0.5, 0.5, 0.5],
            "complexity": complexity,
            "interpretability": ["high", "high", "medium"],
            "engine_args": [{"max_iter": 50}, {"max_iter": 100}, {"trees": 100}],
            "preprocessor_args": [{"center": True}, {"center": False}, {}],
        }
    )
    return GP3MLTuningGrid(candidates=candidates, created=None, call="fixture")


def _base_comparison(*, unique_primary: bool = False, low_success_second: bool = False) -> pd.DataFrame:
    primary_1 = 0.82 if unique_primary else 0.80
    success_2 = 0.70 if low_success_second else 1.0
    return pd.DataFrame(
        [
            {"candidate_id": "candidate_001", "label": "glm A", "engine": "glm", "threshold": 0.5, "complexity": "2", "interpretability": "high", "candidate_status": "pass", "success_prop": 1.0, "failed_folds": 0, "error": None, "metric": "roc_auc", "mean": primary_1, "sd": 0.02, "n_folds": 3, "direction": "maximize"},
            {"candidate_id": "candidate_001", "label": "glm A", "engine": "glm", "threshold": 0.5, "complexity": "2", "interpretability": "high", "candidate_status": "pass", "success_prop": 1.0, "failed_folds": 0, "error": None, "metric": "brier", "mean": 0.20, "sd": 0.01, "n_folds": 3, "direction": "minimize"},
            {"candidate_id": "candidate_002", "label": "glm B", "engine": "glm", "threshold": 0.5, "complexity": "1", "interpretability": "high", "candidate_status": "pass", "success_prop": success_2, "failed_folds": 0, "error": None, "metric": "roc_auc", "mean": 0.80, "sd": 0.03, "n_folds": 3, "direction": "maximize"},
            {"candidate_id": "candidate_002", "label": "glm B", "engine": "glm", "threshold": 0.5, "complexity": "1", "interpretability": "high", "candidate_status": "pass", "success_prop": success_2, "failed_folds": 0, "error": None, "metric": "brier", "mean": 0.15, "sd": 0.02, "n_folds": 3, "direction": "minimize"},
            {"candidate_id": "candidate_003", "label": "ranger C", "engine": "ranger", "threshold": 0.5, "complexity": "3", "interpretability": "medium", "candidate_status": "fail", "success_prop": 0.5, "failed_folds": 2, "error": "fixture failure", "metric": None, "mean": np.nan, "sd": np.nan, "n_folds": 0, "direction": None},
        ]
    )


def _results() -> list[dict[str, Any]]:
    return [
        {"candidate_id": "candidate_001", "status": "pass", "success_prop": 1.0, "warnings": [], "error": None},
        {"candidate_id": "candidate_002", "status": "pass", "success_prop": 1.0, "warnings": ["fixture warning"], "error": None},
        {"candidate_id": "candidate_003", "status": "fail", "success_prop": 0.5, "warnings": [], "error": "fixture failure"},
    ]


def _base_tuning(*, textual_complexity: bool = False, unique_primary: bool = False, low_success_second: bool = False) -> GP3MLModelTuning:
    grid = _base_grid(textual_complexity=textual_complexity)
    task = GP3MLTask(generalization_target="new_participants")
    tuning = GP3MLModelTuning(
        grid=grid,
        results=_results(),
        comparison=_base_comparison(unique_primary=unique_primary, low_success_second=low_success_second),
        task=task,
        predictors=["x"],
        folds_metadata={"generalization_target": "new_participants"},
        metrics_requested=None,
        seed=1,
        keep_evaluations=False,
        selection=None,
        call="fixture",
    )
    tuning.validation = validate_gazepoint_model_tuning(tuning)
    return tuning


def _selection_value(selection: GP3MLModelSelection) -> dict[str, Any]:
    return {
        "class": selection.r_class,
        "candidate_id": selection.candidate_id,
        "primary_metric": selection.primary_metric,
        "direction": selection.direction,
        "primary_value": _jsonable(selection.primary_value),
        "minimum_success_prop": _jsonable(selection.minimum_success_prop),
        "tie_breakers": _frame(selection.tie_breakers),
        "rationale": selection.rationale,
        "eligible_candidates": [_jsonable(x) for x in selection.eligible_candidates],
        "refit_performed": bool(selection.refit_performed),
        "autonomous_selection": bool(selection.autonomous_selection),
        "candidate": _frame(selection.candidate),
    }


def _validation_value(value: Any) -> dict[str, Any]:
    return {"class": value.r_class, "status": value.status, "checks": _frame(value.checks), "issues": _frame(value.issues)}


def _capture(fn) -> dict[str, Any]:
    try:
        return {"status": "success", "value": fn()}
    except Exception as exc:
        return {"status": "error", "error_type": type(exc).__name__, "message": str(exc)}


def _writer_value(tuning: GP3MLModelTuning, selection: GP3MLModelSelection) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        paths = write_gazepoint_model_tuning(tuning, tmp, prefix="parity_tuning", selection=selection, overwrite=False)
        tables = {name: _frame(pd.read_csv(path)) for name, path in paths.items()}
        return {
            "path_names": sorted(paths),
            "basenames": {name: Path(path).name for name, path in paths.items()},
            "tables": tables,
        }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python parity/run_python_tuning_objects.py <fixture.json> <output.json>")
    fixture_path, output_path = map(Path, sys.argv[1:])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    g = fixture["grid"]
    cases: dict[str, dict[str, Any]] = {}

    cases["create_gazepoint_tuning_grid::cartesian_grid"] = _capture(lambda: _grid_value(create_gazepoint_tuning_grid(engine=g["engine"], engine_grid=g["engine_grid"], preprocessor_grid=g["preprocessor_grid"], thresholds=g["thresholds"], complexity=g["complexity"], interpretability=g["interpretability"])))
    cases["create_gazepoint_tuning_grid::custom_labels"] = _capture(lambda: _grid_value(create_gazepoint_tuning_grid(engine="glm", thresholds=[0.4, 0.6], complexity=[1, 2], interpretability=["high", "medium"], labels=["A", "B"])))
    cases["create_gazepoint_tuning_grid::invalid_threshold"] = _capture(lambda: _grid_value(create_gazepoint_tuning_grid("glm", thresholds=[0, 0.5])))
    cases["create_gazepoint_tuning_grid::metadata_length"] = _capture(lambda: _grid_value(create_gazepoint_tuning_grid("glm", thresholds=[0.4, 0.6], complexity=[1, 2, 3])))
    cases["create_gazepoint_tuning_grid::empty_parameter"] = _capture(lambda: _grid_value(create_gazepoint_tuning_grid("glm", engine_grid={"alpha": []})))

    cases["compare_gazepoint_models::all_candidates"] = _capture(lambda: _frame(compare_gazepoint_models(_base_tuning())))
    cases["compare_gazepoint_models::metric_filter"] = _capture(lambda: _frame(compare_gazepoint_models(_base_tuning(), metrics=["brier"])))
    cases["compare_gazepoint_models::invalid_object"] = _capture(lambda: _frame(compare_gazepoint_models("not tuning")))

    cases["select_gazepoint_model::tie_breaker_selection"] = _capture(lambda: _selection_value(select_gazepoint_model(_base_tuning(), metric="roc_auc", direction="maximize", tie_breakers=["accuracy", "brier"], rationale="predeclared review")))
    cases["select_gazepoint_model::complexity_selection"] = _capture(lambda: _selection_value(select_gazepoint_model(_base_tuning(), metric="roc_auc", direction="maximize", tie_breakers=[], rationale="prefer simpler tied candidate")))
    cases["select_gazepoint_model::minimum_success_filter"] = _capture(lambda: _selection_value(select_gazepoint_model(_base_tuning(low_success_second=True), metric="roc_auc", direction="maximize", minimum_success_prop=0.8, rationale="require stable fold success")))
    cases["select_gazepoint_model::accuracy_rejected"] = _capture(lambda: _selection_value(select_gazepoint_model(_base_tuning(), metric="accuracy", direction="maximize", rationale="not allowed")))
    cases["select_gazepoint_model::invalid_direction"] = _capture(lambda: _selection_value(select_gazepoint_model(_base_tuning(), metric="roc_auc", direction="up", rationale="invalid direction")))
    cases["select_gazepoint_model::missing_rationale"] = _capture(lambda: _selection_value(select_gazepoint_model(_base_tuning(), metric="roc_auc", direction="maximize")))
    cases["select_gazepoint_model::no_eligible_metric"] = _capture(lambda: _selection_value(select_gazepoint_model(_base_tuning(), metric="pr_auc", direction="maximize", rationale="requested metric")))
    cases["select_gazepoint_model::unresolved_tie"] = _capture(lambda: _selection_value(select_gazepoint_model(_base_tuning(textual_complexity=True), metric="roc_auc", direction="maximize", tie_breakers=[], rationale="tie remains")))

    cases["validate_gazepoint_model_tuning::clean_validation"] = _capture(lambda: _validation_value(validate_gazepoint_model_tuning(_base_tuning())))

    def selected_validation():
        x = _base_tuning(); x.selection = GP3MLModelSelection(candidate_id="candidate_001")
        return _validation_value(validate_gazepoint_model_tuning(x))
    cases["validate_gazepoint_model_tuning::selection_review"] = _capture(selected_validation)

    def missing_result_validation():
        x = _base_tuning(); x.results = x.results[:-1]
        return _validation_value(validate_gazepoint_model_tuning(x))
    cases["validate_gazepoint_model_tuning::missing_result"] = _capture(missing_result_validation)

    def incomplete_comparison_validation():
        x = _base_tuning(); x.comparison = x.comparison.loc[x.comparison.candidate_id != "candidate_003"].reset_index(drop=True)
        return _validation_value(validate_gazepoint_model_tuning(x))
    cases["validate_gazepoint_model_tuning::incomplete_comparison"] = _capture(incomplete_comparison_validation)

    def target_mismatch_validation():
        x = _base_tuning(); x.folds_metadata = {"generalization_target": "new_stimuli"}
        return _validation_value(validate_gazepoint_model_tuning(x))
    cases["validate_gazepoint_model_tuning::target_mismatch"] = _capture(target_mismatch_validation)
    cases["validate_gazepoint_model_tuning::invalid_object"] = _capture(lambda: _validation_value(validate_gazepoint_model_tuning("not tuning")))

    def write_success():
        x = _base_tuning()
        selection = select_gazepoint_model(x, metric="roc_auc", direction="maximize", tie_breakers=["brier"], rationale="record reviewed candidate")
        return _writer_value(x, selection)
    cases["write_gazepoint_model_tuning::write_with_selection"] = _capture(write_success)

    def overwrite_refusal():
        x = _base_tuning()
        with tempfile.TemporaryDirectory() as tmp:
            write_gazepoint_model_tuning(x, tmp, prefix="parity_tuning", overwrite=False)
            return write_gazepoint_model_tuning(x, tmp, prefix="parity_tuning", overwrite=False)
    cases["write_gazepoint_model_tuning::overwrite_refusal"] = _capture(overwrite_refusal)
    cases["write_gazepoint_model_tuning::invalid_selection"] = _capture(lambda: write_gazepoint_model_tuning(_base_tuning(), tempfile.mkdtemp(), selection="not selection"))
    cases["write_gazepoint_model_tuning::invalid_object"] = _capture(lambda: write_gazepoint_model_tuning("not tuning", tempfile.mkdtemp()))

    output = {"runtime": "python", "r_reference_version": fixture["r_reference"]["version"], "cases": cases}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
