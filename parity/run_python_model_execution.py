from __future__ import annotations

import json
import math
from pathlib import Path
import sys
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
        return {"status": "success", "value": fn()}
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
            fixation_duration = 150.0 + ((participant_index * 11 + stimulus_index * 13) % 90)
            quality = "review" if ((participant_index + 2 * stimulus_index + (participant_index * stimulus_index) % 3) % 4) in {0, 1} else "pass"
            observed_duration = 1.0 + 0.6 * tracking_ratio + 0.04 * blink_rate + ((participant_index * stimulus_index) % 5) / 10.0
            rows.append(
                {
                    "participant_id": f"P{participant_index:02d}",
                    "trial_id": f"T{trial:03d}",
                    "stimulus_id": f"S{stimulus_index:02d}",
                    "tracking_ratio": tracking_ratio,
                    "blink_rate": blink_rate,
                    "fixation_duration": fixation_duration,
                    "quality_status": quality,
                    "observed_duration": observed_duration,
                }
            )
    out = pd.DataFrame(rows)
    out["quality_status"] = pd.Categorical(out["quality_status"], categories=["pass", "review"])
    return out


def _classification_task(data: pd.DataFrame, target: str = "new_participants"):
    return gp.create_gazepoint_synthetic_task(data, "recording_quality", target)


def _regression_task(data: pd.DataFrame, target: str = "new_participants"):
    return gp.create_gazepoint_synthetic_task(data, "observed_duration", target)


def _folds(data: pd.DataFrame, seed: int):
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


def _model_value(model: Any, data: pd.DataFrame) -> dict[str, Any]:
    prediction_type = "probability" if model.task.task_type == "classification" else "response"
    prediction = model.predict(data, type=prediction_type)
    classes = model.predict(data, type="class") if model.task.task_type == "classification" else None
    distribution = sorted((str(k), int(v)) for k, v in model.outcome_distribution.items())
    return {
        "class": model.r_class,
        "engine": model.engine,
        "task_type": model.task.task_type,
        "predictors": [str(x) for x in model.predictors],
        "threshold": float(model.threshold),
        "seed": int(model.seed),
        "training_n": int(model.training_n),
        "outcome_distribution": [[k, v] for k, v in distribution],
        "preprocessor_columns": [str(x) for x in model.preprocessor.columns],
        "prediction": [float(x) for x in np.asarray(prediction, dtype=float)],
        "classes": None if classes is None else [str(x) for x in classes.astype(str)],
    }


def _engine_value(engine: Any) -> dict[str, Any]:
    return {
        "class": engine.r_class,
        "name": str(engine.name),
        "supports": [str(x) for x in engine.supports],
        "probability": bool(engine.probability),
        "metadata": _jsonable(engine.metadata),
        "safety_declaration": _jsonable(engine.safety_declaration),
    }


def _evaluation_value(value: Any) -> dict[str, Any]:
    status_counts = value.fold_status["status"].value_counts().to_dict()
    metric_names = sorted(str(x) for x in pd.unique(value.metrics["metric"])) if len(value.metrics) else []
    fold_sizes = sorted(
        [
            [int(row.n_analysis), int(row.n_assessment), int(row.n_excluded)]
            for row in value.fold_status.itertuples(index=False)
        ]
    )
    models_retained = sum(1 for result in value.fold_results if result.get("model") is not None)
    return {
        "class": value.r_class,
        "engine": str(value.engine),
        "generalization_target": str(value.generalization_target),
        "predictors": [str(x) for x in value.predictors],
        "threshold": float(value.threshold),
        "seed": int(value.seed),
        "n_folds": int(len(value.fold_status)),
        "status_counts": {str(k): int(v) for k, v in sorted(status_counts.items())},
        "fold_sizes": fold_sizes,
        "n_predictions": int(len(value.predictions)),
        "n_metrics": int(len(value.metrics)),
        "metric_names": metric_names,
        "validation_status": str(value.validation.status),
        "keep_models": bool(value.keep_models),
        "models_retained": int(models_retained),
        "failed_folds": int((value.fold_status.status == "fail").sum()),
        "missing_predictions": int(value.fold_status.n_missing_predictions.sum()),
    }


def _tuning_value(value: Any) -> dict[str, Any]:
    statuses = []
    evaluation_retained = 0
    for result in value.results:
        statuses.append(
            {
                "candidate_id": str(result["candidate_id"]),
                "status": str(result["status"]),
                "success_prop": float(result["success_prop"]),
                "seed": int(result["seed"]),
                "has_error": result["error"] is not None,
            }
        )
        evaluation_retained += int(result.get("evaluation") is not None)
    metric_names = sorted(str(x) for x in pd.unique(value.comparison["metric"].dropna())) if len(value.comparison) else []
    return {
        "class": value.r_class,
        "candidate_ids": [str(x) for x in value.grid.candidates.candidate_id],
        "results": statuses,
        "comparison_rows": int(len(value.comparison)),
        "comparison_metrics": metric_names,
        "validation_status": str(value.validation.status),
        "metrics_requested": None if value.metrics_requested is None else [str(x) for x in value.metrics_requested],
        "keep_evaluations": bool(value.keep_evaluations),
        "evaluations_retained": int(evaluation_retained),
        "generalization_target": str(value.folds_metadata["generalization_target"]),
    }


def _safe_engine():
    def fit_fun(*, x, y, task, args):
        del x, task, args
        return {"mean": float(np.mean(y))}

    def predict_fun(*, fit, newdata, type, task, **kwargs):
        del type, task, kwargs
        return np.repeat(float(fit["mean"]), len(newdata))

    return gp.integrate_black_box_model(
        "constant_mean",
        fit_fun,
        predict_fun,
        supports=["classification"],
        probability=True,
        metadata={"backend": "fixture"},
        safety_declaration={
            "prohibited_uses_acknowledged": True,
            "prediction_time_inputs_only": True,
            "group_aware_evaluation_required": True,
        },
    )


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python parity/run_python_model_execution.py <fixture.json> <output.json>")
    fixture_path, output_path = map(Path, sys.argv[1:])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    seed = int(fixture["seed"])
    data = _data()
    cls_task = _classification_task(data)
    reg_task = _regression_task(data)
    folds = _folds(data, seed)
    cases: dict[str, dict[str, Any]] = {}

    cases["integrate_black_box_model::safe_constructor"] = _capture(lambda: _engine_value(_safe_engine()))
    cases["integrate_black_box_model::invalid_functions"] = _capture(
        lambda: gp.integrate_black_box_model(
            "bad",
            1,
            lambda **_: None,
            safety_declaration={
                "prohibited_uses_acknowledged": True,
                "prediction_time_inputs_only": True,
                "group_aware_evaluation_required": True,
            },
        )
    )
    cases["integrate_black_box_model::missing_safety"] = _capture(
        lambda: gp.integrate_black_box_model(
            "bad",
            lambda **_: None,
            lambda **_: None,
            safety_declaration={"prohibited_uses_acknowledged": True},
        )
    )

    cases["fit_gazepoint_model::classification_glm"] = _capture(
        lambda: _model_value(gp.fit_gazepoint_model(data, cls_task, PREDICTORS, engine="glm", seed=seed, threshold=0.45), data)
    )
    cases["fit_gazepoint_model::regression_glm_maps_lm"] = _capture(
        lambda: _model_value(gp.fit_gazepoint_model(data, reg_task, PREDICTORS, engine="glm", seed=seed), data)
    )
    cases["fit_gazepoint_model::custom_engine"] = _capture(
        lambda: _model_value(gp.fit_gazepoint_model(data, cls_task, PREDICTORS, engine=_safe_engine(), seed=seed), data)
    )
    cases["fit_gazepoint_model::classification_lm_rejected"] = _capture(
        lambda: gp.fit_gazepoint_model(data, cls_task, PREDICTORS, engine="lm")
    )
    cases["fit_gazepoint_model::forbidden_predictor"] = _capture(
        lambda: gp.fit_gazepoint_model(data, cls_task, ["quality_status"], engine="glm")
    )
    cases["fit_gazepoint_model::missing_outcome"] = _capture(
        lambda: gp.fit_gazepoint_model(data.assign(quality_status=data.quality_status.astype(object).where(data.index != 0, None)), cls_task, PREDICTORS, engine="glm")
    )
    cases["fit_gazepoint_model::unknown_engine"] = _capture(
        lambda: gp.fit_gazepoint_model(data, cls_task, PREDICTORS, engine="unknown")
    )

    cases["train_gazepoint_classifier::wrapper_success"] = _capture(
        lambda: _model_value(gp.train_gazepoint_classifier(data, cls_task, PREDICTORS, engine="glm", seed=seed), data)
    )
    cases["train_gazepoint_classifier::regression_rejected"] = _capture(
        lambda: gp.train_gazepoint_classifier(data, reg_task, PREDICTORS)
    )

    cases["evaluate_gazepoint_group_folds::successful_glm"] = _capture(
        lambda: _evaluation_value(
            gp.evaluate_gazepoint_group_folds(
                folds,
                cls_task,
                PREDICTORS,
                engine="glm",
                seed=seed,
                assess_calibration=False,
                keep_models=False,
            )
        )
    )
    cases["evaluate_gazepoint_group_folds::keep_models"] = _capture(
        lambda: _evaluation_value(
            gp.evaluate_gazepoint_group_folds(
                folds,
                cls_task,
                PREDICTORS,
                engine="glm",
                seed=seed,
                assess_calibration=False,
                keep_models=True,
            )
        )
    )
    cases["evaluate_gazepoint_group_folds::retain_fold_failures"] = _capture(
        lambda: _evaluation_value(
            gp.evaluate_gazepoint_group_folds(
                folds,
                cls_task,
                PREDICTORS,
                engine="unknown",
                seed=seed,
                continue_on_error=True,
            )
        )
    )
    cases["evaluate_gazepoint_group_folds::invalid_folds"] = _capture(
        lambda: gp.evaluate_gazepoint_group_folds("not folds", cls_task, PREDICTORS)
    )
    cases["evaluate_gazepoint_group_folds::undeclared_predictor"] = _capture(
        lambda: gp.evaluate_gazepoint_group_folds(folds, cls_task, ["fixation_duration"])
    )
    cases["evaluate_gazepoint_group_folds::target_mismatch"] = _capture(
        lambda: gp.evaluate_gazepoint_group_folds(folds, _classification_task(data, "new_stimuli"), PREDICTORS)
    )
    cases["evaluate_gazepoint_group_folds::stop_on_failure"] = _capture(
        lambda: gp.evaluate_gazepoint_group_folds(
            folds,
            cls_task,
            PREDICTORS,
            engine="unknown",
            seed=seed,
            continue_on_error=False,
        )
    )

    grid = gp.create_gazepoint_tuning_grid(
        "glm",
        thresholds=[0.4, 0.6],
        complexity=[1, 2],
        interpretability="high",
    )
    cases["tune_gazepoint_model::successful_candidates"] = _capture(
        lambda: _tuning_value(
            gp.tune_gazepoint_model(
                folds,
                cls_task,
                grid,
                predictors=PREDICTORS,
                metrics=["roc_auc", "brier"],
                seed=seed,
                keep_evaluations=False,
            )
        )
    )
    cases["tune_gazepoint_model::keep_evaluations"] = _capture(
        lambda: _tuning_value(
            gp.tune_gazepoint_model(
                folds,
                cls_task,
                gp.create_gazepoint_tuning_grid("glm", thresholds=0.5),
                predictors=PREDICTORS,
                metrics=["roc_auc"],
                seed=seed,
                keep_evaluations=True,
            )
        )
    )
    mixed_grid = gp.create_gazepoint_tuning_grid(["glm", "unknown"], thresholds=0.5, complexity=[1, 2])
    cases["tune_gazepoint_model::retain_candidate_failure"] = _capture(
        lambda: _tuning_value(
            gp.tune_gazepoint_model(
                folds,
                cls_task,
                mixed_grid,
                predictors=PREDICTORS,
                metrics=["roc_auc"],
                seed=seed,
                continue_on_error=True,
                keep_evaluations=False,
            )
        )
    )
    cases["tune_gazepoint_model::invalid_grid"] = _capture(
        lambda: gp.tune_gazepoint_model(folds, cls_task, "not grid", predictors=PREDICTORS)
    )
    cases["tune_gazepoint_model::stop_on_candidate_failure"] = _capture(
        lambda: gp.tune_gazepoint_model(
            folds,
            cls_task,
            mixed_grid,
            predictors=PREDICTORS,
            metrics=["roc_auc"],
            seed=seed,
            continue_on_error=False,
            keep_evaluations=False,
        )
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
