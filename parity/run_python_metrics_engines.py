from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd

import gp3mlpy


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
    return {
        "columns": [str(name) for name in frame.columns],
        "rows": [
            {str(name): _scalar(value) for name, value in row.items()}
            for _, row in frame.iterrows()
        ],
    }


def _capture(
    call: Callable[[], Any],
    normalize: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    try:
        value = call()
        return {
            "status": "success",
            "value": value if normalize is None else normalize(value),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _rng_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _make_tasks(
    classification: dict[str, Any],
    regression: dict[str, Any],
) -> tuple[pd.DataFrame, Any, pd.DataFrame, Any]:
    n = len(classification["truth"])
    class_data = pd.DataFrame(
        {
            "trial_id": [f"T{i:02d}" for i in range(1, n + 1)],
            "participant_id": [f"P{((i - 1) // 2) + 1:02d}" for i in range(1, n + 1)],
            "stimulus_id": [f"S{((i - 1) % 3) + 1:02d}" for i in range(1, n + 1)],
            "quality_status": pd.Categorical(
                classification["truth"], categories=["pass", "review"]
            ),
        }
    )
    class_task = gp3mlpy.declare_gazepoint_task(
        class_data,
        outcome="quality_status",
        purpose="Predict observed recording quality.",
        task_type="classification",
        unit_id="trial_id",
        participant_id="participant_id",
        stimulus_id="stimulus_id",
        generalization_target="new_participants",
        positive="review",
    )

    m = len(regression["truth"])
    regression_data = pd.DataFrame(
        {
            "trial_id": [f"R{i:02d}" for i in range(1, m + 1)],
            "observed_duration": regression["truth"],
        }
    )
    regression_task = gp3mlpy.declare_gazepoint_task(
        regression_data,
        outcome="observed_duration",
        purpose="Predict observed response duration.",
        task_type="regression",
        unit_id="trial_id",
        generalization_target="new_trials_known_participants",
    )
    return class_data, class_task, regression_data, regression_task


def _normalize_uncertainty(value: Any, *, global_rng_preserved: bool, reproducible: bool) -> dict[str, Any]:
    intervals = value.intervals.sort_values("metric", kind="stable").reset_index(drop=True)
    point = value.point.copy()
    draw_columns = [str(name) for name in value.draws.columns]
    metric_columns = [
        name
        for name in draw_columns
        if name not in {"n", "threshold"}
        and pd.api.types.is_numeric_dtype(value.draws[name])
    ]
    interval_rows = []
    ordered = True
    estimates_match_point = True
    for row in intervals.itertuples(index=False):
        lower = _scalar(row.lower)
        upper = _scalar(row.upper)
        estimate = _scalar(row.estimate)
        if lower is not None and upper is not None and lower > upper:
            ordered = False
        point_value = _scalar(point[str(row.metric)].iloc[0])
        if estimate is None or point_value is None:
            if estimate != point_value:
                estimates_match_point = False
        elif not math.isclose(float(estimate), float(point_value), abs_tol=1e-12, rel_tol=1e-12):
            estimates_match_point = False
        interval_rows.append(
            {
                "metric": str(row.metric),
                "estimate": estimate,
                "lower": lower,
                "upper": upper,
            }
        )
    return {
        "class": value.r_class,
        "bootstrap": int(value.bootstrap),
        "conf_level": float(value.conf_level),
        "seed": int(value.seed),
        "point": _frame(point),
        "intervals": {"columns": ["metric", "estimate", "lower", "upper"], "rows": interval_rows},
        "draw_columns": draw_columns,
        "draw_nrow": int(len(value.draws)),
        "metric_columns": metric_columns,
        "checks": {
            "global_rng_preserved": bool(global_rng_preserved),
            "reproducible_with_seed": bool(reproducible),
            "intervals_ordered": bool(ordered),
            "estimates_match_point": bool(estimates_match_point),
            "draw_count_matches_bootstrap": bool(len(value.draws) == int(value.bootstrap)),
        },
    }


def _bootstrap_case(
    task: Any,
    truth: Any,
    prediction: Any,
    probability: Any,
    *,
    bootstrap: int,
    conf_level: float,
    seed: int,
) -> dict[str, Any]:
    np.random.seed(20260831)
    before = np.random.get_state()
    first = gp3mlpy.bootstrap_gazepoint_metrics(
        task,
        truth=truth,
        prediction=prediction,
        probability=probability,
        bootstrap=bootstrap,
        conf_level=conf_level,
        seed=seed,
    )
    after = np.random.get_state()
    second = gp3mlpy.bootstrap_gazepoint_metrics(
        task,
        truth=truth,
        prediction=prediction,
        probability=probability,
        bootstrap=bootstrap,
        conf_level=conf_level,
        seed=seed,
    )
    reproducible = (
        first.point.equals(second.point)
        and first.intervals.equals(second.intervals)
        and first.draws.equals(second.draws)
    )
    return _normalize_uncertainty(
        first,
        global_rng_preserved=_rng_equal(before, after),
        reproducible=reproducible,
    )


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python parity/run_python_metrics_engines.py "
            "<fixture.json> <output.json>"
        )

    fixture_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    classification = fixture["classification"]
    regression = fixture["regression"]
    class_data, class_task, regression_data, regression_task = _make_tasks(
        classification, regression
    )

    probability = np.asarray(classification["probability"], dtype=float)
    prediction = pd.Categorical(
        classification["prediction"], categories=["pass", "review"]
    )
    truth = class_data["quality_status"]

    performance_cases = {
        "classification_primary": _capture(
            lambda: gp3mlpy.gazepoint_performance_metrics(
                class_task,
                truth=truth,
                prediction=prediction,
                probability=probability,
                threshold=float(classification["threshold"]),
            ),
            _frame,
        ),
        "classification_threshold_dispatch": _capture(
            lambda: gp3mlpy.gazepoint_performance_metrics(
                class_task,
                truth=truth,
                prediction=None,
                probability=probability,
                threshold=0.6,
            ),
            _frame,
        ),
        "regression_primary": _capture(
            lambda: gp3mlpy.gazepoint_performance_metrics(
                regression_task,
                truth=regression["truth"],
                prediction=regression["prediction"],
            ),
            _frame,
        ),
        "classification_probability_length_mismatch": _capture(
            lambda: gp3mlpy.gazepoint_performance_metrics(
                class_task,
                truth=truth,
                prediction=None,
                probability=probability[:-1],
            ),
            _frame,
        ),
    }

    bootstrap_cases = {
        "classification_bootstrap": _capture(
            lambda: _bootstrap_case(
                class_task,
                truth,
                prediction,
                probability,
                bootstrap=int(classification["bootstrap"]),
                conf_level=float(classification["conf_level"]),
                seed=int(classification["seed"]),
            )
        ),
        "regression_bootstrap": _capture(
            lambda: _bootstrap_case(
                regression_task,
                regression["truth"],
                regression["prediction"],
                None,
                bootstrap=int(regression["bootstrap"]),
                conf_level=float(regression["conf_level"]),
                seed=int(regression["seed"]),
            )
        ),
        "bootstrap_zero": _capture(
            lambda: gp3mlpy.bootstrap_gazepoint_metrics(
                class_task, truth, probability=probability, bootstrap=0
            )
        ),
        "truth_too_short": _capture(
            lambda: gp3mlpy.bootstrap_gazepoint_metrics(
                class_task,
                pd.Categorical(["pass"], categories=["pass", "review"]),
                probability=[0.2],
            )
        ),
        "probability_length_mismatch": _capture(
            lambda: gp3mlpy.bootstrap_gazepoint_metrics(
                class_task, truth, probability=probability[:-1]
            )
        ),
        "prediction_length_mismatch": _capture(
            lambda: gp3mlpy.bootstrap_gazepoint_metrics(
                regression_task,
                regression["truth"],
                prediction=regression["prediction"][:-1],
            )
        ),
    }

    engines = gp3mlpy.gp3ml_available_engines()
    engine_cases = {
        "runtime_registry": {
            "status": "success",
            "value": _frame(engines),
        }
    }

    result = {
        "schema_version": 1,
        "runtime": "python",
        "r_reference_version": fixture["r_reference"]["version"],
        "performance_cases": performance_cases,
        "bootstrap_cases": bootstrap_cases,
        "engine_cases": engine_cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
