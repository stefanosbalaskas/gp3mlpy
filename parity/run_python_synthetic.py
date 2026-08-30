from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import gp3mlpy


RANDOM_NUMERIC = (
    "tracking_ratio",
    "blink_rate",
    "fixation_duration",
    "gaze_dispersion",
    "pupil_change",
    "observed_duration",
)

DESIGN_COLUMNS = (
    "participant_id",
    "trial_id",
    "stimulus_id",
    "replicate",
    "assigned_condition",
    "site_label",
)


def _json_scalar(value: Any) -> Any:
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


def _normalize_frame(data: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(name): _json_scalar(value) for name, value in row.items()}
        for _, row in data.iterrows()
    ]


def _normalize_task(task: Any) -> dict[str, Any]:
    keys = [
        "outcome",
        "purpose",
        "task_type",
        "unit_id",
        "participant_id",
        "stimulus_id",
        "generalization_target",
        "positive",
        "observed_outcome",
        "sensitive_outcome",
        "levels",
        "negative",
    ]
    components: dict[str, Any] = {}
    for key in keys:
        value = getattr(task, key, None)
        if isinstance(value, (list, tuple)):
            components[key] = [_json_scalar(item) for item in value]
        else:
            components[key] = _json_scalar(value)
    return {"class": task.r_class, "components": components}


def _normalize_manifest(manifest: pd.DataFrame) -> dict[str, Any]:
    return {
        "class": manifest.attrs.get("r_class"),
        "columns": [str(name) for name in manifest.columns],
        "rows": _normalize_frame(manifest),
    }


def _categorical_levels(series: pd.Series) -> list[str]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(value) for value in series.cat.categories]
    return sorted(str(value) for value in pd.unique(series.dropna()))


def _counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.astype(object).value_counts(dropna=False).sort_index().items()
    }


def _rng_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _simulate(args: dict[str, Any]) -> pd.DataFrame:
    return gp3mlpy.simulate_gazepoint_governed_data(**args)


def _simulation_evidence(args: dict[str, Any], *, include_distribution: bool) -> dict[str, Any]:
    np.random.seed(20260831)
    before = np.random.get_state()
    data = _simulate(args)
    after = np.random.get_state()

    repeated = _simulate(args)
    alternate_args = dict(args)
    alternate_args["seed"] = int(args["seed"]) + 1
    alternate = _simulate(alternate_args)

    participant_values = data["participant_id"].astype(str)
    stimulus_values = data["stimulus_id"].astype(str)
    trial_values = data["trial_id"].astype(str)
    replicate_values = sorted(int(value) for value in pd.unique(data["replicate"]))

    exact = {
        "columns": [str(name) for name in data.columns],
        "nrow": int(len(data)),
        "ncol": int(data.shape[1]),
        "n_participants": int(data["participant_id"].nunique()),
        "n_stimuli": int(data["stimulus_id"].nunique()),
        "n_trials": int(data["trial_id"].nunique()),
        "participant_first": participant_values.iloc[0],
        "participant_last": participant_values.iloc[-1],
        "stimulus_first": stimulus_values.iloc[0],
        "stimulus_last": stimulus_values.iloc[-1],
        "trial_first": trial_values.iloc[0],
        "trial_last": trial_values.iloc[-1],
        "replicate_values": replicate_values,
        "assigned_condition_levels": _categorical_levels(data["assigned_condition"]),
        "quality_status_levels": _categorical_levels(data["quality_status"]),
        "observed_response_levels": _categorical_levels(data["observed_response"]),
        "site_label_levels": _categorical_levels(data["site_label"]),
        "assigned_condition_counts": _counts(data["assigned_condition"]),
        "site_label_counts": _counts(data["site_label"]),
        "design_head": _normalize_frame(data.loc[:, DESIGN_COLUMNS].head(8)),
        "design_tail": _normalize_frame(data.loc[:, DESIGN_COLUMNS].tail(4)),
    }
    checks = {
        "same_seed_reproducible": bool(data.equals(repeated)),
        "different_seed_changes_random_columns": bool(
            not data.loc[:, RANDOM_NUMERIC].equals(alternate.loc[:, RANDOM_NUMERIC])
        ),
        "global_rng_preserved": _rng_state_equal(before, after),
        "tracking_ratio_bounds": bool(data["tracking_ratio"].between(0.45, 1.0).all()),
        "blink_rate_nonnegative": bool((data["blink_rate"] >= 0.0).all()),
        "fixation_duration_floor": bool((data["fixation_duration"] >= 80.0).all()),
        "gaze_dispersion_floor": bool((data["gaze_dispersion"] >= 0.05).all()),
        "observed_duration_floor": bool((data["observed_duration"] >= 0.1).all()),
    }
    result: dict[str, Any] = {"exact": exact, "checks": checks}
    if include_distribution:
        distribution: dict[str, float] = {}
        for name in RANDOM_NUMERIC:
            values = data[name].to_numpy(dtype=float)
            distribution[f"{name}_mean"] = float(np.mean(values))
            distribution[f"{name}_sd"] = float(np.std(values, ddof=1))
        distribution["quality_review_rate"] = float(
            np.mean(data["quality_status"].astype(object).to_numpy() == "review")
        )
        distribution["response_yes_rate"] = float(
            np.mean(data["observed_response"].astype(object).to_numpy() == "recorded_yes")
        )
        result["distribution"] = distribution
    return result


def _capture(call: Callable[[], Any], normalize: Callable[[Any], Any]) -> dict[str, Any]:
    try:
        return {"status": "success", "value": normalize(call())}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python parity/run_python_synthetic.py <fixture.json> <output.json>")

    fixture_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected = fixture["r_reference"]["version"]
    if gp3mlpy.r_reference_version != expected:
        raise RuntimeError(
            f"gp3mlpy reference {gp3mlpy.r_reference_version!r} does not match fixture {expected!r}."
        )

    primary_args = fixture["primary_simulation"]
    primary = _simulate(primary_args)

    simulations = {
        "simulator_primary": {
            "status": "success",
            "value": _simulation_evidence(primary_args, include_distribution=True),
        },
        "simulator_integer_coercion": {
            "status": "success",
            "value": _simulation_evidence(
                fixture["coercion_simulation"], include_distribution=False
            ),
        },
        "simulator_n_participants_error": _capture(
            lambda: gp3mlpy.simulate_gazepoint_governed_data(3, 2, 1, 1),
            lambda value: value,
        ),
        "simulator_n_stimuli_error": _capture(
            lambda: gp3mlpy.simulate_gazepoint_governed_data(4, 1, 1, 1),
            lambda value: value,
        ),
        "simulator_trials_per_cell_error": _capture(
            lambda: gp3mlpy.simulate_gazepoint_governed_data(4, 2, 0, 1),
            lambda value: value,
        ),
    }

    manifest_args = {
        "outcome": "quality_status",
        "predictors": ["tracking_ratio", "blink_rate", "gaze_dispersion"],
    }
    manifests = {
        "manifest_primary": _capture(
            lambda: gp3mlpy.create_gazepoint_synthetic_manifest(**manifest_args),
            _normalize_manifest,
        ),
        "manifest_custom_identifiers": _capture(
            lambda: gp3mlpy.create_gazepoint_synthetic_manifest(
                **manifest_args,
                participant_id="subject_key",
                stimulus_id="item_key",
                trial_id="event_key",
            ),
            _normalize_manifest,
        ),
    }

    task_cases = {
        "task_recording_quality": ("recording_quality", "new_participants"),
        "task_assigned_condition": ("assigned_condition", "new_stimuli"),
        "task_observed_behavior": ("observed_behavior", "new_trials_known_participants"),
        "task_observed_duration": (
            "observed_duration",
            "new_participants_and_new_stimuli",
        ),
    }
    tasks: dict[str, dict[str, Any]] = {}
    for case_id, (workflow, target) in task_cases.items():
        tasks[case_id] = _capture(
            lambda workflow=workflow, target=target: gp3mlpy.create_gazepoint_synthetic_task(
                primary,
                workflow=workflow,
                generalization_target=target,
            ),
            _normalize_task,
        )
    tasks["task_invalid_workflow"] = _capture(
        lambda: gp3mlpy.create_gazepoint_synthetic_task(
            primary, workflow="latent_personality"
        ),
        _normalize_task,
    )
    tasks["task_invalid_generalization_target"] = _capture(
        lambda: gp3mlpy.create_gazepoint_synthetic_task(
            primary,
            workflow="recording_quality",
            generalization_target="same_rows",
        ),
        _normalize_task,
    )

    result = {
        "runtime": "Python",
        "package": "gp3mlpy",
        "package_version": gp3mlpy.__version__,
        "r_reference_version": gp3mlpy.r_reference_version,
        "simulations": simulations,
        "manifests": manifests,
        "tasks": tasks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
