from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import gp3mlpy


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
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        rows.append({str(name): _json_scalar(value) for name, value in row.items()})
    return rows


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


def _normalize_manifest_validation(validation: Any) -> dict[str, Any]:
    return {
        "class": validation.r_class,
        "status": validation.status,
        "n_features": int(validation.n_features),
        "summary": _normalize_frame(validation.summary),
        "checks": _normalize_frame(validation.checks),
        "issues": _normalize_frame(validation.issues),
    }


def _normalize_role_validation(validation: Any) -> dict[str, Any]:
    manifest_status = None
    if validation.manifest_validation is not None:
        manifest_status = validation.manifest_validation.status
    return {
        "class": validation.r_class,
        "status": validation.status,
        "checks": _normalize_frame(validation.checks),
        "issues": _normalize_frame(validation.issues),
        "manifest_validation_status": manifest_status,
    }


def _capture(call: Callable[[], Any], normalize: Callable[[Any], Any]) -> dict[str, Any]:
    try:
        return {"status": "success", "value": normalize(call())}
    except Exception as exc:  # parity evidence intentionally records public error contracts
        return {"status": "error", "message": str(exc)}


def _make_dataset(spec: dict[str, Any]) -> pd.DataFrame:
    kind = spec["kind"]
    n = int(spec["n"])
    if kind == "classification_governance":
        pass_n = int(spec["pass_n"])
        levels = [str(value) for value in spec["outcome_levels"]]
        outcome = [levels[0]] * pass_n + [levels[1]] * (n - pass_n)
        return pd.DataFrame(
            {
                "participant_id": [f"P{index:02d}" for index in range(1, n + 1)],
                "trial_id": [f"T{index:02d}" for index in range(1, n + 1)],
                "stimulus_id": [f"S{((index - 1) % 2) + 1:02d}" for index in range(1, n + 1)],
                "fixation_duration": np.arange(181, 181 + n, dtype=float),
                "pupil_change": np.arange(n, dtype=float) / 10.0,
                "quality_status": pd.Categorical(outcome, categories=levels),
            }
        )
    if kind == "regression_governance":
        return pd.DataFrame(
            {
                "participant_id": [f"R{index:02d}" for index in range(1, n + 1)],
                "trial_id": [f"RT{index:02d}" for index in range(1, n + 1)],
                "stimulus_id": [f"RS{((index - 1) % 2) + 1:02d}" for index in range(1, n + 1)],
                "fixation_duration": np.arange(203, 203 + 3 * n, 3, dtype=float),
                "score": 1.0 + np.arange(n, dtype=float) * 0.5,
            }
        )
    raise RuntimeError(f"Unknown parity dataset kind: {kind}")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python parity/run_python_governance.py <fixture.json> <output.json>")

    fixture_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    expected_reference = fixture["r_reference"]["version"]
    if gp3mlpy.r_reference_version != expected_reference:
        raise RuntimeError(
            f"gp3mlpy reference {gp3mlpy.r_reference_version!r} does not match fixture {expected_reference!r}."
        )

    datasets = {name: _make_dataset(spec) for name, spec in fixture["datasets"].items()}
    task_cases = {case["id"]: case for case in fixture["task_cases"]}
    manifest_cases = {case["id"]: case for case in fixture["manifest_cases"]}

    def make_task(case_id: str) -> Any:
        case = task_cases[case_id]
        return gp3mlpy.declare_gazepoint_task(data=datasets[case["dataset"]], **case["args"])

    def make_manifest(case_id: str) -> pd.DataFrame:
        case = manifest_cases[case_id]
        return gp3mlpy.create_gazepoint_feature_manifest(**case["args"])

    task_declarations: dict[str, dict[str, Any]] = {}
    for case in fixture["task_cases"]:
        task_declarations[case["id"]] = _capture(
            lambda case=case: gp3mlpy.declare_gazepoint_task(
                data=datasets[case["dataset"]],
                **case["args"],
            ),
            _normalize_task,
        )

    use_case_assertions: dict[str, dict[str, Any]] = {}
    for case in fixture["assert_cases"]:
        def run_assert(case: dict[str, Any] = case) -> bool:
            if case.get("plain_object"):
                return gp3mlpy.assert_gp3ml_use_case({})  # type: ignore[arg-type]
            task = make_task(case["task_case"])
            for name, value in case.get("mutations", {}).items():
                setattr(task, name, value)
            data = datasets[task_cases[case["task_case"]]["dataset"]]
            return gp3mlpy.assert_gp3ml_use_case(task, data)

        use_case_assertions[case["id"]] = _capture(run_assert, bool)

    feature_manifest_create: dict[str, dict[str, Any]] = {}
    feature_manifest_validate: dict[str, dict[str, Any]] = {}
    for case in fixture["manifest_cases"]:
        case_id = case["id"]
        operation = case["operation"]
        if operation in {"create", "create_validate"}:
            try:
                manifest = gp3mlpy.create_gazepoint_feature_manifest(**case["args"])
                feature_manifest_create[case_id] = {
                    "status": "success",
                    "value": _normalize_manifest(manifest),
                }
                if operation == "create_validate":
                    feature_manifest_validate[case_id] = _capture(
                        lambda manifest=manifest: gp3mlpy.validate_gazepoint_feature_manifest(manifest),
                        _normalize_manifest_validation,
                    )
            except Exception as exc:
                feature_manifest_create[case_id] = {"status": "error", "message": str(exc)}
        elif operation == "validate_raw":
            raw_manifest = pd.DataFrame(case["raw_manifest"])
            feature_manifest_validate[case_id] = _capture(
                lambda raw_manifest=raw_manifest: gp3mlpy.validate_gazepoint_feature_manifest(raw_manifest),
                _normalize_manifest_validation,
            )
        else:
            raise RuntimeError(f"Unknown manifest parity operation: {operation}")

    role_validations: dict[str, dict[str, Any]] = {}
    for case in fixture["role_cases"]:
        def run_role(case: dict[str, Any] = case) -> Any:
            manifest = None
            if case["manifest_case"] is not None:
                manifest = make_manifest(case["manifest_case"])
            return gp3mlpy.validate_gazepoint_ml_roles(
                data=datasets[case["dataset"]],
                task=make_task(case["task_case"]),
                predictors=case["predictors"],
                feature_manifest=manifest,
            )

        role_validations[case["id"]] = _capture(run_role, _normalize_role_validation)

    result = {
        "runtime": "Python",
        "package": "gp3mlpy",
        "package_version": gp3mlpy.__version__,
        "r_reference_version": gp3mlpy.r_reference_version,
        "task_declarations": task_declarations,
        "use_case_assertions": use_case_assertions,
        "role_validations": role_validations,
        "feature_manifest_create": feature_manifest_create,
        "feature_manifest_validate": feature_manifest_validate,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
