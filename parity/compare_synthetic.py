from __future__ import annotations

import json
import numbers
import sys
from pathlib import Path
from typing import Any


STATUS_ONLY = {
    "task_invalid_workflow",
    "task_invalid_generalization_target",
}
EXACT_ERROR_CASES = {
    "simulator_n_participants_error",
    "simulator_n_stimuli_error",
    "simulator_trials_per_cell_error",
}


def _numeric(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _compare_exact(left: Any, right: Any, path: str) -> list[str]:
    errors: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            only_left = sorted(set(left) - set(right))
            only_right = sorted(set(right) - set(left))
            if only_left:
                errors.append(f"{path}: keys only in R result: {only_left}")
            if only_right:
                errors.append(f"{path}: keys only in Python result: {only_right}")
        for key in sorted(set(left) & set(right)):
            errors.extend(_compare_exact(left[key], right[key], f"{path}.{key}"))
        return errors
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            errors.append(f"{path}: list lengths differ ({len(left)} != {len(right)})")
            return errors
        for index, (lv, rv) in enumerate(zip(left, right, strict=True)):
            errors.extend(_compare_exact(lv, rv, f"{path}[{index}]"))
        return errors
    if _numeric(left) and _numeric(right):
        if float(left) != float(right):
            errors.append(f"{path}: {left!r} != {right!r}")
        return errors
    if left != right:
        errors.append(f"{path}: {left!r} != {right!r}")
    return errors


def _status_only(r_value: Any, py_value: Any, path: str) -> list[str]:
    errors: list[str] = []
    r_status = r_value.get("status") if isinstance(r_value, dict) else None
    py_status = py_value.get("status") if isinstance(py_value, dict) else None
    if r_status != "error":
        errors.append(f"{path}: R did not reject invalid input (status={r_status!r})")
    if py_status != "error":
        errors.append(f"{path}: Python did not reject invalid input (status={py_status!r})")
    return errors


def _require_true_checks(value: Any, runtime: str, path: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{path}: {runtime} result is not an object"]
    checks = value.get("checks")
    if not isinstance(checks, dict):
        return [f"{path}: {runtime} result has no checks object"]
    for name, flag in checks.items():
        if flag is not True:
            errors.append(f"{path}.checks.{name}: {runtime} invariant is not true ({flag!r})")
    return errors


def _compare_simulation(
    r_case: Any,
    py_case: Any,
    *,
    fixture: dict[str, Any],
    case_id: str,
) -> list[str]:
    path = f"simulate_gazepoint_governed_data.{case_id}"
    errors: list[str] = []
    if not isinstance(r_case, dict) or not isinstance(py_case, dict):
        return [f"{path}: missing runtime evidence"]
    if r_case.get("status") != "success" or py_case.get("status") != "success":
        return _compare_exact(r_case, py_case, path)

    r_value = r_case.get("value")
    py_value = py_case.get("value")
    errors.extend(_require_true_checks(r_value, "R", path))
    errors.extend(_require_true_checks(py_value, "Python", path))
    errors.extend(
        _compare_exact(
            r_value.get("exact") if isinstance(r_value, dict) else None,
            py_value.get("exact") if isinstance(py_value, dict) else None,
            f"{path}.exact",
        )
    )

    if case_id == "simulator_primary":
        r_distribution = r_value.get("distribution", {})
        py_distribution = py_value.get("distribution", {})
        tolerances = fixture["distribution_tolerance"]
        ranges = fixture["distribution_ranges"]
        if set(r_distribution) != set(py_distribution):
            errors.append(
                f"{path}.distribution: metric keys differ "
                f"({sorted(r_distribution)} != {sorted(py_distribution)})"
            )
        for metric in sorted(set(r_distribution) & set(py_distribution)):
            r_number = float(r_distribution[metric])
            py_number = float(py_distribution[metric])
            tolerance = float(tolerances[metric])
            if abs(r_number - py_number) > tolerance:
                errors.append(
                    f"{path}.distribution.{metric}: R={r_number:.12g}, "
                    f"Python={py_number:.12g}, abs diff>{tolerance}"
                )
            lower, upper = map(float, ranges[metric])
            if not lower <= r_number <= upper:
                errors.append(
                    f"{path}.distribution.{metric}: R={r_number:.12g} "
                    f"outside [{lower}, {upper}]"
                )
            if not lower <= py_number <= upper:
                errors.append(
                    f"{path}.distribution.{metric}: Python={py_number:.12g} "
                    f"outside [{lower}, {upper}]"
                )
    return errors


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: python parity/compare_synthetic.py "
            "<fixture.json> <r.json> <python.json> <report.json>"
        )

    fixture_path, r_path, py_path, report_path = map(Path, sys.argv[1:])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    r_result = json.loads(r_path.read_text(encoding="utf-8"))
    py_result = json.loads(py_path.read_text(encoding="utf-8"))
    expected = fixture["r_reference"]["version"]

    provenance_errors: list[str] = []
    if r_result.get("package_version") != expected:
        provenance_errors.append(
            f"R package version {r_result.get('package_version')!r} != frozen {expected!r}"
        )
    if py_result.get("r_reference_version") != expected:
        provenance_errors.append(
            "Python r_reference_version "
            f"{py_result.get('r_reference_version')!r} != frozen {expected!r}"
        )

    cases: list[dict[str, Any]] = []

    for case_id in sorted(set(r_result.get("simulations", {})) | set(py_result.get("simulations", {}))):
        r_case = r_result.get("simulations", {}).get(case_id)
        py_case = py_result.get("simulations", {}).get(case_id)
        path = f"simulate_gazepoint_governed_data.{case_id}"
        if case_id in EXACT_ERROR_CASES:
            errors = _compare_exact(r_case, py_case, path)
        else:
            errors = _compare_simulation(
                r_case, py_case, fixture=fixture, case_id=case_id
            )
        cases.append(
            {
                "function": "simulate_gazepoint_governed_data",
                "case_id": case_id,
                "status": "PASS" if not errors else "FAIL",
                "errors": errors,
            }
        )

    for case_id in sorted(set(r_result.get("manifests", {})) | set(py_result.get("manifests", {}))):
        path = f"create_gazepoint_synthetic_manifest.{case_id}"
        errors = _compare_exact(
            r_result.get("manifests", {}).get(case_id),
            py_result.get("manifests", {}).get(case_id),
            path,
        )
        cases.append(
            {
                "function": "create_gazepoint_synthetic_manifest",
                "case_id": case_id,
                "status": "PASS" if not errors else "FAIL",
                "errors": errors,
            }
        )

    for case_id in sorted(set(r_result.get("tasks", {})) | set(py_result.get("tasks", {}))):
        path = f"create_gazepoint_synthetic_task.{case_id}"
        if case_id in STATUS_ONLY:
            errors = _status_only(
                r_result.get("tasks", {}).get(case_id),
                py_result.get("tasks", {}).get(case_id),
                path,
            )
        else:
            errors = _compare_exact(
                r_result.get("tasks", {}).get(case_id),
                py_result.get("tasks", {}).get(case_id),
                path,
            )
        cases.append(
            {
                "function": "create_gazepoint_synthetic_task",
                "case_id": case_id,
                "status": "PASS" if not errors else "FAIL",
                "errors": errors,
            }
        )

    failed = bool(provenance_errors) or any(case["status"] == "FAIL" for case in cases)
    report = {
        "schema_version": 1,
        "overall_status": "FAIL" if failed else "PASS",
        "r_reference": fixture["r_reference"],
        "distribution_tolerance": fixture["distribution_tolerance"],
        "provenance_errors": provenance_errors,
        "cases": cases,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"synthetic parity: {report['overall_status']}")
    for case in cases:
        print(f"  {case['status']}: {case['function']}::{case['case_id']}")
        for error in case["errors"]:
            print(f"    {error}")
    for error in provenance_errors:
        print(f"  provenance: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
