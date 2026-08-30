from __future__ import annotations

import json
import math
import numbers
import sys
from pathlib import Path
from typing import Any


def _numeric(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _compare(left: Any, right: Any, *, atol: float, rtol: float, path: str) -> list[str]:
    errors: list[str] = []
    if left is None or right is None:
        if left is not None or right is not None:
            errors.append(f"{path}: {left!r} != {right!r}")
        return errors
    if _numeric(left) and _numeric(right):
        if not math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol):
            errors.append(f"{path}: {left!r} != {right!r} within atol={atol} rtol={rtol}")
        return errors
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            missing_left = sorted(set(right) - set(left))
            missing_right = sorted(set(left) - set(right))
            if missing_left:
                errors.append(f"{path}: missing from R result: {missing_left}")
            if missing_right:
                errors.append(f"{path}: missing from Python result: {missing_right}")
        for key in sorted(set(left) & set(right)):
            errors.extend(_compare(left[key], right[key], atol=atol, rtol=rtol, path=f"{path}.{key}"))
        return errors
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            errors.append(f"{path}: list lengths differ ({len(left)} != {len(right)})")
            return errors
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            errors.extend(
                _compare(left_value, right_value, atol=atol, rtol=rtol, path=f"{path}[{index}]")
            )
        return errors
    if left != right:
        errors.append(f"{path}: {left!r} != {right!r}")
    return errors


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: python parity/compare_governance.py <fixture.json> <r-output.json> <python-output.json> <report.json>"
        )

    fixture_path, r_path, python_path, report_path = map(Path, sys.argv[1:])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    r_result = json.loads(r_path.read_text(encoding="utf-8"))
    python_result = json.loads(python_path.read_text(encoding="utf-8"))

    tolerance = fixture["numeric_tolerance"]
    atol = float(tolerance["absolute"])
    rtol = float(tolerance["relative"])
    expected_r_version = fixture["r_reference"]["version"]

    provenance_errors: list[str] = []
    if r_result.get("package_version") != expected_r_version:
        provenance_errors.append(
            f"R package version {r_result.get('package_version')!r} != frozen {expected_r_version!r}"
        )
    if python_result.get("r_reference_version") != expected_r_version:
        provenance_errors.append(
            "Python r_reference_version "
            f"{python_result.get('r_reference_version')!r} != frozen {expected_r_version!r}"
        )

    groups = (
        ("task_declarations", "declare_gazepoint_task"),
        ("use_case_assertions", "assert_gp3ml_use_case"),
        ("role_validations", "validate_gazepoint_ml_roles"),
        ("feature_manifest_create", "create_gazepoint_feature_manifest"),
        ("feature_manifest_validate", "validate_gazepoint_feature_manifest"),
    )

    cases: list[dict[str, Any]] = []
    for group, function_name in groups:
        r_cases = r_result.get(group, {})
        python_cases = python_result.get(group, {})
        for case_id in sorted(set(r_cases) | set(python_cases)):
            errors = _compare(
                r_cases.get(case_id),
                python_cases.get(case_id),
                atol=atol,
                rtol=rtol,
                path=f"{function_name}.{case_id}",
            )
            cases.append(
                {
                    "function": function_name,
                    "case_id": case_id,
                    "status": "PASS" if not errors else "FAIL",
                    "errors": errors,
                }
            )

    failed = provenance_errors or any(case["status"] != "PASS" for case in cases)
    report = {
        "schema_version": 1,
        "overall_status": "FAIL" if failed else "PASS",
        "r_reference": fixture["r_reference"],
        "numeric_tolerance": fixture["numeric_tolerance"],
        "r_runtime": {
            "package": r_result.get("package"),
            "package_version": r_result.get("package_version"),
        },
        "python_runtime": {
            "package": python_result.get("package"),
            "package_version": python_result.get("package_version"),
            "r_reference_version": python_result.get("r_reference_version"),
        },
        "provenance_errors": provenance_errors,
        "cases": cases,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"governance parity: {report['overall_status']}")
    for case in cases:
        print(f"  {case['status']}: {case['function']}::{case['case_id']}")
        for error in case["errors"]:
            print(f"    {error}")
    for error in provenance_errors:
        print(f"  provenance: {error}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
