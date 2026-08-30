from __future__ import annotations

import json
import math
import numbers
import sys
from pathlib import Path
from typing import Any


STATUS_ONLY = {
    "fit_invalid_method_error",
    "fit_length_mismatch_error",
    "apply_invalid_calibrator_error",
}

EXPECTED_DIFFERENCES = {
    "assess_length_mismatch_error": (
        "Frozen R 0.3.0 recycles unequal truth/probability vectors with warnings, "
        "while gp3mlpy rejects unequal lengths to prevent silent statistical recycling."
    )
}


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
            errors.append(
                f"{path}: {left!r} != {right!r} within atol={atol} rtol={rtol}"
            )
        return errors
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            missing_r = sorted(set(right) - set(left))
            missing_py = sorted(set(left) - set(right))
            if missing_r:
                errors.append(f"{path}: missing from R result: {missing_r}")
            if missing_py:
                errors.append(f"{path}: missing from Python result: {missing_py}")
        for key in sorted(set(left) & set(right)):
            errors.extend(
                _compare(
                    left[key],
                    right[key],
                    atol=atol,
                    rtol=rtol,
                    path=f"{path}.{key}",
                )
            )
        return errors
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            errors.append(f"{path}: list lengths differ ({len(left)} != {len(right)})")
            return errors
        for index, (lv, rv) in enumerate(zip(left, right, strict=True)):
            errors.extend(
                _compare(lv, rv, atol=atol, rtol=rtol, path=f"{path}[{index}]")
            )
        return errors
    if left != right:
        errors.append(f"{path}: {left!r} != {right!r}")
    return errors


def _status_only_errors(case_id: str, r_value: Any, py_value: Any, path: str) -> list[str]:
    errors: list[str] = []
    r_status = r_value.get("status") if isinstance(r_value, dict) else None
    py_status = py_value.get("status") if isinstance(py_value, dict) else None
    if r_status != "error":
        errors.append(f"{path}: R did not reject the invalid input (status={r_status!r})")
    if py_status != "error":
        errors.append(f"{path}: Python did not reject the invalid input (status={py_status!r})")
    return errors


def _expected_difference(
    case_id: str,
    r_value: Any,
    py_value: Any,
    path: str,
) -> tuple[str, list[str], str]:
    reason = EXPECTED_DIFFERENCES[case_id]
    r_status = r_value.get("status") if isinstance(r_value, dict) else None
    py_status = py_value.get("status") if isinstance(py_value, dict) else None
    errors: list[str] = []
    if r_status != "success":
        errors.append(
            f"{path}: expected frozen R to preserve recycling behavior, got status={r_status!r}"
        )
    if py_status != "error":
        errors.append(
            f"{path}: expected gp3mlpy safety guard to reject unequal lengths, got status={py_status!r}"
        )
    return ("FAIL" if errors else "EXPECTED-DIFFERENCE", errors, reason)


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: python parity/compare_calibration.py "
            "<fixture.json> <r.json> <python.json> <report.json>"
        )

    fixture_path, r_path, py_path, report_path = map(Path, sys.argv[1:])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    r_result = json.loads(r_path.read_text(encoding="utf-8"))
    py_result = json.loads(py_path.read_text(encoding="utf-8"))
    atol = float(fixture["numeric_tolerance"]["absolute"])
    rtol = float(fixture["numeric_tolerance"]["relative"])
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

    groups = (
        ("fits", "fit_gazepoint_calibrator"),
        ("fit_errors", "fit_gazepoint_calibrator"),
        ("applications", "apply_gazepoint_calibrator"),
        ("apply_errors", "apply_gazepoint_calibrator"),
        ("assessments", "assess_gazepoint_calibration"),
        ("assessment_errors", "assess_gazepoint_calibration"),
    )
    cases: list[dict[str, Any]] = []
    for group, function_name in groups:
        r_cases = r_result.get(group, {})
        py_cases = py_result.get(group, {})
        for case_id in sorted(set(r_cases) | set(py_cases)):
            path = f"{function_name}.{case_id}"
            reason = ""
            if case_id in EXPECTED_DIFFERENCES:
                status, errors, reason = _expected_difference(
                    case_id, r_cases.get(case_id), py_cases.get(case_id), path
                )
            elif case_id in STATUS_ONLY:
                errors = _status_only_errors(
                    case_id, r_cases.get(case_id), py_cases.get(case_id), path
                )
                status = "PASS" if not errors else "FAIL"
            else:
                errors = _compare(
                    r_cases.get(case_id),
                    py_cases.get(case_id),
                    atol=atol,
                    rtol=rtol,
                    path=path,
                )
                status = "PASS" if not errors else "FAIL"
            cases.append(
                {
                    "function": function_name,
                    "case_id": case_id,
                    "status": status,
                    "errors": errors,
                    "reason": reason,
                }
            )

    failed = bool(provenance_errors) or any(case["status"] == "FAIL" for case in cases)
    report = {
        "schema_version": 1,
        "overall_status": "FAIL" if failed else "PASS",
        "r_reference": fixture["r_reference"],
        "numeric_tolerance": fixture["numeric_tolerance"],
        "provenance_errors": provenance_errors,
        "cases": cases,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"calibration parity: {report['overall_status']}")
    for case in cases:
        print(f"  {case['status']}: {case['function']}::{case['case_id']}")
        if case["reason"]:
            print(f"    {case['reason']}")
        for error in case["errors"]:
            print(f"    {error}")
    for error in provenance_errors:
        print(f"  provenance: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
