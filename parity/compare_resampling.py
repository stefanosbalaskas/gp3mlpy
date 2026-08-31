from __future__ import annotations

import copy
import json
import math
import numbers
import sys
from pathlib import Path
from typing import Any


def _numeric(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _normalize_escaped_quotes(message: str) -> str:
    """Normalize literal R backslashes that only escape displayed double quotes."""
    previous = None
    while message != previous:
        previous = message
        message = message.replace('\\"', '"')
    return message


def _coverage_signature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare damaged coverage by count pattern, not RNG-dependent source identity."""
    counts: dict[tuple[Any, Any, Any, Any], int] = {}
    for row in rows:
        key = (
            row.get("repeat"),
            row.get("n_analysis"),
            row.get("n_assessment"),
            row.get("n_excluded"),
        )
        counts[key] = counts.get(key, 0) + 1
    result = [
        {
            "repeat": key[0],
            "n_analysis": key[1],
            "n_assessment": key[2],
            "n_excluded": key[3],
            "n_rows": count,
        }
        for key, count in counts.items()
    ]
    return sorted(
        result,
        key=lambda row: (
            str(row["repeat"]),
            str(row["n_analysis"]),
            str(row["n_assessment"]),
            str(row["n_excluded"]),
        ),
    )


def _normalize_case(group: str, case_id: str, value: Any) -> Any:
    normalized = copy.deepcopy(value)
    if group == "errors" and isinstance(normalized, dict):
        message = normalized.get("message")
        if isinstance(message, str):
            normalized["message"] = _normalize_escaped_quotes(message)
    if (
        group == "validations"
        and case_id == "validation_assignment_damage"
        and isinstance(normalized, dict)
    ):
        payload = normalized.get("value")
        if isinstance(payload, dict):
            coverage = payload.get("assessment_coverage")
            if isinstance(coverage, list):
                payload["assessment_coverage"] = _coverage_signature(coverage)
    return normalized


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


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: python parity/compare_resampling.py "
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
        ("successes", "create_gazepoint_group_folds"),
        ("errors", "create_gazepoint_group_folds"),
        ("audits", "audit_gazepoint_group_folds"),
        ("validations", "validate_gazepoint_group_folds"),
        ("writers", "write_gazepoint_group_folds_csv"),
    )
    cases: list[dict[str, Any]] = []
    for group, function_name in groups:
        r_cases = r_result.get(group, {})
        py_cases = py_result.get(group, {})
        for case_id in sorted(set(r_cases) | set(py_cases)):
            left = _normalize_case(group, case_id, r_cases.get(case_id))
            right = _normalize_case(group, case_id, py_cases.get(case_id))
            errors = _compare(
                left,
                right,
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

    failed = bool(provenance_errors) or any(
        case["status"] == "FAIL" for case in cases
    )
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
    print(f"resampling parity: {report['overall_status']}")
    for case in cases:
        print(f"  {case['status']}: {case['function']}::{case['case_id']}")
        for error in case["errors"]:
            print(f"    {error}")
    for error in provenance_errors:
        print(f"  provenance: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
