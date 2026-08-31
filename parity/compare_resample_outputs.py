from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
import sys
from typing import Any


SECTION_BY_FUNCTION = {
    "write_gazepoint_feature_manifest_csv": "feature_writer_cases",
    "collect_gazepoint_fold_predictions": "collector_cases",
    "summarize_gazepoint_resample_performance": "summary_cases",
    "validate_gazepoint_resample_evaluation": "validation_cases",
    "write_gazepoint_resample_evaluation": "writer_cases",
}

STATUS_ONLY = {
    ("write_gazepoint_feature_manifest_csv", "overwrite_refusal"),
    ("write_gazepoint_feature_manifest_csv", "invalid_extension"),
    ("write_gazepoint_feature_manifest_csv", "plain_checks_rejection"),
    ("collect_gazepoint_fold_predictions", "invalid_object"),
    ("summarize_gazepoint_resample_performance", "no_predictions"),
    ("summarize_gazepoint_resample_performance", "invalid_aggregation"),
    ("summarize_gazepoint_resample_performance", "invalid_object"),
    ("validate_gazepoint_resample_evaluation", "invalid_object"),
    ("write_gazepoint_resample_evaluation", "overwrite_refusal"),
    ("write_gazepoint_resample_evaluation", "invalid_object"),
}


def _numeric(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _compare(left: Any, right: Any, path: str, *, atol: float, rtol: float) -> list[str]:
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
            only_r = sorted(set(left) - set(right))
            only_py = sorted(set(right) - set(left))
            if only_r:
                errors.append(f"{path}: keys only in R result: {only_r}")
            if only_py:
                errors.append(f"{path}: keys only in Python result: {only_py}")
        for key in sorted(set(left) & set(right)):
            errors.extend(_compare(left[key], right[key], f"{path}.{key}", atol=atol, rtol=rtol))
        return errors
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            errors.append(f"{path}: list lengths differ ({len(left)} != {len(right)})")
            return errors
        for index, (lv, rv) in enumerate(zip(left, right, strict=True)):
            errors.extend(_compare(lv, rv, f"{path}[{index}]", atol=atol, rtol=rtol))
        return errors
    if left != right:
        errors.append(f"{path}: {left!r} != {right!r}")
    return errors


def _status_only(r_case: Any, py_case: Any, path: str) -> list[str]:
    errors: list[str] = []
    r_status = r_case.get("status") if isinstance(r_case, dict) else None
    py_status = py_case.get("status") if isinstance(py_case, dict) else None
    if r_status != "error":
        errors.append(f"{path}: R did not reject invalid input (status={r_status!r})")
    if py_status != "error":
        errors.append(f"{path}: Python did not reject invalid input (status={py_status!r})")
    return errors


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: python parity/compare_resample_outputs.py "
            "<fixture.json> <r.json> <python.json> <report.json>"
        )

    fixture_path, r_path, py_path, report_path = map(Path, sys.argv[1:])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    r_result = json.loads(r_path.read_text(encoding="utf-8"))
    py_result = json.loads(py_path.read_text(encoding="utf-8"))
    expected = fixture["r_reference"]["version"]
    atol = float(fixture["numeric_tolerance"]["absolute"])
    rtol = float(fixture["numeric_tolerance"]["relative"])

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
    for function_name, section in SECTION_BY_FUNCTION.items():
        r_cases = r_result.get(section, {})
        py_cases = py_result.get(section, {})
        for case_id in sorted(set(r_cases) | set(py_cases)):
            path = f"{function_name}.{case_id}"
            if (function_name, case_id) in STATUS_ONLY:
                errors = _status_only(r_cases.get(case_id), py_cases.get(case_id), path)
            else:
                errors = _compare(
                    r_cases.get(case_id), py_cases.get(case_id), path,
                    atol=atol, rtol=rtol,
                )
            cases.append(
                {
                    "function": function_name,
                    "case_id": case_id,
                    "status": "PASS" if not errors else "FAIL",
                    "errors": errors,
                    "reason": "",
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

    print(f"resample outputs parity: {report['overall_status']}")
    for case in cases:
        print(f"  {case['status']}: {case['function']}::{case['case_id']}")
        for error in case["errors"]:
            print(f"    {error}")
    for error in provenance_errors:
        print(f"  provenance: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
