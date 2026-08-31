from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any


ERROR_KEYWORDS = {
    ("integrate_black_box_model", "invalid_functions"): ["must be functions"],
    ("integrate_black_box_model", "missing_safety"): ["safety declarations"],
    ("fit_gazepoint_model", "classification_lm_rejected"): ["classification engine"],
    ("fit_gazepoint_model", "forbidden_predictor"): ["cannot be predictors"],
    ("fit_gazepoint_model", "missing_outcome"): ["may not be missing"],
    ("fit_gazepoint_model", "unknown_engine"): ["unknown engine"],
    ("train_gazepoint_classifier", "regression_rejected"): ["classification task"],
    ("evaluate_gazepoint_group_folds", "invalid_folds"): ["gazepoint_group_folds"],
    ("evaluate_gazepoint_group_folds", "undeclared_predictor"): ["declared"],
    ("evaluate_gazepoint_group_folds", "target_mismatch"): ["generalization target"],
    ("evaluate_gazepoint_group_folds", "stop_on_failure"): ["fold", "failed"],
    ("tune_gazepoint_model", "invalid_grid"): ["gp3ml_tuning_grid"],
    ("tune_gazepoint_model", "stop_on_candidate_failure"): ["candidate", "failed"],
}


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare(left: Any, right: Any, *, atol: float, rtol: float, path: str) -> list[str]:
    errors: list[str] = []
    if left is None or right is None:
        if left is not None or right is not None:
            errors.append(f"{path}: {left!r} != {right!r}")
        return errors
    if _numeric(left) and _numeric(right):
        if not math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol):
            errors.append(f"{path}: {left!r} != {right!r}")
        return errors
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            only_r = sorted(set(left) - set(right))
            only_py = sorted(set(right) - set(left))
            if only_r:
                errors.append(f"{path}: keys only in R: {only_r}")
            if only_py:
                errors.append(f"{path}: keys only in Python: {only_py}")
        for key in sorted(set(left) & set(right)):
            errors.extend(_compare(left[key], right[key], atol=atol, rtol=rtol, path=f"{path}.{key}"))
        return errors
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            errors.append(f"{path}: list lengths differ ({len(left)} != {len(right)})")
            return errors
        for index, (lv, rv) in enumerate(zip(left, right, strict=True)):
            errors.extend(_compare(lv, rv, atol=atol, rtol=rtol, path=f"{path}[{index}]"))
        return errors
    if left != right:
        errors.append(f"{path}: {left!r} != {right!r}")
    return errors


def _error_contract(function: str, case_id: str, r_case: dict[str, Any], py_case: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    if r_case.get("status") != "error":
        errors.append(f"{path}: R status is {r_case.get('status')!r}, expected error")
    if py_case.get("status") != "error":
        errors.append(f"{path}: Python status is {py_case.get('status')!r}, expected error")
    for runtime, case in (("R", r_case), ("Python", py_case)):
        message = str(case.get("message", "")).lower()
        for keyword in ERROR_KEYWORDS.get((function, case_id), []):
            if keyword.lower() not in message:
                errors.append(f"{path}: {runtime} error lacks token {keyword!r}: {message!r}")
    return errors


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit("Usage: python parity/compare_model_execution.py <fixture.json> <case-registry.csv> <r.json> <python.json> <report.json>")
    fixture_path, registry_path, r_path, py_path, report_path = map(Path, sys.argv[1:])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    with registry_path.open(newline="", encoding="utf-8") as handle:
        registry = list(csv.DictReader(handle))
    r_result = json.loads(r_path.read_text(encoding="utf-8"))
    py_result = json.loads(py_path.read_text(encoding="utf-8"))
    atol = float(fixture["numeric_tolerance"]["absolute"])
    rtol = float(fixture["numeric_tolerance"]["relative"])
    expected_version = fixture["r_reference"]["version"]

    provenance_errors: list[str] = []
    if r_result.get("package_version") != expected_version:
        provenance_errors.append(f"R package version {r_result.get('package_version')!r} != frozen {expected_version!r}")
    if py_result.get("r_reference_version") != expected_version:
        provenance_errors.append(f"Python evidence reference {py_result.get('r_reference_version')!r} != frozen {expected_version!r}")

    r_cases = r_result.get("cases", {})
    py_cases = py_result.get("cases", {})
    report_cases: list[dict[str, Any]] = []
    failures = list(provenance_errors)

    for row in registry:
        function = row["function"]
        case_id = row["case_id"]
        comparison = row["comparison"]
        key = f"{function}::{case_id}"
        r_case = r_cases.get(key)
        py_case = py_cases.get(key)
        errors: list[str] = []
        if not isinstance(r_case, dict):
            errors.append(f"{key}: missing R evidence")
        if not isinstance(py_case, dict):
            errors.append(f"{key}: missing Python evidence")
        if not errors:
            if comparison == "error_contract":
                errors.extend(_error_contract(function, case_id, r_case, py_case, key))
            else:
                if r_case.get("status") != "success" or py_case.get("status") != "success":
                    errors.append(f"{key}: expected success; R={r_case.get('status')!r}, Python={py_case.get('status')!r}")
                else:
                    errors.extend(_compare(r_case.get("value"), py_case.get("value"), atol=atol, rtol=rtol, path=key))
        status = "PASS" if not errors else "FAIL"
        report_cases.append({"function": function, "case_id": case_id, "comparison": comparison, "status": status, "errors": errors})
        failures.extend(errors)

    registered = {f"{row['function']}::{row['case_id']}" for row in registry}
    extra_r = sorted(set(r_cases) - registered)
    extra_py = sorted(set(py_cases) - registered)
    if extra_r:
        failures.append(f"Unregistered R cases: {extra_r}")
    if extra_py:
        failures.append(f"Unregistered Python cases: {extra_py}")

    report = {
        "schema_version": 1,
        "r_reference_version": expected_version,
        "overall_status": "PASS" if not failures else "FAIL",
        "provenance_errors": provenance_errors,
        "cases": report_cases,
        "errors": failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    counts: dict[str, int] = {}
    for case in report_cases:
        counts[case["status"]] = counts.get(case["status"], 0) + 1
    print("model-execution parity: " + ", ".join(f"{name}={counts[name]}" for name in sorted(counts)))
    if failures:
        for error in failures[:80]:
            print("FAIL:", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
