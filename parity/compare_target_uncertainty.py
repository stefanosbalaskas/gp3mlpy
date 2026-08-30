from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
import sys
from typing import Any


STATUS_ONLY = {
    ("bootstrap_gazepoint_metrics_by_unit", "invalid_unit"),
    ("summarize_gazepoint_resample_uncertainty", "invalid_unit"),
    ("summarize_gazepoint_resample_uncertainty", "repeat_missing_column"),
    ("write_gazepoint_target_uncertainty", "overwrite_refusal"),
}

EXPECTED_DIFFERENCES = {
    ("summarize_gazepoint_resample_uncertainty", "repeat_summary"): (
        "Frozen R 0.3.0 fails when repeat-level uncertainty evaluates its reserved "
        "`repeat` column in the aggregate formula, while gp3mlpy returns the intended "
        "repeat-level uncertainty summary. The Python behavior is retained rather than "
        "porting the frozen R reference defect."
    )
}

SECTION_BY_FUNCTION = {
    "bootstrap_gazepoint_metrics_by_unit": "bootstrap_cases",
    "summarize_gazepoint_resample_uncertainty": "summarize_cases",
    "validate_gazepoint_target_uncertainty": "validation_cases",
    "write_gazepoint_target_uncertainty": "writer_cases",
}

_EMPTY_FAILURE_COLUMN_SHAPES = {(), ("replicate", "error")}
_EMPTY_FAILURE_COLUMN_SUFFIXES = (
    "failure_columns",
    "failures.columns",
    "failures.table.columns",
)


def _numeric(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _empty_failure_columns_equivalent(left: Any, right: Any, path: str) -> bool:
    if not path.endswith(_EMPTY_FAILURE_COLUMN_SUFFIXES):
        return False
    if not isinstance(left, list) or not isinstance(right, list):
        return False
    return tuple(left) in _EMPTY_FAILURE_COLUMN_SHAPES and tuple(right) in _EMPTY_FAILURE_COLUMN_SHAPES


def _compare(left: Any, right: Any, path: str, *, tol: float = 1e-12) -> list[str]:
    errors: list[str] = []
    if _empty_failure_columns_equivalent(left, right, path):
        return errors
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            only_left = sorted(set(left) - set(right))
            only_right = sorted(set(right) - set(left))
            if only_left:
                errors.append(f"{path}: keys only in R result: {only_left}")
            if only_right:
                errors.append(f"{path}: keys only in Python result: {only_right}")
        for key in sorted(set(left) & set(right)):
            errors.extend(_compare(left[key], right[key], f"{path}.{key}", tol=tol))
        return errors
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            errors.append(f"{path}: list lengths differ ({len(left)} != {len(right)})")
            return errors
        for index, (lv, rv) in enumerate(zip(left, right, strict=True)):
            errors.extend(_compare(lv, rv, f"{path}[{index}]", tol=tol))
        return errors
    if _numeric(left) and _numeric(right):
        lv = float(left)
        rv = float(right)
        if not math.isclose(lv, rv, rel_tol=tol, abs_tol=tol):
            errors.append(f"{path}: {left!r} != {right!r} within tolerance {tol}")
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


def _expected_difference(
    function_name: str,
    case_id: str,
    r_case: Any,
    py_case: Any,
    path: str,
) -> tuple[str, list[str], str]:
    reason = EXPECTED_DIFFERENCES[(function_name, case_id)]
    errors: list[str] = []
    r_status = r_case.get("status") if isinstance(r_case, dict) else None
    py_status = py_case.get("status") if isinstance(py_case, dict) else None
    if r_status != "error":
        errors.append(
            f"{path}: expected frozen R repeat-summary defect, got status={r_status!r}"
        )
    else:
        message = str(r_case.get("message", ""))
        if "invalid argument to unary operator" not in message:
            errors.append(
                f"{path}: frozen R failed for an unexpected reason: {message!r}"
            )
    if py_status != "success":
        errors.append(
            f"{path}: expected gp3mlpy repeat summary to succeed, got status={py_status!r}"
        )
    else:
        value = py_case.get("value")
        if not isinstance(value, dict):
            errors.append(f"{path}: Python success value is not an object")
        else:
            if value.get("class") != "gp3ml_resample_uncertainty":
                errors.append(
                    f"{path}: Python returned unexpected class {value.get('class')!r}"
                )
            if value.get("unit") != "repeat":
                errors.append(
                    f"{path}: Python returned unexpected unit {value.get('unit')!r}"
                )
    return ("FAIL" if errors else "EXPECTED-DIFFERENCE", errors, reason)


def _checks_true(value: Any, runtime: str, path: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path}: {runtime} success value is not an object"]
    checks = value.get("checks")
    if not isinstance(checks, dict):
        return [f"{path}: {runtime} success value has no checks object"]
    return [
        f"{path}.checks.{name}: {runtime} invariant is not true ({flag!r})"
        for name, flag in checks.items()
        if flag is not True
    ]


def _rows_by_metric(frame: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(frame, dict):
        return {}
    rows = frame.get("rows")
    if not isinstance(rows, list):
        return {}
    return {
        str(row["metric"]): row
        for row in rows
        if isinstance(row, dict) and row.get("metric") is not None
    }


def _compare_regression_bootstrap(
    r_case: Any,
    py_case: Any,
    fixture: dict[str, Any],
    path: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(r_case, dict) or not isinstance(py_case, dict):
        return [f"{path}: missing runtime evidence"]
    if r_case.get("status") != "success" or py_case.get("status") != "success":
        return _compare(r_case, py_case, path)

    r_value = r_case.get("value")
    py_value = py_case.get("value")
    errors.extend(_checks_true(r_value, "R", path))
    errors.extend(_checks_true(py_value, "Python", path))

    stochastic_keys = {"intervals", "draw_summary"}
    r_deterministic = {key: value for key, value in r_value.items() if key not in stochastic_keys}
    py_deterministic = {key: value for key, value in py_value.items() if key not in stochastic_keys}
    errors.extend(_compare(r_deterministic, py_deterministic, f"{path}.deterministic"))

    r_intervals = _rows_by_metric(r_value.get("intervals"))
    py_intervals = _rows_by_metric(py_value.get("intervals"))
    if set(r_intervals) != set(py_intervals):
        errors.append(
            f"{path}.intervals: metric sets differ "
            f"({sorted(r_intervals)} != {sorted(py_intervals)})"
        )
    tolerances = fixture["regression_interval_tolerance"]
    for metric in sorted(set(r_intervals) & set(py_intervals)):
        r_row = r_intervals[metric]
        py_row = py_intervals[metric]
        for key in ("metric", "estimate", "successful_replicates"):
            errors.extend(_compare(r_row.get(key), py_row.get(key), f"{path}.intervals.{metric}.{key}"))
        tolerance = float(tolerances[metric])
        for key in ("lower", "upper"):
            rv = r_row.get(key)
            pv = py_row.get(key)
            if not _numeric(rv) or not _numeric(pv):
                errors.append(f"{path}.intervals.{metric}.{key}: non-numeric interval endpoint")
            elif abs(float(rv) - float(pv)) > tolerance:
                errors.append(
                    f"{path}.intervals.{metric}.{key}: R={float(rv):.12g}, "
                    f"Python={float(pv):.12g}, abs diff>{tolerance}"
                )
        if _numeric(r_row.get("lower")) and _numeric(r_row.get("upper")) and float(r_row["lower"]) > float(r_row["upper"]):
            errors.append(f"{path}.intervals.{metric}: R lower exceeds upper")
        if _numeric(py_row.get("lower")) and _numeric(py_row.get("upper")) and float(py_row["lower"]) > float(py_row["upper"]):
            errors.append(f"{path}.intervals.{metric}: Python lower exceeds upper")

    r_draws = {row["metric"]: row for row in r_value.get("draw_summary", [])}
    py_draws = {row["metric"]: row for row in py_value.get("draw_summary", [])}
    if set(r_draws) != set(py_draws):
        errors.append(
            f"{path}.draw_summary: metric sets differ "
            f"({sorted(r_draws)} != {sorted(py_draws)})"
        )
    for metric in sorted(set(r_draws) & set(py_draws)):
        errors.extend(
            _compare(
                r_draws[metric].get("n"),
                py_draws[metric].get("n"),
                f"{path}.draw_summary.{metric}.n",
            )
        )
        for runtime, row in (("R", r_draws[metric]), ("Python", py_draws[metric])):
            minimum = row.get("min")
            maximum = row.get("max")
            if _numeric(minimum) and _numeric(maximum) and float(minimum) > float(maximum):
                errors.append(f"{path}.draw_summary.{metric}: {runtime} min exceeds max")
    return errors


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: python parity/compare_target_uncertainty.py "
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
    for function_name, section in SECTION_BY_FUNCTION.items():
        r_cases = r_result.get(section, {})
        py_cases = py_result.get(section, {})
        for case_id in sorted(set(r_cases) | set(py_cases)):
            r_case = r_cases.get(case_id)
            py_case = py_cases.get(case_id)
            path = f"{function_name}.{case_id}"
            reason = ""
            if (function_name, case_id) in EXPECTED_DIFFERENCES:
                status, errors, reason = _expected_difference(
                    function_name, case_id, r_case, py_case, path
                )
            elif (function_name, case_id) in STATUS_ONLY:
                errors = _status_only(r_case, py_case, path)
                status = "PASS" if not errors else "FAIL"
            elif function_name == "bootstrap_gazepoint_metrics_by_unit" and case_id == "regression_observation":
                errors = _compare_regression_bootstrap(r_case, py_case, fixture, path)
                status = "PASS" if not errors else "FAIL"
            else:
                errors = _compare(r_case, py_case, path)
                if (
                    function_name == "bootstrap_gazepoint_metrics_by_unit"
                    and isinstance(r_case, dict)
                    and isinstance(py_case, dict)
                    and r_case.get("status") == "success"
                    and py_case.get("status") == "success"
                ):
                    errors.extend(_checks_true(r_case.get("value"), "R", path))
                    errors.extend(_checks_true(py_case.get("value"), "Python", path))
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
        "regression_interval_tolerance": fixture["regression_interval_tolerance"],
        "provenance_errors": provenance_errors,
        "cases": cases,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"target uncertainty parity: {report['overall_status']}")
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
