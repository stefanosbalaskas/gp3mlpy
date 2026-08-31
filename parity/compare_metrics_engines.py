from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
import sys
from typing import Any


PERFORMANCE_EXPECTED_DIFFERENCE = {
    "classification_probability_length_mismatch": (
        "Frozen R 0.3.0 recycles a shortened probability vector through binary metric "
        "operations with warnings, while gp3mlpy rejects row misalignment explicitly "
        "to prevent silent statistical recycling."
    )
}

EXACT_ERROR_CASES = {
    "bootstrap_zero",
    "truth_too_short",
    "probability_length_mismatch",
    "prediction_length_mismatch",
}


def _numeric(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _compare(
    left: Any,
    right: Any,
    *,
    atol: float,
    rtol: float,
    path: str,
) -> list[str]:
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
            only_r = sorted(set(left) - set(right))
            only_py = sorted(set(right) - set(left))
            if only_r:
                errors.append(f"{path}: keys only in R result: {only_r}")
            if only_py:
                errors.append(f"{path}: keys only in Python result: {only_py}")
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
                _compare(
                    lv,
                    rv,
                    atol=atol,
                    rtol=rtol,
                    path=f"{path}[{index}]",
                )
            )
        return errors
    if left != right:
        errors.append(f"{path}: {left!r} != {right!r}")
    return errors


def _expected_performance_difference(
    case_id: str,
    r_case: Any,
    py_case: Any,
    path: str,
) -> tuple[str, list[str], str]:
    reason = PERFORMANCE_EXPECTED_DIFFERENCE[case_id]
    errors: list[str] = []
    r_status = r_case.get("status") if isinstance(r_case, dict) else None
    py_status = py_case.get("status") if isinstance(py_case, dict) else None
    if r_status != "success":
        errors.append(
            f"{path}: expected frozen R recycling behavior, got status={r_status!r}"
        )
    if py_status != "error":
        errors.append(
            f"{path}: expected gp3mlpy to reject row misalignment, got status={py_status!r}"
        )
    elif "probability" not in str(py_case.get("message", "")).lower():
        errors.append(f"{path}: Python rejected input for an unexpected reason")
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


def _interval_rows(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    frame = value.get("intervals")
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


def _compare_bootstrap(
    r_case: Any,
    py_case: Any,
    *,
    atol: float,
    rtol: float,
    path: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(r_case, dict) or not isinstance(py_case, dict):
        return [f"{path}: missing runtime evidence"]
    if r_case.get("status") != "success" or py_case.get("status") != "success":
        return _compare(r_case, py_case, atol=atol, rtol=rtol, path=path)

    r_value = r_case.get("value")
    py_value = py_case.get("value")
    errors.extend(_checks_true(r_value, "R", path))
    errors.extend(_checks_true(py_value, "Python", path))
    if not isinstance(r_value, dict) or not isinstance(py_value, dict):
        return errors

    for key in ("class", "bootstrap", "conf_level", "seed", "point", "draw_columns", "draw_nrow", "metric_columns"):
        errors.extend(
            _compare(
                r_value.get(key),
                py_value.get(key),
                atol=atol,
                rtol=rtol,
                path=f"{path}.{key}",
            )
        )

    r_intervals = _interval_rows(r_value)
    py_intervals = _interval_rows(py_value)
    if set(r_intervals) != set(py_intervals):
        errors.append(
            f"{path}.intervals: metric sets differ "
            f"({sorted(r_intervals)} != {sorted(py_intervals)})"
        )
    for metric in sorted(set(r_intervals) & set(py_intervals)):
        for key in ("metric", "estimate"):
            errors.extend(
                _compare(
                    r_intervals[metric].get(key),
                    py_intervals[metric].get(key),
                    atol=atol,
                    rtol=rtol,
                    path=f"{path}.intervals.{metric}.{key}",
                )
            )
        for runtime, row in (("R", r_intervals[metric]), ("Python", py_intervals[metric])):
            lower = row.get("lower")
            upper = row.get("upper")
            if lower is not None and upper is not None and float(lower) > float(upper):
                errors.append(f"{path}.intervals.{metric}: {runtime} lower exceeds upper")
    return errors


def _engine_map(case: Any) -> tuple[list[str], dict[str, bool], list[str]]:
    errors: list[str] = []
    if not isinstance(case, dict) or case.get("status") != "success":
        return [], {}, ["engine registry did not succeed"]
    value = case.get("value")
    if not isinstance(value, dict):
        return [], {}, ["engine registry value is not a frame"]
    rows = value.get("rows")
    if not isinstance(rows, list):
        return [], {}, ["engine registry rows are missing"]
    names: list[str] = []
    availability: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, dict):
            errors.append("engine registry contains a non-object row")
            continue
        engine = str(row.get("engine"))
        available = row.get("available")
        names.append(engine)
        if not isinstance(available, bool):
            errors.append(f"engine {engine!r} has non-boolean availability {available!r}")
        else:
            availability[engine] = available
    return names, availability, errors


def _compare_engines(
    r_case: Any,
    py_case: Any,
    fixture: dict[str, Any],
    path: str,
) -> list[str]:
    expected = list(fixture["engines"])
    mandatory = list(fixture["mandatory_available"])
    r_names, r_available, errors = _engine_map(r_case)
    py_names, py_available, py_errors = _engine_map(py_case)
    errors.extend(py_errors)
    if r_names != expected:
        errors.append(f"{path}: R engine order {r_names!r} != expected {expected!r}")
    if py_names != expected:
        errors.append(f"{path}: Python engine order {py_names!r} != expected {expected!r}")
    for engine in mandatory:
        if r_available.get(engine) is not True:
            errors.append(f"{path}: mandatory R engine {engine!r} is not available")
        if py_available.get(engine) is not True:
            errors.append(f"{path}: mandatory Python engine {engine!r} is not available")
    if set(r_available) != set(expected):
        errors.append(f"{path}: R availability registry is incomplete")
    if set(py_available) != set(expected):
        errors.append(f"{path}: Python availability registry is incomplete")
    return errors


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: python parity/compare_metrics_engines.py "
            "<fixture.json> <r.json> <python.json> <report.json>"
        )

    fixture_path, r_path, py_path, report_path = map(Path, sys.argv[1:])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    r_result = json.loads(r_path.read_text(encoding="utf-8"))
    py_result = json.loads(py_path.read_text(encoding="utf-8"))
    atol = float(fixture["numeric_tolerance"]["absolute"])
    rtol = float(fixture["numeric_tolerance"]["relative"])
    expected_version = fixture["r_reference"]["version"]

    provenance_errors: list[str] = []
    if r_result.get("package_version") != expected_version:
        provenance_errors.append(
            f"R package version {r_result.get('package_version')!r} != frozen {expected_version!r}"
        )
    if py_result.get("r_reference_version") != expected_version:
        provenance_errors.append(
            "Python r_reference_version "
            f"{py_result.get('r_reference_version')!r} != frozen {expected_version!r}"
        )

    cases: list[dict[str, Any]] = []

    r_performance = r_result.get("performance_cases", {})
    py_performance = py_result.get("performance_cases", {})
    for case_id in sorted(set(r_performance) | set(py_performance)):
        path = f"gazepoint_performance_metrics.{case_id}"
        reason = ""
        if case_id in PERFORMANCE_EXPECTED_DIFFERENCE:
            status, errors, reason = _expected_performance_difference(
                case_id,
                r_performance.get(case_id),
                py_performance.get(case_id),
                path,
            )
        else:
            errors = _compare(
                r_performance.get(case_id),
                py_performance.get(case_id),
                atol=atol,
                rtol=rtol,
                path=path,
            )
            status = "PASS" if not errors else "FAIL"
        cases.append(
            {
                "function": "gazepoint_performance_metrics",
                "case_id": case_id,
                "status": status,
                "errors": errors,
                "reason": reason,
            }
        )

    r_bootstrap = r_result.get("bootstrap_cases", {})
    py_bootstrap = py_result.get("bootstrap_cases", {})
    for case_id in sorted(set(r_bootstrap) | set(py_bootstrap)):
        path = f"bootstrap_gazepoint_metrics.{case_id}"
        if case_id in {"classification_bootstrap", "regression_bootstrap"}:
            errors = _compare_bootstrap(
                r_bootstrap.get(case_id),
                py_bootstrap.get(case_id),
                atol=atol,
                rtol=rtol,
                path=path,
            )
        elif case_id in EXACT_ERROR_CASES:
            errors = _compare(
                r_bootstrap.get(case_id),
                py_bootstrap.get(case_id),
                atol=atol,
                rtol=rtol,
                path=path,
            )
        else:
            errors = [f"{path}: unregistered bootstrap case"]
        cases.append(
            {
                "function": "bootstrap_gazepoint_metrics",
                "case_id": case_id,
                "status": "PASS" if not errors else "FAIL",
                "errors": errors,
                "reason": "",
            }
        )

    r_engines = r_result.get("engine_cases", {})
    py_engines = py_result.get("engine_cases", {})
    engine_case_id = "runtime_registry"
    engine_errors = _compare_engines(
        r_engines.get(engine_case_id),
        py_engines.get(engine_case_id),
        fixture,
        f"gp3ml_available_engines.{engine_case_id}",
    )
    cases.append(
        {
            "function": "gp3ml_available_engines",
            "case_id": engine_case_id,
            "status": "PASS" if not engine_errors else "FAIL",
            "errors": engine_errors,
            "reason": (
                "Optional package availability is runtime-native. Cross-runtime parity "
                "requires the frozen engine set/order and mandatory base/custom engines, "
                "not equality of R package installation and Python adapter installation."
            ),
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
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"metrics/engines parity: {report['overall_status']}")
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
