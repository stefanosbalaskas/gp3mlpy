from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import gp3mlpy


def _scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return str(value)


def _frame(data: pd.DataFrame) -> list[dict[str, Any]]:
    if not isinstance(data, pd.DataFrame) or data.empty:
        return []
    return [
        {str(name): _scalar(value) for name, value in row.items()}
        for _, row in data.iterrows()
    ]


def _vector(values: Any) -> list[Any]:
    return [_scalar(value) for value in np.asarray(values, dtype=object).reshape(-1)]


def _capture(call: Callable[[], Any], normalize: Callable[[Any], Any]) -> dict[str, Any]:
    try:
        value = call()
        return {"status": "success", "value": normalize(value)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _normalize_calibrator(value: Any, probes: list[float]) -> dict[str, Any]:
    applied = gp3mlpy.apply_gazepoint_calibrator(value, probes)
    return {
        "class": getattr(value, "r_class", type(value).__name__),
        "method": value.method,
        "positive": value.positive,
        "negative": value.negative,
        "probe_probability": probes,
        "calibrated_probability": _vector(applied),
    }


def _normalize_assessment(value: Any, *, bootstrap_structure: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "class": getattr(value, "r_class", type(value).__name__),
        "summary": _frame(value.summary),
        "reliability": _frame(value.reliability),
        "positive": value.positive,
        "bins": int(value.bins),
        "bootstrap": int(value.bootstrap),
        "seed": int(value.seed),
    }
    if not bootstrap_structure:
        result["intervals"] = _frame(value.intervals)
    else:
        intervals = value.intervals
        lower = pd.to_numeric(intervals.get("lower", pd.Series(dtype=float)), errors="coerce")
        upper = pd.to_numeric(intervals.get("upper", pd.Series(dtype=float)), errors="coerce")
        result["interval_structure"] = {
            "metrics": intervals.get("metric", pd.Series(dtype=object)).astype(str).tolist(),
            "n_intervals": int(len(intervals)),
            "bounds_ordered": bool(((lower <= upper) | (lower.isna() & upper.isna())).all()),
            "finite_bound_pairs": int((np.isfinite(lower) & np.isfinite(upper)).sum()),
        }
    return result


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python parity/run_python_calibration.py <fixture.json> <output.json>"
        )

    fixture_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected_reference = fixture["r_reference"]["version"]
    if gp3mlpy.r_reference_version != expected_reference:
        raise RuntimeError(
            f"gp3mlpy reference {gp3mlpy.r_reference_version!r} "
            f"does not match fixture {expected_reference!r}."
        )

    primary = fixture["primary"]
    isotonic = fixture["isotonic"]
    assessment = fixture["assessment"]
    bootstrap = fixture["bootstrap"]

    fits: dict[str, Any] = {}
    fits["fit_platt_primary"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_calibrator(
            primary["truth"],
            primary["probability"],
            positive=primary["positive"],
            method="platt",
        ),
        lambda value: _normalize_calibrator(value, primary["probes"]),
    )
    fits["fit_isotonic_ties"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_calibrator(
            isotonic["truth"],
            isotonic["probability"],
            positive=isotonic["positive"],
            method="isotonic",
        ),
        lambda value: _normalize_calibrator(value, isotonic["probes"]),
    )
    fits["fit_default_positive"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_calibrator(
            primary["truth"],
            primary["probability"],
        ),
        lambda value: _normalize_calibrator(value, primary["probes"]),
    )

    fit_errors: dict[str, Any] = {}
    fit_errors["fit_binary_levels_error"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_calibrator(
            ["pass", "pass", "pass"],
            [0.1, 0.2, 0.3],
            method="platt",
        ),
        lambda value: True,
    )
    fit_errors["fit_unknown_positive_error"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_calibrator(
            primary["truth"],
            primary["probability"],
            positive="unknown",
            method="platt",
        ),
        lambda value: True,
    )
    fit_errors["fit_invalid_method_error"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_calibrator(
            primary["truth"],
            primary["probability"],
            positive=primary["positive"],
            method="not-a-method",
        ),
        lambda value: True,
    )
    fit_errors["fit_length_mismatch_error"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_calibrator(
            primary["truth"],
            primary["probability"][:-1],
            positive=primary["positive"],
            method="platt",
        ),
        lambda value: True,
    )

    platt = gp3mlpy.fit_gazepoint_calibrator(
        primary["truth"],
        primary["probability"],
        positive=primary["positive"],
        method="platt",
    )
    iso = gp3mlpy.fit_gazepoint_calibrator(
        isotonic["truth"],
        isotonic["probability"],
        positive=isotonic["positive"],
        method="isotonic",
    )
    applications: dict[str, Any] = {}
    applications["apply_platt_primary"] = _capture(
        lambda: gp3mlpy.apply_gazepoint_calibrator(platt, primary["probes"]),
        _vector,
    )
    applications["apply_isotonic_ties"] = _capture(
        lambda: gp3mlpy.apply_gazepoint_calibrator(iso, isotonic["probes"]),
        _vector,
    )
    applications["apply_boundary_clipping"] = _capture(
        lambda: gp3mlpy.apply_gazepoint_calibrator(platt, [0.0, 1.0]),
        _vector,
    )

    apply_errors = {
        "apply_invalid_calibrator_error": _capture(
            lambda: gp3mlpy.apply_gazepoint_calibrator(object(), [0.2, 0.8]),
            lambda value: True,
        )
    }

    assessments: dict[str, Any] = {}
    assessments["assess_no_bootstrap"] = _capture(
        lambda: gp3mlpy.assess_gazepoint_calibration(
            assessment["truth"],
            assessment["probability"],
            positive=assessment["positive"],
            bins=5,
            bootstrap=0,
            conf_level=0.95,
            seed=101,
        ),
        _normalize_assessment,
    )
    assessments["assess_bin_boundaries"] = _capture(
        lambda: gp3mlpy.assess_gazepoint_calibration(
            ["pass", "review", "pass", "review", "pass", "review"],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            positive="review",
            bins=5,
            bootstrap=0,
            seed=7,
        ),
        _normalize_assessment,
    )
    assessments["assess_bootstrap_structure"] = _capture(
        lambda: gp3mlpy.assess_gazepoint_calibration(
            assessment["truth"],
            assessment["probability"],
            positive=assessment["positive"],
            bins=int(bootstrap["bins"]),
            bootstrap=int(bootstrap["bootstrap"]),
            conf_level=float(bootstrap["conf_level"]),
            seed=int(bootstrap["seed"]),
        ),
        lambda value: _normalize_assessment(value, bootstrap_structure=True),
    )

    assessment_errors: dict[str, Any] = {}
    assessment_errors["assess_binary_levels_error"] = _capture(
        lambda: gp3mlpy.assess_gazepoint_calibration(
            ["pass", "pass", "pass"],
            [0.1, 0.2, 0.3],
            bootstrap=0,
        ),
        lambda value: True,
    )
    assessment_errors["assess_unknown_positive_error"] = _capture(
        lambda: gp3mlpy.assess_gazepoint_calibration(
            assessment["truth"],
            assessment["probability"],
            positive="unknown",
            bootstrap=0,
        ),
        lambda value: True,
    )
    assessment_errors["assess_length_mismatch_error"] = _capture(
        lambda: gp3mlpy.assess_gazepoint_calibration(
            assessment["truth"],
            assessment["probability"][:-1],
            positive=assessment["positive"],
            bootstrap=0,
        ),
        lambda value: True,
    )

    result = {
        "runtime": "Python",
        "package": "gp3mlpy",
        "package_version": gp3mlpy.__version__,
        "r_reference_version": gp3mlpy.r_reference_version,
        "fits": fits,
        "fit_errors": fit_errors,
        "applications": applications,
        "apply_errors": apply_errors,
        "assessments": assessments,
        "assessment_errors": assessment_errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
