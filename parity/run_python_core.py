from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gp3mlpy


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def _normalize_row(data: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(data, pd.DataFrame) or len(data) != 1:
        raise RuntimeError("Expected a one-row DataFrame from the gp3mlpy metric function.")
    row = data.iloc[0]
    return {str(name): _json_scalar(value) for name, value in row.items()}


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python parity/run_python_core.py <fixture.json> <output.json>")

    fixture_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    expected_reference = fixture["r_reference"]["version"]
    if gp3mlpy.r_reference_version != expected_reference:
        raise RuntimeError(
            f"gp3mlpy reference {gp3mlpy.r_reference_version!r} does not match fixture {expected_reference!r}."
        )

    classification: dict[str, dict[str, Any]] = {}
    for case in fixture["classification_cases"]:
        result = gp3mlpy.gazepoint_classification_metrics(
            truth=case["truth"],
            probability=case["probability"],
            predicted=case["predicted"],
            positive=case["positive"],
            threshold=float(case["threshold"]),
        )
        classification[case["id"]] = _normalize_row(result)

    regression: dict[str, dict[str, Any]] = {}
    for case in fixture["regression_cases"]:
        result = gp3mlpy.gazepoint_regression_metrics(
            truth=case["truth"],
            prediction=case["prediction"],
        )
        regression[case["id"]] = _normalize_row(result)

    result = {
        "runtime": "Python",
        "package": "gp3mlpy",
        "package_version": gp3mlpy.__version__,
        "r_reference_version": gp3mlpy.r_reference_version,
        "prohibited_uses": gp3mlpy.gp3ml_prohibited_uses(),
        "classification": classification,
        "regression": regression,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
