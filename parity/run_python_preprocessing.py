from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import gp3mlpy


def _data_frame(spec: dict[str, Any]) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for name, values in spec["columns"].items():
        kind = spec["types"][name]
        if kind == "numeric":
            columns[name] = pd.Series(values, dtype=float)
        elif kind == "logical":
            columns[name] = pd.Series(values, dtype="boolean")
        else:
            columns[name] = pd.Series(values, dtype=object)
    return pd.DataFrame(columns)


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


def _named_numeric(values: dict[str, Any], order: list[str]) -> list[dict[str, Any]]:
    return [
        {"name": name, "value": _scalar(values[name])}
        for name in order
        if name in values
    ]


def _named_levels(values: dict[str, list[str]], order: list[str]) -> list[dict[str, Any]]:
    return [
        {"name": name, "levels": [str(value) for value in values[name]]}
        for name in order
        if name in values
    ]


def _normalize_preprocessor(value: Any) -> dict[str, Any]:
    columns = [str(name) for name in value.columns]
    predictors = [str(name) for name in value.predictors]
    return {
        "class": getattr(value, "r_class", type(value).__name__),
        "predictors": predictors,
        "numeric_imputation": value.numeric_imputation,
        "numeric_imputation_values": _named_numeric(
            value.numeric_imputation_values, predictors
        ),
        "factor_levels": _named_levels(value.factor_levels, predictors),
        "novel_level": value.novel_level,
        "columns": columns,
        "center": _named_numeric(value.center, columns),
        "scale": _named_numeric(value.scale, columns),
        "remove_zero_variance": bool(value.remove_zero_variance),
    }


def _normalize_matrix(value: Any, columns: list[str]) -> dict[str, Any]:
    matrix = np.asarray(value, dtype=float)
    return {
        "columns": [str(name) for name in columns],
        "nrow": int(matrix.shape[0]),
        "ncol": int(matrix.shape[1]),
        "values": [[_scalar(item) for item in row] for row in matrix.tolist()],
    }


def _capture(call: Callable[[], Any], normalize: Callable[[Any], Any]) -> dict[str, Any]:
    try:
        value = call()
        return {"status": "success", "value": normalize(value)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python parity/run_python_preprocessing.py <fixture.json> <output.json>"
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

    training = _data_frame(fixture["training"])
    new_data = _data_frame(fixture["new_data"])

    fits: dict[str, Any] = {}
    fits["fit_median_mixed"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_preprocessor(
            training,
            ["num1", "cat", "constant"],
            numeric_imputation="median",
            center=True,
            scale=True,
            novel_level="other",
            remove_zero_variance=True,
        ),
        _normalize_preprocessor,
    )
    fits["fit_mean_no_center_scale"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_preprocessor(
            training,
            ["num1", "num2", "cat"],
            numeric_imputation="mean",
            center=False,
            scale=False,
            novel_level="other",
            remove_zero_variance=False,
        ),
        _normalize_preprocessor,
    )
    fits["fit_all_missing_numeric"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_preprocessor(
            training,
            ["all_missing"],
            remove_zero_variance=True,
        ),
        _normalize_preprocessor,
    )
    fits["fit_boolean_predictor"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_preprocessor(
            training,
            ["logical"],
            center=False,
            scale=False,
            novel_level="other",
            remove_zero_variance=False,
        ),
        _normalize_preprocessor,
    )
    fits["fit_keep_zero_variance"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_preprocessor(
            training,
            ["constant"],
            center=True,
            scale=True,
            remove_zero_variance=False,
        ),
        _normalize_preprocessor,
    )

    fit_errors: dict[str, Any] = {}
    fit_errors["fit_missing_predictor_error"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_preprocessor(
            training,
            ["num1", "not_present"],
        ),
        lambda value: True,
    )
    fit_errors["fit_too_few_rows_error"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_preprocessor(
            training.iloc[:1].copy(),
            ["num1"],
        ),
        lambda value: True,
    )
    fit_errors["fit_invalid_numeric_imputation"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_preprocessor(
            training,
            ["num1"],
            numeric_imputation="mode",
        ),
        lambda value: True,
    )
    fit_errors["fit_invalid_novel_level"] = _capture(
        lambda: gp3mlpy.fit_gazepoint_preprocessor(
            training,
            ["cat"],
            novel_level="silent",
        ),
        lambda value: True,
    )

    median_pp = gp3mlpy.fit_gazepoint_preprocessor(
        training,
        ["num1", "cat", "constant"],
        numeric_imputation="median",
        center=True,
        scale=True,
        novel_level="other",
        remove_zero_variance=True,
    )
    mean_pp = gp3mlpy.fit_gazepoint_preprocessor(
        training,
        ["num1", "num2", "cat"],
        numeric_imputation="mean",
        center=False,
        scale=False,
        novel_level="other",
        remove_zero_variance=False,
    )
    category_pp = gp3mlpy.fit_gazepoint_preprocessor(
        training,
        ["cat"],
        center=False,
        scale=False,
        novel_level="other",
        remove_zero_variance=False,
    )
    strict_category_pp = gp3mlpy.fit_gazepoint_preprocessor(
        training,
        ["cat"],
        center=False,
        scale=False,
        novel_level="error",
        remove_zero_variance=False,
    )

    bakes: dict[str, Any] = {}
    bakes["bake_training_roundtrip"] = _capture(
        lambda: gp3mlpy.bake_gazepoint_preprocessor(median_pp, training),
        lambda value: _normalize_matrix(value, median_pp.columns),
    )
    bakes["bake_missing_and_novel"] = _capture(
        lambda: gp3mlpy.bake_gazepoint_preprocessor(mean_pp, new_data),
        lambda value: _normalize_matrix(value, mean_pp.columns),
    )
    bakes["bake_novel_to_other"] = _capture(
        lambda: gp3mlpy.bake_gazepoint_preprocessor(category_pp, new_data),
        lambda value: _normalize_matrix(value, category_pp.columns),
    )
    only_a = pd.DataFrame({"cat": pd.Series(["a", "a", "a"], dtype=object)})
    bakes["bake_missing_trained_level"] = _capture(
        lambda: gp3mlpy.bake_gazepoint_preprocessor(category_pp, only_a),
        lambda value: _normalize_matrix(value, category_pp.columns),
    )

    bake_errors: dict[str, Any] = {}
    bake_errors["bake_novel_error"] = _capture(
        lambda: gp3mlpy.bake_gazepoint_preprocessor(strict_category_pp, new_data),
        lambda value: True,
    )
    bake_errors["bake_missing_predictor_error"] = _capture(
        lambda: gp3mlpy.bake_gazepoint_preprocessor(
            mean_pp,
            new_data.drop(columns=["num2"]),
        ),
        lambda value: True,
    )
    bake_errors["bake_invalid_preprocessor_error"] = _capture(
        lambda: gp3mlpy.bake_gazepoint_preprocessor(object(), new_data),
        lambda value: True,
    )

    result = {
        "runtime": "Python",
        "package": "gp3mlpy",
        "package_version": gp3mlpy.__version__,
        "r_reference_version": gp3mlpy.r_reference_version,
        "fits": fits,
        "fit_errors": fit_errors,
        "bakes": bakes,
        "bake_errors": bake_errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
