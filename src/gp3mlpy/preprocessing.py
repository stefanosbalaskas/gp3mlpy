from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._utils import assert_columns, assert_data
from .exceptions import GP3MLError
from .objects import GP3MLPreprocessor


def _is_numeric_series(x: pd.Series) -> bool:
    # R treats logical separately as a factor.
    return pd.api.types.is_numeric_dtype(x.dtype) and not pd.api.types.is_bool_dtype(x.dtype)


def _prepare_raw_frame(
    data: pd.DataFrame,
    *,
    preprocessor: GP3MLPreprocessor | None = None,
    predictors: Sequence[str] | None = None,
    fit: bool = False,
    numeric_imputation: str = "median",
    novel_level: str = "other",
) -> tuple[pd.DataFrame, dict[str, float], dict[str, list[str]]]:
    predictors = list(predictors if predictors is not None else preprocessor.predictors)
    frame = data.loc[:, predictors].copy()
    numeric_values: dict[str, float] = {}
    factor_levels: dict[str, list[str]] = {}
    for name in predictors:
        x = frame[name]
        if _is_numeric_series(x):
            numeric = pd.to_numeric(x, errors="coerce").astype(float)
            if fit:
                finite = numeric[np.isfinite(numeric)]
                value = float(finite.mean()) if numeric_imputation == "mean" and len(finite) else (
                    float(finite.median()) if len(finite) else 0.0
                )
                if not np.isfinite(value):
                    value = 0.0
                numeric_values[name] = value
            else:
                value = float(preprocessor.numeric_imputation_values[name])
            frame[name] = numeric.fillna(value).astype(float)
        else:
            vals = x.astype(object).to_numpy()
            text = np.asarray(["<missing>" if pd.isna(v) or str(v) == "" else str(v) for v in vals], dtype=object)
            if fit:
                base = set(text.tolist())
                if novel_level == "other":
                    base.add("<other>")
                levels = sorted(base)
                factor_levels[name] = levels
            else:
                levels = list(preprocessor.factor_levels[name])
                novel = np.asarray([v not in levels for v in text])
                if np.any(novel) and preprocessor.novel_level == "error":
                    raise GP3MLError(f"Novel levels in `{name}`.")
                text[novel] = "<other>"
            frame[name] = pd.Categorical(text, categories=levels, ordered=False)
    return frame, numeric_values, factor_levels


def _model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    columns: dict[str, np.ndarray] = {}
    for name in frame.columns:
        x = frame[name]
        if _is_numeric_series(x):
            columns[name] = np.asarray(x, dtype=float)
        else:
            if isinstance(x.dtype, pd.CategoricalDtype):
                levels = [str(v) for v in x.cat.categories]
                values = x.astype(object).to_numpy()
            else:
                values = x.astype(object).to_numpy()
                levels = sorted({str(v) for v in values if not pd.isna(v)})
            for level in levels:
                # stats::model.matrix(~ . - 1) naming: factor name immediately followed by level.
                columns[f"{name}{level}"] = np.asarray([1.0 if str(v) == level else 0.0 for v in values], dtype=float)
    return pd.DataFrame(columns, index=frame.index, dtype=float)


def fit_gazepoint_preprocessor(
    data: pd.DataFrame,
    predictors: Sequence[str],
    numeric_imputation: str = "median",
    center: bool = True,
    scale: bool = True,
    novel_level: str = "other",
    remove_zero_variance: bool = True,
) -> GP3MLPreprocessor:
    """Fit preprocessing parameters on an analysis partition only."""
    assert_data(data)
    predictors = list(predictors)
    assert_columns(data, predictors, "predictors")
    if numeric_imputation not in {"median", "mean"}:
        raise GP3MLError("`numeric_imputation` must be one of: median, mean.")
    if novel_level not in {"other", "error"}:
        raise GP3MLError("`novel_level` must be one of: other, error.")
    frame, numeric_values, factor_levels = _prepare_raw_frame(
        data,
        predictors=predictors,
        fit=True,
        numeric_imputation=numeric_imputation,
        novel_level=novel_level,
    )
    matrix = _model_matrix(frame)
    if remove_zero_variance and matrix.shape[1]:
        keep = [c for c in matrix.columns if matrix[c].nunique(dropna=False) > 1]
        matrix = matrix.loc[:, keep]
    means = matrix.mean(axis=0) if center and matrix.shape[1] else pd.Series(0.0, index=matrix.columns, dtype=float)
    sds = matrix.std(axis=0, ddof=1) if scale and matrix.shape[1] else pd.Series(1.0, index=matrix.columns, dtype=float)
    sds = sds.mask(~np.isfinite(sds) | (sds == 0), 1.0)
    return GP3MLPreprocessor(
        predictors=predictors,
        numeric_imputation=numeric_imputation,
        numeric_imputation_values=numeric_values,
        factor_levels=factor_levels,
        novel_level=novel_level,
        columns=list(matrix.columns),
        center=means.to_dict(),
        scale=sds.to_dict(),
        remove_zero_variance=bool(remove_zero_variance),
    )


def bake_gazepoint_preprocessor(preprocessor: GP3MLPreprocessor, new_data: pd.DataFrame) -> np.ndarray:
    """Apply a fitted gp3ml-compatible preprocessor without re-estimation."""
    if not isinstance(preprocessor, GP3MLPreprocessor):
        raise GP3MLError("`preprocessor` must be fitted by gp3ml.")
    assert_columns(new_data, preprocessor.predictors, "predictors")
    frame, _, _ = _prepare_raw_frame(new_data, preprocessor=preprocessor, fit=False)
    matrix = _model_matrix(frame)
    for name in preprocessor.columns:
        if name not in matrix.columns:
            matrix[name] = 0.0
    matrix = matrix.loc[:, preprocessor.columns]
    if matrix.shape[1]:
        center = pd.Series(preprocessor.center, dtype=float).reindex(preprocessor.columns)
        scale = pd.Series(preprocessor.scale, dtype=float).reindex(preprocessor.columns)
        matrix = (matrix - center) / scale
    return matrix.to_numpy(dtype=float, copy=True)
