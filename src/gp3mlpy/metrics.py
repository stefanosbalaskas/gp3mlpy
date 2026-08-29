from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ._utils import clip_probability
from .exceptions import GP3MLError
from .objects import GP3MLMetricUncertainty, GP3MLTask
from .task_governance import assert_gp3ml_use_case


def _as_object_array(x: Sequence[Any] | pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.astype(object).to_numpy()
    return np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=object)


def _missing_mask(values: np.ndarray) -> np.ndarray:
    return pd.isna(values)


def _binary_values(truth: Sequence[Any] | pd.Series, positive: str | None = None) -> tuple[np.ndarray, str, str]:
    values = _as_object_array(truth)
    nonmissing = values[~_missing_mask(values)]
    levels_found = sorted({str(v) for v in nonmissing})
    if len(levels_found) != 2:
        raise GP3MLError("Binary metrics require exactly two truth levels.")
    positive = levels_found[1] if positive is None else str(positive)
    if positive not in levels_found:
        raise GP3MLError("Unknown positive level.")
    negative = next(v for v in levels_found if v != positive)
    y = np.full(len(values), np.nan, dtype=float)
    keep = ~_missing_mask(values)
    y[keep] = np.asarray([1.0 if str(v) == positive else 0.0 for v in values[keep]])
    return y, positive, negative


def _auc(y: np.ndarray, probability: np.ndarray) -> float:
    keep = np.isfinite(y) & ~np.isnan(probability)
    yy = y[keep]
    pp = probability[keep]
    n1 = int(np.sum(yy == 1))
    n0 = int(np.sum(yy == 0))
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(pp, method="average")
    return float((np.sum(ranks[yy == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))


def _pr_auc(y: np.ndarray, probability: np.ndarray) -> float:
    keep = np.isfinite(y) & ~np.isnan(probability)
    yy = y[keep]
    pp = probability[keep]
    positives = int(np.sum(yy == 1))
    if positives == 0:
        return float("nan")
    # R's order(..., decreasing=TRUE) is stable for ties in the current implementation.
    order = np.argsort(-pp, kind="stable")
    yy = yy[order]
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _safe(num: float, den: float) -> float:
    return float("nan") if den == 0 else float(num / den)


def gazepoint_classification_metrics(
    truth: Sequence[Any] | pd.Series,
    probability: Sequence[float] | np.ndarray,
    predicted: Sequence[Any] | pd.Series | None = None,
    positive: str | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute gp3ml 0.3.0 binary classification metrics."""
    y, positive_level, _ = _binary_values(truth, positive)
    p = clip_probability(probability)
    if len(p) != len(y):
        # R recycles in some vector operations, but prediction workflows require aligned rows.
        # Keep a deterministic, explicit failure rather than accidental NumPy broadcasting.
        raise GP3MLError("`probability` must match `truth`.")
    if predicted is None:
        pred_y = np.where(np.isnan(p), np.nan, (p >= threshold).astype(float))
    else:
        vals = _as_object_array(predicted)
        if len(vals) != len(y):
            raise GP3MLError("`predicted` must match `truth`.")
        pred_y = np.full(len(vals), np.nan, dtype=float)
        keep = ~_missing_mask(vals)
        pred_y[keep] = np.asarray([1.0 if str(v) == positive_level else 0.0 for v in vals[keep]])

    tp = int(np.sum((pred_y == 1) & (y == 1)))
    tn = int(np.sum((pred_y == 0) & (y == 0)))
    fp = int(np.sum((pred_y == 1) & (y == 0)))
    fn = int(np.sum((pred_y == 0) & (y == 1)))
    sensitivity = _safe(tp, tp + fn)
    specificity = _safe(tn, tn + fp)
    precision = _safe(tp, tp + fp)
    if np.isnan(precision) or np.isnan(sensitivity) or precision + sensitivity == 0:
        f1 = float("nan")
    else:
        f1 = float(2 * precision * sensitivity / (precision + sensitivity))
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    available_balanced = [z for z in (sensitivity, specificity) if not np.isnan(z)]
    balanced_accuracy = float(np.mean(available_balanced)) if available_balanced else float("nan")
    valid = np.isfinite(y) & ~np.isnan(p)
    brier = float(np.mean((p[valid] - y[valid]) ** 2)) if np.any(valid) else float("nan")
    log_loss = (
        float(-np.mean(y[valid] * np.log(p[valid]) + (1 - y[valid]) * np.log(1 - p[valid])))
        if np.any(valid)
        else float("nan")
    )
    return pd.DataFrame(
        [{
            "n": int(np.sum(valid)),
            "threshold": threshold,
            "accuracy": _safe(tp + tn, tp + tn + fp + fn),
            "balanced_accuracy": balanced_accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "recall": sensitivity,
            "f1": f1,
            "mcc": float("nan") if mcc_den == 0 else float((tp * tn - fp * fn) / mcc_den),
            "roc_auc": _auc(y, p),
            "pr_auc": _pr_auc(y, p),
            "brier": brier,
            "log_loss": log_loss,
        }]
    )


def gazepoint_regression_metrics(
    truth: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Compute gp3ml 0.3.0 regression metrics."""
    truth_arr = np.asarray(truth, dtype=float)
    pred_arr = np.asarray(prediction, dtype=float)
    if len(truth_arr) != len(pred_arr):
        raise GP3MLError("`prediction` must match `truth`.")
    keep = np.isfinite(truth_arr) & np.isfinite(pred_arr)
    truth_arr = truth_arr[keep]
    pred_arr = pred_arr[keep]
    residual = truth_arr - pred_arr
    n = len(truth_arr)
    rmse = float(np.sqrt(np.mean(residual**2))) if n else float("nan")
    mae = float(np.mean(np.abs(residual))) if n else float("nan")
    if n:
        sst = float(np.sum((truth_arr - np.mean(truth_arr)) ** 2))
        r_squared = float("nan") if sst == 0 else float(1 - np.sum(residual**2) / sst)
    else:
        r_squared = float("nan")
    correlation = float("nan")
    if n >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = float(np.corrcoef(truth_arr, pred_arr)[0, 1])
    return pd.DataFrame([{"n": n, "rmse": rmse, "mae": mae, "r_squared": r_squared, "correlation": correlation}])


def gazepoint_performance_metrics(
    task: GP3MLTask,
    truth: Sequence[Any] | pd.Series,
    prediction: Sequence[Any] | pd.Series | None = None,
    probability: Sequence[float] | np.ndarray | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Dispatch to classification or regression metrics for a governed task."""
    assert_gp3ml_use_case(task)
    if task.task_type == "classification":
        if probability is None:
            probability = []
        return gazepoint_classification_metrics(truth, probability, prediction, task.positive, threshold)
    if prediction is None:
        prediction = []
    return gazepoint_regression_metrics(truth, prediction)


def bootstrap_gazepoint_metrics(
    task: GP3MLTask,
    truth: Sequence[Any] | pd.Series,
    prediction: Sequence[Any] | pd.Series | None = None,
    probability: Sequence[float] | np.ndarray | None = None,
    threshold: float = 0.5,
    bootstrap: int = 1000,
    conf_level: float = 0.95,
    seed: int = 1,
) -> GP3MLMetricUncertainty:
    """Percentile bootstrap uncertainty matching gp3ml's row/stratified scheme."""
    assert_gp3ml_use_case(task)
    bootstrap = int(bootstrap)
    if bootstrap < 1:
        raise GP3MLError("`bootstrap` must be positive.")
    truth_arr = _as_object_array(truth)
    n = len(truth_arr)
    if n < 2:
        raise GP3MLError("At least two observations are required.")
    pred_arr = None if prediction is None else _as_object_array(prediction)
    prob_arr = None if probability is None else np.asarray(probability, dtype=float)
    if task.task_type == "classification" and (prob_arr is None or len(prob_arr) != n):
        raise GP3MLError("`probability` must match `truth`.")
    if task.task_type == "regression" and (pred_arr is None or len(pred_arr) != n):
        raise GP3MLError("`prediction` must match `truth`.")
    point = gazepoint_performance_metrics(task, truth_arr, pred_arr, prob_arr, threshold)
    rng = np.random.RandomState(int(seed))
    draws: list[pd.DataFrame] = []
    if task.task_type == "classification":
        class_strings = np.asarray([None if pd.isna(v) else str(v) for v in truth_arr], dtype=object)
        classes = sorted({v for v in class_strings if v is not None})
        if len(classes) != 2:
            raise GP3MLError("Binary bootstrap requires two observed classes.")
        class_indices = [np.flatnonzero(class_strings == level) for level in classes]
        for _ in range(bootstrap):
            idx = np.concatenate([rng.choice(x, size=len(x), replace=True) for x in class_indices])
            draws.append(
                gazepoint_classification_metrics(
                    truth_arr[idx], prob_arr[idx], None if pred_arr is None else pred_arr[idx], task.positive, threshold
                )
            )
    else:
        assert pred_arr is not None
        for _ in range(bootstrap):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            draws.append(gazepoint_regression_metrics(np.asarray(truth_arr, dtype=float)[idx], np.asarray(pred_arr, dtype=float)[idx]))
    draws_df = pd.concat(draws, ignore_index=True)
    metric_columns = [c for c in draws_df.columns if pd.api.types.is_numeric_dtype(draws_df[c]) and c not in {"n", "threshold"}]
    alpha = (1 - conf_level) / 2
    intervals = pd.DataFrame([
        {
            "metric": name,
            "estimate": float(point[name].iloc[0]),
            "lower": float(draws_df[name].quantile(alpha, interpolation="linear")),
            "upper": float(draws_df[name].quantile(1 - alpha, interpolation="linear")),
        }
        for name in metric_columns
    ])
    return GP3MLMetricUncertainty(
        point=point,
        intervals=intervals,
        draws=draws_df,
        bootstrap=bootstrap,
        conf_level=conf_level,
        seed=seed,
        task=task,
    )
