from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ._utils import clip_probability
from .exceptions import GP3MLError
from .metrics import _binary_values
from .objects import GP3MLCalibrationAssessment, GP3MLObject


class GP3MLCalibrator(GP3MLObject):
    r_class = "gp3ml_calibrator"


def _fit_platt(y: np.ndarray, p: np.ndarray):
    keep = np.isfinite(y) & np.isfinite(p)
    x = np.log(p[keep] / (1 - p[keep]))
    X = sm.add_constant(x, prepend=True, has_constant="add")
    return sm.GLM(y[keep], X, family=sm.families.Binomial()).fit()


def _pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    means: list[float] = []
    ws: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for i, (value, weight) in enumerate(zip(values, weights, strict=True)):
        means.append(float(value)); ws.append(float(weight)); starts.append(i); ends.append(i)
        while len(means) >= 2 and means[-2] > means[-1]:
            total = ws[-2] + ws[-1]
            pooled = (means[-2] * ws[-2] + means[-1] * ws[-1]) / total
            means[-2:] = [pooled]; ws[-2:] = [total]
            starts[-2:] = [starts[-2]]; ends[-2:] = [ends[-1]]
    fitted = np.empty(len(values), dtype=float)
    for mean, start, end in zip(means, starts, ends, strict=True):
        fitted[start:end + 1] = mean
    return fitted


def _fit_isotonic(p: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    keep = np.isfinite(p) & np.isfinite(y)
    pp, yy = p[keep], y[keep]
    order = np.argsort(pp, kind="stable")
    pp, yy = pp[order], yy[order]
    unique_x, inverse = np.unique(pp, return_inverse=True)
    sums = np.bincount(inverse, weights=yy)
    counts = np.bincount(inverse).astype(float)
    means = sums / counts
    fitted_unique = _pava(means, counts)
    # Equivalent interpolation representation after tie pooling.
    return {"x": unique_x.astype(float), "yf": fitted_unique.astype(float)}


def fit_gazepoint_calibrator(
    truth: Sequence[object] | pd.Series,
    probability: Sequence[float] | np.ndarray,
    positive: str | None = None,
    method: str = "platt",
) -> GP3MLCalibrator:
    """Fit Platt or isotonic probability calibration using gp3ml semantics."""
    if method not in {"platt", "isotonic"}:
        raise GP3MLError("`method` must be one of: platt, isotonic.")
    y, pos, neg = _binary_values(truth, positive)
    p = clip_probability(probability)
    if len(y) != len(p):
        raise GP3MLError("`probability` must match `truth`.")
    fit = _fit_platt(y, p) if method == "platt" else _fit_isotonic(p, y)
    return GP3MLCalibrator(method=method, fit=fit, positive=pos, negative=neg)


def apply_gazepoint_calibrator(calibrator: GP3MLCalibrator, probability: Sequence[float] | np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator and clip results to the open unit interval."""
    if not isinstance(calibrator, GP3MLCalibrator):
        raise GP3MLError("`calibrator` must be fitted by gp3ml.")
    p = clip_probability(probability)
    if calibrator.method == "platt":
        x = np.log(p / (1 - p))
        X = sm.add_constant(x, prepend=True, has_constant="add")
        result = np.asarray(calibrator.fit.predict(X), dtype=float)
    else:
        result = np.interp(p, calibrator.fit["x"], calibrator.fit["yf"], left=calibrator.fit["yf"][0], right=calibrator.fit["yf"][-1])
    return clip_probability(result)


def _calibration_core(y: np.ndarray, p: np.ndarray, bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = clip_probability(p)
    intercept = slope = float("nan")
    try:
        fit = _fit_platt(y, p)
        params = np.asarray(fit.params, dtype=float)
        if len(params) >= 2:
            intercept, slope = float(params[0]), float(params[1])
    except Exception:
        pass
    breaks = np.linspace(0.0, 1.0, int(bins) + 1)
    # R cut(... include.lowest=TRUE, right=TRUE): intervals [0,b1], (b1,b2], ...
    bin_idx = np.searchsorted(breaks, p, side="left")
    bin_idx = np.clip(bin_idx, 1, int(bins)).astype(int)
    rows = []
    for b in sorted(np.unique(bin_idx)):
        idx = np.flatnonzero(bin_idx == b)
        rows.append({"bin": int(b), "n": len(idx), "mean_probability": float(np.mean(p[idx])), "observed_rate": float(np.mean(y[idx]))})
    reliability = pd.DataFrame(rows, columns=["bin", "n", "mean_probability", "observed_rate"])
    ece = float(np.sum(reliability["n"] / reliability["n"].sum() * np.abs(reliability["mean_probability"] - reliability["observed_rate"])))
    summary = pd.DataFrame([{
        "intercept": intercept,
        "slope": slope,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "ece": ece,
    }])
    return summary, reliability


def assess_gazepoint_calibration(
    truth: Sequence[object] | pd.Series,
    probability: Sequence[float] | np.ndarray,
    positive: str | None = None,
    bins: int = 10,
    bootstrap: int = 200,
    conf_level: float = 0.95,
    seed: int = 1,
) -> GP3MLCalibrationAssessment:
    """Assess calibration and percentile-bootstrap uncertainty."""
    y, pos, _ = _binary_values(truth, positive)
    p = clip_probability(probability)
    if len(y) != len(p):
        raise GP3MLError("`probability` must match `truth`.")
    summary, reliability = _calibration_core(y, p, int(bins))
    bootstrap = int(bootstrap)
    rng = np.random.RandomState(int(seed))
    draws: list[pd.DataFrame] = []
    if bootstrap > 0:
        for _ in range(bootstrap):
            idx = rng.choice(np.arange(len(y)), size=len(y), replace=True)
            draw_summary, _ = _calibration_core(y[idx], p[idx], int(bins))
            draws.append(draw_summary)
    draws_df = pd.concat(draws, ignore_index=True) if draws else pd.DataFrame()
    intervals = pd.DataFrame()
    if len(draws_df):
        alpha = (1 - conf_level) / 2
        intervals = pd.DataFrame({
            "metric": list(draws_df.columns),
            "lower": [float(draws_df[c].quantile(alpha, interpolation="linear")) for c in draws_df.columns],
            "upper": [float(draws_df[c].quantile(1 - alpha, interpolation="linear")) for c in draws_df.columns],
        })
    return GP3MLCalibrationAssessment(
        summary=summary,
        reliability=reliability,
        intervals=intervals,
        positive=pos,
        bins=int(bins),
        bootstrap=bootstrap,
        seed=seed,
    )
