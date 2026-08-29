from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import GP3MLError
from .objects import GazepointFoldDiagnostics, GazepointFoldDiagnosticsValidation, GazepointGroupFolds


def _ratio(numerator: float, denominator: float) -> float:
    if pd.isna(denominator):
        return np.nan
    if denominator == 0:
        return np.nan if numerator == 0 else np.inf
    return float(numerator) / float(denominator)


def _threshold(x: Any, argument: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)) or pd.isna(x) or not np.isfinite(x) or float(x) < 1:
        raise GP3MLError(f"`{argument}` must be one finite numeric value greater than or equal to 1.")
    return float(x)


def _numeric_summary(values: pd.Series) -> dict[str, float]:
    observed = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(observed):
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(len(observed)),
        "mean": float(np.mean(observed)),
        "sd": float(np.std(observed, ddof=1)) if len(observed) > 1 else np.nan,
        "median": float(np.median(observed)),
        "min": float(np.min(observed)),
        "max": float(np.max(observed)),
    }


def diagnose_gazepoint_group_folds(
    x: GazepointGroupFolds,
    imbalance_review: float = 1.5,
    imbalance_fail: float = 2,
) -> GazepointFoldDiagnostics:
    """Create fold-size, grouping, coverage, outcome-balance and exclusion diagnostics."""
    if not isinstance(x, GazepointGroupFolds):
        raise GP3MLError("`x` must be a `gazepoint_group_folds` object.")
    imbalance_review = _threshold(imbalance_review, "imbalance_review")
    imbalance_fail = _threshold(imbalance_fail, "imbalance_fail")
    if imbalance_fail < imbalance_review:
        raise GP3MLError("`imbalance_fail` must be greater than or equal to `imbalance_review`.")
    required = ["folds", "assignments", "fold_summary", "group_counts", "metadata", "audit", "validation"]
    missing = [name for name in required if name not in x]
    if missing:
        raise GP3MLError("Fold object is missing components: " + ", ".join(missing) + ".")
    required_summary = ["repeat", "fold", "fold_id", "n_total", "n_analysis", "n_assessment", "n_excluded", "assessment_prop_all", "assessment_prop_retained", "leakage_status"]
    miss = [c for c in required_summary if c not in x.fold_summary.columns]
    if miss:
        raise GP3MLError("Fold summary is missing columns: " + ", ".join(miss) + ".")
    outcome = x.metadata.get("outcome")
    if not isinstance(outcome, str) or not outcome:
        raise GP3MLError("Fold metadata does not contain one valid outcome name.")

    fold_metrics = x.fold_summary.copy()
    fold_metrics["analysis_assessment_ratio"] = [_ratio(a, b) for a, b in zip(fold_metrics.n_analysis, fold_metrics.n_assessment, strict=True)]
    fold_metrics["excluded_prop"] = np.where(fold_metrics.n_total > 0, fold_metrics.n_excluded / fold_metrics.n_total, np.nan)

    repeat_rows = []
    for repeat_value, current in fold_metrics.groupby("repeat", sort=True):
        assessment_min, assessment_max = int(current.n_assessment.min()), int(current.n_assessment.max())
        analysis_min, analysis_max = int(current.n_analysis.min()), int(current.n_analysis.max())
        repeat_rows.append({
            "repeat": int(repeat_value), "n_folds": int(len(current)),
            "assessment_min": assessment_min, "assessment_max": assessment_max,
            "assessment_size_ratio": _ratio(assessment_max, assessment_min),
            "analysis_min": analysis_min, "analysis_max": analysis_max,
            "analysis_size_ratio": _ratio(analysis_max, analysis_min),
            "total_excluded": int(current.n_excluded.sum()),
            "mean_assessment_prop_all": float(current.assessment_prop_all.mean()),
            "mean_assessment_prop_retained": float(current.assessment_prop_retained.mean()),
        })
    repeat_metrics = pd.DataFrame(repeat_rows)

    required_group = ["repeat", "fold", "fold_id", "partition", "unit", "n_groups"]
    miss = [c for c in required_group if c not in x.group_counts.columns]
    if miss:
        raise GP3MLError("Group counts are missing columns: " + ", ".join(miss) + ".")
    group_rows = []
    for key, current in x.group_counts.groupby(["repeat", "fold", "fold_id", "unit"], sort=False, dropna=False):
        rep, fold, fold_id, unit = key
        count = lambda p: int(current.loc[current.partition == p, "n_groups"].sum()) if (current.partition == p).any() else 0
        na, ne, nx = count("analysis"), count("assessment"), count("excluded")
        group_rows.append({
            "repeat": int(rep), "fold": int(fold), "fold_id": str(fold_id), "unit": str(unit),
            "n_analysis_groups": na, "n_assessment_groups": ne, "n_excluded_groups": nx,
            "analysis_assessment_group_ratio": _ratio(na, ne),
        })
    group_balance = pd.DataFrame(group_rows)

    required_assignment = ["repeat", "source_row", "partition"]
    miss = [c for c in required_assignment if c not in x.assignments.columns]
    if miss:
        raise GP3MLError("Assignments are missing columns: " + ", ".join(miss) + ".")
    a = x.assignments.copy()
    a["n_analysis"] = (a.partition == "analysis").astype(int)
    a["n_assessment"] = (a.partition == "assessment").astype(int)
    a["n_excluded"] = (a.partition == "excluded").astype(int)
    assessment_coverage = a.groupby(["repeat", "source_row"], as_index=False, sort=True)[["n_analysis", "n_assessment", "n_excluded"]].sum()
    assessment_coverage[["repeat", "source_row", "n_analysis", "n_assessment", "n_excluded"]] = assessment_coverage[["repeat", "source_row", "n_analysis", "n_assessment", "n_excluded"]].astype(int)

    first_fold = next(iter(x.folds.values()))
    source_series = []
    for partition_name in ("analysis", "assessment", "excluded"):
        frame = first_fold[partition_name]
        if not isinstance(frame, pd.DataFrame) or outcome not in frame.columns:
            raise GP3MLError(f"Fold partitions must contain outcome column `{outcome}`.")
        source_series.append(frame[outcome])
    prototype = next((s for s in source_series if len(s)), first_fold.analysis[outcome])
    all_values = pd.concat([fo[p][outcome] for fo in x.folds.values() for p in ("analysis", "assessment", "excluded")], ignore_index=True)
    outcome_is_continuous = pd.api.types.is_numeric_dtype(prototype) and all_values.dropna().nunique() > 10
    if outcome_is_continuous:
        categorical_levels: list[str] = []
    elif isinstance(prototype.dtype, pd.CategoricalDtype):
        categorical_levels = list(dict.fromkeys([*[str(v) for v in prototype.cat.categories], *sorted(all_values.dropna().astype(str).unique().tolist())]))
    else:
        categorical_levels = sorted(all_values.dropna().astype(str).unique().tolist())

    outcome_rows = []
    for fo in x.folds.values():
        for partition_name in ("analysis", "assessment", "excluded"):
            values = fo[partition_name][outcome]
            n_missing = int(values.isna().sum())
            if outcome_is_continuous:
                s = _numeric_summary(values)
                outcome_rows.append({
                    "repeat": int(fo["repeat"]), "fold": int(fo.fold), "fold_id": str(fo.fold_id), "partition": partition_name,
                    "metric_type": "numeric", "outcome_level": None, "n": int(s["n"]), "proportion": np.nan, "n_missing": n_missing,
                    "mean": s["mean"], "sd": s["sd"], "median": s["median"], "min": s["min"], "max": s["max"],
                })
            else:
                observed = values.dropna().astype(str)
                denominator = len(observed)
                levels = categorical_levels or [None]
                for level in levels:
                    n = 0 if level is None else int((observed == level).sum())
                    outcome_rows.append({
                        "repeat": int(fo["repeat"]), "fold": int(fo.fold), "fold_id": str(fo.fold_id), "partition": partition_name,
                        "metric_type": "categorical", "outcome_level": level, "n": n,
                        "proportion": n / denominator if denominator else np.nan, "n_missing": n_missing,
                        "mean": np.nan, "sd": np.nan, "median": np.nan, "min": np.nan, "max": np.nan,
                    })
    outcome_balance = pd.DataFrame(outcome_rows)
    exclusion_summary = fold_metrics.loc[:, ["repeat", "fold", "fold_id", "n_total", "n_excluded", "excluded_prop"]].copy()
    metadata = {
        "outcome": outcome,
        "generalization_target": x.metadata["generalization_target"],
        "repeats": int(x.metadata["repeats"]),
        "n_source_rows": int(x.metadata["n_source_rows"]),
        "n_folds_total": int(x.metadata["n_folds_total"]),
        "imbalance_review": imbalance_review,
        "imbalance_fail": imbalance_fail,
        "outcome_type": "numeric" if outcome_is_continuous else "categorical",
        "source_validation_status": x.validation.status,
        "source_audit_status": x.audit.status,
    }
    result = GazepointFoldDiagnostics(
        fold_metrics=fold_metrics, repeat_metrics=repeat_metrics, outcome_balance=outcome_balance,
        group_balance=group_balance, assessment_coverage=assessment_coverage,
        exclusion_summary=exclusion_summary, metadata=metadata, call="diagnose_gazepoint_group_folds",
    )
    result.validation = validate_gazepoint_fold_diagnostics(result)
    return result


def validate_gazepoint_fold_diagnostics(x: GazepointFoldDiagnostics) -> GazepointFoldDiagnosticsValidation:
    if not isinstance(x, GazepointFoldDiagnostics):
        raise GP3MLError("`x` must be a `gazepoint_fold_diagnostics` object.")
    required = ["fold_metrics", "repeat_metrics", "outcome_balance", "group_balance", "assessment_coverage", "exclusion_summary", "metadata"]
    missing = [name for name in required if name not in x]
    if missing:
        raise GP3MLError("Diagnostics object is missing components: " + ", ".join(missing) + ".")
    checks: list[dict[str, str]] = []
    def add(check_id: str, status: str, message: str, remediation: str) -> None:
        checks.append({"check_id": check_id, "status": status, "message": message, "remediation": remediation})
    fm = x.fold_metrics
    ok = len(fm) == int(x.metadata["n_folds_total"])
    add("fold_count", "pass" if ok else "fail", f"Observed {len(fm)} diagnostic fold rows; expected {x.metadata['n_folds_total']}.", "Recreate diagnostics from the complete fold-plan object.")
    ids = fm.fold_id.astype(str)
    ok = len(fm) > 0 and fm.fold_id.notna().all() and ids.str.len().gt(0).all() and not ids.duplicated().any()
    add("fold_identifiers", "pass" if ok else "fail", "Fold identifiers are complete and unique." if ok else "Fold identifiers are missing, empty, or duplicated.", "Recreate the underlying resampling plan before diagnostics.")
    observed = sorted(int(v) for v in fm["repeat"].unique())
    expected = list(range(1, int(x.metadata["repeats"]) + 1))
    ok = observed == expected
    add("repeat_structure", "pass" if ok else "fail", "Repeat identifiers match the declared repeat count." if ok else "Repeat identifiers do not match the declared repeat count.", "Recreate diagnostics from an unmodified fold-plan object.")
    ok = bool((fm.n_total == fm.n_analysis + fm.n_assessment + fm.n_excluded).all())
    add("partition_row_accounting", "pass" if ok else "fail", "Analysis, assessment, and excluded rows reconcile to fold totals." if ok else "At least one fold has inconsistent partition row accounting.", "Inspect and recreate the affected fold assignments.")
    ok = bool(((fm.n_analysis > 0) & (fm.n_assessment > 0)).all())
    add("nonempty_analysis_and_assessment", "pass" if ok else "fail", "All folds contain non-empty analysis and assessment partitions." if ok else "At least one fold has an empty analysis or assessment partition.", "Reduce the fold count or increase the number of grouping units.")
    coverage = x.assessment_coverage
    ok = len(coverage) > 0 and bool((coverage.n_assessment == 1).all())
    add("assessment_coverage_once_per_repeat", "pass" if ok else "fail", "Every source row is assessed exactly once per repeat." if ok else "At least one source row is assessed zero or multiple times in a repeat.", "Recreate diagnostics from the original, undamaged fold assignments.")
    ratios = pd.to_numeric(x.repeat_metrics.assessment_size_ratio, errors="coerce").dropna().to_numpy(float)
    maximum_ratio = np.max(ratios) if len(ratios) else np.nan
    if np.isinf(ratios).any() or (np.isfinite(maximum_ratio) and maximum_ratio > x.metadata["imbalance_fail"]): status = "fail"
    elif np.isfinite(maximum_ratio) and maximum_ratio > x.metadata["imbalance_review"]: status = "review"
    else: status = "pass"
    msg = "Assessment fold-size balance could not be calculated." if pd.isna(maximum_ratio) else f"Maximum within-repeat assessment fold-size ratio is {maximum_ratio:.3f}."
    add("assessment_fold_size_balance", status, msg, "Inspect grouping-unit sizes or use a smaller fold count.")
    ok = len(x.group_balance) > 0 and bool((x.group_balance.n_assessment_groups > 0).all())
    add("assessment_group_presence", "pass" if ok else "fail", "Every fold contains assessment groups for each recorded unit type." if ok else "At least one fold lacks assessment groups for a recorded unit type.", "Inspect the requested generalization target and grouping identifiers.")
    ca = x.outcome_balance.loc[(x.outcome_balance.metric_type == "categorical") & (x.outcome_balance.partition == "assessment")]
    missing_levels = len(ca) > 0 and bool((ca.n == 0).any())
    add("assessment_outcome_level_presence", "review" if missing_levels else "pass", "Outcome-level presence is not applicable to a continuous numeric outcome." if len(ca) == 0 else ("At least one categorical outcome level is absent from an assessment fold." if missing_levels else "Every categorical outcome level is represented in every assessment fold."), "Inspect stratification feasibility and report sparse assessment folds.")
    leakage_status = "fail" if (fm.leakage_status == "fail").any() else "review" if (fm.leakage_status == "review").any() else "pass"
    add("embedded_leakage_audits", leakage_status, f"Embedded fold leakage statuses: {', '.join(sorted(fm.leakage_status.unique()))}.", "Resolve all non-passing embedded leakage-audit findings.")
    src = [x.metadata["source_validation_status"], x.metadata["source_audit_status"]]
    source_status = "fail" if "fail" in src else "review" if "review" in src else "pass"
    add("source_plan_status", source_status, f"Source fold validation is `{src[0]}`; source audit is `{src[1]}`.", "Resolve non-passing source fold validation or audit results.")
    check_df = pd.DataFrame(checks)
    status = "fail" if (check_df.status == "fail").any() else "review" if (check_df.status == "review").any() else "pass"
    summary = pd.DataFrame({"status": ["pass", "review", "fail"], "n_checks": [int((check_df.status == s).sum()) for s in ["pass", "review", "fail"]]})
    return GazepointFoldDiagnosticsValidation(status=status, summary=summary, checks=check_df, issues=check_df.loc[check_df.status != "pass"].reset_index(drop=True), call="validate_gazepoint_fold_diagnostics")


def write_gazepoint_fold_diagnostics_csv(
    x: GazepointFoldDiagnostics,
    directory: str | Path,
    prefix: str = "gazepoint_fold_diagnostics",
    tables: Any = ("fold_metrics", "repeat_metrics", "outcome_balance", "group_balance", "assessment_coverage", "exclusion_summary", "validation_checks", "validation_issues"),
    overwrite: bool = False,
    na: str = "",
) -> dict[str, str]:
    if not isinstance(x, GazepointFoldDiagnostics):
        raise GP3MLError("`x` must be a `gazepoint_fold_diagnostics` object.")
    if not isinstance(directory, (str, Path)) or not str(directory): raise GP3MLError("`directory` must be one non-empty path.")
    if not isinstance(prefix, str) or not prefix: raise GP3MLError("`prefix` must be one non-empty string.")
    if not isinstance(overwrite, bool): raise GP3MLError("`overwrite` must be TRUE or FALSE.")
    available = {
        "fold_metrics": x.fold_metrics, "repeat_metrics": x.repeat_metrics, "outcome_balance": x.outcome_balance,
        "group_balance": x.group_balance, "assessment_coverage": x.assessment_coverage, "exclusion_summary": x.exclusion_summary,
        "validation_checks": x.validation.checks, "validation_issues": x.validation.issues,
    }
    requested = [tables] if isinstance(tables, str) else list(tables)
    if not requested or any(not isinstance(t, str) or not t for t in requested): raise GP3MLError("`tables` must contain one or more table names.")
    unknown = [t for t in requested if t not in available]
    if unknown: raise GP3MLError("Unknown diagnostic tables: " + ", ".join(unknown) + ".")
    requested = list(dict.fromkeys(requested))
    directory = Path(directory).expanduser(); directory.mkdir(parents=True, exist_ok=True)
    paths = {t: directory / f"{prefix}_{t}.csv" for t in requested}
    existing = [p for p in paths.values() if p.exists()]
    if existing and not overwrite: raise GP3MLError("Refusing to overwrite existing files: " + ", ".join(p.name for p in existing) + ".")
    for name, p in paths.items(): available[name].to_csv(p, index=False, na_rep=na)
    return {name: str(p) for name, p in paths.items()}


def _diag_repr(self: GazepointFoldDiagnostics) -> str:
    ratio = pd.to_numeric(self.repeat_metrics.assessment_size_ratio, errors="coerce").max()
    extra = f"\nMaximum assessment-size ratio: {ratio:.3f}" if np.isfinite(ratio) else ""
    return f"<gazepoint_fold_diagnostics>\nTarget: {self.metadata['generalization_target']}\nRepeats: {self.metadata['repeats']}\nFolds: {len(self.fold_metrics)}\nOutcome type: {self.metadata['outcome_type']}\nDiagnostic status: {self.validation.status.upper()}{extra}"


def _validation_repr(self: GazepointFoldDiagnosticsValidation) -> str:
    counts = {row.status: int(row.n_checks) for row in self.summary.itertuples()}
    return f"<gazepoint_fold_diagnostics_validation>\nOverall status: {self.status.upper()}\nPass: {counts.get('pass',0)}\nReview: {counts.get('review',0)}\nFail: {counts.get('fail',0)}"

GazepointFoldDiagnostics.__repr__ = _diag_repr  # type: ignore[method-assign]
GazepointFoldDiagnosticsValidation.__repr__ = _validation_repr  # type: ignore[method-assign]
