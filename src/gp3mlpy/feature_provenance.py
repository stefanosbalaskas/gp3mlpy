from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._utils import as_list, is_missing_text, recycle
from .exceptions import GP3MLError
from .objects import GazepointFeatureManifestValidation

REQUIRED_COLUMNS = [
    "feature", "scientific_source", "source_table", "transformation", "availability_stage",
    "prediction_time_available", "outcome_derived", "post_outcome", "identifier",
    "preprocessing_scope", "fold_local_required", "reviewer_notes",
]
AVAILABILITY_STAGES = ["pre_exposure", "during_exposure", "post_exposure_pre_outcome", "at_prediction", "post_outcome", "unknown"]
PREPROCESSING_SCOPES = ["none", "global", "analysis_partition", "resampling_fold", "unknown"]
CHARACTER_COLUMNS = ["feature", "scientific_source", "source_table", "transformation", "availability_stage", "preprocessing_scope", "reviewer_notes"]
LOGICAL_COLUMNS = ["prediction_time_available", "outcome_derived", "post_outcome", "identifier", "fold_local_required"]


def _as_feature_manifest(x: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise GP3MLError("`x` must be a feature-manifest data frame.")
    missing = [c for c in REQUIRED_COLUMNS if c not in x.columns]
    if missing:
        raise GP3MLError("Feature manifest is missing required columns: " + ", ".join(missing) + ".")
    if x.columns.duplicated().any():
        raise GP3MLError("Feature-manifest column names must be unique.")
    out = x.copy()
    invalid_char = [c for c in CHARACTER_COLUMNS if not (pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]))]
    if invalid_char:
        raise GP3MLError("Manifest columns must be character: " + ", ".join(invalid_char) + ".")
    # Nullable boolean and object booleans are accepted to represent R logical NA.
    for c in LOGICAL_COLUMNS:
        vals = out[c].dropna().tolist()
        if any(not isinstance(v, (bool, np.bool_)) for v in vals):
            raise GP3MLError("Manifest columns must be logical: " + c + ".")
    features = out.feature.map(lambda v: "" if pd.isna(v) else str(v).strip())
    if len(out) == 0 or (features == "").any() or features.duplicated().any():
        raise GP3MLError("`feature` must contain unique, non-missing, non-empty names.")
    out["feature"] = features
    if out.availability_stage.isna().any() or (~out.availability_stage.isin(AVAILABILITY_STAGES)).any():
        raise GP3MLError("`availability_stage` must use one of: " + ", ".join(AVAILABILITY_STAGES) + ".")
    if out.preprocessing_scope.isna().any() or (~out.preprocessing_scope.isin(PREPROCESSING_SCOPES)).any():
        raise GP3MLError("`preprocessing_scope` must use one of: " + ", ".join(PREPROCESSING_SCOPES) + ".")
    out = out[REQUIRED_COLUMNS + [c for c in out.columns if c not in REQUIRED_COLUMNS]]
    out.attrs["r_class"] = "gazepoint_feature_manifest"
    return out


def create_gazepoint_feature_manifest(
    features: list[str] | tuple[str, ...] | str,
    scientific_source: Any = None,
    source_table: Any = None,
    transformation: Any = "none",
    availability_stage: Any = "unknown",
    prediction_time_available: Any = None,
    outcome_derived: Any = False,
    post_outcome: Any = False,
    identifier: Any = False,
    preprocessing_scope: Any = "unknown",
    fold_local_required: Any = None,
    reviewer_notes: Any = "",
) -> pd.DataFrame:
    """Create a Gazepoint feature-provenance manifest."""
    vals = as_list(features)
    if not vals or any(not isinstance(v, str) for v in vals):
        raise GP3MLError("`features` must be a non-empty character vector.")
    vals = [v.strip() for v in vals]
    if any(not v for v in vals) or len(set(vals)) != len(vals):
        raise GP3MLError("`features` must contain unique, non-missing, non-empty names.")
    n = len(vals)
    # None represents R NA_* for optional metadata.
    ss = recycle(scientific_source, n, "scientific_source", kind="character")
    st = recycle(source_table, n, "source_table", kind="character")
    tr = recycle(transformation, n, "transformation", kind="character")
    av = recycle(availability_stage, n, "availability_stage", kind="character", allow_na=False)
    pa = recycle(prediction_time_available, n, "prediction_time_available", kind="logical")
    od = recycle(outcome_derived, n, "outcome_derived", kind="logical", allow_na=False)
    po = recycle(post_outcome, n, "post_outcome", kind="logical", allow_na=False)
    ident = recycle(identifier, n, "identifier", kind="logical", allow_na=False)
    ps = recycle(preprocessing_scope, n, "preprocessing_scope", kind="character", allow_na=False)
    fl = recycle(fold_local_required, n, "fold_local_required", kind="logical")
    rn = recycle(reviewer_notes, n, "reviewer_notes", kind="character")
    df = pd.DataFrame({
        "feature": vals, "scientific_source": ss, "source_table": st, "transformation": tr,
        "availability_stage": av, "prediction_time_available": pa, "outcome_derived": od,
        "post_outcome": po, "identifier": ident, "preprocessing_scope": ps,
        "fold_local_required": fl, "reviewer_notes": rn,
    })
    return _as_feature_manifest(df)


def validate_gazepoint_feature_manifest(x: pd.DataFrame) -> GazepointFeatureManifestValidation:
    """Validate a feature-provenance manifest against gp3ml 0.3.0 safeguards."""
    manifest = _as_feature_manifest(x)
    rows: list[dict[str, str]] = []
    def add(feature: str, check_id: str, status: str, field: str, message: str, remediation: str) -> None:
        rows.append({"feature": feature, "check_id": check_id, "status": status, "field": field, "message": message, "remediation": remediation})
    for _, row in manifest.iterrows():
        feature = row.feature
        missing = [c for c in ["scientific_source", "source_table", "transformation"] if is_missing_text(row[c])]
        add(feature, "provenance_metadata_complete", "review" if missing else "pass", ", ".join(missing), "Required provenance metadata is incomplete." if missing else "Required provenance metadata is complete.", "Document the missing provenance fields before predictive evaluation." if missing else "None.")
        unknown_stage = row.availability_stage == "unknown"
        add(feature, "availability_stage_declared", "review" if unknown_stage else "pass", "availability_stage", "The feature availability stage is unknown." if unknown_stage else "The feature availability stage is declared.", "Declare when the feature becomes available." if unknown_stage else "None.")
        pred = row.prediction_time_available
        pred_missing = pd.isna(pred)
        add(feature, "prediction_time_available", "review" if pred_missing else "fail" if not bool(pred) else "pass", "prediction_time_available", "Prediction-time availability has not been declared." if pred_missing else "The feature is unavailable at the intended prediction time." if not bool(pred) else "The feature is available at the intended prediction time.", "Declare prediction-time availability." if pred_missing else "Remove the feature from the intended predictor set." if not bool(pred) else "None.")
        add(feature, "outcome_derived", "fail" if bool(row.outcome_derived) else "pass", "outcome_derived", "The feature is declared as outcome-derived." if bool(row.outcome_derived) else "The feature is not declared as outcome-derived.", "Remove outcome-derived features from the predictor set." if bool(row.outcome_derived) else "None.")
        stage_post = row.availability_stage == "post_outcome"
        post = bool(row.post_outcome) or stage_post
        add(feature, "post_outcome", "fail" if post else "pass", "post_outcome, availability_stage", "The feature is declared as post-outcome." if post else "The feature is not declared as post-outcome.", "Remove variables unavailable before the outcome." if post else "None.")
        add(feature, "identifier", "fail" if bool(row.identifier) else "pass", "identifier", "The feature is declared as an identifier." if bool(row.identifier) else "The feature is not declared as an identifier.", "Remove identifiers and row-location variables from predictors." if bool(row.identifier) else "None.")
        scope_unknown = row.preprocessing_scope == "unknown"
        add(feature, "preprocessing_scope_declared", "review" if scope_unknown else "pass", "preprocessing_scope", "The preprocessing estimation scope is unknown." if scope_unknown else "The preprocessing estimation scope is declared.", "Declare where data-dependent preprocessing was estimated." if scope_unknown else "None.")
        fold = row.fold_local_required
        fold_missing = pd.isna(fold)
        add(feature, "fold_local_requirement_declared", "review" if fold_missing else "pass", "fold_local_required", "The fold-local preprocessing requirement is unknown." if fold_missing else "The fold-local preprocessing requirement is declared.", "Declare whether preprocessing must be estimated inside each resampling fold." if fold_missing else "None.")
        compatible: bool | None = None if fold_missing else (row.preprocessing_scope == "resampling_fold" if bool(fold) else True)
        add(feature, "preprocessing_scope_compatible", "review" if compatible is None else "pass" if compatible else "fail", "preprocessing_scope, fold_local_required", "Preprocessing compatibility cannot be assessed without a fold-local requirement." if compatible is None else "The declared preprocessing scope is compatible." if compatible else "Fold-local estimation is required, but preprocessing is not declared at resampling-fold scope.", "Declare the fold-local preprocessing requirement." if compatible is None else "None." if compatible else "Estimate preprocessing separately inside each resampling fold.")
        consistent = bool(row.post_outcome) == stage_post
        add(feature, "post_outcome_metadata_consistent", "pass" if consistent else "review", "post_outcome, availability_stage", "Post-outcome declarations are internally consistent." if consistent else "Post-outcome declarations are internally inconsistent.", "None." if consistent else "Reconcile `post_outcome` with `availability_stage`.")
        availability_consistent = not (not pred_missing and bool(pred) and post)
        add(feature, "availability_metadata_consistent", "pass" if availability_consistent else "fail", "prediction_time_available, post_outcome, availability_stage", "Prediction-time availability metadata is consistent." if availability_consistent else "The feature is marked available at prediction time and also marked post-outcome.", "None." if availability_consistent else "Correct the availability declarations and remove unsafe features.")
    checks = pd.DataFrame(rows)
    issues = checks.loc[checks.status != "pass"].reset_index(drop=True)
    status = "fail" if (checks.status == "fail").any() else "review" if (checks.status == "review").any() else "pass"
    summary = pd.DataFrame({"status": ["pass", "review", "fail"], "n_checks": [(checks.status == s).sum() for s in ["pass", "review", "fail"]]})
    return GazepointFeatureManifestValidation(status=status, n_features=len(manifest), summary=summary, checks=checks, issues=issues, manifest=manifest)


def write_gazepoint_feature_manifest_csv(x: Any, file: str | Path, table: str = "manifest", overwrite: bool = False, na: str = "") -> str:
    """Write a feature manifest or validation table to UTF-8 CSV."""
    if not isinstance(file, (str, Path)) or not str(file): raise GP3MLError("`file` must be a single non-empty file path.")
    path = Path(file).expanduser()
    if path.suffix.lower() != ".csv": raise GP3MLError("`file` must use a .csv extension.")
    if table not in {"manifest", "issues", "checks"}: raise GP3MLError("`table` is invalid.")
    if isinstance(x, GazepointFeatureManifestValidation): output = x[table]
    else:
        output = _as_feature_manifest(x)
        if table != "manifest": raise GP3MLError('Plain manifest inputs support only `table = "manifest"`.')
    if not path.parent.exists(): raise GP3MLError(f"Output directory does not exist: {path.parent}.")
    if path.exists() and not overwrite: raise GP3MLError(f"Output file already exists: {path}.")
    output.to_csv(path, index=False, na_rep=na, encoding="utf-8")
    return path.resolve().as_posix()


def _validation_repr(self: GazepointFeatureManifestValidation) -> str:
    return f"<gazepoint_feature_manifest_validation>\nOverall status: {self.status.upper()}\nFeatures: {self.n_features}\nNon-passing checks: {len(self.issues)}\n{self.summary.to_string(index=False)}"
GazepointFeatureManifestValidation.__repr__ = _validation_repr  # type: ignore[method-assign]
