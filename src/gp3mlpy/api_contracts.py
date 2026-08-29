from __future__ import annotations

import inspect
from importlib import metadata
from typing import Any

import numpy as np
import pandas as pd

from ._utils import write_tables, worst_status
from .exceptions import GP3MLError
from .objects import (
    GP3MLAPIContractRegistry,
    GP3MLAPIStabilityAudit,
    GP3MLObject,
    GP3MLObjectContractValidation,
)

_STABLE_EXPORTS = [
    "apply_gazepoint_calibrator", "assert_gp3ml_use_case", "assess_gazepoint_calibration",
    "audit_gazepoint_group_folds", "audit_gazepoint_ml_leakage", "audit_gazepoint_nested_resampling",
    "bake_gazepoint_preprocessor", "bootstrap_gazepoint_metrics", "bootstrap_gazepoint_metrics_by_unit",
    "collect_gazepoint_fold_predictions", "compare_gazepoint_models", "create_external_validation_report",
    "create_gazepoint_feature_manifest", "create_gazepoint_group_folds", "create_gazepoint_model_card",
    "create_gazepoint_nested_folds", "create_gazepoint_release_evidence", "create_gazepoint_release_model_card",
    "create_gazepoint_reproducibility_report", "create_gazepoint_synthetic_manifest", "create_gazepoint_synthetic_task",
    "create_gazepoint_tuning_grid", "declare_gazepoint_external_dataset", "declare_gazepoint_task",
    "diagnose_gazepoint_group_folds", "evaluate_external_validation", "evaluate_gazepoint_external_transportability",
    "evaluate_gazepoint_group_folds", "evaluate_gazepoint_nested_resampling", "fit_gazepoint_calibrator",
    "fit_gazepoint_deep_model", "fit_gazepoint_model", "fit_gazepoint_preprocessor",
    "gazepoint_classification_metrics", "gazepoint_performance_metrics", "gazepoint_regression_metrics",
    "gp3ml_available_engines", "gp3ml_prohibited_uses", "integrate_black_box_model", "select_gazepoint_model",
    "simulate_gazepoint_governed_data", "split_gazepoint_ml_data", "summarize_gazepoint_resample_performance",
    "summarize_gazepoint_resample_uncertainty", "train_gazepoint_classifier", "tune_gazepoint_model",
    "validate_gazepoint_feature_manifest", "validate_gazepoint_fold_diagnostics", "validate_gazepoint_group_folds",
    "validate_gazepoint_ml_roles", "validate_gazepoint_ml_split", "validate_gazepoint_model_tuning",
    "validate_gazepoint_nested_evaluation", "validate_gazepoint_nested_folds", "validate_gazepoint_resample_evaluation",
    "validate_gazepoint_target_uncertainty", "validate_gazepoint_transportability", "write_external_validation_report",
    "write_gazepoint_feature_manifest_csv", "write_gazepoint_fold_diagnostics_csv", "write_gazepoint_group_folds_csv",
    "write_gazepoint_ml_leakage_audit_csv", "write_gazepoint_ml_split_csv", "write_gazepoint_model_card",
    "write_gazepoint_model_tuning", "write_gazepoint_nested_evaluation", "write_gazepoint_release_model_card",
    "write_gazepoint_reproducibility_report", "write_gazepoint_resample_evaluation",
    "write_gazepoint_target_uncertainty", "write_gazepoint_transportability_report",
]

_EXPERIMENTAL_EXPORTS = [
    "gp3ml_api_contracts", "gp3ml_object_schema", "validate_gp3ml_object_contract", "audit_gp3ml_api_stability",
    "write_gp3ml_api_contracts", "gp3ml_interop_contracts", "create_gazepoint_handoff", "validate_gazepoint_handoff",
    "combine_gazepoint_handoffs", "as_gp3ml_data", "normalize_gazepoint_artifact_text",
    "audit_gazepoint_reproducibility", "write_gazepoint_reproducibility_audit", "with_gazepoint_reproducible_output",
    "gp3ml_engine_capabilities", "assert_gp3ml_engine_available", "simulate_gazepoint_research_handoffs",
    "validate_gazepoint_research_bundle", "create_gazepoint_decision_rule", "validate_gazepoint_decision_rule",
    "evaluate_gazepoint_thresholds", "select_gazepoint_threshold", "apply_gazepoint_decision_rule",
    "audit_gazepoint_abstention", "fit_gazepoint_conformal", "predict_gazepoint_interval", "predict_gazepoint_set",
    "assess_gazepoint_conformal_coverage", "validate_gazepoint_conformal", "audit_gazepoint_dataset_shift",
    "audit_gazepoint_missingness_shift", "summarize_gazepoint_shift", "declare_gazepoint_analysis_plan",
    "validate_gazepoint_analysis_plan", "lock_gazepoint_analysis_plan", "audit_gazepoint_plan_deviations",
    "write_gazepoint_analysis_plan", "create_gazepoint_model_artifact", "restore_gazepoint_model_artifact",
    "validate_gazepoint_model_artifact", "test_gazepoint_model_portability", "evaluate_gazepoint_seed_stability",
    "evaluate_gazepoint_feature_stability", "evaluate_gazepoint_threshold_stability",
    "evaluate_gazepoint_missingness_sensitivity", "audit_gazepoint_model_robustness", "capture_gazepoint_environment",
    "compare_gazepoint_environments", "validate_gazepoint_environment", "write_gazepoint_ro_crate",
    "validate_gazepoint_ro_crate", "create_gp3ml_governance_profile", "audit_gp3ml_governance_profile",
    "write_gp3ml_governance_profile", "write_gazepoint_release_checksums", "validate_gazepoint_release_checksums",
]

_STABLE_CLASSES = [
    "gazepoint_feature_manifest_validation", "gazepoint_fold_diagnostics", "gazepoint_fold_diagnostics_validation",
    "gazepoint_group_folds", "gazepoint_group_folds_audit", "gazepoint_group_folds_validation",
    "gazepoint_ml_leakage_audit", "gazepoint_ml_split", "gazepoint_ml_split_validation",
    "gp3ml_calibration_assessment", "gp3ml_external_dataset_declaration", "gp3ml_external_validation",
    "gp3ml_metric_uncertainty", "gp3ml_model", "gp3ml_model_card", "gp3ml_model_selection", "gp3ml_model_tuning",
    "gp3ml_model_tuning_validation", "gp3ml_nested_evaluation", "gp3ml_nested_evaluation_validation",
    "gp3ml_nested_folds", "gp3ml_nested_folds_validation", "gp3ml_nested_resampling_audit", "gp3ml_preprocessor",
    "gp3ml_release_evidence", "gp3ml_release_model_card", "gp3ml_reproducibility_report", "gp3ml_resample_evaluation",
    "gp3ml_resample_evaluation_validation", "gp3ml_resample_performance_summary", "gp3ml_resample_uncertainty",
    "gp3ml_role_validation", "gp3ml_target_uncertainty", "gp3ml_task", "gp3ml_transportability_report",
    "gp3ml_transportability_validation", "gp3ml_tuning_grid", "gp3ml_uncertainty_validation",
]


def _current_exports() -> set[str]:
    import gp3mlpy
    return {name for name in (_STABLE_EXPORTS + _EXPERIMENTAL_EXPORTS) if callable(getattr(gp3mlpy, name, None))}


def gp3ml_api_contracts() -> GP3MLAPIContractRegistry:
    declared = sorted(set(_STABLE_EXPORTS + _EXPERIMENTAL_EXPORTS))
    current = _current_exports()
    exports = pd.DataFrame({
        "name": declared,
        "stability": ["stable" if x in _STABLE_EXPORTS else "experimental" for x in declared],
        "present": [x in current for x in declared],
    })
    classes = pd.DataFrame({
        "class": _STABLE_CLASSES,
        "stability": "stable",
        "schema_policy": "additive_only_within_minor_line",
    })
    policy = pd.DataFrame({
        "contract": ["exported_function_name", "public_s3_class", "function_formals", "named_return_components"],
        "stable_rule": [
            "No removal or rename within the 0.3.x line.",
            "No removal or rename within the 0.3.x line.",
            "Existing arguments retain meaning; new arguments require defaults.",
            "Existing named components retain meaning; additive components are allowed.",
        ],
    })
    try: package_version = metadata.version("gp3mlpy")
    except metadata.PackageNotFoundError: package_version = "development"
    return GP3MLAPIContractRegistry(contract_version="0.3.0", package_version=package_version, exports=exports, classes=classes, policy=policy)


def _rish_class(value: Any) -> str:
    if isinstance(value, GP3MLObject): return value.r_class
    if isinstance(value, pd.DataFrame): return "data.frame"
    if isinstance(value, pd.Series): return "numeric" if pd.api.types.is_numeric_dtype(value) else "character"
    if isinstance(value, str): return "character"
    if isinstance(value, bool): return "logical"
    if isinstance(value, (int, np.integer)): return "integer"
    if isinstance(value, (float, np.floating)): return "numeric"
    if isinstance(value, dict): return "list"
    if isinstance(value, (list, tuple)): return "list"
    return value.__class__.__name__


def _rish_type(value: Any) -> str:
    if isinstance(value, bool): return "logical"
    if isinstance(value, (int, np.integer)): return "integer"
    if isinstance(value, (float, np.floating)): return "double"
    if isinstance(value, str): return "character"
    if isinstance(value, (dict, GP3MLObject, pd.DataFrame)): return "list"
    if isinstance(value, (list, tuple)): return "list"
    if callable(value): return "closure"
    return type(value).__name__


def _rish_length(value: Any) -> int:
    if isinstance(value, (str, int, float, bool, np.generic)) or value is None: return 1
    try: return len(value)
    except TypeError: return 1


def gp3ml_object_schema(x: Any, recursive: bool = False) -> pd.DataFrame:
    def describe(value: Any, path: str) -> dict[str, Any]:
        return {
            "component": path,
            "class": _rish_class(value),
            "typeof": _rish_type(value),
            "length": _rish_length(value),
            "nrow": len(value) if isinstance(value, pd.DataFrame) else np.nan,
            "ncol": len(value.columns) if isinstance(value, pd.DataFrame) else np.nan,
        }
    if isinstance(x, pd.DataFrame) or not isinstance(x, (GP3MLObject, dict, list, tuple)):
        return pd.DataFrame([describe(x, ".")])
    if isinstance(x, GP3MLObject): items = list(x.items())
    elif isinstance(x, dict): items = list(x.items())
    else: items = [(f"[[{i}]]", v) for i, v in enumerate(x, start=1)]
    rows: list[dict[str, Any]] = []
    for key, value in items:
        path = str(key)
        rows.append(describe(value, path))
        if recursive and isinstance(value, (GP3MLObject, dict)):
            nested = value.items()
            for nkey, nvalue in nested:
                rows.append(describe(nvalue, f"{path}${nkey}"))
    return pd.DataFrame(rows)


def validate_gp3ml_object_contract(x: Any, registry: GP3MLAPIContractRegistry | None = None) -> GP3MLObjectContractValidation:
    registry = gp3ml_api_contracts() if registry is None else registry
    classes = [x.r_class] if isinstance(x, GP3MLObject) else [_rish_class(x)]
    registered = [c for c in classes if c in set(registry.classes["class"])]
    # Python mappings necessarily have named unique string keys; list-like objects do not.
    named_list = isinstance(x, (GP3MLObject, dict))
    if named_list:
        names = list(x.keys())
        names_valid = all(bool(str(k)) for k in names) and len(set(map(str, names))) == len(names)
    else:
        names_valid = True
    checks = pd.DataFrame({
        "check": ["registered_public_class", "named_components", "schema_observable"],
        "status": ["pass" if registered else "review", "pass" if names_valid else "fail", "pass"],
        "detail": [
            ", ".join(registered) if registered else "Class is not in the stable public-class registry.",
            "Named components are structurally valid." if names_valid else "List-like public objects require unique non-empty component names.",
            "Schema can be represented by gp3ml_object_schema().",
        ],
    })
    return GP3MLObjectContractValidation(status=worst_status(checks.status), **{"class": classes}, checks=checks, schema=gp3ml_object_schema(x, recursive=True))


def audit_gp3ml_api_stability(registry: GP3MLAPIContractRegistry | None = None) -> GP3MLAPIStabilityAudit:
    import gp3mlpy
    from . import objects as object_module
    registry = gp3ml_api_contracts() if registry is None else registry
    current = {name for name in dir(gp3mlpy) if not name.startswith("_") and callable(getattr(gp3mlpy, name, None))}
    stable = set(registry.exports.loc[registry.exports.stability == "stable", "name"])
    experimental = set(registry.exports.loc[registry.exports.stability == "experimental", "name"])
    missing_stable = sorted(stable - current)
    # Restrict unexpected-export audit to the explicit package-function surface; imported classes are not R namespace exports.
    unexpected = sorted(name for name in current - stable - experimental if inspect.isfunction(getattr(gp3mlpy, name, None)))
    available_classes = {getattr(v, "r_class", None) for v in vars(object_module).values() if inspect.isclass(v)}
    missing_classes = sorted(set(registry.classes["class"]) - available_classes - {"gp3ml_engine", "gp3ml_external_validation_report"})
    checks = pd.DataFrame({
        "check": ["stable_exports_present", "declared_exports_only", "stable_public_classes_present"],
        "status": ["pass" if not missing_stable else "fail", "pass" if not unexpected else "review", "pass" if not missing_classes else "fail"],
        "n_issues": [len(missing_stable), len(unexpected), len(missing_classes)],
    })
    differences = pd.DataFrame(
        [("missing_stable_export", x) for x in missing_stable]
        + [("unexpected_export", x) for x in unexpected]
        + [("missing_stable_class", x) for x in missing_classes],
        columns=["type", "name"],
    )
    return GP3MLAPIStabilityAudit(status=worst_status(checks.status), checks=checks, differences=differences, registry=registry)


def write_gp3ml_api_contracts(registry: GP3MLAPIContractRegistry | None = None, directory: str = ".", prefix: str = "gp3ml_api_contracts", overwrite: bool = False) -> dict[str, str]:
    registry = gp3ml_api_contracts() if registry is None else registry
    if not isinstance(registry, GP3MLAPIContractRegistry):
        raise GP3MLError("`registry` must be created by gp3ml_api_contracts().")
    return write_tables({"exports": registry.exports, "classes": registry.classes, "policy": registry.policy}, directory, prefix, overwrite)
