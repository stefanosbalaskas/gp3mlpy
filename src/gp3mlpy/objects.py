from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from copy import deepcopy
import json
from typing import Any, ClassVar

import pandas as pd

from ._reprs import render_r_print
from .exceptions import GP3MLError


class GP3MLObject(MutableMapping[str, Any]):
    """Dictionary-compatible base for R S3 list objects."""
    r_class: ClassVar[str] = "gp3ml_object"

    def __init__(self, **components: Any) -> None:
        object.__setattr__(self, "_data", dict(components))

    def __getitem__(self, key: str) -> Any: return self._data[key]
    def __setitem__(self, key: str, value: Any) -> None: self._data[key] = value
    def __delitem__(self, key: str) -> None: del self._data[key]
    def __iter__(self) -> Iterator[str]: return iter(self._data)
    def __len__(self) -> int: return len(self._data)

    def __getattribute__(self, name: str) -> Any:
        if not name.startswith("_") and name not in {"r_class"}:
            try:
                data = object.__getattribute__(self, "_data")
                if name in data:
                    return data[name]
            except AttributeError:
                pass
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        try: return self._data[name]
        except KeyError as exc: raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name == "r_class": object.__setattr__(self, name, value)
        else: self._data[name] = value

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    @classmethod
    def from_dict(cls, components: dict[str, Any]) -> "GP3MLObject":
        """Construct an object from named components without changing their meaning."""
        if not isinstance(components, dict):
            raise TypeError("`components` must be a dictionary.")
        return cls(**deepcopy(components))

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize governance metadata to JSON using non-executable representations."""
        def convert(value: Any) -> Any:
            if isinstance(value, GP3MLObject):
                return {key: convert(item) for key, item in value._data.items()}
            if isinstance(value, pd.DataFrame):
                return value.to_dict(orient="records")
            if isinstance(value, pd.Series):
                return value.tolist()
            if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
                try:
                    return value.tolist()
                except (TypeError, ValueError):
                    pass
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return repr(value)

        return json.dumps(convert(self), indent=indent, ensure_ascii=False, sort_keys=True)

    def __deepcopy__(self, memo: dict[int, Any]):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        object.__setattr__(result, "_data", deepcopy(self._data, memo))
        return result

    def __repr__(self) -> str: return render_r_print(self)

    def __str__(self) -> str: return render_r_print(self)


_CLASS_SPECS = {
"GazepointFeatureManifestValidation":"gazepoint_feature_manifest_validation",
"GazepointFoldDiagnostics":"gazepoint_fold_diagnostics",
"GazepointFoldDiagnosticsValidation":"gazepoint_fold_diagnostics_validation",
"GazepointGroupFolds":"gazepoint_group_folds",
"GazepointGroupFoldsAudit":"gazepoint_group_folds_audit",
"GazepointGroupFoldsValidation":"gazepoint_group_folds_validation",
"GazepointMLLeakageAudit":"gazepoint_ml_leakage_audit",
"GazepointMLSplit":"gazepoint_ml_split",
"GazepointMLSplitValidation":"gazepoint_ml_split_validation",
"GP3MLCalibrationAssessment":"gp3ml_calibration_assessment",
"GP3MLExternalDatasetDeclaration":"gp3ml_external_dataset_declaration",
"GP3MLExternalValidation":"gp3ml_external_validation",
"GP3MLMetricUncertainty":"gp3ml_metric_uncertainty",
"GP3MLModel":"gp3ml_model",
"GP3MLModelCard":"gp3ml_model_card",
"GP3MLModelSelection":"gp3ml_model_selection",
"GP3MLModelTuning":"gp3ml_model_tuning",
"GP3MLModelTuningValidation":"gp3ml_model_tuning_validation",
"GP3MLNestedEvaluation":"gp3ml_nested_evaluation",
"GP3MLNestedEvaluationValidation":"gp3ml_nested_evaluation_validation",
"GP3MLNestedFolds":"gp3ml_nested_folds",
"GP3MLNestedFoldsValidation":"gp3ml_nested_folds_validation",
"GP3MLNestedResamplingAudit":"gp3ml_nested_resampling_audit",
"GP3MLPreprocessor":"gp3ml_preprocessor",
"GP3MLReleaseEvidence":"gp3ml_release_evidence",
"GP3MLReleaseModelCard":"gp3ml_release_model_card",
"GP3MLReproducibilityReport":"gp3ml_reproducibility_report",
"GP3MLResampleEvaluation":"gp3ml_resample_evaluation",
"GP3MLResampleEvaluationValidation":"gp3ml_resample_evaluation_validation",
"GP3MLResamplePerformanceSummary":"gp3ml_resample_performance_summary",
"GP3MLResampleUncertainty":"gp3ml_resample_uncertainty",
"GP3MLRoleValidation":"gp3ml_role_validation",
"GP3MLTargetUncertainty":"gp3ml_target_uncertainty",
"GP3MLTask":"gp3ml_task",
"GP3MLTransportabilityReport":"gp3ml_transportability_report",
"GP3MLTransportabilityValidation":"gp3ml_transportability_validation",
"GP3MLTuningGrid":"gp3ml_tuning_grid",
"GP3MLUncertaintyValidation":"gp3ml_uncertainty_validation",
}
for _name, _rclass in _CLASS_SPECS.items():
    globals()[_name] = type(_name, (GP3MLObject,), {"r_class": _rclass})

# Experimental public S3 objects used by v0.3.0.
for _name, _rclass in {
"GP3MLAPIContractRegistry":"gp3ml_api_contract_registry",
"GP3MLAPIStabilityAudit":"gp3ml_api_stability_audit",
"GP3MLObjectContractValidation":"gp3ml_object_contract_validation",
"GP3MLDecisionRule":"gp3ml_decision_rule",
"GP3MLThresholdEvaluation":"gp3ml_threshold_evaluation",
"GP3MLAbstentionAudit":"gp3ml_abstention_audit",
"GP3MLConformal":"gp3ml_conformal_fit",
"GP3MLConformalCoverage":"gp3ml_conformal_coverage",
"GP3MLConformalValidation":"gp3ml_conformal_validation",
"GP3MLDatasetShiftAudit":"gp3ml_dataset_shift_audit",
"GP3MLMissingnessShiftAudit":"gp3ml_missingness_shift_audit",
"GP3MLAnalysisPlan":"gp3ml_analysis_plan",
"GP3MLPlanDeviationAudit":"gp3ml_plan_deviation_audit",
"GP3MLModelArtifact":"gp3ml_model_artifact",
"GP3MLModelArtifactValidation":"gp3ml_model_artifact_validation",
"GP3MLModelPortability":"gp3ml_model_portability_test",
"GP3MLStabilityEvaluation":"gp3ml_stability_evaluation",
"GP3MLThresholdStability":"gp3ml_threshold_stability",
"GP3MLModelRobustnessAudit":"gp3ml_model_robustness_audit",
"GP3MLEnvironment":"gp3ml_environment_record",
"GP3MLEnvironmentComparison":"gp3ml_environment_comparison",
"GP3MLEnvironmentValidation":"gp3ml_environment_validation",
"GP3MLHandoff":"gp3ml_handoff",
"GP3MLHandoffValidation":"gp3ml_handoff_validation",
"GP3MLHandoffBundle":"gp3ml_handoff_bundle",
"GP3MLResearchBundle":"gp3ml_research_bundle",
"GP3MLResearchBundleValidation":"gp3ml_research_bundle_validation",
"GP3MLROCrateValidation":"gp3ml_ro_crate_validation",
"GP3MLGovernanceProfile":"gp3ml_governance_profile",
"GP3MLGovernanceProfileAudit":"gp3ml_governance_profile_audit",
"GP3MLReleaseChecksumValidation":"gp3ml_release_checksum_validation",
"GP3MLReproducibilityAudit":"gp3ml_reproducibility_audit",
"GP3MLEngineCapabilities":"gp3ml_engine_capabilities",
}.items():
    globals()[_name] = type(_name, (GP3MLObject,), {"r_class": _rclass})

# Locked analysis plans are immutable through ordinary Python mutation paths.
_BaseGP3MLAnalysisPlan = globals().get("GP3MLAnalysisPlan")
class GP3MLAnalysisPlan(_BaseGP3MLAnalysisPlan):
    r_class = "gp3ml_analysis_plan"

    def _assert_mutable(self) -> None:
        data = object.__getattribute__(self, "_data")
        if bool(data.get("locked", False)):
            raise GP3MLError("A locked gp3ml analysis plan is immutable; create a new plan to record changes.")

    def __setitem__(self, key: str, value: Any) -> None:
        self._assert_mutable()
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._assert_mutable()
        super().__delitem__(key)

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("_") and name != "r_class":
            self._assert_mutable()
        super().__setattr__(name, value)

# Preserve the R multiple-class contract for release model cards.
_BaseReleaseModelCard = globals().get("GP3MLReleaseModelCard")
class GP3MLReleaseModelCard(GP3MLModelCard):
    r_class = "gp3ml_release_model_card"

class GP3MLEngine(GP3MLObject):
    r_class = "gp3ml_engine"
class GP3MLExternalValidationReport(GP3MLObject):
    r_class = "gp3ml_external_validation_report"

class GP3MLDecisionRuleValidation(GP3MLObject):
    r_class = "gp3ml_decision_rule_validation"

class GP3MLShiftSummary(GP3MLObject):
    r_class = "gp3ml_shift_summary"

class GP3MLAnalysisPlanValidation(GP3MLObject):
    r_class = "gp3ml_analysis_plan_validation"

class GP3MLReleaseChecksumManifest(GP3MLObject):
    r_class = "gp3ml_release_checksum_manifest"

class GP3MLROCrate(GP3MLObject):
    r_class = "gp3ml_ro_crate"
