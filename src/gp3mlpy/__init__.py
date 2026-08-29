"""gp3mlpy: Python port of gp3ml 0.3.0."""
from .exceptions import GP3MLError, OptionalDependencyError
from .task_governance import (
    gp3ml_prohibited_uses, declare_gazepoint_task, assert_gp3ml_use_case, validate_gazepoint_ml_roles,
)
from .feature_provenance import (
    create_gazepoint_feature_manifest, validate_gazepoint_feature_manifest, write_gazepoint_feature_manifest_csv,
)
__version__ = "0.1.0.dev0"
r_reference_version = "0.3.0"

from .leakage import audit_gazepoint_ml_leakage, write_gazepoint_ml_leakage_audit_csv

from .splitting import split_gazepoint_ml_data, validate_gazepoint_ml_split, write_gazepoint_ml_split_csv

from .resampling import create_gazepoint_group_folds, audit_gazepoint_group_folds, validate_gazepoint_group_folds, write_gazepoint_group_folds_csv

from .metrics import (
    gazepoint_classification_metrics,
    gazepoint_regression_metrics,
    gazepoint_performance_metrics,
    bootstrap_gazepoint_metrics,
)
from .preprocessing import fit_gazepoint_preprocessor, bake_gazepoint_preprocessor
from .calibration import fit_gazepoint_calibrator, apply_gazepoint_calibrator, assess_gazepoint_calibration
from .model_engines import (
    gp3ml_available_engines,
    integrate_black_box_model,
    fit_gazepoint_model,
    train_gazepoint_classifier,
)
from .deep_learning import fit_gazepoint_deep_model
from .engine_capabilities import gp3ml_engine_capabilities, assert_gp3ml_engine_available
from .synthetic import simulate_gazepoint_governed_data, create_gazepoint_synthetic_manifest, create_gazepoint_synthetic_task
from .target_uncertainty import bootstrap_gazepoint_metrics_by_unit, summarize_gazepoint_resample_uncertainty, validate_gazepoint_target_uncertainty, write_gazepoint_target_uncertainty
from .resample_evaluation import evaluate_gazepoint_group_folds, collect_gazepoint_fold_predictions, summarize_gazepoint_resample_performance, validate_gazepoint_resample_evaluation, write_gazepoint_resample_evaluation
from .governance_reports import create_gazepoint_model_card, write_gazepoint_model_card, evaluate_external_validation, create_external_validation_report, write_external_validation_report, create_gazepoint_reproducibility_report, write_gazepoint_reproducibility_report
from .external_validation import (
    declare_gazepoint_external_dataset,
    evaluate_gazepoint_external_transportability,
    validate_gazepoint_transportability,
    write_gazepoint_transportability_report,
)
from .resampling_diagnostics import diagnose_gazepoint_group_folds, validate_gazepoint_fold_diagnostics, write_gazepoint_fold_diagnostics_csv
from .model_tuning import create_gazepoint_tuning_grid, tune_gazepoint_model, compare_gazepoint_models, select_gazepoint_model, validate_gazepoint_model_tuning, write_gazepoint_model_tuning
from .nested_resampling import create_gazepoint_nested_folds, audit_gazepoint_nested_resampling, validate_gazepoint_nested_folds, evaluate_gazepoint_nested_resampling, validate_gazepoint_nested_evaluation, write_gazepoint_nested_evaluation

from .roadmap_reporting import (
    create_gazepoint_release_evidence,
    create_gazepoint_release_model_card,
    write_gazepoint_release_model_card,
)

from .decision_governance import (
    apply_gazepoint_decision_rule,
    audit_gazepoint_abstention,
    create_gazepoint_decision_rule,
    evaluate_gazepoint_thresholds,
    select_gazepoint_threshold,
    validate_gazepoint_decision_rule,
)

from .conformal import (
    assess_gazepoint_conformal_coverage,
    fit_gazepoint_conformal,
    predict_gazepoint_interval,
    predict_gazepoint_set,
    validate_gazepoint_conformal,
)

from .dataset_shift import (
    audit_gazepoint_dataset_shift,
    audit_gazepoint_missingness_shift,
    summarize_gazepoint_shift,
)

from .analysis_plan import (audit_gazepoint_plan_deviations, declare_gazepoint_analysis_plan, lock_gazepoint_analysis_plan, validate_gazepoint_analysis_plan, write_gazepoint_analysis_plan)

from .environment import capture_gazepoint_environment, compare_gazepoint_environments, validate_gazepoint_environment

from .reproducibility import audit_gazepoint_reproducibility, normalize_gazepoint_artifact_text, with_gazepoint_reproducible_output, write_gazepoint_reproducibility_audit

from .release_provenance import validate_gazepoint_release_checksums, write_gazepoint_release_checksums

from .governance_profiles import audit_gp3ml_governance_profile, create_gp3ml_governance_profile, write_gp3ml_governance_profile

from .api_contracts import gp3ml_api_contracts, gp3ml_object_schema, validate_gp3ml_object_contract, audit_gp3ml_api_stability, write_gp3ml_api_contracts
from .interoperability import gp3ml_interop_contracts, create_gazepoint_handoff, validate_gazepoint_handoff, combine_gazepoint_handoffs, as_gp3ml_data
from .research_workflow import simulate_gazepoint_research_handoffs, validate_gazepoint_research_bundle
from .model_artifacts import create_gazepoint_model_artifact, restore_gazepoint_model_artifact, validate_gazepoint_model_artifact, test_gazepoint_model_portability
from .robustness import evaluate_gazepoint_seed_stability, evaluate_gazepoint_feature_stability, evaluate_gazepoint_threshold_stability, evaluate_gazepoint_missingness_sensitivity, audit_gazepoint_model_robustness
from .ro_crate import write_gazepoint_ro_crate, validate_gazepoint_ro_crate
from . import plotting as _plotting

# Attach source-derived gp3ml 0.3.0 documentation to every compatibility export.
from ._reference_docs import REFERENCE_DOCS as _REFERENCE_DOCS
for _doc_name, _doc_text in _REFERENCE_DOCS.items():
    _doc_object = globals().get(_doc_name)
    if callable(_doc_object):
        _doc_object.__doc__ = _doc_text
