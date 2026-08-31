# API map

The compatibility layer contains 127 public exports. This page groups the most important entry points by workflow so you can find the right family quickly. The [complete alphabetical index](reference/index.md) remains available for exact-name lookup, and every compatibility export has its own reference page.

<div class="gp-api-intro" markdown>
<div markdown>
<span class="gp-kicker">Find by research stage</span>
## Start with the decision you are making
Use these fast paths instead of scanning 127 names. Each section pairs the core functions with the scientific role they play.
</div>
<div class="gp-api-jump">
<a href="#1-task-and-use-case-governance"><span>01</span>Task</a>
<a href="#2-feature-provenance-and-leakage"><span>02</span>Provenance</a>
<a href="#3-splitting-grouping-and-resampling"><span>03</span>Resampling</a>
<a href="#4-preprocessing-and-model-fitting"><span>04</span>Fit</a>
<a href="#5-tuning-and-nested-evaluation"><span>05</span>Tuning</a>
<a href="#6-performance-calibration-and-uncertainty"><span>06</span>Metrics</a>
<a href="#7-decision-thresholds-and-abstention"><span>07</span>Decisions</a>
<a href="#8-conformal-prediction"><span>08</span>Conformal</a>
<a href="#9-external-validation-shift-and-robustness"><span>09</span>External</a>
<a href="#10-analysis-plans-and-reproducibility"><span>10</span>Reproducibility</a>
<a href="#11-artifacts-handoffs-and-governance-evidence"><span>11</span>Artifacts</a>
<a href="#12-api-and-object-contracts"><span>12</span>Contracts</a>
</div>
</div>

## 1. Task and use-case governance

| Function | Purpose |
|---|---|
| [`declare_gazepoint_task`](reference/declare_gazepoint_task.md) | Declare outcome, identifiers, scientific workflow, and generalization target. |
| [`assert_gp3ml_use_case`](reference/assert_gp3ml_use_case.md) | Enforce the permitted-use boundary. |
| [`validate_gazepoint_ml_roles`](reference/validate_gazepoint_ml_roles.md) | Validate declared data roles. |
| [`gp3ml_prohibited_uses`](reference/gp3ml_prohibited_uses.md) | Inspect the prohibited-use contract. |

## 2. Feature provenance and leakage

| Function | Purpose |
|---|---|
| [`create_gazepoint_feature_manifest`](reference/create_gazepoint_feature_manifest.md) | Declare predictor provenance and roles. |
| [`validate_gazepoint_feature_manifest`](reference/validate_gazepoint_feature_manifest.md) | Validate a manifest before modelling. |
| [`audit_gazepoint_ml_leakage`](reference/audit_gazepoint_ml_leakage.md) | Audit outcome, partition, identifier, and provenance leakage risks. |
| [`write_gazepoint_ml_leakage_audit_csv`](reference/write_gazepoint_ml_leakage_audit_csv.md) | Export the audit for review. |

## 3. Splitting, grouping, and resampling

| Function | Purpose |
|---|---|
| [`split_gazepoint_ml_data`](reference/split_gazepoint_ml_data.md) | Create a governed analysis/assessment split. |
| [`create_gazepoint_group_folds`](reference/create_gazepoint_group_folds.md) | Create repeated group-aware folds. |
| [`audit_gazepoint_group_folds`](reference/audit_gazepoint_group_folds.md) | Audit group overlap and fold structure. |
| [`diagnose_gazepoint_group_folds`](reference/diagnose_gazepoint_group_folds.md) | Produce detailed fold diagnostics. |
| [`create_gazepoint_nested_folds`](reference/create_gazepoint_nested_folds.md) | Create outer/inner grouped resampling. |
| [`audit_gazepoint_nested_resampling`](reference/audit_gazepoint_nested_resampling.md) | Audit nested-resampling independence. |

## 4. Preprocessing and model fitting

| Function | Purpose |
|---|---|
| [`fit_gazepoint_preprocessor`](reference/fit_gazepoint_preprocessor.md) | Fit analysis-local preprocessing. |
| [`bake_gazepoint_preprocessor`](reference/bake_gazepoint_preprocessor.md) | Apply a fitted preprocessor. |
| [`gp3ml_engine_capabilities`](reference/gp3ml_engine_capabilities.md) | Inspect engine/task/backend capabilities. |
| [`assert_gp3ml_engine_available`](reference/assert_gp3ml_engine_available.md) | Fail explicitly when a requested engine is unavailable. |
| [`fit_gazepoint_model`](reference/fit_gazepoint_model.md) | Fit a governed model. |
| [`train_gazepoint_classifier`](reference/train_gazepoint_classifier.md) | Classification-oriented governed training. |
| [`fit_gazepoint_deep_model`](reference/fit_gazepoint_deep_model.md) | Explicit optional deep-learning path. |
| [`integrate_black_box_model`](reference/integrate_black_box_model.md) | Integrate an external model under declared safety constraints. |

## 5. Tuning and nested evaluation

| Function | Purpose |
|---|---|
| [`create_gazepoint_tuning_grid`](reference/create_gazepoint_tuning_grid.md) | Declare candidate hyperparameters. |
| [`tune_gazepoint_model`](reference/tune_gazepoint_model.md) | Tune within the permitted partition. |
| [`compare_gazepoint_models`](reference/compare_gazepoint_models.md) | Compare declared candidates. |
| [`select_gazepoint_model`](reference/select_gazepoint_model.md) | Select under an explicit metric/direction. |
| [`evaluate_gazepoint_nested_resampling`](reference/evaluate_gazepoint_nested_resampling.md) | Evaluate nested grouped workflows. |
| [`validate_gazepoint_nested_evaluation`](reference/validate_gazepoint_nested_evaluation.md) | Validate the nested-evaluation object. |

## 6. Performance, calibration, and uncertainty

| Function | Purpose |
|---|---|
| [`gazepoint_classification_metrics`](reference/gazepoint_classification_metrics.md) | Classification metrics. |
| [`gazepoint_regression_metrics`](reference/gazepoint_regression_metrics.md) | Regression metrics. |
| [`gazepoint_performance_metrics`](reference/gazepoint_performance_metrics.md) | Task-aware metric dispatch. |
| [`bootstrap_gazepoint_metrics`](reference/bootstrap_gazepoint_metrics.md) | Bootstrap metric uncertainty. |
| [`bootstrap_gazepoint_metrics_by_unit`](reference/bootstrap_gazepoint_metrics_by_unit.md) | Bootstrap at the declared grouping unit. |
| [`fit_gazepoint_calibrator`](reference/fit_gazepoint_calibrator.md) | Fit probability calibration. |
| [`assess_gazepoint_calibration`](reference/assess_gazepoint_calibration.md) | Audit calibration. |
| [`evaluate_gazepoint_group_folds`](reference/evaluate_gazepoint_group_folds.md) | Run grouped resample evaluation. |
| [`summarize_gazepoint_resample_uncertainty`](reference/summarize_gazepoint_resample_uncertainty.md) | Summarize uncertainty across resamples. |

## 7. Decision thresholds and abstention

| Function | Purpose |
|---|---|
| [`evaluate_gazepoint_thresholds`](reference/evaluate_gazepoint_thresholds.md) | Evaluate explicit threshold candidates. |
| [`select_gazepoint_threshold`](reference/select_gazepoint_threshold.md) | Select from candidates under declared rules. |
| [`create_gazepoint_decision_rule`](reference/create_gazepoint_decision_rule.md) | Store threshold, costs, origin, abstention, and justification. |
| [`apply_gazepoint_decision_rule`](reference/apply_gazepoint_decision_rule.md) | Apply a declared rule. |
| [`audit_gazepoint_abstention`](reference/audit_gazepoint_abstention.md) | Audit abstention coverage and covered error. |

## 8. Conformal prediction

| Function | Purpose |
|---|---|
| [`fit_gazepoint_conformal`](reference/fit_gazepoint_conformal.md) | Fit a conformal object under the declared task. |
| [`predict_gazepoint_set`](reference/predict_gazepoint_set.md) | Produce classification prediction sets. |
| [`predict_gazepoint_interval`](reference/predict_gazepoint_interval.md) | Produce regression prediction intervals. |
| [`assess_gazepoint_conformal_coverage`](reference/assess_gazepoint_conformal_coverage.md) | Audit nominal and observed coverage. |
| [`validate_gazepoint_conformal`](reference/validate_gazepoint_conformal.md) | Validate the conformal contract. |

## 9. External validation, shift, and robustness

| Function | Purpose |
|---|---|
| [`declare_gazepoint_external_dataset`](reference/declare_gazepoint_external_dataset.md) | Declare external-dataset provenance and role. |
| [`evaluate_gazepoint_external_transportability`](reference/evaluate_gazepoint_external_transportability.md) | Evaluate transportability. |
| [`validate_gazepoint_transportability`](reference/validate_gazepoint_transportability.md) | Validate transportability evidence. |
| [`audit_gazepoint_dataset_shift`](reference/audit_gazepoint_dataset_shift.md) | Audit predictor-distribution shift. |
| [`audit_gazepoint_missingness_shift`](reference/audit_gazepoint_missingness_shift.md) | Audit missingness shift separately. |
| [`summarize_gazepoint_shift`](reference/summarize_gazepoint_shift.md) | Summarize shift evidence. |
| [`audit_gazepoint_model_robustness`](reference/audit_gazepoint_model_robustness.md) | Combine declared stability/sensitivity diagnostics. |

## 10. Analysis plans and reproducibility

| Function | Purpose |
|---|---|
| [`declare_gazepoint_analysis_plan`](reference/declare_gazepoint_analysis_plan.md) | Declare the modelling plan. |
| [`lock_gazepoint_analysis_plan`](reference/lock_gazepoint_analysis_plan.md) | Lock a plan before outcome-driven changes. |
| [`audit_gazepoint_plan_deviations`](reference/audit_gazepoint_plan_deviations.md) | Compare execution against the plan. |
| [`capture_gazepoint_environment`](reference/capture_gazepoint_environment.md) | Capture package/runtime environment. |
| [`compare_gazepoint_environments`](reference/compare_gazepoint_environments.md) | Compare environments. |
| [`audit_gazepoint_reproducibility`](reference/audit_gazepoint_reproducibility.md) | Audit reproducibility-sensitive artifacts. |
| [`validate_gazepoint_release_checksums`](reference/validate_gazepoint_release_checksums.md) | Validate release artifact checksums. |

## 11. Artifacts, handoffs, and governance evidence

| Function | Purpose |
|---|---|
| [`create_gazepoint_model_card`](reference/create_gazepoint_model_card.md) | Create a model card. |
| [`create_gazepoint_model_artifact`](reference/create_gazepoint_model_artifact.md) | Create an auditable model artifact. |
| [`test_gazepoint_model_portability`](reference/test_gazepoint_model_portability.md) | Test model portability. |
| [`create_gazepoint_handoff`](reference/create_gazepoint_handoff.md) | Create a typed cross-package handoff. |
| [`validate_gazepoint_research_bundle`](reference/validate_gazepoint_research_bundle.md) | Validate an integrated research bundle. |
| [`write_gazepoint_ro_crate`](reference/write_gazepoint_ro_crate.md) | Export RO-Crate metadata. |
| [`create_gp3ml_governance_profile`](reference/create_gp3ml_governance_profile.md) | Assemble governance evidence. |
| [`audit_gp3ml_governance_profile`](reference/audit_gp3ml_governance_profile.md) | Audit control-level evidence. |

## 12. API and object contracts

| Function | Purpose |
|---|---|
| [`gp3ml_api_contracts`](reference/gp3ml_api_contracts.md) | Inspect stable/experimental public exports. |
| [`gp3ml_object_schema`](reference/gp3ml_object_schema.md) | Inspect public object schemas. |
| [`validate_gp3ml_object_contract`](reference/validate_gp3ml_object_contract.md) | Validate an object against its contract. |
| [`audit_gp3ml_api_stability`](reference/audit_gp3ml_api_stability.md) | Audit a frozen API. |
| [`gp3ml_interop_contracts`](reference/gp3ml_interop_contracts.md) | Inspect interoperability contracts. |

## Synthetic fixtures and examples

For tutorials and smoke tests, start with [`simulate_gazepoint_governed_data`](reference/simulate_gazepoint_governed_data.md), [`create_gazepoint_synthetic_task`](reference/create_gazepoint_synthetic_task.md), and [`create_gazepoint_synthetic_manifest`](reference/create_gazepoint_synthetic_manifest.md).

For visual objects, see the [plot gallery](plots.md). For the exact frozen export list, use the [complete alphabetical index](reference/index.md) or `gp3ml_api_contracts()`.
