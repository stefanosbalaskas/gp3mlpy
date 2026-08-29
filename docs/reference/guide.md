# API map

The compatibility layer contains 127 public exports. This page groups the most important entry points by workflow so you can find the right family quickly. The [complete alphabetical index](index.md) remains available for exact-name lookup, and every compatibility export has its own reference page.

## 1. Task and use-case governance

| Function | Purpose |
|---|---|
| [`declare_gazepoint_task`](declare_gazepoint_task.md) | Declare outcome, identifiers, scientific workflow, and generalization target. |
| [`assert_gp3ml_use_case`](assert_gp3ml_use_case.md) | Enforce the permitted-use boundary. |
| [`validate_gazepoint_ml_roles`](validate_gazepoint_ml_roles.md) | Validate declared data roles. |
| [`gp3ml_prohibited_uses`](gp3ml_prohibited_uses.md) | Inspect the prohibited-use contract. |

## 2. Feature provenance and leakage

| Function | Purpose |
|---|---|
| [`create_gazepoint_feature_manifest`](create_gazepoint_feature_manifest.md) | Declare predictor provenance and roles. |
| [`validate_gazepoint_feature_manifest`](validate_gazepoint_feature_manifest.md) | Validate a manifest before modelling. |
| [`audit_gazepoint_ml_leakage`](audit_gazepoint_ml_leakage.md) | Audit outcome, partition, identifier, and provenance leakage risks. |
| [`write_gazepoint_ml_leakage_audit_csv`](write_gazepoint_ml_leakage_audit_csv.md) | Export the audit for review. |

## 3. Splitting, grouping, and resampling

| Function | Purpose |
|---|---|
| [`split_gazepoint_ml_data`](split_gazepoint_ml_data.md) | Create a governed analysis/assessment split. |
| [`create_gazepoint_group_folds`](create_gazepoint_group_folds.md) | Create repeated group-aware folds. |
| [`audit_gazepoint_group_folds`](audit_gazepoint_group_folds.md) | Audit group overlap and fold structure. |
| [`diagnose_gazepoint_group_folds`](diagnose_gazepoint_group_folds.md) | Produce detailed fold diagnostics. |
| [`create_gazepoint_nested_folds`](create_gazepoint_nested_folds.md) | Create outer/inner grouped resampling. |
| [`audit_gazepoint_nested_resampling`](audit_gazepoint_nested_resampling.md) | Audit nested-resampling independence. |

## 4. Preprocessing and model fitting

| Function | Purpose |
|---|---|
| [`fit_gazepoint_preprocessor`](fit_gazepoint_preprocessor.md) | Fit analysis-local preprocessing. |
| [`bake_gazepoint_preprocessor`](bake_gazepoint_preprocessor.md) | Apply a fitted preprocessor. |
| [`gp3ml_engine_capabilities`](gp3ml_engine_capabilities.md) | Inspect engine/task/backend capabilities. |
| [`assert_gp3ml_engine_available`](assert_gp3ml_engine_available.md) | Fail explicitly when a requested engine is unavailable. |
| [`fit_gazepoint_model`](fit_gazepoint_model.md) | Fit a governed model. |
| [`train_gazepoint_classifier`](train_gazepoint_classifier.md) | Classification-oriented governed training. |
| [`fit_gazepoint_deep_model`](fit_gazepoint_deep_model.md) | Explicit optional deep-learning path. |
| [`integrate_black_box_model`](integrate_black_box_model.md) | Integrate an external model under declared safety constraints. |

## 5. Tuning and nested evaluation

| Function | Purpose |
|---|---|
| [`create_gazepoint_tuning_grid`](create_gazepoint_tuning_grid.md) | Declare candidate hyperparameters. |
| [`tune_gazepoint_model`](tune_gazepoint_model.md) | Tune within the permitted partition. |
| [`compare_gazepoint_models`](compare_gazepoint_models.md) | Compare declared candidates. |
| [`select_gazepoint_model`](select_gazepoint_model.md) | Select under an explicit metric/direction. |
| [`evaluate_gazepoint_nested_resampling`](evaluate_gazepoint_nested_resampling.md) | Evaluate nested grouped workflows. |
| [`validate_gazepoint_nested_evaluation`](validate_gazepoint_nested_evaluation.md) | Validate the nested-evaluation object. |

## 6. Performance, calibration, and uncertainty

| Function | Purpose |
|---|---|
| [`gazepoint_classification_metrics`](gazepoint_classification_metrics.md) | Classification metrics. |
| [`gazepoint_regression_metrics`](gazepoint_regression_metrics.md) | Regression metrics. |
| [`gazepoint_performance_metrics`](gazepoint_performance_metrics.md) | Task-aware metric dispatch. |
| [`bootstrap_gazepoint_metrics`](bootstrap_gazepoint_metrics.md) | Bootstrap metric uncertainty. |
| [`bootstrap_gazepoint_metrics_by_unit`](bootstrap_gazepoint_metrics_by_unit.md) | Bootstrap at the declared grouping unit. |
| [`fit_gazepoint_calibrator`](fit_gazepoint_calibrator.md) | Fit probability calibration. |
| [`assess_gazepoint_calibration`](assess_gazepoint_calibration.md) | Audit calibration. |
| [`evaluate_gazepoint_group_folds`](evaluate_gazepoint_group_folds.md) | Run grouped resample evaluation. |
| [`summarize_gazepoint_resample_uncertainty`](summarize_gazepoint_resample_uncertainty.md) | Summarize uncertainty across resamples. |

## 7. Decision thresholds and abstention

| Function | Purpose |
|---|---|
| [`evaluate_gazepoint_thresholds`](evaluate_gazepoint_thresholds.md) | Evaluate explicit threshold candidates. |
| [`select_gazepoint_threshold`](select_gazepoint_threshold.md) | Select from candidates under declared rules. |
| [`create_gazepoint_decision_rule`](create_gazepoint_decision_rule.md) | Store threshold, costs, origin, abstention, and justification. |
| [`apply_gazepoint_decision_rule`](apply_gazepoint_decision_rule.md) | Apply a declared rule. |
| [`audit_gazepoint_abstention`](audit_gazepoint_abstention.md) | Audit abstention coverage and covered error. |

## 8. Conformal prediction

| Function | Purpose |
|---|---|
| [`fit_gazepoint_conformal`](fit_gazepoint_conformal.md) | Fit a conformal object under the declared task. |
| [`predict_gazepoint_set`](predict_gazepoint_set.md) | Produce classification prediction sets. |
| [`predict_gazepoint_interval`](predict_gazepoint_interval.md) | Produce regression prediction intervals. |
| [`assess_gazepoint_conformal_coverage`](assess_gazepoint_conformal_coverage.md) | Audit nominal and observed coverage. |
| [`validate_gazepoint_conformal`](validate_gazepoint_conformal.md) | Validate the conformal contract. |

## 9. External validation, shift, and robustness

| Function | Purpose |
|---|---|
| [`declare_gazepoint_external_dataset`](declare_gazepoint_external_dataset.md) | Declare external-dataset provenance and role. |
| [`evaluate_gazepoint_external_transportability`](evaluate_gazepoint_external_transportability.md) | Evaluate transportability. |
| [`validate_gazepoint_transportability`](validate_gazepoint_transportability.md) | Validate transportability evidence. |
| [`audit_gazepoint_dataset_shift`](audit_gazepoint_dataset_shift.md) | Audit predictor-distribution shift. |
| [`audit_gazepoint_missingness_shift`](audit_gazepoint_missingness_shift.md) | Audit missingness shift separately. |
| [`summarize_gazepoint_shift`](summarize_gazepoint_shift.md) | Summarize shift evidence. |
| [`audit_gazepoint_model_robustness`](audit_gazepoint_model_robustness.md) | Combine declared stability/sensitivity diagnostics. |

## 10. Analysis plans and reproducibility

| Function | Purpose |
|---|---|
| [`declare_gazepoint_analysis_plan`](declare_gazepoint_analysis_plan.md) | Declare the modelling plan. |
| [`lock_gazepoint_analysis_plan`](lock_gazepoint_analysis_plan.md) | Lock a plan before outcome-driven changes. |
| [`audit_gazepoint_plan_deviations`](audit_gazepoint_plan_deviations.md) | Compare execution against the plan. |
| [`capture_gazepoint_environment`](capture_gazepoint_environment.md) | Capture package/runtime environment. |
| [`compare_gazepoint_environments`](compare_gazepoint_environments.md) | Compare environments. |
| [`audit_gazepoint_reproducibility`](audit_gazepoint_reproducibility.md) | Audit reproducibility-sensitive artifacts. |
| [`validate_gazepoint_release_checksums`](validate_gazepoint_release_checksums.md) | Validate release artifact checksums. |

## 11. Artifacts, handoffs, and governance evidence

| Function | Purpose |
|---|---|
| [`create_gazepoint_model_card`](create_gazepoint_model_card.md) | Create a model card. |
| [`create_gazepoint_model_artifact`](create_gazepoint_model_artifact.md) | Create an auditable model artifact. |
| [`test_gazepoint_model_portability`](test_gazepoint_model_portability.md) | Test model portability. |
| [`create_gazepoint_handoff`](create_gazepoint_handoff.md) | Create a typed cross-package handoff. |
| [`validate_gazepoint_research_bundle`](validate_gazepoint_research_bundle.md) | Validate an integrated research bundle. |
| [`write_gazepoint_ro_crate`](write_gazepoint_ro_crate.md) | Export RO-Crate metadata. |
| [`create_gp3ml_governance_profile`](create_gp3ml_governance_profile.md) | Assemble governance evidence. |
| [`audit_gp3ml_governance_profile`](audit_gp3ml_governance_profile.md) | Audit control-level evidence. |

## 12. API and object contracts

| Function | Purpose |
|---|---|
| [`gp3ml_api_contracts`](gp3ml_api_contracts.md) | Inspect stable/experimental public exports. |
| [`gp3ml_object_schema`](gp3ml_object_schema.md) | Inspect public object schemas. |
| [`validate_gp3ml_object_contract`](validate_gp3ml_object_contract.md) | Validate an object against its contract. |
| [`audit_gp3ml_api_stability`](audit_gp3ml_api_stability.md) | Audit a frozen API. |
| [`gp3ml_interop_contracts`](gp3ml_interop_contracts.md) | Inspect interoperability contracts. |

## Synthetic fixtures and examples

For tutorials and smoke tests, start with [`simulate_gazepoint_governed_data`](simulate_gazepoint_governed_data.md), [`create_gazepoint_synthetic_task`](create_gazepoint_synthetic_task.md), and [`create_gazepoint_synthetic_manifest`](create_gazepoint_synthetic_manifest.md).

For visual objects, see the [plot gallery](../plots.md). For the exact frozen export list, use the [complete alphabetical index](index.md) or `gp3ml_api_contracts()`.
