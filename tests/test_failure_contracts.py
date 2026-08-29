from __future__ import annotations

import pandas as pd
import pytest

import gp3mlpy as gp
from gp3mlpy import GP3MLError, OptionalDependencyError


def _data_manifest_task():
    data = gp.simulate_gazepoint_governed_data(n_participants=12, n_stimuli=4, trials_per_cell=1, seed=7)
    predictors = ["tracking_ratio", "blink_rate"]
    manifest = gp.create_gazepoint_synthetic_manifest("assigned_condition", predictors)
    task = gp.declare_gazepoint_task(
        data=data, outcome="assigned_condition", purpose="assigned-condition prediction",
        task_type="classification", participant_id="participant_id", stimulus_id="stimulus_id",
        unit_id="trial_id", generalization_target="new_participants", positive="B",
    )
    return data, predictors, manifest, task


def test_unknown_engine_is_rejected():
    with pytest.raises(GP3MLError, match="Unknown gp3ml engine"):
        gp.assert_gp3ml_engine_available("not_an_engine")


def test_duplicate_manifest_features_are_rejected():
    with pytest.raises(GP3MLError, match="unique"):
        gp.create_gazepoint_feature_manifest(["x1", "x1"])


def test_invalid_manifest_availability_stage_is_rejected():
    with pytest.raises(GP3MLError):
        gp.create_gazepoint_feature_manifest(["x1"], availability_stage="not_a_stage")


def test_invalid_manifest_preprocessing_scope_is_rejected():
    with pytest.raises(GP3MLError):
        gp.create_gazepoint_feature_manifest(["x1"], preprocessing_scope="not_a_scope")


def test_incomplete_manifest_is_rejected():
    with pytest.raises(GP3MLError, match="missing required columns"):
        gp.validate_gazepoint_feature_manifest(pd.DataFrame({"feature": ["x1"]}))


def test_group_folds_require_manifest():
    data, predictors, _, _ = _data_manifest_task()
    with pytest.raises(GP3MLError):
        gp.create_gazepoint_group_folds(
            data, "assigned_condition", predictors, None, "new_participants", participant_id="participant_id", v=3
        )


def test_group_folds_require_group_identifier_for_target():
    data, predictors, manifest, _ = _data_manifest_task()
    with pytest.raises(GP3MLError, match="Required grouping identifiers"):
        gp.create_gazepoint_group_folds(data, "assigned_condition", predictors, manifest, "new_participants", v=3)


def test_group_folds_reject_invalid_fold_count():
    data, predictors, manifest, _ = _data_manifest_task()
    with pytest.raises(GP3MLError, match="finite integers|single integer"):
        gp.create_gazepoint_group_folds(
            data, "assigned_condition", predictors, manifest, "new_participants", participant_id="participant_id", trial_id="trial_id", stimulus_id="stimulus_id", v=1
        )


def test_group_folds_reject_more_folds_than_groups():
    data, predictors, manifest, _ = _data_manifest_task()
    with pytest.raises(GP3MLError):
        gp.create_gazepoint_group_folds(
            data, "assigned_condition", predictors, manifest, "new_participants", participant_id="participant_id", trial_id="trial_id", stimulus_id="stimulus_id", v=99
        )


def test_accuracy_cannot_be_sole_primary_selection_metric():
    # Frozen R governance contract: accuracy alone cannot drive model selection.
    # The function validates the tuning object before metric selection, so use the
    # smallest valid tuning result produced by the public API.
    data, predictors, manifest, task = _data_manifest_task()
    folds = gp.create_gazepoint_group_folds(
        data, "assigned_condition", predictors, manifest, "new_participants", participant_id="participant_id", trial_id="trial_id", stimulus_id="stimulus_id", v=3, seed=2
    )
    grid = gp.create_gazepoint_tuning_grid("glm")
    tuned = gp.tune_gazepoint_model(folds, task, grid, continue_on_error=False)
    with pytest.raises(GP3MLError, match="accuracy"):
        gp.select_gazepoint_model(tuned, metric="accuracy", direction="maximize")


def test_decision_rule_requires_scientific_justification():
    with pytest.raises(GP3MLError):
        gp.create_gazepoint_decision_rule(metric="youden", generalization_target="new_participants")


def test_conformal_level_must_be_valid():
    with pytest.raises(GP3MLError):
        gp.fit_gazepoint_conformal(
            truth=[1.0, 2.0, 3.0], prediction=[1.1, 1.9, 3.1], task_type="regression", level=1.1,
            generalization_target="new_participants",
        )


def test_locked_analysis_plan_is_immutable():
    plan = gp.declare_gazepoint_analysis_plan(
        research_question="Q", scientific_purpose="prediction", outcome="outcome",
        outcome_definition="binary assigned condition", predictors=["x1", "x2"],
        generalization_target="new_participants", grouping_variables=["participant_id"],
        eligible_population="synthetic participants", preprocessing_plan="fold local",
        candidate_models=["glm"], primary_metric="roc_auc", secondary_metrics=["brier"],
        calibration_metric="brier", uncertainty_method="participant bootstrap",
        threshold_policy="predeclared", seed_strategy="fixed",
    )
    locked = gp.lock_gazepoint_analysis_plan(plan)
    with pytest.raises(GP3MLError):
        locked["primary_metric"] = "accuracy"
