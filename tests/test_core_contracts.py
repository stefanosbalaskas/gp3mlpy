from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

import gp3mlpy as gp


PREDICTORS = [
    "tracking_ratio",
    "blink_rate",
    "fixation_duration",
    "gaze_dispersion",
    "pupil_change",
]


def fixture(target: str = "new_participants"):
    data = gp.simulate_gazepoint_governed_data(n_participants=18, n_stimuli=4, trials_per_cell=1, seed=17)
    task = gp.create_gazepoint_synthetic_task(data, workflow="recording_quality", generalization_target=target)
    manifest = gp.create_gazepoint_synthetic_manifest(task.outcome, PREDICTORS)
    return data, task, manifest


def test_frozen_api_contract():
    reg = gp.gp3ml_api_contracts()
    assert gp.r_reference_version == "0.3.0"
    assert len(reg.exports) == 127
    assert (reg.exports.stability == "stable").sum() == 71
    assert (reg.exports.stability == "experimental").sum() == 56
    assert reg.exports.present.all()
    assert len(reg.classes) == 38
    assert gp.audit_gp3ml_api_stability(reg).status == "pass"


def test_prohibited_use_boundary():
    prohibited = " ".join(gp.gp3ml_prohibited_uses()).lower()
    assert "emotion" in prohibited
    assert "biometric" in prohibited
    assert "health" in prohibited or "diagnos" in prohibited


def test_synthetic_task_manifest_and_roles():
    data, task, manifest = fixture()
    assert len(data) == 18 * 4
    assert gp.assert_gp3ml_use_case(task, data)
    v = gp.validate_gazepoint_feature_manifest(manifest)
    assert v.status == "pass"
    roles = gp.validate_gazepoint_ml_roles(data, task, PREDICTORS, manifest)
    assert roles.status == "pass"


def test_outcome_cannot_be_predictor():
    data, task, manifest = fixture()
    bad = gp.validate_gazepoint_ml_roles(data, task, [task.outcome] + PREDICTORS, manifest)
    assert bad.status == "fail"


def test_new_participant_split_is_disjoint():
    data, task, manifest = fixture("new_participants")
    split = gp.split_gazepoint_ml_data(
        data, task.outcome, PREDICTORS, manifest, "new_participants",
        participant_id=task.participant_id, trial_id=task.unit_id, stimulus_id=task.stimulus_id,
        assessment_prop=0.25, seed=11,
    )
    assert gp.validate_gazepoint_ml_split(split).status == "pass"
    assert set(split.analysis[task.participant_id]).isdisjoint(set(split.assessment[task.participant_id]))


def test_group_folds_are_valid_and_auditable():
    data, task, manifest = fixture("new_participants")
    folds = gp.create_gazepoint_group_folds(
        data, task.outcome, PREDICTORS, manifest, "new_participants",
        participant_id=task.participant_id, trial_id=task.unit_id, stimulus_id=task.stimulus_id,
        v=3, repeats=1, seed=9,
    )
    assert gp.validate_gazepoint_group_folds(folds).status == "pass"
    assert gp.audit_gazepoint_group_folds(folds).status == "pass"
    diagnostics = gp.diagnose_gazepoint_group_folds(folds)
    assert gp.validate_gazepoint_fold_diagnostics(diagnostics).status == "pass"


def test_leakage_audit_clean_and_invalid_predictor():
    data, task, manifest = fixture("new_participants")
    split = gp.split_gazepoint_ml_data(
        data, task.outcome, PREDICTORS, manifest, "new_participants",
        participant_id=task.participant_id, trial_id=task.unit_id, stimulus_id=task.stimulus_id, seed=3,
    )
    audit = gp.audit_gazepoint_ml_leakage(
        split.analysis, split.assessment, task.outcome, PREDICTORS,
        participant_id=task.participant_id, trial_id=task.unit_id, stimulus_id=task.stimulus_id,
        generalization_target="new_participants",
    )
    assert audit.status == "pass"
    bad = gp.audit_gazepoint_ml_leakage(split.analysis, split.assessment, task.outcome, [task.outcome])
    assert bad.status == "fail"


def test_preprocessing_is_train_then_bake():
    data, task, manifest = fixture()
    pre = gp.fit_gazepoint_preprocessor(data.iloc[:40], PREDICTORS)
    baked = gp.bake_gazepoint_preprocessor(pre, data.iloc[40:])
    assert baked.shape[0] == len(data.iloc[40:])
    assert baked.shape[1] > 0


def test_metrics_and_calibration():
    truth = pd.Series(["no", "yes", "no", "yes", "yes", "no"])
    prob = np.array([0.1, 0.8, 0.2, 0.75, 0.9, 0.3])
    metrics = gp.gazepoint_classification_metrics(truth, prob, positive="yes")
    assert {"roc_auc", "brier"}.issubset(set(metrics.columns))
    cal = gp.fit_gazepoint_calibrator(truth, prob, positive="yes", method="platt")
    out = gp.apply_gazepoint_calibrator(cal, prob)
    assert len(out) == len(prob)
    assessment = gp.assess_gazepoint_calibration(truth, out, positive="yes", bins=3)
    assert np.isfinite(float(assessment.summary.iloc[0]["brier"]))


def test_decision_rule_and_thresholds():
    truth = ["no", "yes", "no", "yes", "yes", "no"]
    prob = [0.1, 0.8, 0.2, 0.75, 0.9, 0.3]
    rule = gp.create_gazepoint_decision_rule(metric="balanced_accuracy", threshold=0.5, scientific_justification="Predeclared observed-outcome classification threshold.", generalization_target="new_trials_known_participants")
    assert gp.validate_gazepoint_decision_rule(rule).status == "pass"
    table = gp.evaluate_gazepoint_thresholds(truth, prob, thresholds=[0.3, 0.5, 0.7], positive="yes")
    assert len(table.thresholds) == 3
    chosen = gp.select_gazepoint_threshold(table, metric="balanced_accuracy", direction="maximize", scientific_justification="Inner-resampling threshold selection for an observed outcome.", generalization_target="new_trials_known_participants")
    classified = gp.apply_gazepoint_decision_rule(chosen, prob, positive="yes", negative="no")
    assert len(classified) == len(prob)


def test_conformal_regression():
    truth = np.arange(20, dtype=float)
    pred = truth + np.sin(truth) * 0.2
    fit = gp.fit_gazepoint_conformal(truth, prediction=pred, task_type="regression", level=0.9, generalization_target="new_trials_known_participants")
    assert gp.validate_gazepoint_conformal(fit).status == "pass"
    intervals = gp.predict_gazepoint_interval(fit, pred[:5])
    assert len(intervals) == 5
    coverage = gp.assess_gazepoint_conformal_coverage(fit, truth[:5], interval=intervals)
    assert 0 <= coverage.row_coverage <= 1


def test_dataset_shift_summary():
    dev = pd.DataFrame({"x": np.arange(20), "z": np.linspace(0, 1, 20)})
    ext = pd.DataFrame({"x": np.arange(20) + 2, "z": np.linspace(.1, 1.1, 20)})
    audit = gp.audit_gazepoint_dataset_shift(dev, ext, predictors=["x", "z"])
    summary = gp.summarize_gazepoint_shift(audit)
    assert len(summary) == 2


def test_locked_analysis_plan_is_immutable():
    plan = gp.declare_gazepoint_analysis_plan(
        research_question="Does observed quality generalize?",
        scientific_purpose="methodological validation",
        outcome="quality_flag", outcome_definition="observed binary label",
        predictors=PREDICTORS, generalization_target="new_participants",
        grouping_variables=["participant_id"], primary_metric="roc_auc",
        eligible_population="Eligible study participants",
        preprocessing_plan="Fold-local median imputation and scaling",
        candidate_models=["glm"], uncertainty_method="participant bootstrap",
        seed_strategy="fixed declared seed",
    )
    assert gp.validate_gazepoint_analysis_plan(plan).status == "pass"
    locked = gp.lock_gazepoint_analysis_plan(plan, plan_id="plan-1")
    assert locked.locked is True
    with pytest.raises(Exception):
        locked["outcome"] = "changed"


def test_environment_contract():
    env = gp.capture_gazepoint_environment(packages=["numpy", "pandas"])
    assert gp.validate_gazepoint_environment(env).status == "pass"
    comp = gp.compare_gazepoint_environments(env, env)
    assert comp.status == "pass"


def test_release_checksum_contract(tmp_path: Path):
    p = tmp_path / "artifact.txt"
    p.write_text("gp3mlpy\n")
    manifest = gp.write_gazepoint_release_checksums([p], tmp_path / "SHA256SUMS.csv")
    assert gp.validate_gazepoint_release_checksums(manifest, directory=tmp_path).status == "pass"


def test_handoff_and_governance_profile():
    data, task, manifest = fixture()
    h = gp.create_gazepoint_handoff(data, source_package="gp3tools", keys=[task.participant_id, task.stimulus_id, task.unit_id], outcome=task.outcome, predictors=PREDICTORS, feature_manifest=manifest)
    assert gp.validate_gazepoint_handoff(h).status == "pass"
    profile = gp.create_gp3ml_governance_profile({"purpose_declared": True, "prohibited_use_screened": True})
    audit = gp.audit_gp3ml_governance_profile(profile)
    assert profile.framework == "gp3ml-native"
    assert audit.status in {"pass", "review", "fail"}


def test_engine_capability_registry():
    caps = gp.gp3ml_engine_capabilities()
    assert "glm" in set(caps.engine)
    assert gp.assert_gp3ml_engine_available("glm")


def test_group_fold_evaluation_smoke():
    data, task, manifest = fixture("new_participants")
    folds = gp.create_gazepoint_group_folds(
        data, task.outcome, PREDICTORS, manifest, "new_participants",
        participant_id=task.participant_id, trial_id=task.unit_id, stimulus_id=task.stimulus_id,
        v=3, repeats=1, seed=17,
    )
    result = gp.evaluate_gazepoint_group_folds(folds, task, PREDICTORS, engine="glm", seed=17)
    assert gp.validate_gazepoint_resample_evaluation(result).status == "pass"
    pred = gp.collect_gazepoint_fold_predictions(result)
    assert len(pred) > 0
    perf = gp.summarize_gazepoint_resample_performance(result)
    assert len(perf.summary) > 0


def test_object_schema_and_typing_marker():
    data, task, manifest = fixture()
    schema = gp.gp3ml_object_schema(task)
    assert "component" in schema.columns
    assert Path(gp.__file__).with_name("py.typed").exists()
