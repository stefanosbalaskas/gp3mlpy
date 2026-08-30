from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gp3mlpy import engine_capabilities as ec
from gp3mlpy import leakage as lk
from gp3mlpy import plotting as pl
from gp3mlpy.exceptions import GP3MLError
from gp3mlpy.objects import (
    GP3MLAPIStabilityAudit,
    GP3MLAbstentionAudit,
    GP3MLConformalCoverage,
    GP3MLDatasetShiftAudit,
    GP3MLEnvironmentComparison,
    GP3MLGovernanceProfileAudit,
    GP3MLHandoffValidation,
    GP3MLModelArtifactValidation,
    GP3MLModelRobustnessAudit,
    GP3MLPlanDeviationAudit,
    GP3MLROCrateValidation,
    GP3MLReleaseChecksumValidation,
    GP3MLReproducibilityAudit,
    GP3MLResearchBundleValidation,
    GP3MLThresholdEvaluation,
)


def _base_partitions():
    analysis = pd.DataFrame(
        {
            "participant_id": ["p1", "p2"],
            "trial_id": ["t1", "t2"],
            "stimulus_id": ["s1", "s2"],
            "outcome": ["no", "yes"],
            "x": [1.0, 2.0],
            "row_id": ["r1", "r2"],
            "target_feature": [10.0, 20.0],
            "post_feature": [3.0, 4.0],
        }
    )
    assessment = pd.DataFrame(
        {
            "participant_id": ["p3", "p4"],
            "trial_id": ["t3", "t4"],
            "stimulus_id": ["s3", "s4"],
            "outcome": ["yes", "no"],
            "x": [3.0, 4.0],
            "row_id": ["r3", "r4"],
            "target_feature": [30.0, 40.0],
            "post_feature": [5.0, 6.0],
        }
    )
    return analysis, assessment


def test_leakage_private_helpers_and_canonical_signatures():
    lk._column_name("x", "x")
    lk._column_name(None, "x", True)
    with pytest.raises(GP3MLError, match="column name"):
        lk._column_name(None, "x")
    with pytest.raises(GP3MLError, match="column name"):
        lk._column_name("", "x")

    assert lk._column_vector(None, "x") == []
    assert lk._column_vector("x", "x") == ["x"]
    assert lk._column_vector(np.array(["x", "y"]), "x") == ["x", "y"]
    with pytest.raises(GP3MLError, match="character vector"):
        lk._column_vector(1, "x")
    with pytest.raises(GP3MLError, match="at least one"):
        lk._column_vector([], "x", allow_empty=False)
    with pytest.raises(GP3MLError, match="unique"):
        lk._column_vector(["x", "x"], "x")
    with pytest.raises(GP3MLError, match="unique"):
        lk._column_vector([""], "x")

    text = pd.Series(["a", " ", None], dtype=object)
    assert lk._missing_identifier(text).tolist() == [False, True, True]
    numeric = pd.Series([1.0, np.nan])
    assert lk._missing_identifier(numeric).tolist() == [False, True]
    assert lk._identifier_values(text) == ["a"]

    assert lk._canonical_value(np.nan) == ("<NA>",)
    naive = pd.Timestamp("2026-08-30 12:00:00")
    aware = pd.Timestamp("2026-08-30 12:00:00", tz="Europe/Athens")
    assert "2026-08-30T12:00:00" in lk._canonical_value(naive)
    assert isinstance(lk._canonical_value(aware), str)
    assert isinstance(lk._canonical_value(np.datetime64("2026-08-30")), str)
    assert lk._canonical_value(pd.Timedelta(seconds=3)) == 3.0
    assert lk._canonical_value(np.int64(4)) == 4
    assert lk._canonical_value("x") == "x"
    for value in ([1], {"x": 1}, {1}, (1,), np.array([1])):
        with pytest.raises(GP3MLError, match="list or matrix"):
            lk._canonical_value(value)

    frame = pd.DataFrame({"p": ["a", "b"], "t": ["1", "2"], "v": [1, 2]})
    assert len(lk._row_signatures(frame, ["p", "v"])) == 2
    assert len(lk._trial_signatures(frame, "t", "p")) == 2
    incomplete = frame.copy()
    incomplete["p"] = [None, None]
    assert lk._trial_signatures(incomplete, "t", "p") == []
    summary = lk._partition_summary(frame, frame, None, "t", None)
    assert summary.n_participants.isna().all() and summary.n_stimuli.isna().all()


def test_leakage_input_guards_and_predictor_leakage_branches():
    analysis, assessment = _base_partitions()
    with pytest.raises(GP3MLError, match="invalid"):
        lk.audit_gazepoint_ml_leakage(analysis, assessment, "outcome", ["x"], generalization_target="bad")
    with pytest.raises(GP3MLError, match="data frames"):
        lk.audit_gazepoint_ml_leakage([], assessment, "outcome", ["x"])
    with pytest.raises(GP3MLError, match="at least one row"):
        lk.audit_gazepoint_ml_leakage(analysis.iloc[:0], assessment, "outcome", ["x"])

    duplicated_columns = analysis.copy()
    duplicated_columns.insert(0, "dup", [1, 2])
    duplicated_columns.columns = ["x", *duplicated_columns.columns[1:]]
    with pytest.raises(GP3MLError, match="unique"):
        lk.audit_gazepoint_ml_leakage(duplicated_columns, duplicated_columns.copy(), "outcome", ["x"])

    with pytest.raises(GP3MLError, match="same column"):
        lk.audit_gazepoint_ml_leakage(analysis, assessment.drop(columns=["post_feature"]), "outcome", ["x"])
    with pytest.raises(GP3MLError, match="distinct"):
        lk.audit_gazepoint_ml_leakage(
            analysis, assessment, "outcome", ["x"], participant_id="participant_id", trial_id="participant_id"
        )
    with pytest.raises(GP3MLError, match="outcome.*identifier"):
        lk.audit_gazepoint_ml_leakage(
            analysis, assessment, "outcome", ["x"], participant_id="outcome"
        )
    with pytest.raises(GP3MLError, match="not found"):
        lk.audit_gazepoint_ml_leakage(analysis, assessment, "outcome", ["missing"])

    audit = lk.audit_gazepoint_ml_leakage(
        analysis,
        assessment,
        "outcome",
        ["outcome", "participant_id", "row_id", "target_feature", "post_feature"],
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id="stimulus_id",
        generalization_target="new_participants",
        target_derived="target_feature",
        post_outcome="post_feature",
    )
    assert audit.status == "fail"
    ids = set(audit.issues.check_id)
    assert {
        "outcome_in_predictors",
        "declared_identifier_in_predictors",
        "identifier_like_predictor_names",
        "target_derived_predictors",
        "post_outcome_predictors",
    }.issubset(ids)


def test_leakage_overlap_duplicate_profile_and_generalization_role_paths():
    analysis, assessment = _base_partitions()
    exact = assessment.copy()
    exact.iloc[0] = analysis.iloc[0]
    overlap = lk.audit_gazepoint_ml_leakage(
        analysis,
        exact,
        "outcome",
        ["x"],
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id="stimulus_id",
        generalization_target="new_participants",
    )
    assert "exact_row_overlap" in set(overlap.issues.check_id)

    dup_analysis = pd.concat([analysis.iloc[[0]], analysis.iloc[[0]]], ignore_index=True)
    duplicate = lk.audit_gazepoint_ml_leakage(
        dup_analysis,
        assessment,
        "outcome",
        ["x"],
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id="stimulus_id",
        generalization_target="new_participants",
    )
    assert "duplicate_rows_within_partitions" in set(duplicate.issues.check_id)

    profile_assessment = assessment.copy()
    profile_assessment["x"] = [1.0, 9.0]
    profile = lk.audit_gazepoint_ml_leakage(
        analysis,
        profile_assessment,
        "outcome",
        ["x"],
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id="stimulus_id",
        generalization_target="new_participants",
    )
    assert "predictor_profile_overlap" in set(profile.issues.check_id)

    missing_participant_role = lk.audit_gazepoint_ml_leakage(
        analysis, assessment, "outcome", ["x"], generalization_target="new_participants"
    )
    assert "participant_id_available" in set(missing_participant_role.issues.check_id)
    missing_participant = analysis.copy()
    missing_participant.loc[0, "participant_id"] = None
    miss_audit = lk.audit_gazepoint_ml_leakage(
        missing_participant,
        assessment,
        "outcome",
        ["x"],
        participant_id="participant_id",
        generalization_target="new_participants",
    )
    assert "participant_id_missing" in set(miss_audit.issues.check_id)

    known_trials_assessment = assessment.copy()
    known_trials_assessment["participant_id"] = ["p1", "p2"]
    known_trials = lk.audit_gazepoint_ml_leakage(
        analysis,
        known_trials_assessment,
        "outcome",
        ["x"],
        participant_id="participant_id",
        trial_id="trial_id",
        generalization_target="new_trials_known_participants",
    )
    assert known_trials.checks.loc[
        known_trials.checks.check_id == "participant_partition_compatibility", "status"
    ].iloc[0] == "pass"
    unseen = known_trials_assessment.copy()
    unseen.loc[0, "participant_id"] = "new"
    unseen_audit = lk.audit_gazepoint_ml_leakage(
        analysis,
        unseen,
        "outcome",
        ["x"],
        participant_id="participant_id",
        trial_id="trial_id",
        generalization_target="new_trials_known_participants",
    )
    assert "participant_partition_compatibility" in set(unseen_audit.issues.check_id)

    participant_overlap = assessment.copy()
    participant_overlap["participant_id"] = ["p1", "p4"]
    overlap_audit = lk.audit_gazepoint_ml_leakage(
        analysis,
        participant_overlap,
        "outcome",
        ["x"],
        participant_id="participant_id",
        generalization_target="new_participants",
    )
    assert "participant_partition_compatibility" in set(overlap_audit.issues.check_id)

    stimulus_target = lk.audit_gazepoint_ml_leakage(
        analysis,
        assessment,
        "outcome",
        ["x"],
        participant_id="participant_id",
        stimulus_id="stimulus_id",
        generalization_target="new_stimuli",
    )
    assert stimulus_target.checks.loc[
        stimulus_target.checks.check_id == "participant_partition_compatibility", "status"
    ].iloc[0] == "pass"


def test_leakage_trial_and_stimulus_missing_overlap_and_writer(tmp_path: Path):
    analysis, assessment = _base_partitions()
    missing_trial_role = lk.audit_gazepoint_ml_leakage(
        analysis,
        assessment,
        "outcome",
        ["x"],
        participant_id="participant_id",
        generalization_target="new_trials_known_participants",
    )
    assert "trial_id_available" in set(missing_trial_role.issues.check_id)

    trial_missing = analysis.copy()
    trial_missing.loc[0, "trial_id"] = ""
    trial_missing_audit = lk.audit_gazepoint_ml_leakage(
        trial_missing,
        assessment,
        "outcome",
        ["x"],
        participant_id="participant_id",
        trial_id="trial_id",
        generalization_target="new_trials_known_participants",
    )
    assert "trial_id_missing" in set(trial_missing_audit.issues.check_id)

    trial_overlap_assessment = assessment.copy()
    trial_overlap_assessment.loc[0, "participant_id"] = analysis.loc[0, "participant_id"]
    trial_overlap_assessment.loc[0, "trial_id"] = analysis.loc[0, "trial_id"]
    trial_overlap = lk.audit_gazepoint_ml_leakage(
        analysis,
        trial_overlap_assessment,
        "outcome",
        ["x"],
        participant_id="participant_id",
        trial_id="trial_id",
        generalization_target="new_trials_known_participants",
    )
    assert "trial_partition_overlap" in set(trial_overlap.issues.check_id)
    no_participant_trial = lk.audit_gazepoint_ml_leakage(
        analysis.drop(columns=["participant_id"]),
        trial_overlap_assessment.drop(columns=["participant_id"]),
        "outcome",
        ["x"],
        trial_id="trial_id",
        generalization_target="new_stimuli",
    )
    assert "trial_partition_overlap" in set(no_participant_trial.issues.check_id)

    missing_stimulus_role = lk.audit_gazepoint_ml_leakage(
        analysis,
        assessment,
        "outcome",
        ["x"],
        generalization_target="new_stimuli",
    )
    assert "stimulus_id_available" in set(missing_stimulus_role.issues.check_id)
    stimulus_missing = analysis.copy()
    stimulus_missing.loc[0, "stimulus_id"] = None
    stimulus_missing_audit = lk.audit_gazepoint_ml_leakage(
        stimulus_missing,
        assessment,
        "outcome",
        ["x"],
        stimulus_id="stimulus_id",
        generalization_target="new_stimuli",
    )
    assert "stimulus_id_missing" in set(stimulus_missing_audit.issues.check_id)
    stimulus_overlap = assessment.copy()
    stimulus_overlap.loc[0, "stimulus_id"] = analysis.loc[0, "stimulus_id"]
    stimulus_overlap_audit = lk.audit_gazepoint_ml_leakage(
        analysis,
        stimulus_overlap,
        "outcome",
        ["x"],
        stimulus_id="stimulus_id",
        generalization_target="new_stimuli",
    )
    assert "stimulus_partition_compatibility" in set(stimulus_overlap_audit.issues.check_id)

    safe = lk.audit_gazepoint_ml_leakage(
        analysis,
        assessment,
        "outcome",
        ["x"],
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id="stimulus_id",
        generalization_target="new_participants_and_new_stimuli",
    )
    for table in ("issues", "checks", "partition_summary"):
        path = tmp_path / f"{table}.csv"
        assert Path(lk.write_gazepoint_ml_leakage_audit_csv(safe, path, table=table)).exists()
    with pytest.raises(GP3MLError, match="inherit"):
        lk.write_gazepoint_ml_leakage_audit_csv(object(), tmp_path / "x.csv")
    with pytest.raises(GP3MLError, match="table"):
        lk.write_gazepoint_ml_leakage_audit_csv(safe, tmp_path / "x.csv", table="bad")
    with pytest.raises(GP3MLError, match=".csv"):
        lk.write_gazepoint_ml_leakage_audit_csv(safe, tmp_path / "x.txt")
    with pytest.raises(GP3MLError, match="does not exist"):
        lk.write_gazepoint_ml_leakage_audit_csv(safe, tmp_path / "missing" / "x.csv")
    existing = tmp_path / "existing.csv"
    existing.write_text("x", encoding="utf-8")
    with pytest.raises(GP3MLError, match="already exists"):
        lk.write_gazepoint_ml_leakage_audit_csv(safe, existing)
    assert lk.write_gazepoint_ml_leakage_audit_csv(safe, existing, overwrite=True)


def _checks(statuses=("pass", "review", "fail")):
    return pd.DataFrame({"status": list(statuses), "check": [f"c{i}" for i in range(len(statuses))]})


def test_plotting_all_direct_functions_and_error_branches():
    fig, ax = plt.subplots()
    returned_fig, returned_ax = pl._new_axes(ax)
    assert returned_fig is fig and returned_ax is ax
    plt.close(fig)
    fresh_fig, fresh_ax = pl._new_axes()
    assert fresh_ax.figure is fresh_fig
    plt.close(fresh_fig)
    assert pl._status_counts(_checks()) == [1, 1, 1]

    threshold = GP3MLThresholdEvaluation(
        thresholds=pd.DataFrame({"threshold": [0.2, 0.8], "balanced_accuracy": [0.6, 0.7]})
    )
    with pytest.raises(GP3MLError, match="Unknown metric"):
        pl.plot_threshold_evaluation(threshold, "missing")
    plt.close(pl.plot_threshold_evaluation(threshold))

    abstention = GP3MLAbstentionAudit(coverage=0.8, abstention_rate=0.2, covered_error_rate=0.1)
    plt.close(pl.plot_abstention_audit(abstention))
    conformal = GP3MLConformalCoverage(nominal_coverage=0.9, row_coverage=0.88, unit_coverage=None)
    plt.close(pl.plot_conformal_coverage(conformal))
    conformal_unit = GP3MLConformalCoverage(nominal_coverage=0.9, row_coverage=0.88, unit_coverage=0.85)
    plt.close(pl.plot_conformal_coverage(conformal_unit))

    findings = pd.DataFrame(
        {
            "predictor": ["x", "cat"],
            "type": ["numeric", "categorical"],
            "standardized_difference": [0.3, np.nan],
            "distribution_statistic": [np.nan, 0.2],
            "status": ["review", "review"],
        }
    )
    shift = GP3MLDatasetShiftAudit(findings=findings)
    plt.close(pl.plot_dataset_shift_audit(shift))
    with pytest.raises(GP3MLError, match="No predictor shift"):
        pl.plot_dataset_shift_audit(GP3MLDatasetShiftAudit(findings=pd.DataFrame()))

    envcomp = GP3MLEnvironmentComparison(
        packages=pd.DataFrame({"status": ["pass", "review"]}),
        core=pd.DataFrame({"status": ["pass", "review"]}),
    )
    plt.close(pl.plot_environment_comparison(envcomp))

    simple_objects = [
        (GP3MLHandoffValidation(status="review", checks=_checks()), pl.plot_handoff_validation),
        (GP3MLModelArtifactValidation(checks=_checks()), pl.plot_model_artifact_validation),
        (GP3MLResearchBundleValidation(status="pass", checks=_checks()), pl.plot_research_bundle_validation),
        (GP3MLROCrateValidation(checks=_checks()), pl.plot_ro_crate_validation),
    ]
    for obj, fn in simple_objects:
        plt.close(fn(obj))

    api = GP3MLAPIStabilityAudit(
        status="review",
        checks=pd.DataFrame({"check": ["a", "b"], "n_issues": [0, 2], "status": ["pass", "review"]}),
    )
    plt.close(pl.plot_api_stability_audit(api))

    robust = GP3MLModelRobustnessAudit(
        findings=pd.DataFrame({"dimension": ["seed"], "metric": ["roc_auc"], "indicator": [0.1]})
    )
    plt.close(pl.plot_model_robustness_audit(robust))
    with pytest.raises(GP3MLError, match="No robustness"):
        pl.plot_model_robustness_audit(GP3MLModelRobustnessAudit(findings=pd.DataFrame()))

    deviation = GP3MLPlanDeviationAudit(
        deviations=pd.DataFrame({"status": ["pass", "deviation"]})
    )
    plt.close(pl.plot_plan_deviation_audit(deviation))
    release = GP3MLReleaseChecksumValidation(files=pd.DataFrame({"status": ["pass", "fail"]}))
    plt.close(pl.plot_release_checksum_validation(release))
    governance = GP3MLGovernanceProfileAudit(
        framework="gp3ml-native",
        controls=pd.DataFrame({"status": ["pass", "review", "fail"]}),
    )
    plt.close(pl.plot_governance_profile_audit(governance))

    repro_empty = GP3MLReproducibilityAudit(status="pass", summary=pd.DataFrame())
    plt.close(pl.plot_reproducibility_audit(repro_empty))
    repro = GP3MLReproducibilityAudit(
        status="review", summary=pd.DataFrame({"issue": ["timestamp"], "n": [2]})
    )
    plt.close(pl.plot_reproducibility_audit(repro))

    capabilities = ec.gp3ml_engine_capabilities()
    plt.close(pl.plot_engine_capabilities(capabilities))
    invalid_capabilities = pd.DataFrame({"engine": ["glm"], "package_available": [True]})
    with pytest.raises(GP3MLError, match="gp3ml_engine_capabilities"):
        pl.plot_engine_capabilities(invalid_capabilities)


def test_bound_plot_methods_dispatch_to_every_registered_class():
    objects = [
        GP3MLThresholdEvaluation(thresholds=pd.DataFrame({"threshold": [0.5], "balanced_accuracy": [0.8]})),
        GP3MLAbstentionAudit(coverage=1.0, abstention_rate=0.0, covered_error_rate=0.0),
        GP3MLConformalCoverage(nominal_coverage=0.9, row_coverage=0.9, unit_coverage=np.nan),
        GP3MLDatasetShiftAudit(
            findings=pd.DataFrame(
                {
                    "predictor": ["x"],
                    "type": ["numeric"],
                    "standardized_difference": [0.1],
                    "distribution_statistic": [np.nan],
                    "status": ["pass"],
                }
            )
        ),
        GP3MLEnvironmentComparison(
            packages=pd.DataFrame({"status": ["pass"]}), core=pd.DataFrame({"status": ["pass"]})
        ),
        GP3MLHandoffValidation(status="pass", checks=_checks(("pass",))),
        GP3MLModelArtifactValidation(checks=_checks(("pass",))),
        GP3MLResearchBundleValidation(status="pass", checks=_checks(("pass",))),
        GP3MLROCrateValidation(checks=_checks(("pass",))),
        GP3MLAPIStabilityAudit(
            status="pass",
            checks=pd.DataFrame({"check": ["a"], "n_issues": [0], "status": ["pass"]}),
        ),
        GP3MLModelRobustnessAudit(
            findings=pd.DataFrame({"dimension": ["seed"], "metric": ["m"], "indicator": [0.0]})
        ),
        GP3MLPlanDeviationAudit(deviations=pd.DataFrame({"status": ["pass"]})),
        GP3MLReleaseChecksumValidation(files=pd.DataFrame({"status": ["pass"]})),
        GP3MLGovernanceProfileAudit(
            framework="gp3ml-native", controls=pd.DataFrame({"status": ["pass"]})
        ),
        GP3MLReproducibilityAudit(status="pass", summary=pd.DataFrame()),
    ]
    for obj in objects:
        fig = obj.plot()
        assert fig is not None
        plt.close(fig)
