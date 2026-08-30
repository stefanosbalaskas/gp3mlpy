from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gp3mlpy as gp
from gp3mlpy import nested_resampling as nr
from gp3mlpy import resample_evaluation as reval
from gp3mlpy import resampling as rs
from gp3mlpy import resampling_diagnostics as rd
from gp3mlpy import splitting as sp
from gp3mlpy.exceptions import GP3MLError


PREDICTORS = ["tracking_ratio", "blink_rate", "fixation_duration"]
TARGETS = [
    "new_trials_known_participants",
    "new_participants",
    "new_stimuli",
    "new_participants_and_new_stimuli",
]


def _fixture(target: str = "new_participants"):
    data = gp.simulate_gazepoint_governed_data(
        n_participants=8,
        n_stimuli=4,
        trials_per_cell=2,
        seed=31,
    )
    task = gp.create_gazepoint_synthetic_task(
        data,
        workflow="assigned_condition",
        generalization_target=target,
    )
    manifest = gp.create_gazepoint_synthetic_manifest(task.outcome, PREDICTORS)
    return data, task, manifest


def _split(data, task, manifest, target):
    return sp.split_gazepoint_ml_data(
        data=data,
        outcome=task.outcome,
        predictors=PREDICTORS,
        feature_manifest=manifest,
        generalization_target=target,
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id="stimulus_id",
        assessment_prop=0.25,
        seed=17,
    )


def _folds(data, task, manifest, target="new_participants", repeats=1):
    v = [2, 2] if target == "new_participants_and_new_stimuli" else 2
    return rs.create_gazepoint_group_folds(
        data=data,
        outcome=task.outcome,
        predictors=PREDICTORS,
        feature_manifest=manifest,
        generalization_target=target,
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id="stimulus_id",
        v=v,
        repeats=repeats,
        seed=19,
    )


def test_splitting_helpers_and_all_generalization_targets(tmp_path: Path):
    assert sp._scalar_column(None, "x") is None
    assert sp._scalar_column(" x ", "x") == "x"
    with pytest.raises(GP3MLError, match="non-empty"):
        sp._scalar_column(None, "x", False)
    with pytest.raises(GP3MLError, match="non-empty"):
        sp._scalar_column(" ", "x")

    frame = pd.DataFrame({"id": ["a", "b"]})
    assert sp._group_values(frame, "id", "id").tolist() == ["a", "b"]
    for values in (["a", None], ["a", " "]):
        bad = pd.DataFrame({"id": values})
        with pytest.raises(GP3MLError, match="missing or empty"):
            sp._group_values(bad, "id", "id")
    units = sp._trial_units(np.array(["p", "pp"]), np.array(["t", "tt"]))
    assert units.tolist() == ["1:p|1:t", "2:pp|2:tt"]
    assert sp._holdout_count(2, 0.01) == 1
    assert sp._holdout_count(2, 0.99) == 1
    with pytest.raises(GP3MLError, match="At least two"):
        sp._holdout_count(1, 0.5)

    counts = sp._group_counts(
        pd.DataFrame({"trial": ["t1", "t2"], "stim": ["s1", "s2"]}),
        np.array(["analysis", "assessment"]),
        None,
        "trial",
        "stim",
    )
    assert set(counts.unit) == {"trial", "stimulus"}

    data, _, manifest = _fixture()
    with pytest.raises(GP3MLError, match="feature_manifest"):
        sp._manifest_for_predictors(None, PREDICTORS)
    with pytest.raises(GP3MLError, match="compatible"):
        sp._manifest_for_predictors(pd.DataFrame({"x": [1]}), PREDICTORS)
    with pytest.raises(GP3MLError, match="missing from"):
        sp._manifest_for_predictors(manifest, ["not_present"])

    for target in TARGETS:
        data, task, manifest = _fixture(target)
        split = _split(data, task, manifest, target)
        assert split.validation.status == "pass"
        assert len(split.analysis) > 0 and len(split.assessment) > 0
        if target == "new_participants_and_new_stimuli":
            assert len(split.excluded) > 0
        else:
            assert split.excluded.empty
        paths = gp.write_gazepoint_ml_split_csv(
            split,
            tmp_path / target,
            prefix="split",
            tables=["assignment", "summary"],
        )
        assert set(paths) == {"assignment", "summary"}

    base_data, base_task, base_manifest = _fixture("new_participants")
    invalid = [
        ({"data": [1, 2]}, "data frame"),
        ({"data": base_data.iloc[:1]}, "at least two"),
        ({"outcome": ""}, "outcome"),
        ({"predictors": 2}, "predictors"),
        ({"predictors": []}, "predictors"),
        ({"predictors": [PREDICTORS[0], PREDICTORS[0]]}, "unique"),
        ({"predictors": [base_task.outcome]}, "outcome"),
        ({"predictors": ["participant_id"]}, "Identifier"),
        ({"generalization_target": "bad"}, "generalization_target"),
        ({"assessment_prop": 1}, "strictly between"),
        ({"assessment_prop": np.nan}, "strictly between"),
        ({"seed": True}, "seed"),
        ({"participant_id": None}, "Required grouping"),
        ({"outcome": "missing"}, "missing required"),
    ]
    base = dict(
        data=base_data,
        outcome=base_task.outcome,
        predictors=PREDICTORS,
        feature_manifest=base_manifest,
        generalization_target="new_participants",
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id="stimulus_id",
        assessment_prop=0.25,
        seed=1,
    )
    for changes, match in invalid:
        args = dict(base)
        args.update(changes)
        with pytest.raises(GP3MLError, match=match):
            sp.split_gazepoint_ml_data(**args)

    reserved = base_data.copy()
    reserved[".gp3ml_source_row"] = np.arange(len(reserved))
    with pytest.raises(GP3MLError, match="already contains"):
        sp.split_gazepoint_ml_data(**{**base, "data": reserved})

    sparse = base_data.groupby("participant_id", sort=False).head(1).reset_index(drop=True)
    sparse_task = gp.create_gazepoint_synthetic_task(
        sparse,
        workflow="assigned_condition",
        generalization_target="new_trials_known_participants",
    )
    sparse_manifest = gp.create_gazepoint_synthetic_manifest(sparse_task.outcome, PREDICTORS)
    with pytest.raises(GP3MLError, match="fewer than two"):
        _split(sparse, sparse_task, sparse_manifest, "new_trials_known_participants")


def test_split_validation_corruption_branches():
    data, task, manifest = _fixture("new_participants")
    split = _split(data, task, manifest, "new_participants")
    with pytest.raises(GP3MLError, match="gazepoint_ml_split"):
        sp.validate_gazepoint_ml_split(object())

    missing = split.__deepcopy__({})
    del missing._data["summary"]
    with pytest.raises(GP3MLError, match="missing components"):
        sp.validate_gazepoint_ml_split(missing)

    bad_partition = split.__deepcopy__({})
    bad_partition.analysis = bad_partition.analysis.drop(columns=[".gp3ml_source_row"])
    with pytest.raises(GP3MLError, match="structurally valid"):
        sp.validate_gazepoint_ml_split(bad_partition)

    duplicate = split.__deepcopy__({})
    duplicate.analysis = pd.concat(
        [duplicate.analysis, duplicate.analysis.iloc[[0]]], ignore_index=True
    )
    assert sp.validate_gazepoint_ml_split(duplicate).status == "fail"

    overlap = split.__deepcopy__({})
    overlap.assessment.loc[0, ".gp3ml_source_row"] = overlap.analysis.loc[0, ".gp3ml_source_row"]
    assert sp.validate_gazepoint_ml_split(overlap).status == "fail"

    missing_row = split.__deepcopy__({})
    missing_row.analysis = missing_row.analysis.iloc[1:].reset_index(drop=True)
    assert sp.validate_gazepoint_ml_split(missing_row).status == "fail"

    unexpected_excluded = split.__deepcopy__({})
    unexpected_excluded.excluded = unexpected_excluded.analysis.iloc[[0]].copy()
    assert sp.validate_gazepoint_ml_split(unexpected_excluded).status == "fail"

    bad_manifest = split.__deepcopy__({})
    bad_manifest.feature_manifest_validation.status = "review"
    assert sp.validate_gazepoint_ml_split(bad_manifest).status in {"review", "fail"}

    bad_leakage = split.__deepcopy__({})
    bad_leakage.leakage_audit.status = "review"
    assert sp.validate_gazepoint_ml_split(bad_leakage).status in {"review", "fail"}


def test_resampling_helpers_all_targets_validation_and_writers(tmp_path: Path):
    assert rs._integer(2, "v", 2) == [2]
    assert rs._integer([2, 3], "v", 2) == [2, 3]
    for bad in (True, [], [1], [2, True], "2"):
        with pytest.raises(GP3MLError):
            rs._integer(bad, "v", 2)
    assert rs._v(2, "new_participants") == {"group": 2}
    assert rs._v(2, "new_participants_and_new_stimuli") == {
        "participant": 2,
        "stimulus": 2,
    }
    assert rs._v([2, 3], "new_participants_and_new_stimuli") == {
        "participant": 2,
        "stimulus": 3,
    }
    with pytest.raises(GP3MLError, match="length one or two"):
        rs._v([2, 2, 2], "new_participants_and_new_stimuli")
    with pytest.raises(GP3MLError, match="single integer"):
        rs._v([2, 3], "new_participants")

    row_fold, mapping = rs._assign_groups(
        np.array(["a", "a", "b", "c", "c"]), 2, np.random.default_rng(1)
    )
    assert set(row_fold) == {1, 2}
    assert len(mapping) == 3
    with pytest.raises(GP3MLError, match="distinct groups"):
        rs._assign_groups(np.array(["a", "b"]), 3, np.random.default_rng(1))
    assert rs._fold_id(1, 2) == "Repeat01_Fold02"
    assert rs._fold_id(1, 2, 1, 2) == "Repeat01_P01_S02"

    for target in TARGETS:
        data, task, manifest = _fixture(target)
        folds = _folds(data, task, manifest, target, repeats=2)
        assert folds.validation.status == "pass"
        assert len(folds.folds) == folds.metadata["n_folds_total"]
        assert rs.audit_gazepoint_group_folds(folds).status == "pass"
        assert rs.validate_gazepoint_group_folds(folds).status == "pass"
        assert target in repr(folds)
        paths = rs.write_gazepoint_group_folds_csv(
            folds,
            tmp_path / target,
            prefix="folds",
            tables=["assignments", "fold_summary"],
            include_fold_data=True,
        )
        assert "assignments" in paths
        with pytest.raises(GP3MLError, match="already exist"):
            rs.write_gazepoint_group_folds_csv(
                folds,
                tmp_path / target,
                prefix="folds",
                tables=["assignments", "fold_summary"],
                include_fold_data=True,
            )

    data, task, manifest = _fixture("new_participants")
    with pytest.raises(GP3MLError, match="data frame"):
        rs.create_gazepoint_group_folds(
            [1, 2], task.outcome, PREDICTORS, manifest, "new_participants"
        )
    with pytest.raises(GP3MLError, match="exceeds"):
        rs.create_gazepoint_group_folds(
            data,
            task.outcome,
            PREDICTORS,
            manifest,
            "new_participants",
            participant_id="participant_id",
            v=99,
        )
    with pytest.raises(GP3MLError, match="gazepoint_group_folds"):
        rs.audit_gazepoint_group_folds(object())
    with pytest.raises(GP3MLError, match="gazepoint_group_folds"):
        rs.validate_gazepoint_group_folds(object())
    with pytest.raises(GP3MLError, match="prefix"):
        rs.write_gazepoint_group_folds_csv(
            _folds(data, task, manifest), tmp_path, prefix="bad/name"
        )
    with pytest.raises(GP3MLError, match="tables"):
        rs.write_gazepoint_group_folds_csv(
            _folds(data, task, manifest), tmp_path, tables=["bad"]
        )


def test_fold_diagnostics_categorical_numeric_validation_and_writer(tmp_path: Path):
    data, task, manifest = _fixture("new_participants")
    folds = _folds(data, task, manifest, repeats=2)
    diag = rd.diagnose_gazepoint_group_folds(folds)
    assert diag.validation.status in {"pass", "review"}
    assert diag.metadata["outcome_type"] == "categorical"
    assert "Folds:" in repr(diag)
    assert "Overall status" in repr(diag.validation)
    assert rd._ratio(0, 0) != rd._ratio(0, 0)
    assert np.isinf(rd._ratio(1, 0))
    assert rd._ratio(2, 4) == 0.5
    assert np.isnan(rd._ratio(1, np.nan))
    assert rd._threshold(1.5, "x") == 1.5
    for bad in (True, 0, np.nan, np.inf, "2"):
        with pytest.raises(GP3MLError):
            rd._threshold(bad, "x")
    assert rd._numeric_summary(pd.Series([np.nan]))["n"] == 0
    assert np.isnan(rd._numeric_summary(pd.Series([1.0]))["sd"])
    assert rd._numeric_summary(pd.Series([1.0, 3.0]))["mean"] == 2.0

    paths = rd.write_gazepoint_fold_diagnostics_csv(
        diag,
        tmp_path,
        prefix="diag",
        tables=["fold_metrics", "validation_checks"],
    )
    assert set(paths) == {"fold_metrics", "validation_checks"}
    with pytest.raises(GP3MLError, match="overwrite"):
        rd.write_gazepoint_fold_diagnostics_csv(
            diag,
            tmp_path,
            prefix="diag",
            tables=["fold_metrics"],
        )
    with pytest.raises(GP3MLError, match="diagnostic tables"):
        rd.write_gazepoint_fold_diagnostics_csv(diag, tmp_path, tables=["bad"])
    with pytest.raises(GP3MLError, match="gazepoint_fold_diagnostics"):
        rd.write_gazepoint_fold_diagnostics_csv(object(), tmp_path)
    with pytest.raises(GP3MLError, match="gazepoint_group_folds"):
        rd.diagnose_gazepoint_group_folds(object())
    with pytest.raises(GP3MLError, match="gazepoint_fold_diagnostics"):
        rd.validate_gazepoint_fold_diagnostics(object())
    with pytest.raises(GP3MLError, match="imbalance_fail"):
        rd.diagnose_gazepoint_group_folds(folds, imbalance_review=2, imbalance_fail=1)

    rdata = data.copy()
    rtask = gp.create_gazepoint_synthetic_task(
        rdata,
        workflow="observed_duration",
        generalization_target="new_participants",
    )
    rmanifest = gp.create_gazepoint_synthetic_manifest(rtask.outcome, PREDICTORS)
    rfolds = _folds(rdata, rtask, rmanifest, repeats=1)
    rdiag = rd.diagnose_gazepoint_group_folds(rfolds)
    assert rdiag.metadata["outcome_type"] == "numeric"

    corrupted = diag.__deepcopy__({})
    corrupted.fold_metrics.loc[0, "n_total"] += 1
    corrupted.fold_metrics.loc[0, "fold_id"] = corrupted.fold_metrics.loc[1, "fold_id"]
    corrupted.assessment_coverage.loc[0, "n_assessment"] = 0
    corrupted.group_balance.loc[0, "n_assessment_groups"] = 0
    corrupted.metadata["source_audit_status"] = "review"
    assert rd.validate_gazepoint_fold_diagnostics(corrupted).status in {"review", "fail"}


def test_grouped_evaluation_success_failure_collection_and_summaries(tmp_path: Path):
    data, task, manifest = _fixture("new_participants")
    folds = _folds(data, task, manifest, repeats=1)
    evaluation = reval.evaluate_gazepoint_group_folds(
        folds,
        task,
        predictors=PREDICTORS,
        engine="glm",
        assess_calibration=True,
        calibration_bins=3,
        calibration_bootstrap=2,
        keep_models=True,
        seed=23,
    )
    assert len(evaluation.fold_status) == folds.metadata["n_folds_total"]
    assert evaluation.validation.status in {"pass", "review"}
    assert all(result["model"] is not None for result in evaluation.fold_results)
    assert len(reval.collect_gazepoint_fold_predictions(evaluation, include_failed=False)) > 0
    fold_summary = reval.summarize_gazepoint_resample_performance(evaluation)
    pooled = reval.summarize_gazepoint_resample_performance(
        evaluation, aggregation="pooled_rows"
    )
    assert len(fold_summary.summary) > 0 and len(pooled.summary) > 0
    assert reval._metric_long(pd.DataFrame()).empty
    assert reval._metric_long(pd.DataFrame({"n": [2], "score": [0.5]})).metric.tolist() == [
        "score"
    ]
    prediction, probability = reval._predictions_from_model(
        evaluation.fold_results[0]["model"], folds.folds[next(iter(folds.folds))].assessment, task
    )
    assert len(prediction) == len(probability)

    paths = reval.write_gazepoint_resample_evaluation(evaluation, tmp_path)
    assert "fold_status" in paths

    with pytest.raises(GP3MLError, match="gazepoint_group_folds"):
        reval.evaluate_gazepoint_group_folds(object(), task)
    with pytest.raises(GP3MLError, match="At least one predictor"):
        reval.evaluate_gazepoint_group_folds(folds, task, predictors=[])
    with pytest.raises(GP3MLError, match="declared"):
        reval.evaluate_gazepoint_group_folds(folds, task, predictors=["not_declared"])
    with pytest.raises(GP3MLError, match="must be lists"):
        reval.evaluate_gazepoint_group_folds(folds, task, preprocessor_args=[])
    with pytest.raises(GP3MLError, match="gp3ml_resample_evaluation"):
        reval.collect_gazepoint_fold_predictions(object())
    with pytest.raises(GP3MLError, match="grouped or nested"):
        reval.summarize_gazepoint_resample_performance(object())
    with pytest.raises(GP3MLError, match="Unknown aggregation"):
        reval.summarize_gazepoint_resample_performance(evaluation, aggregation="bad")

    broken = folds.__deepcopy__({})
    first = next(iter(broken.folds.values()))
    first.analysis = first.analysis.drop(columns=[PREDICTORS[0]])
    failed = reval.evaluate_gazepoint_group_folds(
        broken,
        task,
        predictors=PREDICTORS,
        continue_on_error=True,
    )
    assert (failed.fold_status.status == "fail").any()
    with_failed = reval.collect_gazepoint_fold_predictions(failed, include_failed=True)
    assert with_failed.prediction_missing.fillna(False).any()
    with pytest.raises(GP3MLError, match="Fold"):
        reval.evaluate_gazepoint_group_folds(
            broken,
            task,
            predictors=PREDICTORS,
            continue_on_error=False,
        )

    corrupted = evaluation.__deepcopy__({})
    corrupted.generalization_target = "new_stimuli"
    corrupted.predictions.loc[:, "stage"] = "analysis"
    corrupted.fold_status.loc[0, "n_missing_predictions"] = 1
    corrupted.fold_status.loc[0, "status"] = "fail"
    assert reval.validate_gazepoint_resample_evaluation(corrupted).status == "fail"


def test_nested_fold_creation_audit_validation_and_failure_retention():
    data, task, manifest = _fixture("new_participants")
    outer = _folds(data, task, manifest, repeats=1)
    nested = nr.create_gazepoint_nested_folds(outer, inner_v=2, inner_repeats=1, seed=29)
    assert len(nested.folds) == outer.metadata["n_folds_total"]
    assert nested.audit.status == "pass"
    assert nested.validation.status == "pass"
    assert "Outer folds" in repr(nested)
    assert nr.audit_gazepoint_nested_resampling(nested).status == "pass"
    assert nr.validate_gazepoint_nested_folds(nested).status == "pass"

    with pytest.raises(GP3MLError, match="gazepoint_group_folds"):
        nr.create_gazepoint_nested_folds(object())
    for bad in (1, 0, "bad"):
        with pytest.raises(GP3MLError, match="inner_v"):
            nr.create_gazepoint_nested_folds(outer, inner_v=bad)
    with pytest.raises(GP3MLError, match="gp3ml_nested_folds"):
        nr.audit_gazepoint_nested_resampling(object())
    with pytest.raises(GP3MLError, match="gp3ml_nested_folds"):
        nr.validate_gazepoint_nested_folds(object())

    failed = nested.__deepcopy__({})
    failed.folds[0]["inner"] = None
    failed.folds[0]["status"] = "fail"
    failed.folds[0]["error"] = "synthetic inner failure"
    failed.audit = nr.audit_gazepoint_nested_resampling(failed)
    failed.validation = nr.validate_gazepoint_nested_folds(failed)
    assert failed.audit.status == "fail"
    assert failed.validation.status in {"review", "fail"}

    wrong_target = nested.__deepcopy__({})
    wrong_target.folds[0]["inner"].metadata["generalization_target"] = "new_stimuli"
    wrong_target.audit = nr.audit_gazepoint_nested_resampling(wrong_target)
    wrong_target.validation = nr.validate_gazepoint_nested_folds(wrong_target)
    assert wrong_target.validation.status == "fail"
