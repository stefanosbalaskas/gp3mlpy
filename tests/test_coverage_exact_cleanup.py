from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3mlpy as gp
from gp3mlpy import _reprs as reps
from gp3mlpy import api_contracts as ac
from gp3mlpy import calibration as cal
from gp3mlpy import environment as env
from gp3mlpy import feature_provenance as fp
from gp3mlpy import governance_reports as gr
from gp3mlpy import leakage as lk
from gp3mlpy import metrics as met
from gp3mlpy import model_artifacts as ma
from gp3mlpy import model_engines as me
from gp3mlpy import model_tuning as mt
from gp3mlpy import nested_resampling as nr
from gp3mlpy import preprocessing as pp
from gp3mlpy import reproducibility as repro
from gp3mlpy import resample_evaluation as reval
from gp3mlpy import resampling as rs
from gp3mlpy import resampling_diagnostics as rd
from gp3mlpy import ro_crate as roc
from gp3mlpy import roadmap_reporting as rr
from gp3mlpy import robustness as rob
from gp3mlpy import splitting as sp
from gp3mlpy import task_governance as tg
from gp3mlpy.exceptions import GP3MLError
from gp3mlpy.objects import (
    GazepointFeatureManifestValidation,
    GazepointFoldDiagnostics,
    GazepointGroupFolds,
    GP3MLAnalysisPlan,
    GP3MLCalibrationAssessment,
    GP3MLMetricUncertainty,
    GP3MLModel,
    GP3MLModelArtifact,
    GP3MLModelSelection,
    GP3MLModelTuning,
    GP3MLNestedEvaluation,
    GP3MLObject,
    GP3MLResampleEvaluation,
    GP3MLResampleUncertainty,
    GP3MLStabilityEvaluation,
    GP3MLTargetUncertainty,
    GP3MLTask,
)


PREDICTORS = ["tracking_ratio", "blink_rate", "fixation_duration"]


def _classification_fixture(target: str = "new_participants"):
    data = gp.simulate_gazepoint_governed_data(8, 4, 2, seed=121)
    task = gp.create_gazepoint_synthetic_task(data, "assigned_condition", target)
    manifest = gp.create_gazepoint_synthetic_manifest(task.outcome, PREDICTORS)
    return data, task, manifest


def _folds(target: str = "new_participants"):
    data, task, manifest = _classification_fixture(target)
    folds = gp.create_gazepoint_group_folds(
        data,
        task.outcome,
        PREDICTORS,
        manifest,
        target,
        participant_id=task.participant_id,
        trial_id=task.unit_id,
        stimulus_id=task.stimulus_id,
        v=2 if target != "new_participants_and_new_stimuli" else [2, 2],
        repeats=1,
        seed=122,
    )
    return data, task, manifest, folds


def test_repr_helpers_residual_paths(monkeypatch):
    assert reps._nrow(3) == 0
    table = pd.DataFrame({"a": [1]})
    assert "a" in reps._table_text(table, columns=["missing", "a"])
    assert "a" in reps._table_text(table, columns=["missing"])
    assert reps._status_counts(object()) == (0, 0, 0)
    assert reps._component({"x": 1}, "x") == 1
    assert reps._component({"x": 1}, "missing", 2) == 2

    diag = GP3MLObject()
    diag.r_class = "gazepoint_fold_diagnostics"
    diag.metadata = {"generalization_target": "new_participants", "repeats": 1, "outcome_type": "categorical"}
    diag.validation = {"status": "pass"}
    diag.fold_metrics = pd.DataFrame({"fold": [1]})
    diag.repeat_metrics = pd.DataFrame()
    assert "Diagnostic status" in reps.render_r_print(diag)
    diag.repeat_metrics = pd.DataFrame({"assessment_size_ratio": [np.nan]})
    assert "Maximum assessment-size ratio" not in reps.render_r_print(diag)

    validation = GP3MLObject(status="pass", summary=pd.DataFrame())
    validation.r_class = "gazepoint_fold_diagnostics_validation"
    rendered = reps.render_r_print(validation)
    assert "Pass: 0" in rendered and "Review: 0" in rendered and "Fail: 0" in rendered

    leakage = GP3MLObject(
        status="pass",
        generalization_target="new_participants",
        partition_summary=pd.DataFrame({"other": [1]}),
        issues=pd.DataFrame(),
    )
    leakage.r_class = "gazepoint_ml_leakage_audit"
    assert "0 analysis; 0 assessment" in reps.render_r_print(leakage)

    # Exercise the non-callable branch in the import-time reference-doc attachment loop.
    docs = importlib.import_module("gp3mlpy._reference_docs")
    original = dict(docs.REFERENCE_DOCS)
    docs.REFERENCE_DOCS["definitely_missing_doc_target"] = "coverage"
    try:
        importlib.reload(gp)
    finally:
        docs.REFERENCE_DOCS.clear()
        docs.REFERENCE_DOCS.update(original)
        importlib.reload(gp)


def test_api_environment_feature_and_task_residuals(monkeypatch):
    class CustomValue:
        pass

    custom = CustomValue()
    assert ac._rish_class(custom) == "CustomValue"
    assert ac._rish_type(custom) == "CustomValue"
    assert ac.validate_gp3ml_object_contract({"": 1}).status == "fail"

    real_version = env.metadata.version
    monkeypatch.setattr(
        env.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(env.metadata.PackageNotFoundError(name)),
    )
    assert env._pkg_version("gp3mlpy") == "0.1.0.dev0"
    monkeypatch.setattr(env.metadata, "version", real_version)

    manifest = gp.create_gazepoint_synthetic_manifest("quality_status", ["tracking_ratio"])
    validation = fp.validate_gazepoint_feature_manifest(manifest)
    assert "Overall status" in fp._validation_repr(validation)

    data, task, _ = _classification_fixture()
    assert "type:" in tg._task_repr(task)
    assert met.gazepoint_performance_metrics(task, data[task.outcome], probability=None).shape[0] == 1
    one_class = data.iloc[:4].copy()
    one_class[task.outcome] = pd.Categorical(["A"] * 4, categories=["A", "B"])
    with pytest.raises(GP3MLError, match="two observed classes"):
        met.bootstrap_gazepoint_metrics(task, one_class[task.outcome], probability=[0.2] * 4, bootstrap=2)


def test_calibration_preprocessing_and_model_engine_residuals(monkeypatch):
    monkeypatch.setattr(cal, "_fit_platt", lambda y, p: SimpleNamespace(params=np.array([0.0])))
    summary, bins = cal._calibration_core(np.array([0, 1]), np.array([0.2, 0.8]), 2)
    assert np.isnan(summary.intercept.iloc[0]) and len(bins) == 2

    overflow = pd.DataFrame({"x": [1e308, 1e308]})
    _, values, _ = pp._prepare_raw_frame(
        overflow, predictors=["x"], fit=True, numeric_imputation="mean"
    )
    assert values["x"] == 0.0
    raw_matrix = pp._model_matrix(pd.DataFrame({"cat": ["b", "a", None]}))
    assert {"cata", "catb"}.issubset(raw_matrix.columns)

    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])
    fitted = me._fit_nnet_adapter(x, y, True, 1, {"max_iter": 5, "size": 1})
    assert hasattr(fitted, "predict_proba")

    data, task, _ = _classification_fixture()
    trained = me.train_gazepoint_classifier(data, task, PREDICTORS, engine="glm")
    assert trained.task.task_type == "classification"


def test_model_artifact_exact_validation_and_portability_paths(monkeypatch):
    assert ma._schema(GP3MLObject(predictors=["x"]), None) == {"predictors": ["x"], "classes": None}
    with pytest.raises(GP3MLError, match="missing model predictors"):
        ma._schema(GP3MLObject(predictors=["x"]), pd.DataFrame({"y": [1]}))
    assert ma._safe_model_fingerprint(None) == {"model": None}

    dummy = GP3MLObject(
        predictors=["x"],
        engine="glm",
        task=GP3MLObject(outcome="y"),
        preprocessor=GP3MLObject(columns=["x"]),
        fit=object(),
        training_hash="h",
        training_n=2,
        seed=1,
        threshold=0.5,
    )
    payload = {"model": dummy, "task": dummy.task, "preprocessor": dummy.preprocessor}
    converted = ma._artifact_payload_for_hash(payload)
    assert isinstance(converted["model"], dict) and isinstance(converted["task"], dict)

    class PredictStrings:
        def predict(self, newdata):
            return np.array(["a"] * len(newdata), dtype=object)

    class PredictRaises:
        def predict(self, newdata):
            raise RuntimeError("bad")

    assert ma._predict_for_portability(object(), pd.DataFrame({"x": [1]})) is None
    assert ma._predict_for_portability(PredictRaises(), pd.DataFrame({"x": [1]})) is None
    assert ma._predict_for_portability(PredictStrings(), pd.DataFrame({"x": [1]})).tolist() == ["a"]

    data, task, _ = _classification_fixture()
    model = gp.fit_gazepoint_model(data, task, PREDICTORS, engine="glm")
    artifact = ma.create_gazepoint_model_artifact(model, reference_data=data)
    assert ma.validate_gazepoint_model_artifact(artifact, verify_hash=False).status == "pass"
    assert ma.validate_gazepoint_model_artifact(object()).status == "fail"

    missing = artifact.__deepcopy__({})
    missing.model = None
    missing.task = None
    missing.predictor_schema = None
    missing.metadata = None
    missing.artifact_hash = None
    assert ma.validate_gazepoint_model_artifact(missing).status == "fail"
    with pytest.raises(GP3MLError, match="validation failed"):
        ma.restore_gazepoint_model_artifact(missing)

    tampered = artifact.__deepcopy__({})
    tampered.metadata["engine"] = "tampered"
    assert ma.validate_gazepoint_model_artifact(tampered).status == "fail"

    no_prediction_model = GP3MLObject(predictors=["x"], task=task, preprocessor=None)
    no_prediction_artifact = GP3MLModelArtifact(
        model=no_prediction_model,
        task=task,
        predictor_schema={"predictors": ["x"], "classes": {"x": "float64"}},
        metadata={},
        reference_data=pd.DataFrame({"x": [1.0]}),
    )
    payload2 = no_prediction_artifact.to_dict()
    no_prediction_artifact.artifact_hash = ma._artifact_hash(payload2)
    portability = ma.test_gazepoint_model_portability(no_prediction_artifact)
    assert portability.prediction_equal is None
    fresh = ma.test_gazepoint_model_portability(artifact, fresh_process=True)
    assert fresh.status == "fail" and fresh.fresh_process_ok is False


def _fake_tuning(task, folds, comparison, grid, results=None, selection=None):
    return GP3MLModelTuning(
        grid=grid,
        results=[] if results is None else results,
        comparison=comparison,
        task=task,
        predictors=PREDICTORS,
        folds_metadata=folds.metadata,
        metrics_requested=None,
        seed=1,
        keep_evaluations=False,
        selection=selection,
    )


def test_model_tuning_all_failed_secondary_ties_validation_and_repr(tmp_path: Path, monkeypatch):
    _, task, _, folds = _folds()
    grid = mt.create_gazepoint_tuning_grid("glm", thresholds=0.5)

    fake_eval = SimpleNamespace(
        fold_status=pd.DataFrame(
            {
                "status": ["fail", "fail"],
                "error": ["e1", "e2"],
                "warnings": ["w1", "w1"],
            }
        ),
        metrics=pd.DataFrame(),
        validation=SimpleNamespace(status="review"),
    )
    monkeypatch.setattr(mt, "evaluate_gazepoint_group_folds", lambda **kwargs: fake_eval)
    failed = mt.tune_gazepoint_model(folds, task, grid, predictors=PREDICTORS)
    assert failed.results[0]["status"] == "fail"
    assert "e1" in failed.results[0]["error"] and "w1" in failed.results[0]["warnings"]

    fake_no_error = SimpleNamespace(
        fold_status=pd.DataFrame({"status": ["fail"], "error": [None], "warnings": [None]}),
        metrics=pd.DataFrame(),
        validation=SimpleNamespace(status="review"),
    )
    monkeypatch.setattr(mt, "evaluate_gazepoint_group_folds", lambda **kwargs: fake_no_error)
    default_error = mt.tune_gazepoint_model(folds, task, grid, predictors=PREDICTORS)
    assert default_error.results[0]["error"] == "Every grouped fold failed."

    tie_grid = mt.create_gazepoint_tuning_grid(
        "glm", thresholds=[0.3, 0.5, 0.7], complexity=[3, 2, 1]
    )
    comparison = pd.DataFrame(
        {
            "candidate_id": [
                "candidate_001", "candidate_002", "candidate_003",
                "candidate_001", "candidate_002", "candidate_003",
            ],
            "metric": ["roc_auc"] * 3 + ["brier"] * 3,
            "candidate_status": ["pass"] * 6,
            "success_prop": [1.0] * 6,
            "mean": [0.8, 0.8, 0.8, 0.3, 0.2, 0.25],
        }
    )
    tuning = _fake_tuning(task, folds, comparison, tie_grid)
    selection = mt.select_gazepoint_model(
        tuning,
        "roc_auc",
        "maximize",
        tie_breakers=["accuracy", "missing", "brier"],
        rationale="secondary metric resolves the tie",
    )
    assert selection.candidate_id == "candidate_002"
    assert selection.tie_breakers.metric.tolist() == ["brier"]

    corrupted = _fake_tuning(
        task,
        folds,
        comparison.iloc[:1].copy(),
        tie_grid,
        results=[{"status": "fail"}],
        selection=selection,
    )
    corrupted.folds_metadata = dict(folds.metadata)
    corrupted.folds_metadata["generalization_target"] = "new_stimuli"
    validation = mt.validate_gazepoint_model_tuning(corrupted)
    assert validation.status == "fail"
    assert "no_implicit_selection" in set(validation.issues.check_id)

    no_selection_paths = mt.write_gazepoint_model_tuning(
        failed, tmp_path / "no-selection", selection=None
    )
    assert "selection" in no_selection_paths
    assert "Candidates" in mt._tuning_repr(failed)
    assert "candidate" in mt._grid_repr(grid)
    assert "Candidate:" in mt._selection_repr(selection)


def test_nested_resampling_creation_error_and_validation_paths(monkeypatch):
    _, _, _, folds = _folds()
    broken = folds.__deepcopy__({})
    broken.metadata["n_folds_total"] += 1
    with pytest.raises(GP3MLError, match="pass validation"):
        nr.create_gazepoint_nested_folds(broken, inner_v=2)
    with pytest.raises(GP3MLError, match="inner_v"):
        nr.create_gazepoint_nested_folds(folds, inner_v="bad")
    with pytest.raises(GP3MLError, match="inner_v"):
        nr.create_gazepoint_nested_folds(folds, inner_v=1)
    with pytest.raises(GP3MLError, match="Could not create inner folds"):
        nr.create_gazepoint_nested_folds(folds, inner_v=999, continue_on_error=False)
    retained = nr.create_gazepoint_nested_folds(folds, inner_v=999, continue_on_error=True)
    assert retained.validation.status in {"review", "fail"}
    with pytest.raises(GP3MLError, match="gp3ml_nested_folds"):
        nr.audit_gazepoint_nested_resampling(object())
    with pytest.raises(GP3MLError, match="gp3ml_nested_folds"):
        nr.validate_gazepoint_nested_folds(object())

    damaged = retained.__deepcopy__({})
    damaged.outer_metadata["n_folds_total"] += 1
    assert nr.validate_gazepoint_nested_folds(damaged).status == "fail"


def test_resample_evaluation_guards_summaries_validation_and_writer(tmp_path: Path, monkeypatch):
    data, task, _, folds = _folds()
    assert reval._metric_long(pd.DataFrame({"x": [1, 2]})).empty
    assert reval._metric_long(object()).empty

    with pytest.raises(GP3MLError, match="gazepoint_group_folds"):
        reval.evaluate_gazepoint_group_folds(object(), task)
    broken = folds.__deepcopy__({})
    broken.metadata["n_folds_total"] += 1
    with pytest.raises(GP3MLError, match="must pass validation"):
        reval.evaluate_gazepoint_group_folds(broken, task)

    wrong_outcome = task.__deepcopy__({})
    wrong_outcome.outcome = "quality_status"
    with pytest.raises(GP3MLError, match="outcome does not match"):
        reval.evaluate_gazepoint_group_folds(folds, wrong_outcome)
    wrong_target = task.__deepcopy__({})
    wrong_target.generalization_target = "new_stimuli"
    with pytest.raises(GP3MLError, match="generalization target"):
        reval.evaluate_gazepoint_group_folds(folds, wrong_target)
    with pytest.raises(GP3MLError, match="At least one predictor"):
        reval.evaluate_gazepoint_group_folds(folds, task, predictors=[])
    with pytest.raises(GP3MLError, match="declared in the fold metadata"):
        reval.evaluate_gazepoint_group_folds(folds, task, predictors=["gaze_dispersion"])
    with pytest.raises(GP3MLError, match="preprocessor_args"):
        reval.evaluate_gazepoint_group_folds(folds, task, preprocessor_args=[])
    with pytest.raises(GP3MLError, match="preprocessor_args"):
        reval.evaluate_gazepoint_group_folds(folds, task, engine_args=[])

    evaluation = gp.evaluate_gazepoint_group_folds(folds, task, PREDICTORS, engine="glm")
    assert len(reval.collect_gazepoint_fold_predictions(evaluation, include_failed=False)) == len(
        evaluation.predictions
    )
    with pytest.raises(GP3MLError, match="resample_evaluation"):
        reval.collect_gazepoint_fold_predictions(object())
    with pytest.raises(GP3MLError, match="grouped or nested"):
        reval.summarize_gazepoint_resample_performance(object())
    with pytest.raises(GP3MLError, match="Unknown aggregation"):
        reval.summarize_gazepoint_resample_performance(evaluation, "bad")

    nested = GP3MLNestedEvaluation(
        metrics=evaluation.metrics,
        predictions=evaluation.predictions,
        task=task,
        threshold=0.5,
        generalization_target=task.generalization_target,
        fold_status=evaluation.fold_status,
    )
    with pytest.raises(GP3MLError, match="candidate-specific"):
        reval.summarize_gazepoint_resample_performance(nested, "pooled_rows")
    empty_predictions = evaluation.__deepcopy__({})
    empty_predictions.predictions = pd.DataFrame()
    with pytest.raises(GP3MLError, match="No predictions"):
        reval.summarize_gazepoint_resample_performance(empty_predictions, "pooled_rows")

    rdata = gp.simulate_gazepoint_governed_data(8, 4, 1, seed=129)
    rtask = gp.create_gazepoint_synthetic_task(rdata, "observed_duration", "new_participants")
    rmanifest = gp.create_gazepoint_synthetic_manifest(rtask.outcome, PREDICTORS)
    rfolds = gp.create_gazepoint_group_folds(
        rdata,
        rtask.outcome,
        PREDICTORS,
        rmanifest,
        "new_participants",
        participant_id=rtask.participant_id,
        trial_id=rtask.unit_id,
        stimulus_id=rtask.stimulus_id,
        v=2,
        seed=130,
    )
    regression_eval = gp.evaluate_gazepoint_group_folds(rfolds, rtask, PREDICTORS, engine="lm")
    pooled = reval.summarize_gazepoint_resample_performance(regression_eval, "pooled_rows")
    assert "aggregation_warning" in pooled.summary

    damaged = evaluation.__deepcopy__({})
    damaged.fold_status = damaged.fold_status.iloc[:1].copy()
    damaged.predictions = damaged.predictions.copy()
    damaged.predictions["stage"] = "wrong"
    damaged.fold_status["n_missing_predictions"] = 1
    damaged.fold_status["status"] = "fail"
    damaged.generalization_target = "new_stimuli"
    assert reval.validate_gazepoint_resample_evaluation(damaged).status == "fail"
    with pytest.raises(GP3MLError, match="resample_evaluation"):
        reval.validate_gazepoint_resample_evaluation(object())
    with pytest.raises(GP3MLError, match="resample_evaluation"):
        reval.write_gazepoint_resample_evaluation(object(), tmp_path)


def test_resampling_private_guards_validation_writer_and_repr(tmp_path: Path):
    assert rs._integer(np.int64(2), "v") == [2]
    for value in (None, [], True, [1], [2.2]):
        with pytest.raises(GP3MLError, match="finite integers"):
            rs._integer(value, "v", 2)
    assert rs._v(2, "new_participants_and_new_stimuli") == {"participant": 2, "stimulus": 2}
    with pytest.raises(GP3MLError, match="length one or two"):
        rs._v([2, 2, 2], "new_participants_and_new_stimuli")
    with pytest.raises(GP3MLError, match="single integer"):
        rs._v([2, 2], "new_participants")
    with pytest.raises(GP3MLError, match="distinct groups"):
        rs._assign_groups(np.array(["a", "a"]), 2, np.random.default_rng(1))
    assert rs._fold_id(1, 2, 1, 2) == "Repeat01_P01_S02"
    assert rs._fold_id(1, 2) == "Repeat01_Fold02"

    data, task, manifest = _classification_fixture()
    base = dict(
        data=data,
        outcome=task.outcome,
        predictors=PREDICTORS,
        feature_manifest=manifest,
        generalization_target="new_participants",
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id="stimulus_id",
        v=2,
    )
    cases = [
        ({"data": []}, "data frame"),
        ({"data": data.iloc[:1]}, "at least two rows"),
        ({"predictors": 1}, "predictors"),
        ({"predictors": []}, "predictors"),
        ({"predictors": ["tracking_ratio", "tracking_ratio"]}, "unique"),
        ({"predictors": [task.outcome]}, "outcome"),
        ({"predictors": ["participant_id"]}, "Identifier columns"),
        ({"generalization_target": "bad"}, "generalization_target"),
        ({"repeats": [1, 2]}, "repeats"),
        ({"seed": [1, 2]}, "seed"),
        ({"participant_id": None}, "Required grouping identifiers"),
        ({"data": data.assign(**{".gp3ml_source_row": 1})}, "reserved source-row"),
        ({"predictors": ["missing"]}, "missing required columns"),
        ({"v": 99}, "number of distinct participants"),
    ]
    for changes, match in cases:
        args = dict(base)
        args.update(changes)
        with pytest.raises(GP3MLError, match=match):
            rs.create_gazepoint_group_folds(**args)

    stim_args = dict(base)
    stim_args.update(generalization_target="new_stimuli", participant_id=None, v=99)
    with pytest.raises(GP3MLError, match="distinct stimuli"):
        rs.create_gazepoint_group_folds(**stim_args)
    two_args = dict(base)
    two_args.update(generalization_target="new_participants_and_new_stimuli", v=[99, 2])
    with pytest.raises(GP3MLError, match="Participant"):
        rs.create_gazepoint_group_folds(**two_args)
    two_args["v"] = [2, 99]
    with pytest.raises(GP3MLError, match="Stimulus"):
        rs.create_gazepoint_group_folds(**two_args)

    tiny = data.groupby("participant_id", sort=False).head(1).reset_index(drop=True)
    tiny_manifest = gp.create_gazepoint_synthetic_manifest(task.outcome, PREDICTORS)
    with pytest.raises(GP3MLError, match="distinct participant-trial units"):
        rs.create_gazepoint_group_folds(
            tiny,
            task.outcome,
            PREDICTORS,
            tiny_manifest,
            "new_trials_known_participants",
            participant_id="participant_id",
            trial_id="trial_id",
            stimulus_id="stimulus_id",
            v=2,
        )

    _, _, _, folds = _folds()
    with pytest.raises(GP3MLError, match="gazepoint_group_folds"):
        rs.audit_gazepoint_group_folds(object())
    empty = folds.__deepcopy__({})
    empty.folds = {}
    with pytest.raises(GP3MLError, match="does not contain"):
        rs.audit_gazepoint_group_folds(empty)
    bad_audit = folds.__deepcopy__({})
    first = next(iter(bad_audit.folds.values()))
    first.leakage_audit = None
    with pytest.raises(GP3MLError, match="compatible leakage audit"):
        rs.audit_gazepoint_group_folds(bad_audit)
    with pytest.raises(GP3MLError, match="gazepoint_group_folds"):
        rs.validate_gazepoint_group_folds(object())
    incomplete = GazepointGroupFolds(folds={})
    with pytest.raises(GP3MLError, match="missing components"):
        rs.validate_gazepoint_group_folds(incomplete)

    assert "Total folds" in rs._repr(folds)
    with pytest.raises(GP3MLError, match="gazepoint_group_folds"):
        rs.write_gazepoint_group_folds_csv(object(), tmp_path)
    with pytest.raises(GP3MLError, match="prefix"):
        rs.write_gazepoint_group_folds_csv(folds, tmp_path, prefix="")
    with pytest.raises(GP3MLError, match="directory separators"):
        rs.write_gazepoint_group_folds_csv(folds, tmp_path, prefix="bad/name")
    with pytest.raises(GP3MLError, match="tables"):
        rs.write_gazepoint_group_folds_csv(folds, tmp_path, tables=[])
    with pytest.raises(GP3MLError, match="tables"):
        rs.write_gazepoint_group_folds_csv(folds, tmp_path, tables=["bad"])
    paths = rs.write_gazepoint_group_folds_csv(
        folds, tmp_path / "all", tables="assignments", include_fold_data=True
    )
    assert "assignments" in paths and any("analysis" in key for key in paths)
    with pytest.raises(GP3MLError, match="already exist"):
        rs.write_gazepoint_group_folds_csv(folds, tmp_path / "all", tables="assignments")


def test_resampling_diagnostics_all_guard_and_validation_severity_paths(tmp_path: Path):
    assert np.isnan(rd._ratio(1, np.nan))
    assert np.isnan(rd._ratio(0, 0))
    assert np.isinf(rd._ratio(1, 0))
    assert rd._ratio(3, 2) == 1.5
    for value in (True, "x", np.nan, np.inf, 0.5):
        with pytest.raises(GP3MLError, match="finite numeric"):
            rd._threshold(value, "x")
    empty_summary = rd._numeric_summary(pd.Series([np.nan, "x"]))
    assert empty_summary["n"] == 0
    single = rd._numeric_summary(pd.Series([1.0]))
    assert single["n"] == 1 and np.isnan(single["sd"])

    _, _, _, folds = _folds()
    with pytest.raises(GP3MLError, match="gazepoint_group_folds"):
        rd.diagnose_gazepoint_group_folds(object())
    with pytest.raises(GP3MLError, match="imbalance_fail"):
        rd.diagnose_gazepoint_group_folds(folds, imbalance_review=2, imbalance_fail=1)
    missing = folds.__deepcopy__({})
    del missing["audit"]
    with pytest.raises(GP3MLError, match="missing components"):
        rd.diagnose_gazepoint_group_folds(missing)
    bad_summary = folds.__deepcopy__({})
    bad_summary.fold_summary = bad_summary.fold_summary.drop(columns=["n_total"])
    with pytest.raises(GP3MLError, match="Fold summary"):
        rd.diagnose_gazepoint_group_folds(bad_summary)
    no_outcome = folds.__deepcopy__({})
    no_outcome.metadata = dict(no_outcome.metadata)
    no_outcome.metadata["outcome"] = ""
    with pytest.raises(GP3MLError, match="outcome name"):
        rd.diagnose_gazepoint_group_folds(no_outcome)
    bad_groups = folds.__deepcopy__({})
    bad_groups.group_counts = bad_groups.group_counts.drop(columns=["n_groups"])
    with pytest.raises(GP3MLError, match="Group counts"):
        rd.diagnose_gazepoint_group_folds(bad_groups)
    bad_assignments = folds.__deepcopy__({})
    bad_assignments.assignments = bad_assignments.assignments.drop(columns=["source_row"])
    with pytest.raises(GP3MLError, match="Assignments"):
        rd.diagnose_gazepoint_group_folds(bad_assignments)
    bad_partition = folds.__deepcopy__({})
    first = next(iter(bad_partition.folds.values()))
    first.analysis = first.analysis.drop(columns=[folds.metadata["outcome"]])
    with pytest.raises(GP3MLError, match="outcome column"):
        rd.diagnose_gazepoint_group_folds(bad_partition)

    diag = rd.diagnose_gazepoint_group_folds(folds)
    with pytest.raises(GP3MLError, match="gazepoint_fold_diagnostics"):
        rd.validate_gazepoint_fold_diagnostics(object())
    incomplete = GazepointFoldDiagnostics(fold_metrics=diag.fold_metrics)
    with pytest.raises(GP3MLError, match="missing components"):
        rd.validate_gazepoint_fold_diagnostics(incomplete)

    damaged = diag.__deepcopy__({})
    damaged.fold_metrics = damaged.fold_metrics.copy()
    damaged.fold_metrics.loc[0, "fold_id"] = ""
    damaged.fold_metrics.loc[0, "n_total"] += 1
    damaged.fold_metrics.loc[0, "n_assessment"] = 0
    damaged.assessment_coverage = damaged.assessment_coverage.copy()
    damaged.assessment_coverage.loc[0, "n_assessment"] = 0
    damaged.repeat_metrics = pd.DataFrame({"assessment_size_ratio": [np.inf]})
    damaged.group_balance = damaged.group_balance.copy()
    damaged.group_balance.loc[:, "n_assessment_groups"] = 0
    damaged.outcome_balance = damaged.outcome_balance.copy()
    categorical = damaged.outcome_balance.metric_type == "categorical"
    if categorical.any():
        damaged.outcome_balance.loc[categorical, "n"] = 0
    damaged.fold_metrics.loc[:, "leakage_status"] = "fail"
    damaged.metadata = dict(damaged.metadata)
    damaged.metadata["source_validation_status"] = "review"
    damaged.metadata["source_audit_status"] = "fail"
    assert rd.validate_gazepoint_fold_diagnostics(damaged).status == "fail"

    review = diag.__deepcopy__({})
    review.repeat_metrics = pd.DataFrame({"assessment_size_ratio": [1.75]})
    review.metadata = dict(review.metadata)
    review.metadata["imbalance_review"] = 1.5
    review.metadata["imbalance_fail"] = 2.0
    assert rd.validate_gazepoint_fold_diagnostics(review).status in {"review", "pass"}
    nan_ratio = diag.__deepcopy__({})
    nan_ratio.repeat_metrics = pd.DataFrame({"assessment_size_ratio": [np.nan]})
    rd.validate_gazepoint_fold_diagnostics(nan_ratio)

    assert "Maximum assessment-size ratio" in rd._diag_repr(diag)
    assert "Overall status" in rd._validation_repr(diag.validation)
    with pytest.raises(GP3MLError, match="gazepoint_fold_diagnostics"):
        rd.write_gazepoint_fold_diagnostics_csv(object(), tmp_path)
    for kwargs, match in [
        ({"directory": ""}, "directory"),
        ({"prefix": ""}, "prefix"),
        ({"overwrite": 1}, "overwrite"),
        ({"tables": []}, "tables"),
        ({"tables": [""]}, "tables"),
        ({"tables": ["bad"]}, "Unknown diagnostic"),
    ]:
        args = dict(x=diag, directory=tmp_path / "diag")
        args.update(kwargs)
        with pytest.raises(GP3MLError, match=match):
            rd.write_gazepoint_fold_diagnostics_csv(**args)
    rd.write_gazepoint_fold_diagnostics_csv(diag, tmp_path / "existing", tables="fold_metrics")
    with pytest.raises(GP3MLError, match="overwrite"):
        rd.write_gazepoint_fold_diagnostics_csv(diag, tmp_path / "existing", tables="fold_metrics")


def test_governance_report_helpers_json_model_card_external_and_reproducibility(tmp_path: Path, monkeypatch):
    assert gr._markdown_table(object()) == ["_No rows._"]
    table = pd.DataFrame({"a": [np.nan, "x|y"]})
    rendered = gr._markdown_table(table)
    assert "x\\|y" in rendered[-1]

    p = tmp_path / "x.md"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(GP3MLError, match="File exists"):
        gr._safe_write(p, False)
    assert gr._safe_write(p, True) == p

    assert gr._json_ready(float("inf")) is None
    assert gr._json_ready(np.int64(2)) == 2
    assert gr._json_ready(Path("x")) == "x"
    assert gr._json_ready(pd.Series([1, 2])) == [1, 2]
    assert gr._json_ready(pd.Categorical(["a", "b"])) == ["a", "b"]
    assert gr._json_ready({"x": np.nan}) == {"x": None}
    assert gr._json_ready((1, 2)) == [1, 2]
    assert gr._json_ready(lambda: None) == "<function>"
    assert isinstance(gr._json_ready(object()), str)

    data, task, _ = _classification_fixture()
    model = gp.fit_gazepoint_model(data, task, PREDICTORS, engine="glm")
    with pytest.raises(GP3MLError, match="fitted"):
        gr.create_gazepoint_model_card(object(), "use")
    card = gr.create_gazepoint_model_card(model, "use", limitations="one")
    assert card.limitations == ["one"]
    card_none = gr.create_gazepoint_model_card(model, "use", limitations=[])
    assert "No limitations" in "\n".join(gr._model_card_markdown(card_none))

    uncertainty = GP3MLMetricUncertainty(intervals=pd.DataFrame({"metric": ["roc_auc"], "estimate": [0.8]}))
    card_unc = gr.create_gazepoint_model_card(model, "use", evaluation=uncertainty)
    assert "roc_auc" in "\n".join(gr._model_card_markdown(card_unc))
    card_df = gr.create_gazepoint_model_card(model, "use", evaluation=pd.DataFrame({"m": [1.0]}))
    assert "m" in "\n".join(gr._model_card_markdown(card_df))
    card_empty = gr.create_gazepoint_model_card(model, "use", evaluation=object())
    assert "_No rows._" in "\n".join(gr._model_card_markdown(card_empty))

    with pytest.raises(GP3MLError, match="created"):
        gr.write_gazepoint_model_card(object(), tmp_path / "bad.md")
    with pytest.raises(GP3MLError, match="format"):
        gr.write_gazepoint_model_card(card, tmp_path / "bad", format="yaml")
    assert Path(gr.write_gazepoint_model_card(card, tmp_path / "card.json", format="json")).exists()

    numeric_summary_model = model.__deepcopy__({})
    numeric_summary_model.predictor_summary = dict(model.predictor_summary)
    numeric_summary_model.predictor_summary[PREDICTORS[0]] = {
        "type": "numeric", "mean": 0.0, "sd": 0.0, "missing": 0
    }
    shifted = gr._shift_diagnostics(numeric_summary_model, data)
    assert np.isnan(shifted.loc[shifted.feature == PREDICTORS[0], "standardized_mean_difference"]).all()

    categorical_model = model.__deepcopy__({})
    categorical_model.predictors = ["assigned_condition"]
    categorical_model.predictor_summary = {
        "assigned_condition": {"type": "categorical", "levels": ["A"], "missing": 0}
    }
    categorical_data = data.copy()
    categorical_data["assigned_condition"] = categorical_data["assigned_condition"].astype(str)
    assert "B" in gr._shift_diagnostics(categorical_model, categorical_data).novel_levels.iloc[0]

    validation = gr.evaluate_external_validation(model, data, bootstrap=1)
    with pytest.raises(GP3MLError, match="evaluate_external_validation"):
        gr.create_external_validation_report(object())
    report = gr.create_external_validation_report(validation, limitations="limit")
    assert report.limitations == ["limit"]
    with pytest.raises(GP3MLError, match="external-validation report"):
        gr.write_external_validation_report(object(), tmp_path / "bad-report.md")
    report_no_dev = gr.create_external_validation_report(validation, development_metrics=None, limitations=[])
    text = Path(gr.write_external_validation_report(report_no_dev, tmp_path / "ext.md")).read_text(encoding="utf-8")
    assert "Not supplied." in text

    rdata = gp.simulate_gazepoint_governed_data(8, 4, 1, seed=135)
    rtask = gp.create_gazepoint_synthetic_task(rdata, "observed_duration", "new_participants")
    rmodel = gp.fit_gazepoint_model(rdata, rtask, PREDICTORS, engine="lm")
    rval = gr.evaluate_external_validation(rmodel, rdata, bootstrap=1)
    rreport = gr.create_external_validation_report(rval, development_metrics=pd.DataFrame({"rmse": [1.0]}))
    rtext = Path(gr.write_external_validation_report(rreport, tmp_path / "reg-ext.md")).read_text(encoding="utf-8")
    assert "Not applicable." in rtext

    assert gr._git_info(tmp_path)["commit"] is None

    class GoodResult:
        def __init__(self, value):
            self.stdout = value

    values = iter(["abc\n", "main\n", "\n"])
    monkeypatch.setattr(gr.subprocess, "run", lambda *args, **kwargs: GoodResult(next(values)))
    info = gr._git_info(tmp_path / "fake")
    # No .git means the subprocess branch is intentionally skipped.
    assert info["commit"] is None

    reproducibility = gr.create_gazepoint_reproducibility_report(
        objects={"model": model}, data=data, seeds={"fit": 1}, notes="note", project_path=tmp_path
    )
    assert reproducibility.notes == ["note"]
    with pytest.raises(GP3MLError, match="reproducibility report"):
        gr.write_gazepoint_reproducibility_report(object(), tmp_path / "bad-repro.md")
    empty_repro = gr.create_gazepoint_reproducibility_report(project_path=tmp_path)
    rtext = Path(gr.write_gazepoint_reproducibility_report(empty_repro, tmp_path / "repro.md")).read_text(encoding="utf-8")
    assert "No objects supplied" in rtext and "No seeds supplied" in rtext and "- None." in rtext


def test_repro_rocrate_roadmap_and_robustness_residuals(tmp_path: Path):
    bad_utf = tmp_path / "bad.txt"
    bad_utf.write_bytes(b"\xff\xfe\xfd")
    audit = repro.audit_gazepoint_reproducibility(bad_utf)
    assert audit.files_scanned == 1 and audit.findings.empty

    crate = tmp_path / "crate"
    crate.mkdir()
    (crate / "ro-crate-metadata.json").write_text(
        '{"@context":"https://w3id.org/ro/crate/1.2/context","@graph":[{"@id":"./"}]}',
        encoding="utf-8",
    )
    (crate / "manifest-sha256.csv").write_text("bad,column\nx,y\n", encoding="utf-8")
    assert roc.validate_gazepoint_ro_crate(crate).status == "fail"

    assert rr._as_character_vector(None) == []
    assert rr._as_character_vector("x") == ["x"]
    assert rr._release_card_metrics(GP3MLObject(evaluation=object())).empty
    assert rr._release_card_selection(None).empty
    target_unc = GP3MLTargetUncertainty(intervals=pd.DataFrame({"metric": ["m"]}), unit="participant")
    assert rr._release_card_uncertainty(target_unc).equals(target_unc.intervals)
    res_unc = GP3MLResampleUncertainty(summary=pd.DataFrame({"metric": ["m"]}), unit="participant")
    assert rr._release_card_uncertainty(res_unc).equals(res_unc.summary)
    assert rr._release_card_uncertainty(None).empty
    with pytest.raises(GP3MLError, match="named vector"):
        rr.create_gazepoint_release_evidence(files=["x"])
    with pytest.raises(GP3MLError, match="existing paths"):
        rr.create_gazepoint_release_evidence(files={"x": tmp_path / "missing"})
    evidence = rr.create_gazepoint_release_evidence(notes=None)
    assert evidence.notes == []
    assert "Object hashes" in rr._release_evidence_repr(evidence)

    with pytest.raises(GP3MLError, match="Evaluator"):
        rob._numeric_metrics({"flag": True})
    assert rob._numeric_metrics(pd.Series({"x": 1.0})) == {"x": 1.0}
    assert rob._numeric_metrics(pd.DataFrame({"x": [1.0]})) == {"x": 1.0}
    with pytest.raises(GP3MLError, match="evaluator"):
        rob.evaluate_gazepoint_seed_stability([1], object())
    with pytest.raises(GP3MLError, match="named list"):
        rob.evaluate_gazepoint_missingness_sensitivity({}, lambda **kwargs: {"x": 1})

    thresholds = pd.DataFrame({"threshold": [0.1, 0.15, 0.2, 0.5], "score": [1.0, 1.0, 1.0, 0.0]})
    evaluation = gp.GP3MLThresholdEvaluation(thresholds=thresholds)
    stable = rob.evaluate_gazepoint_threshold_stability(evaluation, "score", tolerance=0.0)
    assert stable.status == "stable"
    review_eval = gp.GP3MLThresholdEvaluation(
        thresholds=pd.DataFrame({"threshold": [0.1, 0.15], "score": [1.0, 1.0]})
    )
    assert rob.evaluate_gazepoint_threshold_stability(review_eval, "score", tolerance=0).status == "review"
    unstable_eval = gp.GP3MLThresholdEvaluation(
        thresholds=pd.DataFrame({"threshold": [0.1, 0.12], "score": [1.0, 1.0]})
    )
    assert rob.evaluate_gazepoint_threshold_stability(unstable_eval, "score", tolerance=0).status == "unstable"
    with pytest.raises(GP3MLError, match="Invalid threshold"):
        rob.evaluate_gazepoint_threshold_stability(object(), "score")
    with pytest.raises(GP3MLError, match="direction"):
        rob.evaluate_gazepoint_threshold_stability(evaluation, "score", "bad")
    with pytest.raises(GP3MLError, match="Unknown metric"):
        rob.evaluate_gazepoint_threshold_stability(evaluation, "bad")

    nonfinite = GP3MLStabilityEvaluation(
        summary=pd.DataFrame({"metric": ["m"], "minimum": [0.0], "maximum": [0.0], "sd": [np.nan]})
    )
    robust_audit = rob.audit_gazepoint_model_robustness(seed_stability=nonfinite)
    assert robust_audit.status == "review"


def test_splitting_guard_tail_and_two_way_failure(monkeypatch):
    assert sp._scalar_column(None, "x") is None
    with pytest.raises(GP3MLError, match="column name"):
        sp._scalar_column(None, "x", False)
    with pytest.raises(GP3MLError, match="column name"):
        sp._scalar_column("", "x")
    assert sp._holdout_count(2, 0.01) == 1
    with pytest.raises(GP3MLError, match="At least two"):
        sp._holdout_count(1, 0.2)

    data, task, manifest = _classification_fixture()
    with pytest.raises(GP3MLError, match="feature_manifest"):
        sp._manifest_for_predictors(None, PREDICTORS)
    with pytest.raises(GP3MLError, match="compatible"):
        sp._manifest_for_predictors(pd.DataFrame({"x": [1]}), PREDICTORS)
    with pytest.raises(GP3MLError, match="missing from"):
        sp._manifest_for_predictors(manifest, ["missing"])
    review_manifest = manifest.copy()
    review_manifest.loc[:, "scientific_source"] = None
    with pytest.raises(GP3MLError, match="must pass validation"):
        sp._manifest_for_predictors(review_manifest, PREDICTORS)

    partition = np.array(["analysis", "assessment"])
    counts = sp._group_counts(pd.DataFrame({"trial": ["a", "b"]}), partition, None, "trial", None)
    assert "trial" in set(counts.unit)

    with pytest.raises(GP3MLError, match="Could not construct"):
        sp._two_way_assignment(
            np.array(["p1", "p1"]),
            np.array(["s1", "s1"]),
            0.2,
            np.random.default_rng(1),
            max_attempts=1,
        )

    base = dict(
        data=data,
        outcome=task.outcome,
        predictors=PREDICTORS,
        feature_manifest=manifest,
        generalization_target="new_participants",
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id="stimulus_id",
    )
    for changes, match in [
        ({"data": []}, "data frame"),
        ({"data": data.iloc[:1]}, "at least two rows"),
        ({"predictors": 1}, "predictors"),
        ({"predictors": []}, "predictors"),
        ({"predictors": ["tracking_ratio", "tracking_ratio"]}, "unique"),
        ({"predictors": [task.outcome]}, "outcome"),
        ({"predictors": ["participant_id"]}, "Identifier"),
        ({"generalization_target": "bad"}, "generalization_target"),
        ({"assessment_prop": 0}, "strictly between"),
        ({"seed": True}, "seed"),
        ({"participant_id": None}, "Required grouping"),
        ({"data": data.assign(**{".gp3ml_source_row": 1})}, "reserved source-row"),
        ({"predictors": ["missing"]}, "missing required columns"),
    ]:
        args = dict(base)
        args.update(changes)
        with pytest.raises(GP3MLError, match=match):
            sp.split_gazepoint_ml_data(**args)

    malformed_group = data.copy()
    malformed_group.loc[0, "participant_id"] = ""
    args = dict(base)
    args["data"] = malformed_group
    with pytest.raises(GP3MLError, match="missing or empty"):
        sp.split_gazepoint_ml_data(**args)


def test_object_mutation_json_deepcopy_and_locked_plan_paths():
    obj = GP3MLObject(a=1)
    assert obj.a == 1
    del obj["a"]
    with pytest.raises(AttributeError):
        _ = obj.a
    obj._private = "x"
    assert obj._private == "x"
    with pytest.raises(TypeError, match="dictionary"):
        GP3MLObject.from_dict([])

    class ToListFails:
        def tolist(self):
            raise TypeError("no")

        def __repr__(self):
            return "fallback-value"

    encoded = GP3MLObject(
        frame=pd.DataFrame({"x": [1]}),
        series=pd.Series([1]),
        array=np.array([1, 2]),
        nested={"x": [1, 2]},
        fallback=ToListFails(),
    ).to_json(indent=None)
    assert "fallback-value" in encoded
    copied = obj.__deepcopy__({})
    assert isinstance(copied, GP3MLObject)

    plan = GP3MLAnalysisPlan(locked=True, x=1)
    for action in (
        lambda: plan.__setitem__("x", 2),
        lambda: plan.__delitem__("x"),
        lambda: setattr(plan, "x", 2),
    ):
        with pytest.raises(GP3MLError, match="immutable"):
            action()
