from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3mlpy as gp
from gp3mlpy import _reprs as reps
from gp3mlpy import api_contracts as ac
from gp3mlpy import external_validation as ev
from gp3mlpy import governance_reports as gr
from gp3mlpy import model_artifacts as ma
from gp3mlpy import model_tuning as mt
from gp3mlpy import nested_resampling as nr
from gp3mlpy import resample_evaluation as reval
from gp3mlpy import resampling as rs
from gp3mlpy import resampling_diagnostics as rd
from gp3mlpy import ro_crate as roc
from gp3mlpy import roadmap_reporting as rr
from gp3mlpy import robustness as rob
from gp3mlpy import splitting as sp
from gp3mlpy.exceptions import GP3MLError
from gp3mlpy.objects import (
    GP3MLAnalysisPlan,
    GP3MLCalibrationAssessment,
    GP3MLEngineCapabilities,
    GP3MLExternalValidation,
    GP3MLModelArtifact,
    GP3MLModelCard,
    GP3MLModelTuning,
    GP3MLObject,
    GP3MLReleaseModelCard,
    GP3MLResampleEvaluation,
    GP3MLTask,
)


PREDICTORS = ["tracking_ratio", "blink_rate", "fixation_duration"]


def _classification_fixture(target: str = "new_participants"):
    data = gp.simulate_gazepoint_governed_data(8, 4, 2, seed=731)
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
        v=[2, 2] if target == "new_participants_and_new_stimuli" else 2,
        repeats=1,
        seed=732,
    )
    return data, task, manifest, folds


def test_repr_api_and_object_fallbacks():
    assert reps._component(object(), "missing", 7) == 7

    capabilities = GP3MLEngineCapabilities(engine=["glm"], status=["pass"])
    rendered = reps.render_r_print(capabilities)
    assert "gp3ml_engine_capabilities" in rendered

    schema = ac.gp3ml_object_schema({"outer": {"inner": 1}}, recursive=True)
    assert "outer$inner" in schema.component.tolist()

    raw = GP3MLObject.__new__(GP3MLObject)
    assert callable(raw.to_dict)

    class BadList:
        def tolist(self):
            raise TypeError("no list conversion")

        def __repr__(self) -> str:
            return "<bad-list>"

    payload = GP3MLObject(value=BadList())
    assert "<bad-list>" in payload.to_json()
    assert "gp3ml_object" in repr(GP3MLObject(value=1))
    assert "gp3ml_object" in str(GP3MLObject(value=1))

    plan = GP3MLAnalysisPlan(locked=False)
    plan._private_marker = 1
    assert plan._private_marker == 1


def test_governance_report_calibration_and_git_command_paths(tmp_path: Path, monkeypatch):
    task = GP3MLTask(
        outcome="y",
        task_type="classification",
        unit_id="trial_id",
        participant_id="participant_id",
        stimulus_id="stimulus_id",
        generalization_target="new_participants",
        purpose="coverage contract",
    )
    calibration = GP3MLCalibrationAssessment(summary=pd.DataFrame({"ece": [0.1]}))
    card = GP3MLModelCard(
        title="card",
        created_at="now",
        intended_use="test",
        prohibited_uses=[],
        task=task,
        engine="glm",
        predictors=["x"],
        training_n=2,
        training_hash="hash",
        evaluation=pd.DataFrame({"roc_auc": [0.8]}),
        calibration=calibration,
        external_validation=object(),
        limitations=["limit"],
    )
    text = "\n".join(gr._model_card_markdown(card))
    assert "ece" in text
    assert "attached" in text

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)

    class Result:
        def __init__(self, stdout: str):
            self.stdout = stdout

    values = iter(["abc123\n", "main\n", ""])
    monkeypatch.setattr(gr.subprocess, "run", lambda *args, **kwargs: Result(next(values)))
    info = gr._git_info(repo)
    assert info == {"commit": "abc123", "branch": "main", "clean": True}

    def fail_run(*args, **kwargs):
        raise RuntimeError("git unavailable")

    monkeypatch.setattr(gr.subprocess, "run", fail_run)
    failed = gr._git_info(repo)
    assert failed == {"commit": None, "branch": None, "clean": None}


def test_model_artifact_metadata_hash_and_prediction_fallbacks(monkeypatch):
    task = GP3MLTask(
        outcome="y",
        task_type="classification",
        generalization_target="new_participants",
        purpose="artifact test",
    )

    def dummy(engine):
        return SimpleNamespace(
            predictors=[],
            engine=engine,
            task=task,
            preprocessor=None,
            training_hash="hash",
            training_n=1,
            seed=1,
            threshold=0.5,
            fit=None,
        )

    def missing_version(name: str):
        raise ma.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(ma.metadata, "version", missing_version)
    known = ma.create_gazepoint_model_artifact(dummy("keras3"), bundle_model=False)
    assert known.metadata["engine_version"] is None
    assert known.metadata["gp3mlpy_version"] == "development"
    assert known.metadata["bundle_error"] is None

    unknown = ma.create_gazepoint_model_artifact(dummy("custom"), bundle_model=False)
    assert unknown.metadata["engine_version"] is None

    non_string = ma.create_gazepoint_model_artifact(dummy(object()), bundle_model=False)
    assert non_string.metadata["engine_version"] is None

    malformed = GP3MLModelArtifact(model=None, task=None, predictor_schema=None, metadata=None)
    validation = ma.validate_gazepoint_model_artifact(malformed)
    assert validation.status == "fail"

    tampered = deepcopy(known)
    tampered.artifact_hash = "not-the-real-hash"
    assert ma.validate_gazepoint_model_artifact(tampered).status == "fail"

    class BrokenPredictor:
        def predict(self, newdata):
            raise RuntimeError("prediction failed")

    assert ma._predict_for_portability(BrokenPredictor(), pd.DataFrame()) is None


def test_tuning_all_failed_ties_and_writer_guards(tmp_path: Path, monkeypatch):
    grid = mt.create_gazepoint_tuning_grid("glm")
    folds = SimpleNamespace(
        metadata={"predictors": ["x"], "generalization_target": "new_participants"}
    )
    task = SimpleNamespace(task_type="classification", generalization_target="new_participants")
    fake_evaluation = SimpleNamespace(
        fold_status=pd.DataFrame(
            {
                "status": ["fail", "fail"],
                "error": ["first", "second"],
                "warnings": ["warn", None],
            }
        ),
        metrics=pd.DataFrame(),
        validation=SimpleNamespace(status="review"),
    )
    monkeypatch.setattr(mt, "evaluate_gazepoint_group_folds", lambda **kwargs: fake_evaluation)

    tuned = mt.tune_gazepoint_model(
        folds,
        task,
        grid,
        predictors=["x"],
        continue_on_error=True,
    )
    assert tuned.results[0]["status"] == "fail"
    assert tuned.results[0]["error"] == "first | second"

    with pytest.raises(GP3MLError, match="Candidate"):
        mt.tune_gazepoint_model(
            folds,
            task,
            grid,
            predictors=["x"],
            continue_on_error=False,
        )

    with pytest.raises(GP3MLError, match="gp3ml_model_tuning"):
        mt.select_gazepoint_model(object(), "roc_auc", "maximize", rationale="review")

    tie_grid = mt.create_gazepoint_tuning_grid(
        ["glm", "ranger"], complexity=[np.nan, np.nan]
    )
    comparison = pd.DataFrame(
        [
            {"candidate_id": "candidate_001", "metric": "roc_auc", "candidate_status": "pass", "success_prop": 1.0, "mean": 0.8},
            {"candidate_id": "candidate_002", "metric": "roc_auc", "candidate_status": "pass", "success_prop": 1.0, "mean": 0.8},
            {"candidate_id": "candidate_001", "metric": "rmse", "candidate_status": "pass", "success_prop": 1.0, "mean": 0.5},
            {"candidate_id": "candidate_002", "metric": "rmse", "candidate_status": "pass", "success_prop": 1.0, "mean": 0.4},
        ]
    )
    tied = GP3MLModelTuning(grid=tie_grid, comparison=comparison)
    selected = mt.select_gazepoint_model(
        tied,
        "roc_auc",
        "maximize",
        tie_breakers=["accuracy", "missing_metric", "rmse"],
        rationale="predeclared human review",
    )
    assert selected.candidate_id == "candidate_002"

    unresolved_grid = mt.create_gazepoint_tuning_grid(
        ["glm", "ranger"], complexity=["simple", "complex"]
    )
    unresolved = GP3MLModelTuning(
        grid=unresolved_grid,
        comparison=comparison.loc[comparison.metric == "roc_auc"].copy(),
    )
    with pytest.raises(GP3MLError, match="Selection remains tied"):
        mt.select_gazepoint_model(
            unresolved,
            "roc_auc",
            "maximize",
            rationale="reviewed tie",
        )

    with pytest.raises(GP3MLError, match="selection"):
        mt.write_gazepoint_model_tuning(tuned, str(tmp_path), selection=object())


def test_external_transportability_empty_metrics_and_calibration_drift(monkeypatch):
    data, task, _ = _classification_fixture()
    model = gp.fit_gazepoint_model(data, task, PREDICTORS, engine="glm")

    assert "complex" in ev._column_class(pd.Series([1 + 2j]))
    no_external = ev.evaluate_gazepoint_external_transportability(
        model,
        data,
        external_data=None,
        threshold=0.4,
    )
    assert no_external.status == "not_externally_validated"

    external = data.copy()
    external[task.participant_id] = "extp_" + external[task.participant_id].astype(str)
    external[task.stimulus_id] = "exts_" + external[task.stimulus_id].astype(str)
    declaration = ev.declare_gazepoint_external_dataset(
        external,
        label="independent",
        independent=True,
        origin="synthetic external",
        participant_id=task.participant_id,
        stimulus_id=task.stimulus_id,
    )

    empty_validation = GP3MLExternalValidation(
        metrics=pd.DataFrame(),
        shift=pd.DataFrame(),
        calibration=None,
    )
    monkeypatch.setattr(ev, "evaluate_external_validation", lambda *args, **kwargs: empty_validation)
    empty_report = ev.evaluate_gazepoint_external_transportability(
        model,
        data,
        external,
        declaration,
    )
    assert empty_report.performance_comparison.empty

    external_calibration = GP3MLCalibrationAssessment(summary=pd.DataFrame({"ece": [0.1]}))
    calibrated_validation = GP3MLExternalValidation(
        metrics=pd.DataFrame({"roc_auc": [0.8]}),
        shift=pd.DataFrame(),
        calibration=external_calibration,
    )
    monkeypatch.setattr(
        ev,
        "evaluate_external_validation",
        lambda *args, **kwargs: calibrated_validation,
    )
    development_calibration = GP3MLCalibrationAssessment(
        summary=pd.DataFrame({"ece": [0.2], "label": ["ok"], "other": [0.3]})
    )
    development = GP3MLResampleEvaluation(
        fold_results=[{"calibration": development_calibration}],
        metrics=pd.DataFrame({"metric": ["roc_auc"], "value": [0.75]}),
    )
    drift_report = ev.evaluate_gazepoint_external_transportability(
        model,
        data,
        external,
        declaration,
        development_evaluation=development,
    )
    assert "drift_ece" in drift_report.calibration_drift.columns
    assert "development_other" in drift_report.calibration_drift.columns
    assert "drift_other" not in drift_report.calibration_drift.columns


def test_grouped_nested_excluded_rows_and_diagnostic_string_outcome(monkeypatch):
    _, task, _, folds = _folds("new_participants_and_new_stimuli")
    evaluation = reval.evaluate_gazepoint_group_folds(
        folds,
        task,
        predictors=PREDICTORS,
        engine="glm",
        continue_on_error=True,
    )
    assert not evaluation.excluded.empty

    _, simple_task, _, simple_folds = _folds("new_participants")
    with monkeypatch.context() as patch:
        patch.setattr(
            reval,
            "validate_gazepoint_ml_roles",
            lambda *args, **kwargs: SimpleNamespace(status="fail"),
        )
        role_failed = reval.evaluate_gazepoint_group_folds(
            simple_folds,
            simple_task,
            predictors=PREDICTORS,
            engine="glm",
            continue_on_error=True,
        )
        assert (role_failed.fold_status.status == "fail").all()

    leakage_folds = deepcopy(simple_folds)
    first = next(iter(leakage_folds.folds.values()))
    first.leakage_audit.status = "fail"
    leakage_failed = reval.evaluate_gazepoint_group_folds(
        leakage_folds,
        simple_task,
        predictors=PREDICTORS,
        engine="glm",
        continue_on_error=True,
    )
    assert (leakage_failed.fold_status.status == "fail").any()

    string_folds = deepcopy(simple_folds)
    for fold in string_folds.folds.values():
        for part in ("analysis", "assessment", "excluded"):
            fold[part][simple_task.outcome] = fold[part][simple_task.outcome].astype(str)
    diagnostics = rd.diagnose_gazepoint_group_folds(string_folds)
    assert not diagnostics.outcome_distribution.empty

    nested = nr.create_gazepoint_nested_folds(
        folds,
        inner_v=2,
        inner_repeats=1,
        seed=733,
        continue_on_error=True,
    )
    nested_eval = nr.evaluate_gazepoint_nested_resampling(
        nested,
        task,
        mt.create_gazepoint_tuning_grid("glm"),
        selection_metric="roc_auc",
        direction="maximize",
        predictors=PREDICTORS,
        minimum_success_prop=0.0,
        seed=734,
        continue_on_error=True,
    )
    assert not nested_eval.excluded.empty


def test_rocrate_roadmap_and_robustness_final_fallbacks(tmp_path: Path, monkeypatch):
    crate = tmp_path / "crate"
    crate.mkdir()
    (crate / "manifest-sha256.csv").write_text("file,sha256\nx,y\n", encoding="utf-8")

    def fail_read_csv(*args, **kwargs):
        raise ValueError("cannot parse manifest")

    monkeypatch.setattr(roc.pd, "read_csv", fail_read_csv)
    assert roc.validate_gazepoint_ro_crate(crate).status == "fail"

    summary_frame = pd.DataFrame({"metric": ["roc_auc"], "mean": [0.8]})
    monkeypatch.setattr(
        rr,
        "summarize_gazepoint_resample_performance",
        lambda evaluation: SimpleNamespace(summary=summary_frame),
    )
    release_card = GP3MLReleaseModelCard(evaluation=GP3MLResampleEvaluation())
    assert rr._release_card_metrics(release_card).equals(summary_frame)

    task = GP3MLTask(outcome="y")
    printable = GP3MLReleaseModelCard(
        task=task,
        generalization_target="new_participants",
        selection_procedure_recorded=False,
        uncertainty_unit=None,
        external_validation_status="not_externally_validated",
    )
    assert "Uncertainty unit: NA" in rr._release_card_repr(printable)

    with pytest.raises(GP3MLError, match="Evaluator"):
        rob._numeric_metrics(pd.Series({"x": "not numeric"}))
    with pytest.raises(GP3MLError, match="Evaluator"):
        rob._numeric_metrics(pd.DataFrame({"x": ["not numeric"]}))
    unknown = rob.audit_gazepoint_model_robustness(seed_stability=object())
    assert unknown.status == "review" and unknown.findings.empty


def test_split_and_fold_defensive_guards_and_writer_paths(tmp_path: Path, monkeypatch):
    data, task, manifest = _classification_fixture()
    split = sp.split_gazepoint_ml_data(
        data,
        task.outcome,
        PREDICTORS,
        manifest,
        task.generalization_target,
        participant_id=task.participant_id,
        trial_id=task.unit_id,
        stimulus_id=task.stimulus_id,
        seed=735,
    )

    with pytest.raises(GP3MLError, match="gazepoint_ml_split"):
        sp.write_gazepoint_ml_split_csv(object(), tmp_path)
    with pytest.raises(GP3MLError, match="prefix"):
        sp.write_gazepoint_ml_split_csv(split, tmp_path, prefix="")
    with pytest.raises(GP3MLError, match="directory separators"):
        sp.write_gazepoint_ml_split_csv(split, tmp_path, prefix="bad/name")
    with pytest.raises(GP3MLError, match="tables"):
        sp.write_gazepoint_ml_split_csv(split, tmp_path, tables=[])

    with pytest.raises(GP3MLError, match="Could not construct"):
        sp._two_way_assignment(
            np.array(["p1", "p1", "p2", "p2"]),
            np.array(["s1", "s2", "s1", "s2"]),
            0.2,
            np.random.default_rng(1),
            max_attempts=0,
        )

    two_data, two_task, two_manifest = _classification_fixture(
        "new_participants_and_new_stimuli"
    )

    def all_analysis(participant, stimulus, assessment_prop, rng, max_attempts=250):
        n = len(participant)
        return {
            "partition": np.array(["analysis"] * n, dtype=object),
            "participant_held": np.zeros(n, dtype=bool),
            "stimulus_held": np.zeros(n, dtype=bool),
            "held_participants": [],
            "held_stimuli": [],
        }

    with monkeypatch.context() as patch:
        patch.setattr(sp, "_two_way_assignment", all_analysis)
        with pytest.raises(GP3MLError, match="empty partition"):
            sp.split_gazepoint_ml_data(
                two_data,
                two_task.outcome,
                PREDICTORS,
                two_manifest,
                two_task.generalization_target,
                participant_id=two_task.participant_id,
                trial_id=two_task.unit_id,
                stimulus_id=two_task.stimulus_id,
            )

    with pytest.raises(GP3MLError, match="predictors"):
        rs.create_gazepoint_group_folds(
            data,
            task.outcome,
            object(),
            manifest,
            task.generalization_target,
            participant_id=task.participant_id,
        )

    def one_fold(values, v, rng):
        mapping = pd.DataFrame(
            {"group": [str(values[0])], "fold": [1], "n_rows": [len(values)]}
        )
        return np.ones(len(values), dtype=int), mapping

    with monkeypatch.context() as patch:
        patch.setattr(rs, "_assign_groups", one_fold)
        with pytest.raises(GP3MLError, match="empty analysis or assessment"):
            rs.create_gazepoint_group_folds(
                data,
                task.outcome,
                PREDICTORS,
                manifest,
                task.generalization_target,
                participant_id=task.participant_id,
                trial_id=task.unit_id,
                stimulus_id=task.stimulus_id,
                v=2,
            )
