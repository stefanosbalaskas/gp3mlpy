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
from gp3mlpy import governance_reports as gr
from gp3mlpy import model_artifacts as ma
from gp3mlpy import model_tuning as mt
from gp3mlpy import resample_evaluation as reval
from gp3mlpy import resampling as rs
from gp3mlpy import ro_crate as roc
from gp3mlpy import splitting as sp
from gp3mlpy.exceptions import GP3MLError
from gp3mlpy.objects import (
    GP3MLAPIContractRegistry,
    GP3MLModelCard,
    GP3MLModelTuning,
    GP3MLResampleEvaluation,
    GP3MLTask,
)


PREDICTORS = ["tracking_ratio", "blink_rate", "fixation_duration"]


def _fixture():
    data = gp.simulate_gazepoint_governed_data(8, 4, 2, seed=811)
    task = gp.create_gazepoint_synthetic_task(data, "assigned_condition", "new_participants")
    manifest = gp.create_gazepoint_synthetic_manifest(task.outcome, PREDICTORS)
    folds = gp.create_gazepoint_group_folds(
        data,
        task.outcome,
        PREDICTORS,
        manifest,
        task.generalization_target,
        participant_id=task.participant_id,
        trial_id=task.unit_id,
        stimulus_id=task.stimulus_id,
        v=2,
        seed=812,
    )
    return data, task, manifest, folds


def test_repr_contract_and_resample_card_final_arcs():
    registry = GP3MLAPIContractRegistry(exports=[], classes=[])
    assert "0 stable exports" in reps.render_r_print(registry)

    validation = ac.validate_gp3ml_object_contract([1, 2])
    assert validation.status in {"pass", "review"}

    task = GP3MLTask(
        outcome="y",
        task_type="classification",
        unit_id="trial_id",
        generalization_target="new_participants",
        purpose="coverage",
    )
    evaluation = GP3MLResampleEvaluation(metrics=pd.DataFrame({"metric": ["roc_auc"], "value": [0.8]}))
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
        evaluation=evaluation,
        calibration=None,
        external_validation=None,
        limitations=["limit"],
    )
    assert "roc_auc" in "\n".join(gr._model_card_markdown(card))


def test_artifact_explicit_inputs_and_nonnumeric_portability(monkeypatch):
    data, task, _, _ = _fixture()
    model = gp.fit_gazepoint_model(data, task, PREDICTORS, engine="glm")
    artifact = ma.create_gazepoint_model_artifact(
        model,
        preprocessor=model.preprocessor,
        task=model.task,
        bundle_model=False,
    )
    assert artifact.preprocessor is model.preprocessor
    assert artifact.task is model.task

    monkeypatch.setattr(
        ma,
        "_predict_for_portability",
        lambda model, newdata: np.asarray(["class-a", "class-b"], dtype=object),
    )
    portability = ma.test_gazepoint_model_portability(
        artifact,
        newdata=data.iloc[:2].copy(),
        fresh_process=False,
    )
    assert portability.prediction_equal is True


def test_tuning_exception_unique_and_multi_stage_tie_paths(tmp_path: Path, monkeypatch):
    grid = mt.create_gazepoint_tuning_grid("glm")
    folds = SimpleNamespace(metadata={"predictors": ["x"], "generalization_target": "new_participants"})
    task = SimpleNamespace(task_type="classification", generalization_target="new_participants")

    def fail_evaluation(**kwargs):
        raise RuntimeError("forced candidate failure")

    monkeypatch.setattr(mt, "evaluate_gazepoint_group_folds", fail_evaluation)
    failed = mt.tune_gazepoint_model(
        folds,
        task,
        grid,
        predictors=["x"],
        continue_on_error=True,
    )
    assert failed.results[0]["status"] == "fail"
    assert "forced candidate failure" in failed.results[0]["error"]

    two_grid = mt.create_gazepoint_tuning_grid(["glm", "ranger"], complexity=[2, 1])
    unique_comparison = pd.DataFrame(
        [
            {"candidate_id": "candidate_001", "metric": "roc_auc", "candidate_status": "pass", "success_prop": 1.0, "mean": 0.9},
            {"candidate_id": "candidate_002", "metric": "roc_auc", "candidate_status": "pass", "success_prop": 1.0, "mean": 0.8},
        ]
    )
    unique = GP3MLModelTuning(grid=two_grid, comparison=unique_comparison)
    assert mt.select_gazepoint_model(
        unique,
        "roc_auc",
        "maximize",
        rationale="predeclared review",
    ).candidate_id == "candidate_001"

    tied_comparison = pd.DataFrame(
        [
            {"candidate_id": "candidate_001", "metric": "roc_auc", "candidate_status": "pass", "success_prop": 1.0, "mean": 0.8},
            {"candidate_id": "candidate_002", "metric": "roc_auc", "candidate_status": "pass", "success_prop": 1.0, "mean": 0.8},
            {"candidate_id": "candidate_001", "metric": "f1", "candidate_status": "pass", "success_prop": 1.0, "mean": 0.7},
            {"candidate_id": "candidate_002", "metric": "f1", "candidate_status": "pass", "success_prop": 1.0, "mean": 0.7},
            {"candidate_id": "candidate_001", "metric": "rmse", "candidate_status": "pass", "success_prop": 1.0, "mean": 0.5},
            {"candidate_id": "candidate_002", "metric": "rmse", "candidate_status": "pass", "success_prop": 1.0, "mean": 0.4},
        ]
    )
    tied = GP3MLModelTuning(grid=two_grid, comparison=tied_comparison)
    selected = mt.select_gazepoint_model(
        tied,
        "roc_auc",
        "maximize",
        tie_breakers=["f1", "rmse"],
        rationale="predeclared review",
    )
    assert selected.candidate_id == "candidate_002"

    with pytest.raises(GP3MLError, match="gp3ml_model_tuning"):
        mt.write_gazepoint_model_tuning(object(), str(tmp_path))


def test_resample_role_and_leakage_raise_arcs(monkeypatch):
    _, task, _, folds = _fixture()

    with monkeypatch.context() as patch:
        patch.setattr(reval, "_redeclare_task", lambda analysis, original: task)
        patch.setattr(
            reval,
            "validate_gazepoint_ml_roles",
            lambda *args, **kwargs: SimpleNamespace(status="fail"),
        )
        role_failed = reval.evaluate_gazepoint_group_folds(
            folds,
            task,
            predictors=PREDICTORS,
            engine="glm",
            continue_on_error=True,
        )
        assert role_failed.fold_status.error.astype(str).str.contains("role validation").all()

    leakage_folds = deepcopy(folds)
    for fold in leakage_folds.folds.values():
        fold.leakage_audit.status = "fail"
    with monkeypatch.context() as patch:
        patch.setattr(reval, "_redeclare_task", lambda analysis, original: task)
        patch.setattr(
            reval,
            "validate_gazepoint_ml_roles",
            lambda *args, **kwargs: SimpleNamespace(status="pass"),
        )
        leakage_failed = reval.evaluate_gazepoint_group_folds(
            leakage_folds,
            task,
            predictors=PREDICTORS,
            engine="glm",
            continue_on_error=True,
        )
        assert leakage_failed.fold_status.error.astype(str).str.contains("leakage audit").all()


def test_string_predictor_branches_and_rocrate_exception(tmp_path: Path):
    data, task, manifest, _ = _fixture()
    single_manifest = manifest.loc[manifest["feature"] == PREDICTORS[0]].reset_index(drop=True)

    string_folds = rs.create_gazepoint_group_folds(
        data,
        task.outcome,
        PREDICTORS[0],
        single_manifest,
        task.generalization_target,
        participant_id=task.participant_id,
        trial_id=task.unit_id,
        stimulus_id=task.stimulus_id,
        v=2,
        seed=813,
    )
    assert string_folds.metadata["predictors"] == [PREDICTORS[0]]

    string_split = sp.split_gazepoint_ml_data(
        data,
        task.outcome,
        PREDICTORS[0],
        single_manifest,
        task.generalization_target,
        participant_id=task.participant_id,
        trial_id=task.unit_id,
        stimulus_id=task.stimulus_id,
        seed=814,
    )
    assert string_split.metadata["predictors"] == [PREDICTORS[0]]

    crate = tmp_path / "crate"
    crate.mkdir()
    (crate / "sha256-manifest.csv").write_text("bad,column\nx,y\n", encoding="utf-8")
    checked = roc.validate_gazepoint_ro_crate(crate)
    hash_status = checked.checks.loc[checked.checks["check"] == "hashes", "status"].iloc[0]
    assert hash_status == "fail"


def test_split_group_count_writer_collision_and_repr_paths(tmp_path: Path):
    data, task, manifest, _ = _fixture()
    partition = np.where(np.arange(len(data)) % 2 == 0, "analysis", "assessment")
    counts = sp._group_counts(
        data,
        partition,
        participant_id=None,
        trial_id=None,
        stimulus_id=task.stimulus_id,
    )
    assert "stimulus" in set(counts.unit)

    split = sp.split_gazepoint_ml_data(
        data,
        task.outcome,
        PREDICTORS,
        manifest,
        task.generalization_target,
        participant_id=task.participant_id,
        trial_id=task.unit_id,
        stimulus_id=task.stimulus_id,
        seed=815,
    )
    out_dir = tmp_path / "split"
    sp.write_gazepoint_ml_split_csv(split, out_dir, tables="summary")
    with pytest.raises(GP3MLError, match="already exist"):
        sp.write_gazepoint_ml_split_csv(split, out_dir, tables="summary")

    assert "gazepoint_ml_split" in sp._split_repr(split)
    assert "gazepoint_ml_split_validation" in sp._val_repr(split.validation)
