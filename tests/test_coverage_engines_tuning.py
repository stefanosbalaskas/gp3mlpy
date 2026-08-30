from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3mlpy as gp
from gp3mlpy import deep_learning as dl
from gp3mlpy import model_engines as me
from gp3mlpy import model_tuning as mt
from gp3mlpy import nested_resampling as nr
from gp3mlpy import preprocessing as pp
from gp3mlpy.exceptions import GP3MLError, OptionalDependencyError
from gp3mlpy.objects import GP3MLEngine, GP3MLModel, GP3MLModelTuning, GP3MLPreprocessor


PREDICTORS = ["tracking_ratio", "blink_rate", "fixation_duration"]


def _classification_fixture(target: str = "new_participants"):
    data = gp.simulate_gazepoint_governed_data(
        n_participants=8,
        n_stimuli=4,
        trials_per_cell=2,
        seed=81,
    )
    task = gp.create_gazepoint_synthetic_task(
        data,
        workflow="assigned_condition",
        generalization_target=target,
    )
    manifest = gp.create_gazepoint_synthetic_manifest(task.outcome, PREDICTORS)
    return data, task, manifest


def _regression_fixture():
    data = gp.simulate_gazepoint_governed_data(
        n_participants=8,
        n_stimuli=4,
        trials_per_cell=1,
        seed=82,
    )
    task = gp.create_gazepoint_synthetic_task(
        data,
        workflow="observed_duration",
        generalization_target="new_participants",
    )
    return data, task


def _folds(data, task, manifest):
    return gp.create_gazepoint_group_folds(
        data,
        task.outcome,
        PREDICTORS,
        manifest,
        task.generalization_target,
        participant_id=task.participant_id,
        trial_id=task.unit_id,
        stimulus_id=task.stimulus_id,
        v=2,
        repeats=1,
        seed=83,
    )


def test_preprocessor_numeric_factor_novel_zero_variance_and_guards():
    frame = pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0, 5.0],
            "all_missing": [np.nan, np.nan, np.nan, np.nan],
            "cat": ["a", "", None, "b"],
            "flag": [True, False, True, False],
            "constant": [1.0, 1.0, 1.0, 1.0],
        }
    )
    assert pp._is_numeric_series(frame["num"])
    assert not pp._is_numeric_series(frame["flag"])

    raw, values, levels = pp._prepare_raw_frame(
        frame,
        predictors=["num", "all_missing", "cat", "flag"],
        fit=True,
        numeric_imputation="mean",
        novel_level="other",
    )
    assert values["num"] == pytest.approx(3.0)
    assert values["all_missing"] == 0.0
    assert "<missing>" in levels["cat"] and "<other>" in levels["cat"]
    matrix = pp._model_matrix(raw)
    assert "num" in matrix.columns
    assert any(name.startswith("cat") for name in matrix.columns)

    pre = pp.fit_gazepoint_preprocessor(
        frame,
        ["num", "all_missing", "cat", "flag", "constant"],
        numeric_imputation="mean",
        center=False,
        scale=False,
        novel_level="other",
        remove_zero_variance=True,
    )
    baked = pp.bake_gazepoint_preprocessor(pre, frame)
    assert baked.shape[0] == len(frame)
    assert np.isfinite(baked).all()
    assert "constant" not in pre.columns

    keep_constant = pp.fit_gazepoint_preprocessor(
        frame,
        ["constant"],
        center=True,
        scale=True,
        remove_zero_variance=False,
    )
    assert keep_constant.scale["constant"] == 1.0
    assert pp.bake_gazepoint_preprocessor(keep_constant, frame).shape == (4, 1)

    error_pre = pp.fit_gazepoint_preprocessor(
        frame,
        ["cat"],
        novel_level="error",
        remove_zero_variance=False,
    )
    novel = pd.DataFrame({"cat": ["new"]})
    with pytest.raises(GP3MLError, match="Novel levels"):
        pp.bake_gazepoint_preprocessor(error_pre, novel)

    other_pre = pp.fit_gazepoint_preprocessor(
        frame,
        ["cat"],
        novel_level="other",
        remove_zero_variance=False,
    )
    assert pp.bake_gazepoint_preprocessor(other_pre, novel).shape[0] == 1

    # Exercise bake's missing model-matrix-column restoration.
    altered = other_pre.__deepcopy__({})
    altered.columns = [*altered.columns, "synthetic_missing_column"]
    altered.center["synthetic_missing_column"] = 0.0
    altered.scale["synthetic_missing_column"] = 1.0
    baked_altered = pp.bake_gazepoint_preprocessor(altered, frame)
    assert baked_altered.shape[1] == len(altered.columns)
    assert np.allclose(baked_altered[:, -1], 0.0)

    with pytest.raises(GP3MLError, match="numeric_imputation"):
        pp.fit_gazepoint_preprocessor(frame, ["num"], numeric_imputation="bad")
    with pytest.raises(GP3MLError, match="novel_level"):
        pp.fit_gazepoint_preprocessor(frame, ["cat"], novel_level="bad")
    with pytest.raises(GP3MLError, match="preprocessor"):
        pp.bake_gazepoint_preprocessor(object(), frame)


def test_engine_inventory_custom_engine_and_training_summaries(monkeypatch):
    data, task, _ = _classification_fixture()
    defaults = me._default_predictors(data, task)
    assert task.outcome not in defaults
    summary = me._training_summary(
        pd.DataFrame({"num": [1.0, np.nan], "cat": ["a", None], "flag": [True, False]}),
        ["num", "cat", "flag"],
    )
    assert summary["num"]["type"] == "numeric"
    assert summary["cat"]["type"] == "categorical"
    assert summary["flag"]["type"] == "categorical"

    available = me.gp3ml_available_engines()
    assert {"glm", "lm", "ranger", "xgboost", "nnet", "keras3", "custom"} == set(
        available.engine
    )

    with pytest.raises(GP3MLError, match="must be functions"):
        me.integrate_black_box_model("bad", 1, lambda **_: None)
    with pytest.raises(GP3MLError, match="safety declarations"):
        me.integrate_black_box_model("bad", lambda **_: None, lambda **_: None)

    def fit_fun(*, x, y, task, args):
        return {"mean": float(np.mean(y)), "n": len(x), "args": args}

    def predict_fun(*, fit, newdata, type, task, **kwargs):
        del type, task, kwargs
        return np.repeat(fit["mean"], len(newdata))

    custom = me.integrate_black_box_model(
        "safe-custom",
        fit_fun,
        predict_fun,
        supports=["classification"],
        metadata={"backend": "test"},
        safety_declaration={
            "prohibited_uses_acknowledged": True,
            "prediction_time_inputs_only": True,
            "group_aware_evaluation_required": True,
        },
    )
    assert custom.probability is True
    model = me.fit_gazepoint_model(data, task, PREDICTORS, engine=custom, engine_args={"x": 1})
    assert model.python_backend == "custom:safe-custom"
    assert len(me.predict_gazepoint_model(model, data.iloc[:3], type="probability")) == 3

    unsupported = GP3MLEngine(
        name="reg-only",
        fit_fun=fit_fun,
        predict_fun=predict_fun,
        supports=["regression"],
        probability=False,
        metadata={},
        safety_declaration={},
    )
    with pytest.raises(GP3MLError, match="does not support"):
        me.fit_gazepoint_model(data, task, PREDICTORS, engine=unsupported)


def test_statsmodels_ranger_nnet_models_and_prediction_branches():
    data, task, _ = _classification_fixture()
    glm = me.fit_gazepoint_model(data, task, PREDICTORS, engine=None, seed=1)
    assert glm.engine == "glm"
    assert "engine=glm" in repr(glm)
    for ptype in ["response", "probability", "link"]:
        out = me.predict_gazepoint_model(glm, data.iloc[:5], type=ptype)
        assert len(out) == 5
    klass = me.predict_gazepoint_model(glm, data.iloc[:5], type="class")
    assert isinstance(klass, pd.Categorical)
    with pytest.raises(GP3MLError, match="type"):
        me.predict_gazepoint_model(glm, data.iloc[:2], type="bad")

    ranger = me.fit_gazepoint_model(
        data,
        task,
        PREDICTORS,
        engine="ranger",
        engine_args={
            "num.trees": 5,
            "num.threads": 1,
            "mtry": 1,
            "min.node.size": 1,
            "max.depth": 0,
            "replace": False,
            "sample.fraction": 0.8,
        },
        seed=2,
    )
    assert ranger.python_backend == "sklearn_random_forest_semantic_adapter"
    assert len(ranger.predict(data.iloc[:4], type="probability")) == 4

    nnet = me.fit_gazepoint_model(
        data,
        task,
        PREDICTORS,
        engine="nnet",
        engine_args={
            "size": 2,
            "linout": False,
            "trace": False,
            "MaxNWts": 100,
            "maxit": 20,
            "decay": 0.001,
            "reltol": 1e-4,
            "abstol": 1e-8,
        },
        seed=3,
    )
    assert nnet.python_backend == "sklearn_mlp_semantic_adapter"
    assert len(nnet.predict(data.iloc[:4], type="probability")) == 4

    rdata, rtask = _regression_fixture()
    lm = me.fit_gazepoint_model(rdata, rtask, PREDICTORS, engine=None)
    assert lm.engine == "lm"
    assert len(lm.predict(rdata.iloc[:3])) == 3
    mapped = me.fit_gazepoint_model(rdata, rtask, PREDICTORS, engine="glm")
    assert mapped.engine == "lm"
    rforest = me.fit_gazepoint_model(
        rdata, rtask, PREDICTORS, engine="ranger", engine_args={"num_trees": 4}, seed=4
    )
    assert len(rforest.predict(rdata.iloc[:3])) == 3
    rnet = me.fit_gazepoint_model(
        rdata, rtask, PREDICTORS, engine="nnet", engine_args={"size": 2, "maxit": 20}, seed=5
    )
    assert len(rnet.predict(rdata.iloc[:3])) == 3

    with pytest.raises(GP3MLError, match="classification engine"):
        me.fit_gazepoint_model(data, task, PREDICTORS, engine="lm")
    with pytest.raises(GP3MLError, match="Unknown engine"):
        me.fit_gazepoint_model(data, task, PREDICTORS, engine="unknown")
    with pytest.raises(GP3MLError, match="cannot be predictors"):
        me.fit_gazepoint_model(data, task, [task.outcome], engine="glm")
    with pytest.raises(GP3MLError, match="binary classification"):
        me.train_gazepoint_classifier(rdata, rtask, PREDICTORS, engine="ranger")

    missing_y = data.copy()
    missing_y.loc[0, task.outcome] = None
    with pytest.raises(GP3MLError, match="outcomes may not be missing"):
        me.fit_gazepoint_model(missing_y, task, PREDICTORS)
    rmissing = rdata.copy()
    rmissing.loc[0, rtask.outcome] = np.nan
    with pytest.raises(GP3MLError, match="outcomes may not be missing"):
        me.fit_gazepoint_model(rmissing, rtask, PREDICTORS)

    constant = pd.DataFrame({"x": [1.0] * len(data)})
    no_cols = pp.fit_gazepoint_preprocessor(constant, ["x"])
    with pytest.raises(GP3MLError, match="No usable model columns"):
        me.fit_gazepoint_model(data.assign(x=1.0), task, ["x"], preprocessor=no_cols)

    unsupported = glm.__deepcopy__({})
    unsupported.engine = "mystery"
    unsupported.engine_spec = None
    with pytest.raises(GP3MLError, match="Unsupported fitted engine"):
        me.predict_gazepoint_model(unsupported, data.iloc[:2])


def test_xgboost_optional_failure_and_fake_classifier_regressor(monkeypatch):
    data, task, _ = _classification_fixture()
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "xgboost", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(OptionalDependencyError, match="xgboost"):
        me.fit_gazepoint_model(data, task, PREDICTORS, engine="xgboost")
    monkeypatch.setattr(builtins, "__import__", real_import)

    class FakeXGB:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.mean = 0.5

        def fit(self, x, y):
            self.mean = float(np.mean(y))
            return self

        def predict_proba(self, x):
            p = np.repeat(self.mean, len(x))
            return np.column_stack([1 - p, p])

        def predict(self, x):
            return np.repeat(self.mean, len(x))

    fake = SimpleNamespace(XGBClassifier=FakeXGB, XGBRegressor=FakeXGB)
    monkeypatch.setitem(sys.modules, "xgboost", fake)
    cls = me.fit_gazepoint_model(
        data,
        task,
        PREDICTORS,
        engine="xgboost",
        engine_args={"nrounds": 3, "max_depth": 1},
        seed=7,
    )
    assert cls.python_backend == "xgboost"
    assert len(cls.predict(data.iloc[:2], type="probability")) == 2

    rdata, rtask = _regression_fixture()
    reg = me.fit_gazepoint_model(rdata, rtask, PREDICTORS, engine="xgboost", seed=8)
    assert len(reg.predict(rdata.iloc[:2])) == 2


def test_deep_learning_missing_dependency_and_fake_keras(monkeypatch):
    data, task, _ = _classification_fixture()
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "keras":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "keras", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(OptionalDependencyError, match="keras"):
        dl.fit_gazepoint_deep_model(data, task, PREDICTORS, epochs=1)
    monkeypatch.setattr(builtins, "__import__", real_import)

    class FakeLayer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeSequential:
        def __init__(self):
            self.layers_added = []
            self.loss = None

        def add(self, layer):
            self.layers_added.append(layer)

        def compile(self, optimizer, loss, metrics):
            self.optimizer = optimizer
            self.loss = loss
            self.metrics = metrics

        def fit(self, x, y, **kwargs):
            self.mean = float(np.mean(y))
            return {"history": "fake", **kwargs}

        def predict(self, x, verbose=0):
            del verbose
            return np.full((len(x), 1), getattr(self, "mean", 0.5))

    fake_keras = SimpleNamespace(
        utils=SimpleNamespace(set_random_seed=lambda seed: seed),
        Sequential=FakeSequential,
        layers=SimpleNamespace(Input=FakeLayer, Dense=FakeLayer, Dropout=FakeLayer),
    )
    monkeypatch.setitem(sys.modules, "keras", fake_keras)

    model = dl.fit_gazepoint_deep_model(
        data,
        task,
        PREDICTORS,
        hidden_units=(4, 2),
        dropout=0.25,
        epochs=1,
        batch_size=8,
        validation_split=0.1,
        seed=9,
    )
    assert model.engine == "keras3"
    assert model.fit.loss == "binary_crossentropy"
    assert len(me.predict_gazepoint_model(model, data.iloc[:3])) == 3

    rdata, rtask = _regression_fixture()
    rpre = pp.fit_gazepoint_preprocessor(rdata, PREDICTORS)
    rmodel = dl.fit_gazepoint_deep_model(
        rdata,
        rtask,
        predictors=None,
        preprocessor=rpre,
        hidden_units=(3,),
        dropout=0.0,
        epochs=1,
        batch_size=4,
        validation_split=0.0,
        optimizer="sgd",
        seed=10,
    )
    assert rmodel.fit.loss == "mean_squared_error"
    assert len(rmodel.predict(rdata.iloc[:2])) == 2


def test_tuning_grid_helpers_validation_selection_and_writers(tmp_path: Path):
    assert mt._as_values(1) == [1]
    assert mt._as_values(np.array([1, 2])) == [1, 2]
    assert mt._expand_grid_list(None) == [{}]
    expanded = mt._expand_grid_list({"a": [1, 2], "b": ["x", "y"]})
    assert expanded[0] == {"a": 1, "b": "x"}
    assert mt._collapse_args({}) == "default"
    assert "a=1/2" in mt._collapse_args({"a": [1, 2]})
    assert "engine:" in mt._candidate_label("glm", {}, {}, 0.5)
    assert mt._recycle_metadata("x", 2, "x") == ["x", "x"]
    with pytest.raises(GP3MLError, match="candidate count"):
        mt._recycle_metadata([1, 2], 3, "x")
    with pytest.raises(GP3MLError, match="named list"):
        mt._expand_grid_list({"": [1]})
    with pytest.raises(GP3MLError, match="at least one value"):
        mt._expand_grid_list({"x": []})

    grid = mt.create_gazepoint_tuning_grid(
        engine=["glm", "ranger", "glm"],
        engine_grid={"n_estimators": [3, 4]},
        preprocessor_grid={"center": [True, False]},
        thresholds=[0.4, 0.6],
        complexity=1,
        interpretability="high",
    )
    assert len(grid.candidates) == 16
    assert "candidates=16" in repr(grid)
    labeled = mt.create_gazepoint_tuning_grid(
        "glm", thresholds=[0.4, 0.6], labels=["low", "high"], complexity=[1, 2]
    )
    assert labeled.candidates.label.tolist() == ["low", "high"]

    for args, match in [
        ({"engine": []}, "engine"),
        ({"engine": "glm", "thresholds": [0]}, "thresholds"),
        ({"engine": "glm", "thresholds": ["x"]}, "thresholds"),
        ({"engine": "glm", "thresholds": [0.5, 0.6], "labels": ["x"] * 3}, "labels"),
    ]:
        with pytest.raises(GP3MLError, match=match):
            mt.create_gazepoint_tuning_grid(**args)

    assert mt._metric_direction("rmse") == "minimize"
    assert mt._metric_direction("roc_auc") == "maximize"
    assert mt._flatten_args({}) == ""
    assert mt._flatten_args({"a": [1, 2]}) == "a=1/2"
    assert mt._validate_direction("roc_auc", "maximize") == "maximize"
    with pytest.raises(GP3MLError, match="direction"):
        mt._validate_direction("roc_auc", "bad")
    with pytest.raises(GP3MLError, match="accuracy"):
        mt._validate_direction("accuracy", "maximize")

    data, task, manifest = _classification_fixture()
    folds = _folds(data, task, manifest)
    small_grid = mt.create_gazepoint_tuning_grid(
        engine="glm", thresholds=[0.4, 0.6], complexity=[1, 2], interpretability="high"
    )
    tuning = mt.tune_gazepoint_model(
        folds,
        task,
        small_grid,
        predictors=PREDICTORS,
        metrics=["roc_auc", "brier"],
        seed=11,
        keep_evaluations=True,
    )
    assert tuning.validation.status in {"pass", "review"}
    assert len(mt.compare_gazepoint_models(tuning)) > 0
    assert len(mt.compare_gazepoint_models(tuning, metrics="roc_auc")) > 0
    selection = mt.select_gazepoint_model(
        tuning,
        metric="roc_auc",
        direction="maximize",
        minimum_success_prop=0.5,
        rationale="predeclared human-reviewed primary metric",
    )
    assert selection.candidate_id in small_grid.candidates.candidate_id.tolist()
    assert "Human rationale" in repr(selection)
    paths = mt.write_gazepoint_model_tuning(
        tuning, tmp_path, selection=selection, prefix="tuning"
    )
    assert {"candidates", "candidate_status", "comparison", "selection", "validation"} == set(paths)

    with pytest.raises(GP3MLError, match="tuning_grid"):
        mt.tune_gazepoint_model(folds, task, object())
    with pytest.raises(GP3MLError, match="gp3ml_model_tuning"):
        mt.compare_gazepoint_models(object())
    with pytest.raises(GP3MLError, match="explicit primary"):
        mt.select_gazepoint_model(tuning, "", "maximize", rationale="x")
    with pytest.raises(GP3MLError, match="rationale"):
        mt.select_gazepoint_model(tuning, "roc_auc", "maximize", rationale="")
    with pytest.raises(GP3MLError, match="between zero and one"):
        mt.select_gazepoint_model(
            tuning, "roc_auc", "maximize", minimum_success_prop=2, rationale="x"
        )
    with pytest.raises(GP3MLError, match="No eligible"):
        mt.select_gazepoint_model(
            tuning, "missing", "maximize", rationale="x"
        )
    with pytest.raises(GP3MLError, match="gp3ml_model_tuning"):
        mt.validate_gazepoint_model_tuning(object())
    with pytest.raises(GP3MLError, match="selection"):
        mt.write_gazepoint_model_tuning(tuning, tmp_path / "bad", selection=object())

    # Explicit tie resolution through numeric complexity and unresolved tie failure.
    comparison = pd.DataFrame(
        {
            "candidate_id": ["candidate_001", "candidate_002"],
            "metric": ["roc_auc", "roc_auc"],
            "candidate_status": ["pass", "pass"],
            "success_prop": [1.0, 1.0],
            "mean": [0.8, 0.8],
        }
    )
    tie_grid = mt.create_gazepoint_tuning_grid(
        "glm", thresholds=[0.4, 0.6], complexity=[2, 1]
    )
    tie_tuning = GP3MLModelTuning(
        grid=tie_grid,
        results=[],
        comparison=comparison,
        task=task,
        predictors=PREDICTORS,
        folds_metadata=folds.metadata,
        selection=None,
    )
    chosen = mt.select_gazepoint_model(
        tie_tuning, "roc_auc", "maximize", rationale="prefer lower complexity"
    )
    assert chosen.candidate_id == "candidate_002"
    tie_grid.candidates["complexity"] = np.nan
    with pytest.raises(GP3MLError, match="remains tied"):
        mt.select_gazepoint_model(
            tie_tuning, "roc_auc", "maximize", rationale="tie remains"
        )


def test_tuning_failure_retention_nested_evaluation_and_nested_writer(tmp_path: Path):
    data, task, manifest = _classification_fixture()
    folds = _folds(data, task, manifest)
    failing_grid = mt.create_gazepoint_tuning_grid(engine="unknown", thresholds=0.5)
    tuning = mt.tune_gazepoint_model(
        folds,
        task,
        failing_grid,
        predictors=PREDICTORS,
        continue_on_error=True,
        keep_evaluations=False,
    )
    assert tuning.results[0]["status"] == "fail"
    assert tuning.results[0]["evaluation"] is None
    assert tuning.comparison.metric.isna().all()
    with pytest.raises(GP3MLError, match="Candidate"):
        mt.tune_gazepoint_model(
            folds,
            task,
            failing_grid,
            predictors=PREDICTORS,
            continue_on_error=False,
        )

    outer = folds
    nested = nr.create_gazepoint_nested_folds(outer, inner_v=2, inner_repeats=1, seed=12)
    grid = mt.create_gazepoint_tuning_grid("glm", thresholds=0.5, complexity=1)
    evaluated = nr.evaluate_gazepoint_nested_resampling(
        nested,
        task,
        grid,
        selection_metric="roc_auc",
        direction="maximize",
        predictors=PREDICTORS,
        minimum_success_prop=0.5,
        selection_rationale="nested human review",
        seed=13,
        keep_models=True,
        continue_on_error=True,
    )
    assert len(evaluated.fold_status) == len(nested.folds)
    assert evaluated.validation.status in {"pass", "review"}
    assert all(result["model"] is not None for result in evaluated.results if result["status"] != "fail")
    assert "Outer assessment predictions" in repr(evaluated)
    paths = nr.write_gazepoint_nested_evaluation(evaluated, tmp_path / "nested")
    assert "fold_status" in paths

    with pytest.raises(GP3MLError, match="gp3ml_nested_folds"):
        nr.evaluate_gazepoint_nested_resampling(
            object(), task, grid, "roc_auc", "maximize"
        )
    with pytest.raises(GP3MLError, match="gp3ml_tuning_grid"):
        nr.evaluate_gazepoint_nested_resampling(
            nested, task, object(), "roc_auc", "maximize"
        )
    with pytest.raises(GP3MLError, match="gp3ml_nested_evaluation"):
        nr.validate_gazepoint_nested_evaluation(object())
    with pytest.raises(GP3MLError, match="gp3ml_nested_evaluation"):
        nr.write_gazepoint_nested_evaluation(object(), tmp_path)

    failed_nested = nested.__deepcopy__({})
    failed_nested.folds[0]["inner"] = None
    failed_nested.folds[0]["status"] = "fail"
    failed_nested.folds[0]["error"] = "missing inner folds"
    failed_nested.audit = nr.audit_gazepoint_nested_resampling(failed_nested)
    failed_nested.validation = nr.validate_gazepoint_nested_folds(failed_nested)
    retained = nr.evaluate_gazepoint_nested_resampling(
        failed_nested,
        task,
        grid,
        "roc_auc",
        "maximize",
        predictors=PREDICTORS,
        continue_on_error=True,
    )
    assert (retained.fold_status.status == "fail").any()
    with pytest.raises(GP3MLError, match="Outer fold"):
        nr.evaluate_gazepoint_nested_resampling(
            failed_nested,
            task,
            grid,
            "roc_auc",
            "maximize",
            predictors=PREDICTORS,
            continue_on_error=False,
        )
