from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gp3mlpy import calibration as cal
from gp3mlpy import metrics as met
from gp3mlpy import target_uncertainty as tu
from gp3mlpy import task_governance as tg
from gp3mlpy.exceptions import GP3MLError
from gp3mlpy.objects import GP3MLResampleEvaluation, GP3MLTargetUncertainty


def _classification_data(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_id": [f"T{i:02d}" for i in range(n)],
            "participant_id": [f"P{i // 2:02d}" for i in range(n)],
            "stimulus_id": [f"S{i % 4}" for i in range(n)],
            "condition": pd.Categorical(
                ["control", "target"] * (n // 2),
                categories=["control", "target"],
            ),
            "x": np.linspace(-1.0, 1.0, n),
        }
    )


def _classification_task(data: pd.DataFrame):
    return tg.declare_gazepoint_task(
        data,
        outcome="condition",
        purpose="predict explicitly assigned experimental condition",
        task_type="classification",
        unit_id="trial_id",
        participant_id="participant_id",
        stimulus_id="stimulus_id",
        generalization_target="new_participants",
        positive="target",
    )


def _regression_data(n: int = 12) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "trial_id": [f"T{i:02d}" for i in range(n)],
            "participant_id": [f"P{i // 2:02d}" for i in range(n)],
            "stimulus_id": [f"S{i % 3}" for i in range(n)],
            "x": np.linspace(0.0, 1.0, n),
        }
    )
    data["score"] = 2.0 * data["x"] + 1.0
    return data


def _regression_task(data: pd.DataFrame):
    return tg.declare_gazepoint_task(
        data,
        outcome="score",
        purpose="predict an explicitly observed continuous research score",
        task_type="regression",
        unit_id="trial_id",
        participant_id="participant_id",
        stimulus_id="stimulus_id",
        generalization_target="new_trials_known_participants",
    )


def test_task_declaration_use_case_and_role_validation_branches():
    data = _classification_data()
    task = _classification_task(data)
    assert task.levels == ["control", "target"]
    assert task.positive == "target"
    assert task.negative == "control"
    assert tg.assert_gp3ml_use_case(task, data)
    assert len(tg.gp3ml_prohibited_uses()) >= 9
    assert tg._ordered_class_levels(data["condition"]) == ["control", "target"]
    plain = pd.Series(["z", "a", "z"])
    assert tg._ordered_class_levels(plain) == ["a", "z"]

    auto_positive = tg.declare_gazepoint_task(
        data,
        outcome="condition",
        purpose="predict explicitly assigned condition",
        task_type="classification",
        unit_id="trial_id",
        participant_id="participant_id",
        stimulus_id="stimulus_id",
    )
    assert auto_positive.positive == "target"

    reg = _regression_data()
    reg_task = _regression_task(reg)
    assert reg_task.levels is None and reg_task.positive is None

    bad_cases = [
        ({"task_type": "other"}, "task_type"),
        ({"generalization_target": "other"}, "generalization_target"),
        ({"outcome": ""}, "outcome"),
        ({"unit_id": None}, "unit_id"),
        ({"purpose": " "}, "purpose"),
        ({"positive": "missing"}, "positive"),
    ]
    base = dict(
        data=data,
        outcome="condition",
        purpose="predict assigned condition",
        task_type="classification",
        unit_id="trial_id",
        participant_id="participant_id",
        stimulus_id="stimulus_id",
        generalization_target="new_trials_known_participants",
        positive="target",
    )
    for changes, match in bad_cases:
        args = dict(base)
        args.update(changes)
        with pytest.raises(GP3MLError, match=match):
            tg.declare_gazepoint_task(**args)

    three = data.copy()
    three["condition"] = ["a", "b", "c", "a", "b"] * 4
    with pytest.raises(GP3MLError, match="exactly two"):
        tg.declare_gazepoint_task(
            three,
            "condition",
            "predict assigned category",
            unit_id="trial_id",
        )

    bad_reg = reg.copy()
    bad_reg["score"] = ["x"] * len(bad_reg)
    with pytest.raises(GP3MLError, match="numeric"):
        tg.declare_gazepoint_task(
            bad_reg,
            "score",
            "predict observed score",
            task_type="regression",
            unit_id="trial_id",
        )

    with pytest.raises(GP3MLError, match="task declaration"):
        tg.assert_gp3ml_use_case(object())
    prohibited = task.__deepcopy__({})
    prohibited.purpose = "infer stress"
    with pytest.raises(GP3MLError, match="prohibited"):
        tg.assert_gp3ml_use_case(prohibited)
    sensitive = task.__deepcopy__({})
    sensitive.sensitive_outcome = True
    with pytest.raises(GP3MLError, match="prohibited"):
        tg.assert_gp3ml_use_case(sensitive)
    latent = task.__deepcopy__({})
    latent.observed_outcome = False
    with pytest.raises(GP3MLError, match="explicitly observed"):
        tg.assert_gp3ml_use_case(latent)

    no_p = task.__deepcopy__({})
    no_p.participant_id = None
    with pytest.raises(GP3MLError, match="Participant-level"):
        tg.assert_gp3ml_use_case(no_p)
    no_s = task.__deepcopy__({})
    no_s.generalization_target = "new_stimuli"
    no_s.stimulus_id = None
    with pytest.raises(GP3MLError, match="Stimulus-level"):
        tg.assert_gp3ml_use_case(no_s)
    crossed = task.__deepcopy__({})
    crossed.generalization_target = "new_participants_and_new_stimuli"
    crossed.stimulus_id = None
    with pytest.raises(GP3MLError, match="Crossed"):
        tg.assert_gp3ml_use_case(crossed)
    with pytest.raises(GP3MLError, match="Missing task columns"):
        tg.assert_gp3ml_use_case(task, data.drop(columns=["trial_id"]))

    roles = tg.validate_gazepoint_ml_roles(data, task, "x")
    assert roles.status == "review"
    assert "feature_manifest" in set(roles.issues.check)
    bad_roles = tg.validate_gazepoint_ml_roles(
        data.assign(condition=data["condition"].astype(object)),
        task,
        ["missing", "condition", "participant_id"],
    )
    assert bad_roles.status == "fail"
    assert {"predictors_exist", "outcome_not_predictor", "identifiers_not_predictors"}.issubset(
        set(bad_roles.issues.check)
    )

    incomplete = data.copy()
    incomplete.loc[0, "condition"] = None
    assert tg.validate_gazepoint_ml_roles(incomplete, task, ["x"]).status == "fail"

    low_support = data.iloc[:6].copy()
    low_task = tg.declare_gazepoint_task(
        low_support,
        "condition",
        "predict assigned condition",
        unit_id="trial_id",
        participant_id="participant_id",
        stimulus_id="stimulus_id",
        generalization_target="new_participants",
    )
    low_roles = tg.validate_gazepoint_ml_roles(low_support, low_task, ["x"])
    assert low_roles.checks.loc[
        low_roles.checks.check == "classification_level_support", "status"
    ].iloc[0] == "review"

    tiny = data.iloc[:3].copy()
    tiny["condition"] = pd.Categorical(["control", "target", "target"])
    tiny_task = tg.declare_gazepoint_task(
        tiny,
        "condition",
        "predict assigned condition",
        unit_id="trial_id",
        participant_id="participant_id",
        stimulus_id="stimulus_id",
        generalization_target="new_participants",
    )
    assert tg.validate_gazepoint_ml_roles(tiny, tiny_task, ["x"]).status == "fail"

    reg_roles = tg.validate_gazepoint_ml_roles(reg, reg_task, ["x"])
    assert reg_roles.checks.loc[
        reg_roles.checks.check == "classification_level_support", "detail"
    ].iloc[0] == "not applicable"


def test_metric_helpers_and_classification_branch_cases():
    series = pd.Series(["a", "b"])
    assert met._as_object_array(series).dtype == object
    assert met._as_object_array(np.array([1, 2])).tolist() == [1, 2]
    assert met._missing_mask(np.array([1, np.nan], dtype=object)).tolist() == [False, True]

    y, pos, neg = met._binary_values(["a", "b", None])
    assert (pos, neg) == ("b", "a")
    assert np.isnan(y[-1])
    with pytest.raises(GP3MLError, match="exactly two"):
        met._binary_values(["a", "a"])
    with pytest.raises(GP3MLError, match="Unknown positive"):
        met._binary_values(["a", "b"], positive="c")

    assert np.isnan(met._auc(np.array([1.0, 1.0]), np.array([0.2, 0.8])))
    assert np.isnan(met._pr_auc(np.array([0.0, 0.0]), np.array([0.2, 0.8])))
    assert met._safe(1, 0) != met._safe(1, 0)
    assert met._safe(1, 2) == 0.5

    truth = ["control", "control", "target", "target"]
    probability = [0.1, 0.8, 0.4, 0.9]
    metrics = met.gazepoint_classification_metrics(truth, probability, positive="target")
    assert metrics.loc[0, "n"] == 4
    assert np.isfinite(metrics.loc[0, "roc_auc"])
    provided = met.gazepoint_classification_metrics(
        truth,
        probability,
        predicted=["control", None, "target", "target"],
        positive="target",
    )
    assert provided.loc[0, "n"] == 4
    with pytest.raises(GP3MLError, match="probability"):
        met.gazepoint_classification_metrics(truth, [0.1])
    with pytest.raises(GP3MLError, match="predicted"):
        met.gazepoint_classification_metrics(truth, probability, predicted=["control"])

    degenerate = met.gazepoint_classification_metrics(
        ["control", "target"],
        [np.nan, np.nan],
        predicted=[None, None],
        positive="target",
    )
    assert np.isnan(degenerate.loc[0, "accuracy"])
    assert np.isnan(degenerate.loc[0, "balanced_accuracy"])
    assert np.isnan(degenerate.loc[0, "f1"])
    assert np.isnan(degenerate.loc[0, "mcc"])
    assert np.isnan(degenerate.loc[0, "brier"])
    assert np.isnan(degenerate.loc[0, "log_loss"])


def test_regression_dispatch_and_bootstrap_metric_paths():
    reg = _regression_data()
    task = _regression_task(reg)
    truth = reg["score"].to_numpy()
    pred = truth + np.linspace(-0.1, 0.1, len(truth))
    metrics = met.gazepoint_regression_metrics(truth, pred)
    assert metrics.loc[0, "n"] == len(truth)
    assert np.isfinite(metrics.loc[0, "correlation"])
    with pytest.raises(GP3MLError, match="prediction"):
        met.gazepoint_regression_metrics([1, 2], [1])

    empty = met.gazepoint_regression_metrics([np.nan], [np.nan])
    assert empty.loc[0, "n"] == 0
    assert np.isnan(empty.loc[0, "r_squared"])
    one = met.gazepoint_regression_metrics([1.0], [1.0])
    assert np.isnan(one.loc[0, "correlation"])
    constant = met.gazepoint_regression_metrics([1.0, 1.0], [1.0, 2.0])
    assert np.isnan(constant.loc[0, "r_squared"])

    dispatched = met.gazepoint_performance_metrics(task, truth, prediction=pred)
    assert "rmse" in dispatched
    with pytest.raises(GP3MLError, match="prediction"):
        met.gazepoint_performance_metrics(task, truth)

    uncertainty = met.bootstrap_gazepoint_metrics(
        task, truth, prediction=pred, bootstrap=5, seed=2
    )
    assert len(uncertainty.draws) == 5
    with pytest.raises(GP3MLError, match="positive"):
        met.bootstrap_gazepoint_metrics(task, truth, prediction=pred, bootstrap=0)
    with pytest.raises(GP3MLError, match="At least two"):
        met.bootstrap_gazepoint_metrics(task, [1.0], prediction=[1.0])
    with pytest.raises(GP3MLError, match="prediction"):
        met.bootstrap_gazepoint_metrics(task, truth, prediction=[1.0], bootstrap=2)

    cdata = _classification_data(20)
    ctask = _classification_task(cdata)
    probs = np.linspace(0.1, 0.9, len(cdata))
    cunc = met.bootstrap_gazepoint_metrics(
        ctask, cdata.condition, probability=probs, bootstrap=4, seed=3
    )
    assert len(cunc.draws) == 4
    with pytest.raises(GP3MLError, match="probability"):
        met.bootstrap_gazepoint_metrics(ctask, cdata.condition, probability=[0.1], bootstrap=2)

    malformed_task = ctask.__deepcopy__({})
    three_truth = ["a", "b", "c", "a"]
    malformed_task.levels = ["a", "b", "c"]
    malformed_task.positive = "c"
    with pytest.raises(GP3MLError, match="two truth levels|two observed classes"):
        met.bootstrap_gazepoint_metrics(
            malformed_task,
            three_truth,
            probability=[0.1, 0.3, 0.8, 0.2],
            bootstrap=2,
        )


def test_platt_isotonic_calibration_and_assessment_branches(monkeypatch):
    truth = ["control", "control", "target", "target", "control", "target"]
    probability = np.array([0.1, 0.3, 0.7, 0.9, 0.4, 0.6])

    platt = cal.fit_gazepoint_calibrator(truth, probability, positive="target", method="platt")
    applied = cal.apply_gazepoint_calibrator(platt, [0.2, 0.8])
    assert applied.shape == (2,)
    isotonic = cal.fit_gazepoint_calibrator(
        truth, probability, positive="target", method="isotonic"
    )
    iso_applied = cal.apply_gazepoint_calibrator(isotonic, [0.0, 0.5, 1.0])
    assert np.all((iso_applied > 0) & (iso_applied < 1))
    with pytest.raises(GP3MLError, match="method"):
        cal.fit_gazepoint_calibrator(truth, probability, method="bad")
    with pytest.raises(GP3MLError, match="probability"):
        cal.fit_gazepoint_calibrator(truth, [0.1], method="isotonic")
    with pytest.raises(GP3MLError, match="fitted"):
        cal.apply_gazepoint_calibrator(object(), probability)

    pava = cal._pava(np.array([0.8, 0.2, 0.6]), np.array([1.0, 1.0, 1.0]))
    assert pava[0] <= pava[1] <= pava[2]
    iso_fit = cal._fit_isotonic(
        np.array([0.2, 0.2, 0.8]), np.array([0.0, 1.0, 1.0])
    )
    assert len(iso_fit["x"]) == 2

    assessment = cal.assess_gazepoint_calibration(
        truth, probability, positive="target", bins=3, bootstrap=3, seed=4
    )
    assert len(assessment.intervals) > 0
    no_boot = cal.assess_gazepoint_calibration(
        truth, probability, positive="target", bins=2, bootstrap=0
    )
    assert no_boot.bootstrap == 0 and no_boot.intervals.empty
    with pytest.raises(GP3MLError, match="probability"):
        cal.assess_gazepoint_calibration(truth, [0.1])

    original = cal._fit_platt
    monkeypatch.setattr(
        cal,
        "_fit_platt",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()),
    )
    summary, reliability = cal._calibration_core(
        np.array([0.0, 1.0]), np.array([0.2, 0.8]), 2
    )
    assert np.isnan(summary.loc[0, "intercept"])
    assert len(reliability) == 2
    monkeypatch.setattr(cal, "_fit_platt", original)


def test_cluster_indices_target_uncertainty_validation_and_writers(tmp_path: Path, monkeypatch):
    rng = np.random.RandomState(1)
    idx = tu._cluster_bootstrap_indices(["p1", "p1", "p2", "p2"], rng)
    assert len(idx) == 4
    with pytest.raises(GP3MLError, match="missing or empty"):
        tu._cluster_bootstrap_indices(["p1", None], rng)
    with pytest.raises(GP3MLError, match="equal length"):
        tu._two_way_indices(["p1"], ["s1", "s2"], rng)
    with pytest.raises(GP3MLError, match="may not be missing"):
        tu._two_way_indices(["p1", None], ["s1", "s2"], rng)
    two = tu._two_way_indices(
        ["p1", "p1", "p2", "p2"], ["s1", "s2", "s1", "s2"], rng
    )
    assert len(two) > 0

    class EmptyChoiceRng:
        def choice(self, values, size, replace=True):
            return np.asarray(["missing"] * size, dtype=object)

    assert len(
        tu._two_way_indices(
            ["p1", "p2"], ["s1", "s2"], EmptyChoiceRng(), max_attempts=2
        )
    ) == 0
    obs = tu._observation_indices(["a", "a", "b", "b"], True, True, rng)
    assert len(obs) == 4
    assert len(tu._observation_indices(["a", "b", "c"], True, True, rng)) == 0
    assert len(tu._observation_indices([1, 2, 3], False, True, rng)) == 3

    data = _classification_data(20)
    task = _classification_task(data)
    probs = np.where(data.condition.astype(str) == "target", 0.8, 0.2)
    for unit in ["observation", "participant", "stimulus", "participant_and_stimulus"]:
        result = tu.bootstrap_gazepoint_metrics_by_unit(
            task,
            data.condition,
            probability=probs,
            participant_id=data.participant_id,
            stimulus_id=data.stimulus_id,
            unit=unit,
            bootstrap=3,
            seed=7,
        )
        assert result.successful_replicates >= 1
        assert tu.validate_gazepoint_target_uncertainty(result).status == "pass"
        paths = tu.write_gazepoint_target_uncertainty(
            result, tmp_path / unit, prefix="uncertainty"
        )
        assert "intervals" in paths

    unstratified = tu.bootstrap_gazepoint_metrics_by_unit(
        task,
        data.condition,
        probability=probs,
        unit="observation",
        bootstrap=2,
        stratify_observations=False,
    )
    assert unstratified.successful_replicates == 2

    invalid_calls = [
        ({"unit": "bad"}, "Unknown resampling"),
        ({"bootstrap": 0}, "positive"),
        ({"truth": ["control"]}, "At least two"),
        ({"probability": [0.1]}, "probability"),
        ({"unit": "participant", "participant_id": None}, "participant_id"),
        ({"unit": "stimulus", "stimulus_id": None}, "stimulus_id"),
    ]
    base = dict(
        task=task,
        truth=data.condition,
        probability=probs,
        participant_id=data.participant_id,
        stimulus_id=data.stimulus_id,
        bootstrap=2,
    )
    for changes, match in invalid_calls:
        args = dict(base)
        args.update(changes)
        with pytest.raises(GP3MLError, match=match):
            tu.bootstrap_gazepoint_metrics_by_unit(**args)

    reg = _regression_data()
    rtask = _regression_task(reg)
    with pytest.raises(GP3MLError, match="prediction"):
        tu.bootstrap_gazepoint_metrics_by_unit(
            rtask, reg.score, prediction=[1.0], bootstrap=2
        )
    runc = tu.bootstrap_gazepoint_metrics_by_unit(
        rtask,
        reg.score,
        prediction=reg.score + 0.1,
        unit="observation",
        bootstrap=2,
    )
    assert runc.unit == "observation"

    monkeypatch.setattr(tu, "_observation_indices", lambda *args, **kwargs: np.array([], dtype=int))
    with pytest.raises(GP3MLError, match="Every bootstrap replicate failed"):
        tu.bootstrap_gazepoint_metrics_by_unit(
            task, data.condition, probability=probs, bootstrap=2
        )

    with pytest.raises(GP3MLError, match="uncertainty object"):
        tu.validate_gazepoint_target_uncertainty(object())
    invalid = GP3MLTargetUncertainty(
        unit=None,
        generalization_target=None,
        limitations=None,
        failed_replicates=None,
    )
    assert tu.validate_gazepoint_target_uncertainty(invalid).status == "fail"


def test_resample_uncertainty_fold_repeat_empty_nonfinite_and_writer(tmp_path: Path):
    metrics = pd.DataFrame(
        {
            "repeat": [1, 1, 2, 2],
            "fold": [1, 2, 1, 2],
            "fold_id": ["r1f1", "r1f2", "r2f1", "r2f2"],
            "metric": ["roc_auc"] * 4,
            "value": [0.7, 0.8, np.nan, 0.9],
        }
    )
    evaluation = GP3MLResampleEvaluation(
        metrics=metrics,
        generalization_target="new_participants",
    )
    fold = tu.summarize_gazepoint_resample_uncertainty(evaluation, unit="fold")
    repeat = tu.summarize_gazepoint_resample_uncertainty(evaluation, unit="repeat")
    assert fold.summary.loc[0, "n_units"] == 3
    assert repeat.summary.loc[0, "n_units"] == 2
    assert tu.validate_gazepoint_target_uncertainty(fold).status == "pass"
    paths = tu.write_gazepoint_target_uncertainty(fold, tmp_path, prefix="resample")
    assert "summary" in paths

    with pytest.raises(GP3MLError, match="grouped or nested"):
        tu.summarize_gazepoint_resample_uncertainty(object())
    with pytest.raises(GP3MLError, match="unit"):
        tu.summarize_gazepoint_resample_uncertainty(evaluation, unit="participant")
    empty = evaluation.__deepcopy__({})
    empty.metrics = pd.DataFrame()
    with pytest.raises(GP3MLError, match="no metric values"):
        tu.summarize_gazepoint_resample_uncertainty(empty)

    all_nan = evaluation.__deepcopy__({})
    all_nan.metrics = metrics.assign(value=np.nan)
    nan_summary = tu.summarize_gazepoint_resample_uncertainty(all_nan)
    assert np.isnan(nan_summary.summary.loc[0, "mean"])
