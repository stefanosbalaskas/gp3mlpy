from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gp3mlpy as gp
from gp3mlpy import governance_reports as gr
from gp3mlpy import model_artifacts as ma
from gp3mlpy import reproducibility as repro
from gp3mlpy import ro_crate as rc
from gp3mlpy import roadmap_reporting as rr
from gp3mlpy import robustness as rb
from gp3mlpy.exceptions import GP3MLError
from gp3mlpy.objects import (
    GP3MLCalibrationAssessment,
    GP3MLModelSelection,
    GP3MLResampleUncertainty,
    GP3MLStabilityEvaluation,
    GP3MLTargetUncertainty,
    GP3MLThresholdEvaluation,
    GP3MLThresholdStability,
)


PREDICTORS = ["tracking_ratio", "blink_rate", "fixation_duration"]


def _classification_fixture():
    data = gp.simulate_gazepoint_governed_data(
        n_participants=18, n_stimuli=4, trials_per_cell=1, seed=71
    )
    task = gp.create_gazepoint_synthetic_task(
        data, workflow="recording_quality", generalization_target="new_participants"
    )
    model = gp.fit_gazepoint_model(data, task, PREDICTORS, engine="glm", seed=71)
    return data, task, model


def _regression_fixture():
    data = pd.DataFrame(
        {
            "participant_id": [f"P{i // 2:02d}" for i in range(12)],
            "stimulus_id": [f"S{i % 3}" for i in range(12)],
            "trial_id": [f"T{i:02d}" for i in range(12)],
            "x": np.linspace(0.0, 2.0, 12),
            "group": ["a", "b"] * 6,
        }
    )
    data["score"] = 1.0 + 2.0 * data["x"]
    task = gp.declare_gazepoint_task(
        data,
        outcome="score",
        purpose="predict an explicitly observed continuous research score",
        task_type="regression",
        unit_id="trial_id",
        participant_id="participant_id",
        stimulus_id="stimulus_id",
        generalization_target="new_trials_known_participants",
    )
    model = gp.fit_gazepoint_model(data, task, ["x", "group"], engine="lm", seed=9)
    return data, task, model


def test_reproducibility_normalization_audit_and_environment_restore(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    text = (
        f"{project.resolve().as_posix()} RtmpABC123 /tmp/RtmpXYZ/a "
        "C:\\Users\\name\\AppData\\Local\\Temp\\thing 0xABCDEF12 "
        "Generated: 2026-08-30 01:02:03"
    )
    normalized = repro.normalize_gazepoint_artifact_text(text, project_path=project)
    assert "<PROJECT>" in normalized
    assert "<RTMP>" in normalized
    assert "<TEMP_PATH>" in normalized
    assert "<ADDRESS>" in normalized
    assert "Generated: <timestamp>" in normalized
    assert repro.normalize_gazepoint_artifact_text(["plain", "0xABCDEF12"])[0] == "plain"

    clean = tmp_path / "clean.txt"
    clean.write_text("stable output\n", encoding="utf-8")
    noisy = tmp_path / "noisy.md"
    noisy.write_text("Generated: 2026-08-30 01:02:03\nptr=0xABCDEF12\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "more.txt").write_text("/tmp/RtmpABC/file\n", encoding="utf-8")
    (nested / "skip.bin").write_bytes(b"\xff\xfe")

    audit_clean = repro.audit_gazepoint_reproducibility(clean)
    assert audit_clean.status == "pass"
    audit = repro.audit_gazepoint_reproducibility(tmp_path, recursive=True)
    assert audit.status == "review"
    assert audit.files_scanned >= 3
    assert {"generated_timestamp", "memory_address"}.issubset(set(audit.findings.issue))
    shallow = repro.audit_gazepoint_reproducibility(tmp_path, recursive=False)
    assert shallow.files_scanned == 2
    with pytest.raises(GP3MLError, match="at least one"):
        repro.audit_gazepoint_reproducibility([])

    paths = repro.write_gazepoint_reproducibility_audit(audit, tmp_path / "audit")
    assert all(Path(path).exists() for path in paths.values())
    with pytest.raises(GP3MLError):
        repro.write_gazepoint_reproducibility_audit(object(), tmp_path)
    with pytest.raises(GP3MLError, match="overwrite"):
        repro.write_gazepoint_reproducibility_audit(audit, tmp_path / "audit")

    monkeypatch.delenv("GP3MLPY_REPRODUCIBLE_EXAMPLES", raising=False)
    assert repro.with_gazepoint_reproducible_output(
        lambda: os.environ["GP3MLPY_REPRODUCIBLE_EXAMPLES"]
    ) == "1"
    assert "GP3MLPY_REPRODUCIBLE_EXAMPLES" not in os.environ
    monkeypatch.setenv("GP3MLPY_REPRODUCIBLE_EXAMPLES", "old")
    assert repro.with_gazepoint_reproducible_output(7) == 7
    assert os.environ["GP3MLPY_REPRODUCIBLE_EXAMPLES"] == "old"


def test_ro_crate_write_validate_and_corruption_paths(tmp_path: Path):
    source = tmp_path / "result.csv"
    source.write_text("x\n1\n", encoding="utf-8")

    crate = rc.write_gazepoint_ro_crate(
        tmp_path / "crate",
        source,
        name="gp3mlpy research output",
        description="synthetic validation artifact",
        creator_name="Researcher",
        creator_orcid="0000-0000-0000-0000",
    )
    assert rc.validate_gazepoint_ro_crate(crate).status == "pass"
    assert len(crate.file_manifest) == 1

    crate_url = rc.write_gazepoint_ro_crate(
        tmp_path / "crate-url",
        [source],
        name="crate",
        description="description",
        creator_name="Researcher",
        creator_orcid="https://orcid.org/0000-0000-0000-0001",
    )
    assert rc.validate_gazepoint_ro_crate(crate_url.path).status == "pass"

    with pytest.raises(GP3MLError, match="must exist"):
        rc.write_gazepoint_ro_crate(
            tmp_path / "missing", tmp_path / "none.txt", "x", "x", "x"
        )

    no_copy = rc.write_gazepoint_ro_crate(
        tmp_path / "no-copy",
        source,
        name="crate",
        description="description",
        creator_name="Researcher",
        copy_files=False,
    )
    assert rc.validate_gazepoint_ro_crate(no_copy).status == "fail"

    root = Path(crate.path)
    copied = root / source.name
    copied.write_text("changed\n", encoding="utf-8")
    assert rc.validate_gazepoint_ro_crate(root).status == "fail"

    bad_meta = tmp_path / "bad-meta"
    bad_meta.mkdir()
    (bad_meta / "ro-crate-metadata.json").write_text("not-json", encoding="utf-8")
    (bad_meta / "sha256-manifest.csv").write_text("file,sha256,size\n", encoding="utf-8")
    assert rc.validate_gazepoint_ro_crate(bad_meta).status == "fail"

    missing = tmp_path / "empty-crate"
    missing.mkdir()
    assert rc.validate_gazepoint_ro_crate(missing).status == "fail"


def test_robustness_stability_helpers_and_statuses():
    assert rb._numeric_metrics({"a": 1, "flag": True}) == {"a": 1.0}
    assert rb._numeric_metrics(pd.Series({"a": np.float64(2.0)})) == {"a": 2.0}
    assert rb._numeric_metrics(pd.DataFrame({"a": [3.0], "text": ["x"]})) == {"a": 3.0}
    with pytest.raises(GP3MLError, match="Evaluator"):
        rb._numeric_metrics([1, 2])
    with pytest.raises(GP3MLError, match="evaluator"):
        rb.evaluate_gazepoint_seed_stability([1], None)

    seed = rb.evaluate_gazepoint_seed_stability(
        [1, 1, 2], lambda seed: {"metric": float(seed)}
    )
    feature = rb.evaluate_gazepoint_feature_stability(
        ["x", "x", "z"], lambda excluded_feature: pd.Series({"metric": len(excluded_feature)})
    )
    missing = rb.evaluate_gazepoint_missingness_sensitivity(
        {"none": 0.0, "some": 0.2},
        lambda scenario, name: pd.DataFrame({"metric": [1.0 - scenario]}),
    )
    assert list(seed.results.seed) == [1, 2]
    assert list(feature.results.excluded_feature) == ["x", "z"]
    assert set(missing.results.scenario) == {"none", "some"}
    with pytest.raises(GP3MLError, match="named list"):
        rb.evaluate_gazepoint_missingness_sensitivity({}, lambda **_: {"m": 1})

    stable_eval = GP3MLThresholdEvaluation(
        thresholds=pd.DataFrame({"threshold": [0.10, 0.20, 0.30], "score": [1.0, 0.99, 0.8]})
    )
    stable = rb.evaluate_gazepoint_threshold_stability(stable_eval, "score", tolerance=0.02)
    assert stable.status == "stable"
    review_eval = GP3MLThresholdEvaluation(
        thresholds=pd.DataFrame({"threshold": [0.10, 0.15, 0.30], "score": [1.0, 0.99, 0.8]})
    )
    assert rb.evaluate_gazepoint_threshold_stability(
        review_eval, "score", tolerance=0.02
    ).status == "review"
    unstable_eval = GP3MLThresholdEvaluation(
        thresholds=pd.DataFrame({"threshold": [0.10, 0.20], "score": [1.0, 0.5]})
    )
    assert rb.evaluate_gazepoint_threshold_stability(
        unstable_eval, "score", tolerance=0.01
    ).status == "unstable"
    minimize_eval = GP3MLThresholdEvaluation(
        thresholds=pd.DataFrame({"threshold": [0.10, 0.20], "loss": [0.10, 0.11]})
    )
    assert rb.evaluate_gazepoint_threshold_stability(
        minimize_eval, "loss", direction="minimize", tolerance=0.2
    ).metric == "loss"
    with pytest.raises(GP3MLError, match="Invalid"):
        rb.evaluate_gazepoint_threshold_stability(object(), "score")
    with pytest.raises(GP3MLError, match="direction"):
        rb.evaluate_gazepoint_threshold_stability(stable_eval, "score", direction="sideways")
    with pytest.raises(GP3MLError, match="Unknown metric"):
        rb.evaluate_gazepoint_threshold_stability(stable_eval, "missing")

    assert rb.audit_gazepoint_model_robustness().status == "review"
    summary = pd.DataFrame(
        {
            "metric": ["pass", "review", "fail", "nonfinite"],
            "minimum": [1.0, 1.0, 1.0, 0.0],
            "maximum": [1.0, 1.0, 1.0, 0.0],
            "sd": [0.01, 0.08, 0.20, np.nan],
        }
    )
    stability = GP3MLStabilityEvaluation(kind="seed", results=pd.DataFrame(), summary=summary)
    audit = rb.audit_gazepoint_model_robustness(
        seed_stability=stability,
        threshold_stability=GP3MLThresholdStability(
            status="stable", metric="score", threshold_span=np.array([0.1, 0.2])
        ),
    )
    assert {"pass", "review", "fail"}.issubset(set(audit.findings.status))
    assert audit.status == "fail"


def test_research_bundle_end_to_end_and_validation_failures():
    with pytest.raises(GP3MLError, match="at least 6"):
        gp.simulate_gazepoint_research_handoffs(n_participants=5, n_stimuli=2)

    bundle = gp.simulate_gazepoint_research_handoffs(
        n_participants=6, n_stimuli=2, trials_per_stimulus=1, seed=5
    )
    validation = gp.validate_gazepoint_research_bundle(bundle)
    assert validation.status == "pass"
    assert len(validation.bundle.data) == 12
    with pytest.raises(GP3MLError, match="simulate"):
        gp.validate_gazepoint_research_bundle(object())

    missing_source = bundle.__deepcopy__({})
    del missing_source.handoffs["gp3sequences"]
    assert gp.validate_gazepoint_research_bundle(missing_source).status == "fail"

    wrong_target = bundle.__deepcopy__({})
    wrong_target.generalization_target = "new_stimuli"
    assert gp.validate_gazepoint_research_bundle(wrong_target).status == "fail"


def test_model_artifact_validation_portability_and_private_helpers(tmp_path: Path, monkeypatch):
    data, task, model = _classification_fixture()

    assert ma._schema(model) == {"predictors": PREDICTORS, "classes": None}
    schema = ma._schema(model, data)
    assert set(schema["classes"]) == set(PREDICTORS)
    with pytest.raises(GP3MLError, match="missing model predictors"):
        ma._schema(model, data.drop(columns=[PREDICTORS[0]]))

    monkeypatch.setattr(
        ma.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no git")),
    )
    assert ma._git_sha(tmp_path.as_posix()) is None

    assert ma._safe_model_fingerprint(None) == {"model": None}
    fingerprint = ma._safe_model_fingerprint(model)
    assert fingerprint["engine"] == "glm"
    assert fingerprint["task"]["outcome"] == task.outcome

    artifact = ma.create_gazepoint_model_artifact(model, reference_data=data, bundle_model=True)
    assert artifact.metadata["bundled"] is False
    assert artifact.metadata["bundle_error"] is not None
    assert ma.validate_gazepoint_model_artifact(artifact).status == "pass"
    restored = ma.restore_gazepoint_model_artifact(artifact)
    assert restored.artifact_hash == artifact.artifact_hash

    portable = ma.test_gazepoint_model_portability(artifact, newdata=data)
    assert portable.status == "pass"
    fresh = ma.test_gazepoint_model_portability(artifact, newdata=data, fresh_process=True)
    assert fresh.status == "fail"
    assert fresh.fresh_process_ok is False

    without_ref = ma.create_gazepoint_model_artifact(model, bundle_model=False)
    assert without_ref.metadata["bundle_error"] is None
    assert ma.test_gazepoint_model_portability(without_ref).status == "pass"

    assert ma.validate_gazepoint_model_artifact(object()).status == "fail"
    with pytest.raises(GP3MLError, match="validation failed"):
        ma.restore_gazepoint_model_artifact(object())
    with pytest.raises(GP3MLError, match="invalid"):
        ma.test_gazepoint_model_portability(object())
    with pytest.raises(GP3MLError, match="required"):
        ma.create_gazepoint_model_artifact(None)

    review = artifact.__deepcopy__({})
    review.task = None
    assert ma.validate_gazepoint_model_artifact(review, verify_hash=False).status == "review"
    missing_schema = artifact.__deepcopy__({})
    missing_schema.predictor_schema = None
    assert ma.validate_gazepoint_model_artifact(missing_schema, verify_hash=False).status == "fail"
    missing_metadata = artifact.__deepcopy__({})
    missing_metadata.metadata = None
    assert ma.validate_gazepoint_model_artifact(missing_metadata, verify_hash=False).status == "fail"
    missing_hash = artifact.__deepcopy__({})
    missing_hash.artifact_hash = None
    assert ma.validate_gazepoint_model_artifact(missing_hash).status == "fail"
    corrupt = artifact.__deepcopy__({})
    corrupt.metadata["platform"] = "changed"
    assert ma.validate_gazepoint_model_artifact(corrupt).status == "fail"

    class NoPredict:
        pass

    class BadPredict:
        def predict(self, newdata):
            raise RuntimeError("boom")

    class TextPredict:
        def predict(self, newdata):
            return np.asarray(["a", "b"], dtype=object)

    assert ma._predict_for_portability(NoPredict(), data) is None
    assert ma._predict_for_portability(BadPredict(), data) is None
    assert np.array_equal(ma._predict_for_portability(TextPredict(), data.iloc[:2]), ["a", "b"])


def test_governance_reports_model_cards_external_validation_and_reproducibility(tmp_path: Path):
    data, task, model = _classification_fixture()

    assert gr._markdown_table(pd.DataFrame()) == ["_No rows._"]
    escaped = gr._markdown_table(pd.DataFrame({"x": ["a|b", None]}))
    assert "a\\|b" in escaped[2]
    assert "" in escaped[3]

    card = gr.create_gazepoint_model_card(
        model,
        intended_use="research review",
        evaluation=pd.DataFrame({"metric": ["roc_auc"], "value": [0.8]}),
        limitations="synthetic example",
    )
    md = Path(gr.write_gazepoint_model_card(card, tmp_path / "card.md"))
    js = Path(gr.write_gazepoint_model_card(card, tmp_path / "card.json", format="json"))
    assert "Model card" in md.read_text(encoding="utf-8")
    assert json.loads(js.read_text(encoding="utf-8"))["title"].startswith("Model card")
    with pytest.raises(GP3MLError, match="created by"):
        gr.write_gazepoint_model_card(object(), tmp_path / "bad.md")
    with pytest.raises(GP3MLError, match="format"):
        gr.write_gazepoint_model_card(card, tmp_path / "bad.txt", format="html")
    with pytest.raises(GP3MLError, match="File exists"):
        gr.write_gazepoint_model_card(card, md)

    assert gr._json_ready(float("nan")) is None
    assert gr._json_ready(np.int64(3)) == 3
    assert gr._json_ready(tmp_path) == str(tmp_path)
    assert gr._json_ready(pd.Series([1, 2])) == [1, 2]
    assert gr._json_ready(pd.Categorical(["a", "b"])) == ["a", "b"]
    assert gr._json_ready({"x": np.array([1, 2])}) == {"x": [1, 2]}
    assert gr._json_ready(lambda: 1) == "<function>"

    external = data.copy()
    validation = gr.evaluate_external_validation(model, external, label="same-site", bootstrap=4, seed=4)
    assert validation.label == "same-site"
    report = gr.create_external_validation_report(validation, limitations="synthetic")
    path = Path(gr.write_external_validation_report(report, tmp_path / "external.md"))
    assert "External validation" in path.read_text(encoding="utf-8")
    with pytest.raises(GP3MLError, match="fitted"):
        gr.evaluate_external_validation(object(), external)
    with pytest.raises(GP3MLError, match="evaluate_external_validation"):
        gr.create_external_validation_report(object())
    with pytest.raises(GP3MLError, match="external-validation report"):
        gr.write_external_validation_report(object(), tmp_path / "x.md")

    rdata, _, rmodel = _regression_fixture()
    rval = gr.evaluate_external_validation(rmodel, rdata, label="regression", bootstrap=2)
    assert rval.calibration is None
    rreport = gr.create_external_validation_report(rval)
    rpath = Path(gr.write_external_validation_report(rreport, tmp_path / "regression.md"))
    assert "Not applicable" in rpath.read_text(encoding="utf-8")

    no_git = gr._git_info(tmp_path / "not-a-repo")
    assert no_git == {"commit": None, "branch": None, "clean": None}
    repro_report = gr.create_gazepoint_reproducibility_report(
        objects={"task": task},
        data=data.iloc[:3],
        seeds={"model": 71},
        notes="synthetic",
        project_path=tmp_path,
    )
    out = Path(gr.write_gazepoint_reproducibility_report(repro_report, tmp_path / "repro.md"))
    text = out.read_text(encoding="utf-8")
    assert "Fingerprints" in text and "model: 71" in text
    empty_report = gr.create_gazepoint_reproducibility_report(project_path=tmp_path)
    empty_text = Path(
        gr.write_gazepoint_reproducibility_report(empty_report, tmp_path / "repro-empty.md")
    ).read_text(encoding="utf-8")
    assert "No objects supplied" in empty_text
    assert "No seeds supplied" in empty_text
    assert "None." in empty_text
    with pytest.raises(GP3MLError, match="reproducibility report"):
        gr.write_gazepoint_reproducibility_report(object(), tmp_path / "bad-repro.md")


def test_release_reporting_evidence_and_validation_paths(tmp_path: Path):
    data, _, model = _classification_fixture()

    assert rr._as_character_vector(None) == []
    assert rr._as_character_vector("x") == ["x"]
    assert rr._as_character_vector((1, 2)) == ["1", "2"]

    with pytest.raises(GP3MLError, match="fitted"):
        rr.create_gazepoint_release_model_card(object(), "x", limitations=["x"])
    with pytest.raises(GP3MLError, match="explicit limitation"):
        rr.create_gazepoint_release_model_card(model, "research", limitations=[])
    with pytest.raises(GP3MLError, match="selection"):
        rr.create_gazepoint_release_model_card(
            model, "research", selection=object(), limitations=["x"]
        )
    with pytest.raises(GP3MLError, match="uncertainty"):
        rr.create_gazepoint_release_model_card(
            model, "research", uncertainty=object(), limitations=["x"]
        )
    with pytest.raises(GP3MLError, match="transportability"):
        rr.create_gazepoint_release_model_card(
            model, "research", transportability=object(), limitations=["x"]
        )

    card = rr.create_gazepoint_release_model_card(
        model,
        intended_use="research review",
        evaluation=pd.DataFrame({"metric": ["roc_auc"], "value": [0.8]}),
        limitations=["synthetic data", "no deployment"],
    )
    assert card.selection_procedure_recorded is False
    assert card.external_validation_status == "not_externally_validated"
    assert rr._release_card_metrics(card).shape[0] == 1
    assert rr._release_card_selection(None).empty
    assert rr._release_card_uncertainty(None).empty

    md = Path(rr.write_gazepoint_release_model_card(card, tmp_path / "release.md"))
    js = Path(rr.write_gazepoint_release_model_card(card, tmp_path / "release.json", format="json"))
    assert "No governed model-selection procedure" in md.read_text(encoding="utf-8")
    assert json.loads(js.read_text(encoding="utf-8"))["deployment_status"] == "research_review_only"
    with pytest.raises(GP3MLError, match="release_model_card"):
        rr.write_gazepoint_release_model_card(object(), tmp_path / "bad.md")
    with pytest.raises(GP3MLError, match="format"):
        rr.write_gazepoint_release_model_card(card, tmp_path / "bad.html", format="html")

    selection = GP3MLModelSelection(
        candidate_id="glm",
        primary_metric="roc_auc",
        direction="maximize",
        primary_value=0.8,
        minimum_success_prop=0.9,
        rationale="declared",
        autonomous_selection=False,
        refit_performed=False,
    )
    uncertainty = GP3MLResampleUncertainty(
        unit="fold",
        generalization_target=model.task.generalization_target,
        summary=pd.DataFrame({"metric": ["roc_auc"], "mean": [0.8]}),
        distribution=pd.DataFrame(),
        limitations="fold distribution",
    )
    calibration = GP3MLCalibrationAssessment(summary=pd.DataFrame({"brier": [0.1]}))
    detailed = rr.create_gazepoint_release_model_card(
        model,
        intended_use="research review",
        selection=selection,
        uncertainty=uncertainty,
        calibration=calibration,
        limitations=["synthetic"],
    )
    detailed_text = Path(
        rr.write_gazepoint_release_model_card(detailed, tmp_path / "release-detailed.md")
    ).read_text(encoding="utf-8")
    assert "glm" in detailed_text
    assert "fold distribution" in detailed_text
    assert not rr._release_card_selection(selection).empty
    assert not rr._release_card_uncertainty(uncertainty).empty
    target_uncertainty = GP3MLTargetUncertainty(
        intervals=pd.DataFrame({"metric": ["m"], "lower": [0.1]})
    )
    assert rr._release_card_uncertainty(target_uncertainty).shape[0] == 1
    unsupported_card = card.__deepcopy__({})
    unsupported_card.evaluation = object()
    assert rr._release_card_metrics(unsupported_card).empty

    file_path = tmp_path / "artifact.txt"
    file_path.write_text("release\n", encoding="utf-8")
    evidence = rr.create_gazepoint_release_evidence(
        objects={"task": model.task}, files={"artifact": file_path}, notes="release note"
    )
    assert len(evidence.object_hashes) == 1
    assert len(evidence.file_md5["artifact"]) == 32
    assert "Object hashes: 1" in repr(evidence)
    empty_evidence = rr.create_gazepoint_release_evidence()
    assert empty_evidence.object_hashes == {}
    with pytest.raises(GP3MLError, match="named vector"):
        rr.create_gazepoint_release_evidence(files=[file_path])
    with pytest.raises(GP3MLError, match="existing paths"):
        rr.create_gazepoint_release_evidence(files={"missing": tmp_path / "missing"})
