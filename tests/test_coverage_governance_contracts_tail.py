from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3mlpy as gp
from gp3mlpy import analysis_plan as ap
from gp3mlpy import api_contracts as ac
from gp3mlpy import engine_capabilities as ec
from gp3mlpy import environment as env
from gp3mlpy import feature_provenance as fp
from gp3mlpy import governance_profiles as gprof
from gp3mlpy import interoperability as interop
from gp3mlpy import release_provenance as rp
from gp3mlpy import synthetic as syn
from gp3mlpy.exceptions import GP3MLError
from gp3mlpy.objects import GP3MLAPIContractRegistry, GP3MLEnvironment, GP3MLObject


def _valid_plan():
    return ap.declare_gazepoint_analysis_plan(
        research_question="Can observed gaze features predict the prespecified response?",
        scientific_purpose="Evaluate a non-sensitive observed outcome.",
        outcome="observed_response",
        outcome_definition="Explicitly recorded response.",
        predictors=["tracking_ratio", "blink_rate"],
        generalization_target="new_participants",
        grouping_variables=["participant_id"],
        eligible_population="Participants satisfying the study inclusion criteria.",
        exclusion_rules=["invalid recording"],
        preprocessing_plan={"scope": "fold"},
        candidate_models=["glm", "ranger"],
        primary_metric="roc_auc",
        secondary_metrics=["brier", "balanced_accuracy"],
        calibration_metric="ece",
        uncertainty_method="participant bootstrap",
        threshold_policy="analysis-only",
        external_validation_required=True,
        seed_strategy="fixed declared seed",
    )


def test_analysis_plan_helpers_validation_lock_audit_and_writers(tmp_path: Path):
    assert ap._unique_str(None) == []
    assert ap._unique_str("x") == ["x"]
    assert ap._unique_str(["x", "x", 2]) == ["x", "2"]
    assert ap._empty(None)
    assert ap._empty("   ")
    assert ap._empty([])
    assert ap._empty(["", " "])
    assert not ap._empty(["x", ""])
    assert not ap._empty(3)

    plan = _valid_plan()
    assert ap.validate_gazepoint_analysis_plan(plan).status == "pass"
    assert ap.validate_gazepoint_analysis_plan(object()).status == "fail"

    duplicate = plan.__deepcopy__({})
    duplicate.predictors = ["tracking_ratio", "tracking_ratio"]
    assert ap.validate_gazepoint_analysis_plan(duplicate).status == "review"
    unsafe = plan.__deepcopy__({})
    unsafe.predictors = [unsafe.outcome]
    assert ap.validate_gazepoint_analysis_plan(unsafe).status == "fail"
    incomplete = plan.__deepcopy__({})
    incomplete.research_question = ""
    with pytest.raises(GP3MLError, match="pass validation"):
        ap.lock_gazepoint_analysis_plan(incomplete)

    locked = ap.lock_gazepoint_analysis_plan(
        plan,
        plan_id="declared-plan",
        locked_at=datetime(2026, 8, 30, 9, 0, tzinfo=timezone.utc),
    )
    assert locked.locked and locked.plan_id == "declared-plan"
    assert len(locked.plan_hash) == 64
    with pytest.raises(GP3MLError, match="already locked"):
        ap.lock_gazepoint_analysis_plan(locked)

    generated = ap.lock_gazepoint_analysis_plan(_valid_plan(), locked_at="2026-08-30T09:00:00+00:00")
    assert generated.plan_id.startswith("gp3ml-plan-")
    assert generated.locked_at.endswith("UTC")

    assert ap._structure_text(None) == "NULL"
    assert "chr" in ap._structure_text("x")
    assert "[1:2]" in ap._structure_text(["a", "b"])
    assert ap._structure_text({"a": 1}) == "{'a': 1}"
    assert ap._structure_text(3) == "3"

    actual = {
        field: locked.to_dict().get(field)
        for field in (
            "outcome",
            "predictors",
            "generalization_target",
            "primary_metric",
            "secondary_metrics",
            "calibration_metric",
            "uncertainty_method",
            "threshold_policy",
            "candidate_models",
            "preprocessing_plan",
        )
    }
    audit = ap.audit_gazepoint_plan_deviations(locked, actual)
    assert audit.status == "pass"
    changed = dict(actual)
    changed["primary_metric"] = "brier"
    assert ap.audit_gazepoint_plan_deviations(locked, changed).status == "review"
    with pytest.raises(GP3MLError, match="locked"):
        ap.audit_gazepoint_plan_deviations(plan, actual)
    with pytest.raises(GP3MLError, match="named list"):
        ap.audit_gazepoint_plan_deviations(locked, [])

    for fmt in ("json", "rds"):
        out = Path(ap.write_gazepoint_analysis_plan(locked, tmp_path / f"plan.{fmt}", fmt))
        assert out.exists() and "declared-plan" in out.read_text(encoding="utf-8")
    md = Path(ap.write_gazepoint_analysis_plan(locked, tmp_path / "plan.md", "md"))
    assert "Prohibited interpretations" in md.read_text(encoding="utf-8")
    with pytest.raises(GP3MLError, match="format"):
        ap.write_gazepoint_analysis_plan(locked, tmp_path / "bad", "yaml")
    with pytest.raises(GP3MLError, match="invalid"):
        ap.write_gazepoint_analysis_plan(incomplete, tmp_path / "invalid.json", "json")


def test_api_contract_registry_schema_audit_and_writer(tmp_path: Path, monkeypatch):
    registry = ac.gp3ml_api_contracts()
    assert isinstance(registry, GP3MLAPIContractRegistry)
    assert set(registry.exports.stability) == {"stable", "experimental"}
    assert len(ac._current_exports()) > 0

    real_version = ac.metadata.version
    monkeypatch.setattr(
        ac.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(ac.metadata.PackageNotFoundError(name)),
    )
    fallback = ac.gp3ml_api_contracts()
    assert fallback.package_version == "development"
    monkeypatch.setattr(ac.metadata, "version", real_version)

    generic = GP3MLObject(a=1)
    values = [
        (generic, generic.r_class, "list"),
        (pd.DataFrame({"x": [1]}), "data.frame", "list"),
        (pd.Series([1.0]), "numeric", None),
        (pd.Series(["a"]), "character", None),
        ("x", "character", "character"),
        (True, "logical", "logical"),
        (np.int64(2), "integer", "integer"),
        (np.float64(2), "numeric", "double"),
        ({"a": 1}, "list", "list"),
        ([1], "list", "list"),
    ]
    for value, klass, rtype in values:
        assert ac._rish_class(value) == klass
        if rtype is not None:
            assert ac._rish_type(value) == rtype
    assert ac._rish_type(lambda: None) == "closure"
    assert ac._rish_length("x") == 1
    assert ac._rish_length(None) == 1
    assert ac._rish_length([1, 2]) == 2
    assert ac._rish_length(iter([1])) == 1

    scalar_schema = ac.gp3ml_object_schema(1)
    assert scalar_schema.component.tolist() == ["."]
    frame_schema = ac.gp3ml_object_schema(pd.DataFrame({"x": [1, 2]}))
    assert frame_schema.nrow.iloc[0] == 2 and frame_schema.ncol.iloc[0] == 1
    list_schema = ac.gp3ml_object_schema([1, "a"])
    assert list_schema.component.tolist() == ["[[1]]", "[[2]]"]
    nested = ac.gp3ml_object_schema({"outer": {"inner": 1}}, recursive=True)
    assert "outer$inner" in nested.component.tolist()

    unregistered = ac.validate_gp3ml_object_contract({"x": 1}, registry)
    assert unregistered.status == "review"
    stable_obj = GP3MLObject()
    stable_obj.r_class = registry.classes["class"].iloc[0]
    assert ac.validate_gp3ml_object_contract(stable_obj, registry).status == "pass"

    audit = ac.audit_gp3ml_api_stability(registry)
    assert audit.status in {"pass", "review"}
    manipulated = registry.__deepcopy__({})
    extra_export = pd.DataFrame(
        [{"name": "definitely_missing_export", "stability": "stable", "present": True}]
    )
    manipulated.exports = pd.concat([manipulated.exports, extra_export], ignore_index=True)
    manipulated.classes = pd.concat(
        [
            manipulated.classes,
            pd.DataFrame(
                [{
                    "class": "definitely_missing_class",
                    "stability": "stable",
                    "schema_policy": "additive_only_within_minor_line",
                }]
            ),
        ],
        ignore_index=True,
    )
    failed = ac.audit_gp3ml_api_stability(manipulated)
    assert failed.status == "fail"
    assert {"missing_stable_export", "missing_stable_class"}.issubset(set(failed.differences.type))

    monkeypatch.setattr(gp, "temporary_coverage_function", lambda: None, raising=False)
    unexpected = ac.audit_gp3ml_api_stability(registry)
    assert "unexpected_export" in set(unexpected.differences.type)

    paths = ac.write_gp3ml_api_contracts(registry, tmp_path, prefix="contracts")
    assert set(paths) == {"exports", "classes", "policy"}
    with pytest.raises(GP3MLError, match="registry"):
        ac.write_gp3ml_api_contracts(object(), tmp_path / "bad")


def test_engine_capabilities_all_backend_and_assertion_paths(monkeypatch):
    table = ec.gp3ml_engine_capabilities(False)
    assert table.r_class == "gp3ml_engine_capabilities"
    assert table._constructor is ec.GP3MLEngineCapabilitiesFrame
    assert "engine" in repr(table) and "engine" in str(table)
    assert ec._available("sys")

    monkeypatch.setattr(ec, "_available", lambda name: name in {"keras", "xgboost"})
    unverified = ec.gp3ml_engine_capabilities(False)
    assert unverified.loc[unverified.engine == "keras3", "status"].iloc[0] == "backend_unverified"

    fake_keras = SimpleNamespace(backend=SimpleNamespace(backend=lambda: "numpy"))
    monkeypatch.setitem(sys.modules, "keras", fake_keras)
    ready = ec.gp3ml_engine_capabilities(True)
    krow = ready.loc[ready.engine == "keras3"].iloc[0]
    assert krow.backend == "numpy" and bool(krow.backend_ready)

    broken_keras = SimpleNamespace(backend=SimpleNamespace(backend=lambda: (_ for _ in ()).throw(RuntimeError("no backend"))))
    monkeypatch.setitem(sys.modules, "keras", broken_keras)
    unavailable = ec.gp3ml_engine_capabilities(True)
    assert unavailable.loc[unavailable.engine == "keras3", "status"].iloc[0] == "backend_unavailable"

    with pytest.raises(GP3MLError, match="Unknown"):
        ec.assert_gp3ml_engine_available("unknown")

    missing = ec.GP3MLEngineCapabilitiesFrame(
        {
            "engine": ["xgboost"],
            "package": ["xgboost"],
            "package_available": [False],
            "backend_ready": [None],
        }
    )
    monkeypatch.setattr(ec, "gp3ml_engine_capabilities", lambda check_keras_backend=False: missing)
    with pytest.raises(GP3MLError, match="not installed"):
        ec.assert_gp3ml_engine_available("xgboost")

    bad_backend = ec.GP3MLEngineCapabilitiesFrame(
        {
            "engine": ["keras3"],
            "package": ["keras"],
            "package_available": [True],
            "backend_ready": [False],
        }
    )
    monkeypatch.setattr(ec, "gp3ml_engine_capabilities", lambda check_keras_backend=False: bad_backend)
    with pytest.raises(GP3MLError, match="usable backend"):
        ec.assert_gp3ml_engine_available("keras3", True)
    bad_backend.loc[0, "backend_ready"] = True
    assert ec.assert_gp3ml_engine_available("keras3", True)


def test_environment_capture_compare_and_validation(tmp_path: Path, monkeypatch):
    assert env._pkg_version("gp3ml") == "0.3.0-reference"
    assert env._pkg_version("gp3mlpy")
    assert env._pkg_version("definitely-not-installed-gp3mlpy-test") is None

    class Result:
        stdout = "abc123\n"

    monkeypatch.setattr(env.subprocess, "run", lambda *args, **kwargs: Result())
    assert env._git_sha(tmp_path) == "abc123"
    monkeypatch.setattr(env.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("git")))
    assert env._git_sha(tmp_path) is None

    lock = tmp_path / "renv.lock"
    lock.write_text("{}", encoding="utf-8")
    captured = env.capture_gazepoint_environment(
        packages=["gp3ml", "gp3ml", "gp3mlpy", "definitely-not-installed-gp3mlpy-test"],
        root=tmp_path,
        include_renv=True,
    )
    assert captured.package_versions["gp3ml"] == "0.3.0-reference"
    assert captured.renv_lock_sha256 is not None
    assert "definitely-not-installed-gp3mlpy-test" not in captured.package_versions
    one = env.capture_gazepoint_environment("gp3ml", root=tmp_path)
    assert list(one.package_versions) == ["gp3ml"]

    reference = GP3MLEnvironment(
        R_version="Python A",
        R_platform="platform",
        OS="os",
        BLAS="",
        LAPACK="",
        RNGkind=[],
        repositories={},
        package_versions={"same": "1", "missing_current": "1"},
        gp3ml_git_sha=None,
        renv_lock_sha256=None,
    )
    current = GP3MLEnvironment(
        R_version="Python B",
        R_platform="platform",
        OS="os",
        BLAS="",
        LAPACK="",
        RNGkind=[],
        repositories={},
        package_versions={"same": "1", "only_current": "2"},
        gp3ml_git_sha=None,
        renv_lock_sha256="different",
    )
    comparison = env.compare_gazepoint_environments(reference, current)
    assert comparison.status == "review"
    assert set(comparison.packages.status) == {"pass", "review"}
    with pytest.raises(GP3MLError, match="Both objects"):
        env.compare_gazepoint_environments(object(), current)

    monkeypatch.setattr(env, "capture_gazepoint_environment", lambda **kwargs: reference)
    assert env.validate_gazepoint_environment(reference).status == "pass"


def _manifest_kwargs():
    return dict(
        features=["x"],
        scientific_source=["measurement"],
        source_table=["trial"],
        transformation=["none"],
        availability_stage=["during_exposure"],
        prediction_time_available=[True],
        outcome_derived=[False],
        post_outcome=[False],
        identifier=[False],
        preprocessing_scope=["resampling_fold"],
        fold_local_required=[True],
        reviewer_notes=["safe"],
    )


def test_feature_manifest_structure_validation_issue_paths_and_writer(tmp_path: Path):
    manifest = fp.create_gazepoint_feature_manifest(**_manifest_kwargs())
    assert fp.validate_gazepoint_feature_manifest(manifest).status == "pass"
    with pytest.raises(GP3MLError, match="feature-manifest"):
        fp._as_feature_manifest(object())
    with pytest.raises(GP3MLError, match="missing required"):
        fp._as_feature_manifest(manifest.drop(columns=["scientific_source"]))

    dupcols = pd.concat([manifest, manifest[["feature"]]], axis=1)
    with pytest.raises(GP3MLError, match="column names"):
        fp._as_feature_manifest(dupcols)
    bad_char = manifest.copy()
    bad_char["scientific_source"] = [1]
    with pytest.raises(GP3MLError, match="character"):
        fp._as_feature_manifest(bad_char)
    bad_logical = manifest.copy()
    bad_logical["identifier"] = [1]
    with pytest.raises(GP3MLError, match="logical"):
        fp._as_feature_manifest(bad_logical)
    for value in ("", "  "):
        bad_feature = manifest.copy()
        bad_feature["feature"] = [value]
        with pytest.raises(GP3MLError, match="feature"):
            fp._as_feature_manifest(bad_feature)
    duplicate_feature = pd.concat([manifest, manifest], ignore_index=True)
    with pytest.raises(GP3MLError, match="feature"):
        fp._as_feature_manifest(duplicate_feature)
    bad_stage = manifest.copy()
    bad_stage["availability_stage"] = ["bad"]
    with pytest.raises(GP3MLError, match="availability_stage"):
        fp._as_feature_manifest(bad_stage)
    bad_scope = manifest.copy()
    bad_scope["preprocessing_scope"] = ["bad"]
    with pytest.raises(GP3MLError, match="preprocessing_scope"):
        fp._as_feature_manifest(bad_scope)

    with pytest.raises(GP3MLError, match="features"):
        fp.create_gazepoint_feature_manifest([])
    with pytest.raises(GP3MLError, match="features"):
        fp.create_gazepoint_feature_manifest(["x", "x"])

    review_kwargs = _manifest_kwargs()
    review_kwargs.update(
        scientific_source=[None],
        source_table=[None],
        transformation=[None],
        availability_stage=["unknown"],
        prediction_time_available=[None],
        preprocessing_scope=["unknown"],
        fold_local_required=[None],
    )
    review = fp.validate_gazepoint_feature_manifest(fp.create_gazepoint_feature_manifest(**review_kwargs))
    assert review.status == "review"

    fail_kwargs = _manifest_kwargs()
    fail_kwargs.update(
        prediction_time_available=[False],
        outcome_derived=[True],
        identifier=[True],
        post_outcome=[True],
        availability_stage=["post_outcome"],
        preprocessing_scope=["global"],
        fold_local_required=[True],
    )
    failed = fp.validate_gazepoint_feature_manifest(fp.create_gazepoint_feature_manifest(**fail_kwargs))
    assert failed.status == "fail"
    assert {"prediction_time_available", "outcome_derived", "post_outcome", "identifier", "preprocessing_scope_compatible"}.issubset(set(failed.issues.check_id))

    inconsistent_kwargs = _manifest_kwargs()
    inconsistent_kwargs.update(post_outcome=[False], availability_stage=["post_outcome"], prediction_time_available=[True])
    inconsistent = fp.validate_gazepoint_feature_manifest(fp.create_gazepoint_feature_manifest(**inconsistent_kwargs))
    assert inconsistent.status == "fail"
    assert "post_outcome_metadata_consistent" in set(inconsistent.issues.check_id)
    assert "availability_metadata_consistent" in set(inconsistent.issues.check_id)

    out = tmp_path / "manifest.csv"
    assert Path(fp.write_gazepoint_feature_manifest_csv(manifest, out)).exists()
    checks = tmp_path / "checks.csv"
    assert Path(fp.write_gazepoint_feature_manifest_csv(failed, checks, table="checks")).exists()
    issues = tmp_path / "issues.csv"
    assert Path(fp.write_gazepoint_feature_manifest_csv(failed, issues, table="issues")).exists()
    with pytest.raises(GP3MLError, match="file"):
        fp.write_gazepoint_feature_manifest_csv(manifest, "")
    with pytest.raises(GP3MLError, match=".csv"):
        fp.write_gazepoint_feature_manifest_csv(manifest, tmp_path / "x.txt")
    with pytest.raises(GP3MLError, match="table"):
        fp.write_gazepoint_feature_manifest_csv(manifest, tmp_path / "x.csv", table="bad")
    with pytest.raises(GP3MLError, match="Plain manifest"):
        fp.write_gazepoint_feature_manifest_csv(manifest, tmp_path / "x.csv", table="issues")
    with pytest.raises(GP3MLError, match="does not exist"):
        fp.write_gazepoint_feature_manifest_csv(manifest, tmp_path / "missing" / "x.csv")
    with pytest.raises(GP3MLError, match="already exists"):
        fp.write_gazepoint_feature_manifest_csv(manifest, out)
    assert fp.write_gazepoint_feature_manifest_csv(manifest, out, overwrite=True)


def test_governance_profiles_all_frameworks_audit_and_writer(tmp_path: Path):
    frameworks = ["gp3ml-native", "NIST-AI-RMF-1.0", "ISO-23894-oriented", "ISO-42001-oriented"]
    for framework in frameworks:
        profile = gprof.create_gp3ml_governance_profile({"task": GP3MLObject()}, framework)
        assert len(profile.controls) == 10
        audit = gprof.audit_gp3ml_governance_profile(profile)
        assert audit.status == "review"
        path = Path(gprof.write_gp3ml_governance_profile(audit, tmp_path / f"{framework}.md"))
        assert path.exists() and framework in path.read_text(encoding="utf-8")
    complete_keys = set(gprof._domains("gp3ml-native").evidence_key)
    complete = gprof.create_gp3ml_governance_profile({key: GP3MLObject() for key in complete_keys})
    assert gprof.audit_gp3ml_governance_profile(complete).status == "pass"
    with pytest.raises(GP3MLError, match="framework"):
        gprof.create_gp3ml_governance_profile({}, "bad")
    with pytest.raises(GP3MLError, match="named list"):
        gprof.create_gp3ml_governance_profile([], "gp3ml-native")
    with pytest.raises(GP3MLError, match="Invalid governance profile"):
        gprof.audit_gp3ml_governance_profile(object())
    with pytest.raises(GP3MLError, match="Invalid governance audit"):
        gprof.write_gp3ml_governance_profile(object(), tmp_path / "bad.md")


def _handoff_frame():
    return pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0], "outcome": ["no", "yes"]})


def test_interoperability_contract_handoffs_validation_combination_and_extract(monkeypatch):
    contracts = interop.gp3ml_interop_contracts()
    assert len(contracts) == 5 and not contracts.duplicates_upstream_preprocessing.any()
    data = _handoff_frame()
    with pytest.raises(GP3MLError, match="source_package"):
        interop.create_gazepoint_handoff(data, "bad", keys="id")

    real_version = interop.metadata.version
    monkeypatch.setattr(interop.metadata, "version", lambda name: "1.2.3")
    handoff = interop.create_gazepoint_handoff(
        data,
        "gp3tools",
        keys="id",
        outcome="outcome",
        predictors="x",
        notes="note",
    )
    assert handoff.source_version == "1.2.3" and handoff.notes == ["note"]
    monkeypatch.setattr(
        interop.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(interop.metadata.PackageNotFoundError(name)),
    )
    absent_version = interop.create_gazepoint_handoff(data, "gpbiometrics", keys="id", predictors="x", notes=None)
    assert absent_version.source_version is None and absent_version.notes == []
    monkeypatch.setattr(interop.metadata, "version", real_version)

    assert interop._composite_key(data, []).tolist() == ["", ""]
    assert interop._composite_key(data, ["id"]).tolist() == ["a", "b"]
    assert interop.validate_gazepoint_handoff(handoff).status == "pass"
    with pytest.raises(GP3MLError, match="created"):
        interop.validate_gazepoint_handoff(object())

    for mutator in (
        lambda h: setattr(h, "keys", ["missing"]),
        lambda h: h.data.__setitem__("id", [None, "b"]),
        lambda h: h.data.__setitem__("id", ["a", "a"]),
        lambda h: setattr(h, "predictors", ["missing"]),
        lambda h: setattr(h, "outcome", "missing"),
        lambda h: h.data.__setitem__("x", [9.0, 9.0]),
    ):
        broken = handoff.__deepcopy__({})
        mutator(broken)
        assert interop.validate_gazepoint_handoff(broken).status == "fail"

    with pytest.raises(GP3MLError, match="collision"):
        interop.combine_gazepoint_handoffs([handoff], collision="bad")
    with pytest.raises(GP3MLError, match="non-empty"):
        interop.combine_gazepoint_handoffs([])
    broken = handoff.__deepcopy__({})
    broken.data["id"] = ["a", "a"]
    with pytest.raises(GP3MLError, match="pass validation"):
        interop.combine_gazepoint_handoffs([broken])

    second = interop.create_gazepoint_handoff(
        pd.DataFrame({"id": ["a", "b"], "x": [3.0, 4.0], "z": [5, 6]}),
        "custom",
        keys="id",
        predictors=["x", "z"],
    )
    with pytest.raises(GP3MLError, match="collision across"):
        interop.combine_gazepoint_handoffs({"one": handoff, "two": second}, keys="id")
    bundle = interop.combine_gazepoint_handoffs({"one": handoff, "two": second}, collision="prefix")
    assert bundle.keys == ["id"] and len(bundle.data) == 2
    assert any(c.startswith("one__") for c in bundle.data.columns)
    assert interop.as_gp3ml_data(handoff).equals(handoff.data)
    assert interop.as_gp3ml_data(bundle).equals(bundle.data)
    with pytest.raises(GP3MLError, match="handoff or handoff bundle"):
        interop.as_gp3ml_data(object())
    damaged = handoff.__deepcopy__({})
    damaged.data["x"] = [99, 99]
    with pytest.raises(GP3MLError, match="validation must pass"):
        interop.as_gp3ml_data(damaged)


def test_release_checksums_and_synthetic_guard_paths(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")
    manifest = rp.write_gazepoint_release_checksums(a, tmp_path / "one.csv")
    assert manifest.checksums.file.tolist() == ["a.txt"]
    multi = rp.write_gazepoint_release_checksums([a, b], tmp_path / "multi.csv")
    assert rp.validate_gazepoint_release_checksums(multi, tmp_path).status == "pass"
    assert rp.validate_gazepoint_release_checksums(multi.path, tmp_path).status == "pass"
    b.write_text("changed", encoding="utf-8")
    assert rp.validate_gazepoint_release_checksums(multi, tmp_path).status == "fail"
    b.unlink()
    assert rp.validate_gazepoint_release_checksums(multi, tmp_path).status == "fail"
    with pytest.raises(GP3MLError, match="must exist"):
        rp.write_gazepoint_release_checksums(tmp_path / "missing.txt")

    for kwargs, match in (
        ({"n_participants": 3}, "n_participants"),
        ({"n_stimuli": 1}, "n_stimuli"),
        ({"trials_per_cell": 0}, "trials_per_cell"),
    ):
        with pytest.raises(GP3MLError, match=match):
            syn.simulate_gazepoint_governed_data(**kwargs)
    data = syn.simulate_gazepoint_governed_data(4, 2, 1, 2)
    for workflow in ("recording_quality", "assigned_condition", "observed_behavior", "observed_duration"):
        task = syn.create_gazepoint_synthetic_task(data, workflow, "new_participants")
        assert task.outcome in data.columns
    with pytest.raises(GP3MLError, match="workflow"):
        syn.create_gazepoint_synthetic_task(data, "bad")
    with pytest.raises(GP3MLError, match="generalization_target"):
        syn.create_gazepoint_synthetic_task(data, "recording_quality", "bad")
    manifest = syn.create_gazepoint_synthetic_manifest("quality_status", ["tracking_ratio"])
    assert manifest.feature.tolist() == ["tracking_ratio"]
