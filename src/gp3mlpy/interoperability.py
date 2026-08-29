from __future__ import annotations

from importlib import metadata
from typing import Any

import pandas as pd

from ._utils import assert_columns, assert_data, hash_jsonable, as_list
from .exceptions import GP3MLError
from .objects import GP3MLHandoff, GP3MLHandoffBundle, GP3MLHandoffValidation

_INTEROP_SOURCES = ["gp3tools", "gpbiometrics", "gp3sequences", "study_design", "custom"]


def gp3ml_interop_contracts() -> pd.DataFrame:
    upstream = [
        "Gazepoint import, validation, gaze/fixation/AOI/transition preparation.",
        "EDA/HR/DIAL/IBI preparation and signal-quality summaries.",
        "Ordered-sequence validation, encoding, summaries, motifs, transitions.",
        "Experimentally assigned labels and prespecified study-design variables.",
        "Externally prepared observed, non-sensitive variables.",
    ]
    responsibility = "Role declaration, provenance, leakage-safe splitting/resampling, modelling, evaluation, uncertainty, and reporting."
    return pd.DataFrame({"source_package": _INTEROP_SOURCES, "upstream_responsibility": upstream, "gp3ml_responsibility": responsibility, "duplicates_upstream_preprocessing": False})


def create_gazepoint_handoff(data: pd.DataFrame, source_package: str, source_version: str | None = None, producer: str | None = None, keys=(), outcome: str | None = None, predictors=(), feature_manifest=None, notes=()) -> GP3MLHandoff:
    assert_data(data)
    source_package = str(source_package)
    if source_package not in _INTEROP_SOURCES:
        raise GP3MLError("`source_package` must be one of: " + ", ".join(_INTEROP_SOURCES) + ".")
    keys = list(dict.fromkeys(str(x) for x in as_list(keys) if x is not None))
    predictors = list(dict.fromkeys(str(x) for x in as_list(predictors) if x is not None))
    assert_columns(data, keys, "keys")
    assert_columns(data, predictors, "predictors")
    if outcome is not None: assert_columns(data, [outcome], "outcome")
    if source_version is None and source_package in {"gp3tools", "gpbiometrics", "gp3sequences"}:
        try: source_version = metadata.version(source_package)
        except metadata.PackageNotFoundError: source_version = None
    return GP3MLHandoff(source_package=source_package, source_version=source_version, producer=producer, keys=keys, outcome=outcome, predictors=predictors, feature_manifest=feature_manifest, data_hash=hash_jsonable(data, algorithm="md5"), data=data.copy(), notes=[] if notes is None else ([notes] if isinstance(notes, str) else list(notes)))


def _composite_key(data: pd.DataFrame, keys: list[str]) -> pd.Series:
    if not keys: return pd.Series([""] * len(data), index=data.index)
    return data[keys].astype(object).where(pd.notna(data[keys]), "<NA>").astype(str).agg("\r".join, axis=1)


def validate_gazepoint_handoff(x) -> GP3MLHandoffValidation:
    if not isinstance(x, GP3MLHandoff): raise GP3MLError("`x` must be created by create_gazepoint_handoff().")
    data_ok = isinstance(x.data, pd.DataFrame) and len(x.data) >= 2
    keys_exist = data_ok and bool(x.keys) and all(k in x.data.columns for k in x.keys)
    key_missing = True if not keys_exist else bool(x.data[x.keys].isna().any().any())
    duplicated_keys = True if not keys_exist else bool(_composite_key(x.data, x.keys).duplicated().any())
    predictors_exist = data_ok and all(p in x.data.columns for p in x.predictors)
    outcome_exists = x.outcome is None or (data_ok and x.outcome in x.data.columns)
    hash_matches = data_ok and x.data_hash == hash_jsonable(x.data, algorithm="md5")
    source_known = x.source_package in _INTEROP_SOURCES
    status = ["pass" if source_known else "fail", "pass" if data_ok else "fail", "pass" if keys_exist else "fail", "pass" if not key_missing else "fail", "pass" if not duplicated_keys else "fail", "pass" if predictors_exist else "fail", "pass" if outcome_exists else "fail", "pass" if hash_matches else "fail"]
    detail = [x.source_package, f"{len(x.data)} rows x {len(x.data.columns)} columns" if data_ok else "Invalid data frame.", ", ".join(x.keys), "No missing join-key values." if not key_missing else "Missing join-key values detected.", "Composite join key is unique." if not duplicated_keys else "Duplicated composite join keys detected.", ", ".join(x.predictors), x.outcome or "<none>", "Handoff data are unchanged." if hash_matches else "Handoff data differ from the recorded fingerprint."]
    checks = pd.DataFrame({"check": ["supported_source", "tabular_data", "join_keys_present", "join_keys_complete", "join_keys_unique", "predictors_present", "outcome_present", "data_hash_matches"], "status": status, "detail": detail})
    return GP3MLHandoffValidation(status="fail" if "fail" in status else "pass", checks=checks, source_package=x.source_package, data_hash=x.data_hash)


def combine_gazepoint_handoffs(handoffs, keys=None, collision: str = "error") -> GP3MLHandoffBundle:
    if collision not in {"error", "prefix"}: raise GP3MLError("`collision` must be one of: error, prefix.")
    if isinstance(handoffs, dict): items = list(handoffs.items())
    elif isinstance(handoffs, (list, tuple)) and handoffs: items = [(f"source{i}", x) for i, x in enumerate(handoffs, 1)]
    else: raise GP3MLError("`handoffs` must be a non-empty list.")
    validations = {name: validate_gazepoint_handoff(x) for name, x in items}
    if any(v.status != "pass" for v in validations.values()): raise GP3MLError("Every handoff must pass validation before combination.")
    keys = list(dict.fromkeys(items[0][1].keys if keys is None else [str(k) for k in as_list(keys)]))
    frames = []
    all_non_keys: list[str] = []
    for name, handoff in items:
        assert_columns(handoff.data, keys, "keys")
        frame = handoff.data.copy(); non_keys = [c for c in frame.columns if c not in keys]
        all_non_keys.extend(non_keys)
        if collision == "prefix": frame = frame.rename(columns={c: f"{name}__{c}" for c in non_keys})
        frames.append(frame)
    if collision == "error":
        dupes = sorted({c for c in all_non_keys if all_non_keys.count(c) > 1})
        if dupes: raise GP3MLError("Non-key column collision across handoffs: " + ", ".join(dupes) + '. Use `collision = "prefix"` or resolve upstream.')
    combined = frames[0]
    for frame in frames[1:]: combined = combined.merge(frame, on=keys, how="inner", sort=False)
    handoff_map = dict(items)
    return GP3MLHandoffBundle(data=combined, keys=keys, sources=[x.source_package for _, x in items], handoffs=handoff_map, validations=validations, data_hash=hash_jsonable(combined, algorithm="md5"))


def as_gp3ml_data(x, *args, **kwargs) -> pd.DataFrame:
    if isinstance(x, GP3MLHandoff):
        if validate_gazepoint_handoff(x).status != "pass": raise GP3MLError("Handoff validation must pass before extracting data.")
        return x.data
    if isinstance(x, GP3MLHandoffBundle): return x.data
    raise GP3MLError("`x` must be a gp3ml handoff or handoff bundle.")
