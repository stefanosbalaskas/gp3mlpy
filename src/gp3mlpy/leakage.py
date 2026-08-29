from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd

from .exceptions import GP3MLError
from .objects import GazepointMLLeakageAudit

_GENERALIZATION_TARGETS = (
    "new_trials_known_participants", "new_participants", "new_stimuli", "new_participants_and_new_stimuli"
)
_IDENTIFIER_RE = re.compile(r"(^|_)(id|uuid|guid|identifier|record|row|index|filename|file_name|filepath|file_path|session_id)(_|$)", re.I)


def _column_name(x: str | None, argument: str, allow_null: bool = False) -> None:
    if allow_null and x is None: return
    if not isinstance(x, str) or not x: raise GP3MLError(f"`{argument}` must be a single non-empty column name.")


def _column_vector(x: Any, argument: str, allow_empty: bool = True) -> list[str]:
    if x is None: vals=[]
    elif isinstance(x, str): vals=[x]
    elif isinstance(x, (list, tuple, pd.Index, np.ndarray)): vals=list(x)
    else: raise GP3MLError(f"`{argument}` must be a character vector.")
    if not allow_empty and not vals: raise GP3MLError(f"`{argument}` must contain at least one column name.")
    if any(not isinstance(v,str) or not v for v in vals) or len(vals)!=len(set(vals)):
        raise GP3MLError(f"`{argument}` must contain unique, non-missing, non-empty column names.")
    return vals


def _missing_identifier(series: pd.Series) -> pd.Series:
    missing=series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype,pd.CategoricalDtype):
        vals=series.astype("string").str.strip()
        missing=missing | vals.eq("").fillna(False)
    return missing


def _identifier_values(series: pd.Series) -> list[str]:
    miss=_missing_identifier(series)
    return pd.unique(series.loc[~miss].astype(str)).tolist()


def _canonical_value(v: Any) -> Any:
    if pd.isna(v): return ("<NA>",)
    if isinstance(v, pd.Timestamp): return v.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%f") if v.tzinfo else v.strftime("%Y-%m-%dT%H:%M:%S.%f")
    if isinstance(v, np.datetime64): return str(pd.Timestamp(v))
    if isinstance(v, pd.Timedelta): return float(v.total_seconds())
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, (list,dict,set,tuple,np.ndarray)): raise GP3MLError("Leakage signatures do not currently support list or matrix columns.")
    return v


def _row_signatures(data: pd.DataFrame, columns: list[str]) -> list[tuple[Any,...]]:
    return [tuple(_canonical_value(v) for v in row) for row in data.loc[:,columns].itertuples(index=False,name=None)]


def _trial_signatures(data: pd.DataFrame, trial_id: str, participant_id: str | None = None) -> list[tuple[Any,...]]:
    cols=[x for x in [participant_id,trial_id] if x]
    complete=pd.Series(True,index=data.index)
    for col in cols: complete &= ~_missing_identifier(data[col])
    return _row_signatures(data.loc[complete],cols) if complete.any() else []


def _partition_summary(analysis: pd.DataFrame, assessment: pd.DataFrame, participant_id: str|None, trial_id: str|None, stimulus_id: str|None) -> pd.DataFrame:
    def count(d:pd.DataFrame,c:str|None)->Any: return pd.NA if c is None else len(_identifier_values(d[c]))
    return pd.DataFrame({
        "partition":["analysis","assessment"], "n_rows":[len(analysis),len(assessment)],
        "n_participants":[count(analysis,participant_id),count(assessment,participant_id)],
        "n_trials":[count(analysis,trial_id),count(assessment,trial_id)],
        "n_stimuli":[count(analysis,stimulus_id),count(assessment,stimulus_id)],
    })


def audit_gazepoint_ml_leakage(
    analysis: pd.DataFrame,
    assessment: pd.DataFrame,
    outcome: str,
    predictors: list[str] | tuple[str,...] | str,
    participant_id: str | None = None,
    trial_id: str | None = None,
    stimulus_id: str | None = None,
    generalization_target: str = "new_trials_known_participants",
    target_derived: list[str] | tuple[str,...] | str | None = None,
    post_outcome: list[str] | tuple[str,...] | str | None = None,
) -> GazepointMLLeakageAudit:
    """Audit structural leakage between analysis and assessment partitions."""
    if generalization_target not in _GENERALIZATION_TARGETS: raise GP3MLError("`generalization_target` is invalid.")
    if not isinstance(analysis,pd.DataFrame) or not isinstance(assessment,pd.DataFrame): raise GP3MLError("`analysis` and `assessment` must both be data frames.")
    if len(analysis)==0 or len(assessment)==0: raise GP3MLError("`analysis` and `assessment` must each contain at least one row.")
    if analysis.columns.duplicated().any() or assessment.columns.duplicated().any(): raise GP3MLError("Partition column names must be unique.")
    if set(analysis.columns)!=set(assessment.columns): raise GP3MLError("`analysis` and `assessment` must contain the same column names.")
    assessment=assessment.loc[:,analysis.columns].copy()
    _column_name(outcome,"outcome")
    predictors_l=_column_vector(predictors,"predictors",allow_empty=False)
    _column_name(participant_id,"participant_id",True); _column_name(trial_id,"trial_id",True); _column_name(stimulus_id,"stimulus_id",True)
    target_l=_column_vector(target_derived,"target_derived"); post_l=_column_vector(post_outcome,"post_outcome")
    identifiers=[x for x in [participant_id,trial_id,stimulus_id] if x]
    if len(identifiers)!=len(set(identifiers)): raise GP3MLError("Participant, trial, and stimulus roles must use distinct columns.")
    if outcome in identifiers: raise GP3MLError("`outcome` must not also be declared as an identifier.")
    declared=list(dict.fromkeys([outcome,*predictors_l,*identifiers,*target_l,*post_l]))
    missing=[c for c in declared if c not in analysis.columns]
    if missing: raise GP3MLError("Declared columns not found in both partitions: "+", ".join(missing)+".")
    rows=[]
    def add(check_id:str,status:str,n:int,columns:list[str]|str|None,message:str,remediation:str)->None:
        if columns is None: cols=[]
        elif isinstance(columns,str): cols=[columns]
        else: cols=list(columns)
        rows.append({"check_id":check_id,"status":status,"n_affected":int(n),"columns":", ".join(cols),"message":message,"remediation":remediation})
    outpred=[outcome] if outcome in predictors_l else []
    add("outcome_in_predictors","fail" if outpred else "pass",len(outpred),outpred,"The outcome is included in the intended predictor set." if outpred else "The outcome is not included in the intended predictor set.","Remove the outcome from `predictors`." if outpred else "None.")
    idpred=[p for p in predictors_l if p in identifiers]
    add("declared_identifier_in_predictors","fail" if idpred else "pass",len(idpred),idpred,"Declared identifiers are included in the predictor set." if idpred else "No declared identifiers are included in the predictor set.","Remove participant, trial, and stimulus identifiers from predictors." if idpred else "None.")
    like=[p for p in predictors_l if _IDENTIFIER_RE.search(p.lower()) and p not in identifiers]
    add("identifier_like_predictor_names","review" if like else "pass",len(like),like,"Some predictor names appear identifier-like and require manual review." if like else "No additional identifier-like predictor names were detected.","Confirm that these variables contain scientific measurements rather than identifiers or row-location information." if like else "None.")
    td=[p for p in predictors_l if p in target_l]
    add("target_derived_predictors","fail" if td else "pass",len(td),td,"Declared target-derived variables are included as predictors." if td else "No declared target-derived variables are included as predictors.","Remove all outcome-derived variables from the predictor set." if td else "None.")
    po=[p for p in predictors_l if p in post_l]
    add("post_outcome_predictors","fail" if po else "pass",len(po),po,"Declared post-outcome variables are included as predictors." if po else "No declared post-outcome variables are included as predictors.","Remove variables unavailable at the intended prediction time." if po else "None.")
    arows=_row_signatures(analysis,list(analysis.columns)); erows=_row_signatures(assessment,list(assessment.columns)); overlap=set(arows)&set(erows)
    add("exact_row_overlap","fail" if overlap else "pass",len(overlap),list(analysis.columns),"Exact row patterns occur in both partitions." if overlap else "No exact row patterns occur in both partitions.","Reconstruct the partitions so that each sample occurs once." if overlap else "None.")
    da=len(arows)-len(set(arows)); de=len(erows)-len(set(erows)); dup=da+de
    add("duplicate_rows_within_partitions","review" if dup else "pass",dup,list(analysis.columns),f"{dup} duplicate rows occur within the supplied partitions ({da} analysis; {de} assessment)." if dup else "No duplicate rows occur within either partition.","Confirm whether repeated rows are expected or accidental duplicates." if dup else "None.")
    aprof=set(_row_signatures(analysis,predictors_l)); eprof=set(_row_signatures(assessment,predictors_l)); pov=aprof&eprof
    add("predictor_profile_overlap","review" if pov else "pass",len(pov),predictors_l,"Identical predictor profiles occur in both partitions." if pov else "No identical predictor profiles occur in both partitions.","Inspect whether repeated profiles represent legitimate repeated measurements, copied samples, or pre-split aggregation." if pov else "None.")
    participant_required=generalization_target in {"new_trials_known_participants","new_participants","new_participants_and_new_stimuli"}
    add("participant_id_available","pass" if participant_id else "fail" if participant_required else "review",int(participant_id is None),participant_id,"A participant identifier was supplied." if participant_id else "No participant identifier was supplied.","None." if participant_id else "Supply `participant_id` to make participant overlap auditable.")
    if participant_id:
        pmiss=int(_missing_identifier(analysis[participant_id]).sum()+_missing_identifier(assessment[participant_id]).sum())
        add("participant_id_missing","pass" if pmiss==0 else "fail" if participant_required else "review",pmiss,participant_id,"No participant identifiers are missing." if pmiss==0 else "Missing participant identifiers occur in the partitions.","None." if pmiss==0 else "Resolve missing participant identifiers before evaluation.")
        ap=_identifier_values(analysis[participant_id]); ep=_identifier_values(assessment[participant_id])
        if generalization_target=="new_trials_known_participants":
            incompatible=set(ep)-set(ap); msg="Assessment contains participants not represented in the analysis partition." if incompatible else "All assessment participants are represented in the analysis partition."
        elif generalization_target in {"new_participants","new_participants_and_new_stimuli"}:
            incompatible=set(ap)&set(ep); msg="Participant identifiers overlap across partitions." if incompatible else "Participant identifiers are disjoint across partitions."
        else: incompatible=set(); msg="Participant overlap is not prohibited by the declared new-stimulus target."
        add("participant_partition_compatibility","fail" if incompatible else "pass",len(incompatible),participant_id,msg,"Reconstruct partitions to match the declared participant generalization target." if incompatible else "None.")
    trial_required=generalization_target=="new_trials_known_participants"
    add("trial_id_available","pass" if trial_id else "fail" if trial_required else "review",int(trial_id is None),trial_id,"A trial identifier was supplied." if trial_id else "No trial identifier was supplied.","None." if trial_id else "Supply `trial_id` to make trial overlap auditable.")
    if trial_id:
        tmiss=int(_missing_identifier(analysis[trial_id]).sum()+_missing_identifier(assessment[trial_id]).sum())
        add("trial_id_missing","pass" if tmiss==0 else "fail" if trial_required else "review",tmiss,trial_id,"No trial identifiers are missing." if tmiss==0 else "Missing trial identifiers occur in the partitions.","None." if tmiss==0 else "Resolve missing trial identifiers before evaluation.")
        ov=set(_trial_signatures(analysis,trial_id,participant_id)) & set(_trial_signatures(assessment,trial_id,participant_id))
        if ov: msg="Trial identifiers overlap across partitions." if participant_id is None else "Participant-trial units overlap across partitions."
        else: msg="Trial identifiers are disjoint across partitions." if participant_id is None else "Participant-trial units are disjoint across partitions."
        add("trial_partition_overlap","fail" if ov else "pass",len(ov),trial_id,msg,"Keep each participant-trial unit entirely within one partition." if ov else "None.")
    stimulus_required=generalization_target in {"new_stimuli","new_participants_and_new_stimuli"}
    add("stimulus_id_available","pass" if stimulus_id else "fail" if stimulus_required else "review",int(stimulus_id is None),stimulus_id,"A stimulus identifier was supplied." if stimulus_id else "No stimulus identifier was supplied.","None." if stimulus_id else "Supply `stimulus_id` to make stimulus overlap auditable.")
    if stimulus_id:
        smiss=int(_missing_identifier(analysis[stimulus_id]).sum()+_missing_identifier(assessment[stimulus_id]).sum())
        add("stimulus_id_missing","pass" if smiss==0 else "fail" if stimulus_required else "review",smiss,stimulus_id,"No stimulus identifiers are missing." if smiss==0 else "Missing stimulus identifiers occur in the partitions.","None." if smiss==0 else "Resolve missing stimulus identifiers before evaluation.")
        if stimulus_required:
            sov=set(_identifier_values(analysis[stimulus_id]))&set(_identifier_values(assessment[stimulus_id])); msg="Stimulus identifiers overlap across partitions." if sov else "Stimulus identifiers are disjoint across partitions."
        else: sov=set(); msg="Stimulus overlap is not prohibited by the declared generalization target."
        add("stimulus_partition_compatibility","fail" if sov else "pass",len(sov),stimulus_id,msg,"Reconstruct partitions with disjoint stimuli for the declared target." if sov else "None.")
    checks=pd.DataFrame(rows); issues=checks.loc[checks.status!="pass"].reset_index(drop=True)
    status="fail" if (checks.status=="fail").any() else "review" if (checks.status=="review").any() else "pass"
    return GazepointMLLeakageAudit(status=status,generalization_target=generalization_target,outcome=outcome,predictors=predictors_l,roles={"participant_id":participant_id,"trial_id":trial_id,"stimulus_id":stimulus_id,"target_derived":target_l,"post_outcome":post_l},partition_summary=_partition_summary(analysis,assessment,participant_id,trial_id,stimulus_id),checks=checks,issues=issues)


def write_gazepoint_ml_leakage_audit_csv(x:GazepointMLLeakageAudit,file:str|Path,table:str="issues",overwrite:bool=False,na:str="")->str:
    if not isinstance(x,GazepointMLLeakageAudit): raise GP3MLError("`x` must inherit from `gazepoint_ml_leakage_audit`.")
    if table not in {"issues","checks","partition_summary"}: raise GP3MLError("`table` is invalid.")
    path=Path(file).expanduser()
    if path.suffix.lower()!=".csv": raise GP3MLError("`file` must use a .csv extension.")
    if not path.parent.exists(): raise GP3MLError(f"Output directory does not exist: {path.parent}.")
    if path.exists() and not overwrite: raise GP3MLError(f"Output file already exists: {path}.")
    x[table].to_csv(path,index=False,na_rep=na,encoding="utf-8")
    return path.resolve().as_posix()


def _repr(self:GazepointMLLeakageAudit)->str:
    a=int(self.partition_summary.loc[self.partition_summary.partition=="analysis","n_rows"].iloc[0]); b=int(self.partition_summary.loc[self.partition_summary.partition=="assessment","n_rows"].iloc[0])
    return f"<gazepoint_ml_leakage_audit>\nOverall status: {self.status.upper()}\nGeneralization target: {self.generalization_target}\nRows: {a} analysis; {b} assessment\nNon-passing checks: {len(self.issues)}"
GazepointMLLeakageAudit.__repr__=_repr  # type: ignore[method-assign]
