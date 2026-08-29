from __future__ import annotations

from pathlib import Path
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import GP3MLError
from .feature_provenance import validate_gazepoint_feature_manifest
from .leakage import audit_gazepoint_ml_leakage
from .objects import GazepointMLSplit, GazepointMLSplitValidation

TARGETS=("new_trials_known_participants","new_participants","new_stimuli","new_participants_and_new_stimuli")


def _scalar_column(x: str|None, argument:str, allow_null:bool=True)->str|None:
    if x is None and allow_null: return None
    if not isinstance(x,str) or not x.strip(): raise GP3MLError(f"`{argument}` must be a single non-empty column name.")
    return x.strip()


def _group_values(data:pd.DataFrame,column:str,argument:str)->np.ndarray:
    s=data[column]
    vals=s.astype("string").str.strip()
    if s.isna().any() or vals.eq("").fillna(False).any(): raise GP3MLError(f"`{argument}` contains missing or empty grouping identifiers.")
    return vals.astype(str).to_numpy()


def _trial_units(participant:np.ndarray,trial:np.ndarray)->np.ndarray:
    return np.array([f"{len(p)}:{p}|{len(t)}:{t}" for p,t in zip(participant,trial,strict=True)],dtype=object)


def _holdout_count(n_groups:int, proportion:float)->int:
    if n_groups<2: raise GP3MLError("At least two distinct groups are required for splitting.")
    # Python round is banker's rounding, matching R's default tie-to-even semantics.
    count=int(round(n_groups*proportion))
    return max(1,min(n_groups-1,count))


def _manifest_for_predictors(feature_manifest:pd.DataFrame|None,predictors:list[str])->tuple[pd.DataFrame,Any]:
    if feature_manifest is None: raise GP3MLError("`feature_manifest` is required. Create and validate it before group-aware splitting.")
    if not isinstance(feature_manifest,pd.DataFrame) or "feature" not in feature_manifest.columns: raise GP3MLError("`feature_manifest` must be a compatible feature manifest.")
    missing=[p for p in predictors if p not in feature_manifest.feature.tolist()]
    if missing: raise GP3MLError("Predictors missing from `feature_manifest`: "+", ".join(missing)+".")
    manifest=feature_manifest.set_index("feature",drop=False).loc[predictors].reset_index(drop=True)
    manifest.attrs.update(feature_manifest.attrs)
    validation=validate_gazepoint_feature_manifest(manifest)
    if validation.status!="pass": raise GP3MLError(f"The predictor feature manifest must pass validation before splitting; current status is `{validation.status}`.")
    return manifest,validation


def _two_way_assignment(participant:np.ndarray,stimulus:np.ndarray,assessment_prop:float,rng:np.random.Generator,max_attempts:int=250)->dict[str,Any]:
    participants=np.array(sorted(set(participant))); stimuli=np.array(sorted(set(stimulus)))
    pc=_holdout_count(len(participants),math.sqrt(assessment_prop)); sc=_holdout_count(len(stimuli),math.sqrt(assessment_prop))
    best=None; best_score=float("inf")
    for _ in range(max_attempts):
        hp=np.sort(rng.choice(participants,size=pc,replace=False)); hs=np.sort(rng.choice(stimuli,size=sc,replace=False))
        ph=np.isin(participant,hp); sh=np.isin(stimulus,hs)
        partition=np.where(ph & sh,"assessment",np.where(~ph & ~sh,"analysis","excluded"))
        if not np.any(partition=="analysis") or not np.any(partition=="assessment"): continue
        achieved=np.mean(partition=="assessment"); excluded=np.mean(partition=="excluded")
        score=abs(achieved-assessment_prop)+excluded
        if score<best_score:
            best_score=score; best={"partition":partition,"participant_held":ph,"stimulus_held":sh,"held_participants":hp.tolist(),"held_stimuli":hs.tolist()}
    if best is None: raise GP3MLError("Could not construct non-empty analysis and assessment blocks for simultaneous participant and stimulus generalization.")
    return best


def _group_counts(data:pd.DataFrame,partition:np.ndarray,participant_id:str|None,trial_id:str|None,stimulus_id:str|None)->pd.DataFrame:
    rows=[]
    def add(unit:str,values:np.ndarray)->None:
        for name in ["analysis","assessment","excluded"]:
            selected=partition==name; rows.append({"partition":name,"unit":unit,"n_groups":int(len(set(values[selected]))) if selected.any() else 0})
    participant=None
    if participant_id: participant=data[participant_id].astype(str).to_numpy(); add("participant",participant)
    if trial_id:
        trial=data[trial_id].astype(str).to_numpy()
        if participant_id: add("participant_trial",_trial_units(participant,trial))
        else: add("trial",trial)
    if stimulus_id: add("stimulus",data[stimulus_id].astype(str).to_numpy())
    return pd.DataFrame(rows,columns=["partition","unit","n_groups"])


def split_gazepoint_ml_data(
    data:pd.DataFrame,
    outcome:str,
    predictors:list[str]|tuple[str,...]|str,
    feature_manifest:pd.DataFrame,
    generalization_target:str,
    participant_id:str|None=None,
    trial_id:str|None=None,
    stimulus_id:str|None=None,
    assessment_prop:float=0.20,
    seed:int=1,
    source_row_id:str=".gp3ml_source_row",
)->GazepointMLSplit:
    """Create a deterministic group-aware Gazepoint holdout split."""
    if not isinstance(data,pd.DataFrame): raise GP3MLError("`data` must be a data frame.")
    if len(data)<2: raise GP3MLError("`data` must contain at least two rows.")
    outcome=_scalar_column(outcome,"outcome",False); source_row_id=_scalar_column(source_row_id,"source_row_id",False)
    participant_id=_scalar_column(participant_id,"participant_id"); trial_id=_scalar_column(trial_id,"trial_id"); stimulus_id=_scalar_column(stimulus_id,"stimulus_id")
    if isinstance(predictors,str): predictors_l=[predictors.strip()]
    elif isinstance(predictors,(list,tuple,np.ndarray,pd.Index)): predictors_l=[str(x).strip() for x in predictors]
    else: raise GP3MLError("`predictors` must contain non-empty column names.")
    if not predictors_l or any(not p for p in predictors_l): raise GP3MLError("`predictors` must contain non-empty column names.")
    if len(predictors_l)!=len(set(predictors_l)): raise GP3MLError("`predictors` must contain unique column names.")
    if outcome in predictors_l: raise GP3MLError("`outcome` must not be included in `predictors`.")
    ids=[x for x in [participant_id,trial_id,stimulus_id,source_row_id] if x]
    idpred=[p for p in predictors_l if p in ids]
    if idpred: raise GP3MLError("Identifier columns must not be predictors: "+", ".join(idpred)+".")
    if generalization_target not in TARGETS: raise GP3MLError("`generalization_target` must be one of: "+", ".join(TARGETS)+".")
    if not isinstance(assessment_prop,(int,float,np.integer,np.floating)) or not np.isfinite(assessment_prop) or not 0<assessment_prop<1: raise GP3MLError("`assessment_prop` must be strictly between 0 and 1.")
    if not isinstance(seed,(int,np.integer)) or isinstance(seed,bool): raise GP3MLError("`seed` must be a single finite integer.")
    ok={"new_trials_known_participants":participant_id is not None and trial_id is not None,"new_participants":participant_id is not None,"new_stimuli":stimulus_id is not None,"new_participants_and_new_stimuli":participant_id is not None and stimulus_id is not None}[generalization_target]
    if not ok: raise GP3MLError(f'Required grouping identifiers were not supplied for `generalization_target = "{generalization_target}"`.')
    if source_row_id in data.columns: raise GP3MLError(f"`data` already contains the reserved source-row column `{source_row_id}`.")
    required=[outcome,*predictors_l,*[x for x in [participant_id,trial_id,stimulus_id] if x]]
    missing=[c for c in dict.fromkeys(required) if c not in data.columns]
    if missing: raise GP3MLError("`data` is missing required columns: "+", ".join(missing)+".")
    manifest,manifest_validation=_manifest_for_predictors(feature_manifest,predictors_l)
    participant=_group_values(data,participant_id,"participant_id") if participant_id else None
    trial=_group_values(data,trial_id,"trial_id") if trial_id else None
    stimulus=_group_values(data,stimulus_id,"stimulus_id") if stimulus_id else None
    rng=np.random.default_rng(int(seed))
    n=len(data); participant_held=np.zeros(n,dtype=bool); stimulus_held=np.zeros(n,dtype=bool); split_unit=np.empty(n,dtype=object); split_unit[:]=None
    if generalization_target=="new_trials_known_participants":
        assert participant is not None and trial is not None
        trial_units=_trial_units(participant,trial); split_unit=trial_units.copy(); assessment_units=[]
        for pv in sorted(set(participant)):
            units=np.array(sorted(set(trial_units[participant==pv])))
            if len(units)<2: raise GP3MLError(f"Participant `{pv}` has fewer than two distinct participant-trial units.")
            assessment_units.extend(rng.choice(units,size=_holdout_count(len(units),assessment_prop),replace=False).tolist())
        partition=np.where(np.isin(trial_units,assessment_units),"assessment","analysis")
    elif generalization_target=="new_participants":
        assert participant is not None
        groups=np.array(sorted(set(participant))); held=rng.choice(groups,size=_holdout_count(len(groups),assessment_prop),replace=False); participant_held=np.isin(participant,held); split_unit=participant.copy(); partition=np.where(participant_held,"assessment","analysis")
    elif generalization_target=="new_stimuli":
        assert stimulus is not None
        groups=np.array(sorted(set(stimulus))); held=rng.choice(groups,size=_holdout_count(len(groups),assessment_prop),replace=False); stimulus_held=np.isin(stimulus,held); split_unit=stimulus.copy(); partition=np.where(stimulus_held,"assessment","analysis")
    else:
        assert participant is not None and stimulus is not None
        tw=_two_way_assignment(participant,stimulus,assessment_prop,rng); partition=tw["partition"]; participant_held=tw["participant_held"]; stimulus_held=tw["stimulus_held"]
        split_unit=np.array([f"{len(p)}:{p}|{len(s)}:{s}" for p,s in zip(participant,stimulus,strict=True)],dtype=object)
    if not np.any(partition=="analysis") or not np.any(partition=="assessment"): raise GP3MLError("The requested split produced an empty partition.")
    source_rows=np.arange(1,n+1,dtype=int); split_data=data.copy(); split_data[source_row_id]=source_rows
    aidx=source_rows[partition=="analysis"]; eidx=source_rows[partition=="assessment"]; xidx=source_rows[partition=="excluded"]
    # R's data.frame integer indexing is 1-based source row IDs.
    analysis=split_data.loc[partition=="analysis"].reset_index(drop=True); assessment=split_data.loc[partition=="assessment"].reset_index(drop=True); excluded=split_data.loc[partition=="excluded"].reset_index(drop=True)
    assignment=pd.DataFrame({"source_row":source_rows,"partition":partition,"split_unit":split_unit,"participant_held_out":participant_held,"stimulus_held_out":stimulus_held})
    retained=len(aidx)+len(eidx)
    summary=pd.DataFrame([{"generalization_target":generalization_target,"seed":int(seed),"assessment_prop_requested":float(assessment_prop),"assessment_prop_achieved_all":len(eidx)/n,"assessment_prop_achieved_retained":len(eidx)/retained,"n_total":n,"n_analysis":len(aidx),"n_assessment":len(eidx),"n_excluded":len(xidx)}])
    groups=_group_counts(data,partition,participant_id,trial_id,stimulus_id)
    leakage=audit_gazepoint_ml_leakage(analysis,assessment,outcome,predictors_l,participant_id,trial_id,stimulus_id,generalization_target)
    result=GazepointMLSplit(analysis=analysis,assessment=assessment,excluded=excluded,analysis_indices=aidx.tolist(),assessment_indices=eidx.tolist(),excluded_indices=xidx.tolist(),assignment=assignment,summary=summary,group_counts=groups,feature_manifest=manifest,feature_manifest_validation=manifest_validation,leakage_audit=leakage,metadata={"outcome":outcome,"predictors":predictors_l,"participant_id":participant_id,"trial_id":trial_id,"stimulus_id":stimulus_id,"generalization_target":generalization_target,"assessment_prop":float(assessment_prop),"seed":int(seed),"source_row_id":source_row_id,"n_source_rows":n})
    result.validation=validate_gazepoint_ml_split(result)
    return result


def validate_gazepoint_ml_split(x:GazepointMLSplit)->GazepointMLSplitValidation:
    """Validate source-row, manifest and leakage invariants for a split."""
    if not isinstance(x,GazepointMLSplit): raise GP3MLError("`x` must be a `gazepoint_ml_split` object.")
    required=["analysis","assessment","excluded","assignment","summary","group_counts","feature_manifest_validation","leakage_audit","metadata"]
    missing=[c for c in required if c not in x]
    if missing: raise GP3MLError("Split object is missing components: "+", ".join(missing)+".")
    sr=x.metadata["source_row_id"]
    for name in ["analysis","assessment","excluded"]:
        if not isinstance(x[name],pd.DataFrame) or sr not in x[name].columns: raise GP3MLError(f"Partition `{name}` is not structurally valid.")
    rows=[]
    def add(check_id,status,message,remediation): rows.append({"check_id":check_id,"status":status,"message":message,"remediation":remediation})
    ar=x.analysis[sr].tolist(); er=x.assessment[sr].tolist(); xr=x.excluded[sr].tolist()
    add("analysis_non_empty","pass" if len(x.analysis)>0 else "fail","The analysis partition is non-empty." if len(x.analysis)>0 else "The analysis partition is empty.","None." if len(x.analysis)>0 else "Revise the split request.")
    add("assessment_non_empty","pass" if len(x.assessment)>0 else "fail","The assessment partition is non-empty." if len(x.assessment)>0 else "The assessment partition is empty.","None." if len(x.assessment)>0 else "Revise the split request.")
    dup=len(ar)!=len(set(ar)) or len(er)!=len(set(er)) or len(xr)!=len(set(xr)); add("source_rows_unique_within_partitions","fail" if dup else "pass","Source rows are duplicated within a partition." if dup else "Source rows are unique within each partition.","Restore one assignment per source row." if dup else "None.")
    overlap=len(set(ar)&set(er))+len(set(ar)&set(xr))+len(set(er)&set(xr)); add("source_rows_disjoint","fail" if overlap else "pass","Source rows overlap across returned partitions." if overlap else "Source rows are disjoint across returned partitions.","Assign each source row to only one partition." if overlap else "None.")
    all_rows=sorted(ar+er+xr); expected=list(range(1,int(x.metadata["n_source_rows"])+1)); complete=all_rows==expected
    add("source_rows_fully_accounted","pass" if complete else "fail","All source rows are accounted for exactly once." if complete else "Source-row accounting is incomplete or invalid.","None." if complete else "Reconstruct the split from the original data.")
    compatible=len(x.excluded)==0 or x.metadata["generalization_target"]=="new_participants_and_new_stimuli"
    add("excluded_rows_compatible","pass" if compatible else "fail","No rows were excluded." if len(x.excluded)==0 else "Cross-block rows were excluded to preserve simultaneous participant and stimulus generalization." if compatible else "Rows were unexpectedly excluded for this target.","None." if compatible else "Recreate the split using the declared grouping target.")
    ms=x.feature_manifest_validation.status; add("feature_manifest_passed",ms,f"Feature-manifest validation status is `{ms}`.","None." if ms=="pass" else "Resolve feature-provenance issues before evaluation.")
    ls=x.leakage_audit.status; add("leakage_audit_status",ls,f"Leakage-audit status is `{ls}`.","None." if ls=="pass" else "Review and resolve the embedded leakage-audit issues.")
    checks=pd.DataFrame(rows); issues=checks.loc[checks.status!="pass"].reset_index(drop=True); status="fail" if (checks.status=="fail").any() else "review" if (checks.status=="review").any() else "pass"
    summary=pd.DataFrame({"status":["pass","review","fail"],"n_checks":[int((checks.status==s).sum()) for s in ["pass","review","fail"]]})
    return GazepointMLSplitValidation(status=status,summary=summary,checks=checks,issues=issues,leakage_audit=x.leakage_audit,feature_manifest_validation=x.feature_manifest_validation)


def write_gazepoint_ml_split_csv(x:GazepointMLSplit,directory:str|Path,prefix:str="gazepoint_ml_split",tables:list[str]|tuple[str,...]|str=("analysis","assessment","excluded","assignment","summary","group_counts","checks","issues"),overwrite:bool=False,na:str="")->dict[str,str]:
    if not isinstance(x,GazepointMLSplit): raise GP3MLError("`x` must be a `gazepoint_ml_split` object.")
    if not isinstance(prefix,str) or not prefix.strip(): raise GP3MLError("`prefix` must be a single non-empty string.")
    if re.search(r"[/\\]",prefix): raise GP3MLError("`prefix` must not contain directory separators.")
    valid=["analysis","assessment","excluded","assignment","summary","group_counts","checks","issues"]
    requested=[tables] if isinstance(tables,str) else list(tables)
    if not requested or any(t not in valid for t in requested): raise GP3MLError("`tables` must use values from: "+", ".join(valid)+".")
    requested=list(dict.fromkeys(requested)); d=Path(directory).expanduser(); d.mkdir(parents=True,exist_ok=True)
    data={"analysis":x.analysis,"assessment":x.assessment,"excluded":x.excluded,"assignment":x.assignment,"summary":x.summary,"group_counts":x.group_counts,"checks":x.validation.checks,"issues":x.validation.issues}
    paths={t:d/f"{prefix}_{t}.csv" for t in requested}; existing=[p for p in paths.values() if p.exists()]
    if existing and not overwrite: raise GP3MLError("Output files already exist: "+", ".join(str(p) for p in existing)+".")
    for t,p in paths.items(): data[t].to_csv(p,index=False,na_rep=na,encoding="utf-8")
    return {t:p.resolve().as_posix() for t,p in paths.items()}


def _split_repr(self:GazepointMLSplit)->str:
    return f"<gazepoint_ml_split>\nTarget: {self.metadata['generalization_target']}\nStatus: {self.validation.status.upper()}\nRows: analysis={len(self.analysis)}, assessment={len(self.assessment)}, excluded={len(self.excluded)}\nSeed: {self.metadata['seed']}"
GazepointMLSplit.__repr__=_split_repr  # type: ignore[method-assign]
def _val_repr(self:GazepointMLSplitValidation)->str:
    return f"<gazepoint_ml_split_validation>\nOverall status: {self.status.upper()}\nNon-passing checks: {len(self.issues)}\n{self.summary.to_string(index=False)}"
GazepointMLSplitValidation.__repr__=_val_repr  # type: ignore[method-assign]
