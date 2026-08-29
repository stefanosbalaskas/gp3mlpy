from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd

from .exceptions import GP3MLError
from .leakage import audit_gazepoint_ml_leakage
from .objects import GP3MLObject, GazepointGroupFolds, GazepointGroupFoldsAudit, GazepointGroupFoldsValidation
from .splitting import TARGETS, _group_counts, _group_values, _manifest_for_predictors, _scalar_column, _trial_units


class GazepointGroupFold(GP3MLObject):
    r_class="gazepoint_group_fold"


def _integer(x:Any,argument:str,minimum:int=1)->list[int]:
    vals=[x] if isinstance(x,(int,np.integer)) and not isinstance(x,bool) else list(x) if isinstance(x,(list,tuple,np.ndarray)) else None
    if vals is None or not vals or any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or int(v)<minimum for v in vals):
        raise GP3MLError(f"`{argument}` must contain finite integers greater than or equal to {minimum}.")
    return [int(v) for v in vals]


def _v(v:Any,target:str)->dict[str,int]:
    vals=_integer(v,"v",2)
    if target=="new_participants_and_new_stimuli":
        if len(vals) not in (1,2): raise GP3MLError("For simultaneous participant and stimulus generalization, `v` must have length one or two.")
        if len(vals)==1: vals*=2
        return {"participant":vals[0],"stimulus":vals[1]}
    if len(vals)!=1: raise GP3MLError("`v` must be a single integer for this generalization target.")
    return {"group":vals[0]}


def _assign_groups(values:np.ndarray,v:int,rng:np.random.Generator)->tuple[np.ndarray,pd.DataFrame]:
    values=np.asarray(values,dtype=str); groups=np.array(sorted(set(values)))
    if len(groups)<v: raise GP3MLError(f"At least {v} distinct groups are required; only {len(groups)} were found.")
    sizes=np.array([(values==g).sum() for g in groups],dtype=int)
    random_ties=rng.permutation(np.arange(1,len(groups)+1))
    order=np.lexsort((random_ties,-sizes))
    fold_rows=np.zeros(v,dtype=int); fold_groups=np.zeros(v,dtype=int); group_fold=np.zeros(len(groups),dtype=int)
    for gp in order:
        candidates=np.flatnonzero(fold_rows==fold_rows.min())
        candidates=candidates[fold_groups[candidates]==fold_groups[candidates].min()]
        selected=int(candidates[0] if len(candidates)==1 else rng.choice(candidates))
        group_fold[gp]=selected+1; fold_rows[selected]+=sizes[gp]; fold_groups[selected]+=1
    mapping=pd.DataFrame({"group":groups,"fold":group_fold,"n_rows":sizes})
    lookup=dict(zip(groups,group_fold,strict=True)); row_fold=np.array([lookup[v] for v in values],dtype=int)
    return row_fold,mapping


def _fold_id(repeat_id:int,fold_id:int,participant_fold:float|int|None=None,stimulus_fold:float|int|None=None)->str:
    if participant_fold is not None and stimulus_fold is not None and not pd.isna(participant_fold) and not pd.isna(stimulus_fold): return f"Repeat{repeat_id:02d}_P{int(participant_fold):02d}_S{int(stimulus_fold):02d}"
    return f"Repeat{repeat_id:02d}_Fold{fold_id:02d}"


def create_gazepoint_group_folds(
    data:pd.DataFrame,outcome:str,predictors:list[str]|tuple[str,...]|str,feature_manifest:pd.DataFrame,generalization_target:str,
    participant_id:str|None=None,trial_id:str|None=None,stimulus_id:str|None=None,v:Any=5,repeats:int=1,seed:int=1,source_row_id:str=".gp3ml_source_row"
)->GazepointGroupFolds:
    """Create deterministic repeated group-aware resampling folds."""
    if not isinstance(data,pd.DataFrame): raise GP3MLError("`data` must be a data frame.")
    if len(data)<2: raise GP3MLError("`data` must contain at least two rows.")
    outcome=_scalar_column(outcome,"outcome",False); source_row_id=_scalar_column(source_row_id,"source_row_id",False); participant_id=_scalar_column(participant_id,"participant_id"); trial_id=_scalar_column(trial_id,"trial_id"); stimulus_id=_scalar_column(stimulus_id,"stimulus_id")
    if isinstance(predictors,str): pred=[predictors.strip()]
    elif isinstance(predictors,(list,tuple,np.ndarray,pd.Index)): pred=[str(x).strip() for x in predictors]
    else: raise GP3MLError("`predictors` must contain non-empty column names.")
    if not pred or any(not p for p in pred): raise GP3MLError("`predictors` must contain non-empty column names.")
    if len(pred)!=len(set(pred)): raise GP3MLError("`predictors` must contain unique column names.")
    if outcome in pred: raise GP3MLError("`outcome` must not be included in `predictors`.")
    idpred=[p for p in pred if p in [participant_id,trial_id,stimulus_id,source_row_id]]
    if idpred: raise GP3MLError("Identifier columns must not be predictors: "+", ".join(idpred)+".")
    if generalization_target not in TARGETS: raise GP3MLError("`generalization_target` must be one of: "+", ".join(TARGETS)+".")
    vv=_v(v,generalization_target); repeats_i=_integer(repeats,"repeats",1); seed_i=_integer(seed,"seed",0)
    if len(repeats_i)!=1: raise GP3MLError("`repeats` must be a single integer.")
    if len(seed_i)!=1: raise GP3MLError("`seed` must be a single integer.")
    repeats=repeats_i[0]; seed=seed_i[0]
    ok={"new_trials_known_participants":participant_id is not None and trial_id is not None,"new_participants":participant_id is not None,"new_stimuli":stimulus_id is not None,"new_participants_and_new_stimuli":participant_id is not None and stimulus_id is not None}[generalization_target]
    if not ok: raise GP3MLError(f'Required grouping identifiers were not supplied for `generalization_target = "{generalization_target}"`.')
    if source_row_id in data.columns: raise GP3MLError(f"`data` already contains the reserved source-row column `{source_row_id}`.")
    missing=[c for c in dict.fromkeys([outcome,*pred,*[x for x in [participant_id,trial_id,stimulus_id] if x]]) if c not in data.columns]
    if missing: raise GP3MLError("`data` is missing required columns: "+", ".join(missing)+".")
    manifest,mvalid=_manifest_for_predictors(feature_manifest,pred)
    participant=_group_values(data,participant_id,"participant_id") if participant_id else None; trial=_group_values(data,trial_id,"trial_id") if trial_id else None; stimulus=_group_values(data,stimulus_id,"stimulus_id") if stimulus_id else None
    if generalization_target=="new_participants" and len(set(participant))<vv["group"]: raise GP3MLError("`v` exceeds the number of distinct participants.")
    if generalization_target=="new_stimuli" and len(set(stimulus))<vv["group"]: raise GP3MLError("`v` exceeds the number of distinct stimuli.")
    if generalization_target=="new_participants_and_new_stimuli":
        if len(set(participant))<vv["participant"]: raise GP3MLError("Participant `v` exceeds the number of distinct participants.")
        if len(set(stimulus))<vv["stimulus"]: raise GP3MLError("Stimulus `v` exceeds the number of distinct stimuli.")
    trial_units=_trial_units(participant,trial) if generalization_target=="new_trials_known_participants" else None
    if trial_units is not None:
        insufficient=[]
        for p in sorted(set(participant)):
            if len(set(trial_units[participant==p]))<vv["group"]: insufficient.append(p)
        if insufficient: raise GP3MLError(f"Each participant must have at least {vv['group']} distinct participant-trial units. Insufficient participants: {', '.join(insufficient)}.")
    rng=np.random.default_rng(seed); source_rows=np.arange(1,len(data)+1,dtype=int); fold_data=data.copy(); fold_data[source_row_id]=source_rows
    folds={}; assignments=[]; summaries=[]; counts_rows=[]; mappings=[]
    fpr=vv["participant"]*vv["stimulus"] if generalization_target=="new_participants_and_new_stimuli" else vv["group"]
    for ri in range(1,repeats+1):
        group_fold=np.full(len(data),np.nan); participant_fold=np.full(len(data),np.nan); stimulus_fold=np.full(len(data),np.nan); split_unit=np.full(len(data),None,dtype=object)
        if generalization_target=="new_participants":
            gf,mapdf=_assign_groups(participant,vv["group"],rng); group_fold=gf.astype(float); participant_fold=gf.astype(float); split_unit=participant.copy(); mapdf.insert(0,"repeat",ri); mapdf.insert(1,"unit","participant"); mappings.append(mapdf[["repeat","unit","group","fold","n_rows"]])
        elif generalization_target=="new_stimuli":
            gf,mapdf=_assign_groups(stimulus,vv["group"],rng); group_fold=gf.astype(float); stimulus_fold=gf.astype(float); split_unit=stimulus.copy(); mapdf.insert(0,"repeat",ri); mapdf.insert(1,"unit","stimulus"); mappings.append(mapdf[["repeat","unit","group","fold","n_rows"]])
        elif generalization_target=="new_trials_known_participants":
            split_unit=trial_units.copy()
            for p in sorted(set(participant)):
                selected=participant==p; gf,mapdf=_assign_groups(trial_units[selected],vv["group"],rng); group_fold[selected]=gf; mapdf.insert(0,"repeat",ri); mapdf.insert(1,"unit","participant_trial"); mapdf.insert(2,"participant",p); mappings.append(mapdf[["repeat","unit","participant","group","fold","n_rows"]])
        else:
            pf,pmap=_assign_groups(participant,vv["participant"],rng); sf,smap=_assign_groups(stimulus,vv["stimulus"],rng); participant_fold=pf.astype(float); stimulus_fold=sf.astype(float); split_unit=np.array([f"{len(p)}:{p}|{len(s)}:{s}" for p,s in zip(participant,stimulus,strict=True)],dtype=object); pmap.insert(0,"repeat",ri); pmap.insert(1,"unit","participant"); smap.insert(0,"repeat",ri); smap.insert(1,"unit","stimulus"); mappings.extend([pmap[["repeat","unit","group","fold","n_rows"]],smap[["repeat","unit","group","fold","n_rows"]]])
        specs=[]
        if generalization_target=="new_participants_and_new_stimuli":
            for sf in range(1,vv["stimulus"]+1):
                for pf in range(1,vv["participant"]+1): specs.append((pf,sf))  # R expand.grid first factor varies fastest
        else: specs=list(range(1,vv["group"]+1))
        for fi,spec in enumerate(specs,start=1):
            if generalization_target=="new_participants_and_new_stimuli":
                hp,hs=spec; ph=participant_fold==hp; sh=stimulus_fold==hs; partition=np.where(ph&sh,"assessment",np.where(~ph&~sh,"analysis","excluded")); heldp=hp; helds=hs
            else:
                held=int(spec); partition=np.where(group_fold==held,"assessment","analysis"); heldp=held if generalization_target=="new_participants" else None; helds=held if generalization_target=="new_stimuli" else None
            fid=_fold_id(ri,fi,heldp,helds); aidx=source_rows[partition=="analysis"]; eidx=source_rows[partition=="assessment"]; xidx=source_rows[partition=="excluded"]
            if len(aidx)==0 or len(eidx)==0: raise GP3MLError(f"Fold `{fid}` produced an empty analysis or assessment partition.")
            a=fold_data.loc[partition=="analysis"].reset_index(drop=True); e=fold_data.loc[partition=="assessment"].reset_index(drop=True); ex=fold_data.loc[partition=="excluded"].reset_index(drop=True)
            leak=audit_gazepoint_ml_leakage(a,e,outcome,pred,participant_id,trial_id,stimulus_id,generalization_target)
            folds[fid]=GazepointGroupFold(**{"repeat":ri,"fold":fi,"fold_id":fid,"participant_fold":heldp,"stimulus_fold":helds,"analysis":a,"assessment":e,"excluded":ex,"analysis_indices":aidx.tolist(),"assessment_indices":eidx.tolist(),"excluded_indices":xidx.tolist(),"leakage_audit":leak})
            assignments.append(pd.DataFrame({"repeat":ri,"fold":fi,"fold_id":fid,"source_row":source_rows,"partition":partition,"split_unit":split_unit,"group_fold":group_fold,"participant_fold":participant_fold,"stimulus_fold":stimulus_fold}))
            summaries.append({"repeat":ri,"fold":fi,"fold_id":fid,"participant_fold":heldp,"stimulus_fold":helds,"n_total":len(data),"n_analysis":len(aidx),"n_assessment":len(eidx),"n_excluded":len(xidx),"assessment_prop_all":len(eidx)/len(data),"assessment_prop_retained":len(eidx)/(len(aidx)+len(eidx)),"leakage_status":leak.status})
            c=_group_counts(data,partition,participant_id,trial_id,stimulus_id); c.insert(0,"repeat",ri); c.insert(1,"fold",fi); c.insert(2,"fold_id",fid); counts_rows.append(c[["repeat","fold","fold_id","partition","unit","n_groups"]])
    # Mapping columns differ only for trial target; concat union matches intended table semantics.
    result=GazepointGroupFolds(folds=folds,assignments=pd.concat(assignments,ignore_index=True),fold_summary=pd.DataFrame(summaries),group_counts=pd.concat(counts_rows,ignore_index=True),group_mapping=pd.concat(mappings,ignore_index=True,sort=False),feature_manifest=manifest,feature_manifest_validation=mvalid,metadata={"outcome":outcome,"predictors":pred,"participant_id":participant_id,"trial_id":trial_id,"stimulus_id":stimulus_id,"generalization_target":generalization_target,"v":vv,"repeats":repeats,"seed":seed,"source_row_id":source_row_id,"n_source_rows":len(data),"n_folds_per_repeat":int(fpr),"n_folds_total":int(fpr*repeats)})
    result.audit=audit_gazepoint_group_folds(result); result.validation=validate_gazepoint_group_folds(result); return result


def audit_gazepoint_group_folds(x:GazepointGroupFolds)->GazepointGroupFoldsAudit:
    if not isinstance(x,GazepointGroupFolds): raise GP3MLError("`x` must be a `gazepoint_group_folds` object.")
    if not x.folds: raise GP3MLError("`x` does not contain fold audit results.")
    summary=[]; checkrows=[]
    for fo in x.folds.values():
        audit=fo.leakage_audit
        if audit is None or not isinstance(audit.checks,pd.DataFrame): raise GP3MLError(f"Fold `{fo.fold_id}` does not contain a compatible leakage audit.")
        summary.append({"repeat":fo["repeat"],"fold":fo.fold,"fold_id":fo.fold_id,"status":audit.status,"n_pass":int((audit.checks.status=="pass").sum()),"n_review":int((audit.checks.status=="review").sum()),"n_fail":int((audit.checks.status=="fail").sum())})
        c=audit.checks.copy(); c.insert(0,"repeat",fo["repeat"]); c.insert(1,"fold",fo.fold); c.insert(2,"fold_id",fo.fold_id); checkrows.append(c)
    s=pd.DataFrame(summary); checks=pd.concat(checkrows,ignore_index=True); issues=checks.loc[checks.status!="pass"].reset_index(drop=True); status="fail" if (s.status=="fail").any() else "review" if (s.status=="review").any() else "pass"
    return GazepointGroupFoldsAudit(status=status,summary=s,checks=checks,issues=issues)


def validate_gazepoint_group_folds(x:GazepointGroupFolds)->GazepointGroupFoldsValidation:
    if not isinstance(x,GazepointGroupFolds): raise GP3MLError("`x` must be a `gazepoint_group_folds` object.")
    required=["folds","assignments","fold_summary","group_counts","group_mapping","feature_manifest_validation","audit","metadata"]; missing=[r for r in required if r not in x]
    if missing: raise GP3MLError("Fold object is missing components: "+", ".join(missing)+".")
    checks=[]
    def add(i,s,m,r): checks.append({"check_id":i,"status":s,"message":m,"remediation":r})
    expected=int(x.metadata["n_folds_total"]); observed=len(x.folds); ok=observed==expected; add("fold_count","pass" if ok else "fail",f"Expected {expected} folds and found {observed}.","None." if ok else "Recreate the fold plan.")
    ids=[f.fold_id for f in x.folds.values()]; ok=len(ids)==len(set(ids)) and all(ids); add("fold_ids_unique","pass" if ok else "fail","Fold identifiers are unique and non-empty." if ok else "Fold identifiers are duplicated or invalid.","None." if ok else "Recreate fold identifiers.")
    ok=isinstance(x.assignments,pd.DataFrame) and len(x.assignments)>0 and set(x.assignments.partition).issubset({"analysis","assessment","excluded"}); add("assignment_partitions_valid","pass" if ok else "fail","Assignment partitions use valid labels." if ok else "Assignment partitions contain invalid labels.","None." if ok else "Recreate the assignment table.")
    expected_rows=list(range(1,int(x.metadata["n_source_rows"])+1)); groups=list(x.assignments.groupby(["repeat","fold"],sort=False)); ok=len(groups)==expected and all(sorted(g.source_row.tolist())==expected_rows and not g.source_row.duplicated().any() for _,g in groups); add("source_rows_accounted_per_fold","pass" if ok else "fail","Every fold accounts for each source row exactly once." if ok else "Source-row accounting is incomplete or duplicated.","None." if ok else "Reconstruct fold assignments from the source data.")
    ok=all(len(f.analysis)>0 and len(f.assessment)>0 for f in x.folds.values()); add("analysis_assessment_non_empty","pass" if ok else "fail","All folds have non-empty analysis and assessment partitions." if ok else "At least one fold has an empty analysis or assessment partition.","None." if ok else "Reduce `v` or revise grouping.")
    ok=x.metadata["generalization_target"]=="new_participants_and_new_stimuli" or not (x.assignments.partition=="excluded").any(); add("excluded_rows_compatible","pass" if ok else "fail","Excluded rows are compatible with the generalization target." if ok else "Unexpected excluded rows were found.","None." if ok else "Recreate folds using the target.")
    coverage=x.assignments.assign(_a=(x.assignments.partition=="assessment").astype(int)).groupby(["repeat","source_row"],as_index=False)["_a"].sum().rename(columns={"_a":"n_assessment"}); ok=len(coverage)==int(x.metadata["repeats"])*int(x.metadata["n_source_rows"]) and (coverage.n_assessment==1).all(); add("assessment_coverage_once_per_repeat","pass" if ok else "fail","Every source row appears in assessment exactly once per repeat." if ok else "Assessment coverage is incomplete or duplicated.","None." if ok else "Recreate the fold assignments.")
    material=True
    for fo in x.folds.values():
        a=x.assignments[(x.assignments["repeat"]==fo["repeat"])&(x.assignments.fold==fo.fold)]; material &= sorted(a.loc[a.partition=="analysis","source_row"])==sorted(fo.analysis_indices) and sorted(a.loc[a.partition=="assessment","source_row"])==sorted(fo.assessment_indices) and sorted(a.loc[a.partition=="excluded","source_row"])==sorted(fo.excluded_indices)
    add("materialized_partitions_match_assignments","pass" if material else "fail","Materialized partitions match the assignment table." if material else "Materialized partitions and assignments disagree.","None." if material else "Recreate the fold object.")
    ms=x.feature_manifest_validation.status; add("feature_manifest_passed",ms,f"Feature-manifest validation status is `{ms}`.","None." if ms=="pass" else "Resolve feature-provenance issues.")
    au=x.audit.status; add("fold_leakage_audit",au,f"Aggregated fold leakage-audit status is `{au}`.","None." if au=="pass" else "Review the fold-level leakage-audit issues.")
    c=pd.DataFrame(checks); issues=c.loc[c.status!="pass"].reset_index(drop=True); status="fail" if (c.status=="fail").any() else "review" if (c.status=="review").any() else "pass"; summary=pd.DataFrame({"status":["pass","review","fail"],"n_checks":[int((c.status==s).sum()) for s in ["pass","review","fail"]]})
    return GazepointGroupFoldsValidation(status=status,summary=summary,checks=c,issues=issues,assessment_coverage=coverage)


def write_gazepoint_group_folds_csv(
    x:GazepointGroupFolds, directory:str|Path, prefix:str="gazepoint_group_folds",
    tables:Any=("assignments","fold_summary","group_counts","group_mapping","validation_checks","validation_issues","audit_summary","audit_checks","audit_issues"),
    include_fold_data:bool=False, overwrite:bool=False, na:str=""
)->dict[str,str]:
    if not isinstance(x,GazepointGroupFolds): raise GP3MLError("`x` must be a `gazepoint_group_folds` object.")
    if not isinstance(prefix,str) or not prefix.strip(): raise GP3MLError("`prefix` must be a single non-empty string.")
    if re.search(r"[/\\]",prefix): raise GP3MLError("`prefix` must not contain directory separators.")
    data={"assignments":x.assignments,"fold_summary":x.fold_summary,"group_counts":x.group_counts,"group_mapping":x.group_mapping,"validation_checks":x.validation.checks,"validation_issues":x.validation.issues,"audit_summary":x.audit.summary,"audit_checks":x.audit.checks,"audit_issues":x.audit.issues}
    req=[tables] if isinstance(tables,str) else list(tables)
    if not req or any(t not in data for t in req): raise GP3MLError("`tables` must use values from: "+", ".join(data)+".")
    out={t:data[t] for t in dict.fromkeys(req)}
    if include_fold_data:
        for fo in x.folds.values():
            for part in ("analysis","assessment","excluded"): out[f"{fo.fold_id}_{part}"]=fo[part]
    d=Path(directory).expanduser(); d.mkdir(parents=True,exist_ok=True); paths={t:d/f"{prefix}_{t}.csv" for t in out}; existing=[p for p in paths.values() if p.exists()]
    if existing and not overwrite: raise GP3MLError("Output files already exist: "+", ".join(str(p) for p in existing)+".")
    for t,p in paths.items(): out[t].to_csv(p,index=False,na_rep=na,encoding="utf-8")
    return {t:p.resolve().as_posix() for t,p in paths.items()}

def _repr(self:GazepointGroupFolds)->str: return f"<gazepoint_group_folds>\nTarget: {self.metadata['generalization_target']}\nRepeats: {self.metadata['repeats']}\nFolds per repeat: {self.metadata['n_folds_per_repeat']}\nTotal folds: {self.metadata['n_folds_total']}\nStatus: {self.validation.status.upper()}"
GazepointGroupFolds.__repr__=_repr  # type: ignore[method-assign]
