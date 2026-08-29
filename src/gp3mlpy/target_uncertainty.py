from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import numpy as np
import pandas as pd
from ._utils import write_tables, worst_status
from .exceptions import GP3MLError
from .metrics import gazepoint_performance_metrics
from .objects import GP3MLResampleUncertainty, GP3MLTargetUncertainty, GP3MLUncertaintyValidation, GP3MLObject
from .task_governance import assert_gp3ml_use_case

def _cluster_bootstrap_indices(ids:Sequence[Any],rng:np.random.RandomState)->np.ndarray:
    arr=np.asarray(ids,dtype=object)
    if pd.isna(arr).any() or any(str(x)=="" for x in arr):raise GP3MLError("Cluster identifiers may not be missing or empty.")
    clusters=list(dict.fromkeys(str(x) for x in arr));sampled=rng.choice(clusters,size=len(clusters),replace=True);return np.concatenate([np.flatnonzero(np.asarray([str(x)==value for x in arr])) for value in sampled])

def _two_way_indices(participant_id,stimulus_id,rng,max_attempts=100):
    p=np.asarray(participant_id,dtype=object);s=np.asarray(stimulus_id,dtype=object)
    if len(p)!=len(s):raise GP3MLError("Participant and stimulus identifiers must have equal length.")
    if pd.isna(p).any() or pd.isna(s).any():raise GP3MLError("Two-way cluster identifiers may not be missing.")
    participants=list(dict.fromkeys(str(x) for x in p));stimuli=list(dict.fromkeys(str(x) for x in s));ps=np.asarray([str(x) for x in p]);ss=np.asarray([str(x) for x in s])
    for _ in range(max_attempts):
        pdrawing=rng.choice(participants,size=len(participants),replace=True);sdrawing=rng.choice(stimuli,size=len(stimuli),replace=True);pc={x:int(np.sum(pdrawing==x)) for x in participants};sc={x:int(np.sum(sdrawing==x)) for x in stimuli};weights=np.asarray([pc.get(a,0)*sc.get(b,0) for a,b in zip(ps,ss,strict=True)],dtype=int);idx=np.repeat(np.arange(len(weights)),weights)
        if len(idx):return idx
    return np.asarray([],dtype=int)

def _observation_indices(truth,classification,stratify,rng):
    n=len(truth)
    if classification and stratify:
        vals=np.asarray([str(x) for x in truth],dtype=object);groups=sorted(set(vals))
        if len(groups)!=2:return np.asarray([],dtype=int)
        return np.concatenate([rng.choice(np.flatnonzero(vals==g),size=int(np.sum(vals==g)),replace=True) for g in groups])
    return rng.choice(np.arange(n),size=n,replace=True)

def bootstrap_gazepoint_metrics_by_unit(task,truth,prediction=None,probability=None,participant_id=None,stimulus_id=None,unit:str="observation",bootstrap:int=1000,conf_level:float=.95,seed:int=1,threshold:float=.5,stratify_observations:bool=True):
    assert_gp3ml_use_case(task)
    if unit not in {"observation","participant","stimulus","participant_and_stimulus"}:raise GP3MLError("Unknown resampling `unit`.")
    bootstrap=int(bootstrap)
    if bootstrap<1:raise GP3MLError("`bootstrap` must be positive.")
    truth_arr=np.asarray(truth,dtype=object);n=len(truth_arr)
    if n<2:raise GP3MLError("At least two observations are required.")
    pred_arr=None if prediction is None else np.asarray(prediction,dtype=object);prob_arr=None if probability is None else np.asarray(probability,dtype=float)
    if task.task_type=="classification" and (prob_arr is None or len(prob_arr)!=n):raise GP3MLError("`probability` must match `truth`.")
    if task.task_type=="regression" and (pred_arr is None or len(pred_arr)!=n):raise GP3MLError("`prediction` must match `truth`.")
    if unit in {"participant","participant_and_stimulus"} and (participant_id is None or len(participant_id)!=n):raise GP3MLError("`participant_id` must match `truth` for this resampling unit.")
    if unit in {"stimulus","participant_and_stimulus"} and (stimulus_id is None or len(stimulus_id)!=n):raise GP3MLError("`stimulus_id` must match `truth` for this resampling unit.")
    point=gazepoint_performance_metrics(task,truth_arr,pred_arr,prob_arr,threshold);rng=np.random.RandomState(int(seed));draws=[];failures=[];replicate_sizes=[]
    for i in range(1,bootstrap+1):
        if unit=="observation":idx=_observation_indices(truth_arr,task.task_type=="classification",stratify_observations,rng)
        elif unit=="participant":idx=_cluster_bootstrap_indices(participant_id,rng)
        elif unit=="stimulus":idx=_cluster_bootstrap_indices(stimulus_id,rng)
        else:idx=_two_way_indices(participant_id,stimulus_id,rng)
        replicate_sizes.append(len(idx))
        if not len(idx):failures.append({"replicate":i,"error":"No rows were selected."});continue
        try:
            draw=gazepoint_performance_metrics(task,truth_arr[idx],None if pred_arr is None else pred_arr[idx],None if prob_arr is None else prob_arr[idx],threshold).copy();draw["replicate"]=i;draw["resample_n"]=len(idx);draws.append(draw)
        except Exception as exc:failures.append({"replicate":i,"error":str(exc)})
    if not draws:raise GP3MLError("Every bootstrap replicate failed.")
    draws_df=pd.concat(draws,ignore_index=True);failures_df=pd.DataFrame(failures,columns=["replicate","error"]);metrics=[c for c in draws_df.columns if pd.api.types.is_numeric_dtype(draws_df[c]) and c not in {"n","threshold","replicate","resample_n"}];alpha=(1-conf_level)/2;intervals=pd.DataFrame([{"metric":m,"estimate":float(point[m].iloc[0]),"lower":float(draws_df[m].quantile(alpha)),"upper":float(draws_df[m].quantile(1-alpha)),"successful_replicates":int(np.isfinite(draws_df[m]).sum())} for m in metrics]);limitations={"observation":"Observation-level intervals do not represent participant- or stimulus-cluster uncertainty.","participant":"Participant-cluster intervals preserve participant rows but do not independently resample stimuli.","stimulus":"Stimulus-cluster intervals preserve stimulus rows but do not independently resample participants.","participant_and_stimulus":"Two-way product-weight bootstrap reflects simultaneous participant and stimulus resampling and may produce variable replicate sizes."}[unit]
    return GP3MLTargetUncertainty(point=point,intervals=intervals,draws=draws_df,failures=failures_df,bootstrap=bootstrap,successful_replicates=int(draws_df.replicate.nunique()),failed_replicates=len(failures_df),conf_level=conf_level,seed=seed,unit=unit,generalization_target=task.generalization_target,task=task,replicate_sizes=np.asarray(replicate_sizes,dtype=int),limitations=limitations,call="bootstrap_gazepoint_metrics_by_unit")

def summarize_gazepoint_resample_uncertainty(evaluation,unit:str="fold",conf_level:float=.95):
    if not isinstance(evaluation,GP3MLObject) or evaluation.r_class not in {"gp3ml_resample_evaluation","gp3ml_nested_evaluation"}:raise GP3MLError("`evaluation` must be a grouped or nested evaluation object.")
    if unit not in {"fold","repeat"}:raise GP3MLError("`unit` must be one of: fold, repeat.")
    metrics=evaluation.metrics
    if metrics.empty:raise GP3MLError("The evaluation contains no metric values.")
    distribution=metrics.groupby(["repeat","metric"],as_index=False)["value"].mean() if unit=="repeat" else metrics[["repeat","fold","fold_id","metric","value"]].copy();alpha=(1-conf_level)/2;rows=[]
    for metric in pd.unique(distribution.metric):
        vals=pd.to_numeric(distribution.loc[distribution.metric==metric,"value"],errors="coerce");finite=vals[np.isfinite(vals)];rows.append({"metric":metric,"distribution_unit":unit,"n_units":len(finite),"mean":float(finite.mean()) if len(finite) else np.nan,"median":float(finite.median()) if len(finite) else np.nan,"sd":float(finite.std(ddof=1)) if len(finite)>1 else np.nan,"lower":float(finite.quantile(alpha)) if len(finite) else np.nan,"upper":float(finite.quantile(1-alpha)) if len(finite) else np.nan})
    return GP3MLResampleUncertainty(summary=pd.DataFrame(rows),distribution=distribution,unit=unit,conf_level=conf_level,generalization_target=evaluation.generalization_target,limitations=f"Intervals summarize the empirical {unit} distribution and are not a substitute for an undeclared cluster bootstrap.")

def validate_gazepoint_target_uncertainty(x):
    if not isinstance(x,GP3MLObject) or x.r_class not in {"gp3ml_target_uncertainty","gp3ml_resample_uncertainty"}:raise GP3MLError("`x` must be a gp3ml uncertainty object.")
    is_target=x.r_class=="gp3ml_target_uncertainty";statuses=["pass" if getattr(x,"unit",None) else "fail","pass" if getattr(x,"generalization_target",None) else "fail","pass" if getattr(x,"limitations",None) else "fail","pass" if (not is_target or getattr(x,"failed_replicates",None) is not None) else "fail"];checks=pd.DataFrame({"check_id":["unit_recorded","target_recorded","limitations_recorded","failed_replicates_recorded"],"status":statuses,"message":[f"Resampling unit: {getattr(x,'unit',None)}",f"Generalization target: {getattr(x,'generalization_target',None)}",str(getattr(x,'limitations',None)),f"Failed replicates: {x.failed_replicates}" if is_target else "Not a bootstrap object."]});return GP3MLUncertaintyValidation(status=worst_status(statuses),checks=checks,issues=checks[checks.status!="pass"].reset_index(drop=True))

def write_gazepoint_target_uncertainty(x,directory,prefix="gazepoint_target_uncertainty",overwrite=False):
    validation=validate_gazepoint_target_uncertainty(x);tables={"point":x.point,"intervals":x.intervals,"draws":x.draws,"failures":x.failures,"validation":validation.checks} if x.r_class=="gp3ml_target_uncertainty" else {"summary":x.summary,"distribution":x.distribution,"validation":validation.checks};return write_tables(tables,directory,prefix,overwrite)
