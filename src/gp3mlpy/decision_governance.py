from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any
import numpy as np
import pandas as pd
from ._utils import worst_status
from .exceptions import GP3MLError
from .objects import GP3MLAbstentionAudit,GP3MLDecisionRule,GP3MLDecisionRuleValidation,GP3MLThresholdEvaluation

def _assert_probability(probability:Sequence[float]|np.ndarray)->np.ndarray:
    try:values=np.asarray(probability,dtype=float)
    except Exception as exc:raise GP3MLError("`probability` must contain probabilities in [0, 1].") from exc
    finite=values[~np.isnan(values)]
    if np.any(~np.isfinite(finite)) or np.any((finite<0)|(finite>1)):raise GP3MLError("`probability` must contain probabilities in [0, 1].")
    return values

def create_gazepoint_decision_rule(metric:str,direction:str="maximize",threshold:float|None=None,threshold_origin:str="predeclared",cost_false_positive:float=1,cost_false_negative:float=1,abstention_allowed:bool=False,abstention_interval:Sequence[float]|None=None,calibration_source:str="none",training_partition:str="analysis",generalization_target:str|None=None,scientific_justification:str|None=None)->GP3MLDecisionRule:
    if direction not in {"maximize","minimize"}:raise GP3MLError("`direction` must be one of: maximize, minimize.")
    if threshold_origin not in {"predeclared","training","inner_resampling"}:raise GP3MLError("`threshold_origin` must be one of: predeclared, training, inner_resampling.")
    if not isinstance(metric,str) or metric=="":raise GP3MLError("`metric` must be one non-empty metric name.")
    if threshold is not None:
        if not isinstance(threshold,(int,float,np.number)) or not math.isfinite(float(threshold)) or not 0<float(threshold)<1:raise GP3MLError("`threshold` must be NULL or one probability strictly between 0 and 1.")
        threshold=float(threshold)
    costs=[float(cost_false_positive),float(cost_false_negative)]
    if any(not math.isfinite(x) or x<0 for x in costs):raise GP3MLError("Decision costs must be finite and non-negative.")
    interval=None
    if bool(abstention_allowed):
        if abstention_interval is None or len(abstention_interval)!=2:raise GP3MLError("Abstention requires an ordered length-two interval inside [0, 1].")
        interval=[float(x) for x in abstention_interval]
        if any(not math.isfinite(x) for x in interval) or interval[0]<0 or interval[1]>1 or interval[0]>=interval[1]:raise GP3MLError("Abstention requires an ordered length-two interval inside [0, 1].")
    if not isinstance(scientific_justification,str) or scientific_justification.strip()=="":raise GP3MLError("Supply one explicit `scientific_justification`.")
    return GP3MLDecisionRule(metric=metric,direction=direction,threshold=threshold,threshold_origin=threshold_origin,cost_false_positive=costs[0],cost_false_negative=costs[1],abstention_allowed=bool(abstention_allowed),abstention_interval=interval,calibration_source=str(calibration_source),training_partition=str(training_partition),generalization_target="" if generalization_target is None else str(generalization_target),scientific_justification=scientific_justification)

def validate_gazepoint_decision_rule(rule:Any,require_threshold:bool=False)->GP3MLDecisionRuleValidation:
    checks=pd.DataFrame({"check":["class","metric","direction","threshold","threshold_origin","training_partition","generalization_target","scientific_justification","abstention"],"status":"pass","detail":""})
    if not isinstance(rule,GP3MLDecisionRule):checks.loc[checks.check=="class","status"]="fail"
    else:
        if not isinstance(rule.metric,str) or rule.metric=="":checks.loc[checks.check=="metric","status"]="fail"
        if rule.direction not in {"maximize","minimize"}:checks.loc[checks.check=="direction","status"]="fail"
        threshold_ok=rule.threshold is None or (isinstance(rule.threshold,(int,float,np.number)) and math.isfinite(float(rule.threshold)) and 0<float(rule.threshold)<1)
        if not threshold_ok or (require_threshold and rule.threshold is None):checks.loc[checks.check=="threshold","status"]="fail"
        if rule.threshold_origin not in {"predeclared","training","inner_resampling"}:checks.loc[checks.check=="threshold_origin","status"]="fail"
        if rule.training_partition=="assessment" or "external" in str(rule.training_partition).lower():checks.loc[checks.check=="training_partition","status"]="fail"
        if not isinstance(rule.generalization_target,str) or rule.generalization_target=="":checks.loc[checks.check=="generalization_target","status"]="fail"
        if not isinstance(rule.scientific_justification,str) or rule.scientific_justification.strip()=="":checks.loc[checks.check=="scientific_justification","status"]="fail"
        if rule.abstention_allowed:
            z=rule.abstention_interval
            if z is None or len(z)!=2 or z[0]>=z[1]:checks.loc[checks.check=="abstention","status"]="fail"
    return GP3MLDecisionRuleValidation(status=worst_status(checks.status),checks=checks)

def evaluate_gazepoint_thresholds(truth:Sequence[Any],probability:Sequence[float],positive:Any,thresholds:Sequence[float]|None,cost_false_positive:float=1,cost_false_negative:float=1)->GP3MLThresholdEvaluation:
    probability_arr=_assert_probability(probability)
    if thresholds is None or len(thresholds)==0:raise GP3MLError("Supply explicit candidate `thresholds`; gp3ml does not choose a hidden default.")
    thresholds_arr=np.sort(np.unique(np.asarray(thresholds,dtype=float)))
    if np.any(~np.isfinite(thresholds_arr)) or np.any((thresholds_arr<=0)|(thresholds_arr>=1)):raise GP3MLError("All candidate thresholds must be strictly between 0 and 1.")
    truth_arr=np.asarray(truth,dtype=object)
    if len(truth_arr)!=len(probability_arr):raise GP3MLError("`truth` and `probability` lengths differ.")
    keep=pd.notna(truth_arr)&~np.isnan(probability_arr);truth_str=np.asarray([str(x) for x in truth_arr[keep]],dtype=object);prob=probability_arr[keep];positive_str=str(positive);levels_found=sorted(set(truth_str.tolist()))
    if len(levels_found)!=2 or positive_str not in levels_found:raise GP3MLError("`truth` must contain exactly two observed levels and `positive` must identify one.")
    negative=next(x for x in levels_found if x!=positive_str);rows=[]
    for threshold in thresholds_arr:
        predicted=np.where(prob>=threshold,positive_str,negative);tp=int(np.sum((predicted==positive_str)&(truth_str==positive_str)));tn=int(np.sum((predicted==negative)&(truth_str==negative)));fp=int(np.sum((predicted==positive_str)&(truth_str==negative)));fn=int(np.sum((predicted==negative)&(truth_str==positive_str)));sensitivity=tp/(tp+fn) if tp+fn>0 else np.nan;specificity=tn/(tn+fp) if tn+fp>0 else np.nan;precision=tp/(tp+fp) if tp+fp>0 else np.nan;recall=sensitivity;f1=2*precision*recall/(precision+recall) if np.isfinite(precision) and np.isfinite(recall) and precision+recall>0 else np.nan;accuracy=(tp+tn)/len(truth_str);balanced_accuracy=float(np.nanmean([sensitivity,specificity]));expected_cost=(fp*float(cost_false_positive)+fn*float(cost_false_negative))/len(truth_str);rows.append({"threshold":float(threshold),"tp":tp,"tn":tn,"fp":fp,"fn":fn,"sensitivity":sensitivity,"specificity":specificity,"precision":precision,"f1":f1,"accuracy":accuracy,"balanced_accuracy":balanced_accuracy,"expected_cost":expected_cost})
    return GP3MLThresholdEvaluation(positive=positive_str,negative=negative,n=len(truth_str),thresholds=pd.DataFrame(rows),cost_false_positive=cost_false_positive,cost_false_negative=cost_false_negative)

def select_gazepoint_threshold(evaluation:GP3MLThresholdEvaluation,metric:str,direction:str="maximize",threshold_origin:str="inner_resampling",training_partition:str="inner_resampling",generalization_target:str|None=None,scientific_justification:str|None=None,abstention_allowed:bool=False,abstention_interval:Sequence[float]|None=None)->GP3MLDecisionRule:
    if not isinstance(evaluation,GP3MLThresholdEvaluation):raise GP3MLError("`evaluation` must come from `evaluate_gazepoint_thresholds()`.")
    if direction not in {"maximize","minimize"}:raise GP3MLError("`direction` must be one of: maximize, minimize.")
    if threshold_origin not in {"inner_resampling","training"}:raise GP3MLError("`threshold_origin` must be one of: inner_resampling, training.")
    tab=evaluation.thresholds
    if metric not in tab.columns:raise GP3MLError(f"Unknown threshold metric `{metric}`.")
    values=pd.to_numeric(tab[metric],errors="coerce");finite=np.isfinite(values.to_numpy(dtype=float))
    if not finite.any():raise GP3MLError(f"Metric `{metric}` has no finite values.")
    target=values[finite].max() if direction=="maximize" else values[finite].min();candidates=tab.loc[finite&(values==target)];selected=float(candidates.threshold.min());return create_gazepoint_decision_rule(metric=metric,direction=direction,threshold=selected,threshold_origin=threshold_origin,cost_false_positive=evaluation.cost_false_positive,cost_false_negative=evaluation.cost_false_negative,abstention_allowed=abstention_allowed,abstention_interval=abstention_interval,calibration_source="explicit threshold evaluation",training_partition=training_partition,generalization_target=generalization_target,scientific_justification=scientific_justification)

def apply_gazepoint_decision_rule(rule:GP3MLDecisionRule,probability:Sequence[float],positive:Any,negative:Any,abstain_label:str=".abstain")->pd.Categorical:
    validation=validate_gazepoint_decision_rule(rule,require_threshold=True)
    if validation.status!="pass":raise GP3MLError("Decision rule validation failed.")
    probability_arr=_assert_probability(probability);decision=np.where(probability_arr>=rule.threshold,str(positive),str(negative)).astype(object)
    if rule.abstention_allowed:
        inside=(probability_arr>=rule.abstention_interval[0])&(probability_arr<=rule.abstention_interval[1]);decision[inside]=abstain_label
    return pd.Categorical(decision,categories=list(dict.fromkeys([str(negative),str(positive),abstain_label])))

def audit_gazepoint_abstention(truth:Sequence[Any],decision:Sequence[Any],abstain_label:str=".abstain")->GP3MLAbstentionAudit:
    if len(truth)!=len(decision):raise GP3MLError("`truth` and `decision` lengths differ.")
    truth_arr=np.asarray(truth,dtype=object);decision_arr=np.asarray(decision,dtype=object);keep=pd.notna(truth_arr)&pd.notna(decision_arr);truth_str=np.asarray([str(x) for x in truth_arr[keep]],dtype=object);dec_str=np.asarray([str(x) for x in decision_arr[keep]],dtype=object);abstained=dec_str==abstain_label;covered=~abstained;coverage=float(np.mean(covered)) if len(covered) else np.nan;error_rate=float(np.mean(dec_str[covered]!=truth_str[covered])) if np.any(covered) else np.nan;abstention_rate=float(np.mean(abstained)) if len(abstained) else np.nan;levels=sorted(set(truth_str.tolist()));rows=[]
    for level in levels:
        n=int(np.sum(truth_str==level));a=int(np.sum(abstained&(truth_str==level)));rows.append({"truth":level,"n":n,"abstained":a,"abstention_rate":a/n if n else np.nan})
    return GP3MLAbstentionAudit(status="fail" if coverage==0 else "pass",n=len(truth_str),coverage=coverage,abstention_rate=abstention_rate,covered_error_rate=error_rate,by_truth=pd.DataFrame(rows))

def _decision_rule_repr(self:GP3MLDecisionRule)->str:
    threshold="<not selected>" if self.threshold is None else str(self.threshold);return "gp3ml decision rule\n"+f" metric: {self.metric} ({self.direction})\n"+f" threshold: {threshold}\n"+f" origin: {self.threshold_origin}\n"+f" target: {self.generalization_target}\n"+f" abstention: {'enabled' if self.abstention_allowed else 'disabled'}"
GP3MLDecisionRule.__repr__=_decision_rule_repr  # type: ignore[method-assign]
