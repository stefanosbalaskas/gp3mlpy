from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import numpy as np
import pandas as pd
from ._utils import worst_status
from .decision_governance import _assert_probability
from .exceptions import GP3MLError
from .objects import GP3MLConformal, GP3MLConformalCoverage, GP3MLConformalValidation

def _type1_quantile(values:np.ndarray,probability:float)->float:
    finite=np.asarray(values,dtype=float);finite=finite[np.isfinite(finite)]
    if not len(finite):raise GP3MLError("No finite conformity scores are available.")
    return float(np.quantile(finite,np.clip(probability,0,1),method="inverted_cdf"))

def fit_gazepoint_conformal(truth:Sequence[Any],prediction:Sequence[float]|None=None,probability:Sequence[float]|None=None,task_type:str="regression",positive:Any=None,level:float=0.90,calibration_unit:str="observation",unit:Sequence[Any]|None=None,generalization_target:str|None=None)->GP3MLConformal:
    if task_type not in {"regression","classification"}:raise GP3MLError("`task_type` must be one of: regression, classification.")
    if calibration_unit not in {"observation","participant","stimulus","participant_stimulus"}:raise GP3MLError("Invalid `calibration_unit`.")
    if not isinstance(level,(int,float,np.number)) or not np.isfinite(level) or not 0<float(level)<1:raise GP3MLError("`level` must be strictly between 0 and 1.")
    truth_arr=np.asarray(truth,dtype="object");n=len(truth_arr)
    if task_type=="regression":
        if prediction is None or len(prediction)!=n:raise GP3MLError("Regression conformal calibration requires numeric `prediction` aligned with `truth`.")
        try:pred=np.asarray(prediction,dtype=float)
        except Exception as exc:raise GP3MLError("Regression conformal calibration requires numeric `prediction` aligned with `truth`.") from exc
        truth_num=pd.to_numeric(pd.Series(truth_arr),errors="coerce").to_numpy(dtype=float);keep=~np.isnan(truth_num)&~np.isnan(pred);score=np.abs(truth_num[keep]-pred[keep]);negative=None
    else:
        if probability is None or len(probability)!=n:raise GP3MLError("Classification conformal calibration requires `probability` aligned with `truth`.")
        prob=_assert_probability(probability);truth_chr=np.asarray([None if pd.isna(x) else str(x) for x in truth_arr],dtype="object");levels_found=sorted(set(x for x in truth_chr.tolist() if x is not None));positive_str=None if positive is None else str(positive)
        if len(levels_found)!=2 or positive_str not in levels_found:raise GP3MLError("Binary classification requires exactly two truth levels and explicit `positive`.")
        keep=pd.notna(truth_chr)&~np.isnan(prob);p_true=np.where(truth_chr[keep]==positive_str,prob[keep],1-prob[keep]);score=1-p_true;positive=positive_str;negative=next(x for x in levels_found if x!=positive_str)
    if calibration_unit!="observation":
        if unit is None or len(unit)!=n:raise GP3MLError("Grouped conformal calibration requires a complete `unit` identifier.")
        unit_arr=np.asarray(unit,dtype="object")[keep]
        if len(unit_arr)!=len(score) or pd.isna(unit_arr).any():raise GP3MLError("Grouped conformal calibration requires a complete `unit` identifier.")
        frame=pd.DataFrame({"unit":[str(x) for x in unit_arr],"score":score});score_for_quantile=frame.groupby("unit",sort=True).score.max().to_numpy(dtype=float)
    else:score_for_quantile=np.asarray(score,dtype=float)
    m=len(score_for_quantile)
    if m==0:raise GP3MLError("No finite conformity scores are available.")
    probability_index=min(1.0,np.ceil((m+1)*float(level))/m);q=_type1_quantile(score_for_quantile,probability_index);summary=pd.Series(np.quantile(score_for_quantile,[0,.25,.5,.75,1],method="linear"),index=["0%","25%","50%","75%","100%"])
    return GP3MLConformal(task_type=task_type,positive=positive,negative=negative,level=float(level),calibration_unit=calibration_unit,generalization_target="" if generalization_target is None else str(generalization_target),n_rows=len(score),n_calibration_units=m,quantile_probability=float(probability_index),conformity_quantile=q,score_summary=summary,caveat="Coverage claims require exchangeability assumptions appropriate to the declared calibration unit and generalization target.")

def validate_gazepoint_conformal(object:Any)->GP3MLConformalValidation:
    checks=pd.DataFrame({"check":["class","task_type","level","calibration_unit","quantile","target","unit_count"],"status":"pass"})
    if not isinstance(object,GP3MLConformal):checks["status"]="fail"
    else:
        if object.task_type not in {"classification","regression"}:checks.loc[checks.check=="task_type","status"]="fail"
        if not isinstance(object.level,(int,float,np.number)) or not 0<float(object.level)<1:checks.loc[checks.check=="level","status"]="fail"
        if object.calibration_unit not in {"observation","participant","stimulus","participant_stimulus"}:checks.loc[checks.check=="calibration_unit","status"]="fail"
        if not isinstance(object.conformity_quantile,(int,float,np.number)) or not np.isfinite(object.conformity_quantile):checks.loc[checks.check=="quantile","status"]="fail"
        if not isinstance(object.generalization_target,str) or object.generalization_target=="":checks.loc[checks.check=="target","status"]="fail"
        if not isinstance(object.n_calibration_units,(int,float,np.number)) or object.n_calibration_units<2:checks.loc[checks.check=="unit_count","status"]="review"
    return GP3MLConformalValidation(status=worst_status(checks.status),checks=checks)

def predict_gazepoint_interval(object:GP3MLConformal,prediction:Sequence[float])->pd.DataFrame:
    validation=validate_gazepoint_conformal(object)
    if validation.status!="pass" or object.task_type!="regression":raise GP3MLError("A valid regression conformal fit is required.")
    pred=np.asarray(prediction,dtype=float);return pd.DataFrame({"prediction":pred,"lower":pred-object.conformity_quantile,"upper":pred+object.conformity_quantile})

def predict_gazepoint_set(object:GP3MLConformal,probability:Sequence[float])->pd.DataFrame:
    validation=validate_gazepoint_conformal(object)
    if validation.status!="pass" or object.task_type!="classification":raise GP3MLError("A valid classification conformal fit is required.")
    prob=_assert_probability(probability);q=object.conformity_quantile;include_positive=(1-prob)<=q;include_negative=prob<=q;label=np.where(include_positive&include_negative,f"{{{object.negative}, {object.positive}}}",np.where(include_positive,f"{{{object.positive}}}",np.where(include_negative,f"{{{object.negative}}}","{}")));return pd.DataFrame({"probability":prob,"include_negative":include_negative,"include_positive":include_positive,"set":label})

def assess_gazepoint_conformal_coverage(object:GP3MLConformal,truth:Sequence[Any],interval:pd.DataFrame|None=None,set:pd.DataFrame|None=None,unit:Sequence[Any]|None=None)->GP3MLConformalCoverage:
    truth_arr=np.asarray(truth,dtype="object")
    if object.task_type=="regression":
        if interval is None or not {"lower","upper"}.issubset(interval.columns) or len(interval)!=len(truth_arr):raise GP3MLError("Supply regression `interval` rows aligned with `truth`.")
        truth_num=pd.to_numeric(pd.Series(truth_arr),errors="coerce").to_numpy(dtype=float);lower=pd.to_numeric(interval.lower,errors="coerce").to_numpy(dtype=float);upper=pd.to_numeric(interval.upper,errors="coerce").to_numpy(dtype=float);covered=~np.isnan(truth_num)&~np.isnan(lower)&~np.isnan(upper)&(truth_num>=lower)&(truth_num<=upper)
    else:
        if set is None or not {"include_negative","include_positive"}.issubset(set.columns) or len(set)!=len(truth_arr):raise GP3MLError("Supply classification `set` rows aligned with `truth`.")
        truth_chr=np.asarray([None if pd.isna(x) else str(x) for x in truth_arr],dtype="object");covered=np.where(truth_chr==object.positive,set.include_positive.to_numpy(dtype=bool),np.where(truth_chr==object.negative,set.include_negative.to_numpy(dtype=bool),False));covered[pd.isna(truth_chr)]=False
    row_coverage=float(np.mean(covered));by_unit=None;unit_coverage=np.nan
    if unit is not None:
        if len(unit)!=len(covered) or pd.isna(np.asarray(unit,dtype="object")).any():raise GP3MLError("`unit` must be complete and aligned.")
        frame=pd.DataFrame({"unit":[str(x) for x in unit],"covered":covered});all_covered=frame.groupby("unit",sort=True).covered.all();by_unit=pd.DataFrame({"unit":all_covered.index,"all_rows_covered":all_covered.to_numpy(dtype=bool)});unit_coverage=float(all_covered.mean())
    return GP3MLConformalCoverage(status="pass" if row_coverage+1e-12>=object.level else "review",nominal_coverage=object.level,row_coverage=row_coverage,unit_coverage=unit_coverage,calibration_unit=object.calibration_unit,generalization_target=object.generalization_target,by_unit=by_unit,caveat=object.caveat)
