from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any
import pandas as pd
from ._utils import hash_jsonable
from .exceptions import GP3MLError
from .objects import GP3MLAnalysisPlan, GP3MLAnalysisPlanValidation, GP3MLPlanDeviationAudit
from .task_governance import gp3ml_prohibited_uses

def _unique_str(values:Sequence[Any]|str|None)->list[str]:
    if values is None:return []
    if isinstance(values,str):return [values]
    return list(dict.fromkeys(str(x) for x in values))

def declare_gazepoint_analysis_plan(research_question:Any,scientific_purpose:Any,outcome:Any,outcome_definition:Any,predictors:Sequence[str],generalization_target:Any,grouping_variables:Sequence[str]|str=(),eligible_population:Any=None,exclusion_rules:Sequence[str]|str=(),preprocessing_plan:Any=None,candidate_models:Any=None,primary_metric:Any=None,secondary_metrics:Sequence[str]|str=(),calibration_metric:Any=None,uncertainty_method:Any=None,threshold_policy:Any=None,external_validation_required:bool=False,seed_strategy:Any=None,prohibited_interpretations:Sequence[str]|str|None=None)->GP3MLAnalysisPlan:
    return GP3MLAnalysisPlan(research_question=research_question,scientific_purpose=scientific_purpose,outcome=outcome,outcome_definition=outcome_definition,predictors=_unique_str(predictors),generalization_target=generalization_target,grouping_variables=_unique_str(grouping_variables),eligible_population=eligible_population,exclusion_rules=_unique_str(exclusion_rules),preprocessing_plan=preprocessing_plan,candidate_models=candidate_models,primary_metric=primary_metric,secondary_metrics=_unique_str(secondary_metrics),calibration_metric=calibration_metric,uncertainty_method=uncertainty_method,threshold_policy=threshold_policy,external_validation_required=bool(external_validation_required),seed_strategy=seed_strategy,prohibited_interpretations=_unique_str(gp3ml_prohibited_uses() if prohibited_interpretations is None else prohibited_interpretations),locked=False,plan_id=None,plan_hash=None,locked_at=None)

def _empty(value:Any)->bool:
    if value is None:return True
    if isinstance(value,str):return value.strip()==""
    try:
        if len(value)==0:return True
        if isinstance(value,Sequence):
            chars=[str(x).strip() for x in value if x is not None];return bool(chars) and not any(chars)
    except TypeError:pass
    return False

def validate_gazepoint_analysis_plan(plan:Any)->GP3MLAnalysisPlanValidation:
    required=["research_question","scientific_purpose","outcome","outcome_definition","predictors","generalization_target","eligible_population","preprocessing_plan","candidate_models","primary_metric","uncertainty_method","seed_strategy","prohibited_interpretations"];checks=pd.DataFrame({"check":required,"status":"pass","detail":""})
    if not isinstance(plan,GP3MLAnalysisPlan):checks["status"]="fail"
    else:
        for name in required:
            if _empty(plan[name]):checks.loc[checks.check==name,"status"]="fail"
        if len(plan.predictors)!=len(set(plan.predictors)):checks.loc[checks.check=="predictors","status"]="review"
        if plan.outcome in plan.predictors:checks.loc[checks.check=="predictors","status"]="fail"
    status="fail" if (checks.status=="fail").any() else ("review" if (checks.status=="review").any() else "pass");return GP3MLAnalysisPlanValidation(status=status,checks=checks)

def lock_gazepoint_analysis_plan(plan:GP3MLAnalysisPlan,plan_id:str|None=None,locked_at:Any=None)->GP3MLAnalysisPlan:
    if validate_gazepoint_analysis_plan(plan).status!="pass":raise GP3MLError("Analysis plan must pass validation before locking.")
    if plan.locked:raise GP3MLError("Analysis plan is already locked.")
    base=plan.to_dict();[base.pop(k,None) for k in ["locked","plan_id","plan_hash","locked_at"]];digest=hash_jsonable(base,algorithm="sha256")
    if plan_id is None:plan_id=f"gp3ml-plan-{digest[:12]}"
    if locked_at is None:dt=datetime.now(timezone.utc)
    elif isinstance(locked_at,datetime):dt=locked_at.astimezone(timezone.utc)
    else:dt=datetime.fromisoformat(str(locked_at)).astimezone(timezone.utc)
    out=plan.to_dict();out.update(locked=True,plan_id=str(plan_id),plan_hash=digest,locked_at=dt.strftime("%Y-%m-%d %H:%M:%S UTC"));return GP3MLAnalysisPlan(**out)

def _structure_text(value:Any)->str:
    if value is None:return "NULL"
    if isinstance(value,str):return f' chr "{value}"'
    if isinstance(value,(list,tuple)):return f" chr [1:{len(value)}] "+" ".join(f'"{x}"' for x in value)
    if isinstance(value,dict):return str(value)
    return repr(value)

def audit_gazepoint_plan_deviations(plan:GP3MLAnalysisPlan,actual:Mapping[str,Any],fields:Sequence[str]=("outcome","predictors","generalization_target","primary_metric","secondary_metrics","calibration_metric","uncertainty_method","threshold_policy","candidate_models","preprocessing_plan"))->GP3MLPlanDeviationAudit:
    if not isinstance(plan,GP3MLAnalysisPlan) or not plan.locked:raise GP3MLError("A locked analysis plan is required.")
    if not isinstance(actual,Mapping):raise GP3MLError("`actual` must be a named list.")
    rows=[]
    for name in fields:
        planned=plan.to_dict().get(name);observed=actual.get(name);same=type(planned) is type(observed) and planned==observed;rows.append({"field":name,"status":"pass" if same else "deviation","planned":_structure_text(planned),"actual":_structure_text(observed)})
    tab=pd.DataFrame(rows);return GP3MLPlanDeviationAudit(status="review" if (tab.status=="deviation").any() else "pass",plan_id=plan.plan_id,plan_hash=plan.plan_hash,deviations=tab)

def write_gazepoint_analysis_plan(plan:GP3MLAnalysisPlan,path:str|Path,format:str="rds")->str:
    if format not in {"rds","json","md"}:raise GP3MLError("`format` must be one of: rds, json, md.")
    if validate_gazepoint_analysis_plan(plan).status=="fail":raise GP3MLError("Cannot write an invalid analysis plan.")
    target=Path(path).expanduser();target.parent.mkdir(parents=True,exist_ok=True)
    if format in {"rds","json"}:target.write_text(json.dumps(plan.to_dict(),indent=2,default=str,ensure_ascii=False)+"\n",encoding="utf-8")
    else:
        lines=["# gp3ml analysis plan","",f"- Plan ID: {plan.plan_id or '<unlocked>'}",f"- SHA-256: {plan.plan_hash or '<unlocked>'}",f"- Research question: {plan.research_question}",f"- Scientific purpose: {plan.scientific_purpose}",f"- Outcome: {plan.outcome}",f"- Generalization target: {plan.generalization_target}",f"- Primary metric: {plan.primary_metric}",f"- Predictors: {', '.join(plan.predictors)}",f"- External validation required: {plan.external_validation_required}","","## Prohibited interpretations",*[f"- {x}" for x in plan.prohibited_interpretations]];target.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return target.resolve().as_posix()
