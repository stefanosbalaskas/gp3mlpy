from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from ._utils import assert_columns, assert_data, clip_probability, hash_jsonable
from .exceptions import GP3MLError, OptionalDependencyError
from .objects import GP3MLEngine, GP3MLModel, GP3MLPreprocessor, GP3MLTask
from .preprocessing import bake_gazepoint_preprocessor, fit_gazepoint_preprocessor
from .task_governance import assert_gp3ml_use_case

def _default_predictors(data:pd.DataFrame,task:GP3MLTask)->list[str]:
    forbidden={task.outcome,task.unit_id,task.participant_id,task.stimulus_id};return [c for c in data.columns if c not in forbidden and c is not None]

def _training_summary(data:pd.DataFrame,predictors:Sequence[str])->dict[str,dict[str,Any]]:
    out={}
    for name in predictors:
        x=data[name]
        if pd.api.types.is_numeric_dtype(x.dtype) and not pd.api.types.is_bool_dtype(x.dtype):
            numeric=pd.to_numeric(x,errors="coerce");out[name]={"type":"numeric","mean":float(numeric.mean(skipna=True)),"sd":float(numeric.std(skipna=True,ddof=1)),"missing":int(numeric.isna().sum())}
        else:out[name]={"type":"categorical","levels":sorted({str(v) for v in x.dropna()}),"missing":int(x.isna().sum())}
    return out

def gp3ml_available_engines()->pd.DataFrame:
    import importlib.util
    return pd.DataFrame({"engine":["glm","lm","ranger","xgboost","nnet","keras3","custom"],"available":[True,True,True,importlib.util.find_spec("xgboost") is not None,True,importlib.util.find_spec("keras") is not None,True]})

def integrate_black_box_model(name:str,fit_fun:Callable[...,Any],predict_fun:Callable[...,Any],supports:Sequence[str] = ("classification","regression"),probability:bool=True,metadata:dict[str,Any]|None=None,safety_declaration:dict[str,bool]|None=None)->GP3MLEngine:
    if not callable(fit_fun) or not callable(predict_fun):raise GP3MLError("`fit_fun` and `predict_fun` must be functions.")
    required=("prohibited_uses_acknowledged","prediction_time_inputs_only","group_aware_evaluation_required")
    if not isinstance(safety_declaration,dict) or not all(safety_declaration.get(k) is True for k in required):raise GP3MLError("Black-box integration requires explicit TRUE safety declarations for prohibited uses, prediction-time inputs, and group-aware evaluation.")
    return GP3MLEngine(name=name,fit_fun=fit_fun,predict_fun=predict_fun,supports=list(supports),probability=bool(probability),metadata={} if metadata is None else dict(metadata),safety_declaration=dict(safety_declaration))

def _fit_statsmodels(x:np.ndarray,y:np.ndarray,engine_name:str):
    X=sm.add_constant(x,prepend=True,has_constant="add");return sm.GLM(y,X,family=sm.families.Binomial()).fit() if engine_name=="glm" else sm.OLS(y,X).fit()

def _fit_ranger_adapter(x:np.ndarray,y:np.ndarray,classification:bool,seed:int,engine_args:dict[str,Any]):
    args=dict(engine_args);aliases={"num.trees":"n_estimators","num_trees":"n_estimators","num.threads":"n_jobs","num_threads":"n_jobs","mtry":"max_features","min.node.size":"min_samples_leaf","min_node_size":"min_samples_leaf","max.depth":"max_depth","max_depth":"max_depth","replace":"bootstrap","sample.fraction":"max_samples","sample_fraction":"max_samples"};translated={aliases.get(k,k):v for k,v in args.items()}
    if translated.get("max_depth")==0:translated["max_depth"]=None
    if translated.get("bootstrap") is False:translated.pop("max_samples",None)
    translated.setdefault("n_estimators",500);translated.setdefault("random_state",int(seed));translated.setdefault("n_jobs",1);estimator=RandomForestClassifier(**translated) if classification else RandomForestRegressor(**translated);return estimator.fit(x,y)

def _fit_nnet_adapter(x:np.ndarray,y:np.ndarray,classification:bool,seed:int,engine_args:dict[str,Any]):
    args=dict(engine_args);size=int(args.pop("size",5));args.pop("linout",None);args.pop("trace",None);args.pop("MaxNWts",None)
    if "maxit" in args and "max_iter" not in args:args["max_iter"]=int(args.pop("maxit"))
    if "decay" in args and "alpha" not in args:args["alpha"]=float(args.pop("decay"))
    if "reltol" in args and "tol" not in args:args["tol"]=float(args.pop("reltol"))
    args.pop("abstol",None);args.setdefault("hidden_layer_sizes",(size,));args.setdefault("activation","logistic");args.setdefault("solver","lbfgs");args.setdefault("random_state",int(seed));args.setdefault("max_iter",1000);estimator=MLPClassifier(**args) if classification else MLPRegressor(**args);return estimator.fit(x,y)

def fit_gazepoint_model(data:pd.DataFrame,task:GP3MLTask,predictors:Sequence[str]|None=None,engine:str|GP3MLEngine|None=None,preprocessor:GP3MLPreprocessor|None=None,preprocessor_args:dict[str,Any]|None=None,engine_args:dict[str,Any]|None=None,seed:int=1,threshold:float=0.5)->GP3MLModel:
    """Fit a governed model using gp3ml 0.3.0's engine and preprocessing contract."""
    assert_data(data);assert_gp3ml_use_case(task,data);predictors=_default_predictors(data,task) if predictors is None else list(predictors);assert_columns(data,predictors,"predictors");forbidden_values=[v for v in (task.outcome,task.unit_id,task.participant_id,task.stimulus_id) if v];forbidden=[p for p in predictors if p in forbidden_values]
    if forbidden:raise GP3MLError(f"Outcome or identifiers cannot be predictors: {', '.join(forbidden)}.")
    custom_engine=isinstance(engine,GP3MLEngine)
    if engine is None:engine="glm" if task.task_type=="classification" else "lm"
    engine_name=engine.name if custom_engine else str(engine)
    if not custom_engine and engine_name not in {"glm","lm","ranger","xgboost","nnet"}:raise GP3MLError(f"Unknown engine `{engine_name}`. Use `fit_gazepoint_deep_model()` for keras3 or supply a gp3ml engine.")
    if task.task_type=="classification" and engine_name=="lm":raise GP3MLError("Use a classification engine.")
    if task.task_type=="regression" and engine_name=="glm":engine_name="lm"
    preprocessor_args={} if preprocessor_args is None else dict(preprocessor_args);engine_args={} if engine_args is None else dict(engine_args);preprocessor=preprocessor or fit_gazepoint_preprocessor(data,predictors,**preprocessor_args);x=bake_gazepoint_preprocessor(preprocessor,data)
    if x.shape[1]==0:raise GP3MLError("No usable model columns remain after preprocessing.")
    outcome=data[task.outcome]
    if task.task_type=="classification":
        vals=outcome.astype(object).to_numpy()
        if pd.isna(vals).any():raise GP3MLError("Training outcomes may not be missing.")
        y=np.asarray([1 if str(v)==task.positive else 0 for v in vals],dtype=int)
    else:
        y=pd.to_numeric(outcome,errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():raise GP3MLError("Training outcomes may not be missing.")
    engine_spec=None;backend=engine_name
    if custom_engine:
        assert isinstance(engine,GP3MLEngine)
        if task.task_type not in engine.supports:raise GP3MLError("Custom engine does not support this task type.")
        fit=engine.fit_fun(x=x,y=y,task=task,args=engine_args);engine_spec=engine;backend=f"custom:{engine.name}"
    elif engine_name in {"glm","lm"}:fit=_fit_statsmodels(x,y,engine_name);backend="statsmodels"
    elif engine_name=="ranger":fit=_fit_ranger_adapter(x,y,task.task_type=="classification",seed,engine_args);backend="sklearn_random_forest_semantic_adapter"
    elif engine_name=="xgboost":
        try:import xgboost as xgb
        except ImportError as exc:raise OptionalDependencyError("Install `xgboost` to use this engine.") from exc
        defaults={"n_estimators":int(engine_args.pop("nrounds",100)),"objective":"binary:logistic" if task.task_type=="classification" else "reg:squarederror","random_state":int(seed),"verbosity":0,"n_jobs":1};defaults.update(engine_args);estimator=xgb.XGBClassifier(**defaults) if task.task_type=="classification" else xgb.XGBRegressor(**defaults);fit=estimator.fit(x,y);backend="xgboost"
    else:fit=_fit_nnet_adapter(x,y,task.task_type=="classification",seed,engine_args);backend="sklearn_mlp_semantic_adapter"
    counts=outcome.value_counts(dropna=False,sort=False);training_subset=data.loc[:,[task.outcome,*predictors]].copy();return GP3MLModel(fit=fit,engine=engine_name,engine_spec=engine_spec,engine_args=engine_args,task=task,predictors=predictors,preprocessor=preprocessor,threshold=threshold,seed=seed,training_n=len(data),outcome_distribution={str(k):int(v) for k,v in counts.items()},predictor_summary=_training_summary(data,predictors),training_hash=hash_jsonable(training_subset,algorithm="md5"),call="fit_gazepoint_model",python_backend=backend)

def train_gazepoint_classifier(data:pd.DataFrame,task:GP3MLTask,predictors:Sequence[str]|None=None,engine:str|GP3MLEngine="glm",**kwargs:Any)->GP3MLModel:
    if not isinstance(task,GP3MLTask) or task.task_type!="classification":raise GP3MLError("`task` must be a binary classification task.")
    return fit_gazepoint_model(data,task,predictors,engine,**kwargs)

def predict_gazepoint_model(object:GP3MLModel,newdata:pd.DataFrame,type:str="response",**kwargs:Any)->np.ndarray|pd.Categorical:
    if type not in {"response","probability","class","link"}:raise GP3MLError("`type` must be one of: response, probability, class, link.")
    x=bake_gazepoint_preprocessor(object.preprocessor,newdata);task=object.task
    if object.engine_spec is not None:raw=np.asarray(object.engine_spec.predict_fun(fit=object.fit,newdata=x,type=type,task=task,**kwargs),dtype=float)
    elif object.engine=="glm":
        X=sm.add_constant(x,prepend=True,has_constant="add");raw=np.asarray(X@np.asarray(object.fit.params),dtype=float) if type=="link" else np.asarray(object.fit.predict(X),dtype=float)
    elif object.engine=="lm":X=sm.add_constant(x,prepend=True,has_constant="add");raw=np.asarray(object.fit.predict(X),dtype=float)
    elif object.engine=="ranger":raw=np.asarray(object.fit.predict_proba(x)[:,1],dtype=float) if task.task_type=="classification" else np.asarray(object.fit.predict(x),dtype=float)
    elif object.engine=="xgboost":raw=np.asarray(object.fit.predict_proba(x)[:,1],dtype=float) if task.task_type=="classification" and hasattr(object.fit,"predict_proba") else np.asarray(object.fit.predict(x),dtype=float)
    elif object.engine=="nnet":raw=np.asarray(object.fit.predict_proba(x)[:,1],dtype=float) if task.task_type=="classification" else np.asarray(object.fit.predict(x),dtype=float)
    elif object.engine=="keras3":raw=np.asarray(object.fit.predict(x,verbose=0),dtype=float).reshape(-1)
    else:raise GP3MLError(f"Unsupported fitted engine `{object.engine}`.")
    if task.task_type=="classification":
        if type=="class":return pd.Categorical(np.where(raw>=object.threshold,task.positive,task.negative),categories=list(task.levels),ordered=False)
        return clip_probability(raw)
    return raw

def _model_predict(self:GP3MLModel,newdata:pd.DataFrame,type:str="response",**kwargs:Any):return predict_gazepoint_model(self,newdata,type=type,**kwargs)
def _model_repr(self:GP3MLModel)->str:return f"<gp3ml_model> engine={self.engine} task={self.task.task_type} n={self.training_n} predictors={len(self.predictors)}"
setattr(GP3MLModel,"predict",_model_predict);setattr(GP3MLModel,"__repr__",_model_repr)
