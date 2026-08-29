# `declare_gazepoint_analysis_plan`

**R reference:** gp3ml 0.3.0.

## Declare a frozen-analysis-plan contract

Declare a frozen-analysis-plan contract

## R reference usage

```r
declare_gazepoint_analysis_plan(
  research_question,
  scientific_purpose,
  outcome,
  outcome_definition,
  predictors,
  generalization_target,
  grouping_variables = character(),
  eligible_population,
  exclusion_rules = character(),
  preprocessing_plan,
  candidate_models,
  primary_metric,
  secondary_metrics = character(),
  calibration_metric = NULL,
  uncertainty_method,
  threshold_policy = NULL,
  external_validation_required = FALSE,
  seed_strategy,
  prohibited_interpretations = gp3ml_prohibited_uses()
)
```

The Python implementation is exported as `gp3mlpy.declare_gazepoint_analysis_plan`. See the runtime docstring for Python-specific typing and semantic adaptations.
