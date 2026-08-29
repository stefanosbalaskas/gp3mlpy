# `audit_gazepoint_plan_deviations`

**R reference:** gp3ml 0.3.0.

## Audit deviations from a locked analysis plan

Audit deviations from a locked analysis plan

## R reference usage

```r
audit_gazepoint_plan_deviations(
  plan,
  actual,
  fields = c("outcome", "predictors", "generalization_target", "primary_metric",
    "secondary_metrics", "calibration_metric", "uncertainty_method", "threshold_policy",
    "candidate_models", "preprocessing_plan")
)
```

The Python implementation is exported as `gp3mlpy.audit_gazepoint_plan_deviations`. See the runtime docstring for Python-specific typing and semantic adaptations.
