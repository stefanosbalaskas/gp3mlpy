# `fit_gazepoint_conformal`

**R reference:** gp3ml 0.3.0.

## Fit target-aware split-conformal calibration

This function provides conservative split-conformal calibration for explicitly observed regression or binary classification outcomes. When a grouped calibration unit is supplied, row scores are aggregated to the maximum score within each calibration unit before the conformal quantile is estimated. This records and respects the calibration unit but does not claim distribution-free coverage under arbitrary dependence.

## R reference usage

```r
fit_gazepoint_conformal(
  truth,
  prediction = NULL,
  probability = NULL,
  task_type = c("regression", "classification"),
  positive = NULL,
  level = 0.9,
  calibration_unit = c("observation", "participant", "stimulus", "participant_stimulus"),
  unit = NULL,
  generalization_target
)
```

The Python implementation is exported as `gp3mlpy.fit_gazepoint_conformal`. See the runtime docstring for Python-specific typing and semantic adaptations.
