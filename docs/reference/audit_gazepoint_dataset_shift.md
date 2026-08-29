# `audit_gazepoint_dataset_shift`

**R reference:** gp3ml 0.3.0.

## Audit predictor distribution shift

Audit predictor distribution shift

## R reference usage

```r
audit_gazepoint_dataset_shift(
  development,
  external,
  predictors = intersect(names(development), names(external)),
  thresholds = list(smd_review = 0.2, smd_fail = 0.5, outside_review = 0.05, outside_fail
    = 0.2, tv_review = 0.2, tv_fail = 0.4)
)
```

The Python implementation is exported as `gp3mlpy.audit_gazepoint_dataset_shift`. See the runtime docstring for Python-specific typing and semantic adaptations.
