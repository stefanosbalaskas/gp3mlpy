# `audit_gazepoint_model_robustness`

**R reference:** gp3ml 0.3.0.

## Audit multiple robustness dimensions

Audit multiple robustness dimensions

## R reference usage

```r
audit_gazepoint_model_robustness(
  seed_stability = NULL,
  feature_stability = NULL,
  threshold_stability = NULL,
  missingness_stability = NULL,
  relative_sd_review = 0.05,
  relative_sd_fail = 0.15
)
```

The Python implementation is exported as `gp3mlpy.audit_gazepoint_model_robustness`. See the runtime docstring for Python-specific typing and semantic adaptations.
