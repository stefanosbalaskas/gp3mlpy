# `audit_gazepoint_missingness_shift`

**R reference:** gp3ml 0.3.0.

## Audit missingness shift

Audit missingness shift

## R reference usage

```r
audit_gazepoint_missingness_shift(
  development,
  external,
  predictors = intersect(names(development), names(external)),
  review_delta = 0.1,
  fail_delta = 0.25
)
```

The Python implementation is exported as `gp3mlpy.audit_gazepoint_missingness_shift`. See the runtime docstring for Python-specific typing and semantic adaptations.
