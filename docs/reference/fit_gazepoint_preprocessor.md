# `fit_gazepoint_preprocessor`

**R reference:** gp3ml 0.3.0.

## Fit a fold-local preprocessing engine

Fit a fold-local preprocessing engine

## R reference usage

```r
fit_gazepoint_preprocessor(
  data,
  predictors,
  numeric_imputation = c("median", "mean"),
  center = TRUE,
  scale = TRUE,
  novel_level = c("other", "error"),
  remove_zero_variance = TRUE
)
```

The Python implementation is exported as `gp3mlpy.fit_gazepoint_preprocessor`. See the runtime docstring for Python-specific typing and semantic adaptations.
