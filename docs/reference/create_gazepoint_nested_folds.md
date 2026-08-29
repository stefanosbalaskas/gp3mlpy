# `create_gazepoint_nested_folds`

**R reference:** gp3ml 0.3.0.

## Create nested grouped resampling from mature outer folds

Inner folds are constructed only from each outer analysis partition and preserve the declared participant/stimulus generalization target. The outer assessment partition is never used for inner preprocessing or tuning.

## R reference usage

```r
create_gazepoint_nested_folds(
  outer_folds,
  inner_v = 3L,
  inner_repeats = 1L,
  seed = 1L,
  continue_on_error = FALSE
)

\method{print}{gp3ml_nested_folds}(x, ...)
```

The Python implementation is exported as `gp3mlpy.create_gazepoint_nested_folds`. See the runtime docstring for Python-specific typing and semantic adaptations.
