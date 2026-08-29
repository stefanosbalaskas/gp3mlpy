# `tune_gazepoint_model`

**R reference:** gp3ml 0.3.0.

## Evaluate every governed candidate on the same grouped folds

Evaluate every governed candidate on the same grouped folds

## R reference usage

```r
tune_gazepoint_model(
  folds,
  task,
  tuning_grid,
  predictors = NULL,
  metrics = NULL,
  seed = 1L,
  continue_on_error = TRUE,
  keep_evaluations = TRUE
)

\method{print}{gp3ml_model_tuning}(x, ...)
```

The Python implementation is exported as `gp3mlpy.tune_gazepoint_model`. See the runtime docstring for Python-specific typing and semantic adaptations.
