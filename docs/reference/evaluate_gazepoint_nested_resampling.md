# `evaluate_gazepoint_nested_resampling`

**R reference:** gp3ml 0.3.0.

## Evaluate nested grouped resampling with inner governed tuning

Evaluate nested grouped resampling with inner governed tuning

## R reference usage

```r
evaluate_gazepoint_nested_resampling(
  nested_folds,
  task,
  tuning_grid,
  selection_metric,
  direction,
  predictors = NULL,
  minimum_success_prop = 0.8,
  tie_breakers = NULL,
  selection_rationale = .gp3ml_nested_selection_rationale_default,
  seed = 1L,
  keep_models = FALSE,
  continue_on_error = TRUE
)

\method{print}{gp3ml_nested_evaluation}(x, ...)
```

The Python implementation is exported as `gp3mlpy.evaluate_gazepoint_nested_resampling`. See the runtime docstring for Python-specific typing and semantic adaptations.
