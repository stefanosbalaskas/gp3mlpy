# `evaluate_gazepoint_group_folds`

**R reference:** gp3ml 0.3.0.

## Evaluate a governed model specification across materialized grouped folds

Fits preprocessing and the requested model only on each fold's analysis partition, predicts only on the corresponding assessment partition, retains excluded rows, and records fold-level metrics, leakage audits, warnings, and failures. Row-level predictions are never relabelled as participant- or stimulus-level estimates.

## R reference usage

```r
evaluate_gazepoint_group_folds(
  folds,
  task,
  predictors = NULL,
  engine = NULL,
  preprocessor_args = list(),
  engine_args = list(),
  threshold = 0.5,
  seed = 1L,
  assess_calibration = FALSE,
  calibration_bins = 10L,
  calibration_bootstrap = 0L,
  keep_models = FALSE,
  continue_on_error = TRUE
)

\method{print}{gp3ml_resample_evaluation}(x, ...)
```

The Python implementation is exported as `gp3mlpy.evaluate_gazepoint_group_folds`. See the runtime docstring for Python-specific typing and semantic adaptations.
