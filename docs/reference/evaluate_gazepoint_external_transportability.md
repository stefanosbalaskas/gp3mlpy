# `evaluate_gazepoint_external_transportability`

**R reference:** gp3ml 0.3.0.

## Evaluate external transportability and validation status

An internal holdout or a dataset explicitly declared non-independent is labelled \code{not_externally_validated

## R reference usage

```r
evaluate_gazepoint_external_transportability(
  model,
  development_data,
  external_data = NULL,
  declaration = NULL,
  development_evaluation = NULL,
  threshold = model$threshold,
  bootstrap = 200L,
  seed = 1L
)

\method{print}{gp3ml_transportability_report}(x, ...)
```

The Python implementation is exported as `gp3mlpy.evaluate_gazepoint_external_transportability`. See the runtime docstring for Python-specific typing and semantic adaptations.
