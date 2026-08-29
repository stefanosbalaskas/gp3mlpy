# `select_gazepoint_model`

**R reference:** gp3ml 0.3.0.

## Select a governed candidate using an explicit metric and direction

This function records a reviewable decision. It does not refit a model and refuses accuracy as the sole primary metric.

## R reference usage

```r
select_gazepoint_model(
  x,
  metric,
  direction,
  minimum_success_prop = 0.8,
  tie_breakers = NULL,
  rationale
)

\method{print}{gp3ml_model_selection}(x, ...)
```

The Python implementation is exported as `gp3mlpy.select_gazepoint_model`. See the runtime docstring for Python-specific typing and semantic adaptations.
