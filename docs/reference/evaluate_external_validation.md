# `evaluate_external_validation`

**R reference:** gp3ml 0.3.0.

## Evaluate an independent external-validation dataset

Evaluate an independent external-validation dataset

## R reference usage

```r
evaluate_external_validation(
  model,
  external_data,
  label = "external",
  threshold = model$threshold,
  bootstrap = 200L,
  seed = 1L
)
```

The Python implementation is exported as `gp3mlpy.evaluate_external_validation`. See the runtime docstring for Python-specific typing and semantic adaptations.
