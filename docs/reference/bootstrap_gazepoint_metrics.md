# `bootstrap_gazepoint_metrics`

**R reference:** gp3ml 0.3.0.

## Bootstrap uncertainty intervals for performance metrics

Bootstrap uncertainty intervals for performance metrics

## R reference usage

```r
bootstrap_gazepoint_metrics(
  task,
  truth,
  prediction = NULL,
  probability = NULL,
  threshold = 0.5,
  bootstrap = 1000L,
  conf_level = 0.95,
  seed = 1L
)
```

The Python implementation is exported as `gp3mlpy.bootstrap_gazepoint_metrics`. See the runtime docstring for Python-specific typing and semantic adaptations.
