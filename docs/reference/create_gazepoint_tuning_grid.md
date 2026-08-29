# `create_gazepoint_tuning_grid`

**R reference:** gp3ml 0.3.0.

## Create an explicit governed tuning grid

Candidate values are fully materialized before evaluation. No hidden metric, default ranking rule, or automatic winner is created.

## R reference usage

```r
create_gazepoint_tuning_grid(
  engine,
  engine_grid = list(),
  preprocessor_grid = list(),
  thresholds = 0.5,
  complexity = NA,
  interpretability = NA,
  labels = NULL
)

\method{print}{gp3ml_tuning_grid}(x, ...)
```

The Python implementation is exported as `gp3mlpy.create_gazepoint_tuning_grid`. See the runtime docstring for Python-specific typing and semantic adaptations.
