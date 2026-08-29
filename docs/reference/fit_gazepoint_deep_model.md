# `fit_gazepoint_deep_model`

**R reference:** gp3ml 0.3.0.

## Fit an optional governed deep-learning model through keras3

Fit an optional governed deep-learning model through keras3

## R reference usage

```r
fit_gazepoint_deep_model(
  data,
  task,
  predictors = NULL,
  preprocessor = NULL,
  hidden_units = c(64L, 32L),
  dropout = 0.2,
  epochs = 50L,
  batch_size = 32L,
  validation_split = 0.2,
  optimizer = "adam",
  seed = 1L,
  verbose = 0L
)
```

The Python implementation is exported as `gp3mlpy.fit_gazepoint_deep_model`. See the runtime docstring for Python-specific typing and semantic adaptations.
