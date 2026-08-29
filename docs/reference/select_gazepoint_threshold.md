# `select_gazepoint_threshold`

**R reference:** gp3ml 0.3.0.

## Select a threshold from a governed threshold evaluation

Select a threshold from a governed threshold evaluation

## R reference usage

```r
select_gazepoint_threshold(
  evaluation,
  metric,
  direction = c("maximize", "minimize"),
  threshold_origin = c("inner_resampling", "training"),
  training_partition = "inner_resampling",
  generalization_target,
  scientific_justification,
  abstention_allowed = FALSE,
  abstention_interval = NULL
)
```

The Python implementation is exported as `gp3mlpy.select_gazepoint_threshold`. See the runtime docstring for Python-specific typing and semantic adaptations.
