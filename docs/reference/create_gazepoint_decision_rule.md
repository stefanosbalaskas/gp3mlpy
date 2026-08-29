# `create_gazepoint_decision_rule`

**R reference:** gp3ml 0.3.0.

## Create a governed classification decision rule

Create a governed classification decision rule

## R reference usage

```r
create_gazepoint_decision_rule(
  metric,
  direction = c("maximize", "minimize"),
  threshold = NULL,
  threshold_origin = c("predeclared", "training", "inner_resampling"),
  cost_false_positive = 1,
  cost_false_negative = 1,
  abstention_allowed = FALSE,
  abstention_interval = NULL,
  calibration_source = "none",
  training_partition = "analysis",
  generalization_target,
  scientific_justification
)
```

The Python implementation is exported as `gp3mlpy.create_gazepoint_decision_rule`. See the runtime docstring for Python-specific typing and semantic adaptations.
