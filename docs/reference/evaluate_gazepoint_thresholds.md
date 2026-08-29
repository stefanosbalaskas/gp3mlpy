# `evaluate_gazepoint_thresholds`

**R reference:** gp3ml 0.3.0.

## Evaluate explicit classification thresholds

Evaluate explicit classification thresholds

## R reference usage

```r
evaluate_gazepoint_thresholds(
  truth,
  probability,
  positive,
  thresholds,
  cost_false_positive = 1,
  cost_false_negative = 1
)
```

The Python implementation is exported as `gp3mlpy.evaluate_gazepoint_thresholds`. See the runtime docstring for Python-specific typing and semantic adaptations.
