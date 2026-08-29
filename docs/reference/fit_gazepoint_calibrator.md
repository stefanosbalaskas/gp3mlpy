# `fit_gazepoint_calibrator`

**R reference:** gp3ml 0.3.0.

## Fit a probability calibrator

Fit a probability calibrator

## R reference usage

```r
fit_gazepoint_calibrator(
  truth,
  probability,
  positive = NULL,
  method = c("platt", "isotonic")
)
```

The Python implementation is exported as `gp3mlpy.fit_gazepoint_calibrator`. See the runtime docstring for Python-specific typing and semantic adaptations.
