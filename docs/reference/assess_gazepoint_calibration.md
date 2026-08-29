# `assess_gazepoint_calibration`

**R reference:** gp3ml 0.3.0.

## Calibration assessment with bootstrap uncertainty

Calibration assessment with bootstrap uncertainty

## R reference usage

```r
assess_gazepoint_calibration(
  truth,
  probability,
  positive = NULL,
  bins = 10L,
  bootstrap = 200L,
  conf_level = 0.95,
  seed = 1L
)
```

The Python implementation is exported as `gp3mlpy.assess_gazepoint_calibration`. See the runtime docstring for Python-specific typing and semantic adaptations.
