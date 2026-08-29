# `integrate_black_box_model`

**R reference:** gp3ml 0.3.0.

## Integrate a controlled black-box model engine

Integrate a controlled black-box model engine

## R reference usage

```r
integrate_black_box_model(
  name,
  fit_fun,
  predict_fun,
  supports = c("classification", "regression"),
  probability = TRUE,
  metadata = list(),
  safety_declaration
)
```

The Python implementation is exported as `gp3mlpy.integrate_black_box_model`. See the runtime docstring for Python-specific typing and semantic adaptations.
