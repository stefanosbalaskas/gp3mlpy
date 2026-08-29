# `test_gazepoint_model_portability`

**R reference:** gp3ml 0.3.0.

## Test model-artifact serialization and optional fresh-process prediction

Test model-artifact serialization and optional fresh-process prediction

## R reference usage

```r
test_gazepoint_model_portability(
  artifact,
  newdata = artifact$reference_data,
  tolerance = 1e-08,
  fresh_process = FALSE
)
```

The Python implementation is exported as `gp3mlpy.test_gazepoint_model_portability`. See the runtime docstring for Python-specific typing and semantic adaptations.
