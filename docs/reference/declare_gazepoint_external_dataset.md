# `declare_gazepoint_external_dataset`

**R reference:** gp3ml 0.3.0.

## Declare an external dataset and its independence status

Declare an external dataset and its independence status

## R reference usage

```r
declare_gazepoint_external_dataset(
  data,
  label,
  independent,
  origin,
  collection_period = NULL,
  participant_id = "participant_id",
  stimulus_id = "stimulus_id",
  notes = character()
)

\method{print}{gp3ml_external_dataset_declaration}(x, ...)
```

The Python implementation is exported as `gp3mlpy.declare_gazepoint_external_dataset`. See the runtime docstring for Python-specific typing and semantic adaptations.
