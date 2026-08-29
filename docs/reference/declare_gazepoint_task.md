# `declare_gazepoint_task`

**R reference:** gp3ml 0.3.0.

## Declare a governed Gazepoint prediction task

Declare a governed Gazepoint prediction task

## R reference usage

```r
declare_gazepoint_task(
  data,
  outcome,
  purpose,
  task_type = c("classification", "regression"),
  unit_id,
  participant_id = NULL,
  stimulus_id = NULL,
  generalization_target = c("new_trials_known_participants", "new_participants",
    "new_stimuli", "new_participants_and_new_stimuli", "external_validation"),
  positive = NULL,
  observed_outcome = TRUE,
  sensitive_outcome = FALSE
)
```

The Python implementation is exported as `gp3mlpy.declare_gazepoint_task`. See the runtime docstring for Python-specific typing and semantic adaptations.
