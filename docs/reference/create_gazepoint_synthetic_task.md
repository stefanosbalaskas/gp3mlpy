# `create_gazepoint_synthetic_task`

**R reference:** gp3ml 0.3.0.

## Create one of the governed synthetic demonstration tasks

Create one of the governed synthetic demonstration tasks

## R reference usage

```r
create_gazepoint_synthetic_task(
  data,
  workflow = c("recording_quality", "assigned_condition", "observed_behavior",
    "observed_duration"),
  generalization_target = c("new_trials_known_participants", "new_participants",
    "new_stimuli", "new_participants_and_new_stimuli")
)
```

The Python implementation is exported as `gp3mlpy.create_gazepoint_synthetic_task`. See the runtime docstring for Python-specific typing and semantic adaptations.
