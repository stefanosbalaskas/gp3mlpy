# `audit_gazepoint_ml_leakage`

**R reference:** gp3ml 0.3.0.

## Audit leakage between predictive-analysis partitions

Audits already-defined analysis and assessment partitions for common forms of leakage and for incompatibility with a declared generalization target. The function does not create data splits, preprocess variables, select features, or fit predictive models.

## R reference usage

```r
audit_gazepoint_ml_leakage(
  analysis,
  assessment,
  outcome,
  predictors,
  participant_id = NULL,
  trial_id = NULL,
  stimulus_id = NULL,
  generalization_target = c("new_trials_known_participants", "new_participants",
    "new_stimuli", "new_participants_and_new_stimuli"),
  target_derived = character(),
  post_outcome = character()
)
```

The Python implementation is exported as `gp3mlpy.audit_gazepoint_ml_leakage`. See the runtime docstring for Python-specific typing and semantic adaptations.
