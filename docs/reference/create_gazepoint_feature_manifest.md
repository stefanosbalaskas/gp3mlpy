# `create_gazepoint_feature_manifest`

**R reference:** gp3ml 0.3.0.

## Create a Gazepoint feature-provenance manifest

Creates a structured provenance manifest for intended predictive features. Each row records where a feature originated, when it became available, whether it is outcome-derived or post-outcome, and where any data-dependent preprocessing was estimated.

## R reference usage

```r
create_gazepoint_feature_manifest(
  features,
  scientific_source = NA_character_,
  source_table = NA_character_,
  transformation = "none",
  availability_stage = "unknown",
  prediction_time_available = NA,
  outcome_derived = FALSE,
  post_outcome = FALSE,
  identifier = FALSE,
  preprocessing_scope = "unknown",
  fold_local_required = NA,
  reviewer_notes = ""
)
```

The Python implementation is exported as `gp3mlpy.create_gazepoint_feature_manifest`. See the runtime docstring for Python-specific typing and semantic adaptations.
