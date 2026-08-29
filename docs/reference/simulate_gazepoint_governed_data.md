# `simulate_gazepoint_governed_data`

**R reference:** gp3ml 0.3.0.

## Simulate governed synthetic Gazepoint-derived data

Creates deterministic, non-sensitive synthetic data for package examples, tests, and website articles. The generated outcomes are explicitly observed: a predefined recording-quality review status, an experimentally assigned condition, and a non-sensitive recorded response.

## R reference usage

```r
simulate_gazepoint_governed_data(
  n_participants = 30L,
  n_stimuli = 8L,
  trials_per_cell = 2L,
  seed = 1L
)
```

The Python implementation is exported as `gp3mlpy.simulate_gazepoint_governed_data`. See the runtime docstring for Python-specific typing and semantic adaptations.
