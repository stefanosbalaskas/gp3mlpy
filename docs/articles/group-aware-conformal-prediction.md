# Target-Aware Conformal Prediction

> Source-derived companion to `gp3ml` 0.3.0 vignette `group-aware-conformal-prediction.Rmd`. R code blocks are omitted here; the Python companion script is under `examples/group-aware-conformal-prediction.py`.

Performance uncertainty and prediction uncertainty answer different questions.
This workflow calibrates split-conformal prediction to an explicit unit. Grouped
calibration uses the maximum row conformity score within each supplied unit,
which is conservative and records the calibration semantics.

It does **not** assert distribution-free guarantees under arbitrary dependence.



Do not describe observation-level coverage as new-participant coverage merely
because participant identifiers are present elsewhere in the study.
