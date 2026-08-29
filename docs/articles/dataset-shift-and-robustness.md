# Dataset Shift and Robustness Auditing

> Source-derived companion to `gp3ml` 0.3.0 vignette `dataset-shift-and-robustness.Rmd`. R code blocks are omitted here; the Python companion script is under `examples/dataset-shift-and-robustness.py`.

Dataset shift is not one scalar drift score. gp3ml keeps predictor-distribution
shift, missingness shift, prevalence shift, calibration drift, and performance
degradation conceptually separate.



Robustness diagnostics should examine dependence on seeds, folds, features,
thresholds, missingness scenarios, and other declared analytical choices rather
than relabelling one successful analysis as robust.
