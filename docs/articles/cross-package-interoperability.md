# Cross-Package Interoperability Contracts

> Source-derived companion to `gp3ml` 0.3.0 vignette `cross-package-interoperability.Rmd`. R code blocks are omitted here; the Python companion script is under `examples/cross-package-interoperability.py`.

## Boundary, not duplication

The interoperability layer does not import raw Gazepoint files, process
biometric signals, or perform sequence analysis. Those remain upstream
responsibilities.



## Prepared handoffs



## Combine only after validation



The resulting table is a modelling handoff. It does not imply that `gp3ml`
performed the upstream preprocessing represented by those columns.
