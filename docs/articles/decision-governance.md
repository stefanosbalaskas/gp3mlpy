# Governed Decision Thresholds and Abstention

> Source-derived companion to `gp3ml` 0.3.0 vignette `decision-governance.Rmd`. R code blocks are omitted here; the Python companion script is under `examples/decision-governance.py`.

This article separates probability estimation from the scientific decision rule.
gp3ml does not treat 0.5 as a universally justified decision threshold. Thresholds
must be predeclared or selected using analysis/inner-resampling data, never an
outer assessment or independent external-validation set.



An abstention interval can be declared when the scientific protocol permits
withholding a forced classification. Abstentions must be reported explicitly,
including coverage and error among non-abstained predictions.
