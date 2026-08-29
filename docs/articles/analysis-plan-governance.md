# Frozen Analysis-Plan Governance

> Source-derived companion to `gp3ml` 0.3.0 vignette `analysis-plan-governance.Rmd`. R code blocks are omitted here; the Python companion script is under `examples/analysis-plan-governance.py`.

The analysis-plan contract records the scientific design before model selection.
Locking requires the optional `openssl` package and creates a SHA-256 identifier.



Any later change to the outcome, predictor set, generalization target, metric,
preprocessing, or threshold policy should be represented as a deviation rather
than silently rewriting the original plan.
