# Governance

gp3mlpy preserves the governance-first scientific contract of gp3ml 0.3.0.

Users must declare the scientific purpose, observed outcome, predictor roles, grouping identifiers, and intended generalization target. Predictive preprocessing is fitted inside the relevant analysis partition. Group overlap is audited against the declared target. Model tuning and threshold selection must not use the final assessment partition. External validation is explicitly distinguished from reused internal holdouts. Uncertainty procedures retain their declared sampling unit.

The package does not autonomously choose a winning model or silently alter a research design to make validation pass. Review and fail statuses are intended to surface unresolved scientific or reproducibility decisions rather than hide them.
