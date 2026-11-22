# Modelling-4636910 reviewer-specific supplementary validation

**Status:** post-review, explicitly reproducible synthetic verification; not the unarchived execution ledger that produced the original manuscript figures. The archived release tags and publication artifacts are unchanged. This folder is source-level research evidence, not a new installed gp3mlpy API.

## Reproduction

The repository workflow `.github/workflows/reviewer-validation.yml` installs exactly `gp3ml==0.3.0` from its SHA-256-pinned archive and `gp3mlpy==0.1.0` from PyPI, then runs:

1. `Rscript research/reviewer_2026/fold_inventory.R research/reviewer_2026/output`
2. Existing paired `parity/run_r_leakage.R` and `parity/run_python_leakage.py` against `parity/fixtures/leakage.json`
3. Existing `parity/compare_leakage.py` comparison
4. `python research/reviewer_2026/verify.py research/reviewer_2026/output`

All workflow outputs, including complete row- and fold-level records, are uploaded as the `reviewer-validation-evidence` Actions artifact. Failed executions are not represented as passed. The audit prints its numeric result JSON to the workflow log.

## Evidence files and interpretation

- `synthetic_source.csv`: 30 participants × 8 stimuli × 2 trials = 480 rows, synthetic `quality_status` outcome; R simulator seed 1, exactly 24 gaze-dispersion rows set missing with missingness seed 951.
- `fold_assignments.csv`, `fold_inventory.csv`: new 5-fold participant or 5×4 crossed grouping, two repeats, group seed 20261122. 10 participant folds and 40 crossed folds; 150 partition-specific class/participant/stimulus summary rows; 24,000 row–fold assignments. All source rows enter assessment once per repeat; crossed mixed regions are excluded. These are new, explicitly seeded records, not original case-study fold assignments.
- `r-leakage.json`, `python-leakage.json`, `negative_controls.csv`: 13 paired cases; structural negative controls, clean cases, input errors, explicit status, check ID, diagnostic text, and stage. The matched comparison is to the frozen R implementation, not independent scientific correctness.
- `dependence_missingness_sensitivity.csv`: 72 independent synthetic sources (three participant fingerprint strengths 0, 0.35, 1; three MCAR levels 0, 0.10, 0.30; eight seeds/cell); 144 row- or participant-holdout 1-nearest-neighbour pipeline evaluations with training-local imputation and scaling. A scikit-learn pipeline does not automatically emit the `gp3mlpy` participant-compatibility audit. Analysts can implement an equivalent explicit group check in either ecosystem. The results isolate **participant-ID overlap detection**, not arbitrary leakage detection, and do not establish wider superiority.
- `isolated_audit_predictions.csv`, `model_execution_lineage.json`: a separate, fully isolated 18/4/4/4 participant fit/calibration/threshold/audit sequence using the same new R synthetic source. Records trial IDs, synthetic-source digest, preprocessing fitting cohort and parameters, model coefficients, calibration group and score cutoff, threshold grid/rule, and final audit membership. This proves recorded ancestry **within this instrumented example**, not for historical cross-validation predictions or unobserved upstream transformations.
- `reviewer_validation_summary.json`: programmatic assertions for all reviewed components and specification-derived numerical oracles.

## Numerical checks

Independent arithmetic provides balanced accuracy 0.750, Brier 0.175, log loss 0.5108256237659907 for `truth = (pass, pass, review, review)` and positive-class probabilities `(0.1, 0.7, 0.8, 0.6)` at threshold 0.5, with `review` positive. In an independent five-participant grouped conformal fixture, participant maximum nonconformity scores imply a 0.6 cutoff at nominal 0.8 coverage. Both frozen R and Python packages are checked against these specified expected values with absolute numerical tolerance 1e-12.

For the separate 30-participant synthetic source, the locked analysis sequence fits logistic regression on P001–P018 (288 rows), constructs conformal scores on P019–P022 (64 rows), optimizes balanced accuracy over the **declared** 0.10–0.90 grid in 0.05 increments on P023–P026 (64 rows; smallest-threshold tie rule), and evaluates once on P027–P030 (64 rows). The selected threshold is 0.15 and the illustrative abstention half-width is 0.05, yielding 33 decisions, 31 abstentions and 10 errors among decisions. The 90% participant-maximum conformal cutoff is approximately 0.95905 from four calibration participants, giving 0 empty, 3 singleton and 61 two-class sets, and full observed row and all-rows-participant coverage in this one tiny synthetic audit. This is low set informativeness, not general empirical validation.

## Remaining boundaries

- The original submission's exact seeded fold inventory and model-fitting ancestry were not archived in the supplied manuscript or identified in the frozen repositories. The new evidence does **not** retrospectively authenticate those numbers. The manuscript must explicitly replace, qualify or relocate them.
- User declarations of upstream feature roles or preprocessing history cannot be verified by inspecting a manifest alone. Only steps with recorded source IDs, fitted parameters and execution dependencies are described as executed/checked in this supplemental workflow.
- The new scenario is synthetic; it is not evidence of Gazepoint device accuracy, real participant effects, generalizable predictive accuracy or unrestricted exchangeability-based conformal guarantees.
- The existing 67/71 reference-based API result remains separately bounded; the four EXPECTED-DIFFERENCE contracts are documented in `PARITY_STATUS.md`.
