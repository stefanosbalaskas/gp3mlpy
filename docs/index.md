<div class="gp-home-hero">
  <div class="gp-home-copy">
    <div class="gp-release-line">
      <span>gp3mlpy 0.1.0</span>
      <span>Python 3.11+</span>
      <span>gp3ml 0.3.0 parity target</span>
    </div>
    <p class="gp-eyebrow">Governance-first predictive modelling</p>
    <h1>Build models whose validation logic matches the scientific claim.</h1>
    <p class="gp-lead">Leakage-resistant, group-aware modelling for Gazepoint research workflows in Python — with explicit generalization targets, auditable predictors, governed decisions, external validation, and reproducible release evidence.</p>
    <div class="gp-home-actions">
      <a class="gp-button gp-button-primary" href="getting-started/">Get started</a>
      <a class="gp-button" href="articles/">Explore workflows</a>
      <a class="gp-button" href="api-map/">Find an API</a>
    </div>
    <div class="gp-proof-row">
      <span><strong>100%</strong> statements</span>
      <span><strong>100%</strong> branches</span>
      <span><strong>125</strong> tests</span>
      <span><strong>0</strong> parity failures</span>
    </div>
  </div>
  <div class="gp-terminal" aria-label="Install gp3mlpy from PyPI">
    <div class="gp-terminal-bar"><span></span><span></span><span></span><small>terminal</small></div>
    <div class="gp-terminal-body">
      <div><span class="gp-prompt">$</span> python -m pip install gp3mlpy</div>
      <div class="gp-terminal-muted">Successfully installed gp3mlpy-0.1.0</div>
      <div>&nbsp;</div>
      <div><span class="gp-prompt">›</span> import gp3mlpy as gp</div>
      <div><span class="gp-prompt">›</span> gp.__version__</div>
      <div class="gp-terminal-output">'0.1.0'</div>
      <div><span class="gp-prompt">›</span> gp.r_reference_version</div>
      <div class="gp-terminal-output">'0.3.0'</div>
    </div>
    <div class="gp-terminal-footer">
      <a href="https://pypi.org/project/gp3mlpy/">PyPI</a>
      <a href="https://doi.org/10.5281/zenodo.22206729">DOI 10.5281/zenodo.22206729</a>
    </div>
  </div>
</div>

## Predictive modelling with explicit scientific contracts

`gp3mlpy` is the Python port of **gp3ml 0.3.0**. Instead of treating validation as a generic train/test operation, it makes the target of generalization, feature provenance, fitted operations, uncertainty, and reporting evidence visible parts of the analysis.

<div class="gp-feature-grid" markdown>
<div class="gp-feature-card" markdown>
<span class="gp-number">01</span>
### Declare what must generalize
Choose new participants, new stimuli, or both from the scientific claim. Splits that violate that target are rejected.
</div>
<div class="gp-feature-card" markdown>
<span class="gp-number">02</span>
### Keep fitted operations local
Preprocessing, tuning, calibration, thresholds, and related operations remain inside the correct analysis or fold structure.
</div>
<div class="gp-feature-card" markdown>
<span class="gp-number">03</span>
### Make predictors auditable
Feature manifests record source, role, and permission before model performance is known.
</div>
<div class="gp-feature-card" markdown>
<span class="gp-number">04</span>
### Leave an evidence trail
Diagnostics, model cards, external-validation reports, checksums, environment capture, handoffs, and release evidence remain inspectable.
</div>
</div>

## The governed lifecycle

<div class="gp-lifecycle">
  <div class="gp-life-step"><span>01</span><strong>Declare</strong><small>outcome · unit · use · target</small></div>
  <div class="gp-life-step"><span>02</span><strong>Audit</strong><small>provenance · roles · leakage</small></div>
  <div class="gp-life-step"><span>03</span><strong>Split</strong><small>group-aware holdout or resampling</small></div>
  <div class="gp-life-step"><span>04</span><strong>Fit</strong><small>fold-local preprocessing and tuning</small></div>
  <div class="gp-life-step"><span>05</span><strong>Evaluate</strong><small>performance · uncertainty · robustness</small></div>
  <div class="gp-life-step"><span>06</span><strong>Report</strong><small>validation · provenance · release evidence</small></div>
</div>

<div class="gp-inline-cta" markdown>
**New to the package?** The [Quickstart](getting-started.md) takes you from installation to a validated grouped workflow, then points to the next layer only when you need it.
</div>

## Choose the route that matches your study

<div class="gp-route-grid" markdown>
<div class="gp-route-card" markdown>
<span class="gp-route-tag">Generalization</span>
### New participants
Keep assessment participants completely unseen during fitting.

[Participant workflow →](articles/participant-generalization.md)
</div>
<div class="gp-route-card" markdown>
<span class="gp-route-tag">Model selection</span>
### Nested grouped resampling
Separate inner tuning from outer assessment while retaining grouping.

[Nested workflow →](articles/nested-grouped-resampling.md)
</div>
<div class="gp-route-card" markdown>
<span class="gp-route-tag">Transportability</span>
### External validation
Declare an external dataset, quantify shift, and report transportability explicitly.

[External validation →](articles/external-validation-reporting.md)
</div>
<div class="gp-route-card" markdown>
<span class="gp-route-tag">Decision layer</span>
### Governed thresholds
Keep threshold origin, error costs, calibration source, and abstention visible.

[Decision governance →](articles/decision-governance.md)
</div>
</div>

## A validated release, not just a feature list

<div class="gp-evidence-layout" markdown>
<div class="gp-evidence-panel" markdown>
<span class="gp-kicker">Python quality floor</span>
### 100% statement and branch coverage

- **4,020 / 4,020** executable statements
- **1,700 / 1,700** measured branches
- **0** partial branches
- **125** passing Python tests
- Ubuntu, Windows, macOS × Python 3.11, 3.12, 3.13
- Ruff, mypy, strict MkDocs, build/Twine, installed-wheel API checks

CI permanently enforces `--cov-branch --cov-fail-under=100`.
</div>
<div class="gp-evidence-panel gp-evidence-accent" markdown>
<span class="gp-kicker">Frozen stable API</span>
### 67 PASS · 4 expected differences · 0 FAIL

All **71 stable exports** were exercised against the SHA-256-verified `gp3ml 0.3.0` release archive.

The four expected differences preserve safer or functioning Python behavior instead of reproducing frozen-R recycling or reference defects.

[Read the parity status →](https://github.com/stefanosbalaskas/gp3mlpy/blob/main/PARITY_STATUS.md)
</div>
</div>

## Diagnostics are part of the evidence

The documentation build regenerates synthetic plot fixtures from the current package. The plots visualize declared analysis objects; plotting does not perform hidden model selection.

<div class="gp-plot-grid gp-home-plots">
<div class="gp-plot-card">
<img src="assets/plots/threshold-evaluation.svg" alt="Decision-threshold evaluation plot">
<div><strong>Decision thresholds</strong><br><span>Inspect declared candidate thresholds and decision consequences.</span></div>
</div>
<div class="gp-plot-card">
<img src="assets/plots/dataset-shift.svg" alt="Dataset-shift audit plot">
<div><strong>Dataset shift</strong><br><span>Separate predictor shift from calibration and performance change.</span></div>
</div>
<div class="gp-plot-card">
<img src="assets/plots/governance-profile.svg" alt="Governance profile audit plot">
<div><strong>Governance evidence</strong><br><span>Surface controls with evidence and controls that still need review.</span></div>
</div>
</div>

<div class="gp-center-link"><a href="plots/">Open the complete plot gallery →</a></div>

## Where to go next

<div class="gp-next-grid" markdown>
<div markdown>
<span class="gp-kicker">Learn</span>
### [Key concepts](key-concepts.md)
Understand generalization targets, provenance, leakage resistance, parity, and the scientific boundary.
</div>
<div markdown>
<span class="gp-kicker">Apply</span>
### [Workflow articles](articles/index.md)
Follow end-to-end, generalization, governance, validation, and reproducibility workflows.
</div>
<div markdown>
<span class="gp-kicker">Inspect</span>
### [Workflow API map](api-map.md)
Find functions by research stage rather than alphabetically.
</div>
<div markdown>
<span class="gp-kicker">Reference</span>
### [Complete API index](reference/index.md)
Browse all 127 compatibility exports and their dedicated reference pages.
</div>
</div>

!!! danger "Hard scientific boundary"
    `gp3mlpy` is for explicitly observed, non-sensitive outcomes. It must not be used for person identification, biometric authentication, health/diagnosis inference, protected-attribute inference, or direct/indirect inference of emotion, stress, personality, deception, cognition, comprehension, intent, or other mental states.
