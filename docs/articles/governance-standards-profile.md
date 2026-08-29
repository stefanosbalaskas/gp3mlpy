# Governance Evidence and Standards Crosswalks

> Source-derived companion to `gp3ml` 0.3.0 vignette `governance-standards-profile.Rmd`. The runnable Python companion is under `examples/governance-standards-profile.py`.

`gp3mlpy` can organize package evidence into a native governance profile or an orientation crosswalk for NIST AI RMF 1.0, ISO/IEC 23894, or ISO/IEC 42001.

!!! warning "Crosswalk, not certification"
    These are documentation crosswalks only. They are **not** evidence of NIST endorsement, ISO conformity, certification, or legal compliance.

## Evidence-oriented profiles

A profile is built from evidence objects already produced by the workflow. Missing evidence is surfaced for review rather than converted into a synthetic compliance score.

```python
import gp3mlpy as gp

profile = gp.create_gp3ml_governance_profile(
    {
        "task": task,
        "feature_manifest": manifest,
        "folds": folds,
        "performance": evaluation,
        "decision_rule": decision_rule,
        "research_artifact": bundle,
    },
    framework="gp3ml-native",
)

audit = gp.audit_gp3ml_governance_profile(profile)
print(audit.controls[["control", "status", "evidence_class"]])
```

<div class="gp-plot-card">
  <img src="../assets/plots/governance-profile.svg" alt="Counts of governance controls with available evidence and review status">
  <div><strong>Governance evidence.</strong> The plot summarizes control statuses while the audit table retains the individual evidence mapping.</div>
</div>

```python
fig = audit.plot()
```

## What the profile covers

The native profile organizes evidence across controls for:

1. scientific purpose;
2. intended and prohibited use;
3. data and feature provenance;
4. generalization target;
5. leakage-resistant validation;
6. performance and calibration;
7. prediction-level uncertainty;
8. external validation and shift;
9. human oversight and decision rules; and
10. reproducibility and artifact provenance.

For external frameworks, those controls are mapped to orientation domains while retaining an explicit disclaimer that the crosswalk is documentation support only.

## Why missing evidence is not scored away

A single composite compliance score would hide which evidence is present and which remains unresolved. The audit therefore retains control-level `pass`/`review`/`fail` statuses and evidence classes so gaps can be inspected directly.

## Key functions

| Function | Role |
|---|---|
| [`create_gp3ml_governance_profile`](../reference/create_gp3ml_governance_profile.md) | Assemble an evidence profile under a named framework. |
| [`audit_gp3ml_governance_profile`](../reference/audit_gp3ml_governance_profile.md) | Audit control-level evidence availability. |
| [`write_gp3ml_governance_profile`](../reference/write_gp3ml_governance_profile.md) | Write the resulting profile to an inspectable artifact. |

Continue with [Analysis-plan governance](analysis-plan-governance.md), [Reproducibility hardening](reproducibility-hardening.md), or the [plot gallery](../plots.md).
