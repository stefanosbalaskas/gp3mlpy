# Contributing

Changes to gp3mlpy must preserve the frozen gp3ml 0.3.0 compatibility contract unless the change is explicitly classified as a Python-native experimental extension.

Before proposing a change, run the full test suite, API freeze checks, runnable examples, and package build. Stable exports and stable public object classes may not be removed or renamed within the compatibility line. Existing argument meanings and named components must remain compatible; additions require defaults or additive object fields.

Do not add functionality that violates `PROHIBITED-USE.md`, silently weakens leakage safeguards, or performs autonomous model selection.
