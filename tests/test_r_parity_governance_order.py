from __future__ import annotations

import pandas as pd

import gp3mlpy


def test_role_validation_preserves_r_factor_and_predictor_order() -> None:
    data = pd.DataFrame(
        {
            "participant_id": [f"P{index:02d}" for index in range(1, 23)],
            "trial_id": [f"T{index:02d}" for index in range(1, 23)],
            "stimulus_id": [f"S{((index - 1) % 2) + 1:02d}" for index in range(1, 23)],
            "fixation_duration": range(181, 203),
            "quality_status": pd.Categorical(
                ["pass"] * 10 + ["review"] * 12,
                categories=["pass", "review"],
            ),
        }
    )
    task = gp3mlpy.declare_gazepoint_task(
        data=data,
        outcome="quality_status",
        purpose="Predict predefined recording-quality review status",
        task_type="classification",
        unit_id="trial_id",
        participant_id="participant_id",
        stimulus_id="stimulus_id",
        generalization_target="new_participants",
        positive="review",
    )

    validation = gp3mlpy.validate_gazepoint_ml_roles(
        data=data,
        task=task,
        predictors=["trial_id", "participant_id", "fixation_duration"],
    )
    checks = validation.checks.set_index("check")

    assert checks.loc["classification_level_support", "detail"] == "pass=10, review=12"
    assert checks.loc["identifiers_not_predictors", "detail"] == "trial_id, participant_id"
