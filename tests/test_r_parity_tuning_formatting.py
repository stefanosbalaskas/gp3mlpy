from __future__ import annotations

from gp3mlpy import model_tuning as mt


def test_tuning_logical_arguments_use_frozen_r_text_form():
    assert mt._r_arg_text(True) == "TRUE"
    assert mt._r_arg_text(False) == "FALSE"
    assert mt._r_arg_text(2) == "2"
    assert mt._collapse_args({"center": [True, False]}) == "center=TRUE/FALSE"
    assert mt._flatten_args({"center": [True, False]}) == "center=TRUE/FALSE"

    grid = mt.create_gazepoint_tuning_grid(
        engine="glm",
        preprocessor_grid={"center": [True, False]},
        thresholds=0.4,
    )
    assert grid.candidates.label.tolist() == [
        "glm [engine:default; prep:center=TRUE; threshold=0.4]",
        "glm [engine:default; prep:center=FALSE; threshold=0.4]",
    ]
