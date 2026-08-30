from __future__ import annotations

import pandas as pd

from gp3mlpy import model_tuning as mt
from gp3mlpy.objects import GP3MLModelTuning


def test_empty_tie_breakers_fall_through_to_complexity_rule():
    grid = mt.create_gazepoint_tuning_grid(["glm", "ranger"], complexity=[2, 1])
    comparison = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_001",
                "metric": "roc_auc",
                "candidate_status": "pass",
                "success_prop": 1.0,
                "mean": 0.8,
            },
            {
                "candidate_id": "candidate_002",
                "metric": "roc_auc",
                "candidate_status": "pass",
                "success_prop": 1.0,
                "mean": 0.8,
            },
        ]
    )
    tuning = GP3MLModelTuning(grid=grid, comparison=comparison)
    selection = mt.select_gazepoint_model(
        tuning,
        metric="roc_auc",
        direction="maximize",
        tie_breakers=[],
        rationale="predeclared human review",
    )
    assert selection.candidate_id == "candidate_002"
