from __future__ import annotations

import numpy as np
import pandas as pd

from gp3mlpy import bake_gazepoint_preprocessor, fit_gazepoint_preprocessor


def test_logical_predictors_match_r_factor_labels_and_columns() -> None:
    data = pd.DataFrame(
        {
            "logical": pd.Series(
                [True, False, pd.NA, True, False, True],
                dtype="boolean",
            )
        }
    )

    preprocessor = fit_gazepoint_preprocessor(
        data,
        ["logical"],
        center=False,
        scale=False,
        novel_level="other",
        remove_zero_variance=False,
    )

    assert preprocessor.factor_levels["logical"] == [
        "<missing>",
        "<other>",
        "FALSE",
        "TRUE",
    ]
    assert preprocessor.columns == [
        "logical<missing>",
        "logical<other>",
        "logicalFALSE",
        "logicalTRUE",
    ]

    baked = bake_gazepoint_preprocessor(preprocessor, data)
    np.testing.assert_array_equal(
        baked,
        np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
    )
