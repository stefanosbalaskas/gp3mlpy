from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._utils import hash_jsonable
from .exceptions import OptionalDependencyError
from .model_engines import _default_predictors, _training_summary
from .objects import GP3MLModel, GP3MLPreprocessor, GP3MLTask
from .preprocessing import bake_gazepoint_preprocessor, fit_gazepoint_preprocessor
from .task_governance import assert_gp3ml_use_case


def fit_gazepoint_deep_model(
    data: pd.DataFrame,
    task: GP3MLTask,
    predictors: Sequence[str] | None = None,
    preprocessor: GP3MLPreprocessor | None = None,
    hidden_units: Sequence[int] = (64, 32),
    dropout: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.2,
    optimizer: str | Any = "adam",
    seed: int = 1,
    verbose: int = 0,
) -> GP3MLModel:
    """Fit the explicit optional Keras3-equivalent governed deep model."""
    try:
        import keras
    except ImportError as exc:
        raise OptionalDependencyError("Install `keras` and configure a backend to use deep learning.") from exc
    assert_gp3ml_use_case(task, data)
    predictors = _default_predictors(data, task) if predictors is None else list(predictors)
    preprocessor = preprocessor or fit_gazepoint_preprocessor(data, predictors)
    x = bake_gazepoint_preprocessor(preprocessor, data)
    if task.task_type == "classification":
        y = np.asarray([1 if str(v) == task.positive else 0 for v in data[task.outcome]], dtype=int)
    else:
        y = pd.to_numeric(data[task.outcome], errors="coerce").to_numpy(float)
    keras.utils.set_random_seed(int(seed))
    model = keras.Sequential()
    units_seq = [int(v) for v in hidden_units]
    model.add(keras.layers.Input(shape=(x.shape[1],)))
    for units in units_seq:
        model.add(keras.layers.Dense(units=units, activation="relu"))
        if dropout > 0:
            model.add(keras.layers.Dropout(rate=float(dropout)))
    model.add(keras.layers.Dense(units=1, activation="sigmoid" if task.task_type == "classification" else "linear"))
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy" if task.task_type == "classification" else "mean_squared_error",
        metrics=["accuracy"] if task.task_type == "classification" else ["mean_absolute_error"],
    )
    history = model.fit(x, y, epochs=int(epochs), batch_size=int(batch_size), validation_split=float(validation_split), verbose=int(verbose))
    counts = data[task.outcome].value_counts(dropna=False, sort=False)
    subset = data.loc[:, [task.outcome, *predictors]]
    return GP3MLModel(
        fit=model,
        history=history,
        engine="keras3",
        engine_spec=None,
        engine_args={"hidden_units": units_seq, "dropout": dropout, "epochs": epochs, "batch_size": batch_size},
        task=task,
        predictors=predictors,
        preprocessor=preprocessor,
        threshold=0.5,
        seed=seed,
        training_n=len(data),
        outcome_distribution={str(k): int(v) for k, v in counts.items()},
        predictor_summary=_training_summary(data, predictors),
        training_hash=hash_jsonable(subset, algorithm="md5"),
        call="fit_gazepoint_deep_model",
        python_backend="keras",
    )
