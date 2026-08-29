from __future__ import annotations

import importlib.util
import pandas as pd

from .exceptions import GP3MLError


class GP3MLEngineCapabilitiesFrame(pd.DataFrame):
    """Data-frame-compatible engine table with the R print contract."""

    @property
    def _constructor(self):
        return GP3MLEngineCapabilitiesFrame

    @property
    def r_class(self) -> str:
        return "gp3ml_engine_capabilities"

    def __repr__(self) -> str:
        return self.to_string(index=False)

    def __str__(self) -> str:
        return self.to_string(index=False)


def _available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def gp3ml_engine_capabilities(check_keras_backend: bool = False) -> pd.DataFrame:
    """Audit Python engine availability while retaining gp3ml engine labels."""
    engines = ["glm", "lm", "ranger", "xgboost", "nnet", "keras3", "custom"]
    packages = [None, None, "scikit-learn", "xgboost", "scikit-learn", "keras", None]
    package_available = [True, True, True, _available("xgboost"), True, _available("keras"), True]
    backend = [None] * 7
    backend_ready: list[bool | None] = [None] * 7
    keras_row = 5
    if package_available[keras_row] and check_keras_backend:
        try:
            import keras
            value = str(keras.backend.backend())
            backend[keras_row] = value
            backend_ready[keras_row] = bool(value)
        except Exception:
            backend_ready[keras_row] = False
    status = ["available" if x else "unavailable" for x in package_available]
    if package_available[keras_row] and not check_keras_backend:
        status[keras_row] = "backend_unverified"
    elif package_available[keras_row] and check_keras_backend:
        status[keras_row] = "available" if backend_ready[keras_row] else "backend_unavailable"
    result = GP3MLEngineCapabilitiesFrame({
        "engine": engines,
        "package": packages,
        "classification": [True, False, True, True, True, True, True],
        "regression": [False, True, True, True, True, True, True],
        "probability": [True, False, True, True, True, True, None],
        "package_available": package_available,
        "backend": backend,
        "backend_ready": backend_ready,
        "status": status,
        "notes": [
            "statsmodels binomial GLM.",
            "statsmodels linear model.",
            "Python semantic adapter via sklearn RandomForest; not algorithmically identical to R ranger.",
            "Optional xgboost package; governed wrapper.",
            "Python semantic adapter via sklearn MLP; not algorithmically identical to R nnet.",
            "Optional Keras package plus configured backend; deep learning remains explicit.",
            "Externally supplied engine requires safety declarations.",
        ],
    })
    result.attrs["r_class"] = "gp3ml_engine_capabilities"
    return result


def assert_gp3ml_engine_available(engine: str, check_keras_backend: bool = False) -> bool:
    engine = str(engine)
    table = gp3ml_engine_capabilities(check_keras_backend=check_keras_backend)
    row = table.loc[table.engine == engine]
    if row.empty:
        raise GP3MLError(f"Unknown gp3ml engine `{engine}`.")
    item = row.iloc[0]
    if not bool(item.package_available):
        raise GP3MLError(f"Engine `{engine}` requires optional package `{item.package}`, which is not installed.")
    if engine == "keras3" and check_keras_backend and item.backend_ready is not True:
        raise GP3MLError("`keras3` is installed but a usable backend was not confirmed. Configure a supported Keras backend before fitting.")
    return True
