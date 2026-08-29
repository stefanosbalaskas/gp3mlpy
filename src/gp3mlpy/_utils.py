from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from hashlib import md5, sha256
from pathlib import Path
import json
import math
import os
import re
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import GP3MLError

STATUS_ORDER = {"pass": 0, "review": 1, "fail": 2}


def timestamp() -> str:
    if os.getenv("GP3MLPY_REPRODUCIBLE_EXAMPLES", "").lower() in {"1", "true", "yes"}:
        return "<timestamp>"
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def assert_data(data: pd.DataFrame, name: str = "data", min_rows: int = 2) -> None:
    if not isinstance(data, pd.DataFrame):
        raise GP3MLError(f"`{name}` must be a data frame.")
    if len(data) < min_rows:
        raise GP3MLError(f"`{name}` must contain at least {min_rows} rows.")


def clean_columns(columns: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys(c for c in columns if c is not None and str(c) != ""))


def assert_columns(data: pd.DataFrame, columns: Iterable[str | None], argument: str = "columns") -> None:
    cols = clean_columns(columns)
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise GP3MLError(f"Missing {argument}: {', '.join(missing)}.")


def worst_status(statuses: Iterable[str]) -> str:
    vals = list(statuses)
    if not vals:
        return "fail"
    return max(vals, key=lambda x: STATUS_ORDER.get(str(x), 2))


def clip_probability(x: Sequence[float] | np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)


def seed_from(seed: int, *values: Any) -> int:
    flat: list[str] = []
    def visit(x: Any) -> None:
        if isinstance(x, (str, bytes)) or not isinstance(x, Iterable):
            flat.append(str(x))
        else:
            for y in x: visit(y)
    for value in values: visit(value)
    token = "|".join([str(int(seed)), *flat])
    raw = [ord(ch) for ch in token]
    weighted = sum(v * (i + 1) for i, v in enumerate(raw))
    return int(weighted % (2**31 - 2) + 1)


def r_round(x: float) -> int:
    # R uses IEC 60559 rounding-to-even for round().
    return int(round(float(x)))


def hash_jsonable(x: Any, algorithm: str = "sha256") -> str:
    def default(obj: Any) -> Any:
        if isinstance(obj, pd.DataFrame):
            return {"columns": list(obj.columns), "data": obj.astype(object).where(pd.notna(obj), None).values.tolist()}
        if isinstance(obj, pd.Series):
            return obj.astype(object).where(pd.notna(obj), None).tolist()
        if isinstance(obj, np.generic): return obj.item()
        if hasattr(obj, "to_dict"): return obj.to_dict()
        if isinstance(obj, Path): return str(obj)
        raise TypeError(type(obj).__name__)
    payload = json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=default).encode()
    return (sha256(payload) if algorithm == "sha256" else md5(payload)).hexdigest()


def write_tables(tables: Mapping[str, pd.DataFrame], directory: str | Path, prefix: str, overwrite: bool = False) -> dict[str, str]:
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / f"{prefix}_{name}.csv" for name in tables}
    existing = [p for p in paths.values() if p.exists()]
    if existing and not overwrite:
        raise GP3MLError("Refusing to overwrite existing files: " + ", ".join(p.name for p in existing) + ".")
    for name, table in tables.items():
        table.to_csv(paths[name], index=False, na_rep="")
    return {name: p.resolve().as_posix() for name, p in paths.items()}


def ensure_choice(value: str, choices: Sequence[str], name: str) -> str:
    if value not in choices:
        raise GP3MLError(f"`{name}` must be one of: {', '.join(choices)}.")
    return value


def as_list(x: Any) -> list[Any]:
    if isinstance(x, str) or x is None: return [x]
    if isinstance(x, pd.Series): return x.tolist()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Sequence): return list(x)
    return [x]


def recycle(x: Any, n: int, argument: str, *, kind: str, allow_na: bool = True) -> list[Any]:
    vals = as_list(x)
    if len(vals) not in (1, n):
        raise GP3MLError(f"`{argument}` must have length 1 or length {n}.")
    if kind == "character" and any(v is not None and not isinstance(v, str) for v in vals):
        raise GP3MLError(f"`{argument}` must be a character vector.")
    if kind == "logical" and any(v is not None and not isinstance(v, (bool, np.bool_)) for v in vals):
        raise GP3MLError(f"`{argument}` must be a logical vector.")
    if not allow_na and any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        raise GP3MLError(f"`{argument}` must not contain missing values.")
    return vals * n if len(vals) == 1 else vals


def is_missing_text(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip()


def identifier_like(name: str) -> bool:
    return bool(re.search(r"(^|_)(id|index|row|participant|subject|trial|stimulus|session|file|filename)(_|$)", str(name).lower()))
