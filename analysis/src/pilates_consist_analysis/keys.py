from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd

CANONICAL_KEY_COLUMNS = [
    "scenario_id",
    "run_id",
    "model",
    "year",
    "iteration",
    "phys_sim_iteration",
    "beam_sub_iteration",
    "seed",
]


@dataclass(frozen=True)
class AnalysisKey:
    scenario_id: Optional[str] = None
    run_id: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    iteration: Optional[int] = None
    phys_sim_iteration: Optional[int] = None
    beam_sub_iteration: Optional[int] = None
    seed: Optional[int] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AnalysisKey":
        payload: Dict[str, Any] = {}
        for field in cls.__dataclass_fields__:
            payload[field] = data.get(field)
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ensure_canonical_key_columns(
    frame: pd.DataFrame,
    *,
    fill_missing: bool = True,
    default_values: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    output = frame.copy()
    defaults = dict(default_values or {})
    for column in CANONICAL_KEY_COLUMNS:
        if column in output.columns:
            continue
        if not fill_missing:
            raise ValueError(f"Missing canonical key column: {column}")
        output[column] = defaults.get(column)
    return output


def canonical_key_dict(
    row: Mapping[str, Any],
    *,
    columns: Iterable[str] = CANONICAL_KEY_COLUMNS,
) -> Dict[str, Any]:
    return {column: row.get(column) for column in columns}
