from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class RunRecord:
    run_id: str
    parent_run_id: Optional[str]
    name: Optional[str]
    model: Optional[str]
    year: Optional[int]
    iteration: Optional[int]
    status: Optional[str]
    created_at: Optional[str]
    ended_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def find_runs(
    tracker: Any,
    *,
    model: Optional[str] = None,
    status: Optional[str] = None,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
    name: Optional[str] = None,
    limit: int = 100,
) -> List[RunRecord]:
    if not hasattr(tracker, "queries"):
        raise RuntimeError("Tracker does not expose a queries service.")

    runs = tracker.queries.find_runs(
        model=model,
        status=status,
        year=year,
        iteration=iteration,
        name=name,
        limit=limit,
    )
    records: List[RunRecord] = []
    for run in runs:
        records.append(
            RunRecord(
                run_id=str(getattr(run, "id", "") or ""),
                parent_run_id=getattr(run, "parent_run_id", None),
                name=getattr(run, "name", None),
                model=getattr(run, "model_name", None),
                year=getattr(run, "year", None),
                iteration=getattr(run, "iteration", None),
                status=getattr(run, "status", None),
                created_at=_iso(getattr(run, "created_at", None)),
                ended_at=_iso(getattr(run, "ended_at", None)),
            )
        )
    return records


def runs_to_frame(records: List[RunRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "run_id",
                "parent_run_id",
                "name",
                "model",
                "year",
                "iteration",
                "status",
                "created_at",
                "ended_at",
            ]
        )
    return pd.DataFrame([record.to_dict() for record in records])
