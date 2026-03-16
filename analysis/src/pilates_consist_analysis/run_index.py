from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import pandas as pd


RUN_INDEX_COLUMNS = [
    "run_id",
    "parent_run_id",
    "name",
    "status",
    "scenario_id",
    "scenario_id_source",
    "year",
    "year_source",
    "iteration",
    "iteration_source",
    "model",
    "model_source",
    "seed",
    "seed_source",
    "created_at",
    "ended_at",
    "is_complete",
    "is_completed_status",
    "is_converged_candidate",
    "has_parent",
    "archive_run_dir",
]

_FIELD_ALIASES = {
    "scenario_id": (
        "scenario_id",
        "scenario",
        "scenario.id",
        "facet.scenario_id",
    ),
    "year": ("year", "simulation_year", "facet.year"),
    "iteration": (
        "iteration",
        "outer_iteration",
        "simulation_iteration",
        "facet.iteration",
    ),
    "model": ("model_name", "model", "facet.model"),
    "seed": ("seed", "random_seed", "facet.seed"),
    "name": ("name",),
    "status": ("status",),
}


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _metadata_sources(run: Any) -> list[tuple[str, Mapping[str, Any]]]:
    output: list[tuple[str, Mapping[str, Any]]] = []
    for name in ("metadata", "meta"):
        value = getattr(run, name, None)
        if isinstance(value, Mapping):
            output.append((name, value))
    return output


def _lookup_mapping(mapping: Mapping[str, Any], key_path: str) -> Any:
    if key_path in mapping:
        return mapping.get(key_path)
    current: Any = mapping
    for part in key_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_field_with_source(run: Any, field: str) -> tuple[Any, str]:
    aliases = _FIELD_ALIASES.get(field, (field,))
    for key in aliases:
        if "." not in key and hasattr(run, key):
            value = getattr(run, key)
            if value is not None and str(value).strip() != "":
                return value, "run_attr"

    for source_name, source in _metadata_sources(run):
        for key in aliases:
            value = _lookup_mapping(source, key)
            if value is not None and str(value).strip() != "":
                if "." in key:
                    return value, f"{source_name}.{key}"
                return value, source_name

    if field == "model":
        value = getattr(run, "description", None)
        if value is not None and str(value).strip():
            return value, "fallback.description"

    if field == "name":
        value = getattr(run, "description", None)
        if value is not None and str(value).strip():
            return value, "fallback.description"

    return None, "missing"


def _run_status(run: Any) -> str:
    return str(getattr(run, "status", "") or "").strip().lower()


def _collect_runs(tracker: Any, *, limit: int = 200000) -> list[Any]:
    if hasattr(tracker, "run_set"):
        try:
            return list(tracker.run_set(label="run-index", limit=limit))
        except TypeError:
            try:
                return list(tracker.run_set("run-index", limit=limit))
            except Exception:
                pass
        except Exception:
            pass

    queries = getattr(tracker, "queries", None)
    if queries is not None and hasattr(queries, "find_runs"):
        try:
            return list(queries.find_runs(limit=limit))
        except Exception:
            return []
    return []


@dataclass(frozen=True)
class RunIndex:
    frame: pd.DataFrame
    archive_run_dir: Optional[Path] = None

    @classmethod
    def build(
        cls,
        tracker: Any,
        *,
        archive_run_dir: Optional[str | Path] = None,
        limit: int = 200000,
    ) -> "RunIndex":
        resolved_archive = (
            Path(archive_run_dir).expanduser().resolve()
            if archive_run_dir is not None
            else None
        )
        rows: list[dict[str, Any]] = []
        for run in _collect_runs(tracker, limit=limit):
            scenario_id, scenario_id_source = _run_field_with_source(run, "scenario_id")
            year, year_source = _run_field_with_source(run, "year")
            iteration, iteration_source = _run_field_with_source(run, "iteration")
            model, model_source = _run_field_with_source(run, "model")
            seed, seed_source = _run_field_with_source(run, "seed")
            name, _name_source = _run_field_with_source(run, "name")
            status, _status_source = _run_field_with_source(run, "status")

            normalized_year = _as_int(year)
            normalized_iteration = _as_int(iteration)
            normalized_seed = _as_int(seed)
            normalized_status = _normalize_optional_text(status)
            completed_status = (normalized_status or "").lower() == "completed"

            parent_run_id = _normalize_optional_text(getattr(run, "parent_run_id", None))
            rows.append(
                {
                    "run_id": _normalize_optional_text(getattr(run, "id", None)) or "",
                    "parent_run_id": parent_run_id,
                    "name": _normalize_optional_text(name),
                    "status": normalized_status,
                    "scenario_id": _normalize_optional_text(scenario_id),
                    "scenario_id_source": scenario_id_source,
                    "year": normalized_year,
                    "year_source": year_source,
                    "iteration": normalized_iteration,
                    "iteration_source": iteration_source,
                    "model": _normalize_optional_text(model),
                    "model_source": model_source,
                    "seed": normalized_seed,
                    "seed_source": seed_source,
                    "created_at": _iso(getattr(run, "created_at", None)),
                    "ended_at": _iso(getattr(run, "ended_at", None)),
                    "is_complete": bool(completed_status and normalized_iteration is not None),
                    "is_completed_status": bool(completed_status),
                    "is_converged_candidate": bool(
                        completed_status and normalized_iteration is not None
                    ),
                    "has_parent": bool(parent_run_id),
                    "archive_run_dir": str(resolved_archive) if resolved_archive is not None else None,
                }
            )

        if not rows:
            frame = pd.DataFrame(columns=RUN_INDEX_COLUMNS)
        else:
            frame = pd.DataFrame(rows, columns=RUN_INDEX_COLUMNS).sort_values(
                ["scenario_id", "year", "iteration", "model", "run_id"],
                na_position="last",
            )
            frame = frame.reset_index(drop=True)
        return cls(frame=frame, archive_run_dir=resolved_archive)

    def scenarios(self) -> list[str]:
        if self.frame.empty or "scenario_id" not in self.frame.columns:
            return []
        values = self.frame["scenario_id"].dropna().astype(str)
        return sorted(value for value in values.unique().tolist() if value.strip())

    def years(self, *, scenario_id: Optional[str] = None) -> list[int]:
        frame = self.filter(scenario_id=scenario_id)
        if frame.empty or "year" not in frame.columns:
            return []
        values = pd.to_numeric(frame["year"], errors="coerce").dropna().astype(int)
        return sorted(values.unique().tolist())

    def models(self, *, scenario_id: Optional[str] = None) -> list[str]:
        frame = self.filter(scenario_id=scenario_id)
        if frame.empty or "model" not in frame.columns:
            return []
        values = frame["model"].dropna().astype(str)
        return sorted(value for value in values.unique().tolist() if value.strip())

    def filter(
        self,
        *,
        scenario_id: Optional[str] = None,
        year: Optional[int] = None,
        iteration: Optional[int] = None,
        model: Optional[str] = None,
        status: Optional[str] = None,
        completed_only: bool = False,
    ) -> pd.DataFrame:
        frame = self.frame
        if scenario_id is not None:
            frame = frame.loc[frame["scenario_id"] == str(scenario_id)]
        if year is not None:
            frame = frame.loc[pd.to_numeric(frame["year"], errors="coerce") == int(year)]
        if iteration is not None:
            frame = frame.loc[
                pd.to_numeric(frame["iteration"], errors="coerce") == int(iteration)
            ]
        if model is not None:
            frame = frame.loc[frame["model"] == str(model)]
        if status is not None:
            frame = frame.loc[
                frame["status"].astype(str).str.lower() == str(status).strip().lower()
            ]
        if completed_only:
            frame = frame.loc[frame["is_completed_status"] == True]
        return frame.reset_index(drop=True).copy()


def build_run_index(
    tracker: Any,
    *,
    archive_run_dir: Optional[str | Path] = None,
    limit: int = 200000,
) -> RunIndex:
    return RunIndex.build(
        tracker,
        archive_run_dir=archive_run_dir,
        limit=limit,
    )
