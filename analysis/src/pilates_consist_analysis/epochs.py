from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence
import warnings

import pandas as pd

DEFAULT_MODELS = ["urbansim", "activitysim", "beam", "atlas"]

_FIELD_ALIASES = {
    "year": ("year", "simulation_year", "facet.year"),
    "iteration": (
        "iteration",
        "outer_iteration",
        "simulation_iteration",
        "facet.iteration",
    ),
    "scenario_id": (
        "scenario_id",
        "scenario",
        "scenario.id",
        "facet.scenario_id",
    ),
    "model": ("model_name", "model", "facet.model"),
}


@dataclass
class SimulationEpoch:
    year: int
    outer_iteration: int
    scenario_id: Optional[str]
    runs: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        if not self.runs:
            return False
        for run in self.runs.values():
            if run is None:
                return False
            if _run_status(run) != "completed":
                return False
        return True

    @property
    def models(self) -> list[str]:
        return sorted(model for model, run in self.runs.items() if run is not None)

    def model_run(self, model: str) -> Any:
        key = _normalize_model(model)
        if key is None or key not in self.runs or self.runs[key] is None:
            available = ", ".join(self.models) or "<none>"
            raise KeyError(
                f"Model '{model}' not present in epoch "
                f"(year={self.year}, iteration={self.outer_iteration}, "
                f"scenario_id={self.scenario_id}). Available models: {available}."
            )
        return self.runs[key]

    def run_ids(self) -> Dict[str, str]:
        output: Dict[str, str] = {}
        for model, run in self.runs.items():
            if run is None:
                continue
            run_id = str(getattr(run, "id", "") or "").strip()
            if run_id:
                output[model] = run_id
        return output

    def __repr__(self) -> str:
        status = "complete" if self.is_complete else "incomplete"
        return (
            "SimulationEpoch("
            f"year={self.year}, "
            f"iteration={self.outer_iteration}, "
            f"scenario={self.scenario_id!r}, "
            f"models={self.models}, "
            f"{status})"
        )


@dataclass
class EpochPanel:
    scenario_id: Optional[str]
    epochs: list[SimulationEpoch]

    def years(self) -> list[int]:
        return sorted({epoch.year for epoch in self.epochs})

    def converged_epochs(self) -> "EpochPanel":
        best: Dict[tuple[int, Optional[str]], SimulationEpoch] = {}
        for epoch in self.epochs:
            if not epoch.is_complete:
                continue
            key = (epoch.year, epoch.scenario_id)
            current = best.get(key)
            if current is None or epoch.outer_iteration > current.outer_iteration:
                best[key] = epoch
        selected = sorted(
            best.values(),
            key=lambda e: (e.year, str(e.scenario_id or ""), e.outer_iteration),
        )
        return EpochPanel(scenario_id=self.scenario_id, epochs=selected)

    def epoch(self, year: int) -> SimulationEpoch:
        candidates = [item for item in self.epochs if item.year == int(year)]
        if not candidates:
            raise ValueError(f"No epoch found for year={year}.")
        if len(candidates) > 1:
            iterations = sorted(item.outer_iteration for item in candidates)
            raise ValueError(
                f"Multiple epochs found for year={year} with iterations={iterations}. "
                "Call converged_epochs() first."
            )
        return candidates[0]

    def to_frame(self) -> pd.DataFrame:
        rows: list[Dict[str, Any]] = []
        for epoch in self.epochs:
            for model, run in sorted(epoch.runs.items()):
                if run is None:
                    continue
                rows.append(
                    {
                        "year": epoch.year,
                        "outer_iteration": epoch.outer_iteration,
                        "scenario_id": epoch.scenario_id,
                        "model": model,
                        "run_id": getattr(run, "id", None),
                        "status": getattr(run, "status", None),
                        "created_at": getattr(run, "created_at", None),
                        "ended_at": getattr(run, "ended_at", None),
                        "is_complete": epoch.is_complete,
                    }
                )
        columns = [
            "year",
            "outer_iteration",
            "scenario_id",
            "model",
            "run_id",
            "status",
            "created_at",
            "ended_at",
            "is_complete",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows, columns=columns)
        return frame.sort_values(
            ["year", "outer_iteration", "model"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    def __iter__(self) -> Iterator[SimulationEpoch]:
        return iter(
            sorted(
                self.epochs,
                key=lambda e: (e.year, str(e.scenario_id or ""), e.outer_iteration),
            )
        )

    def __len__(self) -> int:
        return len(self.epochs)


def build_epoch_panel(
    tracker: Any,
    scenario_id: Optional[str] = None,
    models: Optional[Sequence[str]] = None,
) -> EpochPanel:
    scenario_filter = _normalize_optional_text(scenario_id)
    allowed_models = {
        model
        for model in (_normalize_model(item) for item in (models or DEFAULT_MODELS))
        if model is not None
    }

    runs = _collect_runs(tracker, models=sorted(allowed_models))
    grouped: Dict[tuple[int, int, Optional[str]], Dict[str, Any]] = {}
    missing_year = 0
    missing_iteration = 0
    missing_model = 0

    for run in runs:
        year = _as_int(_run_field(run, "year"))
        if year is None:
            missing_year += 1
            continue

        iteration = _as_int(_run_field(run, "iteration"))
        if iteration is None:
            missing_iteration += 1
            continue

        model = _normalize_model(_run_field(run, "model"))
        if model is None:
            missing_model += 1
            continue
        if allowed_models and model not in allowed_models:
            continue

        run_scenario_id = _normalize_optional_text(_run_field(run, "scenario_id"))
        if scenario_filter is not None and run_scenario_id != scenario_filter:
            continue

        key = (year, iteration, run_scenario_id)
        model_map = grouped.setdefault(key, {})
        model_map[model] = _preferred_run(model_map.get(model), run)

    _warn_if_nonzero(missing_year, "Runs missing year were excluded from epoch panel.")
    _warn_if_nonzero(
        missing_iteration,
        "Runs missing iteration were excluded from epoch panel.",
    )
    _warn_if_nonzero(
        missing_model, "Runs missing model were excluded from epoch panel."
    )

    epochs: list[SimulationEpoch] = []
    for (year, iteration, run_scenario_id), run_map in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], str(item[0][2] or "")),
    ):
        runs_by_model = {
            model: run_map[model]
            for model in sorted(run_map.keys())
            if run_map[model] is not None
        }
        epochs.append(
            SimulationEpoch(
                year=year,
                outer_iteration=iteration,
                scenario_id=run_scenario_id,
                runs=runs_by_model,
            )
        )

    _validate_parent_links(epochs)
    return EpochPanel(scenario_id=scenario_filter, epochs=epochs)


def converged_epoch(
    tracker: Any,
    year: int,
    scenario_id: Optional[str] = None,
    models: Optional[Sequence[str]] = None,
) -> SimulationEpoch:
    panel = build_epoch_panel(
        tracker,
        scenario_id=scenario_id,
        models=models,
    ).converged_epochs()
    return panel.epoch(year=year)


def _collect_runs(tracker: Any, models: Optional[Sequence[str]] = None) -> list[Any]:
    model_values = [value for value in (models or []) if str(value).strip()]
    collected: list[Any] = []

    if hasattr(tracker, "run_set"):
        if model_values:
            for model in model_values:
                try:
                    collected.extend(
                        list(
                            tracker.run_set(
                                label=f"epochs-{model}", model=model, limit=200000
                            )
                        )
                    )
                except Exception:
                    continue
        else:
            try:
                collected.extend(list(tracker.run_set(label="epochs", limit=200000)))
            except Exception:
                pass

    if (
        not collected
        and hasattr(tracker, "queries")
        and hasattr(tracker.queries, "find_runs")
    ):
        if model_values:
            for model in model_values:
                try:
                    collected.extend(
                        list(tracker.queries.find_runs(model=model, limit=200000))
                    )
                except Exception:
                    continue
        else:
            try:
                collected.extend(list(tracker.queries.find_runs(limit=200000)))
            except Exception:
                pass

    deduped: Dict[str, Any] = {}
    for run in collected:
        run_id = str(getattr(run, "id", "") or "").strip()
        if run_id:
            deduped[run_id] = run
    return list(deduped.values())


def _metadata_sources(run: Any) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    for name in ("metadata", "meta"):
        value = getattr(run, name, None)
        if isinstance(value, Mapping):
            output.append(value)
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


def _run_field(run: Any, field: str) -> Any:
    aliases = _FIELD_ALIASES.get(field, (field,))
    for key in aliases:
        if "." not in key and hasattr(run, key):
            value = getattr(run, key)
            if value is not None and str(value).strip() != "":
                return value

    for source in _metadata_sources(run):
        for key in aliases:
            value = _lookup_mapping(source, key)
            if value is not None and str(value).strip() != "":
                return value

    if field == "model":
        value = getattr(run, "description", None)
        if value is not None and str(value).strip():
            return value
    return None


def _run_status(run: Any) -> str:
    return str(getattr(run, "status", "") or "").strip().lower()


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    output = str(value).strip()
    return output or None


def _normalize_model(value: Any) -> Optional[str]:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return None
    return normalized.lower()


def _timestamp_value(run: Any) -> str:
    for key in ("ended_at", "updated_at", "created_at", "started_at"):
        value = getattr(run, key, None)
        if isinstance(value, datetime):
            return value.isoformat()
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _preferred_run(existing: Optional[Any], candidate: Any) -> Any:
    if existing is None:
        return candidate

    existing_key = (
        1 if _run_status(existing) == "completed" else 0,
        _timestamp_value(existing),
        str(getattr(existing, "id", "") or ""),
    )
    candidate_key = (
        1 if _run_status(candidate) == "completed" else 0,
        _timestamp_value(candidate),
        str(getattr(candidate, "id", "") or ""),
    )
    if candidate_key > existing_key:
        return candidate
    return existing


def _warn_if_nonzero(count: int, message: str) -> None:
    if int(count) <= 0:
        return
    warnings.warn(f"{message} count={int(count)}", RuntimeWarning, stacklevel=3)


def _validate_parent_links(epochs: Sequence[SimulationEpoch]) -> None:
    all_run_ids = {
        str(getattr(run, "id", "") or "").strip()
        for epoch in epochs
        for run in epoch.runs.values()
        if run is not None
    }

    missing_parent_refs = 0
    for epoch in epochs:
        for run in epoch.runs.values():
            parent_run_id = str(getattr(run, "parent_run_id", "") or "").strip()
            if parent_run_id and parent_run_id not in all_run_ids:
                missing_parent_refs += 1
    _warn_if_nonzero(
        missing_parent_refs,
        "Runs reference parent_run_id values that are not present in this panel.",
    )

    for epoch in epochs:
        asim = epoch.runs.get("activitysim")
        beam = epoch.runs.get("beam")
        if asim is None or beam is None:
            continue

        asim_id = str(getattr(asim, "id", "") or "").strip()
        beam_id = str(getattr(beam, "id", "") or "").strip()
        if not asim_id or not beam_id:
            continue

        beam_parent = str(getattr(beam, "parent_run_id", "") or "").strip()
        if not beam_parent:
            warnings.warn(
                "BEAM run missing parent_run_id in epoch "
                f"(year={epoch.year}, iteration={epoch.outer_iteration}, scenario_id={epoch.scenario_id})",
                RuntimeWarning,
                stacklevel=3,
            )
        elif beam_parent != asim_id:
            warnings.warn(
                "BEAM parent_run_id does not match ActivitySim run in epoch "
                f"(year={epoch.year}, iteration={epoch.outer_iteration}, scenario_id={epoch.scenario_id})",
                RuntimeWarning,
                stacklevel=3,
            )

        asim_parent = str(getattr(asim, "parent_run_id", "") or "").strip()
        if asim_parent and asim_parent != beam_id:
            warnings.warn(
                "ActivitySim parent_run_id does not match BEAM run in epoch "
                f"(year={epoch.year}, iteration={epoch.outer_iteration}, scenario_id={epoch.scenario_id})",
                RuntimeWarning,
                stacklevel=3,
            )
