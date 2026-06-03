from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from pilates.runtime.archive_paths import archive_fallback_path
from pilates.utils.coupler_helpers import resolve_existing_path
from pilates.workflows.artifact_keys import ATLAS_VEHICLES2_OUTPUT


@dataclass(frozen=True)
class AtlasVehicles2Source:
    """Resolved ATLAS vehicles2 source for BEAM vehicle staging."""

    selected_path: Path
    forecast_year: int
    source_year: int
    storage_location: str
    resolution_mode: str
    candidates: Tuple[str, ...]

    def as_input_metadata(self) -> dict[str, Any]:
        return {
            "source_semantic_key": ATLAS_VEHICLES2_OUTPUT,
            "source_path": str(self.selected_path),
            "source_year": self.source_year,
            "forecast_year": self.forecast_year,
            "source_storage_location": self.storage_location,
            "source_resolution_mode": self.resolution_mode,
            "candidate_paths": list(self.candidates),
        }


def _candidate_paths_for_year(
    *,
    state: Any,
    workspace: Any,
    source_year: int,
) -> tuple[Path, Optional[Path]]:
    local_path = Path(workspace.get_atlas_output_dir()) / f"vehicles2_{source_year}.csv"
    archive_path = archive_fallback_path(
        state=state,
        workspace=workspace,
        local_path=local_path,
    )
    return local_path, archive_path


def resolve_atlas_vehicles2_source(
    *,
    state: Any,
    workspace: Any,
    require_exact_year: bool,
) -> Optional[AtlasVehicles2Source]:
    """Resolve the ATLAS vehicles2 source used to stage BEAM vehicles.

    Exact mode accepts only ``vehicles2_{forecast_year}.csv``. Legacy mode tries
    the forecast year first and then permits the prior forecast year fallback.
    Both modes search the local run workspace before the archive fallback path.
    """
    forecast_year = getattr(state, "forecast_year", None)
    if forecast_year is None:
        if require_exact_year:
            raise FileNotFoundError(
                "BEAM preprocess requires forecast-year ATLAS vehicles2, but "
                "WorkflowState.forecast_year is not set."
            )
        return None

    forecast_year = int(forecast_year)
    source_years = [forecast_year]
    if not require_exact_year:
        source_years.append(forecast_year - 1)

    candidates: list[str] = []
    for source_year in source_years:
        local_path, archive_path = _candidate_paths_for_year(
            state=state,
            workspace=workspace,
            source_year=source_year,
        )
        ordered = (("local", local_path), ("archive", archive_path))
        for storage_location, candidate in ordered:
            if candidate is None:
                continue
            candidates.append(str(candidate))
            if candidate.exists():
                return _resolved_source(
                    selected_path=candidate,
                    forecast_year=forecast_year,
                    source_year=source_year,
                    storage_location=storage_location,
                    candidates=candidates,
                )
        env_resolved = resolve_existing_path(
            str(local_path),
            workspace=workspace,
            materialize_from_archive=True,
        )
        if env_resolved:
            resolved_path = Path(env_resolved)
            candidates.append(str(resolved_path))
            return _resolved_source(
                selected_path=resolved_path,
                forecast_year=forecast_year,
                source_year=source_year,
                storage_location=(
                    "local" if resolved_path == local_path else "archive"
                ),
                candidates=candidates,
            )

    if require_exact_year:
        raise FileNotFoundError(
            "BEAM preprocess requires exact forecast-year ATLAS vehicles2, but "
            f"vehicles2_{forecast_year}.csv was not found in local or archive "
            f"candidates: {list(dict.fromkeys(candidates))}"
        )
    return None


def _resolved_source(
    *,
    selected_path: Path,
    forecast_year: int,
    source_year: int,
    storage_location: str,
    candidates: list[str],
) -> AtlasVehicles2Source:
    return AtlasVehicles2Source(
        selected_path=selected_path,
        forecast_year=forecast_year,
        source_year=source_year,
        storage_location=storage_location,
        resolution_mode=(
            "exact_forecast_year"
            if source_year == forecast_year
            else "legacy_prior_year"
        ),
        candidates=tuple(dict.fromkeys(candidates)),
    )
