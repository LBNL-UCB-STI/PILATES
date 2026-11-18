from __future__ import annotations

import os
import posixpath
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Literal

BEAM_INPUT_CONTAINER_DIR = "/app/beam_input"
WORKSPACE_CONTAINER_DIR = "/app/workspace"

_PASSENGER_EXCHANGE_EXTENSIONS = (".csv.gz", ".csv", ".parquet")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _to_container_path(container_root: str, host_path: str, host_root: str) -> str:
    relpath = os.path.relpath(host_path, host_root)
    return posixpath.normpath(posixpath.join(container_root, *Path(relpath).parts))


def _has_exchange_file(directory: Path, stem: str) -> bool:
    return any(
        (directory / f"{stem}{ext}").exists() for ext in _PASSENGER_EXCHANGE_EXTENSIONS
    )


def resolve_population_override(settings: Any, workspace: Any) -> Dict[str, Any]:
    """
    If this run's PILATES-generated population (households/persons fed to BEAM)
    exists, point IMPACTS' population.passenger_folder at it. Otherwise return
    {} so IMPACTS falls back to its configured static reference population.
    """
    cfg = settings.impacts
    if cfg is None or not cfg.population_fallback_to_beam_inputs:
        return {}

    beam_mutable_dir = Path(workspace.get_beam_mutable_data_dir())
    if not (
        _has_exchange_file(beam_mutable_dir, "households")
        and _has_exchange_file(beam_mutable_dir, "persons")
    ):
        return {}

    container_path = _to_container_path(
        WORKSPACE_CONTAINER_DIR, str(beam_mutable_dir), workspace.full_path
    )
    return {"population": {"passenger_folder": container_path}}


def build_pilates_bridge_settings(
    settings: Any,
    state: Any,
    workspace: Any,
    *,
    stage: Literal["presim", "postsim"],
) -> Dict[str, Any]:
    """
    Build the PILATES->IMPACTS bridge settings dict for the given stage,
    consumed by `python -m impacts derive_settings_from_pilates`.
    """
    cfg = settings.impacts
    if cfg is None:
        raise ValueError("Impacts config is missing")

    forecast_year = getattr(state, "forecast_year", None) or settings.run.start_year
    output_run_name = f"{settings.run.output_run_name}_y{forecast_year}"

    geography = settings.shared.geography
    fips = dict(geography.FIPS or {})

    impacts_section: Dict[str, Any] = {
        "local_input_folder": _to_container_path(
            WORKSPACE_CONTAINER_DIR, workspace.get_impacts_input_dir(), workspace.full_path
        ),
        "local_output_folder": _to_container_path(
            WORKSPACE_CONTAINER_DIR, workspace.get_impacts_output_dir(), workspace.full_path
        ),
        "seed": cfg.seed,
        "pipeline": {
            "presim": cfg.pipeline_presim.model_dump(),
            "postsim": cfg.pipeline_postsim.model_dump(),
        },
    }
    impacts_section = _deep_merge(impacts_section, deepcopy(cfg.overrides))

    if stage == "postsim":
        impacts_section = _deep_merge(
            impacts_section, resolve_population_override(settings, workspace)
        )

    return {
        "run": {
            "region": settings.run.region,
            "scenario": settings.run.scenario,
            "start_year": forecast_year,
            "output_run_name": output_run_name,
        },
        "shared": {
            "geography": {
                "FIPS": fips,
                "local_crs": geography.local_crs,
            },
        },
        "beam": {
            "local_input_folder": BEAM_INPUT_CONTAINER_DIR,
            "local_output_folder": _to_container_path(
                WORKSPACE_CONTAINER_DIR, workspace.get_beam_output_dir(), workspace.full_path
            ),
            "router_directory": settings.beam.router_directory,
        },
        "impacts": impacts_section,
    }
