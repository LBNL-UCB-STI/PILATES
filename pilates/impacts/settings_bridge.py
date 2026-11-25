from __future__ import annotations

import os
import posixpath
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Literal

BEAM_INPUT_CONTAINER_DIR = "/app/beam_input"
WORKSPACE_CONTAINER_DIR = "/app/workspace"
INPUT_CACHE_CONTAINER_DIR = "/app/impacts_cache"


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


def resolve_shared_input_cache_folder(settings: Any) -> Path:
    cfg = settings.impacts
    configured = cfg.shared_input_cache_folder
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path(settings.run.output_directory).expanduser().resolve()
        / ".pilates-cache"
        / "impacts"
    )


def presim_fleet_manifest_path(settings: Any, workspace: Any, year: int) -> Path:
    label = f"{settings.run.output_run_name}_y{year}"
    return (
        Path(workspace.get_impacts_output_dir())
        / f"impacts_presim--{settings.run.region}--{label}"
        / "activities"
        / "fleet_manifest.yaml"
    )


def build_pilates_bridge_settings(
    settings: Any,
    state: Any,
    workspace: Any,
    *,
    stage: Literal["presim", "postsim"],
    beam_output_dir: Path | None = None,
    beam_osm_file: Path | None = None,
    beam_osm_artifact_id: str | None = None,
    exposure_population: tuple[Path | None, Path | None, Path | None] | None = None,
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
            WORKSPACE_CONTAINER_DIR,
            workspace.get_impacts_input_dir(),
            workspace.full_path,
        ),
        "local_output_folder": _to_container_path(
            WORKSPACE_CONTAINER_DIR,
            workspace.get_impacts_output_dir(),
            workspace.full_path,
        ),
        "seed": cfg.seed,
        "pipeline": {
            "presim": cfg.pipeline_presim.model_dump(),
            "postsim": cfg.pipeline_postsim.model_dump(),
        },
    }
    impacts_section = _deep_merge(impacts_section, deepcopy(cfg.overrides))
    impacts_section["shared_input_cache_dir"] = INPUT_CACHE_CONTAINER_DIR
    beam_output = beam_output_dir or Path(workspace.get_beam_output_dir())
    if cfg.use_generated_fleet:
        if not cfg.pipeline_presim.fleet:
            raise ValueError(
                "Using the generated IMPACTS fleet requires presim fleet generation."
            )
        if stage == "presim":
            path = (
                Path(impacts_section["local_input_folder"]) / "presim" / output_run_name
            )
            impacts_section["local_input_folder"] = path.as_posix()
        else:
            manifest = presim_fleet_manifest_path(settings, workspace, forecast_year)
            if not manifest.is_file():
                raise FileNotFoundError(
                    f"Missing current IMPACTS fleet manifest: {manifest}"
                )
            impacts_section["presim_fleet_manifest"] = _to_container_path(
                WORKSPACE_CONTAINER_DIR, str(manifest), workspace.full_path
            )

    if stage == "postsim":
        impacts_section["pipeline"]["postsim"]["exposure"] = (
            cfg.pipeline_postsim.exposure
        )
        if cfg.pipeline_postsim.exposure:
            from pilates.impacts.population import validate_exposure_files

            directory = validate_exposure_files(
                exposure_population, workspace=workspace, year=forecast_year
            )
            container_dir = _to_container_path(
                WORKSPACE_CONTAINER_DIR,
                str(directory),
                str(Path(workspace.full_path).resolve()),
            )
            exposure = impacts_section.setdefault("exposure", {})
            for key, filename in (
                ("persons_file", "persons.parquet"),
                ("households_file", "households.parquet"),
                ("source_manifest", "population_source.yaml"),
            ):
                exposure[key] = posixpath.join(container_dir, filename)
        if beam_osm_file is not None:
            if not beam_osm_file.is_file() or not beam_osm_artifact_id:
                raise ValueError(
                    "IMPACTS requires a materialized BEAM OSM artifact and ID."
                )
            relative_osm = beam_osm_file.resolve().relative_to(
                Path(workspace.full_path).resolve()
            )
            emissions = impacts_section.setdefault("emissions", {})
            emissions["osm_network_file"] = (
                Path(WORKSPACE_CONTAINER_DIR) / relative_osm
            ).as_posix()
            emissions["osm_network_artifact_id"] = beam_osm_artifact_id

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
                WORKSPACE_CONTAINER_DIR, str(beam_output), workspace.full_path
            ),
            "router_directory": settings.beam.router_directory,
        },
        "impacts": impacts_section,
    }
