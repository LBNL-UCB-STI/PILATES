from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from collections.abc import Mapping

from consist import (
    CacheOptions,
    OutputArtifactSpec,
    define_step,
    require_runtime_kwargs,
)

from pilates.config.models import PilatesConfig
from pilates.impacts.postsim import ImpactsPostsimRunner
from pilates.impacts.population import (
    exposure_population_dir,
    write_exposure_population,
)
from pilates.impacts.presim import ImpactsPresimRunner
from pilates.impacts.settings_bridge import presim_fleet_manifest_path
from pilates.workflows.artifact_keys import (
    BEAM_EMISSIONS_SKIMS,
    BEAM_NETWORK_FINAL,
    BEAM_R5_OSM_FILE,
    IMPACTS_ACTIVITIES_DIR,
    IMPACTS_EXPOSURE_TABLE,
    IMPACTS_EXPOSURE_HOUSEHOLDS,
    IMPACTS_EXPOSURE_PERSONS,
    IMPACTS_EXPOSURE_SOURCE,
    IMPACTS_FLEET_DIR,
    IMPACTS_FLEET_MANIFEST,
    IMPACTS_PIPELINE_MANIFEST,
    IMPACTS_POSTPROCESS_MANIFEST,
    IMPACTS_POSTSIM_DIR,
    IMPACTS_POSTSIM_SETTINGS,
    IMPACTS_PRESIM_SETTINGS,
    USIM_FORECAST_OUTPUT,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.consist_metadata import consist_step_meta
from pilates.workflows.contracts import (
    ConfigContract,
    InputContract,
    StepCallInputs,
    StepDefinition,
)
from pilates.workflows.sim_state import SimState
from pilates.workflows.state import WorkflowState
from pilates.workflows.state_helpers import resolve_forecast_year
from pilates.workflows.steps.beam import _beam_run_native_output_paths
from pilates.workflows.steps.shared import (
    _build_step_call_inputs,
    _native_contract_output_paths,
    _native_execution_options,
)
from pilates.workspace import Workspace, is_dry_run_workspace

_IMPACTS_INPUT_CONTRACT = InputContract(
    status="incomplete",
    reason="IMPACTS consumes the staged BEAM input tree and generated rates",
    config_contract=ConfigContract.payload(),
)

_EXPOSURE_CONTRACT = InputContract(
    status="complete", config_contract=ConfigContract.payload()
)
_EXPOSURE_FILES = {
    IMPACTS_EXPOSURE_PERSONS: "persons.parquet",
    IMPACTS_EXPOSURE_HOUSEHOLDS: "households.parquet",
    IMPACTS_EXPOSURE_SOURCE: "population_source.yaml",
}


def exposure_population_paths(*, settings, state, workspace, **_):
    year = resolve_forecast_year(state) or settings.run.start_year
    root = exposure_population_dir(workspace, year)
    return {key: root / name for key, name in _EXPOSURE_FILES.items()}


def _exposure_inputs(*, sim_state: SimState, **_):
    forecast = sim_state.land_use.forecast_output
    if forecast is not None and getattr(forecast, "key", None) == USIM_FORECAST_OUTPUT:
        key, source = USIM_FORECAST_OUTPUT, forecast
    else:
        source = sim_state.land_use.population_source
        if source is None or getattr(
            source, "key", USIM_POPULATION_SOURCE_H5
        ) != USIM_POPULATION_SOURCE_H5:
            raise ValueError(
                "IMPACTS exposure requires the active UrbanSim forecast or baseline "
                "UrbanSim population-source HDF5."
            )
        key = USIM_POPULATION_SOURCE_H5
    return StepCallInputs(
        inputs={key: source},
        required_keys=(key,),
    )


def _exposure_execution_options(*, step_inputs: StepCallInputs, **_):
    artifact_key = next(iter(step_inputs.inputs))
    artifact = step_inputs.inputs[artifact_key]
    metadata = getattr(artifact, "meta", {}) or {}
    digest = getattr(artifact, "digest", None)
    source = {
        "source_run_id": metadata.get(
            "source_run_id", getattr(artifact, "run_id", None)
        ),
        "source_artifact_id": metadata.get(
            "source_artifact_id", str(getattr(artifact, "id", artifact))
        ),
        "source_artifact_key": artifact_key,
        "source_type": (
            "urbansim_forecast"
            if artifact_key == USIM_FORECAST_OUTPUT
            else "baseline_urbansim_input"
        ),
        "source_digest": metadata.get("source_digest")
        or (
            str(digest)
            if digest is not None and digest.scheme == "sha256:file"
            else None
        ),
    }
    if artifact_key == USIM_FORECAST_OUTPUT and not source["source_run_id"]:
        raise ValueError(
            "IMPACTS exposure forecast is missing its Consist producing step ID."
        )
    return replace(
        _native_execution_options(),
        runtime_kwargs={"exposure_population_source": source},
    )


@define_step(
    model="impacts_exposure_population",
    name_template="impacts_exposure_population__y{year}__i{iteration}__phase_{phase}",
    inputs={USIM_FORECAST_OUTPUT: None, USIM_POPULATION_SOURCE_H5: None},
    optional_input_keys=(USIM_FORECAST_OUTPUT, USIM_POPULATION_SOURCE_H5),
    schema_outputs=tuple(_EXPOSURE_FILES),
    output_paths=_native_contract_output_paths(exposure_population_paths),
    config=lambda ctx: {
        "population_source": ctx.get_runtime("exposure_population_source", default=None)
    },
    input_binding="paths",
)
@require_runtime_kwargs("settings", "state", "workspace", "exposure_population_source")
def _native_exposure_population(
    usim_forecast_output: Path | None = None,
    usim_population_source_h5: Path | None = None,
    *,
    settings,
    state,
    workspace,
    exposure_population_source,
) -> None:
    source_path = usim_forecast_output or usim_population_source_h5
    if source_path is None:
        raise ValueError("IMPACTS exposure received no UrbanSim population HDF5 input.")
    paths = exposure_population_paths(
        settings=settings, state=state, workspace=workspace
    )
    write_exposure_population(
        source_path,
        paths[IMPACTS_EXPOSURE_SOURCE].parent,
        year=resolve_forecast_year(state) or settings.run.start_year,
        source=exposure_population_source,
    )


def validate_exposure_artifacts(artifacts: Mapping) -> None:
    producers = {
        getattr(artifact, "run_id", None) or getattr(artifact, "producer_step", None)
        for artifact in artifacts.values()
    }
    if (
        set(artifacts) != set(_EXPOSURE_FILES)
        or any(
            getattr(artifact, "key", None) != key for key, artifact in artifacts.items()
        )
        or len(producers) != 1
        or None in producers
    ):
        raise ValueError(
            "IMPACTS exposure requires all three files from the same forecast preparation step."
        )


impacts_exposure_population = StepDefinition(
    name="impacts_exposure_population",
    function=_native_exposure_population,
    resolve_inputs=_exposure_inputs,
    project_outputs=lambda outputs, **_: outputs,
    input_contract=_EXPOSURE_CONTRACT,
    output_paths=exposure_population_paths,
    execution_options=_exposure_execution_options,
    cache_options=lambda **_: CacheOptions(
        cache_hydration="outputs-requested",
        cache_hydration_failure="miss",
        validate_cached_outputs="eager",
    ),
)


def _presim_paths(*, settings, state, workspace, **_):
    if getattr(settings, "impacts", None) is None:
        return {}
    paths = {
        IMPACTS_PRESIM_SETTINGS: Path(workspace.get_impacts_settings_path("presim"))
    }
    year = resolve_forecast_year(state) or settings.run.start_year
    manifest = presim_fleet_manifest_path(settings, workspace, year)
    presim = settings.impacts.pipeline_presim
    if presim.activities or presim.fleet:
        paths[IMPACTS_ACTIVITIES_DIR] = OutputArtifactSpec(
            path=manifest.parent, meta={"directory_artifact": True}
        )
    if settings.impacts.use_generated_fleet:
        paths[IMPACTS_FLEET_MANIFEST] = manifest
        label = f"{settings.run.output_run_name}_y{year}"
        paths[IMPACTS_FLEET_DIR] = OutputArtifactSpec(
            path=Path(workspace.get_impacts_input_dir()) / "presim" / label / "emfac",
            meta={"directory_artifact": True},
        )
    return paths


def _presim_inputs(*, settings, workspace, **_):
    # Check the PBF before Consist can reuse a cached presim result.
    if not is_dry_run_workspace(workspace):
        from pilates.beam.launch_paths import resolve_r5_network_reference
        from pilates.beam.preprocessor import ensure_beam_mutable_workspace

        ensure_beam_mutable_workspace(settings, workspace)
        resolve_r5_network_reference(settings=settings, workspace=workspace)
    return StepCallInputs()


def _postsim_paths(*, settings, state, workspace, **_):
    if getattr(settings, "impacts", None) is None:
        return {}
    year = resolve_forecast_year(state) or settings.run.start_year
    label = f"{settings.run.output_run_name}_y{year}"
    root = (
        Path(workspace.get_impacts_output_dir())
        / f"impacts-postsim--{settings.run.region}--{label}"
    )
    paths = {
        IMPACTS_POSTSIM_DIR: OutputArtifactSpec(
            path=root, meta={"directory_artifact": True}
        ),
        IMPACTS_POSTSIM_SETTINGS: Path(workspace.get_impacts_settings_path("postsim")),
        IMPACTS_PIPELINE_MANIFEST: root / "pipeline_manifest.yaml",
        IMPACTS_POSTPROCESS_MANIFEST: root / "postprocess_manifest.yaml",
    }
    if settings.impacts.pipeline_postsim.exposure:
        paths[IMPACTS_EXPOSURE_TABLE] = (
            root / "exposure/beam_concentration_distribution.parquet"
        )
    return paths


def _postsim_inputs(*, settings, state, workspace, sim_state: SimState, **_):
    year = resolve_forecast_year(state) or settings.run.start_year
    root = Path(workspace.get_impacts_input_dir()) / "beam-output" / str(year)
    required = (BEAM_EMISSIONS_SKIMS, BEAM_NETWORK_FINAL)
    candidates = {role: sim_state.extras.get(role) for role in required}
    destinations = {
        BEAM_EMISSIONS_SKIMS: root / "ITERS/it.0/0.skimsEmissions.parquet",
        BEAM_NETWORK_FINAL: root / "network.csv.gz",
        "impacts_beam_events": root / "ITERS/it.0/0.events.parquet",
    }
    if settings.impacts.pipeline_postsim.exposure:
        population = {key: sim_state.extras.get(key) for key in _EXPOSURE_FILES}
        validate_exposure_artifacts(population)
        required += tuple(_EXPOSURE_FILES)
        candidates.update(population)
        destinations.update(
            exposure_population_paths(
                settings=settings, state=state, workspace=workspace
            )
        )
    osm = None
    if settings.impacts.use_beam_osm_artifact:
        required += (BEAM_R5_OSM_FILE,)
        osm = sim_state.extras.get(BEAM_R5_OSM_FILE)
    if osm is not None:
        suffix = f"_{year}_{state.iteration}"
        producer = getattr(osm, "run_id", None) or getattr(osm, "producer_step", None)
        for role in (BEAM_EMISSIONS_SKIMS, BEAM_NETWORK_FINAL, BEAM_R5_OSM_FILE):
            artifact = osm if role == BEAM_R5_OSM_FILE else candidates[role]
            artifact_producer = getattr(artifact, "run_id", None) or getattr(
                artifact, "producer_step", None
            )
            if (
                not producer
                or artifact_producer != producer
                or getattr(artifact, "key", None) != role + suffix
            ):
                raise ValueError(
                    "IMPACTS requires the OSM PBF, network and skims from the "
                    f"same BEAM run for year {year}, iteration {state.iteration}."
                )
        osm_key = BEAM_R5_OSM_FILE + suffix
        osm_path = _beam_run_native_output_paths(
            settings=settings, state=state, workspace=workspace
        )[osm_key]
        if not osm_path.resolve().is_relative_to(Path(workspace.full_path).resolve()):
            osm_path = root / f"iteration-{state.iteration}" / osm_path.name
        destinations[BEAM_R5_OSM_FILE] = osm_path
        candidates[BEAM_R5_OSM_FILE] = osm
    if settings.impacts.use_generated_fleet:
        required += (IMPACTS_FLEET_MANIFEST,)
    return _build_step_call_inputs(
        candidates={
            **candidates,
            IMPACTS_FLEET_MANIFEST: sim_state.extras.get(IMPACTS_FLEET_MANIFEST),
            "impacts_beam_events": sim_state.network.events_parquet,
        },
        required_roles=required,
        optional_roles=("impacts_beam_events",),
        logical_destinations=destinations,
    )


def _postsim_execution_options(*, step_inputs: StepCallInputs, **_):
    options = _native_execution_options(step_inputs=step_inputs)
    osm = step_inputs.inputs.get(BEAM_R5_OSM_FILE)
    if osm is None:
        return options
    return replace(options, runtime_kwargs={"beam_osm_artifact_id": str(osm.id)})


@define_step(
    model="impacts_presim",
    name_template="impacts_presim__y{year}__i{iteration}__phase_{phase}",
    schema_outputs=[IMPACTS_PRESIM_SETTINGS],
    outputs=_native_contract_output_paths(_presim_paths),
    input_binding="paths",
    **consist_step_meta("impacts_presim", input_contract=_IMPACTS_INPUT_CONTRACT),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_impacts_presim(
    *, settings: PilatesConfig, state: WorkflowState, workspace: Workspace
) -> None:
    from pilates.beam.preprocessor import ensure_beam_mutable_workspace

    ensure_beam_mutable_workspace(settings, workspace)
    ImpactsPresimRunner("impacts_presim", state).run(None, workspace)


@define_step(
    model="impacts_postsim",
    name_template="impacts_postsim__y{year}__i{iteration}__phase_{phase}",
    inputs={BEAM_EMISSIONS_SKIMS: None, BEAM_NETWORK_FINAL: None},
    optional_input_keys=(
        IMPACTS_FLEET_MANIFEST,
        "impacts_beam_events",
        BEAM_R5_OSM_FILE,
        *_EXPOSURE_FILES,
    ),
    schema_outputs=[
        IMPACTS_POSTSIM_SETTINGS,
        IMPACTS_PIPELINE_MANIFEST,
        IMPACTS_POSTPROCESS_MANIFEST,
    ],
    outputs=_native_contract_output_paths(_postsim_paths),
    input_binding="paths",
    **consist_step_meta("impacts_postsim", input_contract=_IMPACTS_INPUT_CONTRACT),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_impacts_postsim(
    skims_emissions_parquet: Path,
    beam_network_final: Path,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    impacts_fleet_manifest: Path | None = None,
    impacts_beam_events: Path | None = None,
    beam_r5_osm_file: Path | None = None,
    beam_osm_artifact_id: str | None = None,
    impacts_exposure_persons: Path | None = None,
    impacts_exposure_households: Path | None = None,
    impacts_exposure_population_source: Path | None = None,
) -> None:
    del settings, impacts_fleet_manifest, impacts_beam_events
    root = beam_network_final.parent
    if skims_emissions_parquet != root / "ITERS/it.0/0.skimsEmissions.parquet":
        raise ValueError(
            "IMPACTS requires the network and skims staged for the same BEAM run."
        )
    ImpactsPostsimRunner(
        "impacts_postsim",
        state,
        beam_osm_file=beam_r5_osm_file,
        beam_osm_artifact_id=beam_osm_artifact_id,
        exposure_population=(
            impacts_exposure_persons,
            impacts_exposure_households,
            impacts_exposure_population_source,
        ),
    ).run(root, workspace)


def _impacts_cache(*, state, **_) -> CacheOptions:
    if state.is_restart_run:
        return CacheOptions(
            cache_hydration="outputs-requested",
            cache_hydration_failure="miss",
            validate_cached_outputs="eager",
        )
    # shortcut: fresh executions rerun until the contract tracks all source data.
    return CacheOptions(cache_mode="overwrite")


impacts_presim = StepDefinition(
    name="impacts_presim",
    function=_native_impacts_presim,
    resolve_inputs=_presim_inputs,
    project_outputs=lambda outputs, **_: outputs,
    input_contract=_IMPACTS_INPUT_CONTRACT,
    outputs=_presim_paths,
    cache_options=_impacts_cache,
)
impacts_postsim = StepDefinition(
    name="impacts_postsim",
    function=_native_impacts_postsim,
    resolve_inputs=_postsim_inputs,
    project_outputs=lambda outputs, **_: outputs,
    input_contract=_IMPACTS_INPUT_CONTRACT,
    outputs=_postsim_paths,
    execution_options=_postsim_execution_options,
    cache_options=_impacts_cache,
)
