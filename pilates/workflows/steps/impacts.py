from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from consist import (
    CacheOptions,
    OutputArtifactSpec,
    define_step,
    require_runtime_kwargs,
)

from pilates.config.models import PilatesConfig
from pilates.impacts.postsim import ImpactsPostsimRunner
from pilates.impacts.presim import ImpactsPresimRunner
from pilates.impacts.settings_bridge import presim_fleet_manifest_path
from pilates.workflows.artifact_keys import (
    BEAM_EMISSIONS_SKIMS,
    BEAM_NETWORK_FINAL,
    BEAM_R5_OSM_FILE,
    IMPACTS_ACTIVITIES_DIR,
    IMPACTS_EXPOSURE_TABLE,
    IMPACTS_FLEET_DIR,
    IMPACTS_FLEET_MANIFEST,
    IMPACTS_PIPELINE_MANIFEST,
    IMPACTS_POSTPROCESS_MANIFEST,
    IMPACTS_POSTSIM_DIR,
    IMPACTS_POSTSIM_SETTINGS,
    IMPACTS_PRESIM_SETTINGS,
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
from pilates.workspace import Workspace

_IMPACTS_INPUT_CONTRACT = InputContract(
    status="incomplete",
    reason="IMPACTS consumes the staged BEAM input tree and generated rates",
    config_contract=ConfigContract.payload(),
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
    osm = None
    if settings.impacts.use_beam_osm_artifact:
        required += (BEAM_R5_OSM_FILE,)
        osm = sim_state.extras.get(BEAM_R5_OSM_FILE)
    if osm is not None:
        suffix = f"_{year}_{state.iteration}"
        producer = getattr(osm, "run_id", None) or getattr(osm, "producer_step", None)
        for role, artifact in {**candidates, BEAM_R5_OSM_FILE: osm}.items():
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
    resolve_inputs=lambda **_: StepCallInputs(inputs={}),
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
