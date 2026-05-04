"""
Runtime launcher for PILATES simulations.

This module assembles and runs the full simulation lifecycle:
- settings and workflow state initialization
- Consist tracker and scenario setup
- restart hydration and bootstrap initialization
- multi-stage yearly orchestration

`run.py` is the thin CLI entrypoint. This module owns the runtime assembly.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
import os
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Sequence, cast

from pilates.config import PilatesConfig
from pilates.workspace import Workspace
from pilates.generic.initialization import (
    Initialization,
    build_bootstrap_artifact_summary,
)
from pilates.utils.formatting import formatted_print
from pilates.utils.io import parse_args_and_settings
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_config import (
    build_scenario_consist_kwargs,
    build_step_consist_kwargs,
)
from pilates.utils.consist_db_snapshot import (
    ConsistDbSnapshotManager,
    mirror_consist_db_to_archive,
    resolve_consist_db_paths,
    restore_local_consist_db_from_snapshot,
    seed_local_consist_db_from_shared,
)
from pilates.utils.coupler_helpers import (
    flush_archive_queue,
    stop_archive_worker,
)
from pilates.atlas.inputs import (
    atlas_static_input_relpaths,
    build_atlas_static_inputs_fallback,
)
from pilates.activitysim.preprocessor import required_asim_config_dirs
from pilates.urbansim.postprocessor import get_usim_datastore_fname
from pilates.utils.consist_types import ScenarioWithCoupler
from pilates.runtime import bootstrap as bootstrap_runtime
from pilates.runtime.consist_audit import (
    emit_artifact_lifecycle_audit_event,
    emit_consist_audit_event,
)
from pilates.runtime.context import WorkflowRuntimeContext
from pilates.runtime.failure_hints import (
    RUN_FAILURE_CONTEXT,
    clear_run_failure_context,
    format_hpc_restart_command,
    format_restart_command,
    log_restart_instructions_on_failure,
    set_run_failure_context,
)
from pilates.runtime import restart as restart_runtime
from pilates.runtime import scenario_runtime
from pilates.runtime.run_notifications import (
    RunNotificationContext,
    register_consist_run_notification_hooks,
)
from pilates.runtime.run_publishers import register_consist_run_publishers
from pilates.runtime.storage_probe import log_local_storage_info_if_enabled
from pilates.workflows._profile import ensure_runtime_flags_initialized
from pilates.workflows.coupler_schema import build_coupler_schema
from pilates.workflows.surface import EnabledWorkflowSurface, build_enabled_workflow_surface
from pilates.workflows.stages import (
    run_land_use_stage,
    run_postprocessing_stage,
    run_supply_demand_stage,
    run_vehicle_ownership_stage,
)
from pilates.workflows.stages.supply_demand_resume import (
    seed_supply_demand_parent_run_ids_for_resume,
)
from pilates.workflows.stages.handoffs import LandUseToSupplyDemandHandoff

warnings.simplefilter(action="ignore", category=FutureWarning)
from workflow_state import WorkflowState  # noqa: E402

from pilates.workflows.steps import StepOutputsHolder, validate_workflow_step_contracts  # noqa: E402
from consist.types import CacheOptions  # noqa: E402

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_SCENARIO_NAME_TEMPLATE = "{func_name}__y{year}__i{iteration}__phase_{phase}"
_RUN_FAILURE_CONTEXT = RUN_FAILURE_CONTEXT


@dataclass(frozen=True)
class PreparedRunContext:
    """
    Runtime objects resolved before bootstrap and scenario execution begin.

    This keeps the launcher lifecycle focused on orchestration while preserving
    the existing storage, tracker, workspace, and failure-context setup order.
    """

    settings: PilatesConfig
    state: WorkflowState
    surface: EnabledWorkflowSurface
    workspace: Workspace
    runtime_context: WorkflowRuntimeContext
    tracker: Any
    snapshot_manager: ConsistDbSnapshotManager
    run_name: str
    scenario_id: str
    seed: Optional[int]
    cache_epoch: int
    is_restart_run: bool
    archive_run_dir: str
    local_run_dir: str
    archive_state_path: str
    local_state_path: str
    local_consist_db_path: Optional[str]
    archive_consist_db_path: Optional[str]


def _resolve_scenario_id(settings: PilatesConfig) -> str:
    return scenario_runtime.resolve_scenario_id(settings)


def _resolve_seed(settings: PilatesConfig) -> Optional[int]:
    return scenario_runtime.resolve_seed(settings)


def _set_run_failure_context(**kwargs: Any) -> None:
    set_run_failure_context(**kwargs)


def _format_restart_command(
    *,
    settings: Optional[Any],
    archive_state_path: Optional[str],
) -> Optional[str]:
    return format_restart_command(
        settings=settings,
        archive_state_path=archive_state_path,
    )


def _format_hpc_restart_command(
    *,
    settings: Optional[Any],
    archive_state_path: Optional[str],
) -> Optional[str]:
    return format_hpc_restart_command(
        settings=settings,
        archive_state_path=archive_state_path,
    )


def _log_restart_instructions_on_failure() -> None:
    log_restart_instructions_on_failure(logger=logger)


def _merge_tag_list(existing: Any, additions: Sequence[str]) -> List[str]:
    return scenario_runtime.merge_tag_list(existing, additions)


def _merge_epoch_facet(
    *,
    existing: Any,
    scenario_id: str,
    seed: Optional[int],
    model: Optional[str],
    year: Optional[int],
    iteration: Optional[int],
) -> Dict[str, Any]:
    return scenario_runtime.merge_epoch_facet(
        existing=existing,
        scenario_id=scenario_id,
        seed=seed,
        model=model,
        year=year,
        iteration=iteration,
    )


_ScenarioParentLinkProxy = scenario_runtime.ScenarioParentLinkProxy


def _resolve_cache_epoch(settings: PilatesConfig) -> int:
    return scenario_runtime.resolve_cache_epoch(settings)


def _resolve_run_storage_roots(settings: PilatesConfig) -> tuple[str, str]:
    """
    Resolve the archive and mutable run roots for the current execution.

    The launcher owns this topology:
    - ``run.output_directory`` is the durable archive root on shared scratch.
    - ``run.local_workspace_root`` is the mutable node-local workspace root.
    """
    output_directory = settings.run.output_directory
    if not output_directory:
        raise ValueError("output_directory not found in config")
    archive_root = os.path.realpath(os.path.expandvars(output_directory))
    local_workspace_root = settings.run.local_workspace_root
    if local_workspace_root:
        local_root = os.path.realpath(os.path.expandvars(local_workspace_root))
    else:
        local_root = archive_root
    return archive_root, local_root


def _configure_run_storage_environment(
    *,
    archive_run_dir: str,
    local_run_dir: str,
    enable_archive_copy: bool,
) -> None:
    """
    Export the runtime storage topology for helpers that archive logged outputs.
    """
    os.environ["PILATES_LOCAL_RUN_DIR"] = local_run_dir
    os.environ["PILATES_ARCHIVE_RUN_DIR"] = archive_run_dir
    os.environ["PILATES_ENABLE_ARCHIVE_COPY"] = "1" if enable_archive_copy else "0"


def _build_schema_steps() -> List[Callable[..., Any]]:
    return scenario_runtime.build_schema_steps()


def _filter_schema_steps_for_enabled_models(
    steps: List[Callable[..., Any]],
    *,
    include_optional: bool = True,
    surface: EnabledWorkflowSurface,
) -> List[Callable[..., Any]]:
    return scenario_runtime.filter_schema_steps_for_enabled_models(
        steps,
        include_optional=include_optional,
        surface=surface,
    )


def _get_consist_schemas() -> Optional[list[type[Any]]]:
    try:
        from pilates.database.schema.registry import get_consist_schemas

        return get_consist_schemas()
    except Exception:
        return None


def _build_scenario_runtime_contract(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    scenario_id: str,
    seed: Optional[int],
    cache_epoch: int,
    surface: EnabledWorkflowSurface,
) -> Dict[str, Any]:
    return scenario_runtime.build_scenario_runtime_contract(
        settings=settings,
        state=state,
        workspace=workspace,
        scenario_id=scenario_id,
        seed=seed,
        cache_epoch=cache_epoch,
        build_scenario_consist_kwargs_fn=build_scenario_consist_kwargs,
        build_coupler_schema_fn=build_coupler_schema,
        validate_workflow_step_contracts_fn=validate_workflow_step_contracts,
        build_schema_steps_fn=_build_schema_steps,
        filter_schema_steps_for_enabled_models_fn=_filter_schema_steps_for_enabled_models,
        merge_epoch_facet_fn=_merge_epoch_facet,
        scenario_name_template=_SCENARIO_NAME_TEMPLATE,
        surface=surface,
    )

def build_manifest_path(workspace: Workspace, year: int, iteration: int) -> Path:
    return (
        Path(workspace.full_path)
        / ".workflow"
        / f"year_{year}_iteration_{iteration}.yaml"
    )


def _log_local_storage_info() -> None:
    log_local_storage_info_if_enabled()


def run_bootstrap_phase(
    *,
    tracker: Any,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    scenario_id: str,
    seed: Optional[int],
    surface: Optional[EnabledWorkflowSurface] = None,
) -> Dict[str, Any]:
    return bootstrap_runtime.run_bootstrap_phase(
        tracker=tracker,
        settings=settings,
        state=state,
        workspace=workspace,
        scenario_id=scenario_id,
        seed=seed,
        surface=surface,
        initialization_cls=Initialization,
        build_bootstrap_artifact_summary_fn=build_bootstrap_artifact_summary,
        build_step_consist_kwargs_fn=build_step_consist_kwargs,
        merge_tag_list_fn=_merge_tag_list,
        merge_epoch_facet_fn=_merge_epoch_facet,
        cache_options_cls=CacheOptions,
    )


def _assert_bootstrap_output_invariant(
    bootstrap_result: Optional[Dict[str, Any]],
) -> None:
    bootstrap_runtime.assert_bootstrap_output_invariant(bootstrap_result)


def _restart_required_local_artifacts(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    surface: Optional[EnabledWorkflowSurface] = None,
) -> List[restart_runtime.RestartArtifactDiagnostic]:
    return restart_runtime.restart_required_local_artifacts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
        get_usim_datastore_fname_fn=get_usim_datastore_fname,
        required_asim_config_dirs_fn=required_asim_config_dirs,
        atlas_static_input_relpaths_fn=atlas_static_input_relpaths,
        workflow_stage=WorkflowState.Stage,
    )


def _find_missing_restart_local_artifacts(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    surface: Optional[EnabledWorkflowSurface] = None,
) -> List[restart_runtime.RestartArtifactDiagnostic]:
    return restart_runtime.find_missing_restart_local_artifacts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
        restart_required_local_artifacts_fn=_restart_required_local_artifacts,
    )


def _read_archive_run_state_year(state_path: str) -> Optional[int]:
    return restart_runtime.read_archive_run_state_year(
        state_path,
        read_current_stage_fn=WorkflowState.read_current_stage,
    )


def _enforce_resume_rewind_guardrail(
    *,
    state: WorkflowState,
    archive_state_path: str,
    allow_rewind_resume: bool,
) -> None:
    restart_runtime.enforce_resume_rewind_guardrail(
        state=state,
        archive_state_path=archive_state_path,
        allow_rewind_resume=allow_rewind_resume,
        read_archive_run_state_year_fn=_read_archive_run_state_year,
    )


def _prepare_run_context(
    *,
    settings: Optional[PilatesConfig] = None,
    state: Optional[WorkflowState] = None,
    clear_failure_context: bool = True,
) -> PreparedRunContext:
    """
    Resolve the run-local objects needed before bootstrap/scenario execution.
    """
    if clear_failure_context:
        clear_run_failure_context()

    if settings is None:
        settings = parse_args_and_settings()
    ensure_runtime_flags_initialized(settings)
    if state is None:
        state = WorkflowState.from_settings(settings)
    surface = build_enabled_workflow_surface(settings, state=state)
    _set_run_failure_context(settings=settings, state=state)

    _log_local_storage_info()

    output_path, local_root = _resolve_run_storage_roots(settings)

    is_restart_run = bool(state.run_info_path)
    if is_restart_run:
        run_name = os.path.basename(os.path.dirname(state.run_info_path))
        logger.info(f"Restarting run. Reusing output folder: {run_name}")
    else:
        custom_label = settings.run.output_run_name
        region = settings.run.region
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"pilates-run--{region}--{custom_label}--{timestamp}"
        logger.info(f"Starting fresh run. Creating new output folder: {run_name}")
    scenario_id = _resolve_scenario_id(settings)
    run_seed = _resolve_seed(settings)
    logger.info(
        "Resolved run tagging metadata: scenario_id=%s seed=%s",
        scenario_id,
        run_seed if run_seed is not None else "n/a",
    )

    archive_run_dir = os.path.join(output_path, run_name)
    local_run_dir = os.path.join(local_root, run_name)
    logger.info(
        "Run storage topology resolved: archive_run_dir=%s local_run_dir=%s",
        archive_run_dir,
        local_run_dir,
    )
    _set_run_failure_context(
        archive_run_dir=archive_run_dir,
        local_run_dir=local_run_dir,
    )
    os.makedirs(local_run_dir, exist_ok=True)
    if archive_run_dir != local_run_dir:
        os.makedirs(archive_run_dir, exist_ok=True)

    _configure_run_storage_environment(
        archive_run_dir=archive_run_dir,
        local_run_dir=local_run_dir,
        enable_archive_copy=settings.run.enable_archive_copy,
    )

    project_root_abs = str(Path(__file__).resolve().parents[2])

    local_consist_db_path, archive_consist_db_path = resolve_consist_db_paths(
        settings=settings,
        local_run_dir=local_run_dir,
        archive_run_dir=archive_run_dir,
    )
    if local_consist_db_path:
        os.makedirs(os.path.dirname(local_consist_db_path), exist_ok=True)
    restored_from_snapshot = restore_local_consist_db_from_snapshot(
        settings=settings,
        local_db_path=local_consist_db_path,
        archive_run_dir=archive_run_dir,
    )
    if not restored_from_snapshot:
        seed_local_consist_db_from_shared(
            settings=settings,
            local_db_path=local_consist_db_path,
            shared_db_path=settings.shared.database.path,
        )
    logger.info(
        "Initializing Consist Tracker in %s (db_path=%s)",
        archive_run_dir,
        local_consist_db_path,
    )

    cache_epoch = _resolve_cache_epoch(settings)

    tracker = cr.create_tracker(
        settings=settings,
        run_dir=archive_run_dir,
        db_path=local_consist_db_path,
        cache_epoch=cache_epoch,
        # PILATES roots the tracker at the archive run dir but rematerializes
        # restart/bootstrap outputs into the mounted local workspace root.
        allow_external_paths=True,
        mounts={
            "inputs": project_root_abs,
            "workspace": local_run_dir,
            "scratch": str(Path(output_path).resolve()),
        },
        project_root=project_root_abs,
        schemas=_get_consist_schemas(),
    )
    if tracker is None:
        raise RuntimeError(
            "Consist tracker initialization failed (received noop/invalid tracker). "
            "Check earlier Consist logs for tracker creation errors, often caused by "
            "a PILATES/Consist API mismatch."
        )
    run_event_context = RunNotificationContext.from_env(
        run_name=run_name,
        scenario_id=scenario_id,
        seed=run_seed,
        archive_run_dir=archive_run_dir,
        local_run_dir=local_run_dir,
        settings_file=settings.settings_file,
    )
    register_consist_run_notification_hooks(tracker, context=run_event_context)
    register_consist_run_publishers(tracker, context=run_event_context)
    snapshot_manager = ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=local_consist_db_path,
        archive_run_dir=archive_run_dir,
    )

    with tracker.trace(
        name="workspace_setup",
        model="pilates_orchestrator",
        tags=[f"scenario_id:{scenario_id}"],
    ):
        workspace = Workspace(
            settings,
            local_root,
            folder_name=run_name,
        )
    archive_state_path = os.path.join(archive_run_dir, "run_state.yaml")
    local_state_path = os.path.join(workspace.full_path, "run_state.yaml")
    _set_run_failure_context(archive_state_path=archive_state_path)
    if is_restart_run:
        _enforce_resume_rewind_guardrail(
            state=state,
            archive_state_path=archive_state_path,
            allow_rewind_resume=settings.allow_rewind_resume,
        )
    state.file_loc = archive_state_path
    state.mirror_file_loc = local_state_path
    if state.run_info_path != archive_state_path:
        state.set_run_info_path(archive_state_path)
    runtime_context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )

    return PreparedRunContext(
        settings=settings,
        state=state,
        surface=surface,
        workspace=workspace,
        runtime_context=runtime_context,
        tracker=tracker,
        snapshot_manager=snapshot_manager,
        run_name=run_name,
        scenario_id=scenario_id,
        seed=run_seed,
        cache_epoch=cache_epoch,
        is_restart_run=is_restart_run,
        archive_run_dir=archive_run_dir,
        local_run_dir=local_run_dir,
        archive_state_path=archive_state_path,
        local_state_path=local_state_path,
        local_consist_db_path=local_consist_db_path,
        archive_consist_db_path=archive_consist_db_path,
    )


def _run_bootstrap_sequence(prepared: PreparedRunContext) -> Optional[Dict[str, Any]]:
    """
    Run restart preflight, bootstrap initialization, and post-bootstrap checks.

    This keeps `main()` focused on the top-level lifecycle while preserving the
    launcher-level wrapper seams used by tests and runtime integrations.
    """
    settings = prepared.settings
    state = prepared.state
    surface = prepared.surface
    workspace = prepared.workspace
    tracker = prepared.tracker
    scenario_id = prepared.scenario_id
    run_seed = prepared.seed
    is_restart_run = prepared.is_restart_run

    if state.data_initialized:
        restart_missing_artifacts_initial = _find_missing_restart_local_artifacts(
            settings=settings,
            state=state,
            workspace=workspace,
            surface=surface,
        )
        restart_runtime.log_prebootstrap_missing_artifacts(
            restart_missing_artifacts_initial,
            surface=surface,
        )

    cr.set_tracker(tracker)
    bootstrap_result: Optional[Dict[str, Any]] = None
    should_run_bootstrap = is_restart_run or not state.data_initialized
    if should_run_bootstrap:
        if is_restart_run and state.data_initialized:
            logger.info(
                "Running bootstrap pre-scenario hydration for restart. "
                "bootstrap re-hydrates workspace invariants through the normal "
                "cached run path before restart frontier hydration runs inside "
                "the scenario context."
            )
        else:
            logger.info("Running bootstrap initialization phase.")
        bootstrap_result = run_bootstrap_phase(
            tracker=tracker,
            settings=settings,
            state=state,
            workspace=workspace,
            scenario_id=scenario_id,
            seed=run_seed,
            surface=surface,
        )
        _assert_bootstrap_output_invariant(bootstrap_result)
        if not state.data_initialized:
            state.set_data_initialized(True)
    else:
        logger.info("Restarting from a previous state. Skipping bootstrap phase.")
    bootstrap_runtime.log_bootstrap_result_summary(bootstrap_result, log=logger)

    if is_restart_run:
        restart_missing_artifacts_after_bootstrap = _find_missing_restart_local_artifacts(
            settings=settings,
            state=state,
            workspace=workspace,
            surface=surface,
        )
        restart_runtime.enforce_postbootstrap_missing_artifacts(
            restart_missing_artifacts_after_bootstrap,
            settings=settings,
        )

    return bootstrap_result


def main(
    *,
    settings: Optional[PilatesConfig] = None,
    state: Optional[WorkflowState] = None,
    clear_failure_context: bool = True,
):
    """
    Run the PILATES simulation lifecycle using the Consist Scenario API.

    This workflow coordinates multiple land use and transportation microsimulation models
    across a multi-year planning horizon:

    1. **Initialization**: Copy immutable input data to mutable workspace
    2. **Land Use Forecasting**: UrbanSim predicts demographic/economic changes
    3. **Vehicle Ownership**: ATLAS models vehicle fleet evolution
    4. **Supply/Demand Loop**: Iterates between activity demand (ActivitySim) and
       traffic assignment (BEAM) until convergence
    5. **Post-Processing**: Validation and output generation

    Architecture:
    - **Consist Scenario**: Manages caching of expensive computations and provenance logging
    - **Coupler**: Passes artifacts (outputs) between models via `scenario.coupler`
    - **Workflow step builders**: Encapsulate model-specific execution logic

    Caching Strategy:
    - ActivitySim compilation: Cached across iterations (inputs unchanged = skip compile)
    - Model outputs: Cached per iteration (convergence check)
    - Bootstrap: pre-scenario cached run with replay-hydrated declared output paths
    - Restart: default path is scenario replay plus cache hits; legacy hydration helpers are manual tooling only
    """
    prepared = _prepare_run_context(
        settings=settings,
        state=state,
        clear_failure_context=clear_failure_context,
    )
    settings = prepared.settings
    state = prepared.state
    surface = prepared.surface
    workspace = prepared.workspace
    runtime_context = prepared.runtime_context
    tracker = prepared.tracker
    snapshot_manager = prepared.snapshot_manager
    run_name = prepared.run_name
    scenario_id = prepared.scenario_id
    run_seed = prepared.seed
    cache_epoch = prepared.cache_epoch
    is_restart_run = prepared.is_restart_run
    archive_run_dir = prepared.archive_run_dir
    local_run_dir = prepared.local_run_dir
    archive_state_path = prepared.archive_state_path
    local_consist_db_path = prepared.local_consist_db_path
    archive_consist_db_path = prepared.archive_consist_db_path

    emit_consist_audit_event(
        workspace=workspace,
        event_type="run_context",
        scenario_id=scenario_id,
        seed=run_seed,
        settings_file=settings.settings_file,
        run_name=run_name,
        workspace_root=workspace.full_path,
        local_run_dir=local_run_dir,
        archive_run_dir=archive_run_dir,
        archive_state_path=archive_state_path if is_restart_run else None,
        restart_run=is_restart_run,
        data_initialized=bool(state.data_initialized),
        bootstrap_cache_enabled=bootstrap_runtime.is_bootstrap_cache_enabled(settings),
    )

    # 5. BOOTSTRAP PHASE (PRE-SCENARIO)
    # Initialization runs before entering scenario step execution so bootstrap
    # lifecycle can evolve independently from normal model steps.
    _run_bootstrap_sequence(prepared)

    # 6. START SCENARIO CONTEXT
    # The scenario context is where all model execution happens. Each step runs inside
    # scenario.run(), which handles:
    #   - Caching checks (skip if inputs identical to previous run)
    #   - Provenance logging (inputs, outputs, dependencies)
    #   - Coupler coordination (step outputs → coupler → next step inputs)
    # The coupler is a shared dict-like object for passing artifacts between steps.
    scenario_contract = _build_scenario_runtime_contract(
        settings=settings,
        state=state,
        workspace=workspace,
        scenario_id=scenario_id,
        seed=run_seed,
        cache_epoch=cache_epoch,
        surface=surface,
    )
    scenario_kwargs = scenario_contract["scenario_kwargs"]
    schema_steps_all = scenario_contract["schema_steps_all"]
    schema_steps_enabled = scenario_contract["schema_steps_enabled"]
    coupler_schema = scenario_contract["coupler_schema"]
    required_output_keys = scenario_contract["required_output_keys"]

    preview_count = 25
    logger.info(
        "Scenario output contract: declared_keys=%d require_outputs=%d "
        "(enabled_steps=%d/%d). Preview: %s",
        len(coupler_schema),
        len(required_output_keys),
        len(schema_steps_enabled),
        len(schema_steps_all),
        required_output_keys[:preview_count],
    )
    logger.info(
        "Scenario Consist defaults: step_tags=%s step_facet=%s",
        scenario_kwargs.get("step_tags"),
        scenario_kwargs.get("step_facet"),
    )
    scenario_tags = _merge_tag_list(
        ["pilates_simulation"],
        [f"scenario_id:{scenario_id}"]
        + ([f"seed:{run_seed}"] if run_seed is not None else []),
    )
    try:
        with cr.scenario(
            run_name,
            tracker=tracker,
            tags=scenario_tags,
            model="pilates_orchestrator",
            **scenario_kwargs,
        ) as scenario:
            tagged_scenario = _ScenarioParentLinkProxy(scenario)
            coupler = tagged_scenario.coupler
            coupler.declare_outputs(
                *coupler_schema.keys(),
                warn_undefined=True,
                description=coupler_schema,
            )
            bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
                settings=settings,
                state=state,
                workspace=workspace,
                coupler=coupler,
            )
            if is_restart_run:
                seed_supply_demand_parent_run_ids_for_resume(
                    scenario=tagged_scenario,
                    workspace=workspace,
                    state=state,
                )
            if is_restart_run and state.data_initialized:
                logger.info(
                    "Restart replay mode active: skipping bespoke restart "
                    "hydration and relying on scenario replay plus Consist cache hits."
                )
                emit_consist_audit_event(
                    workspace=workspace,
                    event_type="restart_hydration",
                    frontier_stage=None,
                    frontier_step=None,
                    success=True,
                    hydrated_keys=[],
                    missing_keys=[],
                    producer_steps_by_key={},
                    fallback_reason="replay_mode",
                    rewind_restore=False,
                    overlay_root=None,
                )

            # 7. MAIN WORKFLOW LOOP
            # Iterates through forecast years. For each year, runs sequential stages:
            # A (Land Use) → B (Vehicle Ownership) → C (Supply/Demand Loop) → D (Post-Processing)
            #
            # Step Pattern (used for all stages):
            #   1. build_*_inputs(...)      - Collect inputs from previous outputs + coupler
            #   2. log_inputs(...)          - Log for provenance
            #   3. build_*_outputs(...)     - Declare what we expect to produce
            #   4. make_*_step(...)         - Create step function with coupler refs
            #   5. build_step_config(...)   - Create config (year, iteration, inputs, outputs, kwargs)
            #   6. scenario.run(...)        - Execute via Consist (handles caching + provenance)
            #
            for year in state:
                formatted_print(f"STARTING YEAR {year}")
                logger.info(
                    "[launcher] starting year=%s run_id=%s",
                    year,
                    cr.current_run_id(),
                )
                land_use_handoff = LandUseToSupplyDemandHandoff()
                outputs_holder_year = StepOutputsHolder()

                if state.should_run(WorkflowState.Stage.land_use):
                    land_use_handoff = run_land_use_stage(
                        scenario=cast(ScenarioWithCoupler, tagged_scenario),
                        coupler=coupler,
                        year=year,
                        outputs_holder_year=outputs_holder_year,
                        context=runtime_context,
                    )
                    state.complete_step(WorkflowState.Stage.land_use)
                    snapshot_manager.maybe_snapshot_interval(
                        reason=f"after_land_use_y{year}"
                    )

                if state.should_run(WorkflowState.Stage.vehicle_ownership_model):
                    formatted_print(
                        f"VEHICLE OWNERSHIP MODEL (ATLAS) FOR YEAR {state.forecast_year}"
                    )
                    run_vehicle_ownership_stage(
                        scenario=cast(ScenarioWithCoupler, tagged_scenario),
                        coupler=coupler,
                        year=year,
                        build_atlas_static_inputs_fallback=build_atlas_static_inputs_fallback,
                        context=runtime_context,
                    )
                    state.complete_step(WorkflowState.Stage.vehicle_ownership_model)
                    snapshot_manager.maybe_snapshot_interval(
                        reason=f"after_vehicle_ownership_y{year}"
                    )

                if state.should_run(WorkflowState.Stage.supply_demand_loop):
                    run_supply_demand_stage(
                        scenario=cast(ScenarioWithCoupler, tagged_scenario),
                        coupler=coupler,
                        year=year,
                        handoff=land_use_handoff,
                        build_manifest_path=build_manifest_path,
                        on_iteration_boundary=(
                            lambda iteration, y=year: (
                                snapshot_manager.on_outer_iteration_boundary(
                                    year=y,
                                    iteration=iteration,
                                )
                            )
                        ),
                        context=runtime_context,
                    )
                    snapshot_manager.maybe_snapshot_interval(
                        reason=f"after_supply_demand_y{year}"
                    )

                if state.should_run(WorkflowState.Stage.postprocessing):
                    formatted_print("POST-PROCESSING")
                    run_postprocessing_stage(
                        scenario=cast(ScenarioWithCoupler, tagged_scenario),
                        coupler=coupler,
                        year=year,
                        context=runtime_context,
                    )
                    state.complete_step(WorkflowState.Stage.postprocessing)
                    snapshot_manager.maybe_snapshot_interval(
                        reason=f"after_postprocessing_y{year}"
                    )
                snapshot_manager.maybe_snapshot_interval(
                    reason=f"year_boundary_y{year}"
                )

        formatted_print("SIMULATION COMPLETE")
        logger.info("[Main] Simulation complete.")
    finally:
        snapshot_ok = snapshot_manager.final_snapshot()
        flush_archive_queue(timeout=300)
        emit_artifact_lifecycle_audit_event(
            workspace=workspace,
            event_type="final_shutdown",
            snapshot_ok=snapshot_ok,
            archive_run_dir=archive_run_dir,
            local_run_dir=local_run_dir,
            local_to_scratch_recovery_roots_written=0,
        )
        stop_archive_worker(timeout=30)
        if not snapshot_ok:
            mirror_consist_db_to_archive(local_consist_db_path, archive_consist_db_path)
