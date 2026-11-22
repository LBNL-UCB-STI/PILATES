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
import signal
import sys
from pathlib import Path
from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    cast,
    runtime_checkable,
)

from consist.models.run import Run
from pilates.config import PilatesConfig
from pilates.workspace import Workspace
from pilates.utils.formatting import formatted_print
from pilates.utils.io import parse_args_and_settings
from pilates.utils import consist_runtime as cr
from pilates.workflows.consist_metadata import (
    build_scenario_consist_kwargs,
)
from pilates.utils.consist_db_snapshot import (
    ConsistDbSnapshotManager,
    mirror_consist_db_to_archive,
    resolve_consist_db_paths,
    restore_local_consist_db_from_snapshot,
    seed_local_consist_db_from_shared,
)
from pilates.activitysim.admission import (
    has_initial_skims_admission_policy,
    preflight_activitysim_initial_skims_admission,
)
from pilates.urbansim.admission import (
    preflight_bootstrap_urbansim_datastore_admission,
)
from pilates.utils.admission_helpers import extract_admission_reports
from pilates.workflows.orchestrator import run_pilates
from pilates.workflows.state_helpers import requires_prior_beam_skim_handoff
from pilates.utils.consist_types import ScenarioWithCoupler
from pilates.runtime.context import WorkflowRuntimeContext
from pilates.runtime.failure_hints import (
    RUN_FAILURE_CONTEXT,
    clear_run_failure_context,
    format_hpc_restart_command,
    format_restart_command,
    log_promotion_instructions_on_success,
    log_restart_instructions_on_failure,
    set_run_failure_context,
)
from pilates.runtime import scenario_runtime
from pilates.runtime.run_notifications import (
    ConsistRunNotifier,
    RunNotificationContext,
    register_consist_run_notification_hooks,
)
from pilates.runtime.run_publishers import (
    ConsistRunPublisher,
    register_consist_run_publishers,
)
from pilates.runtime.storage_probe import log_local_storage_info_if_enabled
from pilates.workflows._profile import ensure_runtime_flags_initialized
from pilates.workflows.state import WorkflowState

warnings.simplefilter(action="ignore", category=FutureWarning)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("pyhocon").setLevel(logging.WARNING)
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
    local_consist_db_path: Optional[str]
    archive_consist_db_path: Optional[str]
    run_notifier: Optional[ConsistRunNotifier]
    run_publisher: Optional[ConsistRunPublisher]


@runtime_checkable
class _TrackerWithDb(Protocol):
    db: Any


def query_expected_prefix_steps(tracker: Any, scenario_name: str) -> int:
    """
    Query tracker DB for the count of completed child step runs from the latest run of scenario_name.
    """
    if tracker is None:
        return 0

    if not isinstance(tracker, _TrackerWithDb):
        return 0

    db = tracker.db
    if db is None:
        return 0
    try:
        con = db.connection
        step_rows = con.execute(
            """
            SELECT count(*) FROM run 
            WHERE parent_run_id = (
                SELECT id FROM run 
                WHERE model_name = 'pilates_orchestrator' 
                  AND status = 'completed' 
                  AND id LIKE ? 
                ORDER BY started_at DESC 
                LIMIT 1
            ) 
            AND status = 'completed'
            """,
            [f"{scenario_name}%"],
        ).fetchone()
        return int(step_rows[0]) if step_rows else 0
    except Exception as exc:
        logger.debug("Failed to query expected_prefix_steps: %s", exc)
        return 0


_format_restart_command = format_restart_command
_format_hpc_restart_command = format_hpc_restart_command
_resolve_cache_epoch = scenario_runtime.resolve_cache_epoch
_build_schema_steps = scenario_runtime.build_schema_steps
_filter_schema_steps_for_enabled_models = (
    scenario_runtime.filter_schema_steps_for_enabled_models
)
_log_local_storage_info = log_local_storage_info_if_enabled


def _log_restart_instructions_on_failure() -> None:
    log_restart_instructions_on_failure(logger=logger)


def _snapshot_activitysim_boundary(
    snapshot_manager: ConsistDbSnapshotManager,
    *,
    year: int,
    iteration: int,
) -> bool:
    """Persist the completed ActivitySim handoff before BEAM begins."""

    return snapshot_manager.snapshot(
        reason=f"after_activitysim_y{year}_i{iteration}",
        checkpoint=True,
    )


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
) -> Dict[str, Any]:
    return scenario_runtime.build_scenario_runtime_contract(
        settings=settings,
        state=state,
        workspace=workspace,
        scenario_id=scenario_id,
        seed=seed,
        cache_epoch=cache_epoch,
        build_scenario_consist_kwargs_fn=build_scenario_consist_kwargs,
        build_schema_steps_fn=_build_schema_steps,
        filter_schema_steps_for_enabled_models_fn=_filter_schema_steps_for_enabled_models,
        merge_epoch_facet_fn=scenario_runtime.merge_epoch_facet,
        scenario_name_template=_SCENARIO_NAME_TEMPLATE,
    )


def _is_at_initial_workflow_boundary(state: Any) -> bool:
    """Return True when state is positioned at the first enabled major stage of start_year."""
    if not isinstance(state, WorkflowState):
        return False
    if not state.is_start_year() or int(state.iteration or 0) != 0:
        return False
    first_enabled = next(
        (stage for stage in state.major_stage_order if state.is_enabled(stage)),
        None,
    )
    return first_enabled is not None and state.current_major_stage == first_enabled


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
    set_run_failure_context(settings=settings, state=state)

    _log_local_storage_info()

    output_path, local_root = _resolve_run_storage_roots(settings)

    runtime_obj = getattr(settings, "runtime", None)
    runtime_options = getattr(runtime_obj, "options", None)
    explicit_run_dir = getattr(
        runtime_options, "run_dir", getattr(settings, "run_dir", None)
    )
    timestamp_opt = getattr(
        runtime_options, "timestamp", getattr(settings, "timestamp", False)
    )

    is_restart_run = False
    if explicit_run_dir:
        run_dir_path = Path(explicit_run_dir)
        run_name = run_dir_path.name
        is_restart_run = run_dir_path.exists()
        if is_restart_run:
            logger.info(f"Resuming run from explicit run folder: {run_name}")
        else:
            logger.info(f"Starting run with explicit run folder: {run_name}")
    elif is_restart_run:
        run_name = os.path.basename(os.path.dirname(state.run_info_path))
        logger.info(f"Restarting run. Reusing output folder: {run_name}")
    else:
        custom_label = settings.run.output_run_name
        region = settings.run.region
        if timestamp_opt:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"pilates-run--{region}--{custom_label}--{timestamp}"
        else:
            run_name = f"pilates-run--{region}--{custom_label}"
        candidate_archive = os.path.join(output_path, run_name)
        if os.path.exists(candidate_archive):
            is_restart_run = True
            logger.info(f"Resuming existing run. Reusing output folder: {run_name}")
        else:
            logger.info(f"Starting fresh run. Creating new output folder: {run_name}")
    scenario_id = scenario_runtime.resolve_scenario_id(settings)
    run_seed = scenario_runtime.resolve_seed(settings)
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
    set_run_failure_context(
        archive_run_dir=archive_run_dir,
        local_run_dir=local_run_dir,
    )
    os.makedirs(local_run_dir, exist_ok=True)
    if archive_run_dir != local_run_dir:
        os.makedirs(archive_run_dir, exist_ok=True)
    dlt_data_dir = os.environ.setdefault(
        "DLT_DATA_DIR",
        os.path.join(local_run_dir, ".dlt"),
    )
    os.makedirs(dlt_data_dir, exist_ok=True)

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
            "beam_input": os.path.join(local_run_dir, "beam", "input"),
            "scratch": str(Path(output_path).resolve()),
        },
        archive_mounts={
            "workspace": ".",
            "beam_input": "beam/input",
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
    run_notifier = register_consist_run_notification_hooks(
        tracker,
        context=run_event_context,
    )
    run_publisher = register_consist_run_publishers(
        tracker,
        context=run_event_context,
    )
    snapshot_manager = ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=local_consist_db_path,
        archive_run_dir=archive_run_dir,
        archive_db_path=archive_consist_db_path,
    )

    expected_prefix_steps = (
        query_expected_prefix_steps(tracker, run_name) if is_restart_run else 0
    )

    on_archive_commit = snapshot_manager.snapshot_after_archive_step

    with tracker.trace(
        name="workspace_setup",
        model="pilates_orchestrator",
        tags=[f"scenario_id:{scenario_id}"],
    ):
        workspace = Workspace(
            settings,
            local_root,
            folder_name=run_name,
            archive_run_dir=archive_run_dir,
            on_archive_commit=on_archive_commit,
            expected_prefix_steps=expected_prefix_steps,
        )
        if isinstance(workspace, Workspace):
            workspace.on_archive_commit = on_archive_commit
            workspace.expected_prefix_steps = expected_prefix_steps
    runtime_context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        on_archive_commit=on_archive_commit,
        expected_prefix_steps=expected_prefix_steps,
    )
    return PreparedRunContext(
        settings=settings,
        state=state,
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
        local_consist_db_path=local_consist_db_path,
        archive_consist_db_path=archive_consist_db_path,
        run_notifier=run_notifier,
        run_publisher=run_publisher,
    )


def _emit_pre_scenario_failure(
    prepared: PreparedRunContext,
    error: Exception,
) -> None:
    """Emit a synthetic scenario-level failure for launcher failures before scenario start."""
    failure_run = Run(
        id=prepared.run_name,
        model_name="pilates_orchestrator",
        tags=["scenario_header", "launcher_failure"],
        status="failed",
        description="PILATES launcher failed before entering the scenario context.",
        phase="launcher",
        meta={"launcher_failure": True},
    )
    if prepared.run_notifier is not None:
        prepared.run_notifier.on_run_failed(failure_run, error)
    if prepared.run_publisher is not None:
        prepared.run_publisher.on_run_failed(failure_run, error)


def main(
    *,
    settings: Optional[PilatesConfig] = None,
    state: Optional[WorkflowState] = None,
    clear_failure_context: bool = True,
):
    """
    Run the PILATES simulation lifecycle using the Consist Scenario API.
    """
    prepared = _prepare_run_context(
        settings=settings,
        state=state,
        clear_failure_context=clear_failure_context,
    )
    settings = prepared.settings
    state = prepared.state
    workspace = prepared.workspace
    tracker = prepared.tracker
    snapshot_manager = prepared.snapshot_manager
    run_name = prepared.run_name
    scenario_id = prepared.scenario_id
    run_seed = prepared.seed
    cache_epoch = prepared.cache_epoch
    is_restart_run = prepared.is_restart_run
    archive_run_dir = prepared.archive_run_dir
    local_run_dir = prepared.local_run_dir
    local_consist_db_path = prepared.local_consist_db_path
    archive_consist_db_path = prepared.archive_consist_db_path
    run_succeeded = False

    original_signal_handlers: Dict[int, Any] = {}

    def _graceful_exit_handler(signum: int, frame: Any) -> None:
        try:
            sig_name = signal.Signals(signum).name
        except (ValueError, AttributeError):
            sig_name = str(signum)
        logger.warning(
            "PILATES received signal %d (%s); initiating graceful shutdown...",
            signum,
            sig_name,
        )
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            original_signal_handlers[sig] = signal.signal(sig, _graceful_exit_handler)
        except (ValueError, OSError, AttributeError):
            pass

    try:
        cr.set_tracker(tracker)

        # 5. START SCENARIO CONTEXT
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
        )
        scenario_kwargs = scenario_contract["scenario_kwargs"]
        schema_steps_all = scenario_contract["schema_steps_all"]
        schema_steps_enabled = scenario_contract["schema_steps_enabled"]
        required_output_keys = scenario_contract["required_output_keys"]

        preview_count = 25
        logger.info(
            "Scenario output contract: require_outputs=%d "
            "(enabled_steps=%d/%d). Preview: %s",
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
        scenario_tags = scenario_runtime.merge_tag_list(
            ["pilates_simulation"],
            [f"scenario_id:{scenario_id}"]
            + ([f"seed:{run_seed}"] if run_seed is not None else []),
        )
    except Exception as exc:
        _emit_pre_scenario_failure(prepared, exc)
        raise

    try:
        with cr.scenario(
            run_name,
            tracker=tracker,
            tags=scenario_tags,
            model="pilates_orchestrator",
            **scenario_kwargs,
        ) as scenario:
            # ponytail: bootstrap_initialization is retired; admit initial inputs
            # directly on fresh start or restart at the initial workflow boundary.
            if not is_restart_run or _is_at_initial_workflow_boundary(state):
                preflight_bootstrap_urbansim_datastore_admission(
                    settings=settings,
                    metadata_logger=tracker,
                    workspace_path=Path(workspace.full_path),
                    report_dir=Path(archive_run_dir),
                    existing_admission_reports=extract_admission_reports(
                        cr.current_run()
                    ),
                )
                if has_initial_skims_admission_policy(
                    settings
                ) and not requires_prior_beam_skim_handoff(
                    settings=settings, state=state
                ):
                    preflight_activitysim_initial_skims_admission(
                        settings=settings,
                        metadata_logger=tracker,
                        tracker=tracker,
                        workspace_path=Path(workspace.full_path),
                        report_dir=Path(archive_run_dir),
                        existing_admission_reports=extract_admission_reports(
                            cr.current_run()
                        ),
                    )
            run_pilates(
                scenario=cast(ScenarioWithCoupler, scenario),
                settings=settings,
                state=state,
                workspace=workspace,
                snapshot_manager=snapshot_manager,
                on_activity_demand_boundary=(
                    lambda y, iteration: _snapshot_activitysim_boundary(
                        snapshot_manager,
                        year=y,
                        iteration=iteration,
                    )
                ),
            )

        formatted_print("SIMULATION COMPLETE")
        logger.info("[Main] Simulation complete.")
        run_succeeded = True
    finally:
        try:
            try:
                snapshot_ok = snapshot_manager.final_snapshot()
                if not snapshot_ok:
                    mirror_consist_db_to_archive(
                        local_consist_db_path,
                        archive_consist_db_path,
                        tracker=tracker,
                    )
                if run_succeeded and snapshot_ok:
                    log_promotion_instructions_on_success(
                        logger=logger,
                        context=RUN_FAILURE_CONTEXT,
                    )
            finally:
                cr.dispose_tracker(tracker)
        finally:
            for sig, handler in original_signal_handlers.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass
