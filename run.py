"""
run.py

Main entrypoint and workflow orchestrator for PILATES simulations.

This module:
- Parses settings and initializes workflow state.
- Initializes the Consist Tracker and Scenario Context.
- Executes the multi-stage simulation loop using the Scenario/Step API.
- Manages provenance for the critical "Data Initialization" step to link
  immutable inputs to the mutable workspace.
"""

import warnings
from datetime import datetime
import os
import logging
import sys
import shutil
import socket
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List

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
from pilates.utils.coupler_helpers import flush_archive_queue, stop_archive_worker
from pilates.workflows.coupler_schema import build_coupler_schema
from pilates.workflows.catalog import schema_step_names, enabled_schema_step_models
from pilates.workflows.stages import (
    run_land_use_stage,
    run_postprocessing_stage,
    run_supply_demand_stage,
    run_vehicle_ownership_stage,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
from workflow_state import WorkflowState

from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_compile_step,
    make_activitysim_postprocess_step,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
    make_atlas_postprocess_step,
    make_atlas_preprocess_step,
    make_atlas_run_step,
    make_beam_postprocess_step,
    make_beam_full_skim_step,
    make_beam_preprocess_step,
    make_beam_run_step,
    make_urbansim_postprocess_step,
    make_urbansim_preprocess_step,
    make_urbansim_run_step,
    validate_workflow_step_contracts,
)
from consist.types import CacheOptions

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_SCENARIO_NAME_TEMPLATE = "{func_name}__y{year}__i{iteration}__phase_{phase}"


class _SchemaCoupler:
    """No-op coupler used to construct decorated step callables for schema introspection."""

    def get(self, _key: str, default: Optional[Any] = None) -> Any:
        return default

    def set(self, _key: str, _value: Any) -> None:
        return None

    def update(self, _mapping: Dict[str, Any]) -> None:
        return None

    def view(self, _namespace: str) -> "_SchemaCoupler":
        return self

    def declare_outputs(self, *args: Any, **kwargs: Any) -> None:
        return None


def _resolve_cache_epoch(settings: Any) -> int:
    value = getattr(getattr(settings, "run", None), "cache_epoch", 1)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _build_schema_steps() -> List[Callable[..., Any]]:
    coupler = _SchemaCoupler()
    outputs_holder = StepOutputsHolder()
    step_factories: Dict[str, Callable[..., Any]] = {
        "urbansim_preprocess": make_urbansim_preprocess_step,
        "urbansim_run": make_urbansim_run_step,
        "urbansim_postprocess": make_urbansim_postprocess_step,
        "atlas_preprocess": make_atlas_preprocess_step,
        "atlas_run": make_atlas_run_step,
        "atlas_postprocess": make_atlas_postprocess_step,
        "activitysim_preprocess": make_activitysim_preprocess_step,
        "activitysim_compile": make_activitysim_compile_step,
        "activitysim_run": make_activitysim_run_step,
        "activitysim_postprocess": make_activitysim_postprocess_step,
        "beam_preprocess": make_beam_preprocess_step,
        "beam_run": make_beam_run_step,
        "beam_postprocess": make_beam_postprocess_step,
        "beam_full_skim": make_beam_full_skim_step,
    }
    ordered_steps = schema_step_names()
    missing_factories = [name for name in ordered_steps if name not in step_factories]
    if missing_factories:
        raise RuntimeError(
            "Missing schema step factories for: " + ", ".join(missing_factories)
        )
    return [
        step_factories[step_name](coupler=coupler, outputs_holder=outputs_holder)
        for step_name in ordered_steps
    ]


def _is_model_enabled(settings: Any, *, flag_attr: str, model_attr: str) -> bool:
    """
    Resolve whether a workflow model is enabled.

    Prefers precomputed flags from ``parse_args_and_settings`` and falls back to
    ``settings.run.models`` when flags are not present.
    """
    explicit_flag = getattr(settings, flag_attr, None)
    if explicit_flag is not None:
        return bool(explicit_flag)
    run_cfg = getattr(settings, "run", None)
    model_cfg = getattr(run_cfg, "models", None) if run_cfg is not None else None
    return bool(getattr(model_cfg, model_attr, None))


def _filter_schema_steps_for_enabled_models(
    steps: List[Callable[..., Any]],
    settings: Any,
    *,
    include_optional: bool = True,
) -> List[Callable[..., Any]]:
    """
    Keep only step definitions for models enabled in the active run settings.

    Parameters
    ----------
    steps : list of callables
        Step functions decorated with ``@define_step`` metadata.
    settings : Any
        Runtime settings object used to resolve enabled model flags.
    include_optional : bool, default True
        Whether optional steps (currently ``beam_full_skim``) should be included.
    """
    enabled_models = enabled_schema_step_models(
        settings,
        is_model_enabled=_is_model_enabled,
        include_optional=include_optional,
    )

    filtered: List[Callable[..., Any]] = []
    for step_func in steps:
        meta = getattr(step_func, "__consist_step__", None)
        model_name = getattr(meta, "model", "") if meta is not None else ""
        if model_name not in enabled_models:
            continue
        filtered.append(step_func)
    return filtered


def _get_consist_schemas() -> Optional[list[type]]:
    try:
        from pilates.database.schema.registry import get_consist_schemas

        return get_consist_schemas()
    except Exception:
        return None


def build_manifest_path(workspace: Workspace, year: int, iteration: int) -> Path:
    return (
        Path(workspace.full_path)
        / ".workflow"
        / f"year_{year}_iteration_{iteration}.yaml"
    )


def build_atlas_static_inputs_fallback(workspace: Workspace) -> Dict[str, str]:
    """
    Enumerate static ATLAS inputs from the mutable input directory.

    This fallback is used when Initialization was skipped (e.g., restart) and the
    in-memory RecordStore of copied inputs is unavailable. It may include files
    produced by prior ATLAS preprocess runs.
    """
    atlas_input_dir = workspace.get_atlas_mutable_input_dir()
    if not os.path.exists(atlas_input_dir):
        return {}
    inputs: Dict[str, str] = {}
    for root, _, files in os.walk(atlas_input_dir):
        for filename in sorted(files):
            path = os.path.join(root, filename)
            relpath = os.path.relpath(path, atlas_input_dir)
            key = f"atlas_static_{relpath.replace(os.sep, '__')}"
            inputs.setdefault(key, path)
    return inputs


def _read_mount_table() -> Dict[str, str]:
    mounts: Dict[str, str] = {}
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3:
                    mountpoint = parts[1]
                    fstype = parts[2]
                    mounts[mountpoint] = fstype
    except OSError:
        return {}
    return mounts


def _mount_for_path(path: str, mounts: Dict[str, str]) -> str:
    path = os.path.realpath(path)
    best_match = ""
    for mountpoint in mounts:
        if path == mountpoint or path.startswith(mountpoint.rstrip("/") + "/"):
            if len(mountpoint) > len(best_match):
                best_match = mountpoint
    return best_match


def _format_bytes(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}EiB"


def _log_local_storage_info() -> None:
    mounts = _read_mount_table()
    hostname = socket.gethostname()
    job_id = os.environ.get("SLURM_JOB_ID")
    node_list = os.environ.get("SLURM_NODELIST")
    logger.info(
        "Storage probe: host=%s job_id=%s nodelist=%s",
        hostname,
        job_id or "n/a",
        node_list or "n/a",
    )

    candidates = []
    for var in ("SLURM_TMPDIR", "TMPDIR", "TMP", "TEMP"):
        value = os.environ.get(var)
        if value:
            candidates.append(value)
    candidates += [
        "/tmp",
        "/var/tmp",
        "/dev/shm",
        "/scratch",
        "/local",
        "/local_scratch",
        "/lscratch",
        "/mnt",
    ]

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if not os.path.exists(path):
            continue
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        mountpoint = _mount_for_path(path, mounts)
        fstype = mounts.get(mountpoint, "unknown")
        logger.info(
            "Storage candidate: path=%s mount=%s fstype=%s free=%s total=%s",
            os.path.realpath(path),
            mountpoint or "unknown",
            fstype,
            _format_bytes(usage.free),
            _format_bytes(usage.total),
        )


def _is_bootstrap_cache_enabled(settings: Any) -> bool:
    run_cfg = getattr(settings, "run", None)
    return bool(getattr(run_cfg, "bootstrap_cache_enabled", True))


def _build_bootstrap_manifest_reference(
    *,
    probe_run_id: Optional[str] = None,
    materialization_run_id: Optional[str] = None,
) -> Dict[str, str]:
    reference: Dict[str, str] = {}
    if probe_run_id:
        reference["probe_run_id"] = probe_run_id
    if materialization_run_id:
        reference["materialization_run_id"] = materialization_run_id
    return reference


def run_bootstrap_phase(
    *,
    tracker: Any,
    settings: Any,
    state: WorkflowState,
    workspace: Workspace,
) -> Dict[str, Any]:
    """
    Execute initialization in a dedicated pre-scenario bootstrap phase.

    Phase 1 behavior:
    - probe cache via tracker.run(...) before scenario starts;
    - if cache hit, force one overwrite run to materialize workspace safely;
    - return cache status plus lightweight artifact summary metadata.
    """
    staged_artifact_summary: Dict[str, Any] = {}

    def _execute_initialization() -> None:
        nonlocal staged_artifact_summary
        init_model = Initialization("initialization", state)
        copied_records = init_model.run(settings, workspace)
        staged_artifact_summary = build_bootstrap_artifact_summary(
            workspace,
            copied_records,
        )

    run_kwargs: Dict[str, Any] = {
        "fn": _execute_initialization,
        "name": "bootstrap_initialization",
        "model": "initialization",
        "year": state.start_year,
        "iteration": 0,
        "phase": "bootstrap",
        "stage": "bootstrap",
        "tags": ["bootstrap", "init"],
        **build_step_consist_kwargs(
            "initialization",
            settings,
            workspace_path=workspace.full_path,
        ),
    }

    if not _is_bootstrap_cache_enabled(settings):
        logger.info("Bootstrap cache disabled; running initialization once.")
        run_result = tracker.run(
            **run_kwargs,
            cache_options=CacheOptions(cache_mode="off"),
        )
        if not staged_artifact_summary:
            staged_artifact_summary = build_bootstrap_artifact_summary(workspace)
        return {
            "bootstrap_cache_hit": False,
            "staged_artifact_summary": staged_artifact_summary,
            "manifest_reference": _build_bootstrap_manifest_reference(
                probe_run_id=getattr(getattr(run_result, "run", None), "id", None)
            ),
        }

    probe_result = tracker.run(**run_kwargs)
    probe_run_id = getattr(getattr(probe_result, "run", None), "id", None)
    cache_hit = bool(getattr(probe_result, "cache_hit", False))

    if cache_hit:
        logger.info(
            "BOOTSTRAP CACHE HIT. Running Phase 1 materialization pass to keep workspace safe."
        )
        materialized_result = tracker.run(
            **run_kwargs,
            cache_options=CacheOptions(cache_mode="overwrite"),
        )
        if not staged_artifact_summary:
            staged_artifact_summary = build_bootstrap_artifact_summary(workspace)
        return {
            "bootstrap_cache_hit": True,
            "staged_artifact_summary": staged_artifact_summary,
            "manifest_reference": _build_bootstrap_manifest_reference(
                probe_run_id=probe_run_id,
                materialization_run_id=getattr(
                    getattr(materialized_result, "run", None), "id", None
                ),
            ),
        }

    logger.info("BOOTSTRAP CACHE MISS. Initialization executed for this workspace.")
    if not staged_artifact_summary:
        staged_artifact_summary = build_bootstrap_artifact_summary(workspace)
    return {
        "bootstrap_cache_hit": False,
        "staged_artifact_summary": staged_artifact_summary,
        "manifest_reference": _build_bootstrap_manifest_reference(
            probe_run_id=probe_run_id
        ),
    }


def _assert_bootstrap_output_invariant(
    bootstrap_result: Optional[Dict[str, Any]],
) -> None:
    """
    Ensure bootstrap produced a non-empty artifact summary before state mutation.
    """
    summary = (
        bootstrap_result.get("staged_artifact_summary")
        if isinstance(bootstrap_result, dict)
        else None
    )
    copied_total = (
        summary.get("copied_records_total") if isinstance(summary, dict) else None
    )
    if isinstance(copied_total, int) and copied_total > 0:
        return

    diagnostics = {
        "bootstrap_result_type": type(bootstrap_result).__name__,
        "bootstrap_cache_hit": (
            bootstrap_result.get("bootstrap_cache_hit")
            if isinstance(bootstrap_result, dict)
            else None
        ),
        "manifest_reference": (
            bootstrap_result.get("manifest_reference")
            if isinstance(bootstrap_result, dict)
            else None
        ),
        "staged_artifact_summary": summary,
    }
    raise RuntimeError(
        "Bootstrap initialization invariant failed: expected "
        "'staged_artifact_summary.copied_records_total' > 0 before setting "
        f"data_initialized=True. diagnostics={diagnostics}"
    )


def main():
    """
    Main entrypoint for PILATES simulation orchestration using Consist Scenario API.

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
    - **StepConfig**: Declarative config for each model step
    - **Step Builders**: Encapsulate model-specific execution logic

    Caching Strategy:
    - ActivitySim compilation: Cached across iterations (inputs unchanged = skip compile)
    - Model outputs: Cached per iteration (convergence check)
    - Restarting: Skips initialization if run_state.yaml exists
    """
    # 1. PARSE SETTINGS AND SET UP WORKFLOW STATE
    settings = parse_args_and_settings()
    state = WorkflowState.from_settings(settings)

    _log_local_storage_info()

    # 2. SETUP PATHS
    output_directory = settings.run.output_directory
    if not output_directory:
        raise ValueError("output_directory not found in config")
    output_path = os.path.realpath(os.path.expandvars(output_directory))
    local_workspace_root = getattr(settings.run, "local_workspace_root", None)
    if local_workspace_root:
        local_root = os.path.realpath(os.path.expandvars(local_workspace_root))
    else:
        local_root = output_path

    # Split run roots:
    # - archive_run_dir (scratch) holds Consist run metadata + archived artifacts
    # - local_run_dir (node-local) holds mutable workspace during execution

    if state.run_info_path:
        run_name = os.path.basename(os.path.dirname(state.run_info_path))
        logger.info(f"Restarting run. Reusing output folder: {run_name}")
    else:
        partial_run_name = settings.run.output_run_name
        run_name = f"{partial_run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        logger.info(f"Starting fresh run. Creating new output folder: {run_name}")

    archive_run_dir = os.path.join(output_path, run_name)
    local_run_dir = os.path.join(local_root, run_name)
    os.makedirs(local_run_dir, exist_ok=True)
    if archive_run_dir != local_run_dir:
        os.makedirs(archive_run_dir, exist_ok=True)

    os.environ["PILATES_LOCAL_RUN_DIR"] = local_run_dir
    os.environ["PILATES_ARCHIVE_RUN_DIR"] = archive_run_dir
    os.environ["PILATES_ENABLE_ARCHIVE_COPY"] = (
        "1" if settings.run.enable_archive_copy else "0"
    )

    # 3. INITIALIZE CONSIST TRACKER
    # Consist provides provenance tracking and computation caching.
    # It is required for PILATES execution.
    #   - Provenance: Full lineage of data transformations (OpenLineage compatible)
    #   - Caching: Skips expensive computations if inputs unchanged
    #   - Coupler: Manages artifact passing between steps
    # Mount Strategy:
    # - 'inputs': The project root. Source files resolve here.
    # - 'workspace': The mutable run dir. Destination files resolve here.
    # NOTE: Do not rely on cwd; production runs may invoke `python run.py` from elsewhere.
    # Use the directory containing `run.py` as the canonical inputs root.
    project_root_abs = str(Path(__file__).resolve().parent)

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
        shared_db_path = getattr(getattr(settings, "shared", None), "database", None)
        seed_local_consist_db_from_shared(
            settings=settings,
            local_db_path=local_consist_db_path,
            shared_db_path=getattr(shared_db_path, "path", None),
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
        mounts={
            "inputs": project_root_abs,  # Immutable Source
            "workspace": local_run_dir,  # Mutable Destination
            "scratch": str(Path(output_path).resolve()),  # For temp files
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
    snapshot_manager = ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=local_consist_db_path,
        archive_run_dir=archive_run_dir,
    )

    # 4. INITIALIZE WORKSPACE
    workspace = Workspace(
        settings,
        local_root,
        folder_name=run_name,
    )
    archive_state_path = os.path.join(archive_run_dir, "run_state.yaml")
    local_state_path = os.path.join(workspace.full_path, "run_state.yaml")
    state.file_loc = archive_state_path
    state.mirror_file_loc = local_state_path
    if state.run_info_path != archive_state_path:
        state.set_run_info_path(archive_state_path)

    # 5. BOOTSTRAP PHASE (PRE-SCENARIO)
    # Initialization runs before entering scenario step execution so bootstrap
    # lifecycle can evolve independently from normal model steps.
    cr.set_tracker(tracker)
    bootstrap_result: Optional[Dict[str, Any]] = None
    if not state.data_initialized:
        logger.info("Running bootstrap initialization phase.")
        bootstrap_result = run_bootstrap_phase(
            tracker=tracker,
            settings=settings,
            state=state,
            workspace=workspace,
        )
        _assert_bootstrap_output_invariant(bootstrap_result)
        state.set_data_initialized(True)
    else:
        logger.info("Restarting from a previous state. Skipping bootstrap phase.")
    if bootstrap_result is not None:
        logger.info(
            "Bootstrap phase complete: cache_hit=%s manifest_ref=%s summary=%s",
            bootstrap_result.get("bootstrap_cache_hit"),
            bootstrap_result.get("manifest_reference"),
            bootstrap_result.get("staged_artifact_summary"),
        )

    # 6. START SCENARIO CONTEXT
    # The scenario context is where all model execution happens. Each step runs inside
    # scenario.run(), which handles:
    #   - Caching checks (skip if inputs identical to previous run)
    #   - Provenance logging (inputs, outputs, dependencies)
    #   - Coupler coordination (step outputs → coupler → next step inputs)
    # The coupler is a shared dict-like object for passing artifacts between steps.
    scenario_kwargs = build_scenario_consist_kwargs(settings)
    scenario_kwargs.setdefault("name_template", _SCENARIO_NAME_TEMPLATE)
    scenario_kwargs.setdefault("cache_epoch", cache_epoch)
    schema_steps_all = _build_schema_steps()
    validate_workflow_step_contracts(declared_steps=schema_steps_all)
    schema_steps_enabled = _filter_schema_steps_for_enabled_models(
        schema_steps_all,
        settings,
        include_optional=True,
    )
    coupler_schema = build_coupler_schema(schema_steps_enabled, settings=settings)
    required_schema = build_coupler_schema(
        _filter_schema_steps_for_enabled_models(
            schema_steps_all,
            settings,
            include_optional=False,
        ),
        settings=settings,
        include_extras=False,
    )
    required_output_keys = list(required_schema.keys())
    scenario_kwargs["require_outputs"] = required_output_keys

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
    try:
        with cr.scenario(
            run_name,
            tracker=tracker,
            tags=["pilates_simulation"],
            model="pilates_orchestrator",
            **scenario_kwargs,
        ) as scenario:
            coupler = scenario.coupler
            coupler.declare_outputs(
                *coupler_schema.keys(),
                warn_undefined=True,
                description=coupler_schema,
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
                usim_inputs: Dict[str, Any] = {}
                outputs_holder_year = StepOutputsHolder()

                if state.should_run(WorkflowState.Stage.land_use):
                    usim_inputs = run_land_use_stage(
                        scenario=scenario,
                        state=state,
                        settings=settings,
                        workspace=workspace,
                        coupler=coupler,
                        year=year,
                        outputs_holder_year=outputs_holder_year,
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
                        scenario=scenario,
                        state=state,
                        settings=settings,
                        workspace=workspace,
                        coupler=coupler,
                        year=year,
                        build_atlas_static_inputs_fallback=build_atlas_static_inputs_fallback,
                    )
                    state.complete_step(WorkflowState.Stage.vehicle_ownership_model)
                    snapshot_manager.maybe_snapshot_interval(
                        reason=f"after_vehicle_ownership_y{year}"
                    )

                if state.should_run(WorkflowState.Stage.supply_demand_loop):
                    run_supply_demand_stage(
                        scenario=scenario,
                        state=state,
                        settings=settings,
                        workspace=workspace,
                        coupler=coupler,
                        year=year,
                        usim_inputs=usim_inputs,
                        build_manifest_path=build_manifest_path,
                        on_iteration_boundary=(
                            lambda iteration, y=year: snapshot_manager.on_outer_iteration_boundary(
                                year=y,
                                iteration=iteration,
                            )
                        ),
                    )
                    snapshot_manager.maybe_snapshot_interval(
                        reason=f"after_supply_demand_y{year}"
                    )

                if state.should_run(WorkflowState.Stage.postprocessing):
                    formatted_print("POST-PROCESSING")
                    run_postprocessing_stage(
                        scenario=scenario,
                        state=state,
                        settings=settings,
                        workspace=workspace,
                        coupler=coupler,
                        year=year,
                    )
                    state.complete_step(WorkflowState.Stage.postprocessing)
                    snapshot_manager.maybe_snapshot_interval(
                        reason=f"after_postprocessing_y{year}"
                    )
                snapshot_manager.maybe_snapshot_interval(reason=f"year_boundary_y{year}")

        formatted_print("SIMULATION COMPLETE")
        logger.info("[Main] Simulation complete.")
    finally:
        snapshot_ok = snapshot_manager.final_snapshot()
        flush_archive_queue(timeout=300)
        stop_archive_worker(timeout=30)
        if not snapshot_ok:
            mirror_consist_db_to_archive(local_consist_db_path, archive_consist_db_path)


if __name__ == "__main__":
    main()
