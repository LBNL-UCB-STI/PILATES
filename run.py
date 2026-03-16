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
import shlex
import sys
import shutil
import socket
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Tuple, Sequence, cast

from pilates.workspace import Workspace
from pilates.generic.records import sanitize_artifact_key
from pilates.generic.model_factory import ModelFactory
from pilates.generic.initialization import (
    Initialization,
    build_bootstrap_artifact_summary,
)
from pilates.utils.formatting import formatted_print
from pilates.utils.io import get_traffic_assignment_model, parse_args_and_settings
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
    snapshot_latest_dir,
)
from pilates.utils.coupler_helpers import (
    enqueue_archive_copy,
    flush_archive_queue,
    stop_archive_worker,
)
from pilates.utils.restart_bundle import (
    build_restart_bundle_manifest,
    manifest_entries_to_local_artifacts,
    write_restart_bundle_manifest,
)
from pilates.atlas.inputs import atlas_static_input_relpaths
from pilates.activitysim.preprocessor import required_asim_config_dirs
from pilates.urbansim.postprocessor import get_usim_datastore_fname
from pilates.utils.consist_types import ScenarioWithCoupler
from pilates.runtime import bootstrap as bootstrap_runtime
from pilates.runtime import restart as restart_runtime
from pilates.runtime import scenario_runtime
from pilates.workflows.coupler_schema import build_coupler_schema
from pilates.workflows.stages import (
    run_land_use_stage,
    run_postprocessing_stage,
    run_supply_demand_stage,
    run_vehicle_ownership_stage,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
from workflow_state import WorkflowState

from pilates.workflows.steps import StepOutputsHolder, validate_workflow_step_contracts
from consist.types import CacheOptions

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_SCENARIO_NAME_TEMPLATE = "{func_name}__y{year}__i{iteration}__phase_{phase}"
_RUN_FAILURE_CONTEXT: Dict[str, Any] = {}


def _coerce_int(value: Any) -> Optional[int]:
    return scenario_runtime.coerce_int(value)


def _resolve_scenario_id(settings: Any) -> str:
    return scenario_runtime.resolve_scenario_id(settings)


def _resolve_seed(settings: Any) -> Optional[int]:
    return scenario_runtime.resolve_seed(settings)


def _set_run_failure_context(**kwargs: Any) -> None:
    for key, value in kwargs.items():
        if value is None:
            continue
        _RUN_FAILURE_CONTEXT[key] = value


def _format_restart_command(
    *,
    settings: Optional[Any],
    archive_state_path: Optional[str],
) -> Optional[str]:
    config_path = None
    if settings is not None:
        config_path = getattr(settings, "settings_file", None)
    if not config_path and not archive_state_path:
        return None

    command = ["python", "run.py"]
    if config_path:
        command.extend(["-c", str(config_path)])
    if archive_state_path:
        command.extend(["-S", str(archive_state_path)])
    return " ".join(shlex.quote(part) for part in command)

def _format_hpc_restart_command(
    *,
    settings: Optional[Any],
    archive_state_path: Optional[str],
) -> Optional[str]:
    config_path = None
    if settings is not None:
        config_path = getattr(settings, "settings_file", None)
    if not config_path and not archive_state_path:
        return None

    command = ["./hpc/job_runner.sh"]
    if config_path:
        command.extend(["-c", str(config_path)])
    command.extend(["-a", "<slurm_account>"])
    if archive_state_path:
        command.extend(["-s", str(archive_state_path)])
    return " ".join(shlex.quote(part) for part in command)


def _log_restart_instructions_on_failure() -> None:
    settings = _RUN_FAILURE_CONTEXT.get("settings")
    state = _RUN_FAILURE_CONTEXT.get("state")
    archive_run_dir = _RUN_FAILURE_CONTEXT.get("archive_run_dir")
    local_run_dir = _RUN_FAILURE_CONTEXT.get("local_run_dir")
    archive_state_path = _RUN_FAILURE_CONTEXT.get("archive_state_path")
    if archive_state_path is None and state is not None:
        archive_state_path = getattr(state, "run_info_path", None)

    command = _format_restart_command(
        settings=settings,
        archive_state_path=archive_state_path,
    )
    if command is None:
        return

    logger.error("Run failed. Restart command:")
    logger.error("  %s", command)
    if archive_run_dir:
        command_hpc = _format_hpc_restart_command(
            settings=settings,
            archive_state_path=archive_state_path,
        )
        logger.error("  HPC command: %s", command_hpc)
    if archive_state_path:
        logger.error("  state file: %s", archive_state_path)
    if archive_run_dir:
        logger.error("  archive run dir: %s", archive_run_dir)
    if local_run_dir:
        logger.error("  local run dir: %s", local_run_dir)


def _merge_tag_list(existing: Any, additions: Sequence[str]) -> List[str]:
    return scenario_runtime.merge_tag_list(existing, additions)


def _facet_to_mapping(facet: Any) -> Dict[str, Any]:
    return scenario_runtime.facet_to_mapping(facet)


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


_EpochTaggingScenarioProxy = scenario_runtime.EpochTaggingScenarioProxy
_SchemaCoupler = scenario_runtime.SchemaCoupler


def _resolve_cache_epoch(settings: Any) -> int:
    return scenario_runtime.resolve_cache_epoch(settings)


def _build_schema_steps() -> List[Callable[..., Any]]:
    return scenario_runtime.build_schema_steps()


def _is_model_enabled(settings: Any, *, flag_attr: str, model_attr: str) -> bool:
    return scenario_runtime.is_model_enabled(
        settings,
        flag_attr=flag_attr,
        model_attr=model_attr,
    )


def _filter_schema_steps_for_enabled_models(
    steps: List[Callable[..., Any]],
    settings: Any,
    *,
    include_optional: bool = True,
) -> List[Callable[..., Any]]:
    return scenario_runtime.filter_schema_steps_for_enabled_models(
        steps,
        settings,
        include_optional=include_optional,
    )


def _get_consist_schemas() -> Optional[list[type[Any]]]:
    try:
        from pilates.database.schema.registry import get_consist_schemas

        return get_consist_schemas()
    except Exception:
        return None


def _build_scenario_runtime_contract(
    *,
    settings: Any,
    scenario_id: str,
    seed: Optional[int],
    cache_epoch: int,
) -> Dict[str, Any]:
    return scenario_runtime.build_scenario_runtime_contract(
        settings=settings,
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
    )


def build_manifest_path(workspace: Workspace, year: int, iteration: int) -> Path:
    return (
        Path(workspace.full_path)
        / ".workflow"
        / f"year_{year}_iteration_{iteration}.yaml"
    )


def _atlas_static_input_key(relpath: str) -> str:
    normalized_relpath = relpath.replace("\\", "/")
    rel_no_ext = os.path.splitext(normalized_relpath)[0]
    return sanitize_artifact_key(rel_no_ext) or rel_no_ext


def _iter_existing_atlas_static_inputs(
    settings: Any,
    atlas_input_dir: str,
):
    for relpath in atlas_static_input_relpaths(settings):
        normalized_relpath = relpath.replace("\\", "/")
        path = os.path.join(atlas_input_dir, normalized_relpath)
        if not os.path.exists(path):
            continue
        yield normalized_relpath, _atlas_static_input_key(normalized_relpath), path


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

    settings = getattr(workspace, "settings", None)
    if settings is not None:
        inputs: Dict[str, str] = {}
        for _relpath, key, path in _iter_existing_atlas_static_inputs(
            settings, atlas_input_dir
        ):
            inputs.setdefault(key, path)
        if inputs:
            return inputs

    inputs: Dict[str, str] = {}
    for root, _, files in os.walk(atlas_input_dir):
        for filename in sorted(files):
            path = os.path.join(root, filename)
            relpath = os.path.relpath(path, atlas_input_dir)
            key = _atlas_static_input_key(relpath)
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
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}EiB"


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
    return bootstrap_runtime.is_bootstrap_cache_enabled(settings)


def _build_bootstrap_manifest_reference(
    *,
    probe_run_id: Optional[str] = None,
    materialization_run_id: Optional[str] = None,
) -> Dict[str, str]:
    return bootstrap_runtime.build_bootstrap_manifest_reference(
        probe_run_id=probe_run_id,
        materialization_run_id=materialization_run_id,
    )


def _archive_bootstrap_restart_artifacts(
    *,
    settings: Any,
    workspace: Workspace,
) -> None:
    bootstrap_runtime.archive_bootstrap_restart_artifacts(
        settings=settings,
        workspace=workspace,
        enqueue_archive_copy_fn=enqueue_archive_copy,
        flush_archive_queue_fn=flush_archive_queue,
        get_usim_datastore_fname_fn=get_usim_datastore_fname,
    )


def run_bootstrap_phase(
    *,
    tracker: Any,
    settings: Any,
    state: WorkflowState,
    workspace: Workspace,
    scenario_id: str,
    seed: Optional[int],
) -> Dict[str, Any]:
    return bootstrap_runtime.run_bootstrap_phase(
        tracker=tracker,
        settings=settings,
        state=state,
        workspace=workspace,
        scenario_id=scenario_id,
        seed=seed,
        initialization_cls=Initialization,
        build_bootstrap_artifact_summary_fn=build_bootstrap_artifact_summary,
        build_step_consist_kwargs_fn=build_step_consist_kwargs,
        merge_tag_list_fn=_merge_tag_list,
        merge_epoch_facet_fn=_merge_epoch_facet,
        archive_bootstrap_restart_artifacts_fn=_archive_bootstrap_restart_artifacts,
        cache_options_cls=CacheOptions,
    )


def _assert_bootstrap_output_invariant(
    bootstrap_result: Optional[Dict[str, Any]],
) -> None:
    bootstrap_runtime.assert_bootstrap_output_invariant(bootstrap_result)


def _restart_required_local_artifacts(
    *,
    settings: Any,
    state: WorkflowState,
    workspace: Workspace,
) -> List[Dict[str, str]]:
    return restart_runtime.restart_required_local_artifacts(
        settings=settings,
        state=state,
        workspace=workspace,
        get_usim_datastore_fname_fn=get_usim_datastore_fname,
        required_asim_config_dirs_fn=required_asim_config_dirs,
        atlas_static_input_relpaths_fn=atlas_static_input_relpaths,
        workflow_stage=WorkflowState.Stage,
    )


def _find_missing_restart_local_artifacts(
    *,
    settings: Any,
    state: WorkflowState,
    workspace: Workspace,
) -> List[Dict[str, str]]:
    return restart_runtime.find_missing_restart_local_artifacts(
        settings=settings,
        state=state,
        workspace=workspace,
        restart_required_local_artifacts_fn=_restart_required_local_artifacts,
    )


def _format_missing_artifact_summary(artifacts: List[Dict[str, str]]) -> str:
    return restart_runtime.format_missing_artifact_summary(artifacts)


def _resolve_restart_rehydrate_mode(settings: Any) -> str:
    return restart_runtime.resolve_restart_rehydrate_mode(settings)


def _is_restart_strict(settings: Any) -> bool:
    return restart_runtime.is_restart_strict(settings)


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


def _map_local_path_to_archive(
    *,
    local_path: str,
    local_run_dir: str,
    archive_run_dir: str,
) -> Optional[str]:
    return restart_runtime.map_local_path_to_archive(
        local_path=local_path,
        local_run_dir=local_run_dir,
        archive_run_dir=archive_run_dir,
    )


def _repair_restart_state_for_incomplete_atlas_outputs(
    *,
    settings: Any,
    state: WorkflowState,
    archive_run_dir: str,
) -> bool:
    return restart_runtime.repair_restart_state_for_incomplete_atlas_outputs(
        settings=settings,
        state=state,
        archive_run_dir=archive_run_dir,
    )


def _repair_restart_beam_inputs_from_source(
    *,
    settings: Any,
    state: WorkflowState,
    workspace: Workspace,
) -> bool:
    if get_traffic_assignment_model(settings) != "beam":
        return False

    beam_cfg = getattr(settings, "beam", None)
    region = getattr(getattr(settings, "run", None), "region", None)
    beam_config_name = getattr(beam_cfg, "config", None)
    if beam_cfg is None or not region or not beam_config_name:
        return False

    beam_root = Path(workspace.get_beam_mutable_data_dir())
    region_dir = beam_root / region
    config_path = beam_root / region / beam_config_name
    if beam_root.exists() and region_dir.exists() and config_path.exists():
        return False

    logger.warning(
        "[RestartRepair] BEAM primary config missing at %s; repopulating BEAM "
        "mutable inputs from source production inputs.",
        config_path,
    )
    beam_root.mkdir(parents=True, exist_ok=True)
    beam_preprocessor = ModelFactory().get_preprocessor("beam", state)
    beam_preprocessor.copy_data_to_mutable_location(settings, str(beam_root))
    repaired = config_path.exists()
    if repaired:
        logger.info(
            "[RestartRepair] Restored BEAM primary config from source: %s",
            config_path,
        )
    else:
        logger.warning(
            "[RestartRepair] BEAM source repopulation completed but primary config is "
            "still missing: %s",
            config_path,
        )
    return repaired


def _copy_archive_entry_preserve_existing(
    *,
    archive_path: str,
    local_path: str,
) -> Tuple[int, int]:
    return restart_runtime.copy_archive_entry_preserve_existing(
        archive_path=archive_path,
        local_path=local_path,
    )


def _rehydrate_missing_local_artifacts_from_archive(
    *,
    missing_artifacts: List[Dict[str, str]],
    local_run_dir: str,
    archive_run_dir: str,
) -> Dict[str, int]:
    return restart_runtime.rehydrate_missing_local_artifacts_from_archive(
        missing_artifacts=missing_artifacts,
        local_run_dir=local_run_dir,
        archive_run_dir=archive_run_dir,
        map_local_path_to_archive_fn=_map_local_path_to_archive,
        copy_archive_entry_fn=_copy_archive_entry_preserve_existing,
    )


def _rehydrate_full_local_run_from_archive(
    *,
    local_run_dir: str,
    archive_run_dir: str,
) -> Dict[str, int]:
    return restart_runtime.rehydrate_full_local_run_from_archive(
        local_run_dir=local_run_dir,
        archive_run_dir=archive_run_dir,
        copy_archive_entry_fn=_copy_archive_entry_preserve_existing,
    )


def _rehydrate_bundle_local_artifacts_from_archive(
    *,
    bundle_manifest: Optional[Dict[str, Any]],
    local_run_dir: str,
    archive_run_dir: str,
) -> Dict[str, int]:
    return restart_runtime.rehydrate_bundle_local_artifacts_from_archive(
        bundle_manifest=bundle_manifest,
        local_run_dir=local_run_dir,
        archive_run_dir=archive_run_dir,
        manifest_entries_to_local_artifacts_fn=manifest_entries_to_local_artifacts,
        rehydrate_missing_local_artifacts_fn=_rehydrate_missing_local_artifacts_from_archive,
    )


def _log_resume_doctor_check(
    *,
    check: str,
    ok: bool,
    detail: str,
) -> None:
    restart_runtime.log_resume_doctor_check(check=check, ok=ok, detail=detail)


def _run_resume_doctor_diagnostics(
    *,
    state: WorkflowState,
    workspace: Workspace,
    local_run_dir: str,
    archive_run_dir: str,
    archive_state_path: str,
    local_state_path: str,
    local_consist_db_path: Optional[str],
    restart_missing_artifacts_initial: List[Dict[str, str]],
    restart_missing_artifacts_after_rehydrate: List[Dict[str, str]],
) -> None:
    restart_runtime.run_resume_doctor_diagnostics(
        state=state,
        workspace=workspace,
        local_run_dir=local_run_dir,
        archive_run_dir=archive_run_dir,
        archive_state_path=archive_state_path,
        local_state_path=local_state_path,
        local_consist_db_path=local_consist_db_path,
        restart_missing_artifacts_initial=restart_missing_artifacts_initial,
        restart_missing_artifacts_after_rehydrate=restart_missing_artifacts_after_rehydrate,
        snapshot_latest_dir_fn=snapshot_latest_dir,
        build_manifest_path_fn=build_manifest_path,
        map_local_path_to_archive_fn=_map_local_path_to_archive,
        format_missing_artifact_summary_fn=_format_missing_artifact_summary,
        log_resume_doctor_check_fn=_log_resume_doctor_check,
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
    _RUN_FAILURE_CONTEXT.clear()
    # 1. PARSE SETTINGS AND SET UP WORKFLOW STATE
    settings = parse_args_and_settings()
    state = WorkflowState.from_settings(settings)
    _set_run_failure_context(settings=settings, state=state)

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

    is_restart_run = bool(state.run_info_path)
    if is_restart_run:
        run_name = os.path.basename(os.path.dirname(state.run_info_path))
        logger.info(f"Restarting run. Reusing output folder: {run_name}")
    else:
        partial_run_name = settings.run.output_run_name
        run_name = f"{partial_run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
    _set_run_failure_context(
        archive_run_dir=archive_run_dir,
        local_run_dir=local_run_dir,
    )
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
        # PILATES roots the tracker at the archive run dir but rematerializes
        # restart/bootstrap outputs into the mounted local workspace root.
        allow_external_paths=True,
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
    _set_run_failure_context(archive_state_path=archive_state_path)
    if is_restart_run:
        _enforce_resume_rewind_guardrail(
            state=state,
            archive_state_path=archive_state_path,
            allow_rewind_resume=bool(getattr(settings, "allow_rewind_resume", False)),
        )
        _repair_restart_state_for_incomplete_atlas_outputs(
            settings=settings,
            state=state,
            archive_run_dir=archive_run_dir,
        )
    state.file_loc = archive_state_path
    state.mirror_file_loc = local_state_path
    if state.run_info_path != archive_state_path:
        state.set_run_info_path(archive_state_path)

    restart_rehydrate_mode = _resolve_restart_rehydrate_mode(settings)
    restart_strict = _is_restart_strict(settings)
    restart_bundle_manifest = build_restart_bundle_manifest(
        archive_run_dir=archive_run_dir,
        local_run_dir=local_run_dir,
        settings=settings,
        workspace=workspace,
        state=state,
        local_consist_db_path=local_consist_db_path,
    )
    restart_bundle_manifest_path = write_restart_bundle_manifest(
        archive_run_dir=archive_run_dir,
        manifest=restart_bundle_manifest,
    )
    logger.info(
        "Restart bundle manifest ready: path=%s artifacts=%s mode=%s strict=%s",
        restart_bundle_manifest_path,
        len(restart_bundle_manifest.get("artifacts", [])),
        restart_rehydrate_mode,
        restart_strict,
    )

    restart_missing_artifacts_initial: List[Dict[str, str]] = []
    restart_missing_artifacts_after_rehydrate: List[Dict[str, str]] = []
    if state.data_initialized:
        restart_missing_artifacts_initial = _find_missing_restart_local_artifacts(
            settings=settings,
            state=state,
            workspace=workspace,
        )
        if restart_missing_artifacts_initial:
            logger.warning(
                "Restart preflight found missing local workspace inputs while "
                "data_initialized=True: %s",
                _format_missing_artifact_summary(restart_missing_artifacts_initial),
            )
            if restart_rehydrate_mode == "off":
                logger.info(
                    "Restart rehydration disabled (run.restart_rehydrate_mode=off)."
                )
            elif restart_rehydrate_mode == "full":
                _rehydrate_full_local_run_from_archive(
                    local_run_dir=local_run_dir,
                    archive_run_dir=archive_run_dir,
                )
            else:
                _rehydrate_bundle_local_artifacts_from_archive(
                    bundle_manifest=restart_bundle_manifest,
                    local_run_dir=local_run_dir,
                    archive_run_dir=archive_run_dir,
                )
            restart_missing_artifacts_after_rehydrate = (
                _find_missing_restart_local_artifacts(
                    settings=settings,
                    state=state,
                    workspace=workspace,
                )
            )
            if restart_missing_artifacts_after_rehydrate:
                logger.warning(
                    "Restart preflight still missing required local workspace inputs "
                    "after archive rehydration: %s",
                    _format_missing_artifact_summary(
                        restart_missing_artifacts_after_rehydrate
                    ),
                )
                if restart_strict:
                    raise RuntimeError(
                        "Strict restart preflight failed; required restart artifacts are "
                        "still missing after hydration. missing="
                        + _format_missing_artifact_summary(
                            restart_missing_artifacts_after_rehydrate
                        )
                    )
    if is_restart_run and _repair_restart_beam_inputs_from_source(
        settings=settings,
        state=state,
        workspace=workspace,
    ):
        restart_missing_artifacts_after_rehydrate = _find_missing_restart_local_artifacts(
            settings=settings,
            state=state,
            workspace=workspace,
        )
    if is_restart_run:
        _run_resume_doctor_diagnostics(
            state=state,
            workspace=workspace,
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
            archive_state_path=archive_state_path,
            local_state_path=local_state_path,
            local_consist_db_path=local_consist_db_path,
            restart_missing_artifacts_initial=restart_missing_artifacts_initial,
            restart_missing_artifacts_after_rehydrate=restart_missing_artifacts_after_rehydrate,
        )

    # 5. BOOTSTRAP PHASE (PRE-SCENARIO)
    # Initialization runs before entering scenario step execution so bootstrap
    # lifecycle can evolve independently from normal model steps.
    cr.set_tracker(tracker)
    bootstrap_result: Optional[Dict[str, Any]] = None
    force_restart_bootstrap = bool(restart_missing_artifacts_after_rehydrate)
    if (
        restart_missing_artifacts_initial
        and not restart_missing_artifacts_after_rehydrate
    ):
        logger.info(
            "Restart preflight recovered missing local inputs from archive; "
            "continuing without forced bootstrap."
        )
    if not state.data_initialized or force_restart_bootstrap:
        if force_restart_bootstrap:
            logger.warning(
                "Forcing bootstrap initialization on restart because required local "
                "inputs were missing during preflight. initial_missing=%s "
                "remaining_after_rehydrate=%s",
                _format_missing_artifact_summary(restart_missing_artifacts_initial),
                _format_missing_artifact_summary(
                    restart_missing_artifacts_after_rehydrate
                ),
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
        )
        _assert_bootstrap_output_invariant(bootstrap_result)
        if not state.data_initialized:
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
    scenario_contract = _build_scenario_runtime_contract(
        settings=settings,
        scenario_id=scenario_id,
        seed=run_seed,
        cache_epoch=cache_epoch,
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
            tagged_scenario = _EpochTaggingScenarioProxy(
                scenario,
                scenario_id=scenario_id,
                seed=run_seed,
            )
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
                        scenario=cast(ScenarioWithCoupler, tagged_scenario),
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
                        scenario=cast(ScenarioWithCoupler, tagged_scenario),
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
                        scenario=cast(ScenarioWithCoupler, tagged_scenario),
                        state=state,
                        settings=settings,
                        workspace=workspace,
                        coupler=coupler,
                        year=year,
                        usim_inputs=usim_inputs,
                        build_manifest_path=build_manifest_path,
                        on_iteration_boundary=(
                            lambda iteration, y=year: (
                                snapshot_manager.on_outer_iteration_boundary(
                                    year=y,
                                    iteration=iteration,
                                )
                            )
                        ),
                    )
                    snapshot_manager.maybe_snapshot_interval(
                        reason=f"after_supply_demand_y{year}"
                    )

                if state.should_run(WorkflowState.Stage.postprocessing):
                    formatted_print("POST-PROCESSING")
                    run_postprocessing_stage(
                        scenario=cast(ScenarioWithCoupler, tagged_scenario),
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
                snapshot_manager.maybe_snapshot_interval(
                    reason=f"year_boundary_y{year}"
                )

        formatted_print("SIMULATION COMPLETE")
        logger.info("[Main] Simulation complete.")
    finally:
        snapshot_ok = snapshot_manager.final_snapshot()
        flush_archive_queue(timeout=300)
        stop_archive_worker(timeout=30)
        if not snapshot_ok:
            mirror_consist_db_to_archive(local_consist_db_path, archive_consist_db_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _log_restart_instructions_on_failure()
        raise
