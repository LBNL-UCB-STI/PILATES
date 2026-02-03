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
from typing import Optional, cast, Dict, Any

# Consist Imports (optional)
try:
    import consist
except ImportError:  # Consist optional dependency
    consist = None

# Legacy/PILATES Imports
from pilates.generic.records import RecordStore
from pilates.workspace import Workspace
from pilates.generic.initialization import Initialization
from pilates.utils.formatting import formatted_print
from pilates.utils.io import parse_args_and_settings
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_config import (
    build_scenario_consist_kwargs,
    build_step_consist_kwargs,
)
from pilates.utils.consist_types import ScenarioWithCoupler, TrackerLike
from pilates.utils.input_logging import log_inputs
from pilates.workflows.artifact_constants import (
    ASIM_MUTABLE_DATA_DIR,
    ASIM_OUTPUT_DIR,
    ATLAS_OUTPUT_DIR,
    ATLAS_VEHICLES2_INPUT,
    BEAM_MUTABLE_DATA_DIR,
    BEAM_OUTPUT_DIR,
    FINAL_SKIMS_OMX,
    USIM_DATASTORE_H5,
    USIM_MUTABLE_DATA_DIR,
    ZARR_SKIMS,
)
from pilates.workflows.coupler_schema import PILATES_COUPLER_SCHEMA
from pilates.workflows.stages import (
    run_land_use_stage,
    run_postprocessing_stage,
    run_supply_demand_stage,
    run_vehicle_ownership_stage,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
from workflow_state import WorkflowState

from pilates.workflows.steps import StepOutputsHolder

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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

    if state.run_info_path:
        run_name = os.path.basename(os.path.dirname(state.run_info_path))
        logger.info(f"Restarting run. Reusing output folder: {run_name}")
    else:
        partial_run_name = settings.run.output_run_name
        run_name = f"{partial_run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        logger.info(f"Starting fresh run. Creating new output folder: {run_name}")

    full_run_dir = os.path.join(output_path, run_name)
    os.makedirs(full_run_dir, exist_ok=True)

    # 3. INITIALIZE CONSIST TRACKER (OPTIONAL)
    # Consist provides provenance tracking and computation caching. It's optional; PILATES
    # works without it but gains:
    #   - Provenance: Full lineage of data transformations (OpenLineage compatible)
    #   - Caching: Skips expensive computations if inputs unchanged
    #   - Coupler: Manages artifact passing between steps
    # Mount Strategy:
    # - 'inputs': The project root. Source files resolve here.
    # - 'workspace': The mutable run dir. Destination files resolve here.
    # NOTE: Do not rely on cwd; production runs may invoke `python run.py` from elsewhere.
    # Use the directory containing `run.py` as the canonical inputs root.
    project_root_abs = str(Path(__file__).resolve().parent)

    consist_enabled = cr.consist_available(settings)
    tracker: Optional[TrackerLike] = None
    if consist_enabled:
        logger.info(f"Initializing Consist Tracker in {full_run_dir}")
    else:
        logger.info("Consist disabled/unavailable; running without Consist tracker.")

    tracker = cr.create_tracker(
        settings=settings,
        enabled=consist_enabled,
        run_dir=full_run_dir,
        db_path=(
            settings.shared.database.path if settings.shared.database.enabled else None
        ),
        mounts={
            "inputs": project_root_abs,  # Immutable Source
            "workspace": full_run_dir,  # Mutable Destination
            "scratch": str(Path(output_path).resolve()),  # For temp files
        },
        project_root=project_root_abs,
        schemas=_get_consist_schemas(),
    )
    if tracker is None and consist_enabled:
        raise RuntimeError(
            "Consist enabled but tracker could not be created. "
            "Install Consist or set settings.shared.database.use_consist=False."
        )
    if consist_enabled:
        assert tracker is not None

    # 4. INITIALIZE WORKSPACE
    workspace = Workspace(
        settings,
        output_path,
        folder_name=run_name,
    )
    state.file_loc = os.path.join(workspace.full_path, "run_state.yaml")
    if not state.run_info_path:
        state.set_run_info_path(state.file_loc)

    # 5. START SCENARIO CONTEXT
    # The scenario context is where all model execution happens. Each step runs inside
    # scenario.run(), which handles:
    #   - Caching checks (skip if inputs identical to previous run)
    #   - Provenance logging (inputs, outputs, dependencies)
    #   - Coupler coordination (step outputs → coupler → next step inputs)
    # The coupler is a shared dict-like object for passing artifacts between steps.
    if tracker is not None:
        cr.set_tracker(tracker)
    scenario_kwargs = build_scenario_consist_kwargs(settings)
    if consist_enabled:
        scenario_kwargs["require_outputs"] = list(PILATES_COUPLER_SCHEMA.keys())
    with cr.scenario(
        run_name,
        tracker=tracker,
        enabled=consist_enabled,
        tags=["pilates_simulation"],
        model="pilates_orchestrator",
        **scenario_kwargs,
    ) as scenario:
        scenario = cast(ScenarioWithCoupler, scenario)
        coupler = scenario.coupler
        if consist_enabled:
            require_outputs = getattr(scenario, "require_outputs", None)
            if callable(require_outputs):
                require_outputs(
                    *PILATES_COUPLER_SCHEMA.keys(),
                    warn_undocumented=True,
                    description=PILATES_COUPLER_SCHEMA,
                )
        scenario.declare_outputs(
            USIM_DATASTORE_H5,
            ASIM_MUTABLE_DATA_DIR,
            ASIM_OUTPUT_DIR,
            BEAM_OUTPUT_DIR,
            ATLAS_OUTPUT_DIR,
            ZARR_SKIMS,
            FINAL_SKIMS_OMX,
        )

        # 6. DATA INITIALIZATION STEP
        # Copies immutable input data to the mutable workspace. This is the critical
        # "data initialization" event in provenance: it links original sources (inputs://)
        # to the working copies (workspace://). Only runs if state.data_initialized is
        # False (first run) or if resuming from a checkpoint (uses previous workspace).
        #
        # ProvenanceNote: scenario.trace() logs this step, creating a provenance entry
        # that later steps reference as inputs.
        if not state.data_initialized:
            logger.info("Running Initialization Step (Copying mutable data)")

            with scenario.trace(
                "initialization",
                model="initialization",
                year=state.start_year,
                iteration=0,
                tags=["init"],
                **build_step_consist_kwargs(
                    "initialization", settings, workspace_path=workspace.full_path
                ),
            ):
                init_model = Initialization("initialization", state)

                # This performs the copy.
                # Source files -> recorded as inputs (inputs://...)
                # Dest files -> recorded as outputs (workspace://...)
                init_model.run(settings, workspace)

            state.set_data_initialized(True)
        else:
            logger.info(
                "Restarting from a previous state. Skipping data initialization."
            )

        # 6. MAIN WORKFLOW LOOP
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
                )

            if state.should_run(WorkflowState.Stage.postprocessing):
                formatted_print("POST-PROCESSING")
                run_postprocessing_stage(
                    scenario=scenario,
                    state=state,
                    settings=settings,
                    workspace=workspace,
                    year=year,
                )
                state.complete_step(WorkflowState.Stage.postprocessing)

    formatted_print("SIMULATION COMPLETE")
    logger.info("[Main] Simulation complete.")


if __name__ == "__main__":
    main()
