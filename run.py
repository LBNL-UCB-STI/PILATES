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
import uuid
import os
import logging
import sys
from pathlib import Path
from typing import Any

# Consist Imports (optional)
try:
    import consist
    from consist import Tracker
except ImportError:  # Consist optional dependency
    consist = None
    Tracker = None

# Legacy/PILATES Imports
from pilates.generic.model_factory import ModelFactory
from pilates.generic.records import RecordStore
from pilates.generic.runner import GenericRunner
from pilates.workspace import Workspace
from pilates.generic.initialization import Initialization
from pilates.config.models import PilatesConfig
from pilates.utils.io import parse_args_and_settings
from pilates.postprocessing.postprocessor import process_event_file, copy_outputs_to_mep
from pilates.utils.consist_adapter import ConsistProvenanceTracker
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_config import (
    build_scenario_consist_kwargs,
    build_step_consist_kwargs,
)
from pilates.utils.coupler_helpers import (
    artifact_to_path,
    clean_expected_outputs,
    update_coupler_from_beam_outputs,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
from workflow_state import WorkflowState

from pilates.activitysim import postprocessor as asim_post
from pilates.urbansim import postprocessor as usim_post

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def formatted_print(string, width=50, fill_char="#"):
    """
    Print a formatted banner for major workflow steps.
    """
    print("\n")
    if len(string) + 2 > width:
        width = len(string) + 4
    string = string.upper()
    print(fill_char * width)
    print("{:#^{width}}".format(" " + string + " ", width=width))
    print(fill_char * width, "\n")


def _expected_inputs_for(model_name: str, settings, state, workspace) -> dict:
    expected = {}
    components = ModelFactory._registry.get(model_name, {})
    for component_name in ("preprocessor", "runner", "postprocessor"):
        component_cls = components.get(component_name)
        if component_cls is None:
            continue
        expected_fn = getattr(component_cls, "expected_inputs", None)
        if callable(expected_fn):
            _merge_expected_inputs(expected, expected_fn(settings, state, workspace) or {})
    return expected


def _expected_outputs_for(
    model_name: str, settings, state, workspace, *, components=None
) -> dict:
    expected = {}
    registry_components = ModelFactory._registry.get(model_name, {})
    component_names = components or ("runner", "postprocessor")
    for component_name in component_names:
        component_cls = registry_components.get(component_name)
        if component_cls is None:
            continue
        expected_fn = getattr(component_cls, "expected_outputs", None)
        if callable(expected_fn):
            _merge_expected_outputs(expected, expected_fn(settings, state, workspace) or {})
    return expected


def _merge_expected_inputs(
    target: dict, expected: dict, *, prefer_expected: bool = False
) -> dict:
    for key, value in expected.items():
        if value is None:
            continue
        if key in target:
            if prefer_expected:
                target[key] = value
            else:
                logger.debug(
                    "Input '%s' already provided; keeping existing value.", key
                )
            continue
        target[key] = value
    return target


def _merge_expected_outputs(
    target: dict, expected: dict, *, prefer_expected: bool = False
) -> dict:
    for key, value in expected.items():
        if value is None:
            continue
        if key in target:
            if prefer_expected:
                target[key] = value
            else:
                logger.debug(
                    "Output '%s' already provided; keeping existing value.", key
                )
            continue
        target[key] = value
    return target






def warm_start_activities(
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: ConsistProvenanceTracker,
):
    """
    TODO: THIS IS BROKEN

    Run ActivitySim warm-start to update UrbanSim inputs with long-term choices.

    This function is typically executed only in the initial start year as part of
    the Land Use step when ActivitySim is used as the activity demand model.

    Sequence:
      - Instantiate ActivitySim preprocessor/runner via ModelFactory
      - Preprocess to prepare ActivitySim inputs
      - Run ActivitySim runner to produce outputs
      - Apply a specialized post-processing function that updates UrbanSim H5 input
        files with warm-started choices (workplace, school, auto ownership)
      - Record the postprocessor model run with the provenance tracker

    Args:
        settings (PilatesConfig): Parsed simulation settings.
        state (WorkflowState): Current workflow state.
        workspace (Workspace): Workspace object with file paths.
        provenance_tracker (OpenLineageTracker): Provenance tracker instance.
    """
    factory = ModelFactory()
    activity_demand_model, _ = GenericRunner.get_model_and_image(
        settings, "activity_demand_model"
    )

    if activity_demand_model == "polaris":
        logger.info("POLARIS module is not activated due to missing polarisruntime library")

    elif activity_demand_model == "activitysim":
        runner = factory.get_runner("activitysim", state, provenance_tracker)
        preprocessor = factory.get_preprocessor("activitysim", state, provenance_tracker)

        # Preprocess
        input_data = preprocessor.preprocess(workspace)

        # Run ActivitySim
        runner.run(input_data, workspace)

        logger.info("Appending warm start activities/choices to UrbanSim base year input data")

        asim_post.update_usim_inputs_after_warm_start(
            settings,
            state,
            workspace,
            provenance_tracker,
            model_run_hash=None,
        )

    logger.info("Done!")


def forecast_land_use(
    settings: PilatesConfig,
    year: int,
    workflow_state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: ConsistProvenanceTracker,
):
    """
    High-level wrapper to start an UrbanSim (land use) run.
    """
    land_use_model, _ = GenericRunner.get_model_and_image(
        settings, "land_use_model"
    )

    run_land_use(
        year,
        workflow_state.forecast_year,
        land_use_model,
        workflow_state,
        workspace,
        provenance_tracker,
    )

    # Verify Output
    usim_output_store_name = usim_post.get_usim_datastore_fname(
        settings, io="output", year=workflow_state.forecast_year
    )
    usim_datastore_fpath = os.path.join(
        workspace.get_usim_mutable_data_dir(), usim_output_store_name
    )

    if not os.path.exists(usim_datastore_fpath):
        logger.critical(
            f"No UrbanSim output data found at {usim_datastore_fpath}. Run failed."
        )
        sys.exit(1)


def run_land_use(
    year,
    forecast_year,
    land_use_model,
    state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: ConsistProvenanceTracker,
):
    """
    Prepare inputs, run UrbanSim, and postprocess outputs for a land-use forecast.

    Args:
        year (int): Current simulation year (start of forecast).
        forecast_year (int): Forecast target year.
        land_use_model (str): Land use model identifier (e.g., 'urbansim').
        state (WorkflowState): Workflow state.
        workspace (Workspace): Workspace instance for data paths.
        provenance_tracker (OpenLineageTracker): Provenance tracker instance.
    """
    logger.info("Running land use")

    factory = ModelFactory()

    preprocessor = factory.get_preprocessor(
        "urbansim", state, provenance_tracker, major_stage=WorkflowState.Stage.land_use
    )
    runner = factory.get_runner(
        "urbansim", state, provenance_tracker, major_stage=WorkflowState.Stage.land_use
    )
    postprocessor = factory.get_postprocessor(
        "urbansim", state, provenance_tracker, major_stage=WorkflowState.Stage.land_use
    )

    # 1. PREPROCESS
    formatted_print(f"Preparing {year} input data for land use development simulation.")
    input_data = preprocessor.preprocess(workspace)

    # 2. RUN
    formatted_print(f"Simulating land use development from {year} to {forecast_year} with {land_use_model}.")
    raw_outputs = runner.run(input_data, workspace)

    # 3. POSTPROCESS
    postprocessor.postprocess(raw_outputs, workspace)

    logger.info("Done!")


def run_activity_demand(
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: ConsistProvenanceTracker,
    input_store: RecordStore = None,
) -> RecordStore:
    """
    Generate activity plans for the current year using the configured activity demand model.

    Supports:
      - ActivitySim: full preprocess -> run -> postprocess sequence
      - Polaris: logged but not implemented here

    Args:
        settings (PilatesConfig): Simulation settings.
        state (WorkflowState): Current workflow state.
        workspace (Workspace): Workspace instance for file operations.
        provenance_tracker (OpenLineageTracker): Provenance tracker for model events.
        input_store (RecordStore, optional): Preprocessed inputs to reuse if available.

    Returns:
        RecordStore: Processed outputs (file records) from the activity demand postprocessor.
    """
    factory = ModelFactory()
    activity_demand_model = settings.run.models.activity_demand

    if activity_demand_model == "polaris":
        logger.info("POLARIS module is not activated")
        return RecordStore()

    elif activity_demand_model == "activitysim":
        preprocessor = factory.get_preprocessor(
            "activitysim", state, provenance_tracker, major_stage=WorkflowState.Stage.activity_demand,
        )
        runner = factory.get_runner(
            "activitysim", state, provenance_tracker, major_stage=WorkflowState.Stage.activity_demand,
        )
        postprocessor = factory.get_postprocessor(
            "activitysim", state, provenance_tracker, major_stage=WorkflowState.Stage.activity_demand,
        )

        input_data = input_store or preprocessor.preprocess(workspace)
        raw_outputs = runner.run(input_data, workspace)
        processed_outputs = postprocessor.postprocess(raw_outputs, workspace)

        return processed_outputs

    else:
        logger.warning(f"Unknown activity demand model: {activity_demand_model}")
        return RecordStore()


def run_traffic_assignment(
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: ConsistProvenanceTracker,
    activity_demand_outputs: RecordStore = None,
    previous_beam_outputs: RecordStore = None,
):
    """
    Run the configured traffic assignment (supply) model for the current year/iteration.

    Currently supports:
      - BEAM: obtains preprocessor, runner, postprocessor via ModelFactory and runs the
        preprocessor with the activity demand outputs as input.

    Args:
        settings (PilatesConfig): Simulation settings.
        state (WorkflowState): Current workflow state.
        workspace (Workspace): Workspace instance.
        provenance_tracker (OpenLineageTracker): Provenance tracker instance.
        activity_demand_outputs (RecordStore, optional): Processed activity demand outputs.
        previous_beam_outputs (RecordStore, optional): Outputs from previous BEAM iteration.
    """
    factory = ModelFactory()
    travel_model = settings.run.models.travel

    if travel_model == "polaris":
        logger.info("POLARIS module is not activated")
        return RecordStore()
    elif travel_model == "beam":
        preprocessor = factory.get_preprocessor(
            "beam", state, provenance_tracker, major_stage=WorkflowState.Stage.traffic_assignment,
        )
        runner = factory.get_runner(
            "beam", state, provenance_tracker, major_stage=WorkflowState.Stage.traffic_assignment,
        )
        postprocessor = factory.get_postprocessor(
            "beam", state, provenance_tracker, major_stage=WorkflowState.Stage.traffic_assignment,
        )

        combined_inputs = RecordStore()
        if activity_demand_outputs:
            combined_inputs += activity_demand_outputs
        if previous_beam_outputs:
            combined_inputs += previous_beam_outputs
        input_data = preprocessor.preprocess(workspace, combined_inputs)
        raw_outputs = runner.run(input_data, workspace)
        processed_outputs = postprocessor.postprocess(raw_outputs, workspace)
        return processed_outputs

    else:
        logger.warning(f"Unknown travel model: {travel_model}")
        return RecordStore()


def main():
    """
    Main entrypoint refactored to use Consist Scenario API.
    """
    # 1. PARSE SETTINGS AND SET UP WORKFLOW STATE
    settings = parse_args_and_settings()
    state = WorkflowState.from_settings(settings)

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
    # Mount Strategy:
    # - 'inputs': The project root. Source files resolve here.
    # - 'workspace': The mutable run dir. Destination files resolve here.
    # NOTE: Do not rely on cwd; production runs may invoke `python run.py` from elsewhere.
    # Use the directory containing `run.py` as the canonical inputs root.
    project_root_abs = str(Path(__file__).resolve().parent)

    consist_enabled = cr.consist_available(settings)
    tracker = None
    if consist_enabled:
        logger.info(f"Initializing Consist Tracker in {full_run_dir}")
        tracker = Tracker(
            run_dir=full_run_dir,
            db_path=settings.shared.database.path if settings.shared.database.enabled else None,
            mounts={
                "inputs": project_root_abs,  # Immutable Source
                "workspace": full_run_dir,  # Mutable Destination
            },
            project_root=project_root_abs,
        )
    else:
        logger.info("Consist disabled/unavailable; running without Consist tracker.")

    # 4. INITIALIZE WORKSPACE & ADAPTER
    # We pass the native 'tracker' to the adapter.
    # The adapter will detect active scenario steps and "attach" to them.
    adapter = None
    if tracker is not None:
        adapter = ConsistProvenanceTracker(
            run_id="placeholder_id",  # Will be overwritten by attach mode
            output_path=full_run_dir,
            folder_name=run_name,
            tracker=tracker,
        )

    workspace = Workspace(
        settings,
        output_path,
        folder_name=run_name,
        provenance_tracker=adapter,
    )
    state.file_loc = os.path.join(workspace.full_path, "run_state.yaml")


    # 5. START SCENARIO
    if tracker is not None:
        cr.set_tracker(tracker)
    with cr.scenario(
        run_name,
        tracker=tracker,
        enabled=consist_enabled,
        tags=["pilates_simulation"],
        model="pilates_orchestrator",
        **build_scenario_consist_kwargs(settings),
    ) as scenario:
        coupler = scenario.coupler
        scenario.declare_outputs(
            "usim_datastore_h5",
            "asim_output_dir",
            "beam_output_dir",
            "atlas_output_dir",
            "zarr_skims",
            "final_skims_omx",
        )
        typed_coupler = None
        if consist is not None and consist_enabled:
            @consist.coupler_schema
            class WorkflowCoupler:
                usim_datastore_h5: Any
                asim_output_dir: Any
                beam_output_dir: Any
                atlas_output_dir: Any
                zarr_skims: Any
                final_skims_omx: Any

            typed_coupler = scenario.coupler_schema(WorkflowCoupler)


        # 6. DATA INITIALIZATION STEP
        if not state.data_initialized:
            logger.info("Running Initialization Step (Copying mutable data)")

            # We use 'initialization' as the step name.
            # Initialization.py calls `adapter.start_model_run("initialization")` internally.
            # The Adapter will see this active step and ATTACH to it, rather than creating a new one.
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

                init_model = Initialization("initialization", state, provenance_tracker=adapter)

                # This performs the copy.
                # Source files -> recorded as inputs (inputs://...)
                # Dest files -> recorded as outputs (workspace://...)
                init_model.run(settings, workspace)

            state.set_data_initialized(True)
        else:
            logger.info("Restarting from a previous state. Skipping data initialization.")

        # 6. MAIN WORKFLOW LOOP
        for year in state:
            formatted_print(f"STARTING YEAR {year}")

            # A. LAND USE FORECASTING
            if state.should_run(WorkflowState.Stage.land_use):
                formatted_print(f"LAND USE MODEL FOR YEAR {state.forecast_year}")

                step_name = f"urbansim_{year}"

                usim_inputs = {}
                usim_data_dir = workspace.get_usim_mutable_data_dir()
                usim_input_fname = usim_post.get_usim_datastore_fname(
                    settings, io="input"
                )
                usim_input_path = os.path.join(usim_data_dir, usim_input_fname)
                if os.path.exists(usim_input_path):
                    usim_inputs["usim_datastore_h5"] = usim_input_path
                elif not state.is_start_year():
                    usim_output_fname = usim_post.get_usim_datastore_fname(
                        settings, io="output", year=year
                    )
                    usim_output_path = os.path.join(usim_data_dir, usim_output_fname)
                    if os.path.exists(usim_output_path):
                        usim_inputs["usim_datastore_h5"] = usim_output_path
                if os.path.exists(usim_data_dir):
                    usim_inputs["usim_mutable_data_dir"] = usim_data_dir
                if usim_inputs:
                    if "usim_datastore_h5" in usim_inputs:
                        cr.log_input(
                            usim_inputs["usim_datastore_h5"],
                            key="usim_datastore_h5",
                            description=f"UrbanSim input datastore for year {year}",
                        )
                    if "usim_mutable_data_dir" in usim_inputs:
                        cr.log_input(
                            usim_inputs["usim_mutable_data_dir"],
                            key="usim_mutable_data_dir",
                            description="UrbanSim mutable data directory",
                        )

                expected_usim_inputs = _expected_inputs_for(
                    "urbansim", settings, state, workspace
                )
                _merge_expected_inputs(usim_inputs, expected_usim_inputs)
                expected_usim_outputs = clean_expected_outputs(
                    _expected_outputs_for("urbansim", settings, state, workspace)
                )

                def _run_urbansim_step(
                    *,
                    settings: PilatesConfig,
                    state: WorkflowState,
                    workspace: Workspace,
                    adapter: ConsistProvenanceTracker,
                    usim_data_dir: str,
                    typed_coupler,
                    expected_outputs: dict,
                ):
                    if state.is_start_year() and settings.activitysim.warm_start_activities:
                        logger.info("[Main] Running warm start activities for ActivitySim.")
                        warm_start_activities(settings, state, workspace, adapter)

                    forecast_land_use(settings, year, state, workspace, adapter)
                    usim_output_path = expected_outputs.get("usim_datastore_h5")
                    if not usim_output_path:
                        usim_output_fname = usim_post.get_usim_datastore_fname(
                            settings, io="output", year=state.forecast_year
                        )
                        usim_output_path = os.path.join(usim_data_dir, usim_output_fname)
                    if os.path.exists(usim_output_path):
                        artifact = cr.log_output(
                            usim_output_path,
                            key="usim_datastore_h5",
                            description=f"UrbanSim datastore output for year {state.forecast_year}",
                        )
                        coupler.set(
                            "usim_datastore_h5", artifact or usim_output_path
                        )
                        if typed_coupler is not None and artifact is not None:
                            typed_coupler.usim_datastore_h5 = artifact

                scenario.run(
                    fn=_run_urbansim_step,
                    name=step_name,
                    model="urbansim",
                    year=year,
                    iteration=0,
                    inputs=usim_inputs,
                    output_paths=expected_usim_outputs or None,
                    load_inputs=False,
                    runtime_kwargs={
                        "settings": settings,
                        "state": state,
                        "workspace": workspace,
                        "adapter": adapter,
                        "usim_data_dir": usim_data_dir,
                        "typed_coupler": typed_coupler,
                        "expected_outputs": expected_usim_outputs,
                    },
                    **build_step_consist_kwargs("urbansim", settings),
                )

                state.complete_step(WorkflowState.Stage.land_use)

            # B. VEHICLE OWNERSHIP MODEL (ATLAS)
            if state.should_run(WorkflowState.Stage.vehicle_ownership_model):
                formatted_print(f"VEHICLE OWNERSHIP MODEL (ATLAS) FOR YEAR {state.forecast_year}")
                logger.info("[Main] Running ATLAS vehicle ownership model.")

                # Explicitly thread the mutable UrbanSim datastore across ATLAS sub-years.
                # This makes the sequential dependency clear (each sub-year reads the updated file
                # produced by the previous sub-year) and improves provenance clarity.
                coupler.pop("usim_datastore_h5", None)
                if state.run_info_path and os.path.exists(state.run_info_path):
                    previous_run_dir = os.path.dirname(state.run_info_path)
                    urbansim_datastore_dir = os.path.join(previous_run_dir, "urbansim", "data")
                else:
                    urbansim_datastore_dir = workspace.get_usim_mutable_data_dir()

                if state.is_start_year():
                    region = settings.run.region
                    region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
                    usim_datastore_fname = settings.urbansim.input_file_template.format(region_id=region_id)
                else:
                    usim_datastore_fname = settings.urbansim.output_file_template.format(year=state.forecast_year)

                usim_datastore_h5_path = os.path.join(urbansim_datastore_dir, usim_datastore_fname)

                # ATLAS Logic extraction
                factory = ModelFactory()
                preprocessor = factory.get_preprocessor("atlas", state, adapter, major_stage=WorkflowState.Stage.vehicle_ownership_model)
                runner = factory.get_runner("atlas", state, adapter, major_stage=WorkflowState.Stage.vehicle_ownership_model)
                postprocessor = factory.get_postprocessor("atlas", state, adapter, major_stage=WorkflowState.Stage.vehicle_ownership_model)

                warm_start_atlas = state.is_start_year()
                forecast = True
                yrs = [state.year] + [y + 2 for y in range(state.year, state.forecast_year, 2)] if forecast else [state.year]
                if not yrs and forecast: yrs = [state.forecast_year]

                # ATLAS Sub-loop
                for atlas_year in yrs:
                    # Create SubState
                    class AtlasSubState:
                        def __init__(self, parent_state, year):
                            self.__dict__ = parent_state.__dict__.copy()
                            self.year = year
                            self.current_year = year
                            self.forecast_year = year
                            self.main_forecast_year = parent_state.forecast_year
                            self.start_year = parent_state.start_year
                            self.full_settings = parent_state.full_settings
                            self.is_start_year = lambda: (year == parent_state.start_year)
                        def set_sub_stage_progress(self, sub_stage_progress):
                            state.set_sub_stage_progress(sub_stage_progress)

                    atlas_state = AtlasSubState(state, atlas_year)
                    step_name = f"atlas_{atlas_year}"

                    atlas_usim_input = coupler.get("usim_datastore_h5") or usim_datastore_h5_path
                    atlas_mutable_input_dir = workspace.get_atlas_mutable_input_dir()
                    step_inputs = {
                        "usim_datastore_h5": atlas_usim_input,
                        "atlas_mutable_input_dir": atlas_mutable_input_dir,
                    }
                    expected_atlas_inputs = _expected_inputs_for(
                        "atlas", settings, atlas_state, workspace
                    )
                    _merge_expected_inputs(step_inputs, expected_atlas_inputs)
                    expected_atlas_outputs = clean_expected_outputs(
                        _expected_outputs_for("atlas", settings, atlas_state, workspace)
                    )

                    def _run_atlas_step(
                        *,
                        atlas_state,
                        preprocessor,
                        runner,
                        postprocessor,
                        workspace: Workspace,
                        typed_coupler,
                        coupler,
                        usim_datastore_h5_path: str,
                        atlas_year: int,
                        expected_outputs: dict,
                    ):
                        # 1. Preprocess
                        preprocessor.update_state(atlas_state)
                        input_data = preprocessor.preprocess(workspace)

                        # 2. Run
                        runner.update_state(atlas_state)
                        try:
                            raw_outputs = runner.run(input_data, workspace)
                            # 3. Postprocess
                            postprocessor.update_state(atlas_state)
                            postprocessor.postprocess(raw_outputs, workspace)

                            atlas_output_dir = expected_outputs.get("atlas_output_dir")
                            if not atlas_output_dir:
                                atlas_output_dir = workspace.get_atlas_output_dir()
                            if os.path.exists(atlas_output_dir):
                                artifact = cr.log_output(
                                    atlas_output_dir,
                                    key="atlas_output_dir",
                                    description=f"ATLAS output directory for year {atlas_year}",
                                )
                                coupler.set(
                                    "atlas_output_dir", artifact or atlas_output_dir
                                )
                                if typed_coupler is not None and artifact is not None:
                                    typed_coupler.atlas_output_dir = artifact

                            # Capture the updated datastore container as an explicit output and
                            # thread it forward to the next ATLAS sub-year.
                            atlas_usim_output = expected_outputs.get("usim_datastore_h5")
                            if not atlas_usim_output:
                                atlas_usim_output = usim_datastore_h5_path
                            if os.path.exists(atlas_usim_output):
                                updated_h5 = cr.log_output(
                                    atlas_usim_output,
                                    key="usim_datastore_h5",
                                    description=f"UrbanSim datastore after ATLAS update for year {atlas_year}",
                                )
                                coupler.set(
                                    "usim_datastore_h5", updated_h5 or atlas_usim_output
                                )
                                if typed_coupler is not None and updated_h5 is not None:
                                    typed_coupler.usim_datastore_h5 = updated_h5
                            else:
                                logger.warning(
                                    f"[Main] UrbanSim datastore not found after ATLAS postprocess: {atlas_usim_output}"
                                )
                        except Exception as e:
                            logger.error(f"ATLAS failed for {atlas_year}: {e}")
                            sys.exit(1)

                        scenario.run(
                            fn=_run_atlas_step,
                            name=step_name,
                            model="atlas",
                            year=atlas_year,
                            iteration=0,
                            inputs=step_inputs,
                            outputs=list(expected_atlas_outputs.keys()) or None,
                            output_paths=expected_atlas_outputs or None,
                            cache_hydration="outputs-requested",
                            load_inputs=False,
                            runtime_kwargs={
                                "atlas_state": atlas_state,
                                "preprocessor": preprocessor,
                                "runner": runner,
                                "postprocessor": postprocessor,
                                "workspace": workspace,
                                "typed_coupler": typed_coupler,
                                "coupler": coupler,
                                "usim_datastore_h5_path": usim_datastore_h5_path,
                                "atlas_year": atlas_year,
                                "expected_outputs": expected_atlas_outputs,
                            },
                            **build_step_consist_kwargs("atlas", settings),
                        )

                state.complete_step(WorkflowState.Stage.vehicle_ownership_model)

            # C. SUPPLY/DEMAND LOOP
            if state.should_run(WorkflowState.Stage.supply_demand_loop):
                total_iters = settings.run.supply_demand_iters
                previous_beam_outputs = None

                for i in range(state.iteration, total_iters):
                    state.iteration = i
                    formatted_print(f"SUPPLY/DEMAND ITERATION {i+1}/{total_iters}")
                    activity_demand_outputs = None

                    # C1. ACTIVITY DEMAND
                    if state.should_run(WorkflowState.Stage.supply_demand_loop, i, WorkflowState.Stage.activity_demand):
                        formatted_print("ACTIVITY DEMAND MODEL")
                        step_name = f"activitysim_{year}_iter{i}"

                        asim_inputs = {}
                        asim_data_dir = workspace.get_asim_mutable_data_dir()
                        if os.path.exists(asim_data_dir):
                            asim_inputs["asim_mutable_data_dir"] = asim_data_dir
                            cr.log_input(
                                asim_data_dir,
                                key="asim_mutable_data_dir",
                                description=f"ActivitySim mutable data dir for year {year}, iter {i}",
                            )
                        if usim_inputs and "usim_datastore_h5" in usim_inputs:
                            asim_inputs["usim_datastore_h5"] = usim_inputs["usim_datastore_h5"]
                            cr.log_input(
                                usim_inputs["usim_datastore_h5"],
                                key="usim_datastore_h5",
                                description=f"UrbanSim datastore for ActivitySim year {year}, iter {i}",
                            )
                        zarr_skims_input = coupler.get("zarr_skims")
                        if zarr_skims_input:
                            asim_inputs["zarr_skims"] = zarr_skims_input
                            zarr_skims_path = artifact_to_path(
                                zarr_skims_input, workspace
                            )
                            if zarr_skims_path:
                                cr.log_input(
                                    zarr_skims_path,
                                    key="zarr_skims",
                                    description=f"ActivitySim compiled zarr skims for year {year}, iter {i}",
                                )
                        expected_asim_compile_inputs = _expected_inputs_for(
                            "activitysim_compile", settings, state, workspace
                        )
                        _merge_expected_inputs(
                            asim_inputs, expected_asim_compile_inputs
                        )
                        expected_asim_inputs = _expected_inputs_for(
                            "activitysim", settings, state, workspace
                        )
                        _merge_expected_inputs(asim_inputs, expected_asim_inputs)

                        activitysim_compile_holder = {
                            "input_store": None,
                            "compile_outputs": None,
                        }

                        if not state.asim_compiled:
                            compile_step_name = f"activitysim_compile_{year}"
                            expected_compile_outputs = clean_expected_outputs(
                                _expected_outputs_for(
                                    "activitysim_compile",
                                    settings,
                                    state,
                                    workspace,
                                    components=("runner",),
                                )
                            )

                            def _run_activitysim_compile_step(
                                *,
                                settings: PilatesConfig,
                                state: WorkflowState,
                                workspace: Workspace,
                                adapter: ConsistProvenanceTracker,
                                output_holder: dict,
                                typed_coupler,
                                expected_outputs: dict,
                            ):
                                factory = ModelFactory()
                                preprocessor = factory.get_preprocessor(
                                    "activitysim",
                                    state,
                                    adapter,
                                    major_stage=WorkflowState.Stage.activity_demand,
                                )
                                compile_runner = factory.get_runner(
                                    "activitysim_compile",
                                    state,
                                    adapter,
                                    major_stage=WorkflowState.Stage.activity_demand,
                                )

                                input_store = preprocessor.preprocess(workspace)
                                compile_outputs = compile_runner.run(
                                    input_store, workspace
                                )

                                output_holder["input_store"] = input_store
                                output_holder["compile_outputs"] = compile_outputs

                                zarr_record = None
                                if compile_outputs:
                                    for record in compile_outputs.all_records():
                                        if record.short_name == "zarr_skims":
                                            zarr_record = record
                                            break
                                zarr_output_path = expected_outputs.get("zarr_skims")
                                if not zarr_output_path and zarr_record is not None:
                                    zarr_output_path = zarr_record.file_path
                                if zarr_output_path and os.path.exists(zarr_output_path):
                                    artifact = cr.log_output(
                                        zarr_output_path,
                                        key="zarr_skims",
                                        description="ActivitySim compiled zarr skims",
                                    )
                                    coupler.set(
                                        "zarr_skims",
                                        artifact or zarr_output_path,
                                    )
                                    if typed_coupler is not None and artifact is not None:
                                        typed_coupler.zarr_skims = artifact

                            scenario.run(
                                fn=_run_activitysim_compile_step,
                                name=compile_step_name,
                                model="activitysim_compile",
                                year=year,
                                iteration=-1,
                                inputs=asim_inputs,
                                output_paths=expected_compile_outputs or None,
                                cache_mode="overwrite",
                                load_inputs=False,
                                runtime_kwargs={
                                    "settings": settings,
                                    "state": state,
                                    "workspace": workspace,
                                    "adapter": adapter,
                                    "output_holder": activitysim_compile_holder,
                                    "typed_coupler": typed_coupler,
                                    "expected_outputs": expected_compile_outputs,
                                },
                                **build_step_consist_kwargs(
                                    "activitysim_compile",
                                    settings,
                                    workspace_path=workspace.full_path,
                                ),
                            )

                        def _run_activitysim_step(
                            *,
                            settings: PilatesConfig,
                            state: WorkflowState,
                            workspace: Workspace,
                            adapter: ConsistProvenanceTracker,
                            output_holder: dict,
                            typed_coupler,
                            year: int,
                            iteration: int,
                            input_store: RecordStore,
                            compile_outputs: RecordStore,
                            expected_outputs: dict,
                        ):
                            combined_input_store = input_store
                            if combined_input_store is not None and compile_outputs:
                                combined_input_store = combined_input_store + compile_outputs
                            output_holder["activity_demand_outputs"] = run_activity_demand(
                                settings,
                                state,
                                workspace,
                                adapter,
                                input_store=combined_input_store,
                            )
                            asim_output_dir = expected_outputs.get("asim_output_dir")
                            if not asim_output_dir:
                                asim_output_dir = workspace.get_asim_output_dir()
                            if os.path.exists(asim_output_dir):
                                artifact = cr.log_output(
                                    asim_output_dir,
                                    key="asim_output_dir",
                                    description=f"ActivitySim output directory for year {year}, iter {iteration}",
                                )
                                coupler.set(
                                    "asim_output_dir", artifact or asim_output_dir
                                )
                                if typed_coupler is not None and artifact is not None:
                                    typed_coupler.asim_output_dir = artifact

                        activitysim_holder = {"activity_demand_outputs": None}
                        expected_asim_outputs = clean_expected_outputs(
                            _expected_outputs_for("activitysim", settings, state, workspace)
                        )
                        scenario.run(
                            fn=_run_activitysim_step,
                            name=step_name,
                            model="activitysim",
                            year=year,
                            iteration=i,
                            inputs=asim_inputs,
                            output_paths=expected_asim_outputs or None,
                            cache_mode="overwrite",
                            load_inputs=False,
                            runtime_kwargs={
                                "settings": settings,
                                "state": state,
                                "workspace": workspace,
                                "adapter": adapter,
                                "output_holder": activitysim_holder,
                                "typed_coupler": typed_coupler,
                                "year": year,
                                "iteration": i,
                                "input_store": activitysim_compile_holder["input_store"],
                                "compile_outputs": activitysim_compile_holder[
                                    "compile_outputs"
                                ],
                                "expected_outputs": expected_asim_outputs,
                            },
                            **build_step_consist_kwargs(
                                "activitysim",
                                settings,
                                workspace_path=workspace.full_path,
                            ),
                        )
                        activity_demand_outputs = activitysim_holder[
                            "activity_demand_outputs"
                        ]

                        state.complete_step(WorkflowState.Stage.supply_demand_loop, i, WorkflowState.Stage.activity_demand)

                    # C2. TRAFFIC ASSIGNMENT
                    if state.should_run(WorkflowState.Stage.supply_demand_loop, i, WorkflowState.Stage.traffic_assignment):
                        formatted_print("TRAFFIC ASSIGNMENT MODEL")
                        step_name = f"beam_{year}_iter{i}"

                        beam_inputs = {}
                        if activity_demand_outputs:
                            beam_inputs.update(activity_demand_outputs.to_mapping())
                        if previous_beam_outputs:
                            beam_inputs.update(previous_beam_outputs.to_mapping())
                        zarr_skims_input = coupler.get("zarr_skims")
                        if zarr_skims_input:
                            beam_inputs["zarr_skims"] = zarr_skims_input
                        expected_beam_inputs = _expected_inputs_for(
                            "beam", settings, state, workspace
                        )
                        _merge_expected_inputs(beam_inputs, expected_beam_inputs)
                        if beam_inputs:
                            cr.log_artifacts(beam_inputs, direction="input")
                        beam_mutable_dir = workspace.get_beam_mutable_data_dir()
                        if os.path.exists(beam_mutable_dir):
                            cr.log_input(
                                beam_mutable_dir,
                                key="beam_mutable_data_dir",
                                description=f"BEAM mutable data dir for year {year}, iter {i}",
                            )
                        # TODO: Log BEAM outputs from prior iterations as inputs once coupler mappings exist.

                        def _run_beam_step(
                            *,
                            settings: PilatesConfig,
                            state: WorkflowState,
                            workspace: Workspace,
                            adapter: ConsistProvenanceTracker,
                            activity_demand_outputs: RecordStore,
                            previous_beam_outputs: RecordStore,
                            output_holder: dict,
                            typed_coupler,
                            year: int,
                            iteration: int,
                            expected_outputs: dict,
                        ):
                            output_holder["beam_outputs"] = run_traffic_assignment(
                                settings,
                                state,
                                workspace,
                                adapter,
                                activity_demand_outputs,
                                previous_beam_outputs,
                            )
                            beam_output_dir = expected_outputs.get("beam_output_dir")
                            if not beam_output_dir:
                                beam_output_dir = workspace.get_beam_output_dir()
                            if os.path.exists(beam_output_dir):
                                artifact = cr.log_output(
                                    beam_output_dir,
                                    key="beam_output_dir",
                                    description=f"BEAM output directory for year {year}, iter {iteration}",
                                )
                                coupler.set(
                                    "beam_output_dir", artifact or beam_output_dir
                                )
                                if typed_coupler is not None and artifact is not None:
                                    typed_coupler.beam_output_dir = artifact

                            output_store = output_holder.get("beam_outputs")
                            update_coupler_from_beam_outputs(
                                output_store, coupler, typed_coupler, workspace
                            )

                        beam_holder = {"beam_outputs": None}
                        expected_beam_outputs = clean_expected_outputs(
                            _expected_outputs_for("beam", settings, state, workspace)
                        )

                        scenario.run(
                            fn=_run_beam_step,
                            name=step_name,
                            model="beam",
                            year=year,
                            iteration=i,
                            inputs=beam_inputs or None,
                            output_paths=expected_beam_outputs or None,
                            cache_mode="overwrite",
                            load_inputs=False,
                            runtime_kwargs={
                                "settings": settings,
                                "state": state,
                                "workspace": workspace,
                                "adapter": adapter,
                                "activity_demand_outputs": activity_demand_outputs,
                                "previous_beam_outputs": previous_beam_outputs,
                                "output_holder": beam_holder,
                                "typed_coupler": typed_coupler,
                                "year": year,
                                "iteration": i,
                                "expected_outputs": expected_beam_outputs,
                            },
                            **build_step_consist_kwargs(
                                "beam", settings, workspace_path=workspace.full_path
                            ),
                        )
                        previous_beam_outputs = beam_holder["beam_outputs"]

                        state.complete_step(WorkflowState.Stage.supply_demand_loop, i, WorkflowState.Stage.traffic_assignment)

                state.complete_step(WorkflowState.Stage.supply_demand_loop)

            # D. POST-PROCESSING
            if state.should_run(WorkflowState.Stage.postprocessing):
                formatted_print("POST-PROCESSING")
                def _run_postprocessing_step(
                    *,
                    settings: PilatesConfig,
                    state: WorkflowState,
                    workspace: Workspace,
                    tracker,
                ):
                    if "postprocessing" in settings:
                        process_event_file(settings, state, workspace, tracker)
                        copy_outputs_to_mep(settings, state, workspace, tracker)

                scenario.run(
                    fn=_run_postprocessing_step,
                    name=f"postprocessing_{year}",
                    model="postprocessing",
                    year=year,
                    cache_mode="overwrite",
                    load_inputs=False,
                    runtime_kwargs={
                        "settings": settings,
                        "state": state,
                        "workspace": workspace,
                        "tracker": tracker,
                    },
                    **build_step_consist_kwargs("postprocessing", settings),
                )
                state.complete_step(WorkflowState.Stage.postprocessing)

    formatted_print("SIMULATION COMPLETE")
    logger.info("[Main] Simulation complete.")


if __name__ == "__main__":
    main()
