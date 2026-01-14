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
from pathlib import Path
from typing import Optional, cast, Dict, Any

# Consist Imports (optional)
try:
    import consist
except ImportError:  # Consist optional dependency
    consist = None

# Legacy/PILATES Imports
from pilates.generic.model_factory import ModelFactory
from pilates.workspace import Workspace
from pilates.generic.initialization import Initialization
from pilates.utils.formatting import formatted_print
from pilates.utils.io import parse_args_and_settings
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_config import (
    build_scenario_consist_kwargs,
    build_step_consist_kwargs,
)
from pilates.activitysim.inputs import build_activitysim_inputs
from pilates.atlas.inputs import build_atlas_inputs
from pilates.beam.inputs import build_beam_inputs
from pilates.urbansim.inputs import build_urbansim_inputs
from pilates.utils.consist_types import ScenarioWithCoupler, TrackerLike
from pilates.utils.coupler_helpers import clean_expected_outputs
from pilates.utils.input_logging import log_inputs
from pilates.utils.step_manifest import load_step_manifest, save_step_manifest
from pilates.workflows.atlas_state import AtlasSubState
from pilates.workflows.coupler_schema import PILATES_COUPLER_SCHEMA
from pilates.workflows.step_io import (
    build_outputs,
    merge_expected_model_inputs,
    merge_model_expected_inputs,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
from workflow_state import WorkflowState

from pilates.workflows.step_runner import build_step_config, common_runtime_kwargs
from pilates.workflows.steps import (
    make_activitysim_compile_step,
    make_activitysim_postprocess_step,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
    make_atlas_step,
    make_beam_step,
    make_postprocessing_step,
    make_urbansim_step,
    deserialize_step_outputs,
    serialize_step_outputs,
    validate_step_ready,
    StepOutputsHolder,
    STEP_OUTPUTS_CLASSES,
)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_manifest_path(workspace: Workspace, year: int, iteration: int) -> Path:
    return (
        Path(workspace.full_path)
        / ".workflow"
        / f"year_{year}_iteration_{iteration}.yaml"
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
            "scratch": str(Path(output_path).resolve()), # For temp files
        },
        project_root=project_root_abs,
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
        try:
            from consist import SchemaValidatingCoupler
        except Exception:
            SchemaValidatingCoupler = None  # type: ignore[assignment]
        if SchemaValidatingCoupler is not None:
            scenario_kwargs["coupler"] = SchemaValidatingCoupler(
                schema=PILATES_COUPLER_SCHEMA
            )
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
            schema_attr = getattr(coupler, "schema", None)
            if schema_attr is not None:
                coupler.schema = PILATES_COUPLER_SCHEMA
        scenario.declare_outputs(
            "usim_datastore_h5",
            "asim_mutable_data_dir",
            "asim_output_dir",
            "beam_output_dir",
            "atlas_output_dir",
            "zarr_skims",
            "final_skims_omx",
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

            # A. LAND USE FORECASTING
            # Forecasts demographic and economic changes using UrbanSim long-term simulation.
            # Outputs: Households, persons, jobs, land use → coupler → all later stages.
            # Cached: If no land-use-sensitive inputs changed, skip this stage.
            #
            if state.should_run(WorkflowState.Stage.land_use):
                formatted_print(f"LAND USE MODEL FOR YEAR {state.forecast_year}")

                step_name = f"urbansim_{year}"

                usim_data_dir = workspace.get_usim_mutable_data_dir()
                usim_inputs, usim_input_descriptions = build_urbansim_inputs(
                    settings, state, workspace, year
                )
                log_inputs(usim_inputs, usim_input_descriptions)
                usim_inputs = merge_model_expected_inputs(
                    "urbansim", usim_inputs, settings, state, workspace
                )
                expected_usim_outputs = clean_expected_outputs(
                    build_outputs("urbansim", settings, state, workspace)
                )

                urbansim_step = make_urbansim_step(
                    coupler=coupler,
                    year=year,
                )
                urbansim_config = build_step_config(
                    fn=urbansim_step,
                    name=step_name,
                    model="urbansim",
                    state=state,
                    iteration=0,
                    inputs=usim_inputs,
                    output_paths=expected_usim_outputs or None,
                    load_inputs=False,
                    runtime_kwargs=common_runtime_kwargs(
                        settings=settings,
                        state=state,
                        workspace=workspace,
                        usim_data_dir=usim_data_dir,
                        expected_outputs=expected_usim_outputs,
                    ),
                    consist_kwargs=build_step_consist_kwargs("urbansim", settings),
                )
                scenario.run(**urbansim_config.to_kwargs())

                state.complete_step(WorkflowState.Stage.land_use)

            # B. VEHICLE OWNERSHIP MODEL (ATLAS)
            # ATLAS models vehicle fleet evolution between the current year and forecast year.
            # It runs at multiple intermediate years (every 2 years by default) to capture
            # gradual fleet turnover and policy impacts (e.g., vehicle electrification).
            #
            # Key pattern: Each ATLAS sub-year reads the UrbanSim output from the previous
            # iteration, updates it, and passes the updated file to the next sub-year.
            # This creates a sequential dependency chain for provenance clarity.
            #
            if state.should_run(WorkflowState.Stage.vehicle_ownership_model):
                formatted_print(
                    f"VEHICLE OWNERSHIP MODEL (ATLAS) FOR YEAR {state.forecast_year}"
                )
                logger.info("[Main] Running ATLAS vehicle ownership model.")

                # Explicitly thread the mutable UrbanSim datastore across ATLAS sub-years.
                # This makes the sequential dependency clear (each sub-year reads the updated file
                # produced by the previous sub-year) and improves provenance clarity.
                coupler.pop("usim_datastore_h5", None)
                if state.run_info_path and os.path.exists(state.run_info_path):
                    previous_run_dir = os.path.dirname(state.run_info_path)
                    urbansim_datastore_dir = os.path.join(
                        previous_run_dir, "urbansim", "data"
                    )
                else:
                    urbansim_datastore_dir = workspace.get_usim_mutable_data_dir()

                if state.is_start_year():
                    region = settings.run.region
                    region_id = settings.urbansim.region_mappings[
                        "region_to_region_id"
                    ][region]
                    usim_datastore_fname = settings.urbansim.input_file_template.format(
                        region_id=region_id
                    )
                else:
                    usim_datastore_fname = (
                        settings.urbansim.output_file_template.format(
                            year=state.forecast_year
                        )
                    )

                usim_datastore_h5_path = os.path.join(
                    urbansim_datastore_dir, usim_datastore_fname
                )

                # ATLAS Logic extraction
                factory = ModelFactory()
                preprocessor, runner, postprocessor = factory.get_components(
                    "atlas",
                    state,
                    major_stage=WorkflowState.Stage.vehicle_ownership_model,
                )

                warm_start_atlas = state.is_start_year()
                forecast = True
                yrs = (
                    [state.year]
                    + [y + 2 for y in range(state.year, state.forecast_year, 2)]
                    if forecast
                    else [state.year]
                )
                if not yrs and forecast:
                    yrs = [state.forecast_year]

                # ATLAS Sub-loop
                for atlas_year in yrs:
                    atlas_state = AtlasSubState(state, atlas_year)
                    step_name = f"atlas_{atlas_year}"

                    step_inputs, step_input_descriptions = build_atlas_inputs(
                        settings,
                        atlas_state,
                        workspace,
                        atlas_year,
                        coupler,
                        usim_datastore_h5_path,
                    )
                    log_inputs(step_inputs, step_input_descriptions)
                    step_inputs = merge_model_expected_inputs(
                        "atlas", step_inputs, settings, atlas_state, workspace
                    )
                    expected_atlas_outputs = clean_expected_outputs(
                        build_outputs("atlas", settings, atlas_state, workspace)
                    )

                    atlas_step = make_atlas_step(
                        coupler=coupler,
                    )
                    atlas_config = build_step_config(
                        fn=atlas_step,
                        name=step_name,
                        model="atlas",
                        state=atlas_state,
                        iteration=0,
                        inputs=step_inputs,
                        outputs=list(expected_atlas_outputs.keys()) or None,
                        output_paths=expected_atlas_outputs or None,
                        cache_hydration="outputs-requested",
                        load_inputs=False,
                        runtime_kwargs=common_runtime_kwargs(
                            settings=settings,
                            state=atlas_state,
                            workspace=workspace,
                            atlas_state=atlas_state,
                            base_state=state,
                            preprocessor=preprocessor,
                            runner=runner,
                            postprocessor=postprocessor,
                            usim_datastore_h5_path=usim_datastore_h5_path,
                            atlas_year=atlas_year,
                            expected_outputs=expected_atlas_outputs,
                        ),
                        consist_kwargs=build_step_consist_kwargs("atlas", settings),
                    )
                    scenario.run(**atlas_config.to_kwargs())

                state.complete_step(WorkflowState.Stage.vehicle_ownership_model)

            # C. SUPPLY/DEMAND LOOP
            # Iterates between activity demand generation (ActivitySim) and traffic
            # assignment (BEAM) until convergence. Consist handles expensive computation
            # caching across iterations:
            #
            # - ActivitySim Compilation (Iter 1 only): Preprocesses demand data.
            #   - Cached: If UrbanSim output unchanged since last run, skip.
            #   - Outputs: Compiled data passed to all iterations via compile_outputs_holder.
            #
            # - ActivitySim Main (All Iters): Generates activity schedules using compiled data.
            #   - Depends on: UrbanSim outputs (from land use stage) + compiled data.
            #   - Outputs: Activity plans → coupler → BEAM inputs.
            #   - Cached: If activity model inputs unchanged, skip.
            #
            # - BEAM Traffic Assignment (All Iters): Simulates traffic network.
            #   - Depends on: ActivitySim outputs (demand).
            #   - Outputs: Skims (travel times) → coupler → next ActivitySim iteration.
            #   - Cached: If demand unchanged, skip.
            #
            # Loop terminates when:
            # - Supplied iterations exhausted (settings.run.supply_demand_iters)
            #
            if state.should_run(WorkflowState.Stage.supply_demand_loop):
                total_iters = settings.run.supply_demand_iters
                previous_beam_outputs = None

                for i in range(state.iteration, total_iters):
                    state.iteration = i
                    formatted_print(f"SUPPLY/DEMAND ITERATION {i+1}/{total_iters}")
                    activity_demand_outputs = None

                    # C1. ACTIVITY DEMAND
                    if state.should_run(
                        WorkflowState.Stage.supply_demand_loop,
                        i,
                        WorkflowState.Stage.activity_demand,
                    ):
                        formatted_print("ACTIVITY DEMAND MODEL")
                        manifest_path = build_manifest_path(workspace, year, i)
                        manifest = load_step_manifest(manifest_path) or {}

                        outputs_holder = StepOutputsHolder()
                        for step_name, step_info in manifest.items():
                            outputs_class = STEP_OUTPUTS_CLASSES.get(step_name)
                            if outputs_class is None:
                                continue
                            outputs_data = step_info.get("outputs", {})
                            outputs = deserialize_step_outputs(
                                outputs_class, outputs_data
                            )
                            outputs_holder.set_attribute(step_name, outputs)

                        # ActivitySim Compilation: run once per year when needed.
                        if not state.asim_compiled:
                            asim_inputs, asim_input_descriptions = (
                                build_activitysim_inputs(
                                    settings,
                                    state,
                                    workspace,
                                    year,
                                    i,
                                    coupler,
                                    usim_inputs=usim_inputs,
                                    include_omx_skims=True,
                                )
                            )
                            log_inputs(asim_inputs, asim_input_descriptions)
                            asim_inputs = merge_expected_model_inputs(
                                ("activitysim_compile",),
                                asim_inputs,
                                settings,
                                state,
                                workspace,
                            )

                            expected_compile_outputs = clean_expected_outputs(
                                build_outputs(
                                    "activitysim_compile",
                                    settings,
                                    state,
                                    workspace,
                                    components=("runner",),
                                )
                            )
                            activitysim_compile_step = make_activitysim_compile_step(
                                coupler=coupler,
                            )
                            compile_config = build_step_config(
                                fn=activitysim_compile_step,
                                name=f"activitysim_compile_{year}",
                                model="activitysim_compile",
                                state=state,
                                iteration=-1,
                                inputs=asim_inputs,
                                output_paths=expected_compile_outputs or None,
                                cache_mode="overwrite",
                                load_inputs=False,
                                runtime_kwargs=common_runtime_kwargs(
                                    settings=settings,
                                    state=state,
                                    workspace=workspace,
                                    compile_outputs_holder={
                                        "input_store": None,
                                        "compile_outputs": None,
                                    },
                                    expected_outputs=expected_compile_outputs,
                                ),
                                consist_kwargs=build_step_consist_kwargs(
                                    "activitysim_compile",
                                    settings,
                                    workspace_path=workspace.full_path,
                                ),
                            )
                            scenario.run(**compile_config.to_kwargs())

                        activitysim_steps = [
                            (
                                "activitysim_preprocess",
                                make_activitysim_preprocess_step(
                                    coupler=coupler,
                                    outputs_holder=outputs_holder,
                                ),
                                ["usim_datastore_h5"],
                                None,
                            ),
                            (
                                "activitysim_run",
                                make_activitysim_run_step(
                                    coupler=coupler,
                                    outputs_holder=outputs_holder,
                                ),
                                ["asim_mutable_data_dir", "zarr_skims"],
                                None,
                            ),
                            (
                                "activitysim_postprocess",
                                make_activitysim_postprocess_step(
                                    coupler=coupler,
                                    outputs_holder=outputs_holder,
                                ),
                                ["asim_output_dir"],
                                {
                                    "usim_mutable_data_dir": workspace.get_usim_mutable_data_dir()
                                },
                            ),
                        ]

                        for step_key, step_func, input_keys, inputs in activitysim_steps:
                            if step_key in manifest:
                                logger.info(
                                    "%s already completed (skipping)", step_key
                                )
                                continue
                            validate_step_ready(step_key, outputs_holder)
                            step_config = build_step_config(
                                fn=step_func,
                                name=f"{step_key}_{year}_iter{i}",
                                model=step_key,
                                state=state,
                                inputs=inputs or None,
                                input_keys=input_keys or None,
                                output_paths=None,
                                cache_hydration="inputs-missing",
                                load_inputs=False,
                                runtime_kwargs=common_runtime_kwargs(
                                    settings=settings,
                                    state=state,
                                    workspace=workspace,
                                ),
                                consist_kwargs=build_step_consist_kwargs(
                                    step_key,
                                    settings,
                                    workspace_path=workspace.full_path,
                                ),
                            )
                            result = scenario.run(**step_config.to_kwargs())
                            outputs = outputs_holder.get_attribute(step_key)
                            if outputs is None:
                                raise RuntimeError(
                                    f"{step_key} did not populate outputs_holder"
                                )
                            manifest[step_key] = {
                                "completed_at": datetime.now().isoformat(),
                                "cache_hit": bool(
                                    getattr(result, "cache_hit", False)
                                ),
                                "outputs": serialize_step_outputs(outputs),
                            }
                            save_step_manifest(manifest, manifest_path)

                        postprocess_outputs = outputs_holder.activitysim_postprocess
                        activity_demand_outputs = (
                            postprocess_outputs.to_record_store()
                            if postprocess_outputs is not None
                            else None
                        )

                        state.complete_step(
                            WorkflowState.Stage.supply_demand_loop,
                            i,
                            WorkflowState.Stage.activity_demand,
                        )

                    # C2. TRAFFIC ASSIGNMENT
                    if state.should_run(
                        WorkflowState.Stage.supply_demand_loop,
                        i,
                        WorkflowState.Stage.traffic_assignment,
                    ):
                        # Consist integration: scenario.run handles caching/provenance;
                        # coupler bridges BEAM outputs into downstream stages.
                        formatted_print("TRAFFIC ASSIGNMENT MODEL")
                        step_name = f"beam_{year}_iter{i}"

                        (
                            beam_inputs,
                            beam_mutable_dir,
                            beam_mutable_description,
                        ) = build_beam_inputs(
                            settings,
                            state,
                            workspace,
                            year,
                            i,
                            coupler,
                            activity_demand_outputs=activity_demand_outputs,
                            previous_beam_outputs=previous_beam_outputs,
                        )
                        beam_inputs = merge_model_expected_inputs(
                            "beam", beam_inputs, settings, state, workspace
                        )

                        # BEAM Traffic Assignment: Simulate network equilibration.
                        # Uses activity demand from ActivitySim (via activity_demand_outputs).
                        # Generates skims (travel times) that feed back to ActivitySim
                        # for next iteration via coupler (coupler["zarr_skims"]).
                        # Consist handles caching: if demand unchanged → skip BEAM.
                        beam_outputs_holder = {"beam_outputs": None}
                        expected_beam_outputs = clean_expected_outputs(
                            build_outputs("beam", settings, state, workspace)
                        )
                        beam_step = make_beam_step(
                            coupler=coupler,
                        )
                        beam_config = build_step_config(
                            fn=beam_step,
                            name=step_name,
                            model="beam",
                            state=state,
                            inputs=beam_inputs or None,
                            output_paths=expected_beam_outputs or None,
                            cache_mode="overwrite",
                            load_inputs=False,
                            runtime_kwargs=common_runtime_kwargs(
                                settings=settings,
                                state=state,
                                workspace=workspace,
                                activity_demand_outputs=activity_demand_outputs,
                                previous_beam_outputs=previous_beam_outputs,
                                beam_inputs=beam_inputs,
                                beam_mutable_dir=beam_mutable_dir,
                                beam_mutable_description=beam_mutable_description,
                                beam_outputs_holder=beam_outputs_holder,
                                expected_outputs=expected_beam_outputs,
                            ),
                            consist_kwargs=build_step_consist_kwargs(
                                "beam",
                                settings,
                                workspace_path=workspace.full_path,
                            ),
                        )
                        scenario.run(**beam_config.to_kwargs())
                        previous_beam_outputs = beam_outputs_holder["beam_outputs"]

                        state.complete_step(
                            WorkflowState.Stage.supply_demand_loop,
                            i,
                            WorkflowState.Stage.traffic_assignment,
                        )

                state.complete_step(WorkflowState.Stage.supply_demand_loop)

            # D. POST-PROCESSING
            # Validation and output generation. Processes raw model outputs into
            # final deliverables (e.g., travel metrics, visualization-ready data).
            #
            if state.should_run(WorkflowState.Stage.postprocessing):
                formatted_print("POST-PROCESSING")

                postprocessing_step = make_postprocessing_step()
                postprocessing_config = build_step_config(
                    fn=postprocessing_step,
                    name=f"postprocessing_{year}",
                    model="postprocessing",
                    state=state,
                    cache_mode="overwrite",
                    load_inputs=False,
                    runtime_kwargs=common_runtime_kwargs(
                        settings=settings,
                        state=state,
                        workspace=workspace,
                    ),
                    consist_kwargs=build_step_consist_kwargs(
                        "postprocessing", settings
                    ),
                )
                scenario.run(**postprocessing_config.to_kwargs())
                state.complete_step(WorkflowState.Stage.postprocessing)

    formatted_print("SIMULATION COMPLETE")
    logger.info("[Main] Simulation complete.")


if __name__ == "__main__":
    main()
