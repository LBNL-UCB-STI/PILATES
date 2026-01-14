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
from pilates.atlas.inputs import build_atlas_inputs
from pilates.urbansim.inputs import build_urbansim_inputs
from pilates.utils.consist_types import ScenarioWithCoupler, TrackerLike
from pilates.utils.coupler_helpers import (
    artifact_to_path,
    clean_expected_outputs,
    update_coupler_from_beam_outputs,
    resolve_artifact_from_value,
)
from pilates.utils.input_logging import log_inputs
from pilates.utils.step_manifest import load_step_manifest, save_step_manifest
from pilates.workflows.atlas_state import AtlasSubState
from pilates.workflows.coupler_schema import PILATES_COUPLER_SCHEMA
from pilates.workflows.step_io import build_outputs, merge_model_expected_inputs

warnings.simplefilter(action="ignore", category=FutureWarning)
from workflow_state import WorkflowState

from pilates.workflows.step_runner import build_step_config, common_runtime_kwargs
from pilates.workflows.steps import (
    make_activitysim_compile_step,
    make_activitysim_postprocess_step,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
    make_atlas_postprocess_step,
    make_atlas_preprocess_step,
    make_atlas_run_step,
    make_beam_postprocess_step,
    make_beam_preprocess_step,
    make_beam_run_step,
    make_postprocessing_step,
    make_urbansim_postprocess_step,
    make_urbansim_preprocess_step,
    make_urbansim_run_step,
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
            "scratch": str(Path(output_path).resolve()),  # For temp files
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
            outputs_holder_year = StepOutputsHolder()

            # A. LAND USE FORECASTING
            # Forecasts demographic and economic changes using UrbanSim long-term simulation.
            # Outputs: Households, persons, jobs, land use → coupler → all later stages.
            # Cached: If no land-use-sensitive inputs changed, skip this stage.
            #
            if state.should_run(WorkflowState.Stage.land_use):
                formatted_print(f"LAND USE MODEL FOR YEAR {state.forecast_year}")

                usim_inputs, usim_input_descriptions = build_urbansim_inputs(
                    settings, state, workspace, year
                )
                log_inputs(usim_inputs, usim_input_descriptions)
                usim_inputs = merge_model_expected_inputs(
                    "urbansim", usim_inputs, settings, state, workspace
                )

                urbansim_steps = [
                    (
                        "urbansim_preprocess",
                        make_urbansim_preprocess_step(
                            coupler=coupler,
                            outputs_holder=outputs_holder_year,
                        ),
                        None,
                        usim_inputs,
                    ),
                    (
                        "urbansim_run",
                        make_urbansim_run_step(
                            coupler=coupler,
                            outputs_holder=outputs_holder_year,
                        ),
                        ["usim_mutable_data_dir", "usim_datastore_h5"],
                        None,
                    ),
                    (
                        "urbansim_postprocess",
                        make_urbansim_postprocess_step(
                            coupler=coupler,
                            outputs_holder=outputs_holder_year,
                        ),
                        ["usim_datastore_h5"],
                        None,
                    ),
                ]

                for step_key, step_func, input_keys, inputs in urbansim_steps:
                    validate_step_ready(step_key, outputs_holder_year)
                    step_config = build_step_config(
                        fn=step_func,
                        name=f"{step_key}_{year}",
                        model=step_key,
                        state=state,
                        iteration=0,
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
                        consist_kwargs=build_step_consist_kwargs(step_key, settings),
                    )
                    scenario.run(**step_config.to_kwargs())

                postprocess_outputs = outputs_holder_year.urbansim_postprocess
                run_outputs = outputs_holder_year.urbansim_run
                if (
                    postprocess_outputs is not None
                    and postprocess_outputs.usim_datastore_h5 is not None
                ):
                    usim_inputs["usim_datastore_h5"] = str(
                        postprocess_outputs.usim_datastore_h5
                    )
                elif (
                    run_outputs is not None
                    and run_outputs.usim_datastore_h5 is not None
                ):
                    usim_inputs["usim_datastore_h5"] = str(
                        run_outputs.usim_datastore_h5
                    )

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
                    outputs_holder_atlas = StepOutputsHolder()

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
                    atlas_preprocess_inputs = dict(step_inputs)
                    atlas_static_inputs = workspace.input_data.get("atlas")
                    if atlas_static_inputs is not None:
                        for key, value in atlas_static_inputs.to_mapping().items():
                            atlas_preprocess_inputs.setdefault(key, value)
                    else:
                        atlas_preprocess_inputs.update(
                            build_atlas_static_inputs_fallback(workspace)
                        )
                    atlas_run_inputs = {}
                    if "atlas_mutable_input_dir" in step_inputs:
                        atlas_run_inputs["atlas_mutable_input_dir"] = step_inputs[
                            "atlas_mutable_input_dir"
                        ]

                    atlas_steps = [
                        (
                            "atlas_preprocess",
                            make_atlas_preprocess_step(
                                coupler=coupler,
                                outputs_holder=outputs_holder_atlas,
                            ),
                            None,
                            atlas_preprocess_inputs,
                        ),
                        (
                            "atlas_run",
                            make_atlas_run_step(
                                coupler=coupler,
                                outputs_holder=outputs_holder_atlas,
                            ),
                            ["usim_datastore_h5"],
                            atlas_run_inputs or None,
                        ),
                        (
                            "atlas_postprocess",
                            make_atlas_postprocess_step(
                                coupler=coupler,
                                outputs_holder=outputs_holder_atlas,
                            ),
                            ["atlas_output_dir"],
                            None,
                        ),
                    ]

                    try:
                        for step_key, step_func, input_keys, inputs in atlas_steps:
                            validate_step_ready(step_key, outputs_holder_atlas)
                            step_config = build_step_config(
                                fn=step_func,
                                name=f"{step_key}_{atlas_year}",
                                model=step_key,
                                state=atlas_state,
                                iteration=0,
                                inputs=inputs or None,
                                input_keys=input_keys or None,
                                output_paths=None,
                                cache_hydration="inputs-missing",
                                load_inputs=False,
                                runtime_kwargs=common_runtime_kwargs(
                                    settings=settings,
                                    state=atlas_state,
                                    workspace=workspace,
                                ),
                                consist_kwargs=build_step_consist_kwargs(
                                    step_key, settings
                                ),
                            )
                            scenario.run(**step_config.to_kwargs())
                    except Exception:
                        from pilates.utils.failure_handling import (
                            persist_state_on_error,
                        )

                        persist_state_on_error(state, f"ATLAS year {atlas_year}")
                        sys.exit(1)

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
                    outputs_holder = StepOutputsHolder()
                    manifest_path = build_manifest_path(workspace, year, i)
                    manifest = load_step_manifest(manifest_path) or {}
                    stale_steps = []
                    for step_name, step_info in manifest.items():
                        outputs_class = STEP_OUTPUTS_CLASSES.get(step_name)
                        if outputs_class is None:
                            continue
                        outputs_data = step_info.get("outputs", {})
                        try:
                            outputs = deserialize_step_outputs(
                                outputs_class, outputs_data
                            )
                            validate = getattr(outputs, "validate", None)
                            if callable(validate):
                                validate()
                        except (AssertionError, FileNotFoundError) as exc:
                            logger.warning(
                                "Manifest outputs for %s are stale; will re-run (%s)",
                                step_name,
                                exc,
                            )
                            stale_steps.append(step_name)
                            continue
                        outputs_holder.set_attribute(step_name, outputs)

                    if stale_steps:
                        for step_name in stale_steps:
                            manifest.pop(step_name, None)
                        save_step_manifest(manifest, manifest_path)

                    # C1. ACTIVITY DEMAND
                    if state.should_run(
                        WorkflowState.Stage.supply_demand_loop,
                        i,
                        WorkflowState.Stage.activity_demand,
                    ):
                        formatted_print("ACTIVITY DEMAND MODEL")

                        preprocess_step = (
                            "activitysim_preprocess",
                            make_activitysim_preprocess_step(
                                coupler=coupler,
                                outputs_holder=outputs_holder,
                            ),
                            ["usim_datastore_h5"],
                            None,
                        )

                        step_key, step_func, input_keys, inputs = preprocess_step
                        if step_key not in manifest:
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
                                "cache_hit": bool(getattr(result, "cache_hit", False)),
                                "outputs": serialize_step_outputs(outputs),
                            }
                            save_step_manifest(manifest, manifest_path)
                        else:
                            logger.info("%s already completed (skipping)", step_key)

                        # ActivitySim Compilation: run once per year after preprocess.
                        if not state.asim_compiled:
                            upstream = outputs_holder.activitysim_preprocess
                            if upstream is None:
                                raise RuntimeError(
                                    "ActivitySim compile requires preprocess outputs."
                                )
                            compile_inputs = upstream.to_record_store().to_mapping()
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
                                outputs_holder=outputs_holder,
                            )
                            compile_config = build_step_config(
                                fn=activitysim_compile_step,
                                name=f"activitysim_compile_{year}",
                                model="activitysim_compile",
                                state=state,
                                iteration=-1,
                                inputs=compile_inputs or None,
                                output_paths=expected_compile_outputs or None,
                                cache_mode="overwrite",
                                load_inputs=False,
                                runtime_kwargs=common_runtime_kwargs(
                                    settings=settings,
                                    state=state,
                                    workspace=workspace,
                                    expected_outputs=expected_compile_outputs,
                                ),
                                consist_kwargs=build_step_consist_kwargs(
                                    "activitysim_compile",
                                    settings,
                                    workspace_path=workspace.full_path,
                                ),
                            )
                            scenario.run(**compile_config.to_kwargs())
                        else:
                            zarr_value = None
                            get_value = getattr(coupler, "get", None)
                            if callable(get_value):
                                zarr_value = resolve_artifact_from_value(
                                    get_value("zarr_skims"),
                                    key="zarr_skims",
                                    workspace=workspace,
                                )
                            zarr_path = artifact_to_path(zarr_value, workspace)
                            if not zarr_path:
                                candidate = os.path.join(
                                    workspace.get_asim_output_dir(),
                                    "cache",
                                    "skims.zarr",
                                )
                                if os.path.exists(candidate):
                                    zarr_path = candidate
                            if not zarr_path or not os.path.exists(zarr_path):
                                raise RuntimeError(
                                    "ActivitySim run requires zarr_skims, "
                                    "but none were found while asim_compiled=True."
                                )

                        activitysim_steps = [
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

                        for (
                            step_key,
                            step_func,
                            input_keys,
                            inputs,
                        ) in activitysim_steps:
                            if step_key in manifest:
                                logger.info("%s already completed (skipping)", step_key)
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
                                "cache_hit": bool(getattr(result, "cache_hit", False)),
                                "outputs": serialize_step_outputs(outputs),
                            }
                            save_step_manifest(manifest, manifest_path)

                        state.complete_step(
                            WorkflowState.Stage.supply_demand_loop,
                            i,
                            WorkflowState.Stage.activity_demand,
                        )

                    postprocess_outputs = outputs_holder.activitysim_postprocess
                    activity_demand_outputs = (
                        postprocess_outputs.to_record_store()
                        if postprocess_outputs is not None
                        else None
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
                        beam_preprocess_inputs: Dict[str, Any] = {}
                        if activity_demand_outputs is not None:
                            beam_preprocess_inputs.update(
                                activity_demand_outputs.to_mapping()
                            )
                        if previous_beam_outputs is not None:
                            beam_preprocess_inputs.update(
                                previous_beam_outputs.to_mapping()
                            )
                        if (
                            getattr(settings, "vehicle_ownership_model_enabled", False)
                            and i == 0
                        ):
                            if state.run_info_path and os.path.exists(
                                state.run_info_path
                            ):
                                previous_run_dir = os.path.dirname(state.run_info_path)
                                atlas_output_dir = os.path.join(
                                    previous_run_dir, "atlas", "atlas_output"
                                )
                            else:
                                atlas_output_dir = workspace.get_atlas_output_dir()
                            atlas_vehicle_path = os.path.join(
                                atlas_output_dir,
                                f"vehicles2_{state.forecast_year}.csv",
                            )
                            if not os.path.exists(atlas_vehicle_path):
                                atlas_vehicle_path = os.path.join(
                                    atlas_output_dir,
                                    f"vehicles2_{state.forecast_year - 1}.csv",
                                )
                            if os.path.exists(atlas_vehicle_path):
                                beam_preprocess_inputs.setdefault(
                                    "atlas_vehicles2_input", atlas_vehicle_path
                                )

                        beam_preprocess_input_keys = ["asim_output_dir"]
                        if getattr(settings, "vehicle_ownership_model_enabled", False):
                            beam_preprocess_input_keys.append("atlas_output_dir")

                        beam_postprocess_input_keys = ["beam_output_dir"]
                        if getattr(settings, "activity_demand_enabled", False):
                            beam_postprocess_input_keys.append("asim_output_dir")

                        beam_steps = [
                            (
                                "beam_preprocess",
                                make_beam_preprocess_step(
                                    coupler=coupler,
                                    outputs_holder=outputs_holder,
                                ),
                                beam_preprocess_input_keys,
                                beam_preprocess_inputs or None,
                            ),
                            (
                                "beam_run",
                                make_beam_run_step(
                                    coupler=coupler,
                                    outputs_holder=outputs_holder,
                                ),
                                ["beam_mutable_data_dir", "zarr_skims"],
                                None,
                            ),
                            (
                                "beam_postprocess",
                                make_beam_postprocess_step(
                                    coupler=coupler,
                                    outputs_holder=outputs_holder,
                                ),
                                beam_postprocess_input_keys,
                                None,
                            ),
                        ]

                        for step_key, step_func, input_keys, inputs in beam_steps:
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
                                    activity_demand_outputs=activity_demand_outputs,
                                    previous_beam_outputs=previous_beam_outputs,
                                ),
                                consist_kwargs=build_step_consist_kwargs(
                                    step_key,
                                    settings,
                                    workspace_path=workspace.full_path,
                                ),
                            )
                            scenario.run(**step_config.to_kwargs())

                        beam_run_outputs = outputs_holder.beam_run
                        beam_post_outputs = outputs_holder.beam_postprocess
                        combined_beam_outputs = None
                        if (
                            beam_run_outputs is not None
                            or beam_post_outputs is not None
                        ):
                            combined_beam_outputs = RecordStore()
                            if beam_run_outputs is not None:
                                combined_beam_outputs += (
                                    beam_run_outputs.to_record_store()
                                )
                            if beam_post_outputs is not None:
                                combined_beam_outputs += (
                                    beam_post_outputs.to_record_store()
                                )
                        previous_beam_outputs = combined_beam_outputs
                        if combined_beam_outputs is not None:
                            update_coupler_from_beam_outputs(
                                combined_beam_outputs, coupler, workspace
                            )

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
