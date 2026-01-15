from __future__ import annotations

import logging
import os
import sys
from typing import Callable, Dict, Mapping, Union

from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import ScenarioWithCoupler
from pilates.atlas.inputs import build_atlas_inputs
from pilates.utils.input_logging import log_inputs
from pilates.workflows.atlas_state import AtlasSubState
from pilates.workflows.orchestration import WorkflowStage, WorkflowStepSpec
from pilates.workflows.step_io import merge_model_expected_inputs
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_atlas_postprocess_step,
    make_atlas_preprocess_step,
    make_atlas_run_step,
)
from pilates.workflows.artifact_constants import USIM_DATASTORE_H5
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


def run_vehicle_ownership_stage(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: object,
    year: int,
    build_atlas_static_inputs_fallback: Callable[
        [Workspace], Mapping[str, Union[str, os.PathLike]]
    ],
) -> None:
    """
    Run the ATLAS vehicle ownership stage for the current forecast year.

    This stage executes ATLAS preprocess/run/postprocess for one or more
    sub-years, using the UrbanSim datastore (start-year or forecast output) as
    the primary input. It also wires in any static ATLAS inputs and handles
    sub-year execution via AtlasSubState.

    Parameters
    ----------
    scenario : ScenarioWithCoupler
        Consist scenario wrapper used to execute steps with provenance.
    state : WorkflowState
        Workflow state for year/stage coordination.
    settings : PilatesConfig
        Validated run configuration.
    workspace : Workspace
        Workspace managing run-local inputs/outputs.
    coupler : object
        Coupler used to read/write artifacts across steps.
    year : int
        Forecast year being simulated.
    build_atlas_static_inputs_fallback : Callable[[Workspace], Mapping[str, Union[str, os.PathLike]]]
        Fallback builder for static ATLAS inputs when not already present in
        the workspace input registry.
    """
    logger.info("[Main] Running ATLAS vehicle ownership model.")

    coupler.pop(USIM_DATASTORE_H5, None)
    if state.run_info_path and os.path.exists(state.run_info_path):
        previous_run_dir = os.path.dirname(state.run_info_path)
        urbansim_datastore_dir = os.path.join(previous_run_dir, "urbansim", "data")
    else:
        urbansim_datastore_dir = workspace.get_usim_mutable_data_dir()

    if state.is_start_year():
        region = settings.run.region
        region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
        usim_datastore_fname = settings.urbansim.input_file_template.format(
            region_id=region_id
        )
    else:
        usim_datastore_fname = settings.urbansim.output_file_template.format(
            year=state.forecast_year
        )

    usim_datastore_h5_path = os.path.join(urbansim_datastore_dir, usim_datastore_fname)

    forecast = True
    yrs = (
        [state.year] + [y + 2 for y in range(state.year, state.forecast_year, 2)]
        if forecast
        else [state.year]
    )
    if not yrs and forecast:
        yrs = [state.forecast_year]

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
        atlas_run_inputs: Dict[str, Any] = {}

        preprocess_steps = [
            WorkflowStepSpec(
                name="atlas_preprocess",
                step_func=make_atlas_preprocess_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder_atlas,
                ),
                inputs=atlas_preprocess_inputs,
            ),
            WorkflowStepSpec(
                name="atlas_run",
                step_func=make_atlas_run_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder_atlas,
                ),
                input_keys=[USIM_DATASTORE_H5],
                inputs=atlas_run_inputs or None,
            ),
        ]

        try:
            WorkflowStage(
                name="atlas",
                stage_type=state.Stage.vehicle_ownership_model,
                steps=preprocess_steps,
            ).run(
                scenario=scenario,
                state=atlas_state,
                settings=settings,
                workspace=workspace,
                coupler=coupler,
                outputs_holder=outputs_holder_atlas,
                name_suffix=str(atlas_year),
            )

            upstream_run = outputs_holder_atlas.atlas_run
            if upstream_run is None:
                raise RuntimeError("ATLAS run must complete before postprocess")
            postprocess_input_keys = [
                short_name for short_name, _, _ in upstream_run._iter_record_items()
            ]
            if not postprocess_input_keys:
                postprocess_input_keys = None

            postprocess_steps = [
                WorkflowStepSpec(
                    name="atlas_postprocess",
                    step_func=make_atlas_postprocess_step(
                        coupler=coupler,
                        outputs_holder=outputs_holder_atlas,
                    ),
                    input_keys=postprocess_input_keys,
                )
            ]
            WorkflowStage(
                name="atlas",
                stage_type=state.Stage.vehicle_ownership_model,
                steps=postprocess_steps,
            ).run(
                scenario=scenario,
                state=atlas_state,
                settings=settings,
                workspace=workspace,
                coupler=coupler,
                outputs_holder=outputs_holder_atlas,
                name_suffix=str(atlas_year),
            )
        except Exception:
            from pilates.utils.failure_handling import persist_state_on_error

            persist_state_on_error(state, f"ATLAS year {atlas_year}")
            sys.exit(1)
