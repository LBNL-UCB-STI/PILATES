from __future__ import annotations

import os
from typing import Dict, Union

from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.workspace import Workspace
from workflow_state import WorkflowState

from pilates.utils.formatting import formatted_print
from pilates.utils.input_logging import log_inputs
from pilates.workflows.step_io import merge_model_expected_inputs
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_urbansim_postprocess_step,
    make_urbansim_preprocess_step,
    make_urbansim_run_step,
)
from pilates.workflows.orchestration import StepRef, run_workflow
from pilates.workflows.artifact_constants import USIM_DATASTORE_H5
from pilates.urbansim.inputs import build_urbansim_inputs


def run_land_use_stage(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
    year: int,
    outputs_holder_year: StepOutputsHolder,
) -> Dict[str, Union[str, os.PathLike]]:
    """
    Run the UrbanSim land-use stage and return updated UrbanSim inputs.

    This stage is responsible for land-use evolution. It prepares UrbanSim
    inputs (including any pre-existing datastore), executes preprocess/run/
    postprocess steps, and then updates the UrbanSim datastore reference for
    downstream stages. The postprocess output datastore, when present, is
    preferred; otherwise the run output datastore is used.

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
    coupler : CouplerProtocol
        Consist coupler for reading/writing artifacts across steps.
    year : int
        Forecast year being simulated.
    outputs_holder_year : StepOutputsHolder
        Output holder used to capture UrbanSim step outputs for this year.

    Returns
    -------
    Dict[str, Union[str, PathLike]]
        Updated UrbanSim input mapping, including the latest datastore path.
    """
    formatted_print(f"LAND USE MODEL FOR YEAR {state.forecast_year}")

    usim_inputs, usim_input_descriptions = build_urbansim_inputs(
        settings, state, workspace, year
    )
    log_inputs(usim_inputs, usim_input_descriptions)
    usim_inputs = merge_model_expected_inputs(
        "urbansim", usim_inputs, settings, state, workspace
    )

    preprocess_steps = [
        StepRef(
            name="urbansim_preprocess",
            step_func=make_urbansim_preprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder_year,
            ),
            inputs=usim_inputs,
        ),
    ]

    run_workflow(
        stage_name="land_use",
        steps=preprocess_steps,
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder_year,
        name_suffix=str(year),
    )

    upstream_preprocess = outputs_holder_year.urbansim_preprocess
    if upstream_preprocess is None:
        raise RuntimeError("UrbanSim preprocess must complete first")

    run_input_keys = [
        short_name for short_name, _, _ in upstream_preprocess._iter_record_items()
    ]
    if USIM_DATASTORE_H5 in usim_inputs:
        run_input_keys.append(USIM_DATASTORE_H5)
    if not run_input_keys:
        run_input_keys = None

    run_steps = [
        StepRef(
            name="urbansim_run",
            step_func=make_urbansim_run_step(
                coupler=coupler,
                outputs_holder=outputs_holder_year,
            ),
            input_keys=run_input_keys,
        ),
        StepRef(
            name="urbansim_postprocess",
            step_func=make_urbansim_postprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder_year,
            ),
            input_keys=[USIM_DATASTORE_H5],
        ),
    ]

    run_workflow(
        stage_name="land_use",
        steps=run_steps,
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder_year,
        name_suffix=str(year),
    )

    postprocess_outputs = outputs_holder_year.urbansim_postprocess
    run_outputs = outputs_holder_year.urbansim_run
    if postprocess_outputs is not None and postprocess_outputs.usim_datastore_h5:
        usim_inputs[USIM_DATASTORE_H5] = str(postprocess_outputs.usim_datastore_h5)
    elif run_outputs is not None and run_outputs.usim_datastore_h5:
        usim_inputs[USIM_DATASTORE_H5] = str(run_outputs.usim_datastore_h5)

    return usim_inputs
