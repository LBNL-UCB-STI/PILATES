from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union, cast

from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.workspace import Workspace
from workflow_state import WorkflowState

from pilates.utils.formatting import formatted_print
from pilates.utils.coupler_helpers import enqueue_archive_copy, flush_archive_queue
from pilates.utils.input_logging import log_inputs
from pilates.workflows.input_resolution import resolve_step_inputs
from pilates.workflows.step_io import merge_model_expected_inputs
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_urbansim_postprocess_step,
    make_urbansim_preprocess_step,
    make_urbansim_run_step,
)
from pilates.workflows.orchestration import ManifestConfig, StepRef, run_workflow
from pilates.workflows.outputs_base import step_output_handoff_mapping
from pilates.workflows.artifact_keys import (
    FINAL_SKIMS_OMX,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
)
from pilates.urbansim.inputs import build_urbansim_inputs


def _build_land_use_manifest_path(workspace: Workspace, year: int) -> Path:
    return Path(workspace.full_path) / ".workflow" / f"land_use_year_{year}.yaml"


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
    formatted_print(f"LAND USE MODEL FOR YEAR {year}")

    usim_inputs, usim_input_descriptions = build_urbansim_inputs(
        settings, state, workspace, year
    )
    log_inputs(usim_inputs, cast(Dict[str, Optional[str]], usim_input_descriptions))
    usim_inputs = merge_model_expected_inputs(
        "urbansim", usim_inputs, settings, state, workspace
    )
    manifest_config = ManifestConfig(
        path=_build_land_use_manifest_path(workspace=workspace, year=year)
    )
    preprocess_inputs = dict(usim_inputs)
    if preprocess_inputs.get(USIM_DATASTORE_BASE_H5) == preprocess_inputs.get(
        USIM_DATASTORE_CURRENT_H5
    ):
        preprocess_inputs.pop(USIM_DATASTORE_CURRENT_H5, None)
    preprocess_keys = list(preprocess_inputs.keys())
    if FINAL_SKIMS_OMX not in preprocess_keys:
        preprocess_keys.append(FINAL_SKIMS_OMX)
    preprocess_resolution = resolve_step_inputs(
        keys=preprocess_keys,
        coupler=coupler,
        explicit_inputs=preprocess_inputs,
    )

    preprocess_step = make_urbansim_preprocess_step(
        coupler=coupler,
        outputs_holder=outputs_holder_year,
    )
    preprocess_steps = [
        StepRef(
            name="urbansim_preprocess",
            step_func=preprocess_step,
            inputs=preprocess_resolution.stepref_inputs(),
            input_keys=preprocess_resolution.stepref_input_keys(),
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
        manifest_config=manifest_config,
    )

    upstream_preprocess = outputs_holder_year.urbansim_preprocess
    if upstream_preprocess is None:
        raise RuntimeError("UrbanSim preprocess must complete first")

    run_inputs = step_output_handoff_mapping(upstream_preprocess, coupler=coupler)
    # Some preprocessors materialize key artifacts via explicit logging rather
    # than RecordStore outputs. Keep those handoff outputs, but also preserve
    # declared UrbanSim inputs so run identity/provenance still reflects the
    # restart-critical datastore dependencies.
    for key, value in usim_inputs.items():
        if value is not None:
            run_inputs.setdefault(key, value)
    run_resolution = resolve_step_inputs(
        keys=run_inputs.keys(),
        explicit_inputs=run_inputs,
    )
    postprocess_resolution = resolve_step_inputs(
        keys=[USIM_DATASTORE_CURRENT_H5],
        coupler=coupler,
        fallback_inputs=usim_inputs,
        required_keys=[USIM_DATASTORE_CURRENT_H5],
    )
    if postprocess_resolution.missing_required:
        raise RuntimeError(
            "UrbanSim postprocess requires usim_datastore_h5 but it could not be "
            "resolved from explicit inputs, coupler, or fallback inputs."
        )

    run_steps = [
        StepRef(
            name="urbansim_run",
            step_func=make_urbansim_run_step(
                coupler=coupler,
                outputs_holder=outputs_holder_year,
            ),
            inputs=run_resolution.stepref_inputs(),
            input_keys=run_resolution.stepref_input_keys(),
        ),
        StepRef(
            name="urbansim_postprocess",
            step_func=make_urbansim_postprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder_year,
            ),
            inputs=postprocess_resolution.stepref_inputs(),
            input_keys=postprocess_resolution.stepref_input_keys(),
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
        manifest_config=manifest_config,
    )

    postprocess_outputs = outputs_holder_year.urbansim_postprocess
    run_outputs = outputs_holder_year.urbansim_run
    if postprocess_outputs is not None and postprocess_outputs.usim_datastore_h5:
        usim_inputs[USIM_DATASTORE_CURRENT_H5] = str(
            postprocess_outputs.usim_datastore_h5
        )
    elif run_outputs is not None and run_outputs.usim_datastore_h5:
        usim_inputs[USIM_DATASTORE_CURRENT_H5] = str(run_outputs.usim_datastore_h5)

    # Preserve base semantics as the static/exogenous input.
    if (
        USIM_DATASTORE_BASE_H5 not in usim_inputs
        and USIM_DATASTORE_CURRENT_H5 in usim_inputs
    ):
        usim_inputs[USIM_DATASTORE_BASE_H5] = usim_inputs[USIM_DATASTORE_CURRENT_H5]

    # Keep restart-critical UrbanSim H5 artifacts durable at stage boundaries.
    enqueue_archive_copy(
        key=USIM_DATASTORE_BASE_H5,
        path=usim_inputs.get(USIM_DATASTORE_BASE_H5),
    )
    enqueue_archive_copy(
        key=USIM_DATASTORE_CURRENT_H5,
        path=usim_inputs.get(USIM_DATASTORE_CURRENT_H5),
    )
    urbansim_settings = settings.urbansim
    if urbansim_settings is None:
        raise RuntimeError("UrbanSim config is required for the land use stage.")

    forecast_year = state.forecast_year if state.forecast_year is not None else year
    usim_forecast_output_path = os.path.join(
        workspace.get_usim_mutable_data_dir(),
        urbansim_settings.output_file_template.format(year=forecast_year),
    )
    enqueue_archive_copy(
        key=f"usim_year_output_h5_{forecast_year}",
        path=usim_forecast_output_path,
    )
    flush_archive_queue(timeout=300, fail_on_timeout=True)

    return dict(usim_inputs)
