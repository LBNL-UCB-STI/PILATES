from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Union, cast

from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.workspace import Workspace
from workflow_state import WorkflowState

from pilates.utils.formatting import formatted_print
from pilates.utils.coupler_helpers import archive_copy_now, flush_archive_queue
from pilates.utils.input_logging import log_inputs
from pilates.workflows.binding import build_binding_plan
from pilates.workflows.step_io import merge_model_expected_inputs
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_urbansim_postprocess_step,
    make_urbansim_preprocess_step,
    make_urbansim_run_step,
    urbansim_run_output_paths,
)
from pilates.workflows.orchestration import (
    ManifestConfig,
    StageRunner,
    StepRef,
    run_workflow,
)
from pilates.workflows.coupler_namespace import resolve_coupler_value
from pilates.workflows.outputs_base import step_output_handoff_mapping
from pilates.workflows.artifact_keys import (
    FINAL_SKIMS_OMX,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_FORECAST_OUTPUT,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.urbansim.inputs import build_urbansim_inputs

logger = logging.getLogger(__name__)


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
    downstream stages.

    The stage keeps two semantic datastore handles alive:
    - ``usim_datastore_base_h5`` for the static/exogenous baseline role
    - ``usim_datastore_h5`` for the current mutable handoff role

    Those roles may resolve to the same physical H5 in some runs, but the
    distinction is still preserved for restart-sensitive provenance and
    downstream contract clarity. The postprocess output datastore, when
    present, is preferred for the current-role handoff; otherwise the run
    output datastore is used.

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
    stage_runner = StageRunner(
        stage_name="land_use",
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder_year,
        name_suffix=str(year),
        manifest_config=manifest_config,
        run_workflow_fn=run_workflow,
    )
    preprocess_inputs = dict(usim_inputs)
    if preprocess_inputs.get(USIM_DATASTORE_BASE_H5) == preprocess_inputs.get(
        USIM_DATASTORE_CURRENT_H5
    ):
        # Avoid redundantly passing the same physical H5 twice into preprocess
        # while still preserving both semantic handles in ``usim_inputs``.
        preprocess_inputs.pop(USIM_DATASTORE_CURRENT_H5, None)
    preprocess_keys = list(preprocess_inputs.keys())
    if FINAL_SKIMS_OMX not in preprocess_keys:
        preprocess_keys.append(FINAL_SKIMS_OMX)
    preprocess_binding = build_binding_plan(
        step_name="urbansim_preprocess",
        coupler=coupler,
        explicit_inputs=preprocess_inputs,
        optional_keys=preprocess_keys,
    )

    preprocess_step = make_urbansim_preprocess_step(
        coupler=coupler,
        outputs_holder=outputs_holder_year,
    )
    stage_runner.run_step(
        step=StepRef(
            name="urbansim_preprocess",
            step_func=preprocess_step,
            binding=preprocess_binding,
        )
    )

    upstream_preprocess = outputs_holder_year.urbansim_preprocess
    if upstream_preprocess is None:
        raise RuntimeError("UrbanSim preprocess must complete first")

    run_inputs = step_output_handoff_mapping(upstream_preprocess, coupler=coupler)
    # Some preprocessors materialize key artifacts via explicit logging rather
    # than RecordStore outputs. Preserve the restart-critical datastore roles
    # explicitly, but prefer the coupler-published artifact when available so
    # downstream input identity stays stable across workspace restaging. Do not
    # leak preprocess-only component-local inputs such as ``usim_source_data_dir``
    # into the UrbanSim run binding contract.
    for key in (USIM_DATASTORE_BASE_H5, USIM_DATASTORE_CURRENT_H5):
        resolved = resolve_coupler_value(coupler, key)
        value = (
            resolved.value
            if resolved.value is not None
            else usim_inputs.get(key)
        )
        logger.debug(
            "[land_use] Runtime handoff for %s resolved via %s (storage_key=%s, "
            "value_type=%s, fallback_used=%s)",
            key,
            resolved.source,
            resolved.storage_key,
            type(value).__name__ if value is not None else None,
            resolved.value is None and usim_inputs.get(key) is not None,
        )
        if value is not None:
            run_inputs.setdefault(key, value)
    run_binding = build_binding_plan(
        step_name="urbansim_run",
        coupler=coupler,
        explicit_inputs=run_inputs,
        optional_keys=list(run_inputs.keys()),
    )
    postprocess_binding = build_binding_plan(
        step_name="urbansim_postprocess",
        coupler=coupler,
        explicit_inputs={USIM_DATASTORE_CURRENT_H5: usim_inputs.get(USIM_DATASTORE_CURRENT_H5)},
        fallback_inputs=usim_inputs,
        required_keys=[USIM_DATASTORE_CURRENT_H5],
    )
    if postprocess_binding.missing_required:
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
            binding=run_binding,
            output_paths_provider=urbansim_run_output_paths,
        ),
        StepRef(
            name="urbansim_postprocess",
            step_func=make_urbansim_postprocess_step(
                coupler=coupler,
                outputs_holder=outputs_holder_year,
            ),
            binding=postprocess_binding,
        ),
    ]

    stage_runner.run(steps=run_steps)

    postprocess_outputs = outputs_holder_year.urbansim_postprocess
    run_outputs = outputs_holder_year.urbansim_run
    if run_outputs is not None and run_outputs.usim_datastore_h5:
        usim_inputs[USIM_FORECAST_OUTPUT] = str(run_outputs.usim_datastore_h5)
        usim_inputs[USIM_POPULATION_SOURCE_H5] = str(run_outputs.usim_datastore_h5)
    if postprocess_outputs is not None and postprocess_outputs.usim_datastore_h5:
        usim_inputs[USIM_DATASTORE_CURRENT_H5] = str(
            postprocess_outputs.usim_datastore_h5
        )
    elif run_outputs is not None and run_outputs.usim_datastore_h5:
        usim_inputs[USIM_DATASTORE_CURRENT_H5] = str(run_outputs.usim_datastore_h5)

    # Preserve the base-role handle as the static/exogenous input contract. If
    # current/base collapsed earlier in the run, keep that role explicit here.
    if (
        USIM_DATASTORE_BASE_H5 not in usim_inputs
        and USIM_DATASTORE_CURRENT_H5 in usim_inputs
    ):
        usim_inputs[USIM_DATASTORE_BASE_H5] = usim_inputs[USIM_DATASTORE_CURRENT_H5]

    # Keep restart-critical UrbanSim H5 artifacts durable at stage boundaries.
    archive_copy_now(
        key=USIM_DATASTORE_BASE_H5,
        path=usim_inputs.get(USIM_DATASTORE_BASE_H5),
    )
    archive_copy_now(
        key=USIM_DATASTORE_CURRENT_H5,
        path=usim_inputs.get(USIM_DATASTORE_CURRENT_H5),
    )
    archive_copy_now(
        key=USIM_FORECAST_OUTPUT,
        path=usim_inputs.get(USIM_FORECAST_OUTPUT),
    )
    archive_copy_now(
        key=USIM_POPULATION_SOURCE_H5,
        path=usim_inputs.get(USIM_POPULATION_SOURCE_H5),
    )
    urbansim_settings = settings.urbansim
    if urbansim_settings is None:
        raise RuntimeError("UrbanSim config is required for the land use stage.")

    forecast_year = state.forecast_year if state.forecast_year is not None else year
    usim_forecast_output_path = os.path.join(
        workspace.get_usim_mutable_data_dir(),
        urbansim_settings.output_file_template.format(year=forecast_year),
    )
    archive_copy_now(
        key=f"usim_year_output_h5_{forecast_year}",
        path=usim_forecast_output_path,
    )
    flush_archive_queue(timeout=300, fail_on_timeout=False)

    return dict(usim_inputs)
