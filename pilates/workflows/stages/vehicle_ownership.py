from __future__ import annotations

import logging
import os
import sys
from typing import Callable, Dict, Mapping, Union, Any, Optional

from pilates.config.models import PilatesConfig
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.atlas.inputs import build_atlas_inputs, atlas_static_input_keys
from pilates.utils.input_logging import log_inputs
from pilates.workflows.input_resolution import resolve_step_inputs
from pilates.workflows.atlas_state import AtlasSubState
from pilates.workflows.orchestration import StepRef, run_workflow
from pilates.workflows.step_io import merge_model_expected_inputs
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_atlas_postprocess_step,
    make_atlas_preprocess_step,
    make_atlas_run_step,
)
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
)
from pilates.urbansim.inputs import build_urbansim_inputs
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


def _atlas_sub_years(state: WorkflowState) -> list[int]:
    """
    Return ATLAS sub-years within the current workflow interval.

    ATLAS advances in biannual increments. Keep years bounded to the parent
    interval and never overshoot ``state.forecast_year``.
    """
    years = [state.year]
    if state.forecast_year <= state.year:
        return years
    years.extend(range(state.year + 2, state.forecast_year + 1, 2))
    return years


def _resolve_input_path(value: Any, workspace: Workspace) -> Union[str, None]:
    resolved = artifact_to_path(value, workspace)
    if resolved:
        return resolved
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    return None


def select_atlas_usim_input_path(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    fallback_current_path: Optional[Union[str, os.PathLike]],
    fallback_default_path: Optional[Union[str, os.PathLike]],
    prefer_forecast_output: bool = True,
) -> str:
    """
    Resolve the UrbanSim datastore path used by ATLAS preprocess.

    Precedence (when ``prefer_forecast_output=True``):
    1. Forecast output datastore (year-scoped snapshots).
    2. Current datastore resolved from UrbanSim input builder.
    3. Legacy default path.

    Precedence (when ``prefer_forecast_output=False``):
    1. Current datastore resolved from UrbanSim input builder.
    2. Legacy default path.
    3. Forecast output datastore.
    """
    if state.run_info_path and os.path.exists(state.run_info_path):
        previous_run_dir = os.path.dirname(state.run_info_path)
        usim_dir = os.path.join(previous_run_dir, "urbansim", "data")
    else:
        usim_dir = workspace.get_usim_mutable_data_dir()

    forecast_output_path = os.path.join(
        usim_dir,
        settings.urbansim.output_file_template.format(year=state.forecast_year),
    )

    current_candidate = (
        os.fspath(fallback_current_path)
        if isinstance(fallback_current_path, (str, os.PathLike))
        else None
    )
    default_candidate = (
        os.fspath(fallback_default_path)
        if isinstance(fallback_default_path, (str, os.PathLike))
        else None
    )

    if prefer_forecast_output:
        candidates = [forecast_output_path, current_candidate, default_candidate]
    else:
        candidates = [current_candidate, default_candidate, forecast_output_path]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    for candidate in candidates[1:]:
        if candidate:
            return candidate
    return forecast_output_path


def run_vehicle_ownership_stage(
    *,
    scenario: ScenarioWithCoupler,
    state: WorkflowState,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
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
    coupler : CouplerProtocol
        Coupler used to read/write artifacts across steps.
    year : int
        Forecast year being simulated.
    build_atlas_static_inputs_fallback : Callable[[Workspace], Mapping[str, Union[str, os.PathLike]]]
        Fallback builder for static ATLAS inputs when not already present in
        the workspace input registry.
    """
    logger.info("[Main] Running ATLAS vehicle ownership model.")

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

    usim_datastore_h5_default_path = os.path.join(
        urbansim_datastore_dir, usim_datastore_fname
    )
    fallback_usim_inputs, _ = build_urbansim_inputs(settings, state, workspace, year)
    usim_datastore_h5_current_path = str(
        fallback_usim_inputs.get(
            USIM_DATASTORE_CURRENT_H5, usim_datastore_h5_default_path
        )
    )
    usim_datastore_h5_subyear_path = select_atlas_usim_input_path(
        settings=settings,
        state=state,
        workspace=workspace,
        fallback_current_path=usim_datastore_h5_current_path,
        fallback_default_path=usim_datastore_h5_default_path,
        prefer_forecast_output=True,
    )
    usim_datastore_h5_start_year_path = select_atlas_usim_input_path(
        settings=settings,
        state=state,
        workspace=workspace,
        fallback_current_path=usim_datastore_h5_current_path,
        fallback_default_path=usim_datastore_h5_default_path,
        prefer_forecast_output=False,
    )
    logger.info(
        "[ATLAS] Selected UrbanSim datastores for preprocessing: start_year=%s, subyears=%s",
        usim_datastore_h5_start_year_path,
        usim_datastore_h5_subyear_path,
    )

    yrs = _atlas_sub_years(state)

    for atlas_year in yrs:
        atlas_state = AtlasSubState(state, atlas_year)
        atlas_usim_datastore_h5_path = (
            usim_datastore_h5_start_year_path
            if atlas_state.is_start_year()
            else usim_datastore_h5_subyear_path
        )
        logger.debug(
            "[ATLAS] Year %s using UrbanSim datastore: %s",
            atlas_year,
            atlas_usim_datastore_h5_path,
        )
        outputs_holder_atlas = StepOutputsHolder()

        step_inputs, step_input_descriptions = build_atlas_inputs(
            settings,
            atlas_state,
            workspace,
            atlas_year,
            coupler,
            atlas_usim_datastore_h5_path,
        )
        log_inputs(step_inputs, step_input_descriptions)
        step_inputs = merge_model_expected_inputs(
            "atlas", step_inputs, settings, atlas_state, workspace
        )
        step_inputs[USIM_DATASTORE_CURRENT_H5] = atlas_usim_datastore_h5_path
        step_inputs[USIM_DATASTORE_BASE_H5] = atlas_usim_datastore_h5_path
        # Keep ATLAS preprocessor H5 selection artifact-driven.
        atlas_state.atlas_usim_datastore_h5 = _resolve_input_path(
            step_inputs.get(USIM_DATASTORE_CURRENT_H5),
            workspace,
        )
        atlas_state.atlas_usim_datastore_base_h5 = _resolve_input_path(
            step_inputs.get(USIM_DATASTORE_BASE_H5),
            workspace,
        )
        atlas_preprocess_inputs = dict(step_inputs)
        atlas_preprocess_resolution = resolve_step_inputs(
            keys=atlas_preprocess_inputs.keys(),
            explicit_inputs=atlas_preprocess_inputs,
        )
        atlas_run_inputs: Dict[str, Any] = {}
        atlas_static_inputs = workspace.input_data.get("atlas")
        if atlas_static_inputs is not None:
            for key, value in atlas_static_inputs.to_mapping().items():
                atlas_run_inputs.setdefault(key, value)
        else:
            atlas_run_inputs.update(build_atlas_static_inputs_fallback(workspace))

        atlas_static_keys = atlas_static_input_keys(settings)
        atlas_run_fallbacks = dict(atlas_run_inputs)
        atlas_run_fallbacks.setdefault(
            USIM_DATASTORE_CURRENT_H5,
            step_inputs.get(USIM_DATASTORE_CURRENT_H5),
        )
        atlas_run_fallbacks.setdefault(
            USIM_DATASTORE_BASE_H5,
            step_inputs.get(USIM_DATASTORE_BASE_H5),
        )
        atlas_run_resolution = resolve_step_inputs(
            keys=[USIM_DATASTORE_CURRENT_H5, USIM_DATASTORE_BASE_H5, *atlas_static_keys],
            coupler=coupler,
            explicit_inputs={
                USIM_DATASTORE_CURRENT_H5: step_inputs.get(USIM_DATASTORE_CURRENT_H5),
                USIM_DATASTORE_BASE_H5: step_inputs.get(USIM_DATASTORE_BASE_H5),
            },
            fallback_inputs=atlas_run_fallbacks,
            required_keys=[USIM_DATASTORE_CURRENT_H5],
        )
        if atlas_run_resolution.missing_required:
            raise RuntimeError(
                "ATLAS run requires usim_datastore_h5 but it could not be resolved "
                "from explicit inputs, coupler, or fallback static inputs."
            )

        preprocess_steps = [
            StepRef(
                name="atlas_preprocess",
                step_func=make_atlas_preprocess_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder_atlas,
                ),
                inputs=atlas_preprocess_resolution.stepref_inputs(),
                input_keys=atlas_preprocess_resolution.stepref_input_keys(),
            ),
            StepRef(
                name="atlas_run",
                step_func=make_atlas_run_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder_atlas,
                ),
                input_keys=atlas_run_resolution.stepref_input_keys(),
                inputs=atlas_run_resolution.stepref_inputs(),
            ),
        ]

        try:
            run_workflow(
                stage_name="atlas",
                steps=preprocess_steps,
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
                StepRef(
                    name="atlas_postprocess",
                    step_func=make_atlas_postprocess_step(
                        coupler=coupler,
                        outputs_holder=outputs_holder_atlas,
                    ),
                    input_keys=postprocess_input_keys,
                )
            ]
            run_workflow(
                stage_name="atlas",
                steps=postprocess_steps,
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
