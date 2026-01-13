from __future__ import annotations

import logging
import os
import sys
from typing import Any, Callable, Dict, TYPE_CHECKING

from pilates.generic.model_factory import ModelFactory
from pilates.utils import consist_runtime as cr
from pilates.utils.coupler_helpers import (
    log_and_set_output,
    update_coupler_from_beam_outputs,
)
from pilates.workflows.step_exec import (
    forecast_land_use,
    run_activity_demand,
    run_traffic_assignment,
    warm_start_activities,
)

if TYPE_CHECKING:
    from pilates.config.models import PilatesConfig
    from pilates.generic.records import RecordStore
    from pilates.workspace import Workspace
    from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


def require_common_runtime(
    *names: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return cr.require_runtime_kwargs("settings", "state", "workspace", *names)


def make_urbansim_step(
    *,
    coupler: Any,
    year: int,
) -> Callable[..., None]:
    @require_common_runtime("usim_data_dir", "expected_outputs")
    def _run_urbansim_step(
        *,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        usim_data_dir: str,
        expected_outputs: Dict[str, Any],
    ) -> None:
        if state.is_start_year() and settings.activitysim.warm_start_activities:
            logger.info("[Main] Running warm start activities for ActivitySim.")
            warm_start_activities(settings, state, workspace)

        forecast_land_use(settings, year, state, workspace)
        usim_output_path = expected_outputs.get("usim_datastore_h5")
        if not usim_output_path:
            usim_output_fname = settings.urbansim.output_file_template.format(
                year=state.forecast_year
            )
            usim_output_path = os.path.join(usim_data_dir, usim_output_fname)
        if os.path.exists(usim_output_path):
            log_and_set_output(
                key="usim_datastore_h5",
                path=usim_output_path,
                description=(
                    f"UrbanSim datastore output for year {state.forecast_year}"
                ),
                coupler=coupler,
            )

    return _run_urbansim_step


def make_atlas_step(
    *,
    coupler: Any,
) -> Callable[..., None]:
    @cr.require_runtime_kwargs(
        "atlas_state",
        "base_state",
        "preprocessor",
        "runner",
        "postprocessor",
        "workspace",
        "usim_datastore_h5_path",
        "atlas_year",
        "expected_outputs",
    )
    def _run_atlas_step(
        *,
        atlas_state: WorkflowState,
        base_state: WorkflowState,
        preprocessor: Any,
        runner: Any,
        postprocessor: Any,
        workspace: Workspace,
        usim_datastore_h5_path: str,
        atlas_year: int,
        expected_outputs: Dict[str, Any],
    ) -> None:
        preprocessor.update_state(atlas_state)
        input_data = preprocessor.preprocess(workspace)

        runner.update_state(atlas_state)
        try:
            raw_outputs = runner.run(input_data, workspace)
            postprocessor.update_state(atlas_state)
            postprocessor.postprocess(raw_outputs, workspace)

            atlas_output_dir = expected_outputs.get("atlas_output_dir")
            if not atlas_output_dir:
                atlas_output_dir = workspace.get_atlas_output_dir()
            if os.path.exists(atlas_output_dir):
                log_and_set_output(
                    key="atlas_output_dir",
                    path=atlas_output_dir,
                    description=f"ATLAS output directory for year {atlas_year}",
                    coupler=coupler,
                )

            atlas_usim_output = expected_outputs.get("usim_datastore_h5")
            if not atlas_usim_output:
                atlas_usim_output = usim_datastore_h5_path
            if os.path.exists(atlas_usim_output):
                log_and_set_output(
                    key="usim_datastore_h5",
                    path=atlas_usim_output,
                    description=(
                        "UrbanSim datastore after ATLAS update for year "
                        f"{atlas_year}"
                    ),
                    coupler=coupler,
                )
            else:
                logger.warning(
                    "[Main] UrbanSim datastore not found after ATLAS postprocess: %s",
                    atlas_usim_output,
                )
        except Exception:
            from pilates.utils.failure_handling import persist_state_on_error

            persist_state_on_error(base_state, f"ATLAS year {atlas_year}")
            sys.exit(1)

    return _run_atlas_step


def make_activitysim_compile_step(
    *,
    coupler: Any,
) -> Callable[..., None]:
    @require_common_runtime("compile_outputs_holder", "expected_outputs")
    def _run_activitysim_compile_step(
        *,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        compile_outputs_holder: Dict[str, Any],
        expected_outputs: Dict[str, Any],
    ) -> None:
        factory = ModelFactory()
        from workflow_state import WorkflowState as _WorkflowState

        preprocessor = factory.get_preprocessor(
            "activitysim",
            state,
            major_stage=_WorkflowState.Stage.activity_demand,
        )
        compile_runner = factory.get_runner(
            "activitysim_compile",
            state,
            major_stage=_WorkflowState.Stage.activity_demand,
        )

        input_store = preprocessor.preprocess(workspace)
        omx_record = None
        if input_store:
            for record in input_store.all_records():
                if getattr(record, "short_name", None) == "omx_skims":
                    omx_record = record
                    break
        if omx_record is not None:
            omx_path = omx_record.get_absolute_path(base_path=workspace.full_path)
            if omx_path and os.path.exists(omx_path):
                cr.log_input(
                    omx_path,
                    key="omx_skims",
                    description="ActivitySim compile input skims (OMX)",
                )
        compile_outputs = compile_runner.run(input_store, workspace)

        compile_outputs_holder["input_store"] = input_store
        compile_outputs_holder["compile_outputs"] = compile_outputs

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
            log_and_set_output(
                key="zarr_skims",
                path=zarr_output_path,
                description="ActivitySim compiled zarr skims",
                coupler=coupler,
            )

    return _run_activitysim_compile_step


def make_activitysim_step(
    *,
    coupler: Any,
) -> Callable[..., None]:
    @require_common_runtime(
        "asim_outputs_holder",
        "input_store",
        "compile_outputs",
        "expected_outputs",
    )
    def _run_activitysim_step(
        *,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        asim_outputs_holder: Dict[str, Any],
        input_store: RecordStore,
        compile_outputs: RecordStore,
        expected_outputs: Dict[str, Any],
    ) -> None:
        combined_input_store = input_store
        if combined_input_store is not None and compile_outputs:
            combined_input_store = combined_input_store + compile_outputs
        asim_outputs_holder["activity_demand_outputs"] = run_activity_demand(
            settings,
            state,
            workspace,
            input_store=combined_input_store,
        )
        asim_output_dir = expected_outputs.get("asim_output_dir")
        if not asim_output_dir:
            asim_output_dir = workspace.get_asim_output_dir()
        if os.path.exists(asim_output_dir):
            log_and_set_output(
                key="asim_output_dir",
                path=asim_output_dir,
                description=(
                    "ActivitySim output directory for year "
                    f"{state.year}, iter {state.iteration}"
                ),
                coupler=coupler,
            )

    return _run_activitysim_step


def make_beam_step(
    *,
    coupler: Any,
) -> Callable[..., None]:
    @require_common_runtime(
        "activity_demand_outputs",
        "previous_beam_outputs",
        "beam_inputs",
        "beam_mutable_dir",
        "beam_mutable_description",
        "beam_outputs_holder",
        "expected_outputs",
    )
    def _run_beam_step(
        *,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        activity_demand_outputs: RecordStore,
        previous_beam_outputs: RecordStore,
        beam_inputs: Dict[str, Any],
        beam_mutable_dir: str,
        beam_mutable_description: str,
        beam_outputs_holder: Dict[str, Any],
        expected_outputs: Dict[str, Any],
    ) -> None:
        if beam_inputs:
            cr.log_artifacts(beam_inputs, direction="input")
        if beam_mutable_dir:
            cr.log_input(
                beam_mutable_dir,
                key="beam_mutable_data_dir",
                description=beam_mutable_description or "",
            )
        beam_outputs_holder["beam_outputs"] = run_traffic_assignment(
            settings,
            state,
            workspace,
            activity_demand_outputs,
            previous_beam_outputs,
        )
        beam_output_dir = expected_outputs.get("beam_output_dir")
        if not beam_output_dir:
            beam_output_dir = workspace.get_beam_output_dir()
        if os.path.exists(beam_output_dir):
            log_and_set_output(
                key="beam_output_dir",
                path=beam_output_dir,
                description=(
                    f"BEAM output directory for year {state.year}, iter {state.iteration}"
                ),
                coupler=coupler,
            )

        output_store = beam_outputs_holder.get("beam_outputs")
        update_coupler_from_beam_outputs(output_store, coupler, workspace)

    return _run_beam_step


def make_postprocessing_step() -> Callable[..., None]:
    @require_common_runtime()
    def _run_postprocessing_step(
        *,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
    ) -> None:
        if "postprocessing" in settings:
            from pilates.postprocessing.postprocessor import (
                copy_outputs_to_mep,
                process_event_file,
            )

            process_event_file(settings, state.forecast_year, state.current_inner_iter)
            copy_outputs_to_mep(
                settings,
                state.forecast_year,
                state.current_inner_iter,
                workspace,
            )

    return _run_postprocessing_step
