from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict

from pilates.activitysim.runner import (
    asim_sharrow_cache_dir,
    persist_sharrow_cache_enabled,
)
from pilates.config.models import PilatesConfig
from pilates.workflows.artifact_keys import ASIM_SHARROW_CACHE_DIR
from pilates.workspace import Workspace

# Model-specific step factories for ActivitySim.
# Shared helpers/infrastructure are imported from shared.py.
from .shared import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_H5_UPDATED,
    ZARR_SKIMS,
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
    CouplerProtocol,
    FileRecord,
    ModelFactory,
    RecordStore,
    StepOutputsHolder,
    WorkflowState,
    _activitysim_output_facet_meta,
    _decorate_step_with_consist,
    _execute_postprocess,
    _execute_preprocess,
    _execute_run,
    _log_step_records,
    _make_generic_step_function,
    artifact_to_path,
    cr,
    log_and_set_input,
    log_and_set_output,
    log_output_only,
    require_common_runtime,
    resolve_artifact_from_value,
)
from pilates.workflows.input_resolution import (
    first_resolved_key,
    resolve_preferred_step_input,
    resolved_value_for_key,
)

logger = logging.getLogger(__name__)


def _compile_step_schema_outputs(ctx: Any) -> list[str]:
    settings = getattr(ctx, "runtime_settings", None)
    outputs = [ZARR_SKIMS]
    if settings is not None and persist_sharrow_cache_enabled(settings):
        outputs.append(ASIM_SHARROW_CACHE_DIR)
    return outputs


def _is_non_empty_directory(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for _root, _dirs, files in os.walk(path):
        if files:
            return True
    return False


def make_activitysim_compile_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ActivitySim compile step function.

    This step compiles ActivitySim inputs (prepared by preprocess) into
    optimized skim artifacts (Zarr) used by ActivitySim runs and BEAM.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.

    Returns
    -------
    callable
        Step function for ActivitySim compile.
    """

    @require_common_runtime("expected_outputs")
    def _run_activitysim_compile_step(
        *,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        expected_outputs: Dict[str, Any],
    ) -> None:
        factory = ModelFactory()
        from workflow_state import WorkflowState as _WorkflowState

        compile_runner = factory.get_runner(
            "activitysim_compile",
            state,
            major_stage=_WorkflowState.Stage.activity_demand,
        )

        upstream = outputs_holder.activitysim_preprocess
        if upstream is None:
            raise RuntimeError(
                "ActivitySim compile must run after activitysim_preprocess"
            )
        input_store = upstream.to_record_store()
        omx_record = None
        if input_store:
            for record in input_store.all_records():
                if getattr(record, "short_name", None) == ASIM_OMX_SKIMS:
                    omx_record = record
                    break
        if omx_record is not None:
            input_store = RecordStore(recordList=[omx_record])
        else:
            input_store = RecordStore()
        if omx_record is not None:
            omx_path = omx_record.get_absolute_path(base_path=workspace.full_path)
            if omx_path and os.path.exists(omx_path):
                cr.log_input(
                    omx_path,
                    key=ASIM_OMX_SKIMS,
                    description="ActivitySim compile input skims (OMX)",
                )
        compile_outputs = compile_runner.run(input_store, workspace)

        zarr_record = None
        if compile_outputs:
            for record in compile_outputs.all_records():
                if record.short_name == ZARR_SKIMS:
                    zarr_record = record
                    break
        zarr_output_path = expected_outputs.get(ZARR_SKIMS)
        if not zarr_output_path and zarr_record is not None:
            zarr_output_path = zarr_record.file_path
        if zarr_output_path and os.path.exists(zarr_output_path):
            log_and_set_output(
                key=ZARR_SKIMS,
                path=zarr_output_path,
                description="ActivitySim compiled zarr skims",
                coupler=coupler,
                **_activitysim_output_facet_meta(
                    ZARR_SKIMS,
                    year=state.forecast_year,
                    iteration=state.iteration,
                ),
            )

        if persist_sharrow_cache_enabled(settings):
            cache_path = asim_sharrow_cache_dir(workspace)

            if cache_path and _is_non_empty_directory(cache_path):
                log_and_set_output(
                    key=ASIM_SHARROW_CACHE_DIR,
                    path=cache_path,
                    description=(
                        "ActivitySim persisted compile cache directory "
                        "(numba/sharrow)"
                    ),
                    coupler=coupler,
                    **_activitysim_output_facet_meta(
                        ASIM_SHARROW_CACHE_DIR,
                        year=state.forecast_year,
                        iteration=state.iteration,
                    ),
                )
            elif cache_path and os.path.exists(cache_path) and not os.path.isdir(cache_path):
                logger.warning(
                    "ActivitySim compile cache output path is not a directory: %s",
                    cache_path,
                )
            elif cache_path and os.path.isdir(cache_path):
                logger.info(
                    "ActivitySim compile cache output is enabled but directory is empty: %s",
                    cache_path,
                )
            elif cache_path:
                logger.warning(
                    "ActivitySim compile cache output is enabled but directory was not found: %s",
                    cache_path,
                )

    return _decorate_step_with_consist(
        step_func=_run_activitysim_compile_step,
        step_model="activitysim_compile",
        description="activitysim compile workflow step",
        outputs=[ZARR_SKIMS],
        schema_outputs=_compile_step_schema_outputs,
        tags=["activitysim", "compile"],
    )


def make_activitysim_preprocess_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ActivitySim preprocess step function.

    This step prepares ActivitySim inputs from UrbanSim outputs and ensures
    the mutable input directory contains the required tables and skims.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing preprocess outputs.

    Returns
    -------
    callable
        Step function for ActivitySim preprocess.

    Notes
    -----
    This step focuses on preparing ActivitySim inputs. Config canonicalization
    is also performed here so config ingestion is tied to the preprocess phase.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        """
        Log ActivitySim preprocess inputs.

        This helper logs the UrbanSim datastore input if it exists in the coupler.

        Parameters
        ----------
        settings : PilatesConfig
            Simulation settings for config root resolution.
        state : WorkflowState
            Current workflow state (used for log metadata only).
        workspace : Workspace
            Workspace used to resolve mutable config paths.
        holder : StepOutputsHolder
            Outputs holder (unused for this helper).

        Returns
        -------
        dict
            Extra runtime kwargs for the step executor (empty for this helper).
        """
        resolution = resolve_preferred_step_input(
            preferred_keys=[
                USIM_H5_UPDATED,
                USIM_DATASTORE_CURRENT_H5,
                USIM_DATASTORE_BASE_H5,
            ],
            coupler=coupler,
        )
        selected_key = first_resolved_key(
            resolution,
            [USIM_H5_UPDATED, USIM_DATASTORE_CURRENT_H5, USIM_DATASTORE_BASE_H5],
        )
        selected_value = (
            resolved_value_for_key(
                resolved=resolution,
                key=selected_key,
                coupler=coupler,
            )
            if selected_key is not None
            else None
        )
        usim_input = resolve_artifact_from_value(
            selected_value,
            key=selected_key or USIM_DATASTORE_CURRENT_H5,
            workspace=workspace,
        )
        usim_path = artifact_to_path(usim_input, workspace)
        if usim_path and os.path.exists(usim_path):
            input_key = selected_key or USIM_DATASTORE_BASE_H5
            input_desc = (
                f"UrbanSim datastore updated by ATLAS for ActivitySim year {state.year}"
                if input_key == USIM_H5_UPDATED
                else (
                    f"UrbanSim current datastore for ActivitySim year {state.year}"
                    if input_key == USIM_DATASTORE_CURRENT_H5
                    else f"UrbanSim base datastore for ActivitySim year {state.year}"
                )
            )
            h5_tables_used = [
                "households",
                "persons",
                "jobs",
                "blocks",
            ]
            start_year = state.start_year
            if start_year is not None:
                h5_tables_used.extend(
                    [
                        f"/{start_year}/households",
                        f"/{start_year}/persons",
                        f"/{start_year}/jobs",
                        f"/{start_year}/blocks",
                    ]
                )
            log_and_set_input(
                key=input_key,
                path=usim_path,
                description=input_desc,
                coupler=coupler,
                profile_file_schema=True,
                h5_container=True,
                h5_tables_used=h5_tables_used,
            )
        return {}

    def _log_outputs(
        outputs: ActivitySimPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        """
        Log ActivitySim preprocess outputs and update the coupler.

        Parameters
        ----------
        outputs : ActivitySimPreprocessOutputs
            Typed outputs containing ActivitySim input tables and skims.
        settings : PilatesConfig
            Simulation settings (unused for output logging).
        state : WorkflowState
            Current workflow state (used for log metadata only).
        workspace : Workspace
            Workspace for path resolution.
        holder : StepOutputsHolder
            Outputs holder (unused for this helper).
        """
        _log_step_records(
            record_items=outputs._iter_record_items(),
            log_fn=lambda key, path, description, **meta: log_and_set_output(
                key=key,
                path=path,
                description=description,
                coupler=coupler,
                **meta,
            ),
            profile_schema_keys={ASIM_HOUSEHOLDS_IN, ASIM_PERSONS_IN, ASIM_LAND_USE_IN},
        )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="activitysim",
        phase="preprocess",
        outputs_class=ActivitySimPreprocessOutputs,
        component_getter=lambda factory, state: factory.get_preprocessor(
            "activitysim", state, WorkflowState.Stage.activity_demand
        ),
        component_executor=_execute_preprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "activitysim_preprocess", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )


def make_activitysim_run_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ActivitySim run step function.

    This step executes the activity demand model for the year/iteration and
    produces household/person/tour outputs consumed by BEAM and postprocessing.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing run outputs.

    Returns
    -------
    callable
        Step function for ActivitySim run.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        upstream = holder.activitysim_preprocess
        if upstream is None:
            raise RuntimeError("ActivitySim preprocess must complete first")

        profile_keys = {ASIM_HOUSEHOLDS_IN, ASIM_PERSONS_IN, ASIM_LAND_USE_IN}
        for short_name, path, description in upstream._iter_record_items():
            if short_name == ASIM_OMX_SKIMS:
                continue
            meta: Dict[str, Any] = {}
            if short_name in profile_keys:
                meta["profile_file_schema"] = True
            log_and_set_input(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
                **meta,
            )

        extra_inputs = RecordStore()
        zarr_value = None
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            zarr_value = resolve_artifact_from_value(
                get_value(ZARR_SKIMS),
                key=ZARR_SKIMS,
                workspace=workspace,
            )
        zarr_path = artifact_to_path(zarr_value, workspace)
        if not zarr_path:
            candidate = os.path.join(
                workspace.get_asim_output_dir(), "cache", "skims.zarr"
            )
            if os.path.exists(candidate):
                zarr_path = candidate
        if zarr_path and os.path.exists(zarr_path):
            extra_inputs.add_record(
                FileRecord(
                    file_path=zarr_path,
                    short_name=ZARR_SKIMS,
                    description="Compiled ActivitySim skims (Zarr)",
                )
            )
            log_and_set_input(
                key=ZARR_SKIMS,
                path=zarr_path,
                description=(
                    f"ActivitySim compiled skims for year {state.year}, iter {state.iteration}"
                ),
                coupler=coupler,
            )
        return {"extra_inputs": extra_inputs}

    def _log_outputs(
        outputs: ActivitySimRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for short_name, path, description in outputs._iter_record_items():
            artifact = cr.log_output(
                str(path),
                key=short_name,
                description=description,
                **_activitysim_output_facet_meta(
                    short_name,
                    year=state.forecast_year,
                    iteration=state.iteration,
                ),
            )
            if artifact is not None and getattr(artifact, "hash", None):
                outputs.raw_output_hashes[short_name] = artifact.hash

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="activitysim",
        phase="run",
        outputs_class=ActivitySimRunOutputs,
        component_getter=lambda factory, state: factory.get_runner(
            "activitysim", state, WorkflowState.Stage.activity_demand
        ),
        component_executor=_execute_run,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "activitysim_run", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )


def make_activitysim_postprocess_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ActivitySim postprocess step function.

    This step converts ActivitySim outputs into downstream inputs and updates
    the UrbanSim datastore for the next model stages.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing postprocess outputs.

    Returns
    -------
    callable
        Step function for ActivitySim postprocess.
    """

    def _log_outputs(
        outputs: ActivitySimPostprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        def _extra_meta(short_name: str, _path: str, _description: str) -> Dict[str, Any]:
            meta: Dict[str, Any] = _activitysim_output_facet_meta(
                short_name,
                year=state.forecast_year,
                iteration=state.iteration,
            )
            content_hash = outputs.processed_output_hashes.get(short_name)
            if content_hash:
                meta["content_hash"] = content_hash
            return meta

        _log_step_records(
            record_items=outputs._iter_record_items(),
            log_fn=log_output_only,
            profile_schema_keys={
                "persons_asim_out",
                "trips_asim_out",
                "tours_asim_out",
                "beam_plans_asim_out",
                "households_asim_out",
            },
            extra_meta_fn=_extra_meta,
        )
        if outputs.usim_datastore_h5 is not None:
            log_and_set_output(
                key=USIM_DATASTORE_H5,
                path=str(outputs.usim_datastore_h5),
                description=(
                    f"UrbanSim datastore updated by ActivitySim for year {state.forecast_year}"
                ),
                coupler=coupler,
                profile_file_schema=True,
                h5_container=True,
                hash_tables="if_unchanged",
            )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="activitysim",
        phase="postprocess",
        outputs_class=ActivitySimPostprocessOutputs,
        component_getter=lambda factory, state: factory.get_postprocessor(
            "activitysim", state, WorkflowState.Stage.activity_demand
        ),
        component_executor=_execute_postprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "activitysim_postprocess", outputs
        ),
        output_logger=_log_outputs,
    )
