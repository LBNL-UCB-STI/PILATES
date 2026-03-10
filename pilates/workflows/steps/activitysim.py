from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from pilates.activitysim.runner import (
    asim_sharrow_cache_dir,
    persist_sharrow_cache_enabled,
)
from pilates.activitysim.postprocessor import get_usim_datastore_fname
from pilates.config.models import PilatesConfig
from pilates.generic.model_factory import ModelFactory
from pilates.utils.coupler_helpers import record_store_to_outputs
from pilates.workflows.artifact_keys import ASIM_SHARROW_CACHE_DIR
from pilates.workflows.outputs_base import (
    StepOutputsBase,
    ValidationContext,
    iter_step_output_items,
)
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
    USIM_FORECAST_OUTPUT,
    USIM_H5_UPDATED,
    ZARR_SKIMS,
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
    CouplerProtocol,
    FileRecord,
    RecordStore,
    StepOutputsHolder,
    WorkflowState,
    _activitysim_output_facet_meta,
    _decorate_step_with_consist,
    _declared_outputs_from_class,
    _execute_preprocess,
    _execute_postprocess,
    _execute_run,
    _log_named_h5_tables,
    _log_step_records,
    _schema_outputs_from_class,
    _upstream_outputs_view,
    artifact_to_path,
    cr,
    log_and_set_input,
    log_and_set_output,
    log_input_only,
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

StepOutputsT = TypeVar("StepOutputsT", bound=StepOutputsBase)


def _artifact_content_hash(value: Any) -> Optional[str]:
    """
    Extract a content hash from an artifact-like value when available.
    """
    if value is None:
        return None
    for attr_name in ("content_hash", "hash"):
        content_hash = getattr(value, attr_name, None)
        if content_hash:
            return str(content_hash)
    return None


def _make_activitysim_typed_step_function(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    model_name: str,
    phase: str,
    outputs_class: Type[StepOutputsT],
    component_getter: Callable[[ModelFactory, WorkflowState], Any],
    component_executor: Callable[..., RecordStore],
    outputs_holder_setter: Callable[[StepOutputsHolder, StepOutputsT], None],
    input_logger: Optional[Callable[..., Dict[str, Any]]] = None,
    output_logger: Optional[Callable[..., None]] = None,
) -> Callable[..., None]:
    @cr.require_runtime_kwargs("settings", "state", "workspace")
    def _step_func(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        **kwargs: Any,
    ) -> None:
        logger.debug("Starting %s %s step", model_name, phase)
        factory = ModelFactory()
        component = component_getter(factory, state)

        extra_kwargs: Dict[str, Any] = {}
        if input_logger is not None:
            extra_kwargs = (
                input_logger(settings, state, workspace, outputs_holder) or {}
            )

        record_store = component_executor(
            component,
            workspace,
            outputs_holder,
            coupler=coupler,
            context=f"{model_name}_{phase}",
            **extra_kwargs,
            **kwargs,
        )
        step_outputs = record_store_to_outputs(
            record_store=record_store,
            output_class=outputs_class,
            workspace=workspace,
        )
        validation_context = ValidationContext(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name=f"{model_name}_{phase}",
            upstream_outputs=_upstream_outputs_view(
                outputs_holder,
                current_step_name=f"{model_name}_{phase}",
            ),
        )
        step_outputs.validate(context=validation_context)
        outputs_holder_setter(outputs_holder, step_outputs)

        if output_logger is not None:
            output_logger(step_outputs, settings, state, workspace, outputs_holder)

        logger.info("%s %s completed successfully", model_name, phase)

    if output_logger is not None:
        setattr(
            _step_func,
            "__pilates_output_replayer__",
            lambda outputs, settings, state, workspace, holder: output_logger(
                outputs, settings, state, workspace, holder
            ),
        )

    return _decorate_step_with_consist(
        step_func=_step_func,
        step_model=f"{model_name}_{phase}",
        description=f"{model_name} {phase} workflow step",
        schema_outputs=_schema_outputs_from_class(outputs_class),
        outputs=_declared_outputs_from_class(outputs_class),
        tags=[model_name, phase],
    )


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

        compile_runner = factory.get_runner(
            "activitysim_compile",
            state,
        )

        upstream = outputs_holder.activitysim_preprocess
        if upstream is None:
            raise RuntimeError(
                "ActivitySim compile must run after activitysim_preprocess"
            )
        omx_record = None
        omx_path = None
        for short_name, path, description in iter_step_output_items(upstream):
            if short_name != ASIM_OMX_SKIMS:
                continue
            omx_record = FileRecord(
                file_path=str(path),
                short_name=short_name,
                description=description,
            )
            omx_path = artifact_to_path(path, workspace) or str(path)
            break
        input_store = (
            RecordStore(recordList=[omx_record])
            if omx_record is not None
            else RecordStore()
        )
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
            table_keys = {
                "/households": "activitysim_preprocess_usim_households_table_input",
                "/persons": "activitysim_preprocess_usim_persons_table_input",
                "/jobs": "activitysim_preprocess_usim_jobs_table_input",
                "/blocks": "activitysim_preprocess_usim_blocks_table_input",
            }
            table_descriptions = {
                "/households": "UrbanSim households table used by ActivitySim preprocess",
                "/persons": "UrbanSim persons table used by ActivitySim preprocess",
                "/jobs": "UrbanSim jobs table used by ActivitySim preprocess",
                "/blocks": "UrbanSim blocks table used by ActivitySim preprocess",
            }
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
                table_keys.update(
                    {
                        f"/{start_year}/households": (
                            "activitysim_preprocess_usim_households_table_start_year_input"
                        ),
                        f"/{start_year}/persons": (
                            "activitysim_preprocess_usim_persons_table_start_year_input"
                        ),
                        f"/{start_year}/jobs": (
                            "activitysim_preprocess_usim_jobs_table_start_year_input"
                        ),
                        f"/{start_year}/blocks": (
                            "activitysim_preprocess_usim_blocks_table_start_year_input"
                        ),
                    }
                )
                table_descriptions.update(
                    {
                        f"/{start_year}/households": (
                            "UrbanSim start-year households table used by ActivitySim preprocess"
                        ),
                        f"/{start_year}/persons": (
                            "UrbanSim start-year persons table used by ActivitySim preprocess"
                        ),
                        f"/{start_year}/jobs": (
                            "UrbanSim start-year jobs table used by ActivitySim preprocess"
                        ),
                        f"/{start_year}/blocks": (
                            "UrbanSim start-year blocks table used by ActivitySim preprocess"
                        ),
                    }
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
            _log_named_h5_tables(
                path=usim_path,
                direction="input",
                table_keys=table_keys,
                description_by_table=table_descriptions,
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

    return _make_activitysim_typed_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="activitysim",
        phase="preprocess",
        outputs_class=ActivitySimPreprocessOutputs,
        component_getter=lambda factory, state: factory.get_preprocessor(
            "activitysim", state
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

    compile_input_hashes: Dict[str, str] = {}

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

        extra_inputs: Dict[str, str] = {}
        compile_input_hashes.clear()
        zarr_value = None
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            zarr_value = resolve_artifact_from_value(
                get_value(ZARR_SKIMS),
                key=ZARR_SKIMS,
                workspace=workspace,
            )
        zarr_content_hash = _artifact_content_hash(zarr_value)
        if zarr_content_hash:
            compile_input_hashes[ZARR_SKIMS] = zarr_content_hash
        zarr_path = artifact_to_path(zarr_value, workspace)
        if not zarr_path:
            candidate = os.path.join(
                workspace.get_asim_output_dir(), "cache", "skims.zarr"
            )
            if os.path.exists(candidate):
                zarr_path = candidate
        if zarr_path and os.path.exists(zarr_path):
            extra_inputs[ZARR_SKIMS] = zarr_path
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
        upstream = holder.activitysim_preprocess
        if upstream is not None:
            carried_hashes = getattr(upstream, "input_hashes", {}) or {}
            for short_name, path, _description in upstream._iter_record_items():
                outputs.source_input_paths[short_name] = Path(path)
                content_hash = carried_hashes.get(short_name)
                if content_hash:
                    outputs.source_input_hashes[short_name] = content_hash

        get_value = getattr(coupler, "get", None)
        zarr_value = get_value(ZARR_SKIMS) if callable(get_value) else None
        zarr_path = artifact_to_path(zarr_value, workspace)
        if not zarr_path:
            candidate = os.path.join(
                workspace.get_asim_output_dir(), "cache", "skims.zarr"
            )
            if os.path.exists(candidate):
                zarr_path = candidate
        if zarr_path and os.path.exists(zarr_path):
            outputs.source_input_paths[ZARR_SKIMS] = Path(zarr_path)
            content_hash = _artifact_content_hash(zarr_value) or compile_input_hashes.get(
                ZARR_SKIMS
            )
            if content_hash:
                outputs.source_input_hashes[ZARR_SKIMS] = content_hash

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

    return _make_activitysim_typed_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="activitysim",
        phase="run",
        outputs_class=ActivitySimRunOutputs,
        component_getter=lambda factory, state: factory.get_runner(
            "activitysim", state
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

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        asim_input_dir = workspace.get_asim_mutable_data_dir()
        asim_output_dir = workspace.get_asim_output_dir()
        asim_input_sources = [
            (
                ASIM_HOUSEHOLDS_IN,
                os.path.join(asim_input_dir, "households.csv"),
                "ActivitySim postprocess source input households.csv",
            ),
            (
                ASIM_PERSONS_IN,
                os.path.join(asim_input_dir, "persons.csv"),
                "ActivitySim postprocess source input persons.csv",
            ),
            (
                ASIM_LAND_USE_IN,
                os.path.join(asim_input_dir, "land_use.csv"),
                "ActivitySim postprocess source input land_use.csv",
            ),
            (
                ASIM_OMX_SKIMS,
                os.path.join(asim_input_dir, "skims.omx"),
                "ActivitySim postprocess source input skims.omx",
            ),
            (
                ZARR_SKIMS,
                os.path.join(asim_output_dir, "cache", "skims.zarr"),
                "ActivitySim postprocess source input skims.zarr",
            ),
        ]
        for key, path, description in asim_input_sources:
            if os.path.exists(path):
                log_input_only(
                    key=key,
                    path=path,
                    description=description,
                    profile_file_schema="if_changed",
                )

        if state.is_enabled(WorkflowState.Stage.land_use):
            usim_data_dir = workspace.get_usim_mutable_data_dir()
            current_store_path = os.path.join(
                usim_data_dir,
                get_usim_datastore_fname(settings, io="input"),
            )
            if os.path.exists(current_store_path):
                log_input_only(
                    key=USIM_DATASTORE_CURRENT_H5,
                    path=current_store_path,
                    description=(
                        "ActivitySim postprocess source UrbanSim current datastore"
                    ),
                    profile_file_schema="if_changed",
                )

            forecast_store_path = os.path.join(
                usim_data_dir,
                get_usim_datastore_fname(
                    settings, io="output", year=state.forecast_year
                ),
            )
            if os.path.exists(forecast_store_path):
                log_input_only(
                    key=USIM_FORECAST_OUTPUT,
                    path=forecast_store_path,
                    description=(
                        "ActivitySim postprocess source UrbanSim forecast datastore"
                    ),
                    profile_file_schema="if_changed",
                )
        return {}

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
            _log_named_h5_tables(
                path=str(outputs.usim_datastore_h5),
                direction="output",
                table_keys={
                    "/households": "activitysim_postprocess_usim_households_table_updated",
                    "/persons": "activitysim_postprocess_usim_persons_table_updated",
                },
                description_by_table={
                    "/households": "UrbanSim households table updated by ActivitySim postprocess",
                    "/persons": "UrbanSim persons table updated by ActivitySim postprocess",
                },
            )

    return _make_activitysim_typed_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="activitysim",
        phase="postprocess",
        outputs_class=ActivitySimPostprocessOutputs,
        component_getter=lambda factory, state: factory.get_postprocessor(
            "activitysim", state
        ),
        component_executor=_execute_postprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "activitysim_postprocess", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )
