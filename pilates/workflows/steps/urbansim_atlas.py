from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Type, TypeVar

import pandas as pd

from pilates.atlas.postprocessor import resolve_atlas_usim_datastore_path
from pilates.config.models import PilatesConfig
from pilates.generic.model_factory import ModelFactory
from pilates.utils import consist_runtime as cr
from pilates.utils.coupler_helpers import artifact_to_existing_path
from pilates.workflows.artifact_key_migrations import resolve_artifact_key
from pilates.workflows.outputs_base import StepOutputsBase, ValidationContext
from pilates.workspace import Workspace

# Model-specific step factories for UrbanSim and ATLAS.
# Shared helpers/infrastructure are imported from shared.py.
from .shared import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_H5_UPDATED,
    USIM_INPUT_ARCHIVE_PREFIX,
    USIM_INPUT_MERGED_PREFIX,
    AtlasPostprocessOutputs,
    AtlasPreprocessOutputs,
    AtlasRunOutputs,
    CouplerProtocol,
    StepOutputsHolder,
    UrbanSimPostprocessOutputs,
    UrbanSimPreprocessOutputs,
    UrbanSimRunOutputs,
    WorkflowState,
    _atlas_artifact_facet_meta,
    _decorate_step_with_consist,
    _declared_outputs_from_class,
    _log_named_h5_tables,
    _log_step_records,
    _schema_outputs_from_class,
    _upstream_outputs_view,
    _urbansim_output_facet_meta,
    log_and_set_output,
    log_input_only,
    log_output_only,
    logger,
    warm_start_activities,
)

StepOutputsT = TypeVar("StepOutputsT", bound=StepOutputsBase)


def _strip_component_runtime_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    filtered = dict(kwargs)
    filtered.pop("coupler", None)
    filtered.pop("context", None)
    return filtered


def _root_h5_table_keys(
    path: str,
    *,
    key_prefix: str,
    key_suffix: str,
) -> Dict[str, str]:
    """
    Build artifact keys for root-level HDF5 tables in ``path``.

    UrbanSim merged/archived datastores expose the next-iteration tables at the
    HDF5 root (e.g. ``/households``). We log those tables individually so
    Consist can profile the concrete schema of the tables the postprocessor
    reads and writes.
    """
    try:
        with pd.HDFStore(path, mode="r") as store:
            keys = sorted(store.keys())
    except Exception as exc:
        logger.debug(
            "Skipping HDF5 table enumeration for unreadable file %s: %s",
            path,
            exc,
        )
        return {}

    table_keys: Dict[str, str] = {}
    for table_path in keys:
        normalized_path = (
            str(table_path) if str(table_path).startswith("/") else f"/{table_path}"
        )
        if normalized_path.strip("/").count("/") != 0:
            continue
        table_name = normalized_path.split("/")[-1]
        table_keys[normalized_path] = f"{key_prefix}{table_name}{key_suffix}"
    return table_keys


def _root_h5_table_descriptions(path: str, *, action: str) -> Dict[str, str]:
    descriptions: Dict[str, str] = {}
    for table_path in _root_h5_table_keys(
        path,
        key_prefix="unused_",
        key_suffix="_unused",
    ):
        table_name = table_path.split("/")[-1]
        descriptions[table_path] = f"UrbanSim {table_name} table {action}"
    return descriptions


def _make_typed_step_function(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    model_name: str,
    phase: str,
    outputs_class: Type[StepOutputsT],
    component_getter: Callable[[ModelFactory, WorkflowState], Any],
    component_executor: Callable[..., StepOutputsT],
    outputs_holder_setter: Callable[[StepOutputsHolder, StepOutputsT], None],
    input_logger: Optional[Callable[..., Dict[str, Any]]] = None,
    output_logger: Optional[Callable[..., None]] = None,
    output_recoverer: Optional[Callable[..., Optional[StepOutputsT]]] = None,
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

        step_outputs = component_executor(
            component,
            workspace,
            outputs_holder,
            coupler=coupler,
            context=f"{model_name}_{phase}",
            **extra_kwargs,
            **kwargs,
        )
        if not isinstance(step_outputs, outputs_class):
            raise TypeError(
                f"{model_name}_{phase} must return {outputs_class.__name__}, "
                f"got {type(step_outputs).__name__}"
            )

        validation_context = ValidationContext(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name=f"{model_name}_{phase}",
            upstream_outputs=_upstream_outputs_view(
                outputs_holder, current_step_name=f"{model_name}_{phase}"
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
    if output_recoverer is not None:
        setattr(_step_func, "__pilates_output_recoverer__", output_recoverer)
    return _decorate_step_with_consist(
        step_func=_step_func,
        step_model=f"{model_name}_{phase}",
        description=f"{model_name} {phase} workflow step",
        schema_outputs=_schema_outputs_from_class(outputs_class),
        outputs=_declared_outputs_from_class(outputs_class),
        tags=[model_name, phase],
    )


def _resolve_cached_run_outputs(run_id: Optional[str]) -> Dict[str, Any]:
    if not run_id:
        return {}
    tracker = cr.current_tracker()
    if tracker is None:
        return {}
    get_run_outputs = getattr(tracker, "get_run_outputs", None)
    if not callable(get_run_outputs):
        return {}
    try:
        return get_run_outputs(run_id) or {}
    except Exception:
        logger.debug(
            "Failed loading cached run outputs for run_id=%s", run_id, exc_info=True
        )
        return {}


def _recovered_cached_paths(
    *,
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
    workspace: Workspace,
) -> Dict[str, Path]:
    merged: Dict[str, Any] = {}
    if cached_outputs:
        for raw_key, value in cached_outputs.items():
            if value is None:
                continue
            local_key = str(raw_key).split("/", 1)[-1]
            merged[resolve_artifact_key(local_key)] = value
    for raw_key, value in _resolve_cached_run_outputs(run_id).items():
        if value is None:
            continue
        local_key = str(raw_key).split("/", 1)[-1]
        merged[resolve_artifact_key(local_key)] = value
    recovered: Dict[str, Path] = {}
    for key, value in merged.items():
        path = artifact_to_existing_path(
            value,
            workspace=workspace,
            materialize_from_archive=True,
        )
        if path is not None:
            recovered[key] = Path(path)
    return recovered


def _recover_urbansim_run_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[UrbanSimRunOutputs]:
    del settings, state, coupler, outputs_holder, step_inputs
    recovered_paths = _recovered_cached_paths(
        cached_outputs=cached_outputs,
        run_id=run_id,
        workspace=workspace,
    )
    usim_datastore_h5 = recovered_paths.get(
        USIM_FORECAST_OUTPUT
    ) or recovered_paths.get(USIM_DATASTORE_H5)
    if usim_datastore_h5 is None:
        return None
    return UrbanSimRunOutputs(
        usim_datastore_h5=usim_datastore_h5,
        raw_outputs=recovered_paths,
    )


def _recover_urbansim_postprocess_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[UrbanSimPostprocessOutputs]:
    del settings, state, coupler, outputs_holder, step_inputs
    recovered_paths = _recovered_cached_paths(
        cached_outputs=cached_outputs,
        run_id=run_id,
        workspace=workspace,
    )
    usim_datastore_h5 = next(
        (
            path
            for key, path in recovered_paths.items()
            if key.startswith(USIM_INPUT_MERGED_PREFIX)
        ),
        None,
    ) or recovered_paths.get(USIM_DATASTORE_H5)
    if usim_datastore_h5 is None:
        return None
    return UrbanSimPostprocessOutputs(
        usim_datastore_h5=usim_datastore_h5,
        processed_outputs=recovered_paths,
    )


def _recover_atlas_run_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[AtlasRunOutputs]:
    del settings, coupler, outputs_holder, step_inputs
    recovered_paths = _recovered_cached_paths(
        cached_outputs=cached_outputs,
        run_id=run_id,
        workspace=workspace,
    )
    output_year = getattr(state, "forecast_year", None)
    if output_year is None:
        return None
    required_keys = (
        f"householdv_{output_year}",
        f"vehicles_{output_year}",
    )
    if any(key not in recovered_paths for key in required_keys):
        return None
    return AtlasRunOutputs(
        atlas_output_dir=Path(workspace.get_atlas_output_dir()),
        raw_outputs=recovered_paths,
    )


def _recover_atlas_postprocess_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[AtlasPostprocessOutputs]:
    del settings, state, coupler, outputs_holder, step_inputs
    recovered_paths = _recovered_cached_paths(
        cached_outputs=cached_outputs,
        run_id=run_id,
        workspace=workspace,
    )
    if "atlas_vehicles2_output" not in recovered_paths:
        return None
    usim_datastore_h5 = recovered_paths.get(USIM_H5_UPDATED) or recovered_paths.get(
        USIM_DATASTORE_H5
    )
    if usim_datastore_h5 is None:
        return None
    return AtlasPostprocessOutputs(
        atlas_output_dir=Path(workspace.get_atlas_output_dir()),
        usim_datastore_h5=usim_datastore_h5,
        processed_outputs=recovered_paths,
    )


def _execute_urbansim_preprocess_typed(
    preprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> UrbanSimPreprocessOutputs:
    filtered_kwargs = _strip_component_runtime_kwargs(kwargs)
    final_skims_omx = filtered_kwargs.get("final_skims_omx")
    if final_skims_omx is not None:
        return preprocessor.preprocess(
            workspace,
            final_skims_omx=final_skims_omx,
        )
    return preprocessor.preprocess(workspace)


def _execute_urbansim_run_typed(
    runner: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> UrbanSimRunOutputs:
    upstream = outputs_holder.urbansim_preprocess
    if upstream is None:
        raise RuntimeError("UrbanSim preprocess must complete first")
    if not isinstance(upstream, UrbanSimPreprocessOutputs):
        raise TypeError(
            "urbansim_run requires UrbanSimPreprocessOutputs from urbansim_preprocess"
        )
    return runner.run(upstream, workspace)


def _execute_urbansim_postprocess_typed(
    postprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> UrbanSimPostprocessOutputs:
    upstream = outputs_holder.urbansim_run
    if upstream is None:
        raise RuntimeError("UrbanSim run must complete first")
    if not isinstance(upstream, UrbanSimRunOutputs):
        raise TypeError(
            "urbansim_postprocess requires UrbanSimRunOutputs from urbansim_run"
        )
    return postprocessor.postprocess(upstream, workspace)


def _execute_atlas_preprocess_typed(
    preprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> AtlasPreprocessOutputs:
    return preprocessor.preprocess(
        workspace,
        **_strip_component_runtime_kwargs(kwargs),
    )


def _execute_atlas_run_typed(
    runner: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> AtlasRunOutputs:
    upstream = outputs_holder.atlas_preprocess
    if upstream is None:
        raise RuntimeError("ATLAS preprocess must complete first")
    if not isinstance(upstream, AtlasPreprocessOutputs):
        raise TypeError(
            "atlas_run requires AtlasPreprocessOutputs from atlas_preprocess"
        )
    return runner.run(upstream, workspace)


def _execute_atlas_postprocess_typed(
    postprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> AtlasPostprocessOutputs:
    upstream = outputs_holder.atlas_run
    if upstream is None:
        raise RuntimeError("ATLAS run must complete first")
    if not isinstance(upstream, AtlasRunOutputs):
        raise TypeError("atlas_postprocess requires AtlasRunOutputs from atlas_run")
    return postprocessor.postprocess(upstream, workspace)


def make_urbansim_preprocess_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the UrbanSim preprocess step function.

    This step prepares land-use inputs by materializing the UrbanSim mutable
    data directory, ensuring warm-start activities (if enabled), and making
    required input tables available for the land-use runner.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing preprocess outputs.

    Returns
    -------
    callable
        Step function for UrbanSim preprocess.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        activitysim_settings = settings.activitysim
        if (
            state.is_start_year()
            and activitysim_settings is not None
            and activitysim_settings.warm_start_activities
        ):
            logger.info("[Main] Running warm start activities for ActivitySim.")
            warm_start_activities(settings, state, workspace)
        return {}

    def _log_outputs(
        outputs: UrbanSimPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        forecast_year = state.forecast_year
        urbansim_settings = settings.urbansim
        if forecast_year is None:
            raise RuntimeError(
                "WorkflowState.forecast_year must be set before UrbanSim preprocess logging."
            )
        if urbansim_settings is None:
            raise RuntimeError(
                "UrbanSim config is required for UrbanSim preprocess logging."
            )
        for short_name, path, description in outputs._iter_record_items():
            log_and_set_output(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
                **_urbansim_output_facet_meta(short_name, forecast_year=forecast_year),
            )
        usim_data_dir = outputs.usim_mutable_data_dir
        usim_input_fname = urbansim_settings.input_file_template.format(
            region_id=urbansim_settings.region_mappings["region_to_region_id"][
                settings.run.region
            ]
        )
        usim_input_path = usim_data_dir / usim_input_fname
        if usim_input_path.exists():
            log_and_set_output(
                key=USIM_DATASTORE_BASE_H5,
                path=str(usim_input_path),
                description="UrbanSim base datastore for preprocessing",
                coupler=coupler,
                profile_file_schema=True,
                h5_container=True,
                hash_tables="if_unchanged",
                **_urbansim_output_facet_meta(
                    USIM_DATASTORE_BASE_H5, forecast_year=forecast_year
                ),
            )

    return _make_typed_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="urbansim",
        phase="preprocess",
        outputs_class=UrbanSimPreprocessOutputs,
        component_getter=lambda factory, state: factory.get_preprocessor(
            "urbansim", state
        ),
        component_executor=_execute_urbansim_preprocess_typed,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "urbansim_preprocess", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )


def make_urbansim_run_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the UrbanSim run step function.

    This step executes the UrbanSim land-use simulation for the forecast year
    and produces the UrbanSim datastore output.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing run outputs.

    Returns
    -------
    callable
        Step function for UrbanSim run.
    """

    def _log_outputs(
        outputs: UrbanSimRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        forecast_year = state.forecast_year
        if forecast_year is None:
            raise RuntimeError(
                "WorkflowState.forecast_year must be set before UrbanSim run logging."
            )
        if outputs.usim_datastore_h5 is not None:
            log_and_set_output(
                key=USIM_DATASTORE_H5,
                path=str(outputs.usim_datastore_h5),
                description=(f"UrbanSim datastore output for year {forecast_year}"),
                coupler=coupler,
                profile_file_schema=True,
                h5_container=True,
                hash_tables="if_unchanged",
                **_urbansim_output_facet_meta(
                    USIM_DATASTORE_H5, forecast_year=forecast_year
                ),
            )

    return _make_typed_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="urbansim",
        phase="run",
        outputs_class=UrbanSimRunOutputs,
        component_getter=lambda factory, state: factory.get_runner("urbansim", state),
        component_executor=_execute_urbansim_run_typed,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "urbansim_run", outputs
        ),
        output_logger=_log_outputs,
        output_recoverer=_recover_urbansim_run_outputs,
    )


def make_urbansim_postprocess_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the UrbanSim postprocess step function.

    This step merges UrbanSim outputs into the input datastore used by
    downstream models and prepares the HDF5 for the next stage.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing postprocess outputs.

    Returns
    -------
    callable
        Step function for UrbanSim postprocess.
    """

    def _log_outputs(
        outputs: UrbanSimPostprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        forecast_year = state.forecast_year
        if forecast_year is None:
            raise RuntimeError(
                "WorkflowState.forecast_year must be set before UrbanSim postprocess logging."
            )
        for short_name, path, description in outputs._iter_record_items():
            if short_name.startswith(USIM_INPUT_ARCHIVE_PREFIX):
                log_output_only(
                    key=short_name,
                    path=str(path),
                    description=description,
                    profile_file_schema=True,
                    h5_container=True,
                    hash_tables="if_unchanged",
                    **_urbansim_output_facet_meta(
                        short_name, forecast_year=forecast_year
                    ),
                )
                archive_table_keys = _root_h5_table_keys(
                    str(path),
                    key_prefix="urbansim_postprocess_usim_",
                    key_suffix="_table_archived",
                )
                if archive_table_keys:
                    _log_named_h5_tables(
                        path=str(path),
                        direction="output",
                        table_keys=archive_table_keys,
                        description_by_table=_root_h5_table_descriptions(
                            str(path),
                            action="archived by UrbanSim postprocess",
                        ),
                    )
        if outputs.usim_datastore_h5 is not None:
            log_and_set_output(
                key=USIM_DATASTORE_H5,
                path=str(outputs.usim_datastore_h5),
                description=(
                    "UrbanSim datastore prepared for next iteration "
                    f"(year {forecast_year})"
                ),
                coupler=coupler,
                profile_file_schema=True,
                h5_container=True,
                hash_tables="if_unchanged",
                **_urbansim_output_facet_meta(
                    USIM_DATASTORE_H5, forecast_year=forecast_year
                ),
            )
            merged_table_keys = _root_h5_table_keys(
                str(outputs.usim_datastore_h5),
                key_prefix="urbansim_postprocess_usim_",
                key_suffix="_table_updated",
            )
            if merged_table_keys:
                _log_named_h5_tables(
                    path=str(outputs.usim_datastore_h5),
                    direction="output",
                    table_keys=merged_table_keys,
                    description_by_table=_root_h5_table_descriptions(
                        str(outputs.usim_datastore_h5),
                        action="prepared for the next iteration by UrbanSim postprocess",
                    ),
                )

    return _make_typed_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="urbansim",
        phase="postprocess",
        outputs_class=UrbanSimPostprocessOutputs,
        component_getter=lambda factory, state: factory.get_postprocessor(
            "urbansim", state
        ),
        component_executor=_execute_urbansim_postprocess_typed,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "urbansim_postprocess", outputs
        ),
        output_logger=_log_outputs,
        output_recoverer=_recover_urbansim_postprocess_outputs,
    )


def make_atlas_preprocess_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ATLAS preprocess step function.

    This step extracts UrbanSim HDF5 tables into ATLAS input CSVs and optionally
    computes accessibility metrics required by the ATLAS vehicle ownership model.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing preprocess outputs.

    Returns
    -------
    callable
        Step function for ATLAS preprocess.
    """

    def _log_outputs(
        outputs: AtlasPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        forecast_year = state.forecast_year
        if forecast_year is None:
            raise RuntimeError(
                "WorkflowState.forecast_year must be set before ATLAS preprocess logging."
            )
        run_scenario = getattr(getattr(settings, "atlas", None), "scenario", None)
        prepared_meta = getattr(outputs, "prepared_input_meta", {})
        _log_step_records(
            record_items=outputs._iter_record_items(),
            log_fn=log_output_only,
            profile_schema_suffixes=(".csv", ".parquet"),
            extra_meta_fn=lambda key, _path, _description: {
                **prepared_meta.get(key, {}),
                **_atlas_artifact_facet_meta(
                    key,
                    run_scenario=run_scenario,
                    forecast_year=forecast_year,
                    artifact_family="atlas_preprocess_output",
                ),
            },
        )

    return _make_typed_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="atlas",
        phase="preprocess",
        outputs_class=AtlasPreprocessOutputs,
        component_getter=lambda factory, state: factory.get_preprocessor(
            "atlas", state
        ),
        component_executor=_execute_atlas_preprocess_typed,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "atlas_preprocess", outputs
        ),
        output_logger=_log_outputs,
    )


def make_atlas_run_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ATLAS run step function.

    This step runs ATLAS to simulate vehicle ownership for the sub-year and
    produces household/vehicle output CSVs.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing run outputs.

    Returns
    -------
    callable
        Step function for ATLAS run.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        forecast_year = state.forecast_year
        if forecast_year is None:
            raise RuntimeError(
                "WorkflowState.forecast_year must be set before ATLAS run logging."
            )
        upstream = holder.atlas_preprocess
        if upstream is None:
            raise RuntimeError("ATLAS preprocess must complete first")
        run_scenario = getattr(getattr(settings, "atlas", None), "scenario", None)
        prepared_meta = getattr(upstream, "prepared_input_meta", {})
        _log_step_records(
            record_items=upstream._iter_record_items(),
            log_fn=log_input_only,
            profile_schema_suffixes=(".csv", ".parquet"),
            extra_meta_fn=lambda key, _path, _description: {
                **prepared_meta.get(key, {}),
                **_atlas_artifact_facet_meta(
                    key,
                    run_scenario=run_scenario,
                    forecast_year=forecast_year,
                    artifact_family="atlas_run_input",
                ),
            },
        )
        return {}

    def _log_outputs(
        outputs: AtlasRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        _log_step_records(
            record_items=outputs._iter_record_items(),
            log_fn=log_output_only,
            profile_schema_suffixes=(".csv", ".parquet"),
        )

    return _make_typed_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="atlas",
        phase="run",
        outputs_class=AtlasRunOutputs,
        component_getter=lambda factory, state: factory.get_runner("atlas", state),
        component_executor=_execute_atlas_run_typed,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "atlas_run", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
        output_recoverer=_recover_atlas_run_outputs,
    )


def make_atlas_postprocess_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ATLAS postprocess step function.

    This step updates the UrbanSim HDF5 datastore with ATLAS vehicle ownership
    results and writes vehicles2 outputs used by BEAM.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing postprocess outputs.

    Returns
    -------
    callable
        Step function for ATLAS postprocess.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        forecast_year = state.forecast_year
        urbansim_settings = settings.urbansim
        if forecast_year is None:
            raise RuntimeError(
                "WorkflowState.forecast_year must be set before ATLAS postprocess."
            )
        if urbansim_settings is None:
            raise RuntimeError(
                "UrbanSim config is required for ATLAS postprocess logging."
            )
        usim_output_path = resolve_atlas_usim_datastore_path(
            settings, state, workspace
        )
        if usim_output_path.exists():
            log_input_only(
                key=USIM_DATASTORE_H5,
                path=str(usim_output_path),
                description=(
                    "UrbanSim datastore consumed by ATLAS postprocess "
                    f"for year {forecast_year}"
                ),
                profile_file_schema=True,
                h5_container=True,
                hash_tables="if_unchanged",
                **_urbansim_output_facet_meta(
                    USIM_DATASTORE_H5, forecast_year=forecast_year
                ),
            )
            households_table_path = (
                "/households"
                if state.is_start_year()
                else f"/{forecast_year}/households"
            )
            _log_named_h5_tables(
                path=str(usim_output_path),
                direction="input",
                table_keys={
                    households_table_path: "atlas_postprocess_usim_households_table_input"
                },
                description_by_table={
                    households_table_path: (
                        "UrbanSim households table consumed by ATLAS postprocess"
                    )
                },
            )
        return {}

    def _log_outputs(
        outputs: AtlasPostprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        forecast_year = state.forecast_year
        if forecast_year is None:
            raise RuntimeError(
                "WorkflowState.forecast_year must be set before ATLAS postprocess logging."
            )
        _log_step_records(
            record_items=(
                (short_name, path, description)
                for short_name, path, description in outputs._iter_record_items()
                if short_name != USIM_H5_UPDATED
            ),
            log_fn=log_output_only,
            profile_schema_suffixes=(".csv", ".parquet"),
        )
        if outputs.usim_datastore_h5 is not None:
            log_and_set_output(
                key=USIM_DATASTORE_H5,
                path=str(outputs.usim_datastore_h5),
                description=(
                    f"UrbanSim datastore updated by ATLAS for year {forecast_year}"
                ),
                coupler=coupler,
                profile_file_schema=True,
                h5_container=True,
                hash_tables="if_unchanged",
                **_urbansim_output_facet_meta(
                    USIM_DATASTORE_H5, forecast_year=forecast_year
                ),
            )
            households_table_path = (
                "/households"
                if state.is_start_year()
                else f"/{forecast_year}/households"
            )
            _log_named_h5_tables(
                path=str(outputs.usim_datastore_h5),
                direction="output",
                table_keys={
                    households_table_path: "atlas_postprocess_usim_households_table_updated"
                },
                description_by_table={
                    households_table_path: (
                        "UrbanSim households table updated by ATLAS postprocess"
                    )
                },
            )

    return _make_typed_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="atlas",
        phase="postprocess",
        outputs_class=AtlasPostprocessOutputs,
        component_getter=lambda factory, state: factory.get_postprocessor(
            "atlas", state
        ),
        component_executor=_execute_atlas_postprocess_typed,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "atlas_postprocess", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
        output_recoverer=_recover_atlas_postprocess_outputs,
    )
