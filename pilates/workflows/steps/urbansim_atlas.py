from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import pandas as pd
from consist import BindingResult, CacheOptions, ExecutionOptions, define_step
from consist.types import H5ChildSpec

from pilates.atlas.postprocessor import AtlasPostprocessor
from pilates.atlas.preprocessor import (
    _resolve_atlas_h5_table_key,
    _restore_restart_atlas_year_inputs,
)
from pilates.atlas.preprocessor import AtlasPreprocessor
from pilates.atlas.runner import AtlasRunner
from pilates.config.models import PilatesConfig
from pilates.urbansim.outputs import (
    UrbanSimPostprocessOutputs as NativeUrbanSimPostprocessOutputs,
)
from pilates.urbansim.outputs import (
    UrbanSimPreprocessOutputs as NativeUrbanSimPreprocessOutputs,
)
from pilates.urbansim.outputs import UrbanSimRunOutputs as NativeUrbanSimRunOutputs
from pilates.urbansim.postprocessor import UrbansimPostprocessor
from pilates.urbansim.runner import UrbansimRunner
from pilates.urbansim.preprocessor import UrbansimPreprocessor
from pilates.utils.coupler_helpers import artifact_to_existing_path
from pilates.utils.usim_h5 import ensure_usim_population_year_table_aliases
from pilates.workflows.artifact_keys import (
    ATLAS_OUTPUT_DIR,
    ATLAS_VEHICLES2_OUTPUT,
    FINAL_SKIMS_OMX,
    USIM_POPULATION_SOURCE_H5,
    USIM_MUTABLE_DATA_DIR,
)
from pilates.workflows.output_projection import require_output
from pilates.workflows.outputs_base import ValidationContext
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.step_definition import StepDefinition
from pilates.workflows.coupler_namespace import resolve_coupler_value
from pilates.workflows.step_consist_meta import consist_step_meta
from pilates.utils.consist_runtime import require_runtime_kwargs
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
    StandardStepSpec,
    StepOutputsHolder,
    UrbanSimPostprocessOutputs,
    UrbanSimPreprocessOutputs,
    UrbanSimRunOutputs,
    WorkflowState,
    _atlas_artifact_facet_meta,
    build_standard_step,
    _log_step_records,
    make_default_recoverer,
    _urbansim_output_facet_meta,
    log_and_set_output,
    log_input_only,
    log_output_only,
    logger,
    recovered_cached_paths,
    warm_start_activities,
)


def _strip_component_runtime_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    filtered = dict(kwargs)
    filtered.pop("coupler", None)
    filtered.pop("context", None)
    if "omx_skims" in filtered and "final_skims_omx" not in filtered:
        filtered["final_skims_omx"] = filtered.pop("omx_skims")
    return filtered


def _urbansim_preprocess_output_paths(
    settings: PilatesConfig, state: WorkflowState, workspace: Workspace
) -> Dict[str, Any]:
    """Keep H5 container logging under the callback's specialized owner."""
    expected_outputs = UrbansimPreprocessor.expected_outputs(settings, state, workspace)
    return {
        key: path for key, path in expected_outputs.items() if key != USIM_DATASTORE_H5
    }


def _atlas_postprocess_output_paths(
    settings: PilatesConfig, state: WorkflowState, workspace: Workspace
) -> Dict[str, Any]:
    """Declare only the coarse directory; callbacks own H5 and handoff files."""
    expected_outputs = AtlasPostprocessor.expected_outputs(settings, state, workspace)
    return (
        {ATLAS_OUTPUT_DIR: expected_outputs[ATLAS_OUTPUT_DIR]}
        if ATLAS_OUTPUT_DIR in expected_outputs
        else {}
    )


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


def _resolve_atlas_postprocess_households_table_path(
    *,
    path: str,
    forecast_year: int,
    is_start_year: bool,
) -> str:
    """
    Resolve the households table path ATLAS postprocess will actually touch.

    Fall back to the legacy default when the file is missing or unreadable so
    logging remains best-effort and does not block execution.
    """
    default_path = "/households" if is_start_year else f"/{forecast_year}/households"
    try:
        with pd.HDFStore(path, mode="r") as store:
            resolved = _resolve_atlas_h5_table_key(
                store,
                year=forecast_year,
                table="households",
                is_start_year=is_start_year,
            )
    except Exception:
        return default_path

    return resolved if str(resolved).startswith("/") else f"/{resolved}"


def _atlas_postprocess_output_meta(
    *,
    key: str,
    state: WorkflowState,
) -> Dict[str, Any]:
    if key != ATLAS_VEHICLES2_OUTPUT:
        return {}
    return {
        "facet": {
            "artifact_family": ATLAS_VEHICLES2_OUTPUT,
            "source_role": ATLAS_VEHICLES2_OUTPUT,
            "year": getattr(state, "forecast_year", None),
            "iteration": getattr(state, "iteration", None),
        },
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _urbansim_run_recovered_datastore(
    recovered_paths: Mapping[str, Path], _state: WorkflowState
) -> Optional[Path]:
    return recovered_paths.get(USIM_FORECAST_OUTPUT) or recovered_paths.get(
        USIM_DATASTORE_H5
    )


def _urbansim_postprocess_recovered_datastore(
    recovered_paths: Mapping[str, Path], _state: WorkflowState
) -> Optional[Path]:
    return next(
        (
            path
            for key, path in recovered_paths.items()
            if key.startswith(USIM_INPUT_MERGED_PREFIX)
        ),
        None,
    ) or recovered_paths.get(USIM_DATASTORE_H5)


def _atlas_run_required_keys(
    state: WorkflowState, _recovered_paths: Mapping[str, Path]
) -> Sequence[str]:
    output_year = getattr(state, "forecast_year", None)
    if output_year is None:
        return ("__missing_forecast_year__",)
    return (
        f"householdv_{output_year}",
        f"vehicles_{output_year}",
    )


_recover_urbansim_preprocess_outputs = make_default_recoverer(
    outputs_class=UrbanSimPreprocessOutputs,
    mapping_field="prepared_inputs",
    dir_field="usim_mutable_data_dir",
    dir_getter=lambda workspace: workspace.get_usim_mutable_data_dir(),
    step_logger=logger,
    log_context="UrbanSim/ATLAS cached output recovery",
)


_recover_urbansim_run_outputs = make_default_recoverer(
    outputs_class=UrbanSimRunOutputs,
    mapping_field="raw_outputs",
    primary_path_field="usim_datastore_h5",
    primary_path_resolver=_urbansim_run_recovered_datastore,
    step_logger=logger,
    log_context="UrbanSim/ATLAS cached output recovery",
)


_recover_urbansim_postprocess_outputs = make_default_recoverer(
    outputs_class=UrbanSimPostprocessOutputs,
    mapping_field="processed_outputs",
    primary_path_field="usim_datastore_h5",
    primary_path_resolver=_urbansim_postprocess_recovered_datastore,
    step_logger=logger,
    log_context="UrbanSim/ATLAS cached output recovery",
)


_recover_atlas_preprocess_outputs = make_default_recoverer(
    outputs_class=AtlasPreprocessOutputs,
    mapping_field="prepared_inputs",
    dir_field="atlas_mutable_input_dir",
    dir_getter=lambda workspace: workspace.get_atlas_mutable_input_dir(),
    step_logger=logger,
    log_context="UrbanSim/ATLAS cached output recovery",
)


_recover_atlas_run_outputs = make_default_recoverer(
    outputs_class=AtlasRunOutputs,
    mapping_field="raw_outputs",
    dir_field="atlas_output_dir",
    dir_getter=lambda workspace: workspace.get_atlas_output_dir(),
    required_keys=_atlas_run_required_keys,
    step_logger=logger,
    log_context="UrbanSim/ATLAS cached output recovery",
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
    del coupler, outputs_holder
    recovered_paths = recovered_cached_paths(
        cached_outputs=cached_outputs,
        run_id=run_id,
        workspace=workspace,
        step_logger=logger,
        log_context="UrbanSim/ATLAS cached output recovery",
    )
    expected_outputs = AtlasPostprocessor.expected_outputs(settings, state, workspace)
    if ATLAS_VEHICLES2_OUTPUT not in recovered_paths:
        vehicles2_path = artifact_to_existing_path(
            expected_outputs.get(ATLAS_VEHICLES2_OUTPUT),
            workspace=workspace,
        )
        if vehicles2_path is not None:
            recovered_paths[ATLAS_VEHICLES2_OUTPUT] = Path(vehicles2_path)
    if ATLAS_VEHICLES2_OUTPUT not in recovered_paths:
        return None
    usim_datastore_h5 = (
        recovered_paths.get(USIM_POPULATION_SOURCE_H5)
        or recovered_paths.get(USIM_H5_UPDATED)
        or recovered_paths.get(USIM_DATASTORE_H5)
    )
    if usim_datastore_h5 is None:
        candidate_path = artifact_to_existing_path(
            expected_outputs.get(USIM_POPULATION_SOURCE_H5),
            workspace=workspace,
        )
        if candidate_path is not None:
            usim_datastore_h5 = Path(candidate_path)
    if usim_datastore_h5 is None:
        return None

    recovered_paths = dict(recovered_paths)
    recovered_paths[USIM_POPULATION_SOURCE_H5] = usim_datastore_h5
    recovered_paths[USIM_DATASTORE_H5] = usim_datastore_h5
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
    return preprocessor.preprocess(
        workspace,
        **_strip_component_runtime_kwargs(kwargs),
    )


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


def atlas_preprocess_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> Dict[str, Any]:
    return AtlasPreprocessor.expected_outputs(settings, state, workspace)


def atlas_run_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> Dict[str, Any]:
    return AtlasRunner.expected_outputs(settings, state, workspace)


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


def _rehydrate_restart_atlas_preprocess_state(
    *,
    state: WorkflowState,
    workspace: Workspace,
) -> bool:
    """
    Restore restart-critical ATLAS year directories for recovered preprocess outputs.

    ``atlas_preprocess`` can now be restored from manifest/cache without rerunning
    the component. When ``atlas_run`` reruns after that restore, the dynamic ATLAS
    container still expects the base-year and immediately preceding subyear
    ``atlas_input/year*`` directories to exist locally. Rehydrate them here so the
    output replayer makes recovered preprocess state runner-ready.
    """
    if not bool(getattr(state, "is_restart_run", False)):
        return False

    run_info_path = getattr(state, "run_info_path", None)
    start_year = getattr(state, "start_year", None)
    atlas_year = getattr(state, "year", getattr(state, "current_year", None))
    if (
        not run_info_path
        or start_year is None
        or atlas_year is None
        or not os.path.exists(run_info_path)
    ):
        return False

    atlas_input_root = workspace.get_atlas_mutable_input_dir()
    from pilates.atlas.preprocessor import restart_required_atlas_input_paths

    required_paths = restart_required_atlas_input_paths(
        atlas_input_root=atlas_input_root,
        start_year=int(start_year),
        atlas_year=int(atlas_year),
    )
    if all(os.path.exists(path) for path in required_paths.values()):
        return False

    _restore_restart_atlas_year_inputs(
        previous_run_dir=os.path.dirname(run_info_path),
        workspace=workspace,
        start_year=int(start_year),
        atlas_year=int(atlas_year),
    )
    return True


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
            # Apply H5 recovery policies to the datastore if it matches
            h5_meta = {}
            if short_name == "usim_datastore_h5":
                h5_meta = {
                    "profile_file_schema": True,
                    "h5_container": True,
                    "container_recovery_unit": "parent_file",
                    "child_recovery_policy": "descriptive_only",
                }
            log_and_set_output(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
                step_name="urbansim_preprocess",
                **h5_meta,
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
                step_name="urbansim_preprocess",
                profile_file_schema=True,
                h5_container=True,
                container_recovery_unit="parent_file",
                child_recovery_policy="descriptive_only",
                hash_tables="if_unchanged",
                **_urbansim_output_facet_meta(
                    USIM_DATASTORE_BASE_H5, forecast_year=forecast_year
                ),
            )

    step_func = build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="urbansim_preprocess",
            model_name="urbansim",
            phase="preprocess",
            outputs_class=UrbanSimPreprocessOutputs,
            component_getter=lambda factory, state: factory.get_preprocessor(
                "urbansim", state
            ),
            component_executor=_execute_urbansim_preprocess_typed,
            input_logger=_log_inputs,
            output_logger=_log_outputs,
            output_recoverer=_recover_urbansim_preprocess_outputs,
            output_paths=_urbansim_preprocess_output_paths,
            input_binding="none",
            cache_hydration="metadata",
            step_logger=logger,
        ),
    )
    return step_func


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
                description=(
                    f"UrbanSim forecast datastore output for year {forecast_year}"
                ),
                coupler=coupler,
                step_name="urbansim_run",
                profile_file_schema=True,
                h5_container=True,
                container_recovery_unit="parent_file",
                child_recovery_policy="descriptive_only",
                hash_tables="if_unchanged",
                **_urbansim_output_facet_meta(
                    USIM_DATASTORE_H5, forecast_year=forecast_year
                ),
            )
            log_and_set_output(
                key=USIM_FORECAST_OUTPUT,
                path=str(outputs.usim_datastore_h5),
                description=(
                    f"UrbanSim forecast datastore output for year {forecast_year}"
                ),
                coupler=coupler,
                step_name="urbansim_run",
                profile_file_schema=True,
                h5_container=True,
                container_recovery_unit="parent_file",
                child_recovery_policy="descriptive_only",
                hash_tables="if_unchanged",
                **_urbansim_output_facet_meta(
                    USIM_FORECAST_OUTPUT, forecast_year=forecast_year
                ),
            )

    step_func = build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="urbansim_run",
            model_name="urbansim",
            phase="run",
            outputs_class=UrbanSimRunOutputs,
            component_getter=lambda factory, state: factory.get_runner(
                "urbansim", state
            ),
            component_executor=_execute_urbansim_run_typed,
            output_logger=_log_outputs,
            output_recoverer=_recover_urbansim_run_outputs,
            output_paths=None,
            input_binding="none",
            cache_hydration="metadata",
            step_logger=logger,
        ),
    )
    return step_func


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
                archive_table_keys = _root_h5_table_keys(
                    str(path),
                    key_prefix="urbansim_postprocess_usim_",
                    key_suffix="_table_archived",
                )
                if archive_table_keys:
                    archive_descriptions = _root_h5_table_descriptions(
                        str(path),
                        action="archived by UrbanSim postprocess",
                    )
                    child_specs = {
                        table_path: H5ChildSpec(
                            key=artifact_key,
                            description=archive_descriptions.get(table_path),
                            metadata={
                                "h5_parent_key": artifact_key.rsplit("_table_", 1)[0],
                                "h5_table_name": table_path.split("/")[-1],
                            },
                        )
                        for table_path, artifact_key in archive_table_keys.items()
                    }
                else:
                    child_specs = None
                log_output_only(
                    key=short_name,
                    path=str(path),
                    description=description,
                    step_name="urbansim_postprocess",
                    profile_file_schema=True,
                    h5_container=True,
                    container_recovery_unit="parent_file",
                    child_recovery_policy="descriptive_only",
                    hash_tables="if_unchanged",
                    h5_tables_used=list(archive_table_keys.keys()),
                    child_specs=child_specs,
                    child_selection="include_only" if child_specs else "all",
                    **_urbansim_output_facet_meta(
                        short_name, forecast_year=forecast_year
                    ),
                )
        if outputs.usim_datastore_h5 is not None:
            merged_table_keys = _root_h5_table_keys(
                str(outputs.usim_datastore_h5),
                key_prefix="urbansim_postprocess_usim_",
                key_suffix="_table_updated",
            )
            if merged_table_keys:
                merged_descriptions = _root_h5_table_descriptions(
                    str(outputs.usim_datastore_h5),
                    action="prepared for the next iteration by UrbanSim postprocess",
                )
                merged_child_specs = {
                    table_path: H5ChildSpec(
                        key=artifact_key,
                        description=merged_descriptions.get(table_path),
                        metadata={
                            "h5_parent_key": artifact_key.rsplit("_table_", 1)[0],
                            "h5_table_name": table_path.split("/")[-1],
                        },
                    )
                    for table_path, artifact_key in merged_table_keys.items()
                }
            else:
                merged_child_specs = None
            merged_meta = _urbansim_output_facet_meta(
                USIM_DATASTORE_H5, forecast_year=forecast_year
            )
            merged_meta.setdefault("facet", {}).update(
                {
                    "source_role": f"{USIM_INPUT_MERGED_PREFIX}{forecast_year}",
                    "snapshot_role": "usim_input_merged",
                    "snapshot_reason": "post_merge_handoff",
                    "storage_event": "merged_h5_output",
                }
            )
            log_and_set_output(
                key=USIM_DATASTORE_H5,
                path=str(outputs.usim_datastore_h5),
                description=(
                    "UrbanSim datastore prepared for next iteration "
                    f"(year {forecast_year})"
                ),
                coupler=coupler,
                step_name="urbansim_postprocess",
                profile_file_schema=True,
                h5_container=True,
                container_recovery_unit="parent_file",
                child_recovery_policy="descriptive_only",
                hash_tables="if_unchanged",
                h5_tables_used=list(merged_table_keys.keys()),
                child_specs=merged_child_specs,
                child_selection="include_only" if merged_child_specs else "all",
                **merged_meta,
            )

    step_func = build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="urbansim_postprocess",
            model_name="urbansim",
            phase="postprocess",
            outputs_class=UrbanSimPostprocessOutputs,
            component_getter=lambda factory, state: factory.get_postprocessor(
                "urbansim", state
            ),
            component_executor=_execute_urbansim_postprocess_typed,
            output_logger=_log_outputs,
            output_recoverer=_recover_urbansim_postprocess_outputs,
            output_paths=None,
            input_binding="none",
            cache_hydration="metadata",
            step_logger=logger,
        ),
    )
    return step_func


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
            log_fn=lambda key, path, description, **meta: log_output_only(
                key=key,
                path=path,
                description=description,
                step_name="atlas_preprocess",
                **meta,
            ),
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

    def _replay_outputs(
        outputs: AtlasPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        setattr(
            outputs,
            "_restart_rehydration_used",
            _rehydrate_restart_atlas_preprocess_state(
                state=state,
                workspace=workspace,
            ),
        )
        _log_outputs(outputs, settings, state, workspace, holder)

    step_func = build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="atlas_preprocess",
            model_name="atlas",
            phase="preprocess",
            outputs_class=AtlasPreprocessOutputs,
            component_getter=lambda factory, state: factory.get_preprocessor(
                "atlas", state
            ),
            component_executor=_execute_atlas_preprocess_typed,
            output_logger=_log_outputs,
            output_replayer=_replay_outputs,
            output_recoverer=_recover_atlas_preprocess_outputs,
            output_paths=AtlasPreprocessor.expected_outputs,
            input_binding="none",
            cache_hydration="metadata",
            step_logger=logger,
        ),
    )
    return step_func


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
            log_fn=lambda key, path, description, **meta: log_output_only(
                key=key,
                path=path,
                description=description,
                step_name="atlas_run",
                **meta,
            ),
            profile_schema_suffixes=(".csv", ".parquet"),
        )

    step_func = build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="atlas_run",
            model_name="atlas",
            phase="run",
            outputs_class=AtlasRunOutputs,
            component_getter=lambda factory, state: factory.get_runner("atlas", state),
            component_executor=_execute_atlas_run_typed,
            input_logger=_log_inputs,
            output_logger=_log_outputs,
            output_recoverer=_recover_atlas_run_outputs,
            output_paths=AtlasRunner.expected_outputs,
            input_binding="none",
            cache_hydration="metadata",
            step_logger=logger,
        ),
    )
    return step_func


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
        expected_inputs = AtlasPostprocessor.expected_inputs(settings, state, workspace)
        usim_output_path = expected_inputs.get("usim_datastore_h5")
        if usim_output_path is not None:
            usim_output_path = Path(usim_output_path)
        if usim_output_path is not None and usim_output_path.exists():
            households_table_path = _resolve_atlas_postprocess_households_table_path(
                path=str(usim_output_path),
                forecast_year=forecast_year,
                is_start_year=state.is_start_year(),
            )
            artifact_key = "atlas_postprocess_usim_households_table_input"
            log_input_only(
                key=USIM_DATASTORE_H5,
                path=str(usim_output_path),
                description=(
                    "UrbanSim datastore consumed by ATLAS postprocess "
                    f"for year {forecast_year}"
                ),
                profile_file_schema=True,
                h5_container=True,
                container_recovery_unit="parent_file",
                child_recovery_policy="descriptive_only",
                hash_tables="if_unchanged",
                h5_tables_used=[households_table_path],
                child_specs={
                    households_table_path: H5ChildSpec(
                        key=artifact_key,
                        description="UrbanSim households table consumed by ATLAS postprocess",
                        metadata={
                            "h5_parent_key": artifact_key.rsplit("_table_", 1)[0],
                            "h5_table_name": households_table_path.split("/")[-1],
                        },
                    )
                },
                child_selection="include_only",
                **_urbansim_output_facet_meta(
                    USIM_DATASTORE_H5, forecast_year=forecast_year
                ),
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
                if short_name
                not in {USIM_H5_UPDATED, USIM_DATASTORE_H5, USIM_POPULATION_SOURCE_H5}
            ),
            log_fn=lambda key, path, description, **meta: log_output_only(
                key=key,
                path=path,
                description=description,
                step_name="atlas_postprocess",
                **meta,
            ),
            profile_schema_suffixes=(".csv", ".parquet"),
            extra_meta_fn=lambda key, _path, _description: (
                _atlas_postprocess_output_meta(key=key, state=state)
            ),
        )
        if outputs.usim_datastore_h5 is not None:
            if not state.is_start_year():
                try:
                    ensure_usim_population_year_table_aliases(
                        h5_path=str(outputs.usim_datastore_h5),
                        year=forecast_year,
                    )
                except Exception:
                    logger.debug(
                        "Failed to normalize ATLAS population-source H5 aliases for %s",
                        outputs.usim_datastore_h5,
                        exc_info=True,
                    )
            households_table_path = _resolve_atlas_postprocess_households_table_path(
                path=str(outputs.usim_datastore_h5),
                forecast_year=forecast_year,
                is_start_year=state.is_start_year(),
            )
            artifact_key = "atlas_postprocess_usim_households_table_updated"
            log_and_set_output(
                key=USIM_POPULATION_SOURCE_H5,
                path=str(outputs.usim_datastore_h5),
                description=(
                    f"UrbanSim datastore selected as the ActivitySim population source after ATLAS for year {forecast_year}"
                ),
                coupler=coupler,
                step_name="atlas_postprocess",
                profile_file_schema=True,
                h5_container=True,
                container_recovery_unit="parent_file",
                child_recovery_policy="descriptive_only",
                hash_tables="if_unchanged",
                h5_tables_used=[households_table_path],
                child_specs={
                    households_table_path: H5ChildSpec(
                        key=artifact_key,
                        description="UrbanSim households table updated by ATLAS postprocess",
                        metadata={
                            "h5_parent_key": artifact_key.rsplit("_table_", 1)[0],
                            "h5_table_name": households_table_path.split("/")[-1],
                        },
                    )
                },
                child_selection="include_only",
                **_urbansim_output_facet_meta(
                    USIM_POPULATION_SOURCE_H5, forecast_year=forecast_year
                ),
            )

    step_func = build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="atlas_postprocess",
            model_name="atlas",
            phase="postprocess",
            outputs_class=AtlasPostprocessOutputs,
            component_getter=lambda factory, state: factory.get_postprocessor(
                "atlas", state
            ),
            component_executor=_execute_atlas_postprocess_typed,
            input_logger=_log_inputs,
            output_logger=_log_outputs,
            output_recoverer=_recover_atlas_postprocess_outputs,
            output_paths=_atlas_postprocess_output_paths,
            manual_output_keys=[
                USIM_POPULATION_SOURCE_H5,
                ATLAS_VEHICLES2_OUTPUT,
            ],
            input_binding="none",
            cache_mode="overwrite",
            cache_hydration="metadata",
            step_logger=logger,
        ),
    )
    return step_func


# Native Consist step definitions.  The legacy factories above remain available until
# the coordinated stage cutover removes the holder/manifest architecture.  These
# definitions deliberately contain no holder, cache-recovery, or coupler-publication
# behavior: Consist owns the binding and persisted output links, while the projectors
# below reconstruct only the existing typed output values from those links.


def _persisted_output_path(
    outputs: Mapping[str, Any],
    *,
    step_name: str,
    key: str,
    declared_outputs: Mapping[str, Any],
    workspace: Workspace,
) -> Path:
    """Return one output only when it exists at its declared destination.

    A projection represents the current step execution. It must not fall back
    to an artifact's mounted source, archive location, or historical workspace
    merely because that artifact remains available after a cache hit.
    """

    require_output(outputs, step_name=step_name, key=key)
    try:
        output_spec = declared_outputs[key]
    except KeyError as exc:
        raise RuntimeError(
            f"{step_name} output {key!r} has no declared destination"
        ) from exc

    if isinstance(output_spec, os.PathLike):
        path = Path(output_spec)
    elif isinstance(output_spec, str):
        if output_spec.startswith("workspace://"):
            path = Path(workspace.full_path) / output_spec[
                len("workspace://") :
            ].lstrip("/")
        elif "://" in output_spec:
            raise RuntimeError(
                f"{step_name} output {key!r} has a non-local declared destination"
            )
        else:
            path = Path(output_spec)
            if not path.is_absolute():
                path = Path(workspace.full_path) / path
    else:
        raise RuntimeError(
            f"{step_name} output {key!r} has an invalid declared destination"
        )

    if not path.exists():
        raise RuntimeError(
            f"{step_name} output {key!r} is missing at declared destination {path}"
        )
    return path


def _validation_context(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    step_name: str,
) -> ValidationContext:
    return ValidationContext(
        settings=settings,
        state=state,
        workspace=workspace,
        step_name=step_name,
    )


def _resolve_native_inputs(
    *,
    step_name: str,
    required_roles: Sequence[str],
    optional_roles: Sequence[str],
    logical_destinations: Mapping[str, Path],
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    """Select fixed semantic roles once and retain their Consist artifacts."""

    selected_inputs: Dict[str, Any] = {}
    source_by_role: Dict[str, str] = {}
    selected_key_by_role: Dict[str, str] = {}
    selected_destinations: Dict[str, Path] = {}

    for role in (*required_roles, *optional_roles):
        resolved = resolve_coupler_value(coupler, role)
        source_by_role[role] = resolved.source
        if resolved.storage_key is not None:
            selected_key_by_role[role] = resolved.storage_key
        if resolved.value is None:
            continue
        selected_inputs[role] = resolved.value
        destination = logical_destinations.get(role)
        if destination is not None:
            selected_destinations[role] = destination

    return ResolvedStepInputs(
        step_name=step_name,
        binding=BindingResult(inputs=selected_inputs),
        required_roles=tuple(required_roles),
        optional_roles=tuple(optional_roles),
        source_by_role=source_by_role,
        selected_key_by_role=selected_key_by_role,
        logical_destinations=selected_destinations,
    )


def _native_execution_options(
    *,
    resolved_inputs: ResolvedStepInputs | None = None,
    **_: Any,
) -> ExecutionOptions:
    """Request Consist staging at the deterministic model-owned destinations."""

    return ExecutionOptions(
        input_binding="paths",
        input_paths=resolved_inputs.logical_destinations,
        input_materialization="requested",
        input_materialization_mode="copy",
    )


def _strict_requested_output_cache_options(**_: Any) -> CacheOptions:
    """Admit a cached model step only after all requested output paths hydrate."""

    return CacheOptions(
        cache_hydration="outputs-requested",
        cache_hydration_failure="miss",
    )


def _native_contract_output_paths(
    provider: Callable[..., Mapping[str, Any]],
) -> Callable[[Any], Mapping[str, Any]]:
    """Expose a StepDefinition path provider to direct Consist resolution."""

    def resolve(context: Any) -> Mapping[str, Any]:
        settings = context.get_runtime("settings", default=None)
        state = context.get_runtime("state", default=None)
        workspace = context.get_runtime("workspace", default=None)
        if settings is None or state is None or workspace is None:
            return {}
        return provider(settings=settings, state=state, workspace=workspace)

    return resolve


def _urbansim_preprocess_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    return UrbansimPreprocessor.expected_outputs(settings, state, workspace)


def _urbansim_run_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    return UrbansimRunner.expected_outputs(settings, state, workspace)


def _urbansim_postprocess_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    paths = dict(UrbansimPostprocessor.expected_outputs(settings, state, workspace))
    forecast_year = state.forecast_year
    mutable_data_dir = Path(workspace.get_usim_mutable_data_dir())
    input_name = settings.urbansim.input_file_template.format(
        region_id=settings.urbansim.region_mappings["region_to_region_id"][
            settings.run.region
        ]
    )
    merged_path = mutable_data_dir / input_name
    paths[f"{USIM_INPUT_MERGED_PREFIX}{forecast_year}"] = merged_path
    paths[f"{USIM_INPUT_ARCHIVE_PREFIX}{forecast_year}"] = mutable_data_dir / (
        f"input_data_for_{forecast_year}_outputs.h5"
    )
    return paths


def _atlas_preprocess_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    atlas_input_dir = Path(workspace.get_atlas_mutable_input_dir())
    year_input_dir = atlas_input_dir / f"year{state.forecast_year}"
    paths: Dict[str, Any] = {
        "atlas_mutable_input_dir": atlas_input_dir,
        "atlas_households_csv": year_input_dir / "households.csv",
        "atlas_blocks_csv": year_input_dir / "blocks.csv",
        "atlas_persons_csv": year_input_dir / "persons.csv",
        "atlas_residential_csv": year_input_dir / "residential_units.csv",
        "atlas_jobs_csv": year_input_dir / "jobs.csv",
    }
    if state.year > state.start_year:
        paths["atlas_grave_csv"] = year_input_dir / "grave.csv"
    return paths


def _atlas_run_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    output_dir = Path(workspace.get_atlas_output_dir())
    forecast_year = state.forecast_year
    return {
        ATLAS_OUTPUT_DIR: output_dir,
        f"householdv_{forecast_year}": output_dir / f"householdv_{forecast_year}.csv",
        f"vehicles_{forecast_year}": output_dir / f"vehicles_{forecast_year}.csv",
    }


def _atlas_postprocess_native_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> Dict[str, Any]:
    return AtlasPostprocessor.expected_outputs(settings, state, workspace)


def _resolve_urbansim_preprocess_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    output_paths = _urbansim_preprocess_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    destinations = {
        USIM_DATASTORE_H5: Path(output_paths[USIM_DATASTORE_H5]),
        FINAL_SKIMS_OMX: Path(workspace.get_usim_mutable_data_dir())
        / "final_skims.omx",
    }
    return _resolve_native_inputs(
        step_name="urbansim_preprocess",
        required_roles=(USIM_DATASTORE_H5,),
        optional_roles=(FINAL_SKIMS_OMX,),
        logical_destinations=destinations,
        coupler=coupler,
    )


def _resolve_urbansim_run_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    del settings, state
    return _resolve_native_inputs(
        step_name="urbansim_run",
        required_roles=(USIM_MUTABLE_DATA_DIR,),
        optional_roles=(),
        logical_destinations={
            USIM_MUTABLE_DATA_DIR: Path(workspace.get_usim_mutable_data_dir())
        },
        coupler=coupler,
    )


def _resolve_urbansim_postprocess_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    return _resolve_native_inputs(
        step_name="urbansim_postprocess",
        required_roles=(USIM_DATASTORE_H5,),
        optional_roles=(),
        logical_destinations={
            USIM_DATASTORE_H5: Path(
                _urbansim_run_native_output_paths(
                    settings=settings,
                    state=state,
                    workspace=workspace,
                )[USIM_DATASTORE_H5]
            )
        },
        coupler=coupler,
    )


def _resolve_atlas_preprocess_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    atlas_input = AtlasPreprocessor.expected_inputs(settings, state, workspace)[
        USIM_DATASTORE_H5
    ]
    if atlas_input is None:
        atlas_input = _urbansim_run_native_output_paths(
            settings=settings, state=state, workspace=workspace
        )[USIM_DATASTORE_H5]
    return _resolve_native_inputs(
        step_name="atlas_preprocess",
        required_roles=(USIM_DATASTORE_H5,),
        optional_roles=(),
        logical_destinations={USIM_DATASTORE_H5: Path(atlas_input)},
        coupler=coupler,
    )


def _resolve_atlas_run_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    del settings, state
    return _resolve_native_inputs(
        step_name="atlas_run",
        required_roles=("atlas_mutable_input_dir",),
        optional_roles=(),
        logical_destinations={
            "atlas_mutable_input_dir": Path(workspace.get_atlas_mutable_input_dir())
        },
        coupler=coupler,
    )


def _resolve_atlas_postprocess_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> ResolvedStepInputs:
    return _resolve_native_inputs(
        step_name="atlas_postprocess",
        required_roles=(ATLAS_OUTPUT_DIR, USIM_DATASTORE_H5),
        optional_roles=(),
        logical_destinations={
            ATLAS_OUTPUT_DIR: Path(workspace.get_atlas_output_dir()),
            USIM_DATASTORE_H5: Path(
                _urbansim_run_native_output_paths(
                    settings=settings,
                    state=state,
                    workspace=workspace,
                )[USIM_DATASTORE_H5]
            ),
        },
        coupler=coupler,
    )


@define_step(
    model="urbansim_preprocess",
    name_template="urbansim_preprocess__y{year}__i{iteration}__phase_{phase}",
    output_paths=_native_contract_output_paths(
        _urbansim_preprocess_native_output_paths
    ),
    input_binding="paths",
    **consist_step_meta("urbansim_preprocess"),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_urbansim_preprocess(
    usim_datastore_h5: Path,
    final_skims_omx: Path | None = None,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    del usim_datastore_h5, settings
    UrbansimPreprocessor("urbansim", state).preprocess(
        workspace,
        final_skims_omx=final_skims_omx,
    )


@define_step(
    model="urbansim_run",
    name_template="urbansim_run__y{year}__i{iteration}__phase_{phase}",
    output_paths=_native_contract_output_paths(_urbansim_run_native_output_paths),
    input_binding="paths",
    **consist_step_meta("urbansim_run"),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_urbansim_run(
    usim_mutable_data_dir: Path,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    del settings
    UrbansimRunner("urbansim", state).run(
        NativeUrbanSimPreprocessOutputs(usim_mutable_data_dir=usim_mutable_data_dir),
        workspace,
    )


@define_step(
    model="urbansim_postprocess",
    name_template="urbansim_postprocess__y{year}__i{iteration}__phase_{phase}",
    output_paths=_native_contract_output_paths(
        _urbansim_postprocess_native_output_paths
    ),
    input_binding="paths",
    **consist_step_meta("urbansim_postprocess"),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_urbansim_postprocess(
    usim_datastore_h5: Path,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    del settings
    UrbansimPostprocessor("urbansim", state).postprocess(
        NativeUrbanSimRunOutputs(usim_datastore_h5=usim_datastore_h5),
        workspace,
    )


@define_step(
    model="atlas_preprocess",
    name_template="atlas_preprocess__y{year}__i{iteration}__phase_{phase}",
    output_paths=_native_contract_output_paths(_atlas_preprocess_native_output_paths),
    input_binding="paths",
    **consist_step_meta("atlas_preprocess"),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_atlas_preprocess(
    usim_datastore_h5: Path,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    del usim_datastore_h5, settings
    AtlasPreprocessor("atlas", state).preprocess(workspace)


@define_step(
    model="atlas_run",
    name_template="atlas_run__y{year}__i{iteration}__phase_{phase}",
    output_paths=_native_contract_output_paths(_atlas_run_native_output_paths),
    input_binding="paths",
    **consist_step_meta("atlas_run"),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_atlas_run(
    atlas_mutable_input_dir: Path,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    del settings
    AtlasRunner("atlas", state).run(
        AtlasPreprocessOutputs(atlas_mutable_input_dir=atlas_mutable_input_dir),
        workspace,
    )


@define_step(
    model="atlas_postprocess",
    name_template="atlas_postprocess__y{year}__i{iteration}__phase_{phase}",
    output_paths=_native_contract_output_paths(_atlas_postprocess_native_output_paths),
    input_binding="paths",
    **consist_step_meta("atlas_postprocess"),
)
@require_runtime_kwargs("settings", "state", "workspace")
def _native_atlas_postprocess(
    atlas_output_dir: Path,
    usim_datastore_h5: Path,
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> None:
    del settings, usim_datastore_h5
    forecast_year = state.forecast_year
    AtlasPostprocessor("atlas", state).postprocess(
        AtlasRunOutputs(
            atlas_output_dir=atlas_output_dir,
            raw_outputs={
                f"householdv_{forecast_year}": atlas_output_dir
                / f"householdv_{forecast_year}.csv",
                f"vehicles_{forecast_year}": atlas_output_dir
                / f"vehicles_{forecast_year}.csv",
            },
        ),
        workspace,
    )


def _project_urbansim_preprocess(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> NativeUrbanSimPreprocessOutputs:
    declared_outputs = _urbansim_preprocess_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    prepared_inputs = {
        key: _persisted_output_path(
            outputs,
            step_name="urbansim_preprocess",
            key=key,
            declared_outputs=declared_outputs,
            workspace=workspace,
        )
        for key in outputs
        if key != USIM_MUTABLE_DATA_DIR
    }
    projected = NativeUrbanSimPreprocessOutputs(
        usim_mutable_data_dir=_persisted_output_path(
            outputs,
            step_name="urbansim_preprocess",
            key=USIM_MUTABLE_DATA_DIR,
            declared_outputs=declared_outputs,
            workspace=workspace,
        ),
        prepared_inputs=prepared_inputs,
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="urbansim_preprocess",
        )
    )
    return projected


def _project_urbansim_run(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> NativeUrbanSimRunOutputs:
    declared_outputs = _urbansim_run_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    datastore = _persisted_output_path(
        outputs,
        step_name="urbansim_run",
        key=USIM_DATASTORE_H5,
        declared_outputs=declared_outputs,
        workspace=workspace,
    )
    projected = NativeUrbanSimRunOutputs(
        usim_datastore_h5=datastore,
        raw_outputs={USIM_FORECAST_OUTPUT: datastore},
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="urbansim_run",
        )
    )
    return projected


def _project_urbansim_postprocess(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> NativeUrbanSimPostprocessOutputs:
    declared_outputs = _urbansim_postprocess_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    datastore = _persisted_output_path(
        outputs,
        step_name="urbansim_postprocess",
        key=USIM_DATASTORE_H5,
        declared_outputs=declared_outputs,
        workspace=workspace,
    )
    processed_outputs = {
        key: _persisted_output_path(
            outputs,
            step_name="urbansim_postprocess",
            key=key,
            declared_outputs=declared_outputs,
            workspace=workspace,
        )
        for key in outputs
        if key != USIM_DATASTORE_H5
    }
    projected = NativeUrbanSimPostprocessOutputs(
        usim_datastore_h5=datastore,
        processed_outputs=processed_outputs,
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="urbansim_postprocess",
        )
    )
    return projected


def _project_atlas_preprocess(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> AtlasPreprocessOutputs:
    declared_outputs = _atlas_preprocess_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    prepared_inputs = {
        key: _persisted_output_path(
            outputs,
            step_name="atlas_preprocess",
            key=key,
            declared_outputs=declared_outputs,
            workspace=workspace,
        )
        for key in outputs
        if key != "atlas_mutable_input_dir"
    }
    projected = AtlasPreprocessOutputs(
        atlas_mutable_input_dir=_persisted_output_path(
            outputs,
            step_name="atlas_preprocess",
            key="atlas_mutable_input_dir",
            declared_outputs=declared_outputs,
            workspace=workspace,
        ),
        prepared_inputs=prepared_inputs,
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="atlas_preprocess",
        )
    )
    return projected


def _project_atlas_run(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> AtlasRunOutputs:
    declared_outputs = _atlas_run_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    projected = AtlasRunOutputs(
        atlas_output_dir=_persisted_output_path(
            outputs,
            step_name="atlas_run",
            key=ATLAS_OUTPUT_DIR,
            declared_outputs=declared_outputs,
            workspace=workspace,
        ),
        raw_outputs={
            key: _persisted_output_path(
                outputs,
                step_name="atlas_run",
                key=key,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            for key in outputs
            if key != ATLAS_OUTPUT_DIR
        },
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="atlas_run",
        )
    )
    return projected


def _project_atlas_postprocess(
    outputs: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> AtlasPostprocessOutputs:
    declared_outputs = _atlas_postprocess_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    projected = AtlasPostprocessOutputs(
        atlas_output_dir=_persisted_output_path(
            outputs,
            step_name="atlas_postprocess",
            key=ATLAS_OUTPUT_DIR,
            declared_outputs=declared_outputs,
            workspace=workspace,
        ),
        usim_datastore_h5=_persisted_output_path(
            outputs,
            step_name="atlas_postprocess",
            key=USIM_POPULATION_SOURCE_H5,
            declared_outputs=declared_outputs,
            workspace=workspace,
        ),
        processed_outputs={
            ATLAS_VEHICLES2_OUTPUT: _persisted_output_path(
                outputs,
                step_name="atlas_postprocess",
                key=ATLAS_VEHICLES2_OUTPUT,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
        },
    )
    projected.validate(
        _validation_context(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name="atlas_postprocess",
        )
    )
    return projected


URBANSIM_PREPROCESS = StepDefinition(
    name="urbansim_preprocess",
    function=_native_urbansim_preprocess,
    resolve_inputs=_resolve_urbansim_preprocess_inputs,
    project_outputs=_project_urbansim_preprocess,
    output_paths=_urbansim_preprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
)

URBANSIM_RUN = StepDefinition(
    name="urbansim_run",
    function=_native_urbansim_run,
    resolve_inputs=_resolve_urbansim_run_inputs,
    project_outputs=_project_urbansim_run,
    output_paths=_urbansim_run_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
)

URBANSIM_POSTPROCESS = StepDefinition(
    name="urbansim_postprocess",
    function=_native_urbansim_postprocess,
    resolve_inputs=_resolve_urbansim_postprocess_inputs,
    project_outputs=_project_urbansim_postprocess,
    output_paths=_urbansim_postprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
)

ATLAS_PREPROCESS = StepDefinition(
    name="atlas_preprocess",
    function=_native_atlas_preprocess,
    resolve_inputs=_resolve_atlas_preprocess_inputs,
    project_outputs=_project_atlas_preprocess,
    output_paths=_atlas_preprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
)

ATLAS_RUN = StepDefinition(
    name="atlas_run",
    function=_native_atlas_run,
    resolve_inputs=_resolve_atlas_run_inputs,
    project_outputs=_project_atlas_run,
    output_paths=_atlas_run_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
)

ATLAS_POSTPROCESS = StepDefinition(
    name="atlas_postprocess",
    function=_native_atlas_postprocess,
    resolve_inputs=_resolve_atlas_postprocess_inputs,
    project_outputs=_project_atlas_postprocess,
    output_paths=_atlas_postprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache_options,
)

# Canonical registry members use the workflow step names.  Uppercase aliases keep
# the six definitions convenient to import in focused model tests.
urbansim_preprocess = URBANSIM_PREPROCESS
urbansim_run = URBANSIM_RUN
urbansim_postprocess = URBANSIM_POSTPROCESS
atlas_preprocess = ATLAS_PREPROCESS
atlas_run = ATLAS_RUN
atlas_postprocess = ATLAS_POSTPROCESS
