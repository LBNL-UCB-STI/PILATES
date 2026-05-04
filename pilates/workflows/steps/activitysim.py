from __future__ import annotations

import inspect
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, cast

from consist.types import H5ChildSpec

from pilates.activitysim.preprocessor import ActivitysimPreprocessor
from pilates.activitysim.runner import (
    ActivitysimCompileRunner,
    ActivitysimRunner,
    asim_runtime_zarr_path,
    asim_sharrow_cache_dir,
    persist_sharrow_cache_enabled,
)
from pilates.activitysim.outputs import has_asim_run_marker, normalize_asim_output_key
from pilates.activitysim.outputs import ASIM_REQUIRED_RUN_OUTPUT_KEYS
from pilates.activitysim.postprocessor import ActivitysimPostprocessor
from pilates.utils.coupler_helpers import (
    artifact_to_existing_path,
    resolve_existing_path,
)
from pilates.utils.settings_helper import get as get_setting
from pilates.utils.usim_h5 import reconcile_usim_population_table_paths
from pilates.config.models import PilatesConfig
from pilates.generic.model_factory import ModelFactory
from pilates.utils.consist_runtime import artifact_fingerprint
from pilates.workflows.artifact_keys import (
    ASIM_SHARROW_CACHE_DIR,
    USIM_POPULATION_BLOCKS_TABLE,
    USIM_POPULATION_HOUSEHOLDS_TABLE,
    USIM_POPULATION_JOBS_TABLE,
    USIM_POPULATION_PERSONS_TABLE,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.binding import build_binding_plan
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
    ZARR_SKIMS,
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
    CouplerProtocol,
    StandardStepSpec,
    StepOutputsHolder,
    WorkflowState,
    _activitysim_output_facet_meta,
    _decorate_step_with_consist,
    build_standard_step,
    load_recovered_cached_outputs,
    _log_step_records,
    artifact_to_path,
    cr,
    log_and_set_input,
    log_and_set_output,
    log_input_only,
    log_output_only,
    resolve_artifact_from_value,
)
from pilates.workflows.input_resolution import (
    resolved_metadata_value_for_key,
    resolved_value_for_key,
)
from pilates.workflows.tracker_outputs import merge_canonical_output_mappings

logger = logging.getLogger(__name__)

_ACTIVITYSIM_POPULATION_TABLE_KEYS = (
    USIM_POPULATION_HOUSEHOLDS_TABLE,
    USIM_POPULATION_PERSONS_TABLE,
    USIM_POPULATION_JOBS_TABLE,
    USIM_POPULATION_BLOCKS_TABLE,
)


def _canonical_activitysim_run_output_key(short_name: str) -> str:
    clean_name = re.sub(r"_asim_out_temp$", "", short_name)
    return normalize_asim_output_key(clean_name)


def _resolve_activitysim_run_cached_value(
    *,
    key: str,
    coupler: CouplerProtocol,
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Any:
    value = _resolve_cached_value(
        key=key,
        coupler=coupler,
        cached_outputs=cached_outputs,
        run_id=run_id,
    )
    if value is not None:
        return value
    canonical_key = _canonical_activitysim_run_output_key(key)
    if canonical_key == key:
        return None
    return _resolve_cached_value(
        key=canonical_key,
        coupler=coupler,
        cached_outputs=cached_outputs,
        run_id=run_id,
    )


def _strip_component_runtime_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    filtered = dict(kwargs)
    filtered.pop("coupler", None)
    filtered.pop("context", None)
    return filtered


def _filter_kwargs_for_callable(
    func: Callable[..., Any],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    ):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _execute_activitysim_preprocess(
    preprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> ActivitySimPreprocessOutputs:
    runtime_kwargs = _strip_component_runtime_kwargs(kwargs)
    if "population_source_h5_path" not in runtime_kwargs:
        legacy_population_source = (
            runtime_kwargs.get("usim_population_source_h5")
            or runtime_kwargs.get("usim_datastore_h5")
            or runtime_kwargs.get("usim_datastore_base_h5")
        )
        if legacy_population_source is not None:
            runtime_kwargs["population_source_h5_path"] = legacy_population_source
    filtered_kwargs = {
        key: value
        for key, value in runtime_kwargs.items()
        if key
        in {
            "final_skims_omx",
            "population_source_h5_path",
            USIM_POPULATION_HOUSEHOLDS_TABLE,
            USIM_POPULATION_PERSONS_TABLE,
            USIM_POPULATION_JOBS_TABLE,
            USIM_POPULATION_BLOCKS_TABLE,
        }
    }
    callable_kwargs = _filter_kwargs_for_callable(
        preprocessor.preprocess, filtered_kwargs
    )
    if (
        "population_source_h5_path" in filtered_kwargs
        and "population_source_h5_path" not in callable_kwargs
    ):
        legacy_population_kwargs = {
            key: value
            for key, value in (
                ("usim_datastore_h5", filtered_kwargs["population_source_h5_path"]),
                (
                    "usim_datastore_base_h5",
                    filtered_kwargs["population_source_h5_path"],
                ),
            )
            if key in inspect.signature(preprocessor.preprocess).parameters
        }
        callable_kwargs.update(legacy_population_kwargs)
    return preprocessor.preprocess(
        workspace,
        **callable_kwargs,
    )


def _execute_activitysim_run(
    runner: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    *,
    extra_inputs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> ActivitySimRunOutputs:
    upstream = outputs_holder.activitysim_preprocess
    if upstream is None:
        raise RuntimeError("ActivitySim preprocess must complete first")
    if not isinstance(upstream, ActivitySimPreprocessOutputs):
        raise TypeError(
            "activitysim_run requires ActivitySimPreprocessOutputs from activitysim_preprocess"
        )
    return runner.run(upstream, workspace, extra_inputs=extra_inputs)


def _execute_activitysim_postprocess(
    postprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> ActivitySimPostprocessOutputs:
    upstream = outputs_holder.activitysim_run
    if upstream is None:
        raise RuntimeError("ActivitySim run must complete first")
    if not isinstance(upstream, ActivitySimRunOutputs):
        raise TypeError(
            "activitysim_postprocess requires ActivitySimRunOutputs from activitysim_run"
        )
    runtime_kwargs = _strip_component_runtime_kwargs(kwargs)
    runtime_kwargs.setdefault(
        "population_source_h5_path", runtime_kwargs.get("usim_population_source_h5")
    )
    runtime_kwargs.setdefault(
        "current_input_h5_path",
        runtime_kwargs.get("usim_datastore_h5")
        or runtime_kwargs.get("usim_datastore_base_h5"),
    )
    filtered_kwargs = {
        key: value
        for key, value in runtime_kwargs.items()
        if key
        in {
            "population_source_h5_path",
            "current_input_h5_path",
        }
    }
    callable_kwargs = _filter_kwargs_for_callable(
        postprocessor.postprocess, filtered_kwargs
    )
    legacy_postprocess_kwargs: Dict[str, Any] = {}
    if (
        filtered_kwargs.get("current_input_h5_path") is not None
        and "current_input_h5_path" not in callable_kwargs
        and "usim_datastore_h5"
        in inspect.signature(postprocessor.postprocess).parameters
    ):
        legacy_postprocess_kwargs["usim_datastore_h5"] = filtered_kwargs[
            "current_input_h5_path"
        ]
    callable_kwargs.update(legacy_postprocess_kwargs)
    return postprocessor.postprocess(
        upstream,
        workspace,
        **callable_kwargs,
    )


def _resolve_cached_run_outputs(run_id: Optional[str]) -> Dict[str, Any]:
    return load_recovered_cached_outputs(
        run_id,
        step_logger=logger,
        log_context="ActivitySim cached output recovery",
    )


def _resolve_cached_value(
    *,
    key: str,
    coupler: CouplerProtocol,
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Any:
    merged = merge_canonical_output_mappings(
        cached_outputs,
        _resolve_cached_run_outputs(run_id),
    )
    if key in merged:
        return merged[key]
    get_value = getattr(coupler, "get", None)
    if callable(get_value):
        return get_value(key)
    return None


def _existing_artifact_path(value: Any, workspace: Workspace) -> Optional[str]:
    return artifact_to_existing_path(
        value,
        workspace=workspace,
        materialize_from_archive=True,
    )


def _existing_local_path(path: Any, workspace: Workspace) -> Optional[str]:
    if path is None:
        return None
    return resolve_existing_path(
        str(path),
        workspace=workspace,
        materialize_from_archive=True,
    )


def _resolve_activitysim_h5_runtime_path(
    *,
    value: Any,
    key: str,
    workspace: Workspace,
    required: bool = True,
) -> Optional[str]:
    if value is None:
        if required:
            raise ValueError(
                f"ActivitySim step requires bound input '{key}', but no value was provided."
            )
        return None
    resolved_path = artifact_to_existing_path(
        value,
        workspace=workspace,
        materialize_from_archive=True,
    )
    if resolved_path:
        return resolved_path
    candidate_path = artifact_to_path(
        resolve_artifact_from_value(value, key=key, workspace=workspace),
        workspace,
    )
    if candidate_path and os.path.exists(candidate_path):
        return candidate_path
    if required:
        raise FileNotFoundError(
            f"ActivitySim step received '{key}' but it did not resolve to an existing file: {value}"
        )
    return None


def _resolve_activitysim_preprocess_runtime_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    step_inputs: Optional[Mapping[str, Any]] = None,
    surface: Optional["EnabledWorkflowSurface"] = None,
) -> Dict[str, Any]:
    def _requires_exact_population_year() -> bool:
        land_use_enabled = get_setting(settings, "run.models.land_use") is not None
        is_start_year = getattr(state, "is_start_year", None)
        if not land_use_enabled or not callable(is_start_year):
            return False
        try:
            return not bool(is_start_year())
        except Exception:
            return False

    if step_inputs and USIM_POPULATION_SOURCE_H5 in step_inputs:
        population_source_value = step_inputs[USIM_POPULATION_SOURCE_H5]
        resolution = None
    else:
        step_year = getattr(state, "forecast_year", None)
        if step_year is None:
            step_year = getattr(state, "year", None)
        resolution = build_binding_plan(
            step_name="activitysim_preprocess",
            coupler=coupler,
            settings=settings,
            state=state,
            workspace=workspace,
            year=step_year,
            surface=surface,
        )
        population_source_value = resolved_value_for_key(
            resolved=resolution,
            key=USIM_POPULATION_SOURCE_H5,
            coupler=coupler,
        )
    population_source_h5_path = _resolve_activitysim_h5_runtime_path(
        value=population_source_value,
        key=USIM_POPULATION_SOURCE_H5,
        workspace=workspace,
    )
    runtime_inputs: Dict[str, Any] = {
        "population_source_h5_path": population_source_h5_path,
    }
    if step_inputs:
        for table_key in _ACTIVITYSIM_POPULATION_TABLE_KEYS:
            table_value = step_inputs.get(table_key)
            if isinstance(table_value, str) and table_value:
                runtime_inputs[table_key] = table_value
    if resolution is not None:
        for table_key in _ACTIVITYSIM_POPULATION_TABLE_KEYS:
            table_value = resolved_metadata_value_for_key(
                resolved=resolution,
                key=table_key,
            )
            if isinstance(table_value, str) and table_value:
                runtime_inputs.setdefault(table_key, table_value)

    if population_source_h5_path:
        target_year = getattr(state, "forecast_year", None)
        if target_year is None:
            target_year = getattr(state, "year", None)
        try:
            provided_table_paths = {
                table_key: runtime_inputs[table_key]
                for table_key in _ACTIVITYSIM_POPULATION_TABLE_KEYS
                if isinstance(runtime_inputs.get(table_key), str)
                and runtime_inputs.get(table_key)
            }
            resolved_table_paths = reconcile_usim_population_table_paths(
                h5_path=population_source_h5_path,
                year=target_year,
                provided_paths=provided_table_paths,
                require_exact_year=_requires_exact_population_year(),
            )
        except Exception as exc:
            if _requires_exact_population_year():
                raise
            logger.debug(
                "Skipping ActivitySim population table resolution for %s: %s",
                population_source_h5_path,
                exc,
            )
        else:
            for table_key, table_path in resolved_table_paths.items():
                current_value = runtime_inputs.get(table_key)
                if current_value != table_path:
                    if isinstance(current_value, str) and current_value:
                        logger.debug(
                            "Reconciled ActivitySim population table selector %s: %s -> %s",
                            table_key,
                            current_value,
                            table_path,
                        )
                    runtime_inputs[table_key] = table_path
    return runtime_inputs


def _resolve_activitysim_postprocess_runtime_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    step_inputs: Optional[Mapping[str, Any]] = None,
    surface: Optional["EnabledWorkflowSurface"] = None,
) -> Dict[str, Optional[str]]:
    runtime_inputs: Dict[str, Optional[str]] = {
        "population_source_h5_path": None,
        "current_input_h5_path": None,
    }
    if not state.is_enabled(WorkflowState.Stage.land_use):
        return runtime_inputs

    population_resolution = None
    current_resolution = None
    if not step_inputs or USIM_POPULATION_SOURCE_H5 not in step_inputs:
        population_resolution = build_binding_plan(
            step_name="activitysim_postprocess",
            coupler=coupler,
            settings=settings,
            state=state,
            workspace=workspace,
            year=state.forecast_year,
            surface=surface,
        )
    if not step_inputs or USIM_DATASTORE_CURRENT_H5 not in step_inputs:
        current_year = getattr(state, "year", getattr(state, "current_year", None))
        current_resolution = build_binding_plan(
            step_name="activitysim_postprocess",
            coupler=coupler,
            settings=settings,
            state=state,
            workspace=workspace,
            year=current_year,
            surface=surface,
        )
    population_source_value = (
        step_inputs.get(USIM_POPULATION_SOURCE_H5)
        if step_inputs and USIM_POPULATION_SOURCE_H5 in step_inputs
        else resolved_value_for_key(
            resolved=population_resolution,
            key=USIM_POPULATION_SOURCE_H5,
            coupler=coupler,
        )
    )
    current_input_value = (
        step_inputs.get(USIM_DATASTORE_CURRENT_H5)
        if step_inputs and USIM_DATASTORE_CURRENT_H5 in step_inputs
        else resolved_value_for_key(
            resolved=current_resolution,
            key=USIM_DATASTORE_CURRENT_H5,
            coupler=coupler,
        )
    )
    if population_source_value is None:
        population_source_value = current_input_value
    if current_input_value is None:
        current_input_value = population_source_value

    runtime_inputs["population_source_h5_path"] = _resolve_activitysim_h5_runtime_path(
        value=population_source_value,
        key=USIM_POPULATION_SOURCE_H5,
        workspace=workspace,
    )
    runtime_inputs["current_input_h5_path"] = _resolve_activitysim_h5_runtime_path(
        value=current_input_value,
        key=USIM_DATASTORE_CURRENT_H5,
        workspace=workspace,
    )
    return runtime_inputs


def _resolved_content_hash(
    *,
    value: Any,
    key: str,
    workspace: Workspace,
    fallback_path: Any = None,
) -> Optional[str]:
    candidate = value if value is not None else fallback_path
    artifact = resolve_artifact_from_value(
        candidate,
        key=key,
        workspace=workspace,
    )
    return artifact_fingerprint(artifact)


def _recover_activitysim_preprocess_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[ActivitySimPreprocessOutputs]:
    del outputs_holder, step_inputs
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    candidates = {
        key: Path(path)
        for key, path in ActivitysimPreprocessor.expected_outputs(
            settings, state, workspace
        ).items()
        if key
        in {ASIM_HOUSEHOLDS_IN, ASIM_PERSONS_IN, ASIM_LAND_USE_IN, ASIM_OMX_SKIMS}
        and isinstance(path, str)
    }
    recovered_inputs: Dict[str, Path] = {}
    recovered_hashes: Dict[str, str] = {}
    for key, path in candidates.items():
        cached_value = _resolve_cached_value(
            key=key,
            coupler=coupler,
            cached_outputs=cached_outputs,
            run_id=run_id,
        )
        cached_path = _existing_artifact_path(cached_value, workspace)
        if cached_path:
            path = Path(cached_path)
        resolved_candidate = _existing_local_path(path, workspace)
        if resolved_candidate:
            recovered_inputs[key] = Path(resolved_candidate)
            content_hash = _resolved_content_hash(
                value=cached_value,
                key=key,
                workspace=workspace,
                fallback_path=resolved_candidate,
            )
            if content_hash:
                recovered_hashes[key] = content_hash
    if not {
        ASIM_LAND_USE_IN,
        ASIM_HOUSEHOLDS_IN,
        ASIM_PERSONS_IN,
    }.issubset(recovered_inputs):
        return None
    return ActivitySimPreprocessOutputs(
        mutable_data_dir=asim_dir,
        land_use_table=recovered_inputs[ASIM_LAND_USE_IN],
        households_table=recovered_inputs[ASIM_HOUSEHOLDS_IN],
        persons_table=recovered_inputs[ASIM_PERSONS_IN],
        omx_skims=recovered_inputs.get(ASIM_OMX_SKIMS),
        input_hashes=recovered_hashes,
    )


def _recover_activitysim_run_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[ActivitySimRunOutputs]:
    del step_inputs
    asim_output_dir = Path(workspace.get_asim_output_dir())
    raw_outputs: Dict[str, Path] = {}
    raw_output_hashes: Dict[str, str] = {}
    source_input_paths: Dict[str, Path] = {}
    source_input_hashes: Dict[str, str] = {}

    final_pipeline = Path(
        _existing_local_path(asim_output_dir / "final_pipeline", workspace)
        or (asim_output_dir / "final_pipeline")
    )
    allow_final_pipeline = has_asim_run_marker(
        asim_output_dir, state.year, state.iteration
    )
    if final_pipeline.exists() and allow_final_pipeline:
        for child in final_pipeline.iterdir():
            fpath = child / "final.parquet"
            if fpath.is_file():
                short_name = f"{child.name}_asim_out_temp"
                raw_outputs[short_name] = fpath
                content_hash = _resolved_content_hash(
                    value=_resolve_activitysim_run_cached_value(
                        key=short_name,
                        coupler=coupler,
                        cached_outputs=cached_outputs,
                        run_id=run_id,
                    ),
                    key=short_name,
                    workspace=workspace,
                    fallback_path=fpath,
                )
                if content_hash:
                    raw_output_hashes[short_name] = content_hash
    elif final_pipeline.exists() and not allow_final_pipeline:
        logger.warning(
            "Skipping ActivitySim final_pipeline cache recovery for year %s iteration %s because success marker is missing.",
            state.year,
            state.iteration,
        )

    if not raw_outputs:
        iter_dir = Path(
            _existing_local_path(
                asim_output_dir / f"year-{state.year}-iteration-{state.iteration}",
                workspace,
            )
            or (asim_output_dir / f"year-{state.year}-iteration-{state.iteration}")
        )
        if iter_dir.exists():
            for fpath in iter_dir.glob("*.parquet"):
                short_name = f"{fpath.stem}_asim_out_temp"
                raw_outputs[short_name] = fpath
                content_hash = _resolved_content_hash(
                    value=_resolve_activitysim_run_cached_value(
                        key=short_name,
                        coupler=coupler,
                        cached_outputs=cached_outputs,
                        run_id=run_id,
                    ),
                    key=short_name,
                    workspace=workspace,
                    fallback_path=fpath,
                )
                if content_hash:
                    raw_output_hashes[short_name] = content_hash
    if not raw_outputs:
        return None

    upstream_preprocess = outputs_holder.activitysim_preprocess
    if upstream_preprocess is not None:
        carried_hashes = getattr(upstream_preprocess, "input_hashes", {}) or {}
        for short_name, path, _description in upstream_preprocess._iter_record_items():
            source_input_paths[short_name] = Path(path)
            content_hash = carried_hashes.get(short_name)
            if content_hash:
                source_input_hashes[short_name] = str(content_hash)

    runner_expected_inputs = ActivitysimRunner.expected_inputs(
        settings, state, workspace
    )
    runtime_zarr_candidate = Path(
        cast(
            str,
            runner_expected_inputs.get(ZARR_SKIMS) or asim_runtime_zarr_path(workspace),
        )
    )
    zarr_candidate = Path(
        _existing_local_path(runtime_zarr_candidate, workspace)
        or runtime_zarr_candidate
    )
    if zarr_candidate.exists():
        source_input_paths[ZARR_SKIMS] = zarr_candidate
        content_hash = _resolved_content_hash(
            value=_resolve_cached_value(
                key=ZARR_SKIMS,
                coupler=coupler,
                cached_outputs=cached_outputs,
                run_id=run_id,
            ),
            key=ZARR_SKIMS,
            workspace=workspace,
            fallback_path=zarr_candidate,
        )
        if content_hash:
            source_input_hashes[ZARR_SKIMS] = content_hash

    return ActivitySimRunOutputs(
        output_dir=asim_output_dir,
        raw_outputs=raw_outputs,
        raw_output_hashes=raw_output_hashes,
        source_input_paths=source_input_paths,
        source_input_hashes=source_input_hashes,
    )


def _recover_activitysim_postprocess_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[ActivitySimPostprocessOutputs]:
    del outputs_holder
    asim_output_dir = Path(workspace.get_asim_output_dir())
    iter_dir = Path(
        _existing_local_path(
            asim_output_dir / f"year-{state.year}-iteration-{state.iteration}",
            workspace,
        )
        or (asim_output_dir / f"year-{state.year}-iteration-{state.iteration}")
    )
    processed_outputs: Dict[str, Path] = {}
    processed_output_hashes: Dict[str, str] = {}
    if iter_dir.exists():
        required_outputs = {
            "persons_asim_out",
            "households_asim_out",
            "beam_plans_asim_out",
        }
        available_outputs = {
            normalize_asim_output_key(path.stem) for path in iter_dir.glob("*.parquet")
        }
        if not required_outputs.issubset(available_outputs):
            return None
        for fpath in iter_dir.glob("*.parquet"):
            short_name = normalize_asim_output_key(fpath.stem)
            processed_outputs[short_name] = fpath
            content_hash = _resolved_content_hash(
                value=_resolve_cached_value(
                    key=short_name,
                    coupler=coupler,
                    cached_outputs=cached_outputs,
                    run_id=run_id,
                ),
                key=short_name,
                workspace=workspace,
                fallback_path=fpath,
            )
            if content_hash:
                processed_output_hashes[short_name] = content_hash
    else:
        return None

    archived_input_year = getattr(state, "forecast_year", None)
    if archived_input_year is None:
        archived_input_year = getattr(state, "year", None)
    inputs_dir = Path(
        _existing_local_path(
            asim_output_dir
            / f"inputs-year-{archived_input_year}-iteration-{state.iteration}",
            workspace,
        )
        or (
            asim_output_dir
            / f"inputs-year-{archived_input_year}-iteration-{state.iteration}"
        )
    )
    if inputs_dir.exists():
        archived_inputs = {
            "households.csv": "asim_input_households_csv_archived",
            "persons.csv": "asim_input_persons_csv_archived",
            "land_use.csv": "asim_input_land_use_csv_archived",
            "skims.omx": "asim_input_skims_omx_archived",
        }
        for fname, short_name in archived_inputs.items():
            fpath = inputs_dir / fname
            if fpath.exists():
                processed_outputs[short_name] = fpath
                content_hash = _resolved_content_hash(
                    value=_resolve_cached_value(
                        key=short_name,
                        coupler=coupler,
                        cached_outputs=cached_outputs,
                        run_id=run_id,
                    ),
                    key=short_name,
                    workspace=workspace,
                    fallback_path=fpath,
                )
                if content_hash:
                    processed_output_hashes[short_name] = content_hash
        zarr_path = inputs_dir / "skims.zarr"
        if zarr_path.exists():
            short_name = "asim_input_skims_zarr_archived"
            processed_outputs[short_name] = zarr_path
            content_hash = _resolved_content_hash(
                value=_resolve_cached_value(
                    key=short_name,
                    coupler=coupler,
                    cached_outputs=cached_outputs,
                    run_id=run_id,
                ),
                key=short_name,
                workspace=workspace,
                fallback_path=zarr_path,
            )
            if content_hash:
                processed_output_hashes[short_name] = content_hash

    usim_path = None
    land_use_enabled = False
    is_enabled = getattr(state, "is_enabled", None)
    if callable(is_enabled):
        try:
            land_use_enabled = bool(is_enabled(WorkflowState.Stage.land_use))
        except Exception:
            land_use_enabled = False
    if land_use_enabled:
        required_land_use_keys = (
            USIM_POPULATION_SOURCE_H5,
            USIM_DATASTORE_CURRENT_H5,
        )
        if not step_inputs or any(
            key not in step_inputs for key in required_land_use_keys
        ):
            return None
        usim_path = _existing_artifact_path(
            step_inputs[USIM_DATASTORE_CURRENT_H5], workspace
        )
    else:
        if step_inputs and USIM_DATASTORE_CURRENT_H5 in step_inputs:
            usim_path = _existing_artifact_path(
                step_inputs[USIM_DATASTORE_CURRENT_H5], workspace
            )
        if not usim_path and step_inputs and USIM_DATASTORE_BASE_H5 in step_inputs:
            usim_path = _existing_artifact_path(
                step_inputs[USIM_DATASTORE_BASE_H5], workspace
            )
    usim_existing = _existing_local_path(usim_path, workspace)
    if not processed_outputs and not usim_existing:
        return None
    return ActivitySimPostprocessOutputs(
        usim_datastore_h5=Path(usim_existing) if usim_existing else None,
        asim_output_dir=asim_output_dir,
        processed_outputs=processed_outputs,
        processed_output_hashes=processed_output_hashes,
        usim_datastore_key=USIM_DATASTORE_H5 if usim_existing else None,
    )


def _compile_step_schema_outputs(ctx: Any) -> list[str]:
    settings = ctx.require_runtime("settings")
    outputs = [ZARR_SKIMS]
    if settings is not None and persist_sharrow_cache_enabled(settings):
        outputs.append(ASIM_SHARROW_CACHE_DIR)
    return outputs


def activitysim_compile_output_paths(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
) -> Dict[str, Any]:
    return ActivitysimCompileRunner.expected_outputs(settings, state, workspace)


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

    def _run_activitysim_compile_step(
        *,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        expected_outputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        forecast_year = state.forecast_year
        if forecast_year is None:
            raise RuntimeError(
                "WorkflowState.forecast_year must be set before ActivitySim compile."
            )
        factory = ModelFactory()

        compile_runner = cast(
            ActivitysimCompileRunner,
            factory.get_runner(
                "activitysim_compile",
                state,
            ),
        )
        upstream = outputs_holder.activitysim_preprocess
        if upstream is None:
            raise RuntimeError(
                "ActivitySim compile must run after activitysim_preprocess"
            )
        if not isinstance(upstream, ActivitySimPreprocessOutputs):
            raise TypeError(
                "activitysim_compile requires ActivitySimPreprocessOutputs from activitysim_preprocess"
            )
        if expected_outputs is None:
            expected_outputs = activitysim_compile_output_paths(
                settings=settings,
                state=state,
                workspace=workspace,
            )
        omx_path = (
            artifact_to_path(upstream.omx_skims, workspace)
            if upstream.omx_skims is not None
            else None
        )
        if omx_path and os.path.exists(omx_path):
            cr.log_input(
                omx_path,
                key=upstream.record_keys["omx_skims"],
                description="ActivitySim compile input skims (OMX)",
            )
        compile_outputs = compile_runner.run(upstream, workspace)

        zarr_output_path = expected_outputs.get(ZARR_SKIMS)
        if not zarr_output_path and compile_outputs.zarr_skims is not None:
            zarr_output_path = str(compile_outputs.zarr_skims)
        if zarr_output_path and os.path.exists(zarr_output_path):
            log_and_set_output(
                key=ZARR_SKIMS,
                path=zarr_output_path,
                description="ActivitySim compiled zarr skims",
                coupler=coupler,
                step_name="activitysim_compile",
                **_activitysim_output_facet_meta(
                    ZARR_SKIMS,
                    year=forecast_year,
                    iteration=state.iteration,
                ),
            )

        if persist_sharrow_cache_enabled(settings):
            cache_path = (
                str(compile_outputs.sharrow_cache_dir)
                if compile_outputs.sharrow_cache_dir is not None
                else asim_sharrow_cache_dir(workspace)
            )

            if cache_path and _is_non_empty_directory(cache_path):
                log_and_set_output(
                    key=ASIM_SHARROW_CACHE_DIR,
                    path=cache_path,
                    description=(
                        "ActivitySim persisted compile cache directory (numba/sharrow)"
                    ),
                    coupler=coupler,
                    step_name="activitysim_compile",
                    **_activitysim_output_facet_meta(
                        ASIM_SHARROW_CACHE_DIR,
                        year=forecast_year,
                        iteration=state.iteration,
                    ),
                )
            elif (
                cache_path
                and os.path.exists(cache_path)
                and not os.path.isdir(cache_path)
            ):
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

    step_func = _decorate_step_with_consist(
        step_func=_run_activitysim_compile_step,
        step_model="activitysim_compile",
        description="activitysim compile workflow step",
        outputs=[ZARR_SKIMS],
        schema_outputs=_compile_step_schema_outputs,
        output_paths=activitysim_compile_output_paths,
        cache_mode="overwrite",
        cache_hydration="metadata",
        tags=["activitysim", "compile"],
    )
    return step_func


def make_activitysim_preprocess_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    surface: Optional["EnabledWorkflowSurface"] = None,
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
        step_inputs: Optional[Mapping[str, Any]] = None,
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
        runtime_inputs = _resolve_activitysim_preprocess_runtime_inputs(
            settings=settings,
            state=state,
            workspace=workspace,
            coupler=coupler,
            step_inputs=step_inputs,
            surface=surface,
        )
        usim_path = runtime_inputs["population_source_h5_path"]
        if usim_path and os.path.exists(usim_path):
            input_key = USIM_POPULATION_SOURCE_H5
            population_year = getattr(state, "forecast_year", None)
            if population_year is None:
                population_year = getattr(state, "year", None)
            input_desc = (
                "UrbanSim population-source datastore for ActivitySim "
                f"population year {population_year} "
                f"(workflow year {getattr(state, 'year', None)})"
            )
            table_config = (
                (USIM_POPULATION_HOUSEHOLDS_TABLE, "households"),
                (USIM_POPULATION_PERSONS_TABLE, "persons"),
                (USIM_POPULATION_JOBS_TABLE, "jobs"),
                (USIM_POPULATION_BLOCKS_TABLE, "blocks"),
            )
            h5_tables_used = []
            child_specs: Dict[str, H5ChildSpec] = {}
            start_year = getattr(state, "start_year", None)
            for table_key, table_name in table_config:
                table_path = runtime_inputs.get(table_key)
                if not isinstance(table_path, str) or not table_path:
                    continue
                h5_tables_used.append(table_path)
                key_suffix = (
                    "start_year_input"
                    if start_year is not None
                    and table_path.startswith(f"/{start_year}/")
                    else "input"
                )
                year_label = (
                    "start-year"
                    if start_year is not None
                    and table_path.startswith(f"/{start_year}/")
                    else "resolved"
                )
                artifact_key = (
                    f"activitysim_preprocess_usim_{table_name}_table_{key_suffix}"
                )
                child_specs[table_path] = H5ChildSpec(
                    key=artifact_key,
                    description=(
                        f"UrbanSim {year_label} {table_name} table used by ActivitySim preprocess"
                    ),
                    metadata={
                        "h5_parent_key": artifact_key.rsplit("_table_", 1)[0],
                        "h5_table_name": table_path.split("/")[-1],
                    },
                )
            log_and_set_input(
                key=USIM_POPULATION_SOURCE_H5,
                path=usim_path,
                description=input_desc,
                coupler=coupler,
                profile_file_schema=True,
                h5_container=True,
                h5_tables_used=h5_tables_used,
                child_specs=child_specs,
                child_selection="include_only",
            )
        return {
            "population_source_h5_path": usim_path,
            **{
                table_key: runtime_inputs[table_key]
                for table_key in _ACTIVITYSIM_POPULATION_TABLE_KEYS
                if isinstance(runtime_inputs.get(table_key), str)
                and runtime_inputs.get(table_key)
            },
        }

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
                step_name="activitysim_preprocess",
                **meta,
            ),
            profile_schema_keys={ASIM_HOUSEHOLDS_IN, ASIM_PERSONS_IN, ASIM_LAND_USE_IN},
        )

    step_func = build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="activitysim_preprocess",
            model_name="activitysim",
            phase="preprocess",
            outputs_class=ActivitySimPreprocessOutputs,
            component_getter=lambda factory, state: factory.get_preprocessor(
                "activitysim", state
            ),
            component_executor=_execute_activitysim_preprocess,
            input_logger=_log_inputs,
            output_logger=_log_outputs,
            output_recoverer=_recover_activitysim_preprocess_outputs,
            input_paths=ActivitysimPreprocessor.declared_expected_inputs,
            output_paths=ActivitysimPreprocessor.expected_outputs,
            cache_hydration="metadata",
            input_binding="paths",
            input_materialization="requested",
            step_logger=logger,
        ),
    )
    return step_func


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
        step_inputs: Optional[Mapping[str, Any]] = None,
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
        zarr_content_hash = artifact_fingerprint(zarr_value)
        if zarr_content_hash:
            compile_input_hashes[ZARR_SKIMS] = zarr_content_hash
        zarr_path = artifact_to_path(zarr_value, workspace)
        if zarr_path and os.path.exists(zarr_path):
            extra_inputs[ZARR_SKIMS] = zarr_path
            log_input_only(
                key=ZARR_SKIMS,
                path=zarr_path,
                description=(
                    f"ActivitySim compiled skims for year {state.year}, iter {state.iteration}"
                ),
            )

        cache_value = None
        if callable(get_value):
            cache_value = resolve_artifact_from_value(
                get_value(ASIM_SHARROW_CACHE_DIR),
                key=ASIM_SHARROW_CACHE_DIR,
                workspace=workspace,
            )
        cache_content_hash = artifact_fingerprint(cache_value)
        if cache_content_hash:
            compile_input_hashes[ASIM_SHARROW_CACHE_DIR] = cache_content_hash
        cache_path = artifact_to_path(cache_value, workspace)
        if cache_path and _is_non_empty_directory(cache_path):
            extra_inputs[ASIM_SHARROW_CACHE_DIR] = cache_path
            log_input_only(
                key=ASIM_SHARROW_CACHE_DIR,
                path=cache_path,
                description=(
                    "ActivitySim persisted compile cache directory (numba/sharrow)"
                ),
            )
        return {"extra_inputs": extra_inputs}

    def _log_outputs(
        outputs: ActivitySimRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        forecast_year = state.forecast_year
        if forecast_year is None:
            raise RuntimeError(
                "WorkflowState.forecast_year must be set before ActivitySim run logging."
            )
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
        runtime_zarr_path = asim_runtime_zarr_path(workspace)
        zarr_path = (
            runtime_zarr_path
            if os.path.exists(runtime_zarr_path)
            else artifact_to_path(zarr_value, workspace)
        )
        if zarr_path and os.path.exists(zarr_path):
            outputs.source_input_paths[ZARR_SKIMS] = Path(zarr_path)
            content_hash = artifact_fingerprint(zarr_value) or compile_input_hashes.get(
                ZARR_SKIMS
            )
            if content_hash:
                outputs.source_input_hashes[ZARR_SKIMS] = content_hash

        cache_value = get_value(ASIM_SHARROW_CACHE_DIR) if callable(get_value) else None
        runtime_cache_path = asim_sharrow_cache_dir(workspace)
        cache_path = (
            runtime_cache_path
            if _is_non_empty_directory(runtime_cache_path)
            else artifact_to_path(cache_value, workspace)
        )
        if cache_path and _is_non_empty_directory(cache_path):
            outputs.source_input_paths[ASIM_SHARROW_CACHE_DIR] = Path(cache_path)
            content_hash = artifact_fingerprint(
                cache_value
            ) or compile_input_hashes.get(ASIM_SHARROW_CACHE_DIR)
            if content_hash:
                outputs.source_input_hashes[ASIM_SHARROW_CACHE_DIR] = content_hash

        for short_name, path, description in outputs._iter_record_items():
            output_key = _canonical_activitysim_run_output_key(short_name)
            artifact = log_and_set_output(
                key=output_key,
                path=str(path),
                description=description.replace(short_name, output_key),
                coupler=coupler,
                step_name="activitysim_run",
                **_activitysim_output_facet_meta(
                    output_key,
                    year=forecast_year,
                    iteration=state.iteration,
                ),
            )
            content_hash = artifact_fingerprint(artifact)
            if content_hash:
                outputs.raw_output_hashes[short_name] = content_hash
                outputs.raw_output_hashes.setdefault(output_key, content_hash)

    step_func = build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="activitysim_run",
            model_name="activitysim",
            phase="run",
            outputs_class=ActivitySimRunOutputs,
            component_getter=lambda factory, state: factory.get_runner(
                "activitysim", state
            ),
            component_executor=_execute_activitysim_run,
            input_logger=_log_inputs,
            output_logger=_log_outputs,
            output_recoverer=_recover_activitysim_run_outputs,
            declared_outputs=list(ASIM_REQUIRED_RUN_OUTPUT_KEYS),
            schema_outputs=list(ASIM_REQUIRED_RUN_OUTPUT_KEYS),
            input_paths=ActivitysimRunner.declared_expected_inputs,
            output_paths=ActivitysimRunner.expected_outputs,
            cache_hydration="metadata",
            input_binding="paths",
            input_materialization="requested",
            step_logger=logger,
        ),
    )
    return step_func


def make_activitysim_postprocess_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    surface: Optional["EnabledWorkflowSurface"] = None,
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
        step_inputs: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        forecast_year = state.forecast_year
        if forecast_year is None:
            raise RuntimeError(
                "WorkflowState.forecast_year must be set before ActivitySim postprocess."
            )
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
                asim_runtime_zarr_path(workspace),
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

        runtime_inputs = _resolve_activitysim_postprocess_runtime_inputs(
            settings=settings,
            state=state,
            workspace=workspace,
            coupler=coupler,
            step_inputs=step_inputs,
            surface=surface,
        )
        if state.is_enabled(WorkflowState.Stage.land_use):
            population_source_h5_path = runtime_inputs["population_source_h5_path"]
            current_input_h5_path = runtime_inputs["current_input_h5_path"]
            if population_source_h5_path:
                log_input_only(
                    key=USIM_POPULATION_SOURCE_H5,
                    path=population_source_h5_path,
                    description=(
                        "ActivitySim postprocess upstream population-source datastore"
                    ),
                    profile_file_schema="if_changed",
                )
            if current_input_h5_path:
                log_input_only(
                    key=USIM_DATASTORE_CURRENT_H5,
                    path=current_input_h5_path,
                    description=(
                        "ActivitySim postprocess source UrbanSim current-input datastore"
                    ),
                    profile_file_schema="if_changed",
                )
        return runtime_inputs

    def _log_outputs(
        outputs: ActivitySimPostprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        forecast_year = state.forecast_year
        if forecast_year is None:
            raise RuntimeError(
                "WorkflowState.forecast_year must be set before ActivitySim postprocess logging."
            )

        def _extra_meta(
            short_name: str, _path: str, _description: str
        ) -> Dict[str, Any]:
            meta: Dict[str, Any] = _activitysim_output_facet_meta(
                short_name,
                year=forecast_year,
                iteration=state.iteration,
            )
            content_hash = outputs.processed_output_hashes.get(short_name)
            if content_hash:
                meta["content_hash"] = content_hash
            return meta

        handoff_keys = {
            "beam_plans_asim_out",
            "households_asim_out",
            "linkstats",
            "persons_asim_out",
            USIM_DATASTORE_H5,
        }
        profile_schema_keys = {
            "persons_asim_out",
            "trips_asim_out",
            "tours_asim_out",
            "beam_plans_asim_out",
            "households_asim_out",
        }
        handoff_items = []
        other_items = []
        for record in outputs._iter_record_items():
            if record[0] in handoff_keys:
                handoff_items.append(record)
            else:
                other_items.append(record)

        _log_step_records(
            record_items=other_items,
            log_fn=lambda key, path, description, **meta: log_output_only(
                key=key,
                path=path,
                description=description,
                step_name="activitysim_postprocess",
                **meta,
            ),
            profile_schema_keys=profile_schema_keys,
            extra_meta_fn=_extra_meta,
        )
        _log_step_records(
            record_items=(
                record for record in handoff_items if record[0] != USIM_DATASTORE_H5
            ),
            log_fn=lambda key, path, description, **meta: log_and_set_output(
                key=key,
                path=path,
                description=description,
                coupler=coupler,
                step_name="activitysim_postprocess",
                **meta,
            ),
            profile_schema_keys=profile_schema_keys,
            extra_meta_fn=_extra_meta,
        )
        if outputs.usim_datastore_h5 is not None:
            child_specs = {
                "/households": H5ChildSpec(
                    key="activitysim_postprocess_usim_households_table_updated",
                    description="UrbanSim households table updated by ActivitySim postprocess",
                    metadata={
                        "h5_parent_key": "activitysim_postprocess_usim_households",
                        "h5_table_name": "households",
                    },
                ),
                "/persons": H5ChildSpec(
                    key="activitysim_postprocess_usim_persons_table_updated",
                    description="UrbanSim persons table updated by ActivitySim postprocess",
                    metadata={
                        "h5_parent_key": "activitysim_postprocess_usim_persons",
                        "h5_table_name": "persons",
                    },
                ),
            }
            log_and_set_output(
                key=USIM_DATASTORE_H5,
                path=str(outputs.usim_datastore_h5),
                description=(
                    f"UrbanSim datastore updated by ActivitySim for year {forecast_year}"
                ),
                coupler=coupler,
                step_name="activitysim_postprocess",
                profile_file_schema=True,
                h5_container=True,
                hash_tables="if_unchanged",
                h5_tables_used=list(child_specs.keys()),
                child_specs=child_specs,
                child_selection="include_only",
            )

    step_func = build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="activitysim_postprocess",
            model_name="activitysim",
            phase="postprocess",
            outputs_class=ActivitySimPostprocessOutputs,
            component_getter=lambda factory, state: factory.get_postprocessor(
                "activitysim", state
            ),
            component_executor=_execute_activitysim_postprocess,
            input_logger=_log_inputs,
            output_logger=_log_outputs,
            output_recoverer=_recover_activitysim_postprocess_outputs,
            input_paths=ActivitysimPostprocessor.declared_expected_inputs,
            output_paths=ActivitysimPostprocessor.expected_outputs,
            cache_hydration="metadata",
            input_binding="paths",
            input_materialization="requested",
            step_logger=logger,
        ),
    )
    return step_func
