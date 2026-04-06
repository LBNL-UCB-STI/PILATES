from __future__ import annotations

import inspect
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, cast

from pilates.activitysim.runner import (
    ActivitysimCompileRunner,
    asim_sharrow_cache_dir,
    persist_sharrow_cache_enabled,
)
from pilates.activitysim.outputs import has_asim_run_marker, normalize_asim_output_key
from pilates.activitysim.outputs import ASIM_REQUIRED_RUN_OUTPUT_KEYS
from pilates.utils.coupler_helpers import (
    artifact_to_existing_path,
    resolve_existing_path,
)
from pilates.activitysim.postprocessor import get_usim_datastore_fname
from pilates.config.models import PilatesConfig
from pilates.generic.model_factory import ModelFactory
from pilates.workflows.artifact_keys import ASIM_SHARROW_CACHE_DIR
from pilates.workflows.binding import build_binding_plan
from pilates.workflows.outputs_base import (
    StepOutputsBase,
    ValidationContext,
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
    StandardStepSpec,
    StepOutputsHolder,
    WorkflowState,
    _activitysim_output_facet_meta,
    _decorate_step_with_consist,
    build_standard_step,
    _log_named_h5_tables,
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
    selected_candidate_key,
    resolved_value_for_key,
)
from pilates.workflows.tracker_outputs import (
    load_tracker_run_outputs,
    merge_canonical_output_mappings,
)

logger = logging.getLogger(__name__)


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
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


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


def _execute_activitysim_preprocess(
    preprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> ActivitySimPreprocessOutputs:
    filtered_kwargs = _filter_kwargs_for_callable(
        preprocessor.preprocess,
        _strip_component_runtime_kwargs(kwargs),
    )
    return preprocessor.preprocess(
        workspace,
        **filtered_kwargs,
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
    return postprocessor.postprocess(upstream, workspace)


def _resolve_cached_run_outputs(run_id: Optional[str]) -> Dict[str, Any]:
    return load_tracker_run_outputs(
        run_id,
        logger=logger,
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
    return _artifact_content_hash(artifact)


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
    del settings, state, outputs_holder, step_inputs
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    candidates = {
        ASIM_HOUSEHOLDS_IN: asim_dir / "households.csv",
        ASIM_PERSONS_IN: asim_dir / "persons.csv",
        ASIM_LAND_USE_IN: asim_dir / "land_use.csv",
        ASIM_OMX_SKIMS: asim_dir / "skims.omx",
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
    del settings, step_inputs
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

    zarr_candidate = Path(
        _existing_local_path(asim_output_dir / "cache" / "skims.zarr", workspace)
        or (asim_output_dir / "cache" / "skims.zarr")
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

    inputs_dir = Path(
        _existing_local_path(
            asim_output_dir / f"inputs-year-{state.year}-iteration-{state.iteration}",
            workspace,
        )
        or (asim_output_dir / f"inputs-year-{state.year}-iteration-{state.iteration}")
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
    if step_inputs and USIM_H5_UPDATED in step_inputs:
        usim_path = _existing_artifact_path(step_inputs[USIM_H5_UPDATED], workspace)
    if not usim_path and step_inputs and USIM_DATASTORE_CURRENT_H5 in step_inputs:
        usim_path = _existing_artifact_path(
            step_inputs[USIM_DATASTORE_CURRENT_H5], workspace
        )
    if not usim_path and step_inputs and USIM_DATASTORE_BASE_H5 in step_inputs:
        usim_path = _existing_artifact_path(
            step_inputs[USIM_DATASTORE_BASE_H5], workspace
        )
    if not usim_path:
        urbansim_settings = settings.urbansim
        if urbansim_settings is not None:
            usim_path = os.path.join(
                workspace.get_usim_mutable_data_dir(),
                urbansim_settings.output_file_template.format(year=state.forecast_year),
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
    settings = getattr(ctx, "runtime_settings", None)
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
        tags=["activitysim", "compile"],
    )
    return step_func


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
        resolution = build_binding_plan(
            step_name="activitysim_preprocess",
            coupler=coupler,
            settings=settings,
            state=state,
            workspace=workspace,
            year=state.year,
        )
        selected_key = selected_candidate_key(resolution, USIM_DATASTORE_CURRENT_H5)
        selected_value = (
            resolved_value_for_key(
                resolved=resolution,
                key=USIM_DATASTORE_CURRENT_H5,
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
                step_name="activitysim_preprocess",
                **meta,
            ),
            profile_schema_keys={ASIM_HOUSEHOLDS_IN, ASIM_PERSONS_IN, ASIM_LAND_USE_IN},
        )

    return build_standard_step(
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
            step_logger=logger,
        ),
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
        cache_content_hash = _artifact_content_hash(cache_value)
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
        runtime_zarr_path = os.path.join(
            workspace.get_asim_output_dir(), "cache", "skims.zarr"
        )
        zarr_path = (
            runtime_zarr_path
            if os.path.exists(runtime_zarr_path)
            else artifact_to_path(zarr_value, workspace)
        )
        if zarr_path and os.path.exists(zarr_path):
            outputs.source_input_paths[ZARR_SKIMS] = Path(zarr_path)
            content_hash = _artifact_content_hash(
                zarr_value
            ) or compile_input_hashes.get(ZARR_SKIMS)
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
            content_hash = _artifact_content_hash(
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
            content_hash = _artifact_content_hash(artifact)
            if content_hash:
                outputs.raw_output_hashes[short_name] = content_hash
                outputs.raw_output_hashes.setdefault(output_key, content_hash)

    return build_standard_step(
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
            step_logger=logger,
        ),
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
                get_usim_datastore_fname(settings, io="output", year=forecast_year),
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

    return build_standard_step(
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
            step_logger=logger,
        ),
    )
