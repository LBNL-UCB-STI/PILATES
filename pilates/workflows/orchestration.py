from __future__ import annotations

import logging
from dataclasses import dataclass
import os
from datetime import datetime
from pathlib import Path
import warnings
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Sequence, Set

from consist.types import CacheOptions, ExecutionOptions, OutputPolicyOptions

from pilates.utils.coupler_helpers import (
    artifact_to_existing_path,
    artifact_to_path,
    resolve_existing_path,
    resolve_artifact_from_value,
    set_coupler_from_artifact,
)
from pilates.utils import consist_runtime as cr
from pilates.workflows.artifact_key_migrations import resolve_artifact_key
from pilates.activitysim.outputs import (
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
    has_asim_run_marker,
    normalize_asim_output_key,
)
from pilates.atlas.outputs import (
    AtlasPostprocessOutputs,
    AtlasPreprocessOutputs,
    AtlasRunOutputs,
)
from pilates.beam.outputs import (
    BeamFullSkimOutputs,
    BeamPostprocessOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
)
from pilates.urbansim.outputs import (
    UrbanSimPostprocessOutputs,
    UrbanSimPreprocessOutputs,
    UrbanSimRunOutputs,
)
from pilates.workflows.coupler_namespace import resolve_coupler_value
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    BEAM_FULL_SKIMS,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    FINAL_SKIMS_OMX,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_FORECAST_OUTPUT,
    USIM_H5_UPDATED,
    USIM_INPUT_MERGED_PREFIX,
    ZARR_SKIMS,
)
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.step_manifest import load_step_manifest, save_step_manifest
from pilates.workflows.outputs_base import (
    declared_outputs_for_step_outputs_class,
    deserialize_step_outputs,
    step_output_mapping,
    serialize_step_outputs,
)
from pilates.workflows.step_runner import common_runtime_kwargs
from pilates.workflows.steps import (
    STEP_OUTPUTS_CLASSES,
    StepOutputsHolder,
    validate_step_ready,
    validate_workflow_step_contracts,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepRef:
    """
    Declarative reference to a workflow step invocation.
    """

    name: str
    step_func: Callable[..., None]
    input_keys: Optional[Sequence[str]] = None
    inputs: Optional[Dict[str, Any]] = None
    output_paths: Optional[Dict[str, Any]] = None
    cache_hydration: Optional[str] = None
    cache_mode: Optional[str] = None
    load_inputs: Optional[bool] = None
    required_outputs: Optional[Sequence[str]] = None
    required_outputs_rationale: Optional[str] = None
    output_missing: Optional[Literal["warn", "error", "ignore"]] = None
    output_mismatch: Optional[Literal["warn", "error", "ignore"]] = None
    model: Optional[str] = None
    year: Optional[int] = None
    iteration: Optional[int] = None
    phase: Optional[str] = None
    stage: Optional[str] = None


def _infer_phase(step_name: str) -> Optional[str]:
    if "_" not in step_name:
        return None
    return step_name.rsplit("_", 1)[-1] or None


def _build_step_run_kwargs(
    *,
    step: StepRef,
    settings: Any,
    state: Any,
    runtime_kwargs: Mapping[str, Any],
    stage_name: str,
    default_iteration: int,
) -> Dict[str, Any]:
    if not hasattr(step.step_func, "__consist_step__"):
        raise TypeError(
            f"Step '{step.name}' must be decorated with @define_step metadata."
        )

    run_kwargs: Dict[str, Any] = {"fn": step.step_func}
    step_meta = getattr(step.step_func, "__consist_step__", None)
    resolved_year = step.year
    if resolved_year is None:
        resolved_year = getattr(state, "year", None)
    if resolved_year is not None:
        run_kwargs["year"] = resolved_year

    resolved_iteration = step.iteration
    if resolved_iteration is None:
        resolved_iteration = default_iteration
    if resolved_iteration is not None:
        run_kwargs["iteration"] = resolved_iteration

    resolved_phase = step.phase or _infer_phase(step.name)
    if resolved_phase:
        run_kwargs["phase"] = resolved_phase
    resolved_stage = step.stage or stage_name
    if resolved_stage:
        run_kwargs["stage"] = resolved_stage

    if step.inputs is not None:
        run_kwargs["inputs"] = step.inputs
    if step.input_keys is not None:
        run_kwargs["input_keys"] = step.input_keys
    if step.output_paths is not None:
        run_kwargs["output_paths"] = step.output_paths
    run_kwargs["execution_options"] = ExecutionOptions(
        runtime_kwargs=runtime_kwargs,
        load_inputs=step.load_inputs,
    )

    run_cfg = getattr(settings, "run", None)
    code_identity = getattr(run_cfg, "consist_code_identity", None)
    if (
        step.cache_hydration is not None
        or step.cache_mode is not None
        or code_identity is not None
    ):
        run_kwargs["cache_options"] = CacheOptions(
            cache_hydration=step.cache_hydration,
            cache_mode=step.cache_mode,
            code_identity=code_identity,
        )
    def _normalize_output_keys(values: Any) -> Optional[list[str]]:
        if not isinstance(values, Sequence) or isinstance(values, str):
            return None
        return [output for output in values if isinstance(output, str)]

    outputs_class = STEP_OUTPUTS_CLASSES.get(step.name)
    canonical_outputs: list[str] = []
    if outputs_class is not None:
        canonical_outputs = list(declared_outputs_for_step_outputs_class(outputs_class))

    resolved_required_outputs: Optional[Sequence[str]] = None
    if step.required_outputs is not None:
        rationale = (step.required_outputs_rationale or "").strip()
        if not rationale:
            raise ValueError(
                f"Step '{step.name}': StepRef.required_outputs override requires "
                "StepRef.required_outputs_rationale with a non-empty explanation."
            )
        warnings.warn(
            f"Step '{step.name}': StepRef.required_outputs is deprecated. "
            "Use StepOutputs declared_outputs instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        resolved_required_outputs = _normalize_output_keys(step.required_outputs)
    elif outputs_class is not None:
        # Tracked steps use StepOutputs declarations as the canonical source.
        resolved_required_outputs = canonical_outputs or None
    elif step_meta is not None:
        # Metadata-only steps remain supported for non-catalog call sites.
        resolved_required_outputs = _normalize_output_keys(
            getattr(step_meta, "outputs", None)
        )

    if resolved_required_outputs:
        run_kwargs["outputs"] = list(resolved_required_outputs)

    # Apply strict defaults for declared-output steps unless explicitly overridden.
    output_missing = step.output_missing
    output_mismatch = step.output_mismatch
    if output_missing is None and resolved_required_outputs:
        output_missing = "error"
    if output_mismatch is None and resolved_required_outputs:
        output_mismatch = "error"
    if output_missing is not None or output_mismatch is not None:
        run_kwargs["output_policy"] = OutputPolicyOptions(
            output_missing=output_missing,
            output_mismatch=output_mismatch,
        )

    if step.model is not None:
        run_kwargs["model"] = step.model
    return run_kwargs


@dataclass
class WorkflowStage:
    """
    Orchestrates a sequence of related steps.
    """

    name: str
    stage_type: Any
    steps: Sequence[StepRef]

    def run(
        self,
        *,
        scenario: Any,
        state: Any,
        settings: Any,
        workspace: Any,
        coupler: CouplerProtocol,
        outputs_holder: StepOutputsHolder,
        name_suffix: str,
        iteration: int = 0,
        runtime_kwargs_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Execute all steps in sequence for this stage.
        """
        run_workflow(
            stage_name=self.name,
            steps=self.steps,
            scenario=scenario,
            state=state,
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            outputs_holder=outputs_holder,
            name_suffix=name_suffix,
            iteration=iteration,
            runtime_kwargs_extra=runtime_kwargs_extra,
        )


@dataclass(frozen=True)
class ManifestConfig:
    """
    Configuration for manifest-based step checkpointing.
    """

    path: Path


def run_manifested_steps(
    *,
    stage_name: str,
    steps: Sequence[StepRef],
    outputs_holder: StepOutputsHolder,
    manifest_config: ManifestConfig,
    scenario: Any,
    state: Any,
    settings: Any,
    workspace: Any,
    coupler: CouplerProtocol,
    name_suffix: str,
    iteration: int = 0,
    runtime_kwargs_extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Execute steps with manifest checkpointing and stale detection.
    """
    validate_workflow_step_contracts(step_refs=steps)

    manifest = load_step_manifest(manifest_config.path) or {}
    stale_steps = _detect_stale_steps(manifest, outputs_holder, workspace)
    if stale_steps:
        for step_name in stale_steps:
            manifest.pop(step_name, None)
        save_step_manifest(manifest, manifest_config.path)

    runtime_kwargs = common_runtime_kwargs(
        settings=settings,
        state=state,
        workspace=workspace,
        **(runtime_kwargs_extra or {}),
    )

    def _publish_recovered_outputs(step_func: Callable[..., Any], outputs: Any) -> None:
        replayer = getattr(step_func, "__pilates_output_replayer__", None)
        if callable(replayer):
            replayer(outputs, settings, state, workspace, outputs_holder)
        _update_coupler_from_outputs(outputs, coupler=coupler, workspace=workspace)

    for raw_step in steps:
        spec = raw_step
        if spec.name in manifest:
            logger.info("[%s] %s already completed (skipping)", stage_name, spec.name)
            outputs = outputs_holder.get_attribute(spec.name)
            if outputs is None:
                outputs = _restore_outputs_from_manifest(spec.name, manifest, workspace)
                if outputs is not None:
                    outputs_holder.set_attribute(spec.name, outputs)
            if outputs is not None:
                _publish_recovered_outputs(spec.step_func, outputs)
            continue

        validate_step_ready(spec.name, outputs_holder)
        run_kwargs = _build_step_run_kwargs(
            step=spec,
            settings=settings,
            state=state,
            runtime_kwargs=runtime_kwargs,
            stage_name=stage_name,
            default_iteration=iteration,
        )
        result = scenario.run(**run_kwargs)
        outputs = outputs_holder.get_attribute(spec.name)
        if outputs is None and getattr(result, "cache_hit", False):
            outputs = _recover_cached_outputs(
                step_name=spec.name,
                outputs_holder=outputs_holder,
                settings=settings,
                state=state,
                workspace=workspace,
                coupler=coupler,
                step_inputs=spec.inputs,
                cached_outputs=getattr(result, "outputs", None),
                run_id=getattr(getattr(result, "run", None), "id", None),
                publish_outputs=False,
            )
            if outputs is not None:
                _publish_recovered_outputs(spec.step_func, outputs)
        if outputs is None and getattr(result, "cache_hit", False):
            logger.warning(
                "[%s] Cache hit for %s could not hydrate outputs_holder; rerunning with cache_mode=overwrite.",
                stage_name,
                spec.name,
            )
            rerun_kwargs = dict(run_kwargs)
            rerun_kwargs["cache_options"] = CacheOptions(cache_mode="overwrite")
            result = scenario.run(**rerun_kwargs)
            outputs = outputs_holder.get_attribute(spec.name)
            if outputs is None and getattr(result, "cache_hit", False):
                outputs = _recover_cached_outputs(
                    step_name=spec.name,
                    outputs_holder=outputs_holder,
                    settings=settings,
                    state=state,
                    workspace=workspace,
                    coupler=coupler,
                    step_inputs=spec.inputs,
                    cached_outputs=getattr(result, "outputs", None),
                    run_id=getattr(getattr(result, "run", None), "id", None),
                    publish_outputs=False,
                )
                if outputs is not None:
                    _publish_recovered_outputs(spec.step_func, outputs)
        if outputs is None:
            raise RuntimeError(f"{spec.name} did not populate outputs_holder")
        manifest[spec.name] = {
            "completed_at": datetime.now().isoformat(),
            "cache_hit": bool(getattr(result, "cache_hit", False)),
            "outputs": serialize_step_outputs(outputs),
        }
        save_step_manifest(manifest, manifest_config.path)


def run_workflow(
    *,
    stage_name: str,
    steps: Sequence[StepRef],
    scenario: Any,
    state: Any,
    settings: Any,
    workspace: Any,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    name_suffix: str,
    iteration: int = 0,
    runtime_kwargs_extra: Optional[Dict[str, Any]] = None,
    manifest_config: Optional[ManifestConfig] = None,
) -> None:
    """
    Execute a sequence of workflow steps using native step metadata.
    """
    validate_workflow_step_contracts(step_refs=steps)

    if manifest_config is not None:
        run_manifested_steps(
            stage_name=stage_name,
            steps=steps,
            outputs_holder=outputs_holder,
            manifest_config=manifest_config,
            scenario=scenario,
            state=state,
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            name_suffix=name_suffix,
            iteration=iteration,
            runtime_kwargs_extra=runtime_kwargs_extra,
        )
        return

    runtime_kwargs = common_runtime_kwargs(
        settings=settings,
        state=state,
        workspace=workspace,
        **(runtime_kwargs_extra or {}),
    )
    for raw_step in steps:
        spec = raw_step
        validate_step_ready(spec.name, outputs_holder)
        run_kwargs = _build_step_run_kwargs(
            step=spec,
            settings=settings,
            state=state,
            runtime_kwargs=runtime_kwargs,
            stage_name=stage_name,
            default_iteration=iteration,
        )
        coupler_keys = None
        coupler_keys_fn = getattr(coupler, "keys", None)
        if callable(coupler_keys_fn):
            try:
                coupler_keys = list(coupler_keys_fn())
            except TypeError:
                coupler_keys = None
        logger.debug(
            "[%s] Running %s: inputs=%s input_keys=%s outputs=%s",
            stage_name,
            spec.name,
            bool(spec.inputs),
            spec.input_keys,
            spec.output_paths,
        )
        if coupler_keys is not None:
            logger.debug(
                "[%s] Coupler keys before %s: %s",
                stage_name,
                spec.name,
                coupler_keys,
            )

        result = scenario.run(**run_kwargs)
        if (
            outputs_holder.get_attribute(spec.name) is None
            and getattr(result, "cache_hit", False)
        ):
            _recover_cached_outputs(
                step_name=spec.name,
                outputs_holder=outputs_holder,
                settings=settings,
                state=state,
                workspace=workspace,
                coupler=coupler,
                step_inputs=spec.inputs,
                cached_outputs=getattr(result, "outputs", None),
                run_id=getattr(getattr(result, "run", None), "id", None),
            )
        if (
            outputs_holder.get_attribute(spec.name) is None
            and getattr(result, "cache_hit", False)
        ):
            logger.warning(
                "[%s] Cache hit for %s could not hydrate outputs_holder; rerunning with cache_mode=overwrite.",
                stage_name,
                spec.name,
            )
            rerun_kwargs = dict(run_kwargs)
            rerun_kwargs["cache_options"] = CacheOptions(cache_mode="overwrite")
            result = scenario.run(**rerun_kwargs)
            if (
                outputs_holder.get_attribute(spec.name) is None
                and getattr(result, "cache_hit", False)
            ):
                _recover_cached_outputs(
                    step_name=spec.name,
                    outputs_holder=outputs_holder,
                    settings=settings,
                    state=state,
                    workspace=workspace,
                    coupler=coupler,
                    step_inputs=spec.inputs,
                    cached_outputs=getattr(result, "outputs", None),
                    run_id=getattr(getattr(result, "run", None), "id", None),
                )

        if coupler_keys is not None:
            try:
                current_coupler_keys_fn = getattr(coupler, "keys", None)
                if not callable(current_coupler_keys_fn):
                    raise TypeError
                logger.debug(
                    "[%s] Coupler keys after %s: %s",
                    stage_name,
                    spec.name,
                    list(current_coupler_keys_fn()),
                )
            except TypeError:
                logger.debug(
                    "[%s] Coupler keys after %s: <unavailable>",
                    stage_name,
                    spec.name,
                )


def _detect_stale_steps(
    manifest: Dict[str, Any],
    outputs_holder: StepOutputsHolder,
    workspace: Any,
) -> Set[str]:
    """
    Check which manifest entries have stale or missing outputs.
    """
    stale: Set[str] = set()
    for step_name, step_info in manifest.items():
        outputs_class = STEP_OUTPUTS_CLASSES.get(step_name)
        if outputs_class is None:
            continue
        outputs_data = step_info.get("outputs", {})
        try:
            outputs = deserialize_step_outputs(outputs_class, outputs_data)
            validate = getattr(outputs, "validate", None)
            if callable(validate):
                validate()
            outputs_holder.set_attribute(step_name, outputs)
        except (AssertionError, FileNotFoundError) as exc:
            logger.warning(
                "Manifest outputs for %s are stale; will re-run (%s)",
                step_name,
                exc,
            )
            stale.add(step_name)
    return stale


def _recover_cached_outputs(
    *,
    step_name: str,
    outputs_holder: StepOutputsHolder,
    settings: Any,
    state: Any,
    workspace: Any,
    coupler: CouplerProtocol,
    step_inputs: Optional[Mapping[str, Any]] = None,
    cached_outputs: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
    publish_outputs: bool = True,
) -> Optional[Any]:
    """
    Best-effort output recovery for cache hits that skip step execution.
    """
    _cached_run_outputs_by_key: Optional[Dict[str, Any]] = None

    def _cached_run_outputs() -> Dict[str, Any]:
        nonlocal _cached_run_outputs_by_key
        if _cached_run_outputs_by_key is not None:
            return _cached_run_outputs_by_key
        _cached_run_outputs_by_key = {}
        if not run_id:
            return _cached_run_outputs_by_key
        tracker = cr.current_tracker()
        if tracker is None:
            return _cached_run_outputs_by_key
        get_run_outputs = getattr(tracker, "get_run_outputs", None)
        if not callable(get_run_outputs):
            return _cached_run_outputs_by_key
        try:
            run_outputs = get_run_outputs(run_id) or {}
        except Exception:
            logger.debug(
                "Failed loading cached run outputs for run_id=%s", run_id, exc_info=True
            )
            return _cached_run_outputs_by_key
        for raw_key, value in run_outputs.items():
            if value is None:
                continue
            raw_key_str = str(raw_key)
            local_key = raw_key_str.split("/", 1)[-1]
            canonical_key = resolve_artifact_key(local_key)
            _cached_run_outputs_by_key[canonical_key] = value
        return _cached_run_outputs_by_key

    def _recovered_cached_paths() -> Dict[str, Path]:
        merged: Dict[str, Any] = {}
        if cached_outputs:
            for raw_key, value in cached_outputs.items():
                if value is None:
                    continue
                raw_key_str = str(raw_key)
                local_key = raw_key_str.split("/", 1)[-1]
                merged[resolve_artifact_key(local_key)] = value
        for key, value in _cached_run_outputs().items():
            merged[key] = value
        recovered_paths: Dict[str, Path] = {}
        for key, value in merged.items():
            path = _existing_path(value)
            if path is None:
                continue
            recovered_paths[key] = Path(path)
        return recovered_paths

    def _recover_from_cached_artifacts() -> Optional[Any]:
        recovered_paths = _recovered_cached_paths()
        if not recovered_paths:
            return None
        if step_name == "beam_run":
            return BeamRunOutputs(
                beam_output_dir=Path(workspace.get_beam_output_dir()),
                raw_outputs=recovered_paths,
            )
        if step_name == "beam_postprocess":
            return BeamPostprocessOutputs(
                zarr_skims=recovered_paths.get(ZARR_SKIMS),
                final_skims_omx=recovered_paths.get(FINAL_SKIMS_OMX),
                split_events={
                    key: path
                    for key, path in recovered_paths.items()
                    if key.startswith("events_parquet_") and "_type_" in key
                },
                split_event_links={
                    key: path
                    for key, path in recovered_paths.items()
                    if key.startswith("path_traversal_links_")
                },
            )
        if step_name == "beam_full_skim":
            full_skims = recovered_paths.get(BEAM_FULL_SKIMS)
            if full_skims is None:
                return None
            return BeamFullSkimOutputs(full_skims=full_skims)
        if step_name == "urbansim_preprocess":
            return UrbanSimPreprocessOutputs(
                usim_mutable_data_dir=Path(workspace.get_usim_mutable_data_dir()),
                prepared_inputs=recovered_paths,
            )
        if step_name == "urbansim_run":
            usim_datastore_h5 = recovered_paths.get(USIM_FORECAST_OUTPUT) or recovered_paths.get(
                USIM_DATASTORE_H5
            )
            if usim_datastore_h5 is None:
                return None
            return UrbanSimRunOutputs(
                usim_datastore_h5=usim_datastore_h5,
                raw_outputs=recovered_paths,
            )
        if step_name == "urbansim_postprocess":
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
        if step_name == "atlas_preprocess":
            return AtlasPreprocessOutputs(
                atlas_mutable_input_dir=Path(workspace.get_atlas_mutable_input_dir()),
                prepared_inputs=recovered_paths,
            )
        if step_name == "atlas_run":
            return AtlasRunOutputs(
                atlas_output_dir=Path(workspace.get_atlas_output_dir()),
                raw_outputs=recovered_paths,
            )
        if step_name == "atlas_postprocess":
            return AtlasPostprocessOutputs(
                atlas_output_dir=Path(workspace.get_atlas_output_dir()),
                usim_datastore_h5=recovered_paths.get(USIM_H5_UPDATED)
                or recovered_paths.get(USIM_DATASTORE_H5),
                processed_outputs=recovered_paths,
            )
        return None

    def _resolve_cached_value(key: str) -> Any:
        if cached_outputs:
            for raw_key, value in cached_outputs.items():
                if value is None:
                    continue
                raw_key_str = str(raw_key)
                local_key = raw_key_str.split("/", 1)[-1]
                if resolve_artifact_key(local_key) == key:
                    return value
        run_outputs = _cached_run_outputs()
        if key in run_outputs:
            return run_outputs[key]
        resolved, _ = resolve_coupler_value(coupler, key)
        return resolved

    def _existing_path(value: Any) -> Optional[str]:
        return artifact_to_existing_path(
            value,
            workspace=workspace,
            materialize_from_archive=True,
        )

    def _existing_path_str(path: Any) -> Optional[str]:
        if path is None:
            return None
        return resolve_existing_path(
            str(path),
            workspace=workspace,
            materialize_from_archive=True,
        )

    def _finalize_recovered_outputs(outputs: Any) -> Any:
        validate = getattr(outputs, "validate", None)
        if callable(validate):
            validate()
        outputs_holder.set_attribute(step_name, outputs)
        if publish_outputs:
            _update_coupler_from_outputs(outputs, coupler=coupler, workspace=workspace)
        return outputs

    if step_name == "activitysim_preprocess":
        asim_dir = Path(workspace.get_asim_mutable_data_dir())
        candidates = {
            ASIM_HOUSEHOLDS_IN: asim_dir / "households.csv",
            ASIM_PERSONS_IN: asim_dir / "persons.csv",
            ASIM_LAND_USE_IN: asim_dir / "land_use.csv",
            ASIM_OMX_SKIMS: asim_dir / "skims.omx",
        }
        recovered_inputs: Dict[str, Path] = {}
        for key, path in candidates.items():
            cached_value = _resolve_cached_value(key)
            cached_path = _existing_path(cached_value)
            if cached_path:
                path = Path(cached_path)
            resolved_candidate = _existing_path_str(path)
            if resolved_candidate:
                recovered_inputs[key] = Path(resolved_candidate)
        if not {
            ASIM_LAND_USE_IN,
            ASIM_HOUSEHOLDS_IN,
            ASIM_PERSONS_IN,
        }.issubset(recovered_inputs):
            return None
        return _finalize_recovered_outputs(
            ActivitySimPreprocessOutputs(
                mutable_data_dir=asim_dir,
                land_use_table=recovered_inputs[ASIM_LAND_USE_IN],
                households_table=recovered_inputs[ASIM_HOUSEHOLDS_IN],
                persons_table=recovered_inputs[ASIM_PERSONS_IN],
                omx_skims=recovered_inputs.get(ASIM_OMX_SKIMS),
            )
        )
    elif step_name == "activitysim_run":
        asim_output_dir = Path(workspace.get_asim_output_dir())
        raw_outputs: Dict[str, Path] = {}
        final_pipeline = Path(
            _existing_path_str(asim_output_dir / "final_pipeline")
            or (asim_output_dir / "final_pipeline")
        )
        allow_final_pipeline = has_asim_run_marker(
            asim_output_dir, state.year, state.iteration
        )
        if final_pipeline.exists() and allow_final_pipeline:
            for child in final_pipeline.iterdir():
                fpath = child / "final.parquet"
                if fpath.is_file():
                    raw_outputs[f"{child.name}_asim_out_temp"] = fpath
        elif final_pipeline.exists() and not allow_final_pipeline:
            logger.warning(
                "Skipping ActivitySim final_pipeline cache recovery for year %s iteration %s "
                "because success marker is missing.",
                state.year,
                state.iteration,
        )
        if not raw_outputs:
            iter_dir = Path(
                _existing_path_str(
                    asim_output_dir / f"year-{state.year}-iteration-{state.iteration}"
                )
                or (asim_output_dir / f"year-{state.year}-iteration-{state.iteration}")
            )
            if iter_dir.exists():
                for fpath in iter_dir.glob("*.parquet"):
                    raw_outputs[f"{fpath.stem}_asim_out_temp"] = fpath
        if not raw_outputs:
            return None
        return _finalize_recovered_outputs(
            ActivitySimRunOutputs(
                output_dir=asim_output_dir,
                raw_outputs=raw_outputs,
            )
        )
    elif step_name == "activitysim_postprocess":
        asim_output_dir = Path(workspace.get_asim_output_dir())
        iter_dir = Path(
            _existing_path_str(
                asim_output_dir / f"year-{state.year}-iteration-{state.iteration}"
            )
            or (asim_output_dir / f"year-{state.year}-iteration-{state.iteration}")
        )
        processed_outputs: Dict[str, Path] = {}
        if iter_dir.exists():
            required_outputs = {
                "persons_asim_out",
                "households_asim_out",
                "beam_plans_asim_out",
            }
            available_outputs = {
                normalize_asim_output_key(path.stem)
                for path in iter_dir.glob("*.parquet")
            }
            if not required_outputs.issubset(available_outputs):
                return None
            for fpath in iter_dir.glob("*.parquet"):
                short_name = normalize_asim_output_key(fpath.stem)
                processed_outputs[short_name] = fpath
        else:
            return None

        inputs_dir = Path(
            _existing_path_str(
                asim_output_dir / f"inputs-year-{state.year}-iteration-{state.iteration}"
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
            zarr_path = inputs_dir / "skims.zarr"
            if zarr_path.exists():
                processed_outputs["asim_input_skims_zarr_archived"] = zarr_path

        usim_path = None
        if step_inputs and USIM_DATASTORE_BASE_H5 in step_inputs:
            usim_path = artifact_to_existing_path(
                step_inputs[USIM_DATASTORE_BASE_H5], workspace
            )
        if not usim_path and step_inputs and USIM_DATASTORE_CURRENT_H5 in step_inputs:
            usim_path = artifact_to_existing_path(
                step_inputs[USIM_DATASTORE_CURRENT_H5], workspace
            )
        if not usim_path:
            region_id = settings.urbansim.region_id
            if not region_id:
                region_map = settings.urbansim.region_mappings.get(
                    "region_to_region_id", {}
                )
                region_id = region_map.get(settings.run.region)
            if region_id:
                usim_path = os.path.join(
                    workspace.get_usim_mutable_data_dir(),
                    settings.urbansim.input_file_template.format(region_id=region_id),
                )
        usim_existing = _existing_path_str(usim_path)
        if not processed_outputs and not usim_existing:
            return None
        return _finalize_recovered_outputs(
            ActivitySimPostprocessOutputs(
                usim_datastore_h5=Path(usim_existing) if usim_existing else None,
                asim_output_dir=asim_output_dir,
                processed_outputs=processed_outputs,
                usim_datastore_key=(
                    f"usim_input_{state.forecast_year}" if usim_existing else None
                ),
            )
        )
    elif step_name == "beam_preprocess":
        prepared_inputs: Dict[str, Path] = {}
        has_warmstart = False
        if step_inputs:
            allowed_keys = {
                BEAM_PLANS_IN,
                BEAM_HOUSEHOLDS_IN,
                BEAM_PERSONS_IN,
                LINKSTATS_WARMSTART,
            }
            for key, value in step_inputs.items():
                if key not in allowed_keys:
                    continue
                path = _existing_path(value)
                if path:
                    if key == LINKSTATS_WARMSTART:
                        has_warmstart = True
                    prepared_inputs[key] = Path(path)
        if step_inputs and not has_warmstart:
            # If we have any linkstats-like input, recover a warmstart alias.
            candidate_keys = []
            if "linkstats" in step_inputs:
                candidate_keys.append("linkstats")
            candidate_keys.extend(
                key
                for key in sorted(step_inputs)
                if key.startswith("linkstats_parquet") and "_sub" not in key
            )
            candidate_keys.extend(
                key
                for key in sorted(step_inputs)
                if key.startswith("linkstats") and "_sub" not in key
            )
            candidate_keys.extend(
                key for key in sorted(step_inputs) if key.startswith("linkstats")
            )
            for key in candidate_keys:
                path = _existing_path(step_inputs.get(key))
                if path:
                    prepared_inputs[LINKSTATS_WARMSTART] = Path(path)
                    has_warmstart = True
                    break
        if not prepared_inputs:
            return None
        return _finalize_recovered_outputs(
            BeamPreprocessOutputs(
                beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
                prepared_inputs=prepared_inputs,
            )
        )
    else:
        outputs = _recover_from_cached_artifacts()
        if outputs is None:
            return None
        return _finalize_recovered_outputs(outputs)


def _update_coupler_from_outputs(
    outputs: Any,
    *,
    coupler: CouplerProtocol,
    workspace: Any,
) -> None:
    _update_coupler_from_mapping(
        step_output_mapping(outputs),
        coupler=coupler,
        workspace=workspace,
    )


def _update_coupler_from_mapping(
    mapping: Mapping[str, Any],
    *,
    coupler: CouplerProtocol,
    workspace: Any,
) -> None:
    if not mapping:
        return
    for key, value in mapping.items():
        canonical_key = resolve_artifact_key(key)
        resolved = resolve_artifact_from_value(
            value,
            key=canonical_key,
            workspace=workspace,
        )
        path = artifact_to_path(value, workspace)
        if path is None:
            continue
        artifact = (
            resolved
            if (
                hasattr(resolved, "container_uri")
                or hasattr(resolved, "uri")
            )
            else None
        )
        set_coupler_from_artifact(
            coupler=coupler,
            key=canonical_key,
            artifact=artifact,
            fallback=path,
        )


def _restore_outputs_from_manifest(
    step_name: str,
    manifest: Dict[str, Any],
    workspace: Any,
) -> Optional[Any]:
    """
    Restore step outputs from a manifest entry when possible.
    """
    step_info = manifest.get(step_name, {})
    outputs_class = STEP_OUTPUTS_CLASSES.get(step_name)
    if outputs_class is None:
        return None
    outputs_data = step_info.get("outputs", {})
    try:
        outputs = deserialize_step_outputs(outputs_class, outputs_data)
        validate = getattr(outputs, "validate", None)
        if callable(validate):
            validate()
        return outputs
    except (AssertionError, FileNotFoundError):
        return None
