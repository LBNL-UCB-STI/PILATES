from __future__ import annotations

import logging
from dataclasses import dataclass
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Set

from consist.types import CacheOptions, ExecutionOptions, OutputPolicyOptions

from pilates.generic.records import FileRecord, RecordStore
from pilates.utils.coupler_helpers import (
    artifact_to_path,
    record_store_to_outputs,
    resolve_artifact_from_value,
    set_coupler_from_artifact,
)
from pilates.workflows.artifact_key_migrations import resolve_artifact_key
from pilates.activitysim.outputs import normalize_asim_output_key, has_asim_run_marker
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
)
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.step_manifest import load_step_manifest, save_step_manifest
from pilates.workflows.outputs_base import (
    declared_outputs_for_step_outputs_class,
    deserialize_step_outputs,
    serialize_step_outputs,
)
from pilates.workflows.step_runner import common_runtime_kwargs
from pilates.workflows.steps import (
    STEP_OUTPUTS_CLASSES,
    StepOutputsHolder,
    validate_step_ready,
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
    output_missing: Optional[str] = None
    output_mismatch: Optional[str] = None
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

    if step.cache_hydration is not None or step.cache_mode is not None:
        run_kwargs["cache_options"] = CacheOptions(
            cache_hydration=step.cache_hydration,
            cache_mode=step.cache_mode,
        )
    resolved_required_outputs: Optional[Sequence[str]] = step.required_outputs
    if resolved_required_outputs is None and step_meta is not None:
        meta_outputs = getattr(step_meta, "outputs", None)
        if isinstance(meta_outputs, Sequence) and not isinstance(meta_outputs, str):
            resolved_required_outputs = [
                output for output in meta_outputs if isinstance(output, str)
            ]
    if resolved_required_outputs is None:
        outputs_class = STEP_OUTPUTS_CLASSES.get(step.name)
        if outputs_class is not None:
            class_declared_outputs = list(
                declared_outputs_for_step_outputs_class(outputs_class)
            )
            if class_declared_outputs:
                resolved_required_outputs = class_declared_outputs
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
                _update_coupler_from_outputs(
                    outputs, coupler=coupler, workspace=workspace
                )
            continue

        validate_step_ready(spec.name, outputs_holder)
        run_kwargs = _build_step_run_kwargs(
            step=spec,
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
            )
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
            state=state,
            runtime_kwargs=runtime_kwargs,
            stage_name=stage_name,
            default_iteration=iteration,
        )
        coupler_keys = None
        if hasattr(coupler, "keys"):
            try:
                coupler_keys = list(coupler.keys())
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
            )

        if coupler_keys is not None:
            try:
                logger.debug(
                    "[%s] Coupler keys after %s: %s",
                    stage_name,
                    spec.name,
                    list(coupler.keys()),
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
) -> Optional[Any]:
    """
    Best-effort output recovery for cache hits that skip step execution.
    """
    if step_name == "activitysim_preprocess":
        asim_dir = Path(workspace.get_asim_mutable_data_dir())
        candidates = {
            ASIM_HOUSEHOLDS_IN: asim_dir / "households.csv",
            ASIM_PERSONS_IN: asim_dir / "persons.csv",
            ASIM_LAND_USE_IN: asim_dir / "land_use.csv",
            ASIM_OMX_SKIMS: asim_dir / "skims.omx",
        }
        record_store = RecordStore()
        for key, path in candidates.items():
            if path.exists():
                record_store.add_record(
                    FileRecord(
                        file_path=str(path),
                        short_name=key,
                        description=f"Recovered ActivitySim input: {path.name}",
                    )
                )
        if not record_store.all_records():
            return None
    elif step_name == "activitysim_run":
        record_store = RecordStore()
        asim_output_dir = Path(workspace.get_asim_output_dir())
        final_pipeline = asim_output_dir / "final_pipeline"
        allow_final_pipeline = has_asim_run_marker(
            asim_output_dir, state.year, state.iteration
        )
        if final_pipeline.exists() and allow_final_pipeline:
            for child in final_pipeline.iterdir():
                fpath = child / "final.parquet"
                if fpath.is_file():
                    record_store.add_record(
                        FileRecord(
                            file_path=str(fpath),
                            short_name=f"{child.name}_asim_out_temp",
                            description=f"ActivitySim raw output: {child.name}",
                        )
                    )
        elif final_pipeline.exists() and not allow_final_pipeline:
            logger.warning(
                "Skipping ActivitySim final_pipeline cache recovery for year %s iteration %s "
                "because success marker is missing.",
                state.year,
                state.iteration,
            )
        if not record_store.all_records():
            iter_dir = asim_output_dir / f"year-{state.year}-iteration-{state.iteration}"
            if iter_dir.exists():
                for fpath in iter_dir.glob("*.parquet"):
                    record_store.add_record(
                        FileRecord(
                            file_path=str(fpath),
                            short_name=f"{fpath.stem}_asim_out_temp",
                            description=f"ActivitySim raw output: {fpath.stem}",
                        )
                    )
        if not record_store.all_records():
            return None
    elif step_name == "activitysim_postprocess":
        record_store = RecordStore()
        asim_output_dir = Path(workspace.get_asim_output_dir())
        iter_dir = asim_output_dir / f"year-{state.year}-iteration-{state.iteration}"
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
                record_store.add_record(
                    FileRecord(
                        file_path=str(fpath),
                        short_name=short_name,
                        description=f"ActivitySim output file: {fpath.stem}",
                    )
                )
        else:
            return None

        inputs_dir = (
            asim_output_dir / f"inputs-year-{state.year}-iteration-{state.iteration}"
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
                    record_store.add_record(
                        FileRecord(
                            file_path=str(fpath),
                            short_name=short_name,
                            description=f"Archived ActivitySim input: {fname}",
                        )
                    )
            zarr_path = inputs_dir / "skims.zarr"
            if zarr_path.exists():
                record_store.add_record(
                    FileRecord(
                        file_path=str(zarr_path),
                        short_name="asim_input_skims_zarr_archived",
                        description="Archived ActivitySim input: skims.zarr (snapshot)",
                    )
                )

        usim_path = None
        if step_inputs and USIM_DATASTORE_BASE_H5 in step_inputs:
            usim_path = artifact_to_path(
                step_inputs[USIM_DATASTORE_BASE_H5], workspace
            )
        if not usim_path and step_inputs and USIM_DATASTORE_CURRENT_H5 in step_inputs:
            usim_path = artifact_to_path(
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
        if usim_path and os.path.exists(usim_path):
            record_store.add_record(
                FileRecord(
                    file_path=str(usim_path),
                    short_name=f"usim_input_{state.forecast_year}",
                    description="New UrbanSim input data for next iteration",
                )
            )
        if not record_store.all_records():
            return None
    elif step_name == "beam_preprocess":
        record_store = RecordStore()
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
                path = artifact_to_path(value, workspace)
                if path and os.path.exists(path):
                    if key == LINKSTATS_WARMSTART:
                        has_warmstart = True
                    record_store.add_record(
                        FileRecord(
                            file_path=str(path),
                            short_name=key,
                            description=f"Recovered BEAM preprocess input: {key}",
                        )
                    )
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
                path = artifact_to_path(step_inputs.get(key), workspace)
                if path and os.path.exists(path):
                    record_store.add_record(
                        FileRecord(
                            file_path=str(path),
                            short_name=LINKSTATS_WARMSTART,
                            description=(
                                "Recovered BEAM preprocess warmstart from cached "
                                f"input {key}"
                            ),
                        )
                    )
                    has_warmstart = True
                    break
        if not record_store.all_records():
            return None
    else:
        return None

    output_class = STEP_OUTPUTS_CLASSES.get(step_name)
    if output_class is None:
        return None
    outputs = record_store_to_outputs(record_store, output_class, workspace)
    validate = getattr(outputs, "validate", None)
    if callable(validate):
        validate()
    outputs_holder.set_attribute(step_name, outputs)
    _update_coupler_from_record_store(record_store, coupler=coupler, workspace=workspace)
    return outputs


def _update_coupler_from_outputs(
    outputs: Any,
    *,
    coupler: CouplerProtocol,
    workspace: Any,
) -> None:
    to_record_store = getattr(outputs, "to_record_store", None)
    if not callable(to_record_store):
        return
    record_store = to_record_store()
    _update_coupler_from_record_store(record_store, coupler=coupler, workspace=workspace)


def _update_coupler_from_record_store(
    record_store: RecordStore,
    *,
    coupler: CouplerProtocol,
    workspace: Any,
) -> None:
    if record_store is None:
        return
    mapping = record_store.to_mapping()
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
