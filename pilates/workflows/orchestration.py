from __future__ import annotations

import logging
from dataclasses import dataclass
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Set

from pilates.generic.records import FileRecord, RecordStore
from pilates.utils.coupler_helpers import artifact_to_path, record_store_to_outputs
from pilates.workflows.artifact_constants import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_H5,
)
from pilates.utils.consist_config import build_step_consist_kwargs
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.step_manifest import load_step_manifest, save_step_manifest
from pilates.workflows.outputs_base import (
    deserialize_step_outputs,
    serialize_step_outputs,
)
from pilates.workflows.step_runner import build_step_config, common_runtime_kwargs
from pilates.workflows.steps import (
    STEP_OUTPUTS_CLASSES,
    StepOutputsHolder,
    validate_step_ready,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowStepSpec:
    """
    Declarative specification for a workflow step invocation.
    """

    name: str
    step_func: Callable[..., None]
    input_keys: Optional[Sequence[str]] = None
    inputs: Optional[Dict[str, Any]] = None
    output_paths: Optional[Dict[str, Any]] = None
    cache_hydration: str = "inputs-missing"
    cache_mode: Optional[str] = None
    load_inputs: bool = False


@dataclass
class WorkflowStage:
    """
    Orchestrates a sequence of related steps.
    """

    name: str
    stage_type: Any
    steps: Sequence[WorkflowStepSpec]

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
        runtime_kwargs = common_runtime_kwargs(
            settings=settings,
            state=state,
            workspace=workspace,
            **(runtime_kwargs_extra or {}),
        )

        for spec in self.steps:
            validate_step_ready(spec.name, outputs_holder)
            step_config = build_step_config(
                fn=spec.step_func,
                name=f"{spec.name}_{name_suffix}",
                model=spec.name,
                state=state,
                iteration=iteration,
                inputs=spec.inputs or None,
                input_keys=spec.input_keys or None,
                output_paths=spec.output_paths or None,
                cache_hydration=spec.cache_hydration,
                cache_mode=spec.cache_mode,
                load_inputs=spec.load_inputs,
                runtime_kwargs=runtime_kwargs,
                consist_kwargs=build_step_consist_kwargs(
                    spec.name, settings, workspace_path=workspace.full_path
                ),
            )

            coupler_keys = None
            if hasattr(coupler, "keys"):
                try:
                    coupler_keys = list(coupler.keys())
                except TypeError:
                    coupler_keys = None
            logger.debug(
                "[%s] StepConfig for %s: inputs=%s input_keys=%s outputs=%s",
                self.name,
                spec.name,
                bool(spec.inputs),
                spec.input_keys,
                spec.output_paths,
            )
            if coupler_keys is not None:
                logger.debug(
                    "[%s] Coupler keys before %s: %s",
                    self.name,
                    spec.name,
                    coupler_keys,
                )

            result = scenario.run(**step_config.to_kwargs())
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
                        self.name,
                        spec.name,
                        list(coupler.keys()),
                    )
                except TypeError:
                    logger.debug(
                        "[%s] Coupler keys after %s: <unavailable>",
                        self.name,
                        spec.name,
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
    steps: Sequence[WorkflowStepSpec],
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

    for spec in steps:
        if spec.name in manifest:
            logger.info("[%s] %s already completed (skipping)", stage_name, spec.name)
            if outputs_holder.get_attribute(spec.name) is None:
                outputs = _restore_outputs_from_manifest(spec.name, manifest, workspace)
                if outputs is not None:
                    outputs_holder.set_attribute(spec.name, outputs)
            continue

        validate_step_ready(spec.name, outputs_holder)
        step_config = build_step_config(
            fn=spec.step_func,
            name=f"{spec.name}_{name_suffix}",
            model=spec.name,
            state=state,
            iteration=iteration,
            inputs=spec.inputs or None,
            input_keys=spec.input_keys or None,
            output_paths=spec.output_paths or None,
            cache_hydration=spec.cache_hydration,
            cache_mode=spec.cache_mode,
            load_inputs=spec.load_inputs,
            runtime_kwargs=runtime_kwargs,
            consist_kwargs=build_step_consist_kwargs(
                spec.name, settings, workspace_path=workspace.full_path
            ),
        )

        result = scenario.run(**step_config.to_kwargs())
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
        if final_pipeline.exists():
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
            required_outputs = {"persons", "households", "beam_plans"}
            available_outputs = {path.stem for path in iter_dir.glob("*.parquet")}
            if not required_outputs.issubset(available_outputs):
                return None
            for fpath in iter_dir.glob("*.parquet"):
                record_store.add_record(
                    FileRecord(
                        file_path=str(fpath),
                        short_name=fpath.stem,
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
        if step_inputs and USIM_DATASTORE_H5 in step_inputs:
            usim_path = artifact_to_path(step_inputs[USIM_DATASTORE_H5], workspace)
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
                    record_store.add_record(
                        FileRecord(
                            file_path=str(path),
                            short_name=key,
                            description=f"Recovered BEAM preprocess input: {key}",
                        )
                    )
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
    return outputs


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
