from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Set

from pilates.utils.consist_config import build_step_consist_kwargs
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
        coupler: Any,
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

            scenario.run(**step_config.to_kwargs())

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
    coupler: Any,
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
                outputs = _restore_outputs_from_manifest(
                    spec.name, manifest, workspace
                )
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
