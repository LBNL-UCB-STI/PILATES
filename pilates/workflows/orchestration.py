from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from pilates.utils.consist_config import build_step_consist_kwargs
from pilates.workflows.step_runner import build_step_config, common_runtime_kwargs
from pilates.workflows.steps import StepOutputsHolder, validate_step_ready

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
