from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from pilates.config.models import PilatesConfig
from pilates.generic.model_factory import ModelFactory
from pilates.impacts.outputs import ImpactsPostsimOutputs, ImpactsPresimOutputs
from pilates.workflows.outputs_base import StepOutputsBase, ValidationContext
from pilates.workspace import Workspace

from .shared import (
    CouplerProtocol,
    IMPACTS_EXPOSURE_TABLE,
    IMPACTS_PIPELINE_MANIFEST,
    IMPACTS_POSTPROCESS_MANIFEST,
    IMPACTS_POSTSIM_SETTINGS,
    IMPACTS_PRESIM_SETTINGS,
    StepOutputsHolder,
    WorkflowState,
    _decorate_step_with_consist,
    _declared_outputs_from_class,
    _schema_outputs_from_class,
    _upstream_outputs_view,
    cr,
    log_and_set_output,
    log_output_only,
)

logger = logging.getLogger(__name__)
StepOutputsT = TypeVar("StepOutputsT", bound=StepOutputsBase)


def _make_impacts_typed_step_function(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    model_name: str,
    phase: str,
    outputs_class: Type[StepOutputsT],
    component_getter: Callable[[ModelFactory, WorkflowState], Any],
    component_executor: Callable[..., StepOutputsT],
    outputs_holder_setter: Callable[[StepOutputsHolder, StepOutputsT], None],
    output_logger: Optional[Callable[..., None]] = None,
) -> Callable[..., None]:
    @cr.require_runtime_kwargs("settings", "state", "workspace")
    def _step_func(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        **kwargs: Any,
    ) -> None:
        factory = ModelFactory()
        component = component_getter(factory, state)
        step_outputs = component_executor(
            component,
            workspace,
            outputs_holder,
            coupler=coupler,
            **kwargs,
        )
        if not isinstance(step_outputs, outputs_class):
            raise TypeError(
                f"{model_name}_{phase} must return {outputs_class.__name__}, "
                f"got {type(step_outputs).__name__}"
            )

        step_outputs.validate(
            context=ValidationContext(
                settings=settings,
                state=state,
                workspace=workspace,
                step_name=f"{model_name}_{phase}",
                upstream_outputs=_upstream_outputs_view(
                    outputs_holder,
                    current_step_name=f"{model_name}_{phase}",
                ),
            )
        )
        outputs_holder_setter(outputs_holder, step_outputs)
        if output_logger is not None:
            output_logger(step_outputs, settings, state, workspace, outputs_holder)
        logger.info("%s %s completed successfully", model_name, phase)

    return _decorate_step_with_consist(
        step_func=_step_func,
        step_model=f"{model_name}_{phase}",
        description=f"{model_name} {phase} workflow step",
        schema_outputs=_schema_outputs_from_class(outputs_class),
        outputs=_declared_outputs_from_class(outputs_class),
        tags=[model_name, phase],
    )


def _execute_impacts_presim(
    runner: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **_: Any,
) -> ImpactsPresimOutputs:
    del outputs_holder
    return runner.run(None, workspace)


def _execute_impacts_postsim(
    runner: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **_: Any,
) -> ImpactsPostsimOutputs:
    del outputs_holder
    return runner.run(None, workspace)


def make_impacts_presim_step(
    *, coupler: CouplerProtocol, outputs_holder: StepOutputsHolder
) -> Callable[..., None]:
    def _log_outputs(
        outputs: ImpactsPresimOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        del settings
        del state
        del workspace
        del holder
        log_output_only(
            key=IMPACTS_PRESIM_SETTINGS,
            path=str(outputs.settings_path),
            description="Derived IMPACTS presim settings file",
        )

    return _make_impacts_typed_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="impacts",
        phase="presim",
        outputs_class=ImpactsPresimOutputs,
        component_getter=lambda factory, state: factory.get_runner("impacts_presim", state),
        component_executor=_execute_impacts_presim,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "impacts_presim", outputs
        ),
        output_logger=_log_outputs,
    )


def make_impacts_postsim_step(
    *, coupler: CouplerProtocol, outputs_holder: StepOutputsHolder
) -> Callable[..., None]:
    def _log_outputs(
        outputs: ImpactsPostsimOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        del settings
        del state
        del workspace
        del holder
        log_output_only(
            key=IMPACTS_POSTSIM_SETTINGS,
            path=str(outputs.settings_path),
            description="Derived IMPACTS postsim settings file",
        )
        log_output_only(
            key=IMPACTS_PIPELINE_MANIFEST,
            path=str(outputs.pipeline_manifest),
            description="Impacts pipeline manifest",
        )
        log_output_only(
            key=IMPACTS_POSTPROCESS_MANIFEST,
            path=str(outputs.postprocess_manifest),
            description="Impacts postprocess manifest",
        )
        log_and_set_output(
            key=IMPACTS_EXPOSURE_TABLE,
            path=str(outputs.exposure_table),
            description="Impacts finalized exposure table",
            coupler=coupler,
            profile_file_schema=True,
        )

    return _make_impacts_typed_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="impacts",
        phase="postsim",
        outputs_class=ImpactsPostsimOutputs,
        component_getter=lambda factory, state: factory.get_runner("impacts_postsim", state),
        component_executor=_execute_impacts_postsim,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "impacts_postsim", outputs
        ),
        output_logger=_log_outputs,
    )
