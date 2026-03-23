from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from pilates.config.models import PilatesConfig
from pilates.generic.model_factory import ModelFactory
from pilates.impacts.outputs import (
    ImpactsPostprocessOutputs,
    ImpactsPreprocessOutputs,
    ImpactsRunOutputs,
)
from pilates.workflows.outputs_base import StepOutputsBase, ValidationContext
from pilates.workspace import Workspace

from .shared import (
    CouplerProtocol,
    IMPACTS_EXPOSURE_TABLE,
    IMPACTS_INPUT_MANIFEST,
    IMPACTS_POSTPROCESS_MANIFEST,
    IMPACTS_RUN_MANIFEST,
    StepOutputsHolder,
    WorkflowState,
    _decorate_step_with_consist,
    _declared_outputs_from_class,
    _schema_outputs_from_class,
    _upstream_outputs_view,
    cr,
    log_and_set_output,
    log_input_only,
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


def _execute_impacts_preprocess(
    preprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **_: Any,
) -> ImpactsPreprocessOutputs:
    del outputs_holder
    return preprocessor.preprocess(workspace)


def _execute_impacts_run(
    runner: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **_: Any,
) -> ImpactsRunOutputs:
    upstream = outputs_holder.impacts_preprocess
    if upstream is None:
        raise RuntimeError("Impacts preprocess must complete first")
    if not isinstance(upstream, ImpactsPreprocessOutputs):
        raise TypeError(
            "impacts_run requires ImpactsPreprocessOutputs from impacts_preprocess"
        )
    return runner.run(upstream, workspace)


def _execute_impacts_postprocess(
    postprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    **_: Any,
) -> ImpactsPostprocessOutputs:
    upstream = outputs_holder.impacts_run
    if upstream is None:
        raise RuntimeError("Impacts run must complete first")
    if not isinstance(upstream, ImpactsRunOutputs):
        raise TypeError(
            "impacts_postprocess requires ImpactsRunOutputs from impacts_run"
        )
    return postprocessor.postprocess(upstream, workspace)


def make_impacts_preprocess_step(
    *, coupler: CouplerProtocol, outputs_holder: StepOutputsHolder
) -> Callable[..., None]:
    def _log_outputs(
        outputs: ImpactsPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        del settings
        del state
        del workspace
        del holder
        for source_key, source_path in outputs.staged_inputs.items():
            if source_path:
                log_input_only(
                    key=f"impacts_source_{source_key}",
                    path=source_path,
                    description=f"Impacts upstream source: {source_key}",
                )
        log_output_only(
            key=IMPACTS_INPUT_MANIFEST,
            path=str(outputs.input_manifest),
            description="Impacts staged-input manifest",
        )

    return _make_impacts_typed_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="impacts",
        phase="preprocess",
        outputs_class=ImpactsPreprocessOutputs,
        component_getter=lambda factory, state: factory.get_preprocessor("impacts", state),
        component_executor=_execute_impacts_preprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "impacts_preprocess", outputs
        ),
        output_logger=_log_outputs,
    )


def make_impacts_run_step(
    *, coupler: CouplerProtocol, outputs_holder: StepOutputsHolder
) -> Callable[..., None]:
    def _log_outputs(
        outputs: ImpactsRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        del settings
        del state
        del workspace
        del holder
        log_input_only(
            key=IMPACTS_INPUT_MANIFEST,
            path=str(outputs_holder.impacts_preprocess.input_manifest),
            description="Impacts staged-input manifest consumed by Docker runner",
        )
        log_output_only(
            key=IMPACTS_RUN_MANIFEST,
            path=str(outputs.run_manifest),
            description="Impacts Docker run manifest",
        )

    return _make_impacts_typed_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="impacts",
        phase="run",
        outputs_class=ImpactsRunOutputs,
        component_getter=lambda factory, state: factory.get_runner("impacts", state),
        component_executor=_execute_impacts_run,
        outputs_holder_setter=lambda holder, outputs: setattr(holder, "impacts_run", outputs),
        output_logger=_log_outputs,
    )


def make_impacts_postprocess_step(
    *, coupler: CouplerProtocol, outputs_holder: StepOutputsHolder
) -> Callable[..., None]:
    def _log_outputs(
        outputs: ImpactsPostprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        del settings
        del state
        del workspace
        del holder
        log_input_only(
            key=IMPACTS_RUN_MANIFEST,
            path=str(outputs_holder.impacts_run.run_manifest),
            description="Impacts Docker run manifest consumed by postprocess",
        )
        log_and_set_output(
            key=IMPACTS_EXPOSURE_TABLE,
            path=str(outputs.exposure_table),
            description="Impacts finalized exposure table",
            coupler=coupler,
            profile_file_schema=True,
        )
        log_output_only(
            key=IMPACTS_POSTPROCESS_MANIFEST,
            path=str(outputs.postprocess_manifest),
            description="Impacts postprocess manifest",
        )

    return _make_impacts_typed_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="impacts",
        phase="postprocess",
        outputs_class=ImpactsPostprocessOutputs,
        component_getter=lambda factory, state: factory.get_postprocessor("impacts", state),
        component_executor=_execute_impacts_postprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "impacts_postprocess", outputs
        ),
        output_logger=_log_outputs,
    )
