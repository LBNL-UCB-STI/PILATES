from __future__ import annotations

from typing import Any, Mapping

from consist import (
    BindingResult,
    CacheOptions,
    ExecutionOptions,
    define_step,
    require_runtime_kwargs,
)

from pilates.config.models import PilatesConfig
from pilates.workspace import Workspace
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.step_definition import StepDefinition
from workflow_state import WorkflowState


@define_step(
    model="postprocessing",
    name_template="postprocessing__y{year}__i{iteration}__phase_{phase}",
    input_binding="paths",
    tags=["postprocessing"],
)
@require_runtime_kwargs("settings", "state", "workspace")
def _postprocessing_callable(
    *, settings: PilatesConfig, state: WorkflowState, workspace: Workspace
) -> None:
    if "postprocessing" in settings:
        from pilates.postprocessing.postprocessor import (
            copy_outputs_to_mep,
            process_event_file,
        )

        process_event_file(settings, state.forecast_year, state.current_inner_iter)
        copy_outputs_to_mep(
            settings, state.forecast_year, state.current_inner_iter, workspace
        )


def _resolve_postprocessing_inputs(**_: Any) -> ResolvedStepInputs:
    return ResolvedStepInputs(
        step_name="postprocessing", binding=BindingResult(inputs={})
    )


def _project_postprocessing_outputs(outputs: Mapping[str, Any], **_: Any) -> None:
    if outputs:
        raise RuntimeError("postprocessing does not declare typed outputs")
    return None


postprocessing = StepDefinition(
    name="postprocessing",
    function=_postprocessing_callable,
    resolve_inputs=_resolve_postprocessing_inputs,
    project_outputs=_project_postprocessing_outputs,
    execution_options=lambda **_: ExecutionOptions(input_binding="paths"),
    cache_options=lambda **_: CacheOptions(cache_mode="overwrite"),
)
