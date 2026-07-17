"""One native Consist execution and typed-output projection path."""

from __future__ import annotations

from typing import Any, TypeVar

from consist import ExecutionOptions, RunResult

from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.step_definition import StepDefinition

OutputT = TypeVar("OutputT")


def execute_step(
    *,
    scenario: Any,
    definition: StepDefinition[OutputT],
    settings: Any,
    state: Any,
    workspace: Any,
    stage: str,
    year: int | None,
    iteration: int | None,
    phase: str | None,
    runtime_kwargs: dict[str, Any] | None = None,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> tuple[RunResult, OutputT]:
    """Resolve once, run once, then project persisted ``RunResult.outputs`` once."""

    resolved = resolved_inputs or definition.resolve_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=scenario.coupler,
    )
    resolved.require_complete()
    options = (
        definition.execution_options(
            settings=settings,
            state=state,
            workspace=workspace,
            resolved_inputs=resolved,
        )
        if definition.execution_options is not None
        else ExecutionOptions(input_binding="paths")
    )
    options = ExecutionOptions(
        load_inputs=options.load_inputs,
        input_binding=options.input_binding,
        input_paths=options.input_paths,
        input_materialization=options.input_materialization,
        input_materialization_mode=options.input_materialization_mode,
        executor=options.executor,
        container=options.container,
        runtime_kwargs={
            **(options.runtime_kwargs or {}),
            **(runtime_kwargs or {}),
            "settings": settings,
            "state": state,
            "workspace": workspace,
        },
        inject_context=options.inject_context,
    )
    result = scenario.run(
        fn=definition.function,
        binding=resolved.binding,
        year=year,
        iteration=iteration,
        phase=phase,
        stage=stage,
        output_paths=(
            definition.output_paths(settings=settings, state=state, workspace=workspace)
            if definition.output_paths is not None
            else None
        ),
        cache_options=(
            definition.cache_options(
                settings=settings, state=state, workspace=workspace
            )
            if definition.cache_options is not None
            else None
        ),
        execution_options=options,
    )
    return result, definition.project_outputs(
        result.outputs,
        settings=settings,
        state=state,
        workspace=workspace,
    )
