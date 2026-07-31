"""One native Consist execution and typed-output projection path."""

from __future__ import annotations

from typing import Any, TypeVar

from consist import ExecutionOptions, RunResult, StepIdentity

from pilates.runtime.run_output_archive import archive_completed_run
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

    base_runtime_kwargs = {
        **(runtime_kwargs or {}),
        "settings": settings,
        "state": state,
        "workspace": workspace,
    }
    step_identity: StepIdentity | None = None
    if definition.preflight_identity:
        preflight_options = ExecutionOptions(
            input_binding="paths", runtime_kwargs=base_runtime_kwargs
        )
        step_identity = scenario.resolve_step_identity(
            definition.function,
            year=year,
            iteration=iteration,
            phase=phase,
            stage=stage,
            execution_options=preflight_options,
        )
    resolver_kwargs: dict[str, Any] = {
        "settings": settings,
        "state": state,
        "workspace": workspace,
        "coupler": scenario.coupler,
    }
    if step_identity is not None:
        resolver_kwargs["step_identity"] = step_identity
    resolved = resolved_inputs or definition.resolve_inputs(**resolver_kwargs)
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
            **base_runtime_kwargs,
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
            definition.output_paths(
                settings=settings,
                state=state,
                workspace=workspace,
                resolved_inputs=resolved,
            )
            if definition.output_paths is not None
            else None
        ),
        output_sets=(
            definition.output_sets(
                settings=settings,
                state=state,
                workspace=workspace,
                resolved_inputs=resolved,
            )
            if definition.output_sets is not None
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
        step_identity=step_identity,
    )
    archived_result = archive_completed_run(tracker=scenario.tracker, result=result)
    if archived_result is not result:
        scenario.coupler.update(archived_result.outputs)
        result = archived_result
    return result, definition.project_outputs(
        result.outputs,
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )
