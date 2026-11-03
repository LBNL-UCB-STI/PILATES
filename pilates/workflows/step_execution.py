"""One native Consist execution and typed-output projection path."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from consist import (
    CacheOptions,
    ExecutionOptions,
    OutputPolicyOptions,
    RunCompletionArchivePolicy,
    RunResult,
    StepIdentity,
)

from pilates.runtime.archive_paths import archive_roots
from pilates.workflows.contracts import StepDefinition
from pilates.workflows.resolved_inputs import ResolvedStepInputs

OutputT = TypeVar("OutputT")

CodeIdentityMode = Literal["repo_git", "callable_module", "callable_source"]


@runtime_checkable
class _RunCodeIdentityConfig(Protocol):
    consist_code_identity: CodeIdentityMode | None


@runtime_checkable
class _SettingsWithRunCodeIdentity(Protocol):
    run: _RunCodeIdentityConfig


def _resolve_cache_options(
    *,
    definition: StepDefinition[Any],
    settings: Any,
    state: Any,
    workspace: Any,
) -> CacheOptions | None:
    cache_options = (
        definition.cache_options(settings=settings, state=state, workspace=workspace)
        if definition.cache_options is not None
        else None
    )
    configured_code_identity: CodeIdentityMode | None = None
    if isinstance(settings, _SettingsWithRunCodeIdentity) and isinstance(
        settings.run, _RunCodeIdentityConfig
    ):
        configured_code_identity = settings.run.consist_code_identity
    if configured_code_identity is None:
        return cache_options
    if cache_options is None:
        return CacheOptions(code_identity=configured_code_identity)
    if cache_options.code_identity is not None:
        return cache_options
    return replace(cache_options, code_identity=configured_code_identity)


def _configured_step_archive_root(
    *,
    definition: StepDefinition[Any],
    stage: str,
    year: int | None,
    iteration: int | None,
    phase: str | None,
) -> Path | None:
    if not definition.archive_outputs:
        return None
    if os.environ.get("PILATES_ENABLE_ARCHIVE_COPY", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    roots = archive_roots()
    if roots is None:
        return None
    local_root, archive_root = roots
    if os.path.normcase(os.path.normpath(local_root)) == os.path.normcase(
        os.path.normpath(archive_root)
    ):
        return None
    parts = [
        definition.name,
        stage,
        f"y{year}" if year is not None else None,
        f"i{iteration}" if iteration is not None else None,
        phase,
    ]
    scope = "__".join(part for part in parts if part)
    return (Path(archive_root) / "consist-recovery" / scope).resolve()


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
    resolved = (
        resolved_inputs or definition.resolve_inputs(**resolver_kwargs)
    ).with_input_contract(definition.input_contract)
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
    options = replace(
        options,
        runtime_kwargs={
            **(options.runtime_kwargs or {}),
            **base_runtime_kwargs,
        },
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
        cache_options=_resolve_cache_options(
            definition=definition,
            settings=settings,
            state=state,
            workspace=workspace,
        ),
        output_policy=OutputPolicyOptions(
            output_missing="error",
            run_completion_archive=(
                RunCompletionArchivePolicy(archive_root=archive_root, mode="copy")
                if (
                    archive_root := _configured_step_archive_root(
                        definition=definition,
                        stage=stage,
                        year=year,
                        iteration=iteration,
                        phase=phase,
                    )
                )
                is not None
                else None
            ),
        ),
        execution_options=options,
        step_identity=step_identity,
    )
    return result, definition.project_outputs(
        result.outputs,
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )
