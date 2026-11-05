"""One native Consist execution and typed-output projection path."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, TypeVar, runtime_checkable

from consist import (
    CacheOptions,
    ExecutionOptions,
    OutputPolicyOptions,
    RunCompletionArchivePolicy,
    RunResult,
    StepIdentity,
)

from pilates.utils.archive_paths import ArchiveRunDirLike, workspace_storage_roots
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


@runtime_checkable
class _HasEnableArchiveCopy(Protocol):
    enable_archive_copy: bool


@runtime_checkable
class _HasRunConfig(Protocol):
    run: Any


@runtime_checkable
class _HasMainForecastYear(Protocol):
    main_forecast_year: int | None


def _is_step_archive_enabled(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
) -> bool:
    if isinstance(settings, Mapping):
        run_cfg = settings.get("run")
        if isinstance(run_cfg, Mapping) and isinstance(
            run_cfg.get("enable_archive_copy"), bool
        ):
            return run_cfg["enable_archive_copy"]
    elif (
        isinstance(settings, _HasRunConfig)
        and isinstance(settings.run, _HasEnableArchiveCopy)
        and isinstance(settings.run.enable_archive_copy, bool)
    ):
        return settings.run.enable_archive_copy
    return bool(
        (isinstance(workspace, ArchiveRunDirLike) and workspace.archive_run_dir)
        or (isinstance(state, ArchiveRunDirLike) and state.archive_run_dir)
    )


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
    settings: Any,
    state: Any,
    workspace: Any,
    stage: str,
    year: int | None,
    iteration: int | None,
    phase: str | None,
) -> Path | None:
    if not definition.archive_outputs:
        return None
    if not _is_step_archive_enabled(
        settings=settings,
        state=state,
        workspace=workspace,
    ):
        return None
    roots = workspace_storage_roots(workspace=workspace, state=state)
    if roots is None:
        return None
    local_root, archive_root = roots
    if os.path.normcase(os.path.normpath(local_root)) == os.path.normcase(
        os.path.normpath(archive_root)
    ):
        return None
    main_forecast_year = (
        state.main_forecast_year if isinstance(state, _HasMainForecastYear) else None
    )
    parts = [
        definition.name,
        stage,
        f"y{year}" if year is not None else None,
        (
            f"fy{main_forecast_year}"
            if main_forecast_year is not None and main_forecast_year != year
            else None
        ),
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
            run_completion_archive=(
                RunCompletionArchivePolicy(archive_root=archive_root, mode="copy")
                if (
                    archive_root := _configured_step_archive_root(
                        definition=definition,
                        settings=settings,
                        state=state,
                        workspace=workspace,
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
