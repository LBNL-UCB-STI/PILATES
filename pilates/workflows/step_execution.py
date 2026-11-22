"""One native Consist execution and typed-output projection path."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, TypeVar, overload

from consist import (
    BindingResult,
    CacheOptions,
    ExecutionOptions,
    OutputPolicyOptions,
    RunCompletionArchivePolicy,
    RunResult,
)

from pilates.utils.archive_paths import ArchiveRunDirLike, workspace_storage_roots
from pilates.utils.settings_helper import get as get_setting
from pilates.utils.state_access import iteration_index
from pilates.workflows.contracts import StepCallInputs, StepDefinition
from pilates.workflows.sim_state import ArtifactRef, SimState

if TYPE_CHECKING:
    from pilates.config.models import SettingsLike
    from pilates.workflows.dry_run import (
        DryRunRecorder,
        SymbolicOutputs,
        SymbolicRunResult,
    )

OutputT = TypeVar("OutputT")

CodeIdentityMode = Literal["repo_git", "callable_module", "callable_source"]
_UNSET_ITERATION: Any = object()


class ScenarioDriftError(RuntimeError):
    """Raised when strict resume detects cache drift on a previously completed step."""


def _is_step_archive_enabled(
    *,
    settings: SettingsLike,
    state: Any,
    workspace: Any,
) -> bool:
    enable_archive_copy = get_setting(settings, "run.enable_archive_copy")
    if isinstance(enable_archive_copy, bool):
        return enable_archive_copy
    return bool(
        (isinstance(workspace, ArchiveRunDirLike) and workspace.archive_run_dir)
        or (isinstance(state, ArchiveRunDirLike) and state.archive_run_dir)
    )


def _resolve_cache_options(
    *,
    definition: StepDefinition[Any],
    settings: SettingsLike,
    state: Any,
    workspace: Any,
) -> CacheOptions | None:
    cache_options = (
        definition.cache_options(settings=settings, state=state, workspace=workspace)
        if definition.cache_options is not None
        else None
    )
    configured_code_identity: CodeIdentityMode | None = get_setting(
        settings, "run.consist_code_identity"
    )
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
    settings: SettingsLike,
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
    main_forecast_year = getattr(state, "main_forecast_year", None)
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


@overload
def execute_step(
    *,
    scenario: Any,
    definition: StepDefinition[OutputT],
    settings: SettingsLike,
    state: Any,
    workspace: Any,
    stage: str,
    year: int | None,
    iteration: int | None = _UNSET_ITERATION,
    phase: str | None = None,
    sim_state: SimState | None = None,
    prior_outputs: Mapping[str, ArtifactRef] | None = None,
    runtime_kwargs: dict[str, Any] | None = None,
    step_inputs: StepCallInputs | None = None,
    plan_recorder: None = None,
) -> tuple[RunResult, OutputT]: ...


@overload
def execute_step(
    *,
    scenario: Any,
    definition: StepDefinition[OutputT],
    settings: SettingsLike,
    state: Any,
    workspace: Any,
    stage: str,
    year: int | None,
    iteration: int | None = _UNSET_ITERATION,
    phase: str | None = None,
    sim_state: SimState | None = None,
    prior_outputs: Mapping[str, ArtifactRef] | None = None,
    runtime_kwargs: dict[str, Any] | None = None,
    step_inputs: StepCallInputs | None = None,
    plan_recorder: DryRunRecorder,
) -> tuple[SymbolicRunResult, SymbolicOutputs]: ...


def execute_step(
    *,
    scenario: Any,
    definition: StepDefinition[OutputT],
    settings: SettingsLike,
    state: Any,
    workspace: Any,
    stage: str,
    year: int | None,
    iteration: int | None = _UNSET_ITERATION,
    phase: str | None = None,
    sim_state: SimState | None = None,
    prior_outputs: Mapping[str, ArtifactRef] | None = None,
    runtime_kwargs: dict[str, Any] | None = None,
    step_inputs: StepCallInputs | None = None,
    plan_recorder: DryRunRecorder | None = None,
) -> tuple[RunResult | SymbolicRunResult, OutputT | SymbolicOutputs]:
    """Resolve once, run once, then project persisted ``RunResult.outputs`` once."""

    resolved_iteration: int | None = (
        iteration_index(state) if iteration is _UNSET_ITERATION else iteration
    )
    resolver_kwargs: dict[str, Any] = {
        "settings": settings,
        "state": state,
        "workspace": workspace,
        "sim_state": sim_state if sim_state is not None else SimState(),
    }
    if prior_outputs is not None:
        resolver_kwargs["prior_outputs"] = prior_outputs
    resolved_step_inputs = (
        step_inputs
        if step_inputs is not None
        else definition.resolve_inputs(**resolver_kwargs)
    )
    resolved_step_inputs.require_complete(definition.name)

    if plan_recorder is not None:
        from pilates.workflows.dry_run import (
            SymbolicArtifact,
            SymbolicOutputs,
            SymbolicRun,
            SymbolicRunResult,
        )

        output_keys: list[str] = []
        declared_paths: dict[str, Path] = {}
        output_fn = (
            definition.outputs
            if definition.outputs is not None
            else definition.output_paths
        )
        if output_fn is not None:
            declared = output_fn(
                settings=settings,
                state=state,
                workspace=workspace,
                step_inputs=resolved_step_inputs,
            )
            for key, val in declared.items():
                output_keys.append(key)
                if isinstance(val, (Path, str, os.PathLike)):
                    declared_paths[key] = Path(val)
        if definition.output_sets is not None:
            declared_sets = definition.output_sets(
                settings=settings,
                state=state,
                workspace=workspace,
                step_inputs=resolved_step_inputs,
            )
            output_keys.extend(declared_sets.keys())
        deduped_output_keys = tuple(dict.fromkeys(output_keys))

        step_id = plan_recorder.record_step(
            step_name=definition.name,
            year=year if year is not None else 0,
            iteration=resolved_iteration,
            inputs=resolved_step_inputs.inputs,
            output_keys=deduped_output_keys,
        )
        symbolic_outputs = {
            key: SymbolicArtifact(
                key=key,
                producer_step=step_id,
                path=declared_paths.get(key, Path("<symbolic>")),
            )
            for key in deduped_output_keys
        }
        return (
            SymbolicRunResult(run=SymbolicRun(id=step_id), outputs=symbolic_outputs),
            SymbolicOutputs(symbolic_outputs),
        )

    base_runtime_kwargs = {
        **(runtime_kwargs or {}),
        "settings": settings,
        "state": state,
        "workspace": workspace,
    }
    # Track execution ordinal and check strict resume drift expectations
    current_ordinal = 0
    step_counter = getattr(workspace, "step_execution_counter", None)
    expected_prefix = getattr(workspace, "expected_prefix_steps", None)
    if isinstance(step_counter, int) and isinstance(expected_prefix, int):
        workspace.step_execution_counter += 1
        current_ordinal = workspace.step_execution_counter
        resume_policy = get_setting(settings, "run.resume_policy", "strict")

        if (
            resume_policy == "strict"
            and expected_prefix > 0
            and current_ordinal <= expected_prefix
        ):
            import functools
            import consist as cr

            original_func = definition.function

            @functools.wraps(original_func)
            def _drift_guard_wrapper(*args, **kwargs):
                explanation = None
                try:
                    current_run = cr.current_run()
                    meta = getattr(current_run, "meta", None)
                    if isinstance(meta, dict):
                        explanation = meta.get("cache_miss_explanation")
                except Exception:
                    pass
                msg = (
                    f"Strict-resume drift detected in step '{definition.name}': expected a cache hit "
                    f"for previously completed step run, but got a cache miss."
                )
                if explanation:
                    msg += f" Miss explanation: {explanation}"
                raise ScenarioDriftError(msg)

            # Using dataclasses.replace to update definition if it's a dataclass,
            # or just creating a new instance if needed. Since definition is a StepDefinition
            # let's replace it properly.
            definition = replace(definition, function=_drift_guard_wrapper)

    binding = BindingResult(inputs=dict(resolved_step_inputs.inputs))
    options = (
        definition.execution_options(
            settings=settings,
            state=state,
            workspace=workspace,
            step_inputs=resolved_step_inputs,
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
    step_archive_root = _configured_step_archive_root(
        definition=definition,
        settings=settings,
        state=state,
        workspace=workspace,
        stage=stage,
        year=year,
        iteration=resolved_iteration,
        phase=phase,
    )
    result = scenario.run(
        fn=definition.function,
        binding=binding,
        config_plan_ingest=(
            definition.name
            not in (
                "activitysim_preprocess",
                "activitysim_postprocess",
                "beam_preprocess",
                "beam_postprocess",
            )
        ),
        year=year,
        iteration=resolved_iteration,
        phase=phase,
        stage=stage,
        outputs=(
            definition.outputs(
                settings=settings,
                state=state,
                workspace=workspace,
                step_inputs=resolved_step_inputs,
            )
            if definition.outputs is not None
            else None
        ),
        output_paths=(
            definition.output_paths(
                settings=settings,
                state=state,
                workspace=workspace,
                step_inputs=resolved_step_inputs,
            )
            if definition.outputs is None and definition.output_paths is not None
            else None
        ),
        output_sets=(
            definition.output_sets(
                settings=settings,
                state=state,
                workspace=workspace,
                step_inputs=resolved_step_inputs,
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
                RunCompletionArchivePolicy(archive_root=step_archive_root, mode="copy")
                if step_archive_root is not None
                else None
            ),
        ),
        execution_options=options,
    )
    is_hit = bool(getattr(result, "cache_hit", False))
    if not is_hit and isinstance(getattr(workspace, "db_mutation_seq", None), int):
        workspace.db_mutation_seq += 1
    if step_archive_root is not None and not is_hit:
        on_archive_commit = getattr(workspace, "on_archive_commit", None)
        if on_archive_commit:
            on_archive_commit(definition.name)
    return result, definition.project_outputs(
        result.outputs,
        settings=settings,
        state=state,
        workspace=workspace,
        step_inputs=resolved_step_inputs,
    )
