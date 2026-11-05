"""One native Consist execution and typed-output projection path."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
import shutil
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Mapping,
    Protocol,
    TypeVar,
    runtime_checkable,
)
import uuid

from consist import (
    Artifact,
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


@runtime_checkable
class _HasPathAttribute(Protocol):
    path: str | os.PathLike[str]


@runtime_checkable
class _HasWorkspaceFullPath(Protocol):
    full_path: str | os.PathLike[str]


@runtime_checkable
class _HasCurrentConsistInputs(Protocol):
    inputs: list[Artifact]


@runtime_checkable
class _HasEventStartHooks(Protocol):
    _on_start: list[Callable[[Any], None]]


@runtime_checkable
class _SupportsRunStartHookTracker(Protocol):
    is_cached: bool
    current_consist: _HasCurrentConsistInputs | None
    _local_cache_index: Mapping[Any, Any]
    events: _HasEventStartHooks

    def _find_matching_cached_runs(
        self,
        config_hash: str,
        input_hash: str,
        git_hash: str,
        *,
        signature: str | None = None,
    ) -> list[Any]: ...

    def on_run_start(self, callback: Callable[[Any], None]) -> Any: ...
    def resolve_uri(self, uri: str) -> str: ...


@runtime_checkable
class _HasScenarioTracker(Protocol):
    tracker: _SupportsRunStartHookTracker


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


def _resolve_declared_output_destination(
    output_spec: Any,
    *,
    workspace: Any,
) -> Path | None:
    raw_path = (
        output_spec.path if isinstance(output_spec, _HasPathAttribute) else output_spec
    )
    if isinstance(raw_path, os.PathLike):
        path = Path(raw_path)
    elif isinstance(raw_path, str):
        if raw_path.startswith("workspace://"):
            if not isinstance(workspace, _HasWorkspaceFullPath):
                return None
            path = Path(workspace.full_path) / raw_path[len("workspace://") :].lstrip(
                "/"
            )
        elif "://" in raw_path:
            return None
        else:
            path = Path(raw_path)
            if not path.is_absolute():
                if not isinstance(workspace, _HasWorkspaceFullPath):
                    return None
                path = Path(workspace.full_path) / path
    else:
        return None
    return path.resolve()


def _remove_stashed_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


@contextmanager
def _stash_existing_output_destinations(
    *,
    scenario: Any,
    output_paths: Mapping[str, Any] | None,
    workspace: Any,
) -> Iterator[None]:
    """Stash pre-existing declared output destinations during cache-hit hydration.

    Consist's strict requested-output cache admission refuses to overwrite an
    existing file at a declared output destination if its bytes differ from the
    cached candidate (leaving mismatched destinations for the cache-miss path).
    In PILATES, several steps update canonical workspace paths in place across
    steps or years (such as the UrbanSim datastore or BEAM skims). When a step
    lookup finds matching cached candidates in ``begin_run`` (after all inputs
    have already been resolved, logged, and hashed), moving any pre-existing
    output destination aside allows cache-hit hydration to materialize the
    cached output cleanly, while an ``on_run_start`` hook immediately restores
    the pre-existing file if the candidate is rejected.
    """
    if (
        not output_paths
        or not isinstance(scenario, _HasScenarioTracker)
        or not isinstance(scenario.tracker, _SupportsRunStartHookTracker)
    ):
        yield
        return

    tracker = scenario.tracker
    original_find_matching = tracker._find_matching_cached_runs
    stashed: list[tuple[Path, Path]] = []
    modified_artifacts: list[tuple[Artifact, str | None]] = []

    def _find_matching_and_stash(
        config_hash: str,
        input_hash: str,
        git_hash: str,
        *,
        signature: str | None = None,
    ) -> list[Any]:
        tracker._find_matching_cached_runs = original_find_matching
        candidates = original_find_matching(
            config_hash,
            input_hash,
            git_hash,
            signature=signature,
        )
        cache_key = (config_hash, input_hash, git_hash)
        if not candidates and cache_key not in tracker._local_cache_index:
            return candidates
        seen_destinations: set[Path] = set()
        active_inputs = (
            list(tracker.current_consist.inputs)
            if isinstance(tracker.current_consist, _HasCurrentConsistInputs)
            else []
        )
        for spec in output_paths.values():
            destination = _resolve_declared_output_destination(
                spec, workspace=workspace
            )
            if (
                destination is None
                or destination in seen_destinations
                or not destination.exists()
            ):
                continue
            seen_destinations.add(destination)
            stash_path = destination.parent / (
                f".pilates-pre-step-stash-{uuid.uuid4().hex[:8]}-{destination.name}"
            )
            os.replace(destination, stash_path)
            stashed.append((destination, stash_path))
            for artifact in active_inputs:
                resolved_art_path: Path | None = None
                if artifact.abs_path:
                    resolved_art_path = Path(artifact.abs_path).resolve()
                elif artifact.container_uri:
                    uri_str = tracker.resolve_uri(artifact.container_uri)
                    if uri_str:
                        resolved_art_path = Path(uri_str).resolve()
                if resolved_art_path == destination:
                    modified_artifacts.append((artifact, artifact.abs_path))
                    artifact.abs_path = str(stash_path)
        return candidates

    def _cleanup_or_restore() -> None:
        for artifact, original_abs_path in modified_artifacts:
            artifact.abs_path = original_abs_path
        modified_artifacts.clear()
        for destination, stash_path in stashed:
            if not stash_path.exists():
                continue
            if not tracker.is_cached and not destination.exists():
                os.replace(stash_path, destination)
            elif not destination.exists():
                os.replace(stash_path, destination)
            else:
                _remove_stashed_path(stash_path)
        stashed.clear()

    def _on_run_start(_run: Any) -> None:
        if _on_run_start in tracker.events._on_start:
            tracker.events._on_start.remove(_on_run_start)
        tracker._find_matching_cached_runs = original_find_matching
        _cleanup_or_restore()

    tracker._find_matching_cached_runs = _find_matching_and_stash
    tracker.on_run_start(_on_run_start)
    try:
        yield
    finally:
        tracker._find_matching_cached_runs = original_find_matching
        if _on_run_start in tracker.events._on_start:
            tracker.events._on_start.remove(_on_run_start)
        _cleanup_or_restore()


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
    declared_output_paths = (
        definition.output_paths(
            settings=settings,
            state=state,
            workspace=workspace,
            resolved_inputs=resolved,
        )
        if definition.output_paths is not None
        else None
    )
    with _stash_existing_output_destinations(
        scenario=scenario,
        output_paths=declared_output_paths,
        workspace=workspace,
    ):
        result = scenario.run(
            fn=definition.function,
            binding=resolved.binding,
            year=year,
            iteration=iteration,
            phase=phase,
            stage=stage,
            output_paths=declared_output_paths,
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
