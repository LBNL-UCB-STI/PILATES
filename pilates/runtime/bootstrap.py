from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Mapping, Optional

from consist.types import CacheOptions, OutputArtifactSpec

from pilates.config import PilatesConfig
from pilates.generic.initialization import (
    Initialization,
    build_bootstrap_artifact_summary,
)
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.io import get_activity_demand_model, get_traffic_assignment_model
from pilates.workflows.binding import bootstrap_stage_boundary_durability_policy
from pilates.workspace import Workspace

logger = logging.getLogger(__name__)


def is_bootstrap_cache_enabled(settings: PilatesConfig) -> bool:
    run_cfg = getattr(settings, "run", None)
    return bool(getattr(run_cfg, "bootstrap_cache_enabled", True))


def _bootstrap_cache_options(
    settings: PilatesConfig,
    *,
    cache_options_cls: type[CacheOptions],
    cache_mode: Optional[str] = None,
    cache_hydration: Optional[str] = None,
) -> Optional[CacheOptions]:
    run_cfg = getattr(settings, "run", None)
    code_identity = getattr(run_cfg, "consist_code_identity", None)
    if cache_mode is None and code_identity is None and cache_hydration is None:
        return None
    return cache_options_cls(
        cache_mode=cache_mode,
        code_identity=code_identity,
        cache_hydration=cache_hydration,
    )


def _warn_on_bootstrap_fast_hashing(settings: PilatesConfig) -> None:
    run_cfg = getattr(settings, "run", None)
    if not bool(getattr(run_cfg, "bootstrap_cache_enabled", True)):
        return
    hashing_strategy = str(getattr(run_cfg, "consist_hashing_strategy", "fast")).lower()
    if hashing_strategy != "fast":
        return
    logger.warning(
        "Bootstrap cache is enabled with fast hashing. Initialization stages copied "
        "files whose identity may change on restage due to mtime-sensitive hashing, "
        "which can force downstream cache misses. Prefer "
        "run.consist_hashing_strategy: full."
    )


def build_bootstrap_run_reference(
    *,
    probe_run_id: Optional[str] = None,
) -> Dict[str, str]:
    reference: Dict[str, str] = {}
    if probe_run_id:
        reference["probe_run_id"] = probe_run_id
    return reference


def log_bootstrap_result_summary(
    bootstrap_result: Optional[Mapping[str, Any]],
    *,
    log: Optional[logging.Logger] = None,
) -> None:
    if bootstrap_result is None:
        return

    output_logger = log or logger
    cache_miss_explanation = bootstrap_result.get("cache_miss_explanation")
    if isinstance(cache_miss_explanation, dict):
        output_logger.info(
            "Bootstrap phase complete: cache_hit=%s probe_hit=%s "
            "replay_hydration_complete=%s run_ref=%s summary=%s "
            "cache_miss_reason=%s cache_miss_candidate_run_id=%s",
            bootstrap_result.get("bootstrap_cache_hit"),
            bootstrap_result.get("cache_probe_hit"),
            bootstrap_result.get("replay_hydration_complete"),
            bootstrap_result.get("run_reference"),
            bootstrap_result.get("staged_artifact_summary"),
            cache_miss_explanation.get("reason"),
            cache_miss_explanation.get("candidate_run_id"),
        )
        return

    output_logger.info(
        "Bootstrap phase complete: cache_hit=%s probe_hit=%s "
        "replay_hydration_complete=%s run_ref=%s summary=%s",
        bootstrap_result.get("bootstrap_cache_hit"),
        bootstrap_result.get("cache_probe_hit"),
        bootstrap_result.get("replay_hydration_complete"),
        bootstrap_result.get("run_reference"),
        bootstrap_result.get("staged_artifact_summary"),
    )


def _bootstrap_output_paths(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
) -> Dict[str, OutputArtifactSpec]:
    """Declare immutable snapshots of the bootstrap-owned workspace roots.

    The initialization phase mutates these directories as a unit.  They must
    therefore be logged as Consist directory artifacts rather than ordinary
    file outputs: central archival validates and recovers their complete tree,
    including empty directories, and cache replay restores each declared root.
    Exact files remain bootstrap *invariants* for post-hydration validation;
    declaring them again would duplicate bytes already owned by a root.
    """

    output_paths: Dict[str, OutputArtifactSpec] = {}
    run_models = getattr(getattr(settings, "run", None), "models", None)

    def declare_directory(key: str, path: str) -> None:
        output_paths[key] = OutputArtifactSpec(
            path=path,
            meta={
                "directory_artifact": True,
                "directory_allow_extra_files": True,
            },
        )

    get_usim_data_dir = getattr(workspace, "get_usim_mutable_data_dir", None)
    if callable(get_usim_data_dir) and (
        getattr(run_models, "land_use", None) == "urbansim"
        or getattr(run_models, "activity_demand", None) == "activitysim"
        or getattr(run_models, "vehicle_ownership", None) == "atlas"
    ):
        declare_directory("urbansim_mutable_data_dir", get_usim_data_dir())

    if get_activity_demand_model(settings) == "activitysim":
        get_asim_data_dir = getattr(workspace, "get_asim_mutable_data_dir", None)
        if callable(get_asim_data_dir):
            declare_directory("activitysim_mutable_data_dir", get_asim_data_dir())
        get_asim_configs_dir = getattr(workspace, "get_asim_mutable_configs_dir", None)
        if callable(get_asim_configs_dir):
            declare_directory("activitysim_mutable_configs_dir", get_asim_configs_dir())

    if getattr(run_models, "vehicle_ownership", None) == "atlas":
        get_atlas_input_dir = getattr(workspace, "get_atlas_mutable_input_dir", None)
        if callable(get_atlas_input_dir):
            declare_directory("atlas_mutable_input_dir", get_atlas_input_dir())

    if get_traffic_assignment_model(settings) == "beam":
        get_beam_input_dir = getattr(workspace, "get_beam_mutable_data_dir", None)
        if callable(get_beam_input_dir):
            declare_directory("beam_mutable_data_dir", get_beam_input_dir())

    return output_paths


def seed_bootstrap_artifacts_to_coupler(
    *,
    settings: PilatesConfig,
    state: Any,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> None:
    """
    Seed coupler keys for bootstrap-staged artifacts needed by later steps.

    The stage-boundary durability policy lives in the binding layer so runtime
    bootstrap can publish bootstrap-safe files without reconstructing the
    artifact inventory locally.
    """
    get_value = getattr(coupler, "get", None)

    for rule in bootstrap_stage_boundary_durability_policy():
        resolved_artifacts = rule.resolve(
            settings=settings,
            state=state,
            workspace=workspace,
        )
        if not resolved_artifacts:
            continue
        for key, path in resolved_artifacts.items():
            if callable(get_value) and get_value(key) is not None:
                continue
            if not path or not os.path.exists(path):
                continue
            coupler.set(key, path)


def run_bootstrap_phase(
    *,
    tracker: Any,
    settings: PilatesConfig,
    state: Any,
    workspace: Workspace,
    scenario_id: str,
    seed: Optional[int],
    initialization_cls: type[Initialization] = Initialization,
    build_bootstrap_artifact_summary_fn: Callable[
        ..., Dict[str, Any]
    ] = build_bootstrap_artifact_summary,
    build_step_consist_kwargs_fn: Callable[..., Dict[str, Any]],
    merge_tag_list_fn: Callable[..., list[str]],
    merge_epoch_facet_fn: Callable[..., Dict[str, Any]],
    cache_options_cls: type[CacheOptions] = CacheOptions,
) -> Dict[str, Any]:
    """Execute initialization in a dedicated pre-scenario bootstrap phase.

    Bootstrap snapshots remain local cache outputs.  They are reproducible
    workspace staging inputs rather than completed native-model products, so
    they deliberately do not enter the recovery-copy archive boundary.  A
    later cache replay that cannot restore their workspace invariants falls
    back to this initialization run.
    """
    _warn_on_bootstrap_fast_hashing(settings)
    staged_artifact_summary: Dict[str, Any] = {}

    def _execute_initialization() -> None:
        nonlocal staged_artifact_summary
        init_model = initialization_cls("initialization", state)
        copied_records = init_model.run(settings, workspace)
        staged_artifact_summary = build_bootstrap_artifact_summary_fn(
            workspace,
            copied_records,
        )

    def _finalize_bootstrap_result(
        *,
        cache_hit: bool,
        probe_run_id: Optional[str],
        replay_hydration_complete: Optional[bool] = None,
        cache_miss_explanation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        nonlocal staged_artifact_summary
        if not staged_artifact_summary:
            staged_artifact_summary = build_bootstrap_artifact_summary_fn(workspace)
        cache_probe_hit = bool(cache_hit)
        if replay_hydration_complete is None:
            replay_hydration_complete = cache_probe_hit
        return {
            "bootstrap_cache_hit": cache_hit,
            "cache_probe_hit": cache_probe_hit,
            "replay_hydration_complete": bool(replay_hydration_complete),
            "staged_artifact_summary": staged_artifact_summary,
            "run_reference": build_bootstrap_run_reference(
                probe_run_id=probe_run_id,
            ),
            "cache_miss_explanation": cache_miss_explanation,
        }

    run_kwargs: Dict[str, Any] = {
        "fn": _execute_initialization,
        "name": "bootstrap_initialization",
        "model": "initialization",
        "year": state.start_year,
        "iteration": 0,
        "phase": "bootstrap",
        "stage": "bootstrap",
        **build_step_consist_kwargs_fn(
            "initialization",
            settings,
            workspace_path=workspace.full_path,
        ),
    }
    bootstrap_output_paths = _bootstrap_output_paths(
        settings=settings,
        workspace=workspace,
    )
    if bootstrap_output_paths:
        run_kwargs["output_paths"] = bootstrap_output_paths
    run_kwargs["tags"] = merge_tag_list_fn(
        run_kwargs.get("tags"),
        [
            "bootstrap",
            "init",
            f"scenario_id:{scenario_id}",
            "model:initialization",
            f"year:{state.start_year}",
            "iteration:0",
        ]
        + ([f"seed:{seed}"] if seed is not None else []),
    )
    run_kwargs["facet"] = merge_epoch_facet_fn(
        existing=run_kwargs.get("facet"),
        scenario_id=scenario_id,
        seed=seed,
        model="initialization",
        year=state.start_year,
        iteration=0,
    )

    if not is_bootstrap_cache_enabled(settings):
        logger.info("Bootstrap cache disabled; running initialization once.")
        run_result = tracker.run(
            **run_kwargs,
            cache_options=_bootstrap_cache_options(
                settings,
                cache_options_cls=cache_options_cls,
                cache_mode="off",
            ),
        )
        return _finalize_bootstrap_result(
            cache_hit=False,
            probe_run_id=getattr(getattr(run_result, "run", None), "id", None),
        )

    cache_options = _bootstrap_cache_options(
        settings,
        cache_options_cls=cache_options_cls,
        cache_hydration="outputs-requested" if bootstrap_output_paths else None,
    )
    if cache_options is not None:
        probe_result = tracker.run(**run_kwargs, cache_options=cache_options)
    else:
        probe_result = tracker.run(**run_kwargs)
    probe_run_id = getattr(getattr(probe_result, "run", None), "id", None)
    cache_hit = bool(getattr(probe_result, "cache_hit", False))

    if cache_hit:
        logger.info(
            "BOOTSTRAP CACHE HIT. Replaying declared bootstrap output paths into "
            "workspace root=%s run_id=%s",
            workspace.full_path,
            probe_run_id,
        )
        logger.info(
            "BOOTSTRAP CACHE HIT replay hydration restored required workspace artifacts."
        )
        return _finalize_bootstrap_result(
            cache_hit=True,
            probe_run_id=probe_run_id,
            replay_hydration_complete=True,
        )

    run_meta = getattr(getattr(probe_result, "run", None), "meta", None)
    cache_miss_explanation = (
        dict(run_meta["cache_miss_explanation"])
        if isinstance(run_meta, Mapping)
        and isinstance(run_meta.get("cache_miss_explanation"), Mapping)
        else None
    )
    if cache_miss_explanation is not None:
        logger.info(
            "BOOTSTRAP CACHE MISS. Initialization executed for this workspace. "
            "reason=%s candidate_run_id=%s",
            cache_miss_explanation.get("reason"),
            cache_miss_explanation.get("candidate_run_id"),
        )
    else:
        logger.info("BOOTSTRAP CACHE MISS. Initialization executed for this workspace.")
    return _finalize_bootstrap_result(
        cache_hit=False,
        probe_run_id=probe_run_id,
        cache_miss_explanation=cache_miss_explanation,
    )


def assert_bootstrap_output_invariant(
    bootstrap_result: Optional[Dict[str, Any]],
) -> None:
    """
    Ensure bootstrap produced a valid artifact summary before state mutation.

    Some bootstrap modes, such as BEAM-only initialization, can legitimately
    prepare the workspace without emitting copied ``RecordStore`` artifacts.
    In those cases ``copied_records_total == 0`` is still a valid result as
    long as the summary structure is present.
    """
    summary = (
        bootstrap_result.get("staged_artifact_summary")
        if isinstance(bootstrap_result, dict)
        else None
    )
    copied_total = (
        summary.get("copied_records_total") if isinstance(summary, dict) else None
    )
    if (
        isinstance(summary, dict)
        and isinstance(copied_total, int)
        and copied_total >= 0
    ):
        return

    diagnostics = {
        "bootstrap_result_type": type(bootstrap_result).__name__,
        "bootstrap_cache_hit": (
            bootstrap_result.get("bootstrap_cache_hit")
            if isinstance(bootstrap_result, dict)
            else None
        ),
        "run_reference": (
            bootstrap_result.get("run_reference")
            if isinstance(bootstrap_result, dict)
            else None
        ),
        "staged_artifact_summary": summary,
    }
    raise RuntimeError(
        "Bootstrap initialization invariant failed: expected "
        "a valid 'staged_artifact_summary.copied_records_total' before setting "
        f"data_initialized=True. diagnostics={diagnostics}"
    )
