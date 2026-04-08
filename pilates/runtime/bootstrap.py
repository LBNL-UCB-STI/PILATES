from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

from consist import MaterializationResult
from consist.types import CacheOptions

from pilates.config import PilatesConfig
from pilates.activitysim.preprocessor import required_asim_config_dirs
from pilates.generic.model_factory import ModelFactory
from pilates.generic.initialization import (
    Initialization,
    build_bootstrap_artifact_summary,
)
from pilates.runtime.cache_recovery import (
    cache_miss_audit_fields,
    log_cache_miss_explanation,
)
from pilates.runtime.consist_audit import emit_consist_audit_event
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.io import get_activity_demand_model, get_traffic_assignment_model
from pilates.workflows.binding import bootstrap_stage_boundary_durability_policy
from pilates.workspace import Workspace

logger = logging.getLogger(__name__)
_OPTIONAL_BOOTSTRAP_MISSING_SOURCE_NAMES = frozenset(
    {"canonical_zones", "clipped_geoms"}
)


def is_bootstrap_cache_enabled(settings: PilatesConfig) -> bool:
    run_cfg = getattr(settings, "run", None)
    return bool(getattr(run_cfg, "bootstrap_cache_enabled", True))


def _bootstrap_cache_options(
    settings: PilatesConfig,
    *,
    cache_options_cls: type[CacheOptions],
    cache_mode: Optional[str] = None,
) -> Optional[CacheOptions]:
    run_cfg = getattr(settings, "run", None)
    code_identity = getattr(run_cfg, "consist_code_identity", None)
    if cache_mode is None and code_identity is None:
        return None
    return cache_options_cls(
        cache_mode=cache_mode,
        code_identity=code_identity,
    )


def _warn_on_bootstrap_fast_hashing(settings: PilatesConfig) -> None:
    run_cfg = getattr(settings, "run", None)
    if not bool(getattr(run_cfg, "bootstrap_cache_enabled", True)):
        return
    hashing_strategy = str(
        getattr(run_cfg, "consist_hashing_strategy", "fast")
    ).lower()
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
    materialization_run_id: Optional[str] = None,
) -> Dict[str, str]:
    reference: Dict[str, str] = {}
    if probe_run_id:
        reference["probe_run_id"] = probe_run_id
    if materialization_run_id:
        reference["materialization_run_id"] = materialization_run_id
    return reference


def _bootstrap_materialization_metadata(
    result: MaterializationResult,
) -> Dict[str, Any]:
    return {
        "summary": result.summary,
        "complete": result.complete,
        "has_failures": result.has_failures,
        "materialized_from_filesystem_count": len(result.materialized_from_filesystem),
        "materialized_from_db_count": len(result.materialized_from_db),
        "skipped_existing_count": len(result.skipped_existing),
        "skipped_unmapped_count": len(result.skipped_unmapped),
        "skipped_missing_source_count": len(result.skipped_missing_source),
        "failed_count": len(result.failed),
        "skipped_unmapped": list(result.skipped_unmapped),
        "skipped_missing_source": list(result.skipped_missing_source),
        "failed": list(result.failed),
    }


def _bootstrap_materialization_entry_name(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, (tuple, list)) and entry:
        return str(entry[0])
    if isinstance(entry, dict):
        for key in ("key", "name", "short_name"):
            value = entry.get(key)
            if value:
                return str(value)
    return str(entry)


def _prune_optional_bootstrap_missing_sources(
    result: MaterializationResult,
) -> MaterializationResult:
    tolerated = [
        entry
        for entry in list(getattr(result, "skipped_missing_source", []) or [])
        if _bootstrap_materialization_entry_name(entry)
        in _OPTIONAL_BOOTSTRAP_MISSING_SOURCE_NAMES
    ]
    if tolerated:
        result.skipped_missing_source = [
            entry
            for entry in list(getattr(result, "skipped_missing_source", []) or [])
            if _bootstrap_materialization_entry_name(entry)
            not in _OPTIONAL_BOOTSTRAP_MISSING_SOURCE_NAMES
        ]
        logger.info(
            "BOOTSTRAP CACHE HIT ignoring optional missing-source artifacts: %s",
            [_bootstrap_materialization_entry_name(entry) for entry in tolerated],
        )
    return result


def _bootstrap_required_workspace_artifacts(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
) -> Dict[str, str]:
    required: Dict[str, str] = {}

    if get_activity_demand_model(settings) == "activitysim":
        get_asim_configs_dir = getattr(workspace, "get_asim_mutable_configs_dir", None)
        if callable(get_asim_configs_dir):
            asim_configs_dir = get_asim_configs_dir()
            main_configs_dir = (
                getattr(getattr(settings, "activitysim", None), "main_configs_dir", None)
                or "configs"
            )
            for dirname in required_asim_config_dirs(main_configs_dir):
                required[f"activitysim_config_settings_yaml_{dirname}"] = os.path.join(
                    asim_configs_dir,
                    dirname,
                    "settings.yaml",
                )

    if get_traffic_assignment_model(settings) == "beam":
        get_beam_input_dir = getattr(workspace, "get_beam_mutable_data_dir", None)
        region = getattr(getattr(settings, "run", None), "region", None)
        if callable(get_beam_input_dir) and region:
            beam_input_dir = get_beam_input_dir()
            required["beam_mutable_data_dir"] = beam_input_dir
            required["beam_region_input_dir"] = os.path.join(beam_input_dir, region)
            beam_config_name = getattr(getattr(settings, "beam", None), "config", None)
            if beam_config_name:
                required["beam_primary_config_file"] = os.path.join(
                    beam_input_dir,
                    region,
                    beam_config_name,
                )

    return required


def _find_missing_bootstrap_workspace_artifacts(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
) -> list[Dict[str, str]]:
    missing: list[Dict[str, str]] = []
    for key, path in _bootstrap_required_workspace_artifacts(
        settings=settings,
        workspace=workspace,
    ).items():
        if not path:
            continue
        normalized_path = os.path.realpath(path)
        if os.path.exists(normalized_path):
            continue
        missing.append(
            {
                "key": key,
                "path": normalized_path,
                "reason": (
                    "Bootstrap cache-hit validation requires this workspace "
                    "artifact to exist locally after materialization."
                ),
            }
        )
    return missing


def _format_missing_bootstrap_workspace_artifacts(
    artifacts: list[Dict[str, str]],
) -> str:
    if not artifacts:
        return "none"
    return ", ".join(
        f"{item.get('key')}:{item.get('path')}"
        for item in artifacts
    )


def seed_bootstrap_artifacts_to_coupler(
    *,
    settings: PilatesConfig,
    state: Any,
    workspace: Workspace,
    coupler: CouplerProtocol,
    model_factory_cls: type[ModelFactory] = ModelFactory,
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
            model_factory_cls=model_factory_cls,
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
    build_bootstrap_artifact_summary_fn: Callable[..., Dict[str, Any]] = build_bootstrap_artifact_summary,
    build_step_consist_kwargs_fn: Callable[..., Dict[str, Any]],
    merge_tag_list_fn: Callable[..., list[str]],
    merge_epoch_facet_fn: Callable[..., Dict[str, Any]],
    cache_options_cls: type[CacheOptions] = CacheOptions,
) -> Dict[str, Any]:
    """
    Execute initialization in a dedicated pre-scenario bootstrap phase.
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
        materialization_run_id: Optional[str] = None,
        materialization_result: Optional[MaterializationResult] = None,
        fallback_rerun: bool = False,
        resolution_mode: str,
        cache_miss_explanation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        nonlocal staged_artifact_summary
        if not staged_artifact_summary:
            staged_artifact_summary = build_bootstrap_artifact_summary_fn(workspace)
        result = {
            "bootstrap_cache_hit": cache_hit,
            "staged_artifact_summary": staged_artifact_summary,
            "run_reference": build_bootstrap_run_reference(
                probe_run_id=probe_run_id,
                materialization_run_id=materialization_run_id,
            ),
            "materialization": (
                _bootstrap_materialization_metadata(materialization_result)
                if materialization_result is not None
                else None
            ),
            "fallback_rerun": fallback_rerun,
            "cache_miss_explanation": cache_miss_explanation,
        }
        emit_consist_audit_event(
            workspace=workspace,
            event_type="bootstrap_resolution",
            scenario_id=scenario_id,
            seed=seed,
            year=state.start_year,
            iteration=0,
            resolution_mode=resolution_mode,
            bootstrap_cache_enabled=is_bootstrap_cache_enabled(settings),
            bootstrap_cache_hit=cache_hit,
            fallback_rerun=fallback_rerun,
            probe_run_id=probe_run_id,
            materialization_run_id=materialization_run_id,
            materialization=result.get("materialization"),
            staged_artifact_summary=staged_artifact_summary,
            **cache_miss_audit_fields(cache_miss_explanation),
        )
        return result

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
            resolution_mode="cache_disabled_execute",
        )

    cache_options = _bootstrap_cache_options(
        settings,
        cache_options_cls=cache_options_cls,
    )
    if cache_options is not None:
        probe_result = tracker.run(**run_kwargs, cache_options=cache_options)
    else:
        probe_result = tracker.run(**run_kwargs)
    probe_run_id = getattr(getattr(probe_result, "run", None), "id", None)
    cache_hit = bool(getattr(probe_result, "cache_hit", False))

    if cache_hit:
        logger.info(
            "BOOTSTRAP CACHE HIT. Materializing cached bootstrap outputs into "
            "workspace root=%s run_id=%s preserve_existing=True",
            workspace.full_path,
            probe_run_id,
        )
        materialize_run_outputs_fn = getattr(tracker, "materialize_run_outputs", None)
        materialization_exc = None
        if not probe_run_id:
            materialization_result = MaterializationResult(
                failed=[("bootstrap_initialization", "cache hit missing run id")]
            )
        elif not callable(materialize_run_outputs_fn):
            materialization_result = MaterializationResult(
                failed=[("bootstrap_initialization", "tracker does not expose materialize_run_outputs")]
            )
        else:
            try:
                materialization_result = materialize_run_outputs_fn(
                    run_id=probe_run_id,
                    target_root=workspace.full_path,
                    source_root=None,
                    preserve_existing=True,
                )
            except Exception as exc:
                materialization_result = MaterializationResult(
                    failed=[("bootstrap_initialization", f"materialize_run_outputs raised: {exc}")]
                )
                materialization_exc = exc
        if materialization_exc is not None:
            logger.warning(
                "BOOTSTRAP CACHE HIT materialization failed with exception; "
                "falling back to explicit rerun. run_id=%s error=%s",
                probe_run_id,
                materialization_exc,
            )
        materialization_result = _prune_optional_bootstrap_missing_sources(
            materialization_result
        )

        missing_workspace_artifacts: list[Dict[str, str]] = []
        if materialization_result.complete:
            missing_workspace_artifacts = _find_missing_bootstrap_workspace_artifacts(
                settings=settings,
                workspace=workspace,
            )

        if materialization_result.complete and not missing_workspace_artifacts:
            logger.info(
                "BOOTSTRAP CACHE HIT materialization complete. %s",
                materialization_result.summary,
            )
            return _finalize_bootstrap_result(
                cache_hit=True,
                probe_run_id=probe_run_id,
                materialization_result=materialization_result,
                resolution_mode="cache_hit_materialized",
            )

        fallback_cache_options = _bootstrap_cache_options(
            settings,
            cache_options_cls=cache_options_cls,
            cache_mode="off",
        )
        if missing_workspace_artifacts:
            logger.warning(
                "BOOTSTRAP CACHE HIT materialization completed, but required "
                "workspace artifacts are still missing: %s",
                _format_missing_bootstrap_workspace_artifacts(
                    missing_workspace_artifacts
                ),
            )
            logger.warning(
                "BOOTSTRAP fallback rerun triggered because cached bootstrap outputs "
                "did not restore required workspace invariants."
            )
            fallback_result = tracker.run(
                **run_kwargs,
                cache_options=fallback_cache_options,
            )
            return _finalize_bootstrap_result(
                cache_hit=True,
                probe_run_id=probe_run_id,
                materialization_run_id=getattr(
                    getattr(fallback_result, "run", None),
                    "id",
                    None,
                ),
                materialization_result=materialization_result,
                fallback_rerun=True,
                resolution_mode="cache_hit_missing_workspace_invariants_fallback_rerun",
            )

        logger.warning(
            "BOOTSTRAP CACHE HIT materialization incomplete. %s "
            "(skipped_unmapped=%s skipped_missing_source=%s failed=%s)",
            materialization_result.summary,
            materialization_result.skipped_unmapped,
            materialization_result.skipped_missing_source,
            materialization_result.failed,
        )
        logger.warning(
            "BOOTSTRAP fallback rerun triggered because cached output recovery was incomplete."
        )
        fallback_result = tracker.run(
            **run_kwargs,
            cache_options=fallback_cache_options,
        )
        return _finalize_bootstrap_result(
            cache_hit=True,
            probe_run_id=probe_run_id,
            materialization_run_id=getattr(getattr(fallback_result, "run", None), "id", None),
            materialization_result=materialization_result,
            fallback_rerun=True,
            resolution_mode="cache_hit_incomplete_fallback_rerun",
        )

    cache_miss_explanation = log_cache_miss_explanation(
        logger=logger,
        result=probe_result,
        info_message=(
            "BOOTSTRAP CACHE MISS. Initialization executed for this workspace. "
            "reason=%s candidate_run_id=%s"
        ),
        debug_message="BOOTSTRAP cache miss details: %s",
    )
    if cache_miss_explanation is None:
        logger.info("BOOTSTRAP CACHE MISS. Initialization executed for this workspace.")
    return _finalize_bootstrap_result(
        cache_hit=False,
        probe_run_id=probe_run_id,
        resolution_mode="cache_miss_execute",
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
    if isinstance(summary, dict) and isinstance(copied_total, int) and copied_total >= 0:
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
