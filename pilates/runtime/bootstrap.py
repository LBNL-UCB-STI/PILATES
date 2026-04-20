from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

from consist.types import CacheOptions

from pilates.urbansim.postprocessor import get_usim_datastore_fname
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


def _bootstrap_output_paths(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
    surface: Any = None,
) -> Dict[str, str]:
    output_paths: Dict[str, str] = {}
    run_models = getattr(getattr(settings, "run", None), "models", None)

    get_usim_data_dir = getattr(workspace, "get_usim_mutable_data_dir", None)
    if callable(get_usim_data_dir) and (
        getattr(run_models, "land_use", None) == "urbansim"
        or getattr(run_models, "activity_demand", None) == "activitysim"
        or getattr(run_models, "vehicle_ownership", None) == "atlas"
    ):
        output_paths["urbansim_mutable_data_dir"] = get_usim_data_dir()

    if get_activity_demand_model(settings) == "activitysim":
        get_asim_data_dir = getattr(workspace, "get_asim_mutable_data_dir", None)
        if callable(get_asim_data_dir):
            output_paths["activitysim_mutable_data_dir"] = get_asim_data_dir()
        get_asim_configs_dir = getattr(workspace, "get_asim_mutable_configs_dir", None)
        if callable(get_asim_configs_dir):
            output_paths["activitysim_mutable_configs_dir"] = get_asim_configs_dir()

    if getattr(run_models, "vehicle_ownership", None) == "atlas":
        get_atlas_input_dir = getattr(workspace, "get_atlas_mutable_input_dir", None)
        if callable(get_atlas_input_dir):
            output_paths["atlas_mutable_input_dir"] = get_atlas_input_dir()

    if get_traffic_assignment_model(settings) == "beam":
        get_beam_input_dir = getattr(workspace, "get_beam_mutable_data_dir", None)
        if callable(get_beam_input_dir):
            output_paths["beam_mutable_data_dir"] = get_beam_input_dir()

    # Cache-hit bootstrap replay materializes only the explicitly requested
    # output paths. The workspace-invariant check is stricter than the coarse
    # mutable-root set above: later startup code expects specific staged files
    # like ActivitySim settings.yaml overlays and the BEAM primary config to
    # exist. Include those exact invariants here so a bootstrap cache hit can
    # restore them directly instead of replaying only root directories and then
    # falling back to a full rerun.
    output_paths.update(
        _bootstrap_required_workspace_artifacts(
            settings=settings,
            workspace=workspace,
            surface=surface,
        )
    )

    return output_paths


def _bootstrap_required_workspace_artifacts(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
    surface: Any = None,
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

    get_usim_data_dir = getattr(workspace, "get_usim_mutable_data_dir", None)
    run_models = getattr(getattr(settings, "run", None), "models", None)
    if callable(get_usim_data_dir) and run_models is not None:
        if (
            getattr(run_models, "land_use", None) == "urbansim"
            or getattr(run_models, "activity_demand", None) == "activitysim"
            or getattr(run_models, "vehicle_ownership", None) == "atlas"
        ):
            usim_data_dir = get_usim_data_dir()
            required["usim_datastore_base_h5"] = os.path.join(
                usim_data_dir,
                get_usim_datastore_fname(settings, io="input"),
            )

    if surface is not None:
        required = {
            key: path
            for key, path in required.items()
            if surface.is_bootstrap_owned_artifact_key(key)
        }

    return required


def _find_missing_bootstrap_workspace_artifacts(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
    surface: Any = None,
) -> list[Dict[str, str]]:
    missing: list[Dict[str, str]] = []
    for key, path in _bootstrap_required_workspace_artifacts(
        settings=settings,
        workspace=workspace,
        surface=surface,
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
                    "artifact to exist locally after replay hydration."
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
    surface: Any = None,
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
        fallback_run_id: Optional[str] = None,
        fallback_rerun: bool = False,
        replay_hydration_complete: Optional[bool] = None,
        resolution_mode: str,
        cache_miss_explanation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        nonlocal staged_artifact_summary
        if not staged_artifact_summary:
            staged_artifact_summary = build_bootstrap_artifact_summary_fn(workspace)
        cache_probe_hit = bool(cache_hit)
        fallback_rerun_triggered = bool(fallback_rerun)
        if replay_hydration_complete is None:
            replay_hydration_complete = not fallback_rerun_triggered if cache_probe_hit else False
        result = {
            "bootstrap_cache_hit": cache_hit,
            "cache_probe_hit": cache_probe_hit,
            "replay_hydration_complete": bool(replay_hydration_complete),
            "staged_artifact_summary": staged_artifact_summary,
            "run_reference": build_bootstrap_run_reference(
                probe_run_id=probe_run_id,
                materialization_run_id=fallback_run_id,
            ),
            "fallback_rerun": fallback_rerun,
            "fallback_rerun_triggered": fallback_rerun_triggered,
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
            cache_probe_hit=cache_probe_hit,
            replay_hydration_complete=bool(replay_hydration_complete),
            fallback_rerun=fallback_rerun,
            fallback_rerun_triggered=fallback_rerun_triggered,
            probe_run_id=probe_run_id,
            materialization_run_id=fallback_run_id,
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
    bootstrap_output_paths = _bootstrap_output_paths(
        settings=settings,
        workspace=workspace,
        surface=surface,
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
            resolution_mode="cache_disabled_execute",
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
        missing_workspace_artifacts = _find_missing_bootstrap_workspace_artifacts(
            settings=settings,
            workspace=workspace,
            surface=surface,
        )
        if not missing_workspace_artifacts:
            logger.info(
                "BOOTSTRAP CACHE HIT replay hydration restored required workspace artifacts."
            )
            return _finalize_bootstrap_result(
                cache_hit=True,
                probe_run_id=probe_run_id,
                replay_hydration_complete=True,
                resolution_mode="cache_hit_replay_hydrated",
            )

        fallback_cache_options = _bootstrap_cache_options(
            settings,
            cache_options_cls=cache_options_cls,
            cache_mode="off",
        )
        logger.warning(
            "BOOTSTRAP CACHE HIT replay hydration left required workspace "
            "artifacts missing: %s",
            _format_missing_bootstrap_workspace_artifacts(
                missing_workspace_artifacts
            ),
        )
        logger.warning(
            "BOOTSTRAP fallback rerun triggered because replay hydration did not "
            "restore required workspace invariants."
        )
        fallback_result = tracker.run(
            **run_kwargs,
            cache_options=fallback_cache_options,
        )
        return _finalize_bootstrap_result(
            cache_hit=True,
            probe_run_id=probe_run_id,
            fallback_run_id=getattr(getattr(fallback_result, "run", None), "id", None),
            fallback_rerun=True,
            replay_hydration_complete=False,
            resolution_mode="cache_hit_missing_workspace_invariants_fallback_rerun",
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
