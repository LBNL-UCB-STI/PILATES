from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

from consist.types import CacheOptions

from pilates.generic.initialization import (
    Initialization,
    build_bootstrap_artifact_summary,
)
from pilates.urbansim.postprocessor import get_usim_datastore_fname
from pilates.utils.coupler_helpers import enqueue_archive_copy, flush_archive_queue
from pilates.workspace import Workspace

logger = logging.getLogger(__name__)


def is_bootstrap_cache_enabled(settings: Any) -> bool:
    run_cfg = getattr(settings, "run", None)
    return bool(getattr(run_cfg, "bootstrap_cache_enabled", True))


def build_bootstrap_manifest_reference(
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


def archive_bootstrap_restart_artifacts(
    *,
    settings: Any,
    workspace: Workspace,
    enqueue_archive_copy_fn: Callable[..., Any] = enqueue_archive_copy,
    flush_archive_queue_fn: Callable[..., Any] = flush_archive_queue,
    get_usim_datastore_fname_fn: Callable[..., str] = get_usim_datastore_fname,
) -> None:
    """
    Durably archive bootstrap-created local runtime state needed for restart.
    """
    run_cfg = getattr(settings, "run", None)
    model_cfg = getattr(run_cfg, "models", None)
    urbansim_cfg = getattr(settings, "urbansim", None)
    if (
        run_cfg is not None
        and getattr(run_cfg, "region", None)
        and urbansim_cfg is not None
    ):
        usim_data_dir = workspace.get_usim_mutable_data_dir()
        if os.path.isdir(usim_data_dir):
            enqueue_archive_copy_fn(
                key="urbansim_bootstrap_data_root",
                path=usim_data_dir,
            )
        usim_base_path = os.path.join(
            usim_data_dir,
            get_usim_datastore_fname_fn(settings, io="input"),
        )
        if os.path.exists(usim_base_path):
            enqueue_archive_copy_fn(
                key="bootstrap_usim_datastore_base_h5",
                path=usim_base_path,
            )

    if getattr(model_cfg, "activity_demand", None) == "activitysim":
        asim_data_dir = workspace.get_asim_mutable_data_dir()
        asim_configs_dir = workspace.get_asim_mutable_configs_dir()

        if os.path.isdir(asim_data_dir):
            enqueue_archive_copy_fn(
                key="activitysim_bootstrap_data_root",
                path=asim_data_dir,
            )
        if os.path.isdir(asim_configs_dir):
            enqueue_archive_copy_fn(
                key="activitysim_bootstrap_configs_root",
                path=asim_configs_dir,
            )

    beam_model = getattr(model_cfg, "traffic_assignment", None) or getattr(
        model_cfg, "travel", None
    )
    if beam_model == "beam":
        beam_data_dir = workspace.get_beam_mutable_data_dir()
        if os.path.isdir(beam_data_dir):
            enqueue_archive_copy_fn(
                key="beam_mutable_data_dir",
                path=beam_data_dir,
            )

    flush_archive_queue_fn(timeout=300, fail_on_timeout=True)


def run_bootstrap_phase(
    *,
    tracker: Any,
    settings: Any,
    state: Any,
    workspace: Workspace,
    scenario_id: str,
    seed: Optional[int],
    initialization_cls: type[Initialization] = Initialization,
    build_bootstrap_artifact_summary_fn: Callable[..., Dict[str, Any]] = build_bootstrap_artifact_summary,
    build_step_consist_kwargs_fn: Callable[..., Dict[str, Any]],
    merge_tag_list_fn: Callable[..., list[str]],
    merge_epoch_facet_fn: Callable[..., Dict[str, Any]],
    archive_bootstrap_restart_artifacts_fn: Callable[..., None] = archive_bootstrap_restart_artifacts,
    cache_options_cls: type[CacheOptions] = CacheOptions,
) -> Dict[str, Any]:
    """
    Execute initialization in a dedicated pre-scenario bootstrap phase.
    """
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
    ) -> Dict[str, Any]:
        nonlocal staged_artifact_summary
        if not staged_artifact_summary:
            staged_artifact_summary = build_bootstrap_artifact_summary_fn(workspace)
        archive_bootstrap_restart_artifacts_fn(
            settings=settings,
            workspace=workspace,
        )
        return {
            "bootstrap_cache_hit": cache_hit,
            "staged_artifact_summary": staged_artifact_summary,
            "manifest_reference": build_bootstrap_manifest_reference(
                probe_run_id=probe_run_id,
                materialization_run_id=materialization_run_id,
            ),
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
            cache_options=cache_options_cls(cache_mode="off"),
        )
        return _finalize_bootstrap_result(
            cache_hit=False,
            probe_run_id=getattr(getattr(run_result, "run", None), "id", None),
        )

    probe_result = tracker.run(**run_kwargs)
    probe_run_id = getattr(getattr(probe_result, "run", None), "id", None)
    cache_hit = bool(getattr(probe_result, "cache_hit", False))

    if cache_hit:
        logger.info(
            "BOOTSTRAP CACHE HIT. Running Phase 1 materialization pass to keep workspace safe."
        )
        materialized_result = tracker.run(
            **run_kwargs,
            cache_options=cache_options_cls(cache_mode="overwrite"),
        )
        return _finalize_bootstrap_result(
            cache_hit=True,
            probe_run_id=probe_run_id,
            materialization_run_id=getattr(
                getattr(materialized_result, "run", None), "id", None
            ),
        )

    logger.info("BOOTSTRAP CACHE MISS. Initialization executed for this workspace.")
    return _finalize_bootstrap_result(
        cache_hit=False,
        probe_run_id=probe_run_id,
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
        "manifest_reference": (
            bootstrap_result.get("manifest_reference")
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
