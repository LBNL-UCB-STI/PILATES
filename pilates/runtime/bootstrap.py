from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

from consist import MaterializationResult
from consist.types import CacheOptions

from pilates.generic.model_factory import ModelFactory
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.coupler_helpers import set_coupler_from_artifact
from pilates.generic.initialization import (
    Initialization,
    build_bootstrap_artifact_summary,
)
from pilates.utils.io import get_traffic_assignment_model
from pilates.workflows.artifact_keys import (
    ASIM_SHARROW_CACHE_DIR,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
)
from pilates.workspace import Workspace

logger = logging.getLogger(__name__)


def is_bootstrap_cache_enabled(settings: Any) -> bool:
    run_cfg = getattr(settings, "run", None)
    return bool(getattr(run_cfg, "bootstrap_cache_enabled", True))


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


def seed_bootstrap_artifacts_to_coupler(
    *,
    settings: Any,
    state: Any,
    workspace: Workspace,
    coupler: CouplerProtocol,
    model_factory_cls: type[ModelFactory] = ModelFactory,
) -> None:
    """
    Seed coupler keys for bootstrap-staged artifacts needed by later steps.

    In BEAM-only runs, the mutable BEAM repo already contains the canonical
    plans/households/persons inputs after bootstrap. Publish those staged files
    into the scenario coupler so downstream steps can depend on explicit
    coupler-backed artifacts instead of speculative filesystem paths.
    """
    get_value = getattr(coupler, "get", None)

    activity_demand_model = getattr(getattr(settings.run, "models", None), "activity_demand", None)
    if activity_demand_model == "activitysim":
        zarr_candidate = os.path.join(
            workspace.get_asim_output_dir(),
            "cache",
            "skims.zarr",
        )
        if os.path.exists(zarr_candidate) and (
            not callable(get_value) or get_value(ZARR_SKIMS) is None
        ):
            artifact = cr.log_output(
                zarr_candidate,
                key=ZARR_SKIMS,
                description="Bootstrap-discovered ActivitySim compiled zarr skims",
            )
            set_coupler_from_artifact(
                coupler,
                ZARR_SKIMS,
                artifact,
                fallback=zarr_candidate,
            )

        sharrow_cache_dir = os.path.join(workspace.full_path, "shared_cache", "numba")
        has_cache_files = False
        if os.path.isdir(sharrow_cache_dir):
            for _root, _dirs, files in os.walk(sharrow_cache_dir):
                if files:
                    has_cache_files = True
                    break
        if has_cache_files and (
            not callable(get_value) or get_value(ASIM_SHARROW_CACHE_DIR) is None
        ):
            artifact = cr.log_output(
                sharrow_cache_dir,
                key=ASIM_SHARROW_CACHE_DIR,
                description="Bootstrap-discovered ActivitySim sharrow cache",
            )
            set_coupler_from_artifact(
                coupler,
                ASIM_SHARROW_CACHE_DIR,
                artifact,
                fallback=sharrow_cache_dir,
            )

    if get_traffic_assignment_model(settings) != "beam":
        return
    if activity_demand_model is not None:
        return

    model_factory = model_factory_cls()
    beam_preprocessor = model_factory.get_preprocessor("beam", state)
    existing_inputs = getattr(beam_preprocessor, "existing_beam_exchange_inputs", None)
    if not callable(existing_inputs):
        logger.debug(
            "BEAM preprocessor does not expose existing_beam_exchange_inputs(); "
            "skipping bootstrap coupler seeding."
        )
        return

    try:
        record_store = existing_inputs(workspace)
    except FileNotFoundError as exc:
        logger.warning(
            "Bootstrap could not seed default BEAM inputs into coupler: %s",
            exc,
        )
        record_store = None

    allowed_keys = {BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN}
    if record_store is not None:
        for record in record_store.all_records():
            key = getattr(record, "short_name", None)
            if key not in allowed_keys:
                continue
            if callable(get_value) and get_value(key) is not None:
                continue
            path = record.get_absolute_path(base_path=workspace.full_path)
            if not path or not os.path.exists(path):
                continue
            artifact = cr.log_output(
                path,
                key=key,
                description="Bootstrap-staged default BEAM scenario input",
            )
            set_coupler_from_artifact(coupler, key, artifact, fallback=path)

    beam_cfg = getattr(settings, "beam", None)
    router_directory = getattr(beam_cfg, "router_directory", None)
    region = getattr(getattr(settings, "run", None), "region", None)
    if router_directory and region and (
        not callable(get_value) or get_value(LINKSTATS_WARMSTART) is None
    ):
        router_root = os.path.join(
            workspace.get_beam_mutable_data_dir(),
            region,
            router_directory,
        )
        warmstart_candidates = (
            os.path.join(router_root, "init.linkstats.parquet"),
            os.path.join(router_root, "init.linkstats.csv.gz"),
        )
        for candidate in warmstart_candidates:
            if not os.path.exists(candidate):
                continue
            artifact = cr.log_output(
                candidate,
                key=LINKSTATS_WARMSTART,
                description="Bootstrap-staged BEAM warm-start linkstats",
            )
            set_coupler_from_artifact(
                coupler,
                LINKSTATS_WARMSTART,
                artifact,
                fallback=candidate,
            )
            break


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
        materialization_result: Optional[MaterializationResult] = None,
        fallback_rerun: bool = False,
    ) -> Dict[str, Any]:
        nonlocal staged_artifact_summary
        if not staged_artifact_summary:
            staged_artifact_summary = build_bootstrap_artifact_summary_fn(workspace)
        return {
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
            "BOOTSTRAP CACHE HIT. Materializing cached bootstrap outputs into "
            "workspace root=%s run_id=%s preserve_existing=True",
            workspace.full_path,
            probe_run_id,
        )
        materialization_result: Optional[MaterializationResult] = None
        materialize_run_outputs_fn = getattr(tracker, "materialize_run_outputs", None)
        if not probe_run_id:
            materialization_result = MaterializationResult(
                failed=[
                    (
                        "bootstrap_initialization",
                        "cache hit missing run id; cannot materialize cached outputs",
                    )
                ]
            )
        elif not callable(materialize_run_outputs_fn):
            materialization_result = MaterializationResult(
                failed=[
                    (
                        "bootstrap_initialization",
                        "tracker does not expose materialize_run_outputs",
                    )
                ]
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
                    failed=[
                        (
                            "bootstrap_initialization",
                            f"materialize_run_outputs raised: {exc}",
                        )
                    ]
                )
                logger.warning(
                    "BOOTSTRAP CACHE HIT materialization failed with exception; "
                    "falling back to explicit rerun. run_id=%s error=%s",
                    probe_run_id,
                    exc,
                )

        if materialization_result.complete:
            logger.info(
                "BOOTSTRAP CACHE HIT materialization complete. %s",
                materialization_result.summary,
            )
            return _finalize_bootstrap_result(
                cache_hit=True,
                probe_run_id=probe_run_id,
                materialization_result=materialization_result,
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
            cache_options=cache_options_cls(cache_mode="off"),
        )
        return _finalize_bootstrap_result(
            cache_hit=True,
            probe_run_id=probe_run_id,
            materialization_run_id=getattr(getattr(fallback_result, "run", None), "id", None),
            materialization_result=materialization_result,
            fallback_rerun=True,
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
