from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence, Tuple

from consist import MaterializationResult
from consist.types import CacheOptions

logger = logging.getLogger(__name__)


def materialize_cached_run(
    *,
    tracker: Any,
    run_id: Optional[str],
    target_root: str,
    source_root: Optional[str],
    preserve_existing: bool,
    context: str,
) -> tuple[MaterializationResult, Optional[Exception]]:
    """
    Materialize one cached Consist run output set with normalized failures.
    """
    if not run_id:
        return (
            MaterializationResult(
                failed=[
                    (
                        context,
                        "cache hit missing run id; cannot materialize cached outputs",
                    )
                ]
            ),
            None,
        )

    materialize_run_outputs_fn = getattr(tracker, "materialize_run_outputs", None)
    if not callable(materialize_run_outputs_fn):
        return (
            MaterializationResult(
                failed=[(context, "tracker does not expose materialize_run_outputs")]
            ),
            None,
        )

    try:
        return (
            materialize_run_outputs_fn(
                run_id=run_id,
                target_root=target_root,
                source_root=source_root,
                preserve_existing=preserve_existing,
            ),
            None,
        )
    except Exception as exc:
        return (
            MaterializationResult(
                failed=[(context, f"materialize_run_outputs raised: {exc}")]
            ),
            exc,
        )


def merge_materialization_result(
    *,
    aggregate: MaterializationResult,
    result: MaterializationResult,
) -> None:
    """
    Merge one materialization result into an aggregate result in-place.
    """
    aggregate.materialized_from_filesystem.update(
        dict(getattr(result, "materialized_from_filesystem", {}) or {})
    )
    aggregate.materialized_from_db.update(
        dict(getattr(result, "materialized_from_db", {}) or {})
    )
    aggregate.skipped_existing.extend(list(getattr(result, "skipped_existing", []) or []))
    aggregate.skipped_unmapped.extend(list(getattr(result, "skipped_unmapped", []) or []))
    aggregate.skipped_missing_source.extend(
        list(getattr(result, "skipped_missing_source", []) or [])
    )
    aggregate.failed.extend(list(getattr(result, "failed", []) or []))


def materialize_cached_runs(
    *,
    tracker: Any,
    run_ids: Sequence[str],
    target_root: str,
    source_root: Optional[str],
    preserve_existing: bool,
    initial_failures: Optional[Sequence[Tuple[str, str]]] = None,
    missing_api_context: str = "restart_reconstruction",
) -> MaterializationResult:
    """
    Materialize many cached run output sets and return one aggregate result.
    """
    aggregate = MaterializationResult()
    if initial_failures:
        aggregate.failed.extend(list(initial_failures))

    if not run_ids:
        return aggregate

    materialize_run_outputs_fn = getattr(tracker, "materialize_run_outputs", None)
    if not callable(materialize_run_outputs_fn):
        aggregate.failed.append(
            (missing_api_context, "tracker does not expose materialize_run_outputs")
        )
        return aggregate

    for run_id in run_ids:
        result, _exc = materialize_cached_run(
            tracker=tracker,
            run_id=run_id,
            target_root=target_root,
            source_root=source_root,
            preserve_existing=preserve_existing,
            context=run_id,
        )
        merge_materialization_result(aggregate=aggregate, result=result)

    return aggregate


def run_with_cache_recovery(
    *,
    stage_name: str,
    step_name: str,
    run_step: Callable[[Optional[CacheOptions]], Any],
    read_outputs: Callable[[], Optional[Any]],
    recover_outputs: Callable[[Any], Optional[Any]],
) -> tuple[Any, Optional[Any], dict[str, Any]]:
    """
    Run a step once, try cache-hit recovery, and rerun with overwrite as fallback.
    """
    result = run_step(None)
    metadata = {
        "initial_cache_hit": bool(getattr(result, "cache_hit", False)),
        "recovery_attempts": 0,
        "overwrite_rerun": False,
    }
    outputs = read_outputs()
    if outputs is None and getattr(result, "cache_hit", False):
        metadata["recovery_attempts"] += 1
        outputs = recover_outputs(result)
    if outputs is None and getattr(result, "cache_hit", False):
        logger.warning(
            "[%s] Cache hit for %s could not hydrate outputs_holder; rerunning with cache_mode=overwrite.",
            stage_name,
            step_name,
        )
        metadata["overwrite_rerun"] = True
        result = run_step(CacheOptions(cache_mode="overwrite"))
        outputs = read_outputs()
        if outputs is None and getattr(result, "cache_hit", False):
            metadata["recovery_attempts"] += 1
            outputs = recover_outputs(result)
    metadata["final_cache_hit"] = bool(getattr(result, "cache_hit", False))
    return result, outputs, metadata
