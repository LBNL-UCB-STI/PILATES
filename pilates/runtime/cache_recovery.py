from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional, Sequence
from consist.types import CacheOptions

logger = logging.getLogger(__name__)

_CACHE_MISS_EXPLANATION_META_KEYS = (
    "cache_miss_explanation",
    "cache_miss_explplanation",
)
_CACHE_MISS_DETAIL_KEYS = (
    "identity_inputs_changed",
    "config_keys_changed",
    "config_keys_added",
    "config_keys_removed",
    "adapter_identity_changed",
    "input_keys_added",
    "input_keys_removed",
    "input_artifact_changes",
    "code_identity_changed",
    "code_identity_mode_changed",
    "code_identity_extra_deps_changed",
    "repo_git_identity_changed",
    "code_hash_changed",
)


def _has_observability_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def get_cache_miss_explanation(result: Any) -> Optional[dict[str, Any]]:
    """
    Best-effort extraction of Consist cache miss explanation metadata.
    """
    run = getattr(result, "run", None)
    meta = getattr(run, "meta", None)
    if not isinstance(meta, Mapping):
        return None

    for key in _CACHE_MISS_EXPLANATION_META_KEYS:
        explanation = meta.get(key)
        if isinstance(explanation, Mapping):
            return dict(explanation)
    return None


def cache_miss_audit_fields(
    explanation: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(explanation, Mapping):
        return {}
    return {
        "cache_miss_reason": explanation.get("reason"),
        "cache_miss_candidate_run_id": explanation.get("candidate_run_id"),
        "cache_miss_explanation": dict(explanation),
    }


def _cache_miss_debug_payload(
    explanation: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    payload: dict[str, Any] = {}

    for key in ("confidence", "matched_components", "mismatched_components"):
        value = explanation.get(key)
        if _has_observability_value(value):
            payload[key] = value

    details = explanation.get("details")
    if isinstance(details, Mapping):
        selected_details = {
            key: details.get(key)
            for key in _CACHE_MISS_DETAIL_KEYS
            if _has_observability_value(details.get(key))
        }
        if selected_details:
            payload["details"] = selected_details

        fallbacks_used = details.get("fallbacks_used")
        if _has_observability_value(fallbacks_used):
            payload["fallbacks_used"] = fallbacks_used
    else:
        fallbacks_used = explanation.get("fallbacks_used")
        if _has_observability_value(fallbacks_used):
            payload["fallbacks_used"] = fallbacks_used

    return payload or None


def log_cache_miss_explanation(
    *,
    logger: logging.Logger,
    result: Any,
    info_message: str,
    info_args: Sequence[Any] = (),
    debug_message: Optional[str] = None,
    debug_args: Sequence[Any] = (),
) -> Optional[dict[str, Any]]:
    """
    Log optional cache miss explanation metadata for human operators.
    """
    explanation = get_cache_miss_explanation(result)
    if explanation is None:
        return None

    logger.info(
        info_message,
        *info_args,
        explanation.get("reason"),
        explanation.get("candidate_run_id"),
    )

    debug_payload = _cache_miss_debug_payload(explanation)
    if debug_payload:
        if debug_message is not None:
            logger.debug(debug_message, *debug_args, debug_payload)
        else:
            logger.debug("Cache miss details: %s", debug_payload)

    return explanation


def run_with_cache_recovery(
    *,
    stage_name: str,
    step_name: str,
    run_step: Callable[[Optional[CacheOptions]], Any],
    read_outputs: Callable[[], Optional[Any]],
    recover_outputs: Callable[[Any], Optional[Any]],
) -> tuple[Any, Optional[Any], dict[str, Any]]:
    """
    Run a step once and, on cache hit, attempt one explicit output recovery pass.
    """
    result = run_step(None)
    metadata = {
        "initial_cache_hit": bool(getattr(result, "cache_hit", False)),
        "recovery_attempts": 0,
    }
    if not getattr(result, "cache_hit", False):
        cache_miss_explanation = log_cache_miss_explanation(
            logger=logger,
            result=result,
            info_message="[%s] Cache miss for %s. reason=%s candidate_run_id=%s",
            info_args=(stage_name, step_name),
            debug_message="[%s] Cache miss details for %s: %s",
            debug_args=(stage_name, step_name),
        )
        if cache_miss_explanation is not None:
            metadata["cache_miss_explanation"] = cache_miss_explanation
    outputs = read_outputs()
    if outputs is None and getattr(result, "cache_hit", False):
        metadata["recovery_attempts"] += 1
        outputs = recover_outputs(result)
    metadata["final_cache_hit"] = bool(getattr(result, "cache_hit", False))
    return result, outputs, metadata
