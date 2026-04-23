from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

from pilates.utils import consist_runtime as cr
from pilates.workflows.coupler_namespace import canonical_artifact_key_from_raw_key


def canonicalize_output_mapping(
    mapping: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    canonical: Dict[str, Any] = {}
    if not mapping:
        return canonical
    for raw_key, value in mapping.items():
        if value is None:
            continue
        canonical[canonical_artifact_key_from_raw_key(str(raw_key))] = value
    return canonical


def merge_canonical_output_mappings(
    *mappings: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for mapping in mappings:
        if not mapping:
            continue
        merged.update(canonicalize_output_mapping(mapping))
    return merged


def load_tracker_run_outputs(
    run_id: Optional[str],
    *,
    tracker: Any = None,
    logger: Optional[logging.Logger] = None,
    log_context: str = "tracker output query",
) -> Dict[str, Any]:
    if not run_id:
        return {}
    active_tracker = tracker if tracker is not None else cr.current_tracker()
    if active_tracker is None:
        raise RuntimeError(
            f"Cannot load {log_context} for run_id={run_id}: no active Consist tracker."
        )
    get_run_outputs = getattr(active_tracker, "get_run_outputs", None)
    if not callable(get_run_outputs):
        raise RuntimeError(
            f"Cannot load {log_context} for run_id={run_id}: tracker does not expose "
            "get_run_outputs()."
        )
    try:
        run_outputs = get_run_outputs(run_id) or {}
    except Exception:
        if logger is not None:
            logger.debug(
                "Failed loading %s for run_id=%s",
                log_context,
                run_id,
                exc_info=True,
            )
        return {}
    return canonicalize_output_mapping(run_outputs)
