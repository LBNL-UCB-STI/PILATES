from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Protocol, TypeVar, cast

from pilates.utils import consist_runtime as cr
from pilates.workflows.coupler_namespace import canonical_artifact_key_from_raw_key

RunOutputValue = TypeVar("RunOutputValue")


class TrackerRunOutputs(Protocol[RunOutputValue]):
    def get_run_outputs(self, run_id: str) -> Mapping[str, RunOutputValue] | None: ...


def canonicalize_output_mapping(
    mapping: Optional[Mapping[str, RunOutputValue]],
) -> Dict[str, RunOutputValue]:
    canonical: Dict[str, RunOutputValue] = {}
    if not mapping:
        return canonical
    for raw_key, value in mapping.items():
        if value is None:
            continue
        canonical[canonical_artifact_key_from_raw_key(str(raw_key))] = value
    return canonical


def merge_canonical_output_mappings(
    *mappings: Optional[Mapping[str, RunOutputValue]],
) -> Dict[str, RunOutputValue]:
    merged: Dict[str, RunOutputValue] = {}
    for mapping in mappings:
        if not mapping:
            continue
        merged.update(canonicalize_output_mapping(mapping))
    return merged


def load_tracker_run_outputs(
    run_id: Optional[str],
    *,
    tracker: Optional[TrackerRunOutputs[RunOutputValue]] = None,
    logger: Optional[logging.Logger] = None,
    log_context: str = "tracker output query",
) -> Dict[str, RunOutputValue]:
    if not run_id:
        return {}
    active_tracker = (
        tracker
        if tracker is not None
        else cast(Optional[TrackerRunOutputs[RunOutputValue]], cr.current_tracker())
    )
    if active_tracker is None:
        raise RuntimeError(
            f"Cannot load {log_context} for run_id={run_id}: no active Consist tracker."
        )
    try:
        get_run_outputs = active_tracker.get_run_outputs
    except AttributeError as exc:
        raise RuntimeError(
            f"Cannot load {log_context} for run_id={run_id}: tracker does not expose "
            "get_run_outputs()."
        ) from exc
    try:
        run_outputs = get_run_outputs(run_id)
    except Exception:
        if logger is not None:
            logger.debug(
                "Failed loading %s for run_id=%s",
                log_context,
                run_id,
                exc_info=True,
            )
        return {}
    if run_outputs is None:
        return {}
    return canonicalize_output_mapping(run_outputs)
