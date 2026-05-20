from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from pilates.utils.coupler_helpers import enqueue_archive_copy


_AUDIT_LOCK = threading.Lock()
_AUDIT_STATE_BY_ROOT: Dict[str, Dict[str, Any]] = {}
_LIFECYCLE_STATE_BY_ROOT: Dict[str, Dict[str, Any]] = {}

_LIFECYCLE_EVENTS_NAME = "artifact_lifecycle_audit.jsonl"
_LIFECYCLE_SUMMARY_NAME = "artifact_lifecycle_audit_summary.json"
_ARCHIVE_LOCAL_ENV = "PILATES_LOCAL_RUN_DIR"
_ARCHIVE_ROOT_ENV = "PILATES_ARCHIVE_RUN_DIR"
_LIFECYCLE_REQUIRED_SNAPSHOT_FIELDS = {
    "artifact_family",
    "source_role",
    "snapshot_role",
    "snapshot_reason",
    "storage_event",
    "year",
}
_LIFECYCLE_REQUIRED_ITERATION_FAMILIES = {
    "asim_input_archived",
    "beam_input_archived",
}
_LIFECYCLE_SCOPED_KEY_PREFIXES = (
    "asim_input_",
    "beam_input_",
    "usim_input_archive_",
    "usim_input_merged_",
    "atlas_input_year_dir",
)
_LIFECYCLE_SCOPED_KEYS = {
    "usim_datastore_h5",
    "usim_datastore_base_h5",
    "usim_datastore_current_h5",
    "usim_forecast_output",
    "usim_population_source_h5",
    "zarr_skims",
    "asim_sharrow_cache_dir",
}
_LIFECYCLE_STABLE_CONTRACT_FAMILIES = {
    "usim_datastore_base_h5",
    "usim_datastore_h5",
    "usim_input_archive",
    "usim_population_source_h5",
    "zarr_skims",
}
_LIFECYCLE_DEFERRED_CONTRACT_FAMILIES = {
    "asim_sharrow_cache_dir",
    "atlas_observe_only",
    "usim_year_output_h5",
}
_LIFECYCLE_TRANSITIONAL_CONTRACT_FAMILIES = {
    "asim_input_archived",
    "beam_input_archived",
    "usim_forecast_output",
}
_LIFECYCLE_ATLAS_PREFIXES = (
    "atlas_",
    "adopt_",
    "accessbility",
    "modeaccessibility",
    "vehicle_type_mapping_",
)
_LIFECYCLE_PHASE2_CANDIDATE_FAMILIES = {
    "usim_input_archive",
    "usim_population_source_h5",
}
_LIFECYCLE_CONTRACT_STATUS_BY_FAMILY = {
    **{family: "stable" for family in _LIFECYCLE_STABLE_CONTRACT_FAMILIES},
    **{family: "deferred" for family in _LIFECYCLE_DEFERRED_CONTRACT_FAMILIES},
    **{family: "transitional" for family in _LIFECYCLE_TRANSITIONAL_CONTRACT_FAMILIES},
}
_LIFECYCLE_RESTART_SUPPORT_KEYS = {
    "workflow_manifest",
}


def _lifecycle_contract_status(family: str) -> Optional[str]:
    return _LIFECYCLE_CONTRACT_STATUS_BY_FAMILY.get(family)


def _lifecycle_contract_status_by_family(
    families: set[str],
) -> Dict[str, str]:
    return {
        family: status
        for family in sorted(families)
        if (status := _lifecycle_contract_status(family)) is not None
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _workspace_root(workspace: Any) -> Optional[Path]:
    full_path = getattr(workspace, "full_path", None)
    if not full_path:
        return None
    return Path(str(full_path))


def _audit_paths(workspace: Any) -> Optional[Dict[str, Path]]:
    workspace_root = _workspace_root(workspace)
    if workspace_root is None:
        return None
    diagnostics_dir = workspace_root / ".workflow" / "diagnostics"
    return {
        "diagnostics_dir": diagnostics_dir,
        "events": diagnostics_dir / "consist_restart_audit.jsonl",
        "summary": diagnostics_dir / "consist_restart_audit_summary.json",
    }


def _lifecycle_paths(
    workspace: Any = None,
    *,
    run_dir: Any = None,
) -> Optional[Dict[str, Path]]:
    root: Optional[Path] = None
    if run_dir is not None:
        root = Path(str(run_dir))
    elif workspace is not None:
        root = _workspace_root(workspace)
    else:
        local_root = os.environ.get(_ARCHIVE_LOCAL_ENV)
        if local_root:
            root = Path(local_root)
    if root is None:
        return None
    diagnostics_dir = root / ".workflow" / "diagnostics"
    return {
        "diagnostics_dir": diagnostics_dir,
        "events": diagnostics_dir / _LIFECYCLE_EVENTS_NAME,
        "summary": diagnostics_dir / _LIFECYCLE_SUMMARY_NAME,
    }


def _attempt_dir(diagnostics_dir: Path, attempt_id: str) -> Path:
    return diagnostics_dir / "attempts" / attempt_id


def _sanitize_path_component(value: Any, fallback: str) -> str:
    text = str(value).strip()
    if not text:
        return fallback
    sanitized = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    result = "".join(sanitized).strip("._")
    return result or fallback


def _attempt_id_for_event(event: Mapping[str, Any], attempt_number: int) -> str:
    recorded_at = _sanitize_path_component(
        event.get("recorded_at") or datetime.now().isoformat(),
        "unknown-time",
    )
    run_name = _sanitize_path_component(event.get("run_name"), "run")
    pid = os.getpid()
    return f"attempt_{attempt_number:04d}__{recorded_at}__pid{pid}__{run_name}"


def _new_state(event: Mapping[str, Any], attempt_number: int) -> Dict[str, Any]:
    return {
        "attempt_id": _attempt_id_for_event(event, attempt_number),
        "attempt_number": attempt_number,
        "attempt_started_at": event.get("recorded_at"),
        "event_counts": defaultdict(int),
        "resolution_mode_counts_by_step": defaultdict(lambda: defaultdict(int)),
        "steps_with_incomplete_hydration": defaultdict(int),
        "steps_using_custom_recovery": defaultdict(lambda: defaultdict(int)),
        "restart_hydration": {
            "event_count": 0,
            "latest_frontier_stage": None,
            "latest_frontier_step": None,
            "latest_success": None,
            "latest_hydrated_key_count": 0,
            "latest_missing_key_count": 0,
            "latest_fallback_reason": None,
        },
        "last_event_at": None,
    }


def _state_for_root(root_key: str) -> Optional[Dict[str, Any]]:
    state = _AUDIT_STATE_BY_ROOT.get(root_key)
    return state


def reset_consist_audit_state() -> None:
    """Clear in-memory audit state, primarily for tests."""
    with _AUDIT_LOCK:
        _AUDIT_STATE_BY_ROOT.clear()
        _LIFECYCLE_STATE_BY_ROOT.clear()


def _summary_payload(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "attempt_id": state.get("attempt_id"),
        "attempt_number": state.get("attempt_number"),
        "attempt_started_at": state.get("attempt_started_at"),
        "generated_at": state["last_event_at"],
        "event_counts": {
            key: state["event_counts"][key] for key in sorted(state["event_counts"])
        },
        "resolution_mode_counts_by_step": {
            step_name: {mode: counts[mode] for mode in sorted(counts)}
            for step_name, counts in sorted(
                state["resolution_mode_counts_by_step"].items()
            )
        },
        "steps_with_incomplete_hydration": {
            step_name: state["steps_with_incomplete_hydration"][step_name]
            for step_name in sorted(state["steps_with_incomplete_hydration"])
        },
        "steps_using_custom_recovery": {
            step_name: {mode: counts[mode] for mode in sorted(counts)}
            for step_name, counts in sorted(
                state["steps_using_custom_recovery"].items()
            )
        },
        "restart_hydration": _json_safe(state.get("restart_hydration", {})),
    }


def _path_under_root(path: Optional[str], root: Optional[str]) -> bool:
    if not path or not root:
        return False
    try:
        return os.path.commonpath(
            [os.path.abspath(path), os.path.abspath(root)]
        ) == os.path.abspath(root)
    except ValueError:
        return False


def _archive_path_for_local(path: Optional[str]) -> Optional[str]:
    local_root = os.environ.get(_ARCHIVE_LOCAL_ENV)
    archive_root = os.environ.get(_ARCHIVE_ROOT_ENV)
    if not path or not local_root or not archive_root:
        return None
    abs_path = os.path.abspath(path)
    if not _path_under_root(abs_path, local_root):
        return None
    rel_path = os.path.relpath(abs_path, os.path.abspath(local_root))
    return os.path.join(os.path.abspath(archive_root), rel_path)


def _lifecycle_new_state() -> Dict[str, Any]:
    return {
        "events": [],
        "attempts": [],
        "current_attempt": None,
        "last_event_at": None,
    }


def _lifecycle_new_attempt(
    event: Mapping[str, Any],
    attempt_number: int,
) -> Dict[str, Any]:
    attempt_id = event.get("lifecycle_attempt_id") or _attempt_id_for_event(
        event,
        attempt_number,
    )
    return {
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "attempt_started_at": event.get("recorded_at"),
        "events": [],
        "last_event_at": None,
    }


def _lifecycle_state_from_existing(paths: Mapping[str, Path]) -> Dict[str, Any]:
    state = _lifecycle_new_state()
    events_path = paths["events"]
    if not events_path.exists():
        return state
    imported_events: list[Mapping[str, Any]] = []
    try:
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if isinstance(event, Mapping):
                    imported_events.append(event)
    except Exception:
        return state
    if not imported_events:
        return state
    state["events"].extend(imported_events)
    state["last_event_at"] = imported_events[-1].get("recorded_at")
    attempts_root = paths["diagnostics_dir"] / "attempts"
    for attempt_number, attempt_events_path in enumerate(
        sorted(attempts_root.glob(f"*/{_LIFECYCLE_EVENTS_NAME}")),
        start=1,
    ):
        attempt_events: list[Mapping[str, Any]] = []
        try:
            with attempt_events_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    if isinstance(event, Mapping):
                        attempt_events.append(event)
        except Exception:
            continue
        if not attempt_events:
            continue
        state["attempts"].append(
            {
                "attempt_id": attempt_events_path.parent.name,
                "attempt_number": attempt_number,
                "attempt_started_at": attempt_events[0].get("recorded_at"),
                "events": attempt_events,
                "last_event_at": attempt_events[-1].get("recorded_at"),
            }
        )
    if not state["attempts"]:
        imported_attempt = {
            "attempt_id": "imported_existing_lifecycle_events",
            "attempt_number": 1,
            "attempt_started_at": imported_events[0].get("recorded_at"),
            "events": imported_events,
            "last_event_at": imported_events[-1].get("recorded_at"),
        }
        state["attempts"].append(imported_attempt)
    state["current_attempt"] = state["attempts"][-1]
    return state


def _lifecycle_event_in_scope(event: Mapping[str, Any]) -> bool:
    event_type = str(event.get("event_type", ""))
    if event_type in {
        "run_context",
        "stage_boundary",
        "archive_copy_checkpoint",
        "final_shutdown",
        "promotion_status",
        "beam_restart_binding",
        "beam_restart_recovery_readiness",
    }:
        return True
    key = str(event.get("key") or "")
    if key.startswith("workflow_diagnostics_"):
        return False
    if key in _LIFECYCLE_SCOPED_KEYS:
        return True
    if key.startswith(_LIFECYCLE_SCOPED_KEY_PREFIXES):
        return True
    artifact_family = str(event.get("artifact_family") or "")
    if artifact_family in {
        "asim_input_archived",
        "beam_input_archived",
        "usim_input_archive",
        "usim_input_merged",
        "usim_datastore_h5",
        "usim_datastore_base_h5",
        "usim_datastore_current_h5",
        "usim_forecast_output",
        "usim_population_source_h5",
        "usim_year_output_h5",
        "zarr_skims",
    }:
        return True
    if key.startswith(_LIFECYCLE_ATLAS_PREFIXES):
        return True
    return False


def _lifecycle_family(event: Mapping[str, Any]) -> str:
    if str(event.get("event_type")) == "promotion_status":
        return "post_run_promotion"
    artifact_family = str(event.get("artifact_family") or "")
    if artifact_family:
        return artifact_family
    key = str(event.get("key") or "")
    if key.startswith("beam_input_") and key.endswith("_archived"):
        return "beam_input_archived"
    if key.startswith("asim_input_") and key.endswith("_archived"):
        return "asim_input_archived"
    if key.startswith("usim_input_archive_"):
        return "usim_input_archive"
    if key.startswith("usim_input_merged_"):
        return "usim_input_merged"
    if key.startswith("usim_year_output_h5_"):
        return "usim_year_output_h5"
    if key in _LIFECYCLE_SCOPED_KEYS:
        return key
    if key == "atlas_preprocess_output":
        return "atlas_observe_only"
    if key.startswith(_LIFECYCLE_ATLAS_PREFIXES):
        return "atlas_observe_only"
    return "unknown"


def _lifecycle_event_has_artifact_identity(event: Mapping[str, Any]) -> bool:
    if event.get("artifact_family"):
        return True
    if event.get("key"):
        return True
    return str(event.get("event_type")) == "promotion_status"


def _lifecycle_path_kind(path: Optional[str], event: Mapping[str, Any]) -> str:
    artifact_driver = str(
        event.get("artifact_driver")
        or event.get("driver")
        or event.get("artifact_type")
        or ""
    ).lower()
    family = _lifecycle_family(event)
    if family == "zarr_skims" or artifact_driver == "zarr":
        return "zarr"
    if bool(event.get("h5_container")) or str(path or "").lower().endswith(
        (".h5", ".hdf5")
    ):
        return "h5"
    if bool(event.get("is_dir")):
        return "directory"
    if path and os.path.isdir(path):
        return "directory"
    return "file"


def _joined_lifecycle_event(
    copy_event: Mapping[str, Any],
    logged_event: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    joined = dict(copy_event)
    if logged_event is None:
        return joined
    for key, value in logged_event.items():
        if joined.get(key) in (None, ""):
            joined[key] = value
    return joined


def _is_h5_child_artifact(event: Mapping[str, Any]) -> bool:
    artifact_driver = str(
        event.get("artifact_driver")
        or event.get("driver")
        or event.get("artifact_type")
        or ""
    ).lower()
    if artifact_driver == "h5_table":
        return True
    return bool(event.get("h5_parent_key")) and not bool(event.get("h5_container"))


def _h5_parent_policy_allows_registration(event: Mapping[str, Any]) -> bool:
    if _is_h5_child_artifact(event):
        return False
    return (
        str(event.get("container_recovery_unit") or "") == "parent_file"
        and str(event.get("child_recovery_policy") or "") == "descriptive_only"
    )


def _lifecycle_missing_required_facets(event: Mapping[str, Any]) -> list[str]:
    family = _lifecycle_family(event)
    if family not in {
        "asim_input_archived",
        "beam_input_archived",
        "usim_input_archive",
        "usim_input_merged",
    }:
        return []
    required = set(_LIFECYCLE_REQUIRED_SNAPSHOT_FIELDS)
    if family in _LIFECYCLE_REQUIRED_ITERATION_FAMILIES:
        required.add("iteration")
    missing = []
    for field in sorted(required):
        value = event.get(field)
        if value in (None, "") and field == "year":
            value = event.get("artifact_year")
        if value in (None, "") and field == "iteration":
            value = event.get("artifact_iteration")
        if value in (None, ""):
            missing.append(field)
    return missing


def _copy_match_key(event: Mapping[str, Any]) -> tuple[str, str]:
    key = str(event.get("key") or "")
    path = str(event.get("path") or event.get("src") or "")
    return key, os.path.abspath(path) if path and "://" not in path else path


def _lifecycle_core_summary_payload(
    *,
    events: list[Mapping[str, Any]],
    generated_at: Any,
    phase2_recommendation_basis: str,
    attempt_id: Optional[str] = None,
    attempt_number: Optional[int] = None,
    attempt_started_at: Any = None,
) -> Dict[str, Any]:
    logged: Dict[tuple[str, str], tuple[int, Mapping[str, Any]]] = {}
    copied: Dict[tuple[str, str], tuple[int, Mapping[str, Any]]] = {}
    families_seen: set[str] = set()
    phase2_candidate_families = set(_LIFECYCLE_PHASE2_CANDIDATE_FAMILIES)
    phase2_safe_families: set[str] = set()
    phase2_blocked_families: set[str] = set()
    phase2_blocking_reasons_by_family: Dict[str, set[str]] = defaultdict(set)
    blocking_reasons_by_family: Dict[str, set[str]] = defaultdict(set)
    diagnostic_blocking_reasons_by_family: Dict[str, set[str]] = defaultdict(set)
    blocker_counts_by_reason: Dict[str, int] = defaultdict(int)
    phase2_blocker_counts_by_reason: Dict[str, int] = defaultdict(int)
    restart_support_keys: set[str] = set()
    unknown_event_keys: set[str] = set()
    snapshot_logged = 0
    missing_required = 0
    copy_only_promotions = 0
    local_to_scratch_recovery_root_writes = 0
    copied_artifacts_eligible = 0
    phase2_candidate_copied_artifacts_eligible = 0

    for index, event in enumerate(events):
        key = str(event.get("key") or "")
        if key in _LIFECYCLE_RESTART_SUPPORT_KEYS:
            restart_support_keys.add(key)
            continue
        family = _lifecycle_family(event)
        if _lifecycle_event_has_artifact_identity(event):
            families_seen.add(family)
            if family == "unknown":
                unknown_key = str(
                    event.get("key") or event.get("event_type") or "unknown"
                )
                unknown_event_keys.add(unknown_key)
                blocking_reasons_by_family["unknown"].add("unclassified_family")
                blocker_counts_by_reason["unclassified_family"] += 1
        event_type = str(event.get("event_type"))
        if event_type == "artifact_logged":
            logged.setdefault(_copy_match_key(event), (index, event))
            if _is_h5_child_artifact(event):
                blocking_reasons_by_family[family].add("h5_child_table_ineligible")
                blocker_counts_by_reason["h5_child_table_ineligible"] += 1
                if family in phase2_candidate_families:
                    phase2_blocking_reasons_by_family[family].add(
                        "h5_child_table_ineligible"
                    )
                    phase2_blocker_counts_by_reason[
                        "h5_child_table_ineligible"
                    ] += 1
            if family in {
                "asim_input_archived",
                "beam_input_archived",
                "usim_input_archive",
                "usim_input_merged",
            }:
                snapshot_logged += 1
                if _lifecycle_missing_required_facets(event):
                    missing_required += 1
                    blocking_reasons_by_family[family].add(
                        "missing_required_snapshot_facets"
                    )
                    blocker_counts_by_reason["missing_required_snapshot_facets"] += 1
                    if family in phase2_candidate_families:
                        phase2_blocking_reasons_by_family[family].add(
                            "missing_required_snapshot_facets"
                        )
                        phase2_blocker_counts_by_reason[
                            "missing_required_snapshot_facets"
                        ] += 1
        elif event_type == "archive_copy_completed":
            copied[_copy_match_key(event)] = (index, event)
        elif event_type == "promotion_status":
            metadata_updated = bool(event.get("artifact_metadata_updated"))
            copy_only_promotion = (
                not metadata_updated
                and bool(event.get("copy_performed"))
                and str(event.get("status")) == "promoted"
            )
            if copy_only_promotion:
                copy_only_promotions += 1
                blocking_reasons_by_family["post_run_promotion"].add(
                    "copy_only_promotion_metadata_unavailable"
                )
                blocker_counts_by_reason[
                    "copy_only_promotion_metadata_unavailable"
                ] += 1
        if bool(event.get("local_to_scratch_recovery_roots_written")):
            local_to_scratch_recovery_root_writes += 1

    archive_bytes_missing = 0
    copy_before_log = 0
    source_outside_run_tree = 0
    shallow_directory = 0
    h5_policy = 0
    child_h5_policy = sum(1 for event in events if _is_h5_child_artifact(event))
    copied_joined_to_logged = 0

    for match_key, (copy_index, copy_event) in copied.items():
        local_root = os.environ.get(_ARCHIVE_LOCAL_ENV)
        path = str(copy_event.get("src") or copy_event.get("path") or "")
        dest = str(copy_event.get("dest") or _archive_path_for_local(path) or "")
        reasons: list[str] = []
        logged_entry = logged.get(match_key)
        logged_event = logged_entry[1] if logged_entry is not None else None
        joined_event = _joined_lifecycle_event(copy_event, logged_event)
        family = _lifecycle_family(joined_event)
        families_seen.add(family)
        path_kind = _lifecycle_path_kind(path, joined_event)

        if not _path_under_root(path, local_root):
            source_outside_run_tree += 1
            reasons.append("source_outside_run_tree")
        if not dest or not os.path.exists(dest):
            archive_bytes_missing += 1
            reasons.append("archive_bytes_missing")
        if logged_entry is None:
            reasons.append("artifact_not_logged")
        elif copy_index < logged_entry[0]:
            copy_before_log += 1
            reasons.append("artifact_logging_after_copying")
        if path_kind == "directory":
            shallow_directory += 1
            reasons.append("shallow_directory_signature")
        if logged_event is not None:
            copied_joined_to_logged += 1
        if _is_h5_child_artifact(joined_event):
            reasons.append("h5_child_table_ineligible")
        elif path_kind == "h5" and not _h5_parent_policy_allows_registration(
            joined_event
        ):
            h5_policy += 1
            reasons.append("h5_parent_child_policy")
        if path_kind == "directory":
            reasons.append("shallow_directory_signature")

        if reasons:
            for reason in sorted(set(reasons)):
                if family == "atlas_observe_only":
                    diagnostic_blocking_reasons_by_family[family].add(reason)
                else:
                    blocking_reasons_by_family[family].add(reason)
                blocker_counts_by_reason[reason] += 1
                if family in phase2_candidate_families:
                    phase2_blocking_reasons_by_family[family].add(reason)
                    phase2_blocker_counts_by_reason[reason] += 1
        elif family in phase2_candidate_families:
            copied_artifacts_eligible += 1
            phase2_safe_families.add(family)
            phase2_candidate_copied_artifacts_eligible += 1
        else:
            copied_artifacts_eligible += 1
    for family in sorted(phase2_candidate_families - families_seen):
        phase2_blocking_reasons_by_family[family].add("phase2_candidate_missing")
        blocking_reasons_by_family[family].add("phase2_candidate_missing")
        blocker_counts_by_reason["phase2_candidate_missing"] += 1
        phase2_blocker_counts_by_reason["phase2_candidate_missing"] += 1

    phase2_blocked_families.update(phase2_blocking_reasons_by_family.keys())

    if phase2_safe_families and phase2_blocked_families:
        recommendation = "narrow"
        reason = "Some intended Phase 2 candidates look safe, but blocked candidates remain."
    elif phase2_safe_families and not phase2_blocked_families:
        recommendation = "go"
        reason = "All intended Phase 2 candidates passed."
    else:
        recommendation = "defer"
        reason = "The audit cannot yet prove intended Phase 2 candidates are safe."

    payload = {
        "generated_at": generated_at,
        "phase2_recommendation_basis": phase2_recommendation_basis,
        "event_counts": {
            event_type: sum(
                1 for event in events if event.get("event_type") == event_type
            )
            for event_type in sorted({str(event.get("event_type")) for event in events})
        },
        "families_seen": sorted(families_seen),
        "restart_support_keys": sorted(restart_support_keys),
        "unknown_event_keys": sorted(unknown_event_keys),
        "snapshot_artifacts_logged": snapshot_logged,
        "snapshot_artifacts_missing_required_facets": missing_required,
        "copied_artifacts_eligible_for_recovery_root_registration": copied_artifacts_eligible,
        "phase2_candidate_copied_artifacts_eligible_for_recovery_root_registration": phase2_candidate_copied_artifacts_eligible,
        "copied_artifacts_blocked_archive_bytes_missing": archive_bytes_missing,
        "copied_artifacts_blocked_artifact_logging_after_copying": copy_before_log,
        "copied_artifacts_blocked_source_outside_run_tree": source_outside_run_tree,
        "copied_artifacts_joined_to_logged_artifacts": copied_joined_to_logged,
        "directory_artifacts_blocked_shallow_directory_signatures": shallow_directory,
        "h5_parent_child_artifacts_requiring_policy": h5_policy,
        "h5_child_table_artifacts_ineligible": child_h5_policy,
        "copy_only_promotions_db_tracker_metadata_unavailable": copy_only_promotions,
        "local_to_scratch_recovery_roots_written": local_to_scratch_recovery_root_writes,
        "atlas_observe_only_deferred": True,
        "phase2_recommendation": recommendation,
        "phase2_recommendation_reason": reason,
        "contract_status_by_family": _lifecycle_contract_status_by_family(
            families_seen
        ),
        "phase2_candidate_families": sorted(phase2_candidate_families),
        "safe_families_for_phase2": sorted(phase2_safe_families),
        "blocked_families_for_phase2": sorted(phase2_blocked_families),
        "blocking_reasons_by_family": {
            family: sorted(reasons)
            for family, reasons in sorted(blocking_reasons_by_family.items())
        },
        "blocker_counts_by_reason": {
            reason: blocker_counts_by_reason[reason]
            for reason in sorted(blocker_counts_by_reason)
        },
        "phase2_blocker_counts_by_reason": {
            reason: phase2_blocker_counts_by_reason[reason]
            for reason in sorted(phase2_blocker_counts_by_reason)
        },
        "diagnostic_blocking_reasons_by_family": {
            family: sorted(reasons)
            for family, reasons in sorted(diagnostic_blocking_reasons_by_family.items())
        },
    }
    if attempt_id is not None:
        payload["attempt_id"] = attempt_id
    if attempt_number is not None:
        payload["attempt_number"] = attempt_number
    if attempt_started_at is not None:
        payload["attempt_started_at"] = attempt_started_at
    return payload


def _lifecycle_summary_payload(state: Mapping[str, Any]) -> Dict[str, Any]:
    events = list(state.get("events") or [])
    payload = _lifecycle_core_summary_payload(
        events=events,
        generated_at=state.get("last_event_at"),
        phase2_recommendation_basis="aggregate_attempts",
    )
    attempt_summaries = [
        _lifecycle_core_summary_payload(
            events=list(attempt.get("events") or []),
            generated_at=attempt.get("last_event_at"),
            phase2_recommendation_basis="attempt",
            attempt_id=str(attempt.get("attempt_id")),
            attempt_number=attempt.get("attempt_number"),
            attempt_started_at=attempt.get("attempt_started_at"),
        )
        for attempt in list(state.get("attempts") or [])
    ]
    payload["attempt_summaries"] = attempt_summaries
    payload["latest_attempt_summary"] = (
        attempt_summaries[-1] if attempt_summaries else None
    )
    return payload


def _update_summary_state(state: Dict[str, Any], event: Mapping[str, Any]) -> None:
    event_type = str(event.get("event_type"))
    state["event_counts"][event_type] += 1
    state["last_event_at"] = event.get("recorded_at")
    if event_type == "restart_hydration":
        state["restart_hydration"] = {
            "event_count": int(state.get("restart_hydration", {}).get("event_count", 0))
            + 1,
            "latest_frontier_stage": event.get("frontier_stage"),
            "latest_frontier_step": event.get("frontier_step"),
            "latest_success": bool(event.get("success", False)),
            "latest_hydrated_key_count": len(list(event.get("hydrated_keys") or ())),
            "latest_missing_key_count": len(list(event.get("missing_keys") or ())),
            "latest_fallback_reason": event.get("fallback_reason"),
        }

    step_name = event.get("step_name")
    if step_name:
        resolution_mode = (
            event.get("resolution_mode")
            if event.get("event_type") == "step_resolution"
            else None
        )
        if resolution_mode:
            state["resolution_mode_counts_by_step"][str(step_name)][
                str(resolution_mode)
            ] += 1
        if event.get("event_type") == "output_hydration_check" and not bool(
            event.get("hydration_complete", False)
        ):
            state["steps_with_incomplete_hydration"][str(step_name)] += 1
        for mode_field in (
            "used_output_replayer",
            "used_output_recoverer",
            "used_tracker_output_lookup",
            "used_cached_artifact_recovery",
            "used_manifest_restore",
            "used_compatibility_fallback",
        ):
            if bool(event.get(mode_field)):
                state["steps_using_custom_recovery"][str(step_name)][mode_field] += 1


def emit_consist_audit_event(
    *,
    workspace: Any,
    event_type: str,
    **fields: Any,
) -> None:
    paths = _audit_paths(workspace)
    if paths is None:
        return

    event = {
        "event_type": str(event_type),
        "recorded_at": datetime.now().isoformat(),
        **{key: _json_safe(value) for key, value in fields.items()},
    }
    root_key = str(paths["diagnostics_dir"].parent.parent)

    with _AUDIT_LOCK:
        current_state = _state_for_root(root_key)
        is_new_attempt = current_state is None or event_type == "run_context"
        if is_new_attempt:
            attempt_number = (
                1 if current_state is None else int(current_state["attempt_number"]) + 1
            )
            current_state = _new_state(event, attempt_number)
            _AUDIT_STATE_BY_ROOT[root_key] = current_state
        if current_state is None:
            raise RuntimeError("Consist audit state was not initialized.")
        attempt_dir = _attempt_dir(
            paths["diagnostics_dir"], str(current_state["attempt_id"])
        )
        attempt_paths = {
            "diagnostics_dir": attempt_dir,
            "events": attempt_dir / "consist_restart_audit.jsonl",
            "summary": attempt_dir / "consist_restart_audit_summary.json",
        }
        paths["diagnostics_dir"].mkdir(parents=True, exist_ok=True)
        attempt_paths["diagnostics_dir"].mkdir(parents=True, exist_ok=True)
        event_mode = "w" if is_new_attempt else "a"
        for event_path in (paths["events"], attempt_paths["events"]):
            with event_path.open(event_mode, encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        _update_summary_state(current_state, event)
        summary_payload = _summary_payload(current_state)
        for summary_path in (paths["summary"], attempt_paths["summary"]):
            summary_path.write_text(
                json.dumps(summary_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        # Mirror diagnostics through the artifact archive plane only. Consist DB
        # snapshots/mirroring are owned by the launcher and snapshot manager.
        enqueue_archive_copy(
            key="workflow_diagnostics_consist_restart_audit",
            path=paths["events"],
        )
        enqueue_archive_copy(
            key="workflow_diagnostics_consist_restart_audit_summary",
            path=paths["summary"],
        )
        enqueue_archive_copy(
            key="workflow_diagnostics_consist_restart_audit_attempt",
            path=attempt_paths["events"],
        )
        enqueue_archive_copy(
            key="workflow_diagnostics_consist_restart_audit_summary_attempt",
            path=attempt_paths["summary"],
        )
    if event_type == "run_context":
        emit_artifact_lifecycle_audit_event(
            workspace=workspace,
            event_type="run_context",
            run_name=fields.get("run_name"),
            local_run_dir=fields.get("local_run_dir"),
            archive_run_dir=fields.get("archive_run_dir"),
            restart_run=fields.get("restart_run"),
            lifecycle_attempt_id=current_state.get("attempt_id"),
            lifecycle_attempt_number=current_state.get("attempt_number"),
        )
    elif event_type in {"step_resolution", "output_hydration_check"}:
        emit_artifact_lifecycle_audit_event(
            workspace=workspace,
            event_type="stage_boundary",
            stage_name=fields.get("stage_name"),
            step_name=fields.get("step_name"),
            year=fields.get("year"),
            iteration=fields.get("iteration"),
            run_id=fields.get("run_id"),
            resolution_mode=fields.get("resolution_mode"),
            lifecycle_attempt_id=current_state.get("attempt_id"),
            lifecycle_attempt_number=current_state.get("attempt_number"),
        )


def emit_artifact_lifecycle_audit_event(
    *,
    workspace: Any = None,
    run_dir: Any = None,
    event_type: str,
    **fields: Any,
) -> None:
    """Record shadow-mode artifact lifecycle evidence without affecting execution."""
    paths = _lifecycle_paths(workspace, run_dir=run_dir)
    if paths is None:
        return

    event = {
        "event_type": str(event_type),
        "recorded_at": datetime.now().isoformat(),
        **{key: _json_safe(value) for key, value in fields.items()},
    }
    if not _lifecycle_event_in_scope(event):
        return
    root_key = str(paths["diagnostics_dir"].parent.parent)

    try:
        with _AUDIT_LOCK:
            state = _LIFECYCLE_STATE_BY_ROOT.get(root_key)
            if state is None:
                state = _lifecycle_state_from_existing(paths)
                _LIFECYCLE_STATE_BY_ROOT[root_key] = state
            current_attempt = state.get("current_attempt")
            is_new_attempt = current_attempt is None or event_type == "run_context"
            if is_new_attempt:
                attempt_number = int(
                    event.get("lifecycle_attempt_number")
                    or len(list(state.get("attempts") or [])) + 1
                )
                current_attempt = _lifecycle_new_attempt(event, attempt_number)
                state.setdefault("attempts", []).append(current_attempt)
                state["current_attempt"] = current_attempt
            state["events"].append(event)
            current_attempt["events"].append(event)
            state["last_event_at"] = event["recorded_at"]
            current_attempt["last_event_at"] = event["recorded_at"]
            paths["diagnostics_dir"].mkdir(parents=True, exist_ok=True)
            attempt_dir = _attempt_dir(
                paths["diagnostics_dir"],
                str(current_attempt["attempt_id"]),
            )
            attempt_paths = {
                "diagnostics_dir": attempt_dir,
                "events": attempt_dir / _LIFECYCLE_EVENTS_NAME,
                "summary": attempt_dir / _LIFECYCLE_SUMMARY_NAME,
            }
            attempt_paths["diagnostics_dir"].mkdir(parents=True, exist_ok=True)
            with paths["events"].open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            attempt_event_mode = (
                "w" if is_new_attempt and len(current_attempt["events"]) == 1 else "a"
            )
            with attempt_paths["events"].open(
                attempt_event_mode,
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            summary_payload = _lifecycle_summary_payload(state)
            paths["summary"].write_text(
                json.dumps(summary_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            attempt_summary = _lifecycle_core_summary_payload(
                events=list(current_attempt.get("events") or []),
                generated_at=current_attempt.get("last_event_at"),
                phase2_recommendation_basis="attempt",
                attempt_id=str(current_attempt.get("attempt_id")),
                attempt_number=current_attempt.get("attempt_number"),
                attempt_started_at=current_attempt.get("attempt_started_at"),
            )
            attempt_paths["summary"].write_text(
                json.dumps(attempt_summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        enqueue_archive_copy(
            key="workflow_diagnostics_artifact_lifecycle_audit",
            path=paths["events"],
        )
        enqueue_archive_copy(
            key="workflow_diagnostics_artifact_lifecycle_audit_summary",
            path=paths["summary"],
        )
        enqueue_archive_copy(
            key="workflow_diagnostics_artifact_lifecycle_audit_attempt",
            path=attempt_paths["events"],
        )
        enqueue_archive_copy(
            key="workflow_diagnostics_artifact_lifecycle_audit_summary_attempt",
            path=attempt_paths["summary"],
        )
    except Exception:
        # This audit is intentionally shadow-mode and must never alter workflow
        # correctness.
        return
