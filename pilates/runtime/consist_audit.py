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


def _summary_payload(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "attempt_id": state.get("attempt_id"),
        "attempt_number": state.get("attempt_number"),
        "attempt_started_at": state.get("attempt_started_at"),
        "generated_at": state["last_event_at"],
        "event_counts": {
            key: state["event_counts"][key]
            for key in sorted(state["event_counts"])
        },
        "resolution_mode_counts_by_step": {
            step_name: {
                mode: counts[mode]
                for mode in sorted(counts)
            }
            for step_name, counts in sorted(
                state["resolution_mode_counts_by_step"].items()
            )
        },
        "steps_with_incomplete_hydration": {
            step_name: state["steps_with_incomplete_hydration"][step_name]
            for step_name in sorted(state["steps_with_incomplete_hydration"])
        },
        "steps_using_custom_recovery": {
            step_name: {
                mode: counts[mode]
                for mode in sorted(counts)
            }
            for step_name, counts in sorted(
                state["steps_using_custom_recovery"].items()
            )
        },
        "restart_hydration": _json_safe(state.get("restart_hydration", {})),
    }


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
        assert current_state is not None
        attempt_dir = _attempt_dir(paths["diagnostics_dir"], str(current_state["attempt_id"]))
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
