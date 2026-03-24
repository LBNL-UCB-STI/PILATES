from __future__ import annotations

import json
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


def _state_for_root(root_key: str) -> Dict[str, Any]:
    state = _AUDIT_STATE_BY_ROOT.get(root_key)
    if state is not None:
        return state
    state = {
        "event_counts": defaultdict(int),
        "resolution_mode_counts_by_step": defaultdict(lambda: defaultdict(int)),
        "steps_with_incomplete_hydration": defaultdict(int),
        "steps_using_custom_recovery": defaultdict(lambda: defaultdict(int)),
        "last_event_at": None,
    }
    _AUDIT_STATE_BY_ROOT[root_key] = state
    return state


def _summary_payload(root_key: str) -> Dict[str, Any]:
    state = _state_for_root(root_key)
    return {
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
    }


def _update_summary_state(root_key: str, event: Mapping[str, Any]) -> None:
    state = _state_for_root(root_key)
    event_type = str(event.get("event_type"))
    state["event_counts"][event_type] += 1
    state["last_event_at"] = event.get("recorded_at")

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
        paths["diagnostics_dir"].mkdir(parents=True, exist_ok=True)
        with paths["events"].open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        _update_summary_state(root_key, event)
        paths["summary"].write_text(
            json.dumps(_summary_payload(root_key), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        # Mirror diagnostics into the archive run dir when local and archive roots
        # differ, so post-run inspection and restart debugging can use the same
        # audit bundle.
        enqueue_archive_copy(
            key="workflow_diagnostics_consist_restart_audit",
            path=paths["events"],
        )
        enqueue_archive_copy(
            key="workflow_diagnostics_consist_restart_audit_summary",
            path=paths["summary"],
        )
