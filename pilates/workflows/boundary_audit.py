"""Opt-in observations of recovery-boundary successor bindings."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

from consist.protocols import ArtifactRecordLike

from pilates.utils.consist_runtime import artifact_fingerprint
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workspace import Workspace
from workflow_state import WorkflowState

if TYPE_CHECKING:
    from pilates.workflows.surface import EnabledWorkflowSurface


_AUDIT_ENV = "PILATES_RECOVERY_BOUNDARY_AUDIT"
_ARCHIVE_RUN_DIR_ENV = "PILATES_ARCHIVE_RUN_DIR"
_AUDIT_RELATIVE_PATH = Path(".workflow/diagnostics/recovery_boundary_audit.jsonl")


def _audit_enabled() -> bool:
    return os.environ.get(_AUDIT_ENV) == "1"


def _archive_run_dir(*, state: WorkflowState, workspace: Workspace) -> Path:
    configured = os.environ.get(_ARCHIVE_RUN_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    if state.file_loc:
        return Path(state.file_loc).expanduser().resolve().parent
    return Path(workspace.full_path).expanduser().resolve()


def _relative_locator(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _artifact_observation(
    *,
    key: str,
    value: Any,
    workspace: Workspace,
    workspace_root: Path,
    archive_root: Path,
) -> Dict[str, Any]:
    raw_path = artifact_to_path(value, workspace=workspace)
    existing_path: Optional[Path] = None
    if raw_path and "://" not in raw_path:
        candidate = Path(raw_path).expanduser().resolve()
        if candidate.exists():
            existing_path = candidate

    artifact_id = None
    artifact_key = None
    artifact_kind = None
    driver = None
    if isinstance(value, ArtifactRecordLike):
        artifact_id = str(value.id)
        artifact_key = value.key
        artifact_kind = value.meta.get("artifact_kind")
        driver = value.driver

    return {
        "semantic_key": key,
        "value_type": type(value).__name__,
        "existing_path": str(existing_path) if existing_path is not None else None,
        "workspace_relative_locator": _relative_locator(existing_path, workspace_root),
        "archive_relative_locator": _relative_locator(existing_path, archive_root),
        "artifact_id": artifact_id,
        "artifact_key": artifact_key,
        "artifact_kind": artifact_kind,
        "driver": driver,
        "fingerprint": artifact_fingerprint(value),
    }


def preflight_recovery_boundary_audit(
    *, state: WorkflowState, workspace: Workspace
) -> Optional[Path]:
    """Create and verify the enabled audit sink before model execution."""
    if not _audit_enabled():
        return None
    audit_path = (
        _archive_run_dir(state=state, workspace=workspace) / _AUDIT_RELATIVE_PATH
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        audit_path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return audit_path


def emit_recovery_boundary_audit(
    *,
    boundary: str,
    successor_step: str,
    binding: ResolvedStepInputs,
    predecessor_outputs: Optional[Mapping[str, Any]] = None,
    state: WorkflowState,
    workspace: Workspace,
    surface: Optional["EnabledWorkflowSurface"],
) -> Optional[Path]:
    """Append one diagnostic observation without altering workflow state."""
    if not _audit_enabled():
        return None

    workspace_root = Path(workspace.full_path).expanduser().resolve()
    archive_root = _archive_run_dir(state=state, workspace=workspace)
    audit_path = preflight_recovery_boundary_audit(state=state, workspace=workspace)
    if audit_path is None:
        return None

    required_keys = sorted(set(binding.required_roles))
    optional_keys = sorted(set(binding.optional_roles))
    bound_inputs = dict(binding.binding.inputs or {})
    predecessor_values = dict(predecessor_outputs or {})
    payload = {
        "schema_version": "v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "boundary": boundary,
        "successor_step": successor_step,
        "scope": {
            "year": state.year,
            "forecast_year": state.forecast_year,
            "iteration": state.iteration,
        },
        "workspace_root": str(workspace_root),
        "archive_root": str(archive_root),
        "surface": surface.to_dict() if surface is not None else None,
        "binding": {
            "step_name": binding.step_name,
            "required_input_keys": required_keys,
            "optional_input_keys": optional_keys,
            "bound_input_keys": sorted(bound_inputs),
            "missing_required": sorted(
                key
                for key in binding.required_roles
                if binding.source_by_role.get(key) == "missing"
            ),
            "source_by_key": dict(sorted(binding.source_by_role.items())),
            "coupler_key_by_key": dict(sorted(binding.selected_key_by_role.items())),
        },
        "artifacts": {
            key: _artifact_observation(
                key=key,
                value=value,
                workspace=workspace,
                workspace_root=workspace_root,
                archive_root=archive_root,
            )
            for key, value in sorted(bound_inputs.items())
        },
        "predecessor_artifacts": {
            key: _artifact_observation(
                key=key,
                value=value,
                workspace=workspace,
                workspace_root=workspace_root,
                archive_root=archive_root,
            )
            for key, value in sorted(predecessor_values.items())
        },
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        audit_path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(
                f"Short recovery boundary audit write: {written}/{len(encoded)} bytes"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return audit_path
