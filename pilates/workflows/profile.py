from __future__ import annotations

"""Compatibility shim for legacy profile imports.

Runtime flag initialization and profile construction now live in the private
``pilates.workflows._profile`` module. Keep these re-exports while local tests
and any straggling callers still import from ``pilates.workflows.profile``.
"""

from pilates.workflows._profile import (
    WorkflowProfile,
    build_workflow_profile,
    ensure_runtime_flags_initialized,
    workflow_profile_from_flags,
)

__all__ = [
    "WorkflowProfile",
    "build_workflow_profile",
    "ensure_runtime_flags_initialized",
    "workflow_profile_from_flags",
]
