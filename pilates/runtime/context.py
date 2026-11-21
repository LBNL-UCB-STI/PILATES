from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from pilates.workspace import Workspace

if TYPE_CHECKING:
    from pilates.config.models import PilatesConfig
    from pilates.workflows.state import WorkflowState


@dataclass(frozen=True)
class WorkflowRuntimeContext:
    """
    Narrow runtime context for orchestration and stage code.

    The intent is to keep stage/runtime helpers from repeatedly threading the
    same values when they always travel together:

    - validated settings
    - mutable workflow state
    - workspace paths
    - optional archive commit callback
    - expected prefix steps for strict resume

    This is intentionally not a service locator. Tracker/scenario/coupler stay
    outside this object because they are execution handles, not run-shape
    context.
    """

    settings: "PilatesConfig"
    state: "WorkflowState"
    workspace: "Workspace"
    on_archive_commit: Optional[Callable[[str], None]] = None
    expected_prefix_steps: int = 0

    @classmethod
    def from_parts(
        cls,
        *,
        settings: "PilatesConfig",
        state: "WorkflowState",
        workspace: "Workspace",
        on_archive_commit: Optional[Callable[[str], None]] = None,
        expected_prefix_steps: int = 0,
    ) -> "WorkflowRuntimeContext":
        """
        Build a runtime context from validated settings, state, and workspace.
        """
        archive_cb = (
            on_archive_commit
            if on_archive_commit is not None
            else workspace.on_archive_commit
        )
        prefix_steps = (
            expected_prefix_steps
            if expected_prefix_steps != 0
            else workspace.expected_prefix_steps
        )
        return cls(
            settings=settings,
            state=state,
            workspace=workspace,
            on_archive_commit=archive_cb,
            expected_prefix_steps=prefix_steps,
        )
