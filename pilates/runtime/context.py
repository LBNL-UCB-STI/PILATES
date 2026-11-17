from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from pilates.config.models import PilatesConfig
    from pilates.workflows.surface import EnabledWorkflowSurface
    from pilates.workspace import Workspace
    from pilates.workflows.state import WorkflowState


@dataclass(frozen=True)
class WorkflowRuntimeContext:
    """
    Narrow runtime context for orchestration and stage code.

    The intent is to keep stage/runtime helpers from repeatedly threading the
    same values when they always travel together:

    - validated settings
    - mutable workflow state
    - enabled workflow surface for the current run shape
    - workspace paths
    - optional archive commit callback
    - expected prefix steps for strict resume

    This is intentionally not a service locator. Tracker/scenario/coupler stay
    outside this object because they are execution handles, not run-shape
    context.
    """

    settings: "PilatesConfig"
    state: "WorkflowState"
    surface: "EnabledWorkflowSurface"
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
        surface: Optional["EnabledWorkflowSurface"] = None,
        on_archive_commit: Optional[Callable[[str], None]] = None,
        expected_prefix_steps: int = 0,
    ) -> "WorkflowRuntimeContext":
        """
        Build a runtime context with a guaranteed surface/state pairing.
        """
        if surface is None:
            from pilates.workflows.surface import build_enabled_workflow_surface

            surface = build_enabled_workflow_surface(settings, state=state)
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
            surface=surface,
            workspace=workspace,
            on_archive_commit=archive_cb,
            expected_prefix_steps=prefix_steps,
        )
