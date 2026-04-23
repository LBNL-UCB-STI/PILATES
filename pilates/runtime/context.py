from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pilates.config.models import PilatesConfig
    from pilates.workflows.surface import EnabledWorkflowSurface
    from pilates.workspace import Workspace
    from workflow_state import WorkflowState


@dataclass(frozen=True)
class WorkflowRuntimeContext:
    """
    Narrow runtime context for orchestration and stage code.

    The intent is to keep stage/runtime helpers from repeatedly threading the
    same four values when they always travel together:

    - validated settings
    - mutable workflow state
    - enabled workflow surface for the current run shape
    - workspace paths

    This is intentionally not a service locator. Tracker/scenario/coupler stay
    outside this object because they are execution handles, not run-shape
    context.
    """

    settings: "PilatesConfig"
    state: "WorkflowState"
    surface: "EnabledWorkflowSurface"
    workspace: "Workspace"

    @classmethod
    def from_parts(
        cls,
        *,
        settings: "PilatesConfig",
        state: "WorkflowState",
        workspace: "Workspace",
        surface: Optional["EnabledWorkflowSurface"] = None,
    ) -> "WorkflowRuntimeContext":
        """
        Build a runtime context with a guaranteed surface/state pairing.
        """
        if surface is None:
            from pilates.workflows.surface import build_enabled_workflow_surface

            surface = build_enabled_workflow_surface(settings, state=state)
        return cls(
            settings=settings,
            state=state,
            surface=surface,
            workspace=workspace,
        )


def ensure_workflow_runtime_context(
    *,
    context: Optional[WorkflowRuntimeContext] = None,
    settings: Optional["PilatesConfig"] = None,
    state: Optional["WorkflowState"] = None,
    workspace: Optional["Workspace"] = None,
    surface: Optional["EnabledWorkflowSurface"] = None,
) -> WorkflowRuntimeContext:
    """
    Return a runtime context from either an existing context or loose parts.

    This preserves compatibility for direct stage/helper tests that still pass
    the legacy ``settings/state/workspace/surface`` tuple while letting new
    orchestration call sites use one explicit context object.
    """
    if context is not None:
        return context

    missing = [
        name
        for name, value in (
            ("settings", settings),
            ("state", state),
            ("workspace", workspace),
        )
        if value is None
    ]
    if missing:
        raise TypeError(
            "WorkflowRuntimeContext requires either `context=` or explicit "
            + ", ".join(missing)
        )

    return WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
