from __future__ import annotations

# Model-specific postprocessing step factory.
# Shared helpers/infrastructure are imported from shared.py.
from .shared import *  # noqa: F401,F403
from .shared import _decorate_step_with_consist  # noqa: F401

def make_postprocessing_step() -> Callable[..., None]:
    """
    Build the postprocessing step function.

    This step runs optional post-run cleanup/export routines such as event
    processing and copying outputs to external destinations.

    Returns
    -------
    callable
        Step function for postprocessing.
    """

    @require_common_runtime()
    def _run_postprocessing_step(
        *,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
    ) -> None:
        if "postprocessing" in settings:
            from pilates.postprocessing.postprocessor import (
                copy_outputs_to_mep,
                process_event_file,
            )

            process_event_file(settings, state.forecast_year, state.current_inner_iter)
            copy_outputs_to_mep(
                settings,
                state.forecast_year,
                state.current_inner_iter,
                workspace,
            )

    return _decorate_step_with_consist(
        step_func=_run_postprocessing_step,
        step_model="postprocessing",
        description="postprocessing workflow step",
        tags=["postprocessing"],
    )
