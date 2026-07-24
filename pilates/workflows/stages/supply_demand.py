from __future__ import annotations

import logging
from typing import Callable, Optional

from pilates.runtime.context import WorkflowRuntimeContext
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.coupler_helpers import flush_archive_queue
from pilates.utils.formatting import formatted_print
from .handoffs import LandUseToSupplyDemandHandoff

from .supply_demand_activity import (
    ActivityDemandPhaseInputs,
    _run_activity_demand_phase,
)
from .supply_demand_beam import (
    TrafficAssignmentPhaseInputs,
    _run_traffic_assignment_phase,
    beam_checkpoint_resume_requested,
)

logger = logging.getLogger(__name__)


def _rewind_uncheckpointed_activitysim_restart(
    *,
    settings,
    state,
    has_committed_beam_checkpoint: bool,
) -> bool:
    """Re-run ActivitySim rather than reconstructing an uncommitted handoff."""

    if (
        not state.is_restart_run
        or settings.run.models.activity_demand is None
        or has_committed_beam_checkpoint
        or state.current_major_stage != state.Stage.supply_demand_loop
        or state.should_run(
            state.Stage.supply_demand_loop,
            state.iteration,
            state.Stage.activity_demand,
        )
    ):
        return False

    state.current_sub_stage = state.Stage.activity_demand
    state.sub_stage_progress = None
    state.write_state()
    logger.warning(
        "[supply_demand][restart] no committed BEAM checkpoint exists; "
        "rewinding iteration=%s to ActivitySim instead of reconstructing a handoff.",
        state.iteration,
    )
    return True


def run_supply_demand_stage(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    year: int,
    handoff: LandUseToSupplyDemandHandoff,
    on_iteration_boundary: Optional[Callable[[int], None]] = None,
    context: WorkflowRuntimeContext,
) -> None:
    """
    Run the supply-demand loop (ActivitySim + BEAM) for the year.

    This stage iterates the activity-demand and traffic-assignment sub-stages:
    ActivitySim preprocess -> compile (once per year) -> run/postprocess produces
    household/person/activity outputs, which feed BEAM. BEAM then runs traffic
    assignment and postprocessing, producing skims and other artifacts that can
    be fed into the next iteration.

    Parameters
    ----------
    scenario : ScenarioWithCoupler
        Consist scenario wrapper used to execute steps with provenance.
    state : WorkflowState
        Workflow state tracking iterations and sub-stage completion.
    settings : PilatesConfig
        Validated run configuration.
    workspace : Workspace
        Workspace managing run-local inputs/outputs.
    coupler : CouplerProtocol
        Coupler used to read/write artifacts across steps.
    year : int
        Forecast year being simulated.
    on_iteration_boundary : Optional[Callable[[int], None]], optional
        Callback invoked after each outer iteration completes. Intended for
        orchestration-level safe-point actions such as DB snapshots.

    A committed ``beam_run_completed -> beam_postprocess`` checkpoint is the
    only mid-stage restart boundary. Without one, an in-progress restart safely
    rewinds to ActivitySim for that iteration; it never reconstructs an
    ActivitySim handoff from a manifest.
    """
    settings = context.settings
    state = context.state
    logger.info(
        "[supply_demand] year=%s run_id=%s handoff_keys=%s",
        year,
        cr.current_run_id(),
        sorted(handoff.to_input_mapping().keys()),
    )

    total_iters = settings.run.supply_demand_iters
    if settings.run.models.activity_demand is None and total_iters > 1:
        resumed_iteration = int(getattr(state, "iteration", 0) or 0)
        clamped_total_iters = max(1, resumed_iteration + 1)
        logger.warning(
            "BEAM-only supply_demand_iters=%d. Clamping outer supply-demand "
            "iterations to %d because BEAM already manages its own internal "
            "iterations.",
            total_iters,
            clamped_total_iters,
        )
        total_iters = clamped_total_iters
    _rewind_uncheckpointed_activitysim_restart(
        settings=settings,
        state=state,
        has_committed_beam_checkpoint=beam_checkpoint_resume_requested(state=state),
    )
    for i in range(state.iteration, total_iters):
        state.iteration = i
        formatted_print(f"SUPPLY/DEMAND ITERATION {i + 1}/{total_iters}")
        committed_beam_resume = beam_checkpoint_resume_requested(state=state)

        # C1. ACTIVITY DEMAND
        if committed_beam_resume:
            logger.info(
                "[supply_demand][restart] committed BEAM checkpoint dispatch "
                "preempts ActivitySim recovery for iteration=%s",
                i,
            )
        elif state.should_run(
            state.Stage.supply_demand_loop,
            i,
            state.Stage.activity_demand,
        ):
            activity_demand_inputs = ActivityDemandPhaseInputs(
                year=year,
                iteration=i,
            )
            _run_activity_demand_phase(
                scenario=scenario,
                inputs=activity_demand_inputs,
                context=context,
            )
        elif settings.run.models.activity_demand is not None:
            raise RuntimeError(
                "ActivitySim is skipped without a committed BEAM checkpoint "
                "after restart rewind; refusing to reconstruct a mid-stage handoff."
            )

        # C2. TRAFFIC ASSIGNMENT
        if state.should_run(
            state.Stage.supply_demand_loop,
            i,
            state.Stage.traffic_assignment,
        ):
            traffic_inputs = TrafficAssignmentPhaseInputs(
                year=year,
                iteration=i,
            )
            _run_traffic_assignment_phase(
                scenario=scenario,
                inputs=traffic_inputs,
                context=context,
            )
        # Year/iteration boundary durability checkpoint for restart artifacts.
        flush_archive_queue(timeout=300, fail_on_timeout=False)

        if on_iteration_boundary is not None:
            on_iteration_boundary(i)

    # The final substage completion may already have advanced the workflow
    # into the next year/major stage. Only complete the major stage here when
    # the state is still inside the supply-demand loop.
    if state.current_major_stage == state.Stage.supply_demand_loop:
        state.complete_step(state.Stage.supply_demand_loop)
