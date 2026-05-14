from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
from pilates.runtime.context import WorkflowRuntimeContext
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.coupler_helpers import archive_copy_now, flush_archive_queue
from pilates.utils.formatting import formatted_print
from pilates.workflows.orchestration import ManifestConfig
from pilates.workflows.steps import StepOutputsHolder
from pilates.workspace import Workspace
from .handoffs import LandUseToSupplyDemandHandoff

from .supply_demand_activity import (
    ActivityDemandPhaseInputs,
    _run_activity_demand_phase,
)
from .supply_demand_beam import (
    TrafficAssignmentPhaseInputs,
    _run_traffic_assignment_phase,
)
from .supply_demand_resume import (
    _restore_activity_demand_outputs_for_resume,
    _restore_supply_demand_usim_inputs_for_resume,
)

logger = logging.getLogger(__name__)


def run_supply_demand_stage(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    year: int,
    handoff: Optional[LandUseToSupplyDemandHandoff] = None,
    usim_inputs: Optional[Mapping[str, Union[str, os.PathLike]]] = None,
    build_manifest_path: Callable[[Workspace, int, int], os.PathLike],
    on_iteration_boundary: Optional[Callable[[int], None]] = None,
    context: WorkflowRuntimeContext,
) -> None:
    """
    Run the supply-demand loop (ActivitySim + BEAM) for the year.

    This stage iterates the activity-demand and traffic-assignment sub-stages:
    ActivitySim preprocess -> compile (once per year) -> run/postprocess produces
    household/person/activity outputs, which feed BEAM. BEAM then runs traffic
    assignment and postprocessing, producing skims and other artifacts that can
    be fed into the next iteration. Manifest checkpointing is used around
    ActivitySim preprocess and run/postprocess to support restart/resume without
    re-running completed steps.

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
    handoff : Optional[LandUseToSupplyDemandHandoff]
        Typed UrbanSim datastore handoff used to seed ActivitySim preprocessing
        when land use ran earlier in the year.
    usim_inputs : Optional[Mapping[str, Union[str, os.PathLike]]]
        Compatibility alias for older callers still passing a raw mapping.
    build_manifest_path : Callable[[Workspace, int, int], os.PathLike]
        Factory for per-year/per-iteration manifest file locations.
    on_iteration_boundary : Optional[Callable[[int], None]], optional
        Callback invoked after each outer iteration completes. Intended for
        orchestration-level safe-point actions such as DB snapshots.

    State Machine & Resume Behavior
    -------------------------------
    The stage maintains iteration state via ``state.iteration`` and records
    per-step completion in a manifest under ``.workflow/``. On resume:

    1. **ActivityDemand Phase**:
       - If ``should_run(...)`` returns True, preprocess -> compile (once per year)
         -> run/postprocess executes and updates coupler outputs.
       - If it returns False, ActivitySim steps are skipped and downstream
         phases must rely on previously-produced artifacts.

    2. **TrafficAssignment Phase**:
       - Requires ActivitySim outputs or prior BEAM outputs to seed inputs.
         If neither is available on iteration 0, the stage raises an error.
       - For later iterations, warm-start linkstats may be pulled from prior
         BEAM outputs or from an initial warm-start file.

    3. **Convergence Check**:
       - Checked at each iteration boundary. If convergence is detected, the
         loop exits early and the stage marks completion.

    Warnings
    --------
    - Removing ``.workflow/`` manifests forces all iterations to re-run.
    - If resuming mid-iteration, ensure coupler artifacts for ActivitySim
      outputs are available before running BEAM.
    - Do not mutate coupler keys between iterations; they carry warm-start
      state to the next iteration.
    """
    settings = context.settings
    state = context.state
    workspace = context.workspace
    surface = context.surface
    if handoff is None:
        handoff = LandUseToSupplyDemandHandoff.from_mapping(usim_inputs)
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
    previous_beam_outputs: Optional[Dict[str, Any]] = None
    resumed_usim_inputs = handoff.to_input_mapping()
    if bool(getattr(state, "is_restart_run", False)) and not resumed_usim_inputs:
        # On resumed runs, land use may be enabled globally but already complete
        # for the current year. In that case the skipped land-use stage will not
        # republish the year-scoped UrbanSim H5 roles that ActivitySim expects,
        # so rebuild them from deterministic workspace/archive paths here.
        resumed_usim_inputs = _restore_supply_demand_usim_inputs_for_resume(
            coupler=coupler,
            workspace=workspace,
            state=state,
            settings=settings,
        )

    for i in range(state.iteration, total_iters):
        state.iteration = i
        formatted_print(f"SUPPLY/DEMAND ITERATION {i + 1}/{total_iters}")
        activity_demand_outputs = None
        outputs_holder = StepOutputsHolder()
        manifest_path = build_manifest_path(workspace, year, i)
        manifest_config = ManifestConfig(path=Path(manifest_path))

        # C1. ACTIVITY DEMAND
        if state.should_run(
            state.Stage.supply_demand_loop,
            i,
            state.Stage.activity_demand,
        ):
            activity_demand_inputs = ActivityDemandPhaseInputs(
                year=year,
                iteration=i,
                usim_inputs=resumed_usim_inputs,
            )
            activity_demand_outputs = _run_activity_demand_phase(
                scenario=scenario,
                coupler=coupler,
                inputs=activity_demand_inputs,
                outputs_holder=outputs_holder,
                manifest_config=manifest_config,
                context=context,
                surface=surface,
            ).activity_demand_outputs
        elif (
            settings.run.models.activity_demand is None
            and outputs_holder.activitysim_postprocess is None
        ):
            # Satisfy BEAM preprocess dependencies when ActivitySim is disabled.
            outputs_holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
                usim_datastore_h5=None,
                asim_output_dir=None,
            )
        elif outputs_holder.activitysim_postprocess is None:
            activity_demand_outputs = _restore_activity_demand_outputs_for_resume(
                scenario=scenario,
                coupler=coupler,
                workspace=workspace,
                outputs_holder=outputs_holder,
                state=state,
                settings=settings,
                tracker=scenario.tracker,
                manifest_path=Path(manifest_path),
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
                activity_demand_outputs=activity_demand_outputs,
                previous_beam_outputs=previous_beam_outputs,
            )
            previous_beam_outputs = _run_traffic_assignment_phase(
                scenario=scenario,
                coupler=coupler,
                inputs=traffic_inputs,
                outputs_holder=outputs_holder,
                context=context,
                surface=surface,
            ).previous_beam_outputs

        if os.path.exists(manifest_path):
            archive_copy_now(
                key="workflow_manifest",
                path=manifest_path,
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
