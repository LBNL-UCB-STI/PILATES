from __future__ import annotations

import logging
from pathlib import Path

from pilates.runtime.context import WorkflowRuntimeContext
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.workspace import Workspace

from pilates.workflows.orchestration import ManifestConfig, StepRef, run_workflow
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_impacts_presim_step,
)

logger = logging.getLogger(__name__)


def _build_impacts_presim_manifest_path(workspace: Workspace, year: int) -> Path:
    return Path(workspace.full_path) / ".workflow" / f"impacts_presim_year_{year}.yaml"


def run_impacts_presim_stage(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    year: int,
    context: WorkflowRuntimeContext,
) -> None:
    """
    Run the IMPACTS presim stage (EMFAC activities + fleet) before BEAM.

    This stage has no dependency on coupler inputs or BEAM outputs; it derives
    its own settings from the PILATES configuration and writes its outputs
    directly to the workspace output tree.

    Parameters
    ----------
    scenario : ScenarioWithCoupler
        Consist scenario wrapper used to execute steps with provenance.
    coupler : CouplerProtocol
        Coupler forwarded to shared orchestration utilities.
    year : int
        Forecast year being processed.
    context : WorkflowRuntimeContext
        Shared runtime context (settings, state, workspace).
    """
    settings = context.settings
    state = context.state
    workspace = context.workspace

    if not getattr(settings, "impacts_enabled", False):
        return

    logger.info("[impacts_presim] year=%s run_id=%s", year, cr.current_run_id())

    outputs_holder = StepOutputsHolder()
    manifest_config = ManifestConfig(
        path=_build_impacts_presim_manifest_path(workspace=workspace, year=year)
    )
    steps = [
        StepRef(
            name="impacts_presim",
            step_func=make_impacts_presim_step(coupler=coupler, outputs_holder=outputs_holder),
            cache_mode="overwrite",
            load_inputs=False,
            model="impacts_presim",
            year=year,
            iteration=getattr(state, "iteration", None),
            phase="run",
        )
    ]

    run_workflow(
        stage_name="impacts_presim",
        steps=steps,
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix=str(year),
        iteration=getattr(state, "iteration", 0),
        manifest_config=manifest_config,
    )
