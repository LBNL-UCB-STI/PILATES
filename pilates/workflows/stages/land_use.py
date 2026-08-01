from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pilates.runtime.context import WorkflowRuntimeContext
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils import consist_runtime as cr

from pilates.utils.formatting import formatted_print
from pilates.workflows.steps import (
    urbansim_postprocess,
    urbansim_run,
)
from pilates.workflows.coupler_namespace import resolve_coupler_value
from pilates.workflows.step_execution import execute_step
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_FORECAST_OUTPUT,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.stages.handoffs import LandUseToSupplyDemandHandoff

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def run_land_use_stage(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    year: int,
    context: WorkflowRuntimeContext,
) -> LandUseToSupplyDemandHandoff:
    """
    Run the UrbanSim land-use stage and return updated UrbanSim inputs.

    This stage is responsible for land-use evolution. It prepares UrbanSim
    inputs (including any pre-existing datastore), executes run/postprocess
    steps, and then updates the UrbanSim datastore reference for downstream
    stages.

    The stage keeps two semantic datastore handles alive:
    - ``usim_datastore_base_h5`` for the static/exogenous baseline role
    - ``usim_datastore_h5`` for the current mutable handoff role

    The postprocess step owns the immutable forecast/population-source snapshot
    used by downstream stages. The postprocess output datastore, when present,
    is preferred for the current-role handoff; otherwise the run output datastore
    is used.

    Parameters
    ----------
    scenario : ScenarioWithCoupler
        Consist scenario wrapper used to execute steps with provenance.
    state : WorkflowState
        Workflow state for year/stage coordination.
    settings : PilatesConfig
        Validated run configuration.
    workspace : Workspace
        Workspace managing run-local inputs/outputs.
    coupler : CouplerProtocol
        Consist coupler for reading/writing artifacts across steps.
    year : int
        Forecast year being simulated.
    Returns
    -------
    LandUseToSupplyDemandHandoff
        Updated UrbanSim datastore handoff for downstream supply-demand stages.
    """
    settings = context.settings
    state = context.state
    workspace = context.workspace

    formatted_print(f"LAND USE MODEL FOR YEAR {year}")
    logger.info("[land_use] year=%s run_id=%s", year, cr.current_run_id())

    # Definitions own semantic selection.  The stage intentionally sequences
    # native executions only; it neither constructs bindings nor replays output
    # records through a holder.
    _, run_outputs = execute_step(
        scenario=scenario,
        definition=urbansim_run,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="land_use",
        year=year,
        iteration=getattr(state, "iteration", None),
        phase="run",
    )
    _, postprocess_outputs = execute_step(
        scenario=scenario,
        definition=urbansim_postprocess,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="land_use",
        year=year,
        iteration=getattr(state, "iteration", None),
        phase="postprocess",
    )

    usim_inputs: dict[str, str] = {}
    population_source_snapshot = postprocess_outputs.processed_outputs.get(
        USIM_POPULATION_SOURCE_H5
    )
    if population_source_snapshot is None:
        raise RuntimeError(
            "UrbanSim postprocess did not produce the population-source snapshot"
        )
    usim_inputs[USIM_FORECAST_OUTPUT] = str(population_source_snapshot)
    usim_inputs[USIM_POPULATION_SOURCE_H5] = str(population_source_snapshot)
    if postprocess_outputs.usim_datastore_h5:
        usim_inputs[USIM_DATASTORE_CURRENT_H5] = str(
            postprocess_outputs.usim_datastore_h5
        )
    elif run_outputs.usim_datastore_h5:
        usim_inputs[USIM_DATASTORE_CURRENT_H5] = str(run_outputs.usim_datastore_h5)

    # Preserve the base-role handle as the static/exogenous input contract. If
    # current/base collapsed earlier in the run, keep that role explicit here.
    if (
        USIM_DATASTORE_BASE_H5 not in usim_inputs
        and USIM_DATASTORE_CURRENT_H5 in usim_inputs
    ):
        base = resolve_coupler_value(coupler, USIM_DATASTORE_BASE_H5).value
        usim_inputs[USIM_DATASTORE_BASE_H5] = str(
            base if base is not None else usim_inputs[USIM_DATASTORE_CURRENT_H5]
        )

    return LandUseToSupplyDemandHandoff.from_mapping(usim_inputs)
