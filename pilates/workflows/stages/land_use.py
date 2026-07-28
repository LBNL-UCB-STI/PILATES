from __future__ import annotations

import os
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from pilates.runtime.context import WorkflowRuntimeContext
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils import consist_runtime as cr

from pilates.utils.formatting import formatted_print
from pilates.utils.coupler_helpers import archive_copy_now, flush_archive_queue
from pilates.workflows.steps import (
    urbansim_postprocess,
    urbansim_preprocess,
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
from pilates.utils.usim_h5 import ensure_usim_population_year_table_aliases

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def _population_source_snapshot_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_population_source{path.suffix}")


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
    inputs (including any pre-existing datastore), executes preprocess/run/
    postprocess steps, and then updates the UrbanSim datastore reference for
    downstream stages.

    The stage keeps two semantic datastore handles alive:
    - ``usim_datastore_base_h5`` for the static/exogenous baseline role
    - ``usim_datastore_h5`` for the current mutable handoff role

    The forecast/population-source role is snapshotted before postprocess
    rewrites the mutable current datastore so restart-sensitive provenance can
    still read an immutable exact-year source. The postprocess output datastore,
    when present, is preferred for the current-role handoff; otherwise the run
    output datastore is used.

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
    _, preprocess_outputs = execute_step(
        scenario=scenario,
        definition=urbansim_preprocess,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="land_use",
        year=year,
        iteration=getattr(state, "iteration", None),
        phase="preprocess",
    )
    del preprocess_outputs
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
    if run_outputs.usim_datastore_h5:
        forecast_output_path = Path(run_outputs.usim_datastore_h5)
        population_source_snapshot = _population_source_snapshot_path(
            forecast_output_path
        )
        shutil.copy2(forecast_output_path, population_source_snapshot)
        if state.forecast_year is not None and not state.is_start_year():
            alias_result = ensure_usim_population_year_table_aliases(
                h5_path=str(population_source_snapshot),
                year=state.forecast_year,
            )
            missing_root = alias_result.get("missing_root") or []
            if missing_root:
                logger.warning(
                    "Population-source snapshot %s is missing root tables needed "
                    "for year %s aliases: %s",
                    population_source_snapshot,
                    state.forecast_year,
                    missing_root,
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

    # Keep restart-critical UrbanSim H5 artifacts durable at stage boundaries.
    archive_copy_now(
        key=USIM_DATASTORE_BASE_H5,
        path=usim_inputs.get(USIM_DATASTORE_BASE_H5),
    )
    archive_copy_now(
        key=USIM_DATASTORE_CURRENT_H5,
        path=usim_inputs.get(USIM_DATASTORE_CURRENT_H5),
    )
    archive_copy_now(
        key=USIM_POPULATION_SOURCE_H5,
        path=usim_inputs.get(USIM_POPULATION_SOURCE_H5),
    )
    urbansim_settings = settings.urbansim
    if urbansim_settings is None:
        raise RuntimeError("UrbanSim config is required for the land use stage.")

    forecast_year = state.forecast_year if state.forecast_year is not None else year
    usim_forecast_output_path = os.path.join(
        workspace.get_usim_mutable_data_dir(),
        urbansim_settings.output_file_template.format(year=forecast_year),
    )
    archive_copy_now(
        key=f"usim_year_output_h5_{forecast_year}",
        path=usim_forecast_output_path,
    )
    flush_archive_queue(timeout=300, fail_on_timeout=False)

    return LandUseToSupplyDemandHandoff.from_mapping(usim_inputs)
