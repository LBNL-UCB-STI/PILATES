from __future__ import annotations

import logging
import os
import sys
from typing import Any, Union

from pilates.runtime.context import WorkflowRuntimeContext
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.coupler_helpers import (
    set_coupler_from_artifact,
)
from pilates.utils.usim_h5 import resolve_usim_population_table_paths
from pilates.workflows.atlas_state import AtlasSubState
from pilates.workflows.coupler_namespace import resolve_coupler_value
from pilates.workflows.steps import (
    atlas_postprocess,
    atlas_preprocess,
    atlas_run,
)
from pilates.workflows.step_execution import execute_step
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_CURRENT_H5,
    USIM_POPULATION_SOURCE_H5,
)
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


def _validate_population_h5_for_activitysim_year(
    *,
    path: Union[str, os.PathLike],
    year: int,
    context: str,
) -> None:
    resolved = resolve_usim_population_table_paths(
        h5_path=os.fspath(path),
        year=year,
        require_exact_year=True,
    )
    logger.info(
        "Validated ActivitySim population H5 for %s: path=%s year=%s tables=%s",
        context,
        os.fspath(path),
        year,
        resolved,
    )


def _publish_current_h5_alias(
    *,
    coupler: CouplerProtocol,
    fallback_path: Union[str, os.PathLike],
) -> Any:
    """Alias the current H5 role to ATLAS's published population artifact."""
    population_source = resolve_coupler_value(
        coupler,
        USIM_POPULATION_SOURCE_H5,
    ).value
    if population_source is None:
        # Keep a path fallback for runtimes without an active output logger.
        population_source = os.fspath(fallback_path)
        set_coupler_from_artifact(
            coupler,
            USIM_POPULATION_SOURCE_H5,
            None,
            fallback=population_source,
        )
    set_coupler_from_artifact(
        coupler,
        USIM_DATASTORE_CURRENT_H5,
        population_source,
        fallback=os.fspath(fallback_path),
    )
    return population_source


def _atlas_sub_years(state: WorkflowState) -> list[int]:
    """
    Return ATLAS sub-years within the current workflow interval.

    ATLAS advances in biannual increments. Keep years bounded to the parent
    interval and never overshoot ``state.forecast_year``.
    """
    if state.year is None:
        raise RuntimeError("WorkflowState.year must be set before running ATLAS.")
    forecast_year = state.forecast_year
    if forecast_year is None:
        raise RuntimeError(
            "WorkflowState.forecast_year must be set before running ATLAS."
        )
    years = [state.year]
    if forecast_year <= state.year:
        return years
    years.extend(range(state.year + 2, forecast_year + 1, 2))
    return years


def run_vehicle_ownership_stage(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    year: int,
    context: WorkflowRuntimeContext,
) -> None:
    """
    Run the ATLAS vehicle ownership stage for the current forecast year.

    This stage executes ATLAS preprocess/run/postprocess for one or more
    sub-years, using the UrbanSim datastore (start-year or forecast output) as
    the primary input. It also wires in any static ATLAS inputs and handles
    sub-year execution via AtlasSubState.

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
        Coupler used to read/write artifacts across steps.
    year : int
        Forecast year being simulated.
    """
    settings = context.settings
    state = context.state
    workspace = context.workspace

    logger.info(
        "[vehicle_ownership] year=%s forecast_year=%s run_id=%s",
        year,
        state.forecast_year,
        cr.current_run_id(),
    )

    forecast_year = state.forecast_year
    urbansim_settings = settings.urbansim
    if forecast_year is None:
        raise RuntimeError(
            "WorkflowState.forecast_year must be set before running vehicle ownership."
        )
    if urbansim_settings is None:
        raise RuntimeError(
            "UrbanSim config is required for the vehicle ownership stage."
        )

    yrs = _atlas_sub_years(state)

    for atlas_year in yrs:
        atlas_state = AtlasSubState(state, atlas_year)
        try:
            # Native definitions select their semantic roles through the current
            # scenario coupler.  The stage owns only sub-year ordering and the
            # typed output policy below.
            _, preprocess_outputs = execute_step(
                scenario=scenario,
                definition=atlas_preprocess,
                settings=settings,
                state=atlas_state,
                workspace=workspace,
                stage="vehicle_ownership",
                year=atlas_year,
                iteration=getattr(state, "iteration", None),
                phase="preprocess",
            )
            del preprocess_outputs
            _, run_outputs = execute_step(
                scenario=scenario,
                definition=atlas_run,
                settings=settings,
                state=atlas_state,
                workspace=workspace,
                stage="vehicle_ownership",
                year=atlas_year,
                iteration=getattr(state, "iteration", None),
                phase="run",
            )
            del run_outputs
            _, atlas_postprocess_outputs = execute_step(
                scenario=scenario,
                definition=atlas_postprocess,
                settings=settings,
                state=atlas_state,
                workspace=workspace,
                stage="vehicle_ownership",
                year=atlas_year,
                iteration=getattr(state, "iteration", None),
                phase="postprocess",
            )
            if atlas_postprocess_outputs.usim_datastore_h5 is not None:
                _publish_current_h5_alias(
                    coupler=coupler,
                    fallback_path=atlas_postprocess_outputs.usim_datastore_h5,
                )
                if not atlas_state.is_start_year():
                    _validate_population_h5_for_activitysim_year(
                        path=atlas_postprocess_outputs.usim_datastore_h5,
                        year=atlas_year,
                        context=f"ATLAS postprocess local output y{atlas_year}",
                    )
        except Exception:
            from pilates.utils.failure_handling import persist_state_on_error

            persist_state_on_error(state, f"ATLAS year {atlas_year}")
            sys.exit(1)
