from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

from pilates.config.models import PilatesConfig
from pilates.runtime.context import WorkflowRuntimeContext
from pilates.runtime.archive_paths import archive_fallback_path
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import CouplerProtocol, ScenarioWithCoupler
from pilates.utils.coupler_helpers import (
    archive_copy_destination,
    archive_copy_now,
    flush_archive_queue,
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
from pilates.workspace import Workspace
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


def select_atlas_usim_input_path(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    fallback_current_path: Optional[Union[str, os.PathLike]],
    fallback_default_path: Optional[Union[str, os.PathLike]],
    prefer_forecast_output: bool = True,
) -> str:
    """
    Resolve the UrbanSim datastore path used by ATLAS preprocess.

    Precedence (when ``prefer_forecast_output=True``):
    1. Forecast output datastore (year-scoped snapshots).
    2. Current datastore resolved from UrbanSim input builder.
    3. Legacy default path.

    Precedence (when ``prefer_forecast_output=False``):
    1. Current datastore resolved from UrbanSim input builder.
    2. Legacy default path.
    3. Forecast output datastore.
    """
    usim_dir = workspace.get_usim_mutable_data_dir()

    forecast_year = state.forecast_year
    urbansim_settings = settings.urbansim
    if forecast_year is None:
        raise RuntimeError(
            "WorkflowState.forecast_year must be set before ATLAS input resolution."
        )
    if urbansim_settings is None:
        raise RuntimeError(
            "UrbanSim config is required for the vehicle ownership stage."
        )

    forecast_output_path = os.path.join(
        usim_dir,
        urbansim_settings.output_file_template.format(year=forecast_year),
    )
    forecast_output_archive_path = archive_fallback_path(
        state=state,
        workspace=workspace,
        local_path=Path(forecast_output_path),
    )

    current_candidate = (
        os.fspath(fallback_current_path)
        if isinstance(fallback_current_path, (str, os.PathLike))
        else None
    )
    default_candidate = (
        os.fspath(fallback_default_path)
        if isinstance(fallback_default_path, (str, os.PathLike))
        else None
    )

    if prefer_forecast_output:
        candidates = [
            forecast_output_path,
            os.fspath(forecast_output_archive_path)
            if forecast_output_archive_path is not None
            else None,
            current_candidate,
            default_candidate,
        ]
    else:
        candidates = [
            current_candidate,
            default_candidate,
            forecast_output_path,
            os.fspath(forecast_output_archive_path)
            if forecast_output_archive_path is not None
            else None,
        ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    for candidate in candidates[1:]:
        if candidate:
            return candidate
    return forecast_output_path


def _validate_atlas_subyear_usim_datastore(
    *,
    atlas_year: int,
    start_year: int,
    forecast_year: int,
    selected_path: str,
    settings: PilatesConfig,
    state: WorkflowState,
) -> None:
    """
    Reject later ATLAS subyears that resolve to a non-forecast UrbanSim datastore.

    Dynamic ATLAS subyears after ``start_year`` must read from the forecast-year
    UrbanSim output datastore (for example ``model_data_2029.h5`` for the 2029
    land-use interval). Falling back to an older datastore such as
    ``model_data_2023.h5`` silently degrades table selection and produces
    incorrect restart behavior.
    """
    if atlas_year <= start_year:
        return

    urbansim_settings = settings.urbansim
    if urbansim_settings is None:
        return

    expected_name = os.path.basename(
        urbansim_settings.output_file_template.format(year=forecast_year)
    )
    selected_name = os.path.basename(os.fspath(selected_path))
    if selected_name == expected_name:
        return

    restart_note = (
        " during restart resume"
        if bool(getattr(state, "is_restart_run", False))
        else ""
    )
    raise RuntimeError(
        "ATLAS subyear datastore resolution mismatch%s: year %s requires forecast-year "
        "UrbanSim datastore %r, but resolved %r. This would cause ATLAS to fall back "
        "to older year-scoped tables instead of failing cleanly."
        % (restart_note, atlas_year, expected_name, selected_name)
    )


def run_vehicle_ownership_stage(
    *,
    scenario: ScenarioWithCoupler,
    coupler: CouplerProtocol,
    year: int,
    build_atlas_static_inputs_fallback: Callable[
        [Workspace], Mapping[str, Union[str, os.PathLike]]
    ],
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
    build_atlas_static_inputs_fallback : Callable[[Workspace], Mapping[str, Union[str, os.PathLike]]]
        Fallback builder for static ATLAS inputs when not already present in
        the workspace input registry.
    """
    settings = context.settings
    state = context.state
    workspace = context.workspace
    del build_atlas_static_inputs_fallback

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
                    archive_dest = archive_copy_destination(
                        key=USIM_POPULATION_SOURCE_H5,
                        path=atlas_postprocess_outputs.usim_datastore_h5,
                    )
                    archive_copy_now(
                        key=USIM_POPULATION_SOURCE_H5,
                        path=atlas_postprocess_outputs.usim_datastore_h5,
                        force=True,
                    )
                    if archive_dest is not None and os.path.exists(archive_dest):
                        _validate_population_h5_for_activitysim_year(
                            path=archive_dest,
                            year=atlas_year,
                            context=f"ATLAS postprocess archived output y{atlas_year}",
                        )

            atlas_input_root = workspace.get_atlas_mutable_input_dir()
            atlas_year_input_dir = os.path.join(atlas_input_root, f"year{atlas_year}")
            archive_copy_now(
                key=f"atlas_input_year_dir_{atlas_year}",
                path=atlas_year_input_dir,
            )
            for base_dir in (
                atlas_year_input_dir,
                atlas_input_root,
                workspace.get_atlas_output_dir(),
            ):
                for filename in ("vehicles_output.RData", "households_output.RData"):
                    archive_copy_now(
                        key=f"atlas_rdata_{atlas_year}",
                        path=os.path.join(base_dir, filename),
                    )
            # Ensure year N artifacts are durable before year N+1 consumes them.
            flush_archive_queue(timeout=300, fail_on_timeout=False)
        except Exception:
            from pilates.utils.failure_handling import persist_state_on_error

            persist_state_on_error(state, f"ATLAS year {atlas_year}")
            sys.exit(1)
