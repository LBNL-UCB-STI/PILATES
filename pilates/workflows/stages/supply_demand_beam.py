from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING


from pilates.config.models import PilatesConfig
from pilates.runtime.context import WorkflowRuntimeContext
from pilates.utils.consist_types import (
    ScenarioWithCoupler,
)
from pilates.utils.coupler_helpers import (
    artifact_to_existing_path,
)
from pilates.utils.formatting import formatted_print
from pilates.beam.launch_config import BeamLaunchConfig
from pilates.workflows.coupler_namespace import (
    coupler_storage_value,
    set_coupler_from_artifact,
)
from pilates.utils.archive_paths import resolve_archive_run_dir
from pilates.utils.io import get_traffic_assignment_model
from pilates.utils.settings_helper import get as get_setting
from pilates.workflows.steps import (
    beam_full_skim,
    beam_postprocess,
    beam_preprocess,
    beam_run,
)
from pilates.workflows.step_execution import execute_step
from pilates.workflows.artifact_keys import (
    LINKSTATS,
    LINKSTATS_WARMSTART,
)
from pilates.workspace import Workspace
from pilates.workflows.state import WorkflowState
from pilates.workflows.state_helpers import resolve_forecast_year

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


_FAIL_AFTER_BEAM_RUN_ENV = "PILATES_FAIL_AFTER_BEAM_RUN"


@dataclass
class TrafficAssignmentPhaseInputs:
    """
    Inputs for one BEAM (traffic-assignment) iteration.

    Parameters
    ----------
    year : int
        Forecast year being simulated.
    iteration : int
        Supply-demand iteration index for the year.
    """

    year: int
    iteration: int


def _full_skim_run_schedule(settings: PilatesConfig) -> str:
    beam_cfg = settings.beam
    skim_cfg = beam_cfg.full_skim if beam_cfg is not None else None
    if skim_cfg is None:
        return "disabled"
    return skim_cfg.run_schedule


def _should_run_full_skim(
    settings: PilatesConfig,
    iteration: int,
    *,
    state: Optional[WorkflowState] = None,
) -> bool:
    schedule = _full_skim_run_schedule(settings)
    if schedule == "standalone":
        return True
    if schedule == "after_each_iteration":
        return True
    if schedule == "after_final_iteration":
        if isinstance(state, WorkflowState):
            total_iters = state.effective_supply_demand_iterations(
                resumed_iteration=iteration
            )
        else:
            total_iters = int(settings.run.supply_demand_iters)
            models = vars(settings.run).get("models")
            if (
                models is not None
                and models.activity_demand is None
                and total_iters > 1
            ):
                total_iters = max(1, iteration + 1)
        return iteration == total_iters - 1
    return False


def _maybe_fail_after_beam_run_for_canary(*, year: int, iteration: int) -> None:
    """Inject a controlled restart-canary failure after BEAM run completion."""
    if os.environ.get(_FAIL_AFTER_BEAM_RUN_ENV) != "1":
        return

    message = (
        "Injected failure after completed beam_run for restart canary "
        f"({_FAIL_AFTER_BEAM_RUN_ENV}=1, year={year}, iteration={iteration}). "
        "Unset the environment variable before restarting from run_state.yaml."
    )
    logger.error(message)
    raise RuntimeError(message)


def _run_beam_steps(
    *,
    scenario: ScenarioWithCoupler,
    year: int,
    iteration: int,
    context: WorkflowRuntimeContext,
) -> None:
    """Dispatch native BEAM steps with Consist caching and verified archival."""

    settings = context.settings
    state = context.state
    workspace = context.workspace
    archive_run_dir = resolve_archive_run_dir(workspace=workspace, state=state)

    _, _ = execute_step(
        scenario=scenario,
        definition=beam_preprocess,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=year,
        iteration=iteration,
        phase="preprocess",
    )
    run_inputs = beam_run.resolve_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=scenario.coupler,
    )
    run_result, _ = execute_step(
        scenario=scenario,
        definition=beam_run,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=year,
        iteration=iteration,
        phase="run",
        resolved_inputs=run_inputs,
    )

    logger.info("[beam] completed native beam_run run_id=%s", run_result.run.id)
    _maybe_fail_after_beam_run_for_canary(year=year, iteration=iteration)

    postprocess_inputs = beam_postprocess.resolve_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=scenario.coupler,
        beam_run_outputs=run_result.outputs,
    )
    postprocess_inputs.require_complete()

    _, _ = execute_step(
        scenario=scenario,
        definition=beam_postprocess,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=year,
        iteration=iteration,
        phase="postprocess",
        resolved_inputs=postprocess_inputs,
    )

    if _should_run_full_skim(settings, iteration, state=state):
        _run_beam_full_skim_step(
            scenario=scenario,
            year=year,
            iteration=iteration,
            context=context,
        )
    return None


def _run_beam_full_skim_step(
    *,
    scenario: ScenarioWithCoupler,
    year: int,
    iteration: int,
    context: WorkflowRuntimeContext,
    launch_config: BeamLaunchConfig | None = None,
) -> None:
    """Execute full skims through its native definition."""

    full_skim_inputs = beam_full_skim.resolve_inputs(
        settings=context.settings,
        state=context.state,
        workspace=context.workspace,
        coupler=scenario.coupler,
    )
    _, _ = execute_step(
        scenario=scenario,
        definition=beam_full_skim,
        settings=context.settings,
        state=context.state,
        workspace=context.workspace,
        stage="supply_demand",
        year=year,
        iteration=iteration,
        phase="full_skim",
        runtime_kwargs=(
            {"beam_launch_config": launch_config} if launch_config is not None else None
        ),
        resolved_inputs=full_skim_inputs,
    )
    return None


def _standalone_full_skim_preprocess_outputs(
    *,
    scenario: ScenarioWithCoupler,
    workspace: Workspace,
    settings: Optional[PilatesConfig] = None,
) -> Any:
    from pilates.beam.outputs import BeamPreprocessOutputs

    prepared: Dict[str, Path] = {}
    existing_warmstart = artifact_to_existing_path(
        coupler_storage_value(scenario.coupler, LINKSTATS_WARMSTART),
        workspace=workspace,
    )
    if existing_warmstart is not None:
        prepared[LINKSTATS_WARMSTART] = Path(existing_warmstart)
    if LINKSTATS_WARMSTART not in prepared:
        existing_linkstats = artifact_to_existing_path(
            coupler_storage_value(scenario.coupler, LINKSTATS),
            workspace=workspace,
        )
        if existing_linkstats is not None:
            prepared[LINKSTATS_WARMSTART] = Path(existing_linkstats)
    if LINKSTATS_WARMSTART not in prepared and settings is not None:
        from pilates.utils.beam_warmstart import resolve_initial_linkstats_path

        resolved_warmstart = resolve_initial_linkstats_path(settings, workspace)
        if resolved_warmstart is not None:
            prepared[LINKSTATS_WARMSTART] = Path(resolved_warmstart)
            if isinstance(scenario.coupler, dict):
                scenario.coupler[LINKSTATS_WARMSTART] = prepared[LINKSTATS_WARMSTART]
            else:
                set_coupler_from_artifact(
                    scenario.coupler,
                    LINKSTATS_WARMSTART,
                    prepared[LINKSTATS_WARMSTART],
                )
    if (
        settings is not None
        and get_setting(settings, "shared.geography.zones") is not None
    ):
        from pilates.beam.preprocessor import _prepare_beam_zone_shapefile

        _prepare_beam_zone_shapefile(workspace, settings)
    return BeamPreprocessOutputs(
        beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
        prepared_inputs=prepared,
    )


def _run_traffic_assignment_phase(
    *,
    scenario: ScenarioWithCoupler,
    inputs: TrafficAssignmentPhaseInputs,
    context: WorkflowRuntimeContext,
) -> None:
    """
    Run BEAM for a single supply-demand iteration.

    This sequences native BEAM definitions for one supply-demand iteration.

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
    inputs : TrafficAssignmentPhaseInputs
        Inputs required for this iteration.
    """
    settings = context.settings
    state = context.state
    workspace = context.workspace

    traffic_model = get_traffic_assignment_model(settings)
    if traffic_model is not None and traffic_model != "beam":
        raise NotImplementedError(
            f"Runtime traffic_assignment execution for model {traffic_model!r} "
            "is not yet wired in supply_demand."
        )

    formatted_print("TRAFFIC ASSIGNMENT MODEL")
    forecast_year = resolve_forecast_year(state)
    beam_year = int(forecast_year) if forecast_year is not None else inputs.year

    schedule = _full_skim_run_schedule(settings)
    if schedule == "standalone":
        preprocess_outputs = _standalone_full_skim_preprocess_outputs(
            scenario=scenario,
            workspace=workspace,
            settings=settings,
        )
        _run_beam_full_skim_step(
            scenario=scenario,
            year=beam_year,
            iteration=inputs.iteration,
            context=context,
            launch_config=None,
        )
    else:
        _run_beam_steps(
            scenario=scenario,
            year=beam_year,
            iteration=inputs.iteration,
            context=context,
        )

    state.complete_step(
        state.Stage.supply_demand_loop,
        inputs.iteration,
        state.Stage.traffic_assignment,
    )

    return None
