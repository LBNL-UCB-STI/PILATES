"""Imperative PILATES workflow orchestrator over typed ``SimState``."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

from consist import Artifact

from pilates.beam import beam_exchange
from pilates.config.models import PilatesConfig
from pilates.urbansim.preprocessor import resolve_urbansim_initial_datastore_source_path
from pilates.utils import consist_runtime as cr
from pilates.utils.beam_warmstart import resolve_initial_linkstats_path
from pilates.utils.consist_types import ScenarioWithCoupler
from pilates.utils.formatting import formatted_print
from pilates.utils.io import (
    get_activity_demand_model,
    get_land_use_model,
    get_traffic_assignment_model,
    get_vehicle_ownership_model,
)
from pilates.utils.settings_helper import get as get_setting
from pilates.utils.state_access import (
    current_year,
    iteration_index,
    uses_input_datastore,
)
from pilates.utils.usim_h5 import resolve_usim_population_table_paths
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_FULL_SKIMS,
    BEAM_HOUSEHOLDS_IN,
    BEAM_NETWORK_FINAL,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    FINAL_SKIMS_OMX,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_INPUT_ARCHIVE_PREFIX,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.sim_state import (
    ArtifactRef,
    DemandState,
    LandUseState,
    NetworkState,
    SimState,
)
from pilates.workflows.state import WorkflowState
from pilates.workflows.state_helpers import (
    requires_prior_beam_linkstats_handoff,
    requires_prior_beam_skim_handoff,
    resolve_forecast_year,
)
from pilates.workflows.step_execution import _HasDbMutationSeq, execute_step
from pilates.workflows.steps import (
    activitysim_postprocess,
    activitysim_preprocess,
    activitysim_run,
    atlas_postprocess,
    atlas_preprocess,
    atlas_run,
    beam_full_skim,
    beam_postprocess,
    beam_preprocess,
    beam_run,
    urbansim_postprocess,
    urbansim_run,
)
from pilates.workflows.steps.postprocessing import (
    postprocessing as postprocessing_step,
)
from pilates.workspace import Workspace

logger = logging.getLogger(__name__)

_FAIL_AFTER_BEAM_RUN_ENV = "PILATES_FAIL_AFTER_BEAM_RUN"


@runtime_checkable
class _WorkspaceRootLike(Protocol):
    full_path: str | Path | None


@runtime_checkable
class _HasCoupler(Protocol):
    coupler: Any


@runtime_checkable
class _HasRunModels(Protocol):
    models: Any


@runtime_checkable
class _SnapshotManagerLike(Protocol):
    def maybe_snapshot_interval(self, *, reason: str) -> Any: ...
    def on_outer_iteration_boundary(self, *, year: int, iteration: int) -> Any: ...


@runtime_checkable
class _AtlasParentState(Protocol):
    year: int | None
    forecast_year: int | None
    start_year: int | None
    full_settings: Any | None
    iteration: int | None
    __dict__: dict[str, Any]


class AtlasSubState:
    """Workflow state for a single ATLAS sub-year within a forecast interval."""

    def __init__(self, parent_state: _AtlasParentState, year: int) -> None:
        interval_start_year = parent_state.year
        if interval_start_year is None:
            raise RuntimeError(
                "WorkflowState.year must be set before constructing AtlasSubState."
            )
        self.__dict__ = parent_state.__dict__.copy()
        self.year = year
        self.current_year = year
        self.forecast_year = year
        self.main_forecast_year = parent_state.forecast_year
        self.start_year = parent_state.start_year
        self.atlas_interval_start_year = interval_start_year
        self.full_settings = parent_state.full_settings
        self.is_start_year = lambda: year == self.atlas_interval_start_year
        self.atlas_usim_datastore_h5: Any | None = None
        self.atlas_usim_datastore_base_h5: Any | None = None
        self._parent_state = parent_state

    @property
    def iteration(self) -> int:
        it = self._parent_state.iteration
        return int(it) if it is not None else 1

    def set_sub_stage_progress(self, sub_stage_progress: str) -> None:
        """No-op compatibility hook for ATLAS sub-stage progress."""
        del sub_stage_progress


@dataclass(slots=True)
class _OrchestratorEnv:
    """Shared execution environment threaded through ``run_pilates()``."""

    scenario: ScenarioWithCoupler
    settings: PilatesConfig
    state: WorkflowState
    workspace: Workspace
    snapshot_manager: _SnapshotManagerLike | None = None
    on_activity_demand_boundary: Optional[Callable[[int, int], None]] = None
    last_synced: dict[str, ArtifactRef] = field(default_factory=dict)
    last_snapshotted_seq: int = 0

    def maybe_on_activity_demand_boundary(self, *, year: int, iteration: int) -> None:
        if self.on_activity_demand_boundary is None:
            return
        if isinstance(self.workspace, _HasDbMutationSeq):
            seq = self.workspace.db_mutation_seq
            if seq == self.last_snapshotted_seq:
                return
            self.on_activity_demand_boundary(year, iteration)
            self.last_snapshotted_seq = seq
            return
        self.on_activity_demand_boundary(year, iteration)

    def maybe_on_outer_iteration_boundary(self, *, year: int, iteration: int) -> None:
        if self.snapshot_manager is None:
            return
        if isinstance(self.workspace, _HasDbMutationSeq):
            seq = self.workspace.db_mutation_seq
            if seq == self.last_snapshotted_seq:
                return
            self.snapshot_manager.on_outer_iteration_boundary(
                year=year,
                iteration=iteration,
            )
            self.last_snapshotted_seq = seq
            return
        self.snapshot_manager.on_outer_iteration_boundary(
            year=year,
            iteration=iteration,
        )


def _sync_coupler(
    scenario: Any,
    sim_state: SimState,
    last_synced: dict[str, ArtifactRef] | None = None,
) -> None:
    if isinstance(scenario, _HasCoupler) and scenario.coupler is not None:
        role_map = sim_state.to_role_map()
        if last_synced is not None:
            changed = {
                key: value
                for key, value in role_map.items()
                if last_synced.get(key) is not value
            }
            if changed:
                scenario.coupler.update(changed)
                last_synced.update(changed)
        elif role_map:
            scenario.coupler.update(role_map)


def _admit_bootstrap_input(
    scenario: Any,
    *,
    key: str,
    path: str,
    admitted_cache: dict[str, ArtifactRef] | None = None,
) -> ArtifactRef:
    if admitted_cache is not None and path in admitted_cache:
        return admitted_cache[path]
    try:
        first_step_started = bool(scenario._first_step_started)
    except AttributeError:
        first_step_started = False
    if not first_step_started:
        try:
            try:
                artifact = scenario.add_input(path, key=key, reuse_if_unchanged=True)
            except TypeError:
                artifact = scenario.add_input(path, key=key)
            if isinstance(artifact, Artifact):
                if admitted_cache is not None:
                    admitted_cache[path] = artifact
                return artifact
        except AttributeError:
            pass
    ref = Path(path)
    if admitted_cache is not None:
        admitted_cache[path] = ref
    return ref


def bootstrap_initial_state(
    *,
    scenario: Any,
    settings: PilatesConfig,
    state: Any,
    workspace: Workspace,
    last_synced: dict[str, ArtifactRef] | None = None,
) -> SimState:
    """Resolve initial exogenous artifacts into a typed ``SimState`` and sync coupler."""
    land_use = LandUseState()
    demand = DemandState()
    network = NetworkState()
    extras: dict[str, ArtifactRef] = {}
    admitted_cache: dict[str, ArtifactRef] = {}
    workspace_root = (
        Path(str(workspace.full_path))
        if isinstance(workspace, _WorkspaceRootLike) and workspace.full_path is not None
        else None
    )

    # 1. UrbanSim initial datastore
    requires_usim = (
        get_land_use_model(settings) == "urbansim"
        or get_activity_demand_model(settings) == "activitysim"
        or get_vehicle_ownership_model(settings) == "atlas"
    )
    if (
        uses_input_datastore(state)
        and requires_usim
        and get_setting(settings, "urbansim") is not None
    ):
        static_source = resolve_urbansim_initial_datastore_source_path(settings)
        if static_source is not None and static_source.exists():
            h5_path = str(static_source)
            base_ref = _admit_bootstrap_input(
                scenario,
                key=USIM_DATASTORE_BASE_H5,
                path=h5_path,
                admitted_cache=admitted_cache,
            )
            curr_ref = _admit_bootstrap_input(
                scenario,
                key=USIM_DATASTORE_CURRENT_H5,
                path=h5_path,
                admitted_cache=admitted_cache,
            )
            pop_ref = (
                _admit_bootstrap_input(
                    scenario,
                    key=USIM_POPULATION_SOURCE_H5,
                    path=h5_path,
                    admitted_cache=admitted_cache,
                )
                if get_land_use_model(settings) != "urbansim"
                else None
            )
            land_use = replace(
                land_use,
                prior_h5=base_ref,
                current_h5=curr_ref,
                population_source=pop_ref,
            )

    # 2. Initial skims
    has_skim_consumer = (
        get_activity_demand_model(settings) == "activitysim"
        or get_land_use_model(settings) == "urbansim"
        or get_vehicle_ownership_model(settings) == "atlas"
        or get_traffic_assignment_model(settings) == "beam"
    )
    if has_skim_consumer and not requires_prior_beam_skim_handoff(
        settings=settings, state=state
    ):
        from pilates.activitysim.admission import (
            resolve_bootstrap_activitysim_skims_path,
        )

        if workspace_root is not None:
            try:
                candidate = resolve_bootstrap_activitysim_skims_path(
                    settings=settings,
                    workspace_path=workspace_root,
                )
            except Exception:
                candidate = None
            if candidate is not None and candidate.is_file():
                omx_ref = _admit_bootstrap_input(
                    scenario,
                    key=ASIM_OMX_SKIMS,
                    path=str(candidate),
                    admitted_cache=admitted_cache,
                )
                demand = replace(demand, skims_omx_in=omx_ref)
                if get_traffic_assignment_model(settings) is None:
                    final_ref = _admit_bootstrap_input(
                        scenario,
                        key=FINAL_SKIMS_OMX,
                        path=str(candidate),
                        admitted_cache=admitted_cache,
                    )
                    network = replace(network, skims_omx=final_ref)

    # 3. BEAM exchange inputs (when ActivitySim is disabled)
    if (
        get_traffic_assignment_model(settings) == "beam"
        and get_activity_demand_model(settings) is None
    ):
        try:
            record_store = beam_exchange.register_existing_beam_exchange_inputs(
                settings=settings,
                state=state,
                workspace=workspace,
            )
        except FileNotFoundError as exc:
            logger.warning(
                "Bootstrap could not seed default BEAM inputs: %s",
                exc,
            )
            record_store = None
        allowed_keys = {BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN}
        base_path = str(workspace_root) if workspace_root is not None else None
        if record_store is not None:
            for record in record_store.all_records():
                key = record.short_name
                if key not in allowed_keys:
                    continue
                path = record.get_absolute_path(base_path=base_path)
                if not path or not os.path.exists(path):
                    continue
                ref = _admit_bootstrap_input(
                    scenario,
                    key=key,
                    path=path,
                    admitted_cache=admitted_cache,
                )
                extras[key] = ref

    # 4. BEAM warmstart linkstats
    if get_traffic_assignment_model(
        settings
    ) == "beam" and not requires_prior_beam_linkstats_handoff(
        settings=settings, state=state
    ):
        warmstart_path = resolve_initial_linkstats_path(settings, workspace)
        if warmstart_path and os.path.exists(warmstart_path):
            ws_ref = _admit_bootstrap_input(
                scenario,
                key=LINKSTATS_WARMSTART,
                path=warmstart_path,
                admitted_cache=admitted_cache,
            )
            network = replace(network, linkstats_warmstart=ws_ref)

    sim_state = SimState(
        land_use=land_use,
        demand=demand,
        network=network,
        extras=extras,
    )
    _sync_coupler(scenario, sim_state, last_synced=last_synced)
    return sim_state


def run_land_use_stage(
    *,
    scenario: ScenarioWithCoupler,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState,
    year: int,
    last_synced: dict[str, ArtifactRef] | None = None,
) -> SimState:
    """Execute UrbanSim run + postprocess and return updated ``SimState``."""
    formatted_print(f"LAND USE MODEL FOR YEAR {year}")
    logger.info("[land_use] year=%s run_id=%s", year, cr.current_run_id())

    prior_h5 = sim_state.land_use.current_h5
    run_result, run_outputs = execute_step(
        scenario=scenario,
        definition=urbansim_run,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="land_use",
        year=year,
        phase="run",
        sim_state=sim_state,
    )
    forecast_h5: ArtifactRef = run_result.outputs.get(
        USIM_DATASTORE_H5, run_outputs.usim_datastore_h5
    )
    sim_state = replace(
        sim_state,
        land_use=replace(
            sim_state.land_use,
            current_h5=forecast_h5,
            forecast_output=forecast_h5,
            prior_h5=prior_h5,
        ),
    )

    post_result, post_outputs = execute_step(
        scenario=scenario,
        definition=urbansim_postprocess,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="land_use",
        year=year,
        phase="postprocess",
        sim_state=sim_state,
    )
    archive_key = f"{USIM_INPUT_ARCHIVE_PREFIX}{resolve_forecast_year(state) or year}"
    archived_base = post_result.outputs.get(archive_key, sim_state.land_use.prior_h5)
    population_source = post_result.outputs.get(
        USIM_POPULATION_SOURCE_H5,
        post_outputs.processed_outputs.get(USIM_POPULATION_SOURCE_H5),
    )
    if population_source is None:
        raise RuntimeError(
            "UrbanSim postprocess did not produce the population-source snapshot"
        )
    curr_ref: ArtifactRef = post_result.outputs.get(
        USIM_DATASTORE_H5, post_outputs.usim_datastore_h5
    )
    sim_state = replace(
        sim_state,
        land_use=replace(
            sim_state.land_use,
            current_h5=curr_ref,
            merged_h5=curr_ref,
            prior_h5=archived_base,
            population_source=population_source,
        ),
    )
    _sync_coupler(scenario, sim_state, last_synced=last_synced)
    return sim_state


def _atlas_sub_years(state: WorkflowState) -> list[int]:
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


def _validate_population_h5_for_activitysim_year(
    *,
    path: str | os.PathLike[str],
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


def run_vehicle_ownership_stage(
    *,
    scenario: ScenarioWithCoupler,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState,
    year: int,
    last_synced: dict[str, ArtifactRef] | None = None,
) -> SimState:
    """Execute ATLAS vehicle ownership sub-years and return updated ``SimState``."""
    logger.info(
        "[vehicle_ownership] year=%s forecast_year=%s run_id=%s",
        year,
        state.forecast_year,
        cr.current_run_id(),
    )
    forecast_year = state.forecast_year
    if forecast_year is None:
        raise RuntimeError(
            "WorkflowState.forecast_year must be set before running vehicle ownership."
        )
    if settings.urbansim is None:
        raise RuntimeError(
            "UrbanSim config is required for the vehicle ownership stage."
        )

    forecast_datastore_ref = (
        sim_state.land_use.merged_h5
        or sim_state.land_use.current_h5
        or sim_state.land_use.population_source
        or sim_state.land_use.forecast_output
    )

    for atlas_year in _atlas_sub_years(state):
        atlas_state = AtlasSubState(state, atlas_year)
        try:
            prep_result, _ = execute_step(
                scenario=scenario,
                definition=atlas_preprocess,
                settings=settings,
                state=atlas_state,
                workspace=workspace,
                stage="vehicle_ownership",
                year=atlas_year,
                iteration=iteration_index(state),
                phase="preprocess",
                sim_state=sim_state,
            )

            run_result, _ = execute_step(
                scenario=scenario,
                definition=atlas_run,
                settings=settings,
                state=atlas_state,
                workspace=workspace,
                stage="vehicle_ownership",
                year=atlas_year,
                iteration=iteration_index(state),
                phase="run",
                sim_state=sim_state,
                prior_outputs=prep_result.outputs,
            )
            sim_state = replace(
                sim_state,
                atlas=replace(
                    sim_state.atlas,
                    vehicles=run_result.outputs.get(f"vehicles_{atlas_year}"),
                    households_with_cars=run_result.outputs.get(
                        f"householdv_{atlas_year}"
                    ),
                    rdata_bundle=run_result.outputs.get(
                        f"atlas_continuation_{atlas_year}"
                    ),
                    last_simulated_year=atlas_year,
                ),
            )

            post_result, post_outputs = execute_step(
                scenario=scenario,
                definition=atlas_postprocess,
                settings=settings,
                state=atlas_state,
                workspace=workspace,
                stage="vehicle_ownership",
                year=atlas_year,
                iteration=iteration_index(state),
                phase="postprocess",
                sim_state=sim_state,
            )
            hh_cars_ref = post_result.outputs.get(ATLAS_VEHICLES2_OUTPUT)
            pop_ref: ArtifactRef | None = post_result.outputs.get(
                USIM_POPULATION_SOURCE_H5,
                post_outputs.usim_datastore_h5,
            )
            updated_land_use = sim_state.land_use
            if pop_ref is not None:
                if atlas_year != forecast_year:
                    updated_land_use = replace(
                        updated_land_use,
                        population_source=pop_ref,
                        prior_h5=pop_ref,
                        current_h5=(
                            forecast_datastore_ref
                            if forecast_datastore_ref is not None
                            else updated_land_use.current_h5
                        ),
                    )
                else:
                    updated_land_use = replace(
                        updated_land_use,
                        population_source=pop_ref,
                        current_h5=pop_ref,
                    )
            sim_state = replace(
                sim_state,
                land_use=updated_land_use,
                atlas=replace(sim_state.atlas, hh_cars=hh_cars_ref),
            )
        except Exception:
            from pilates.utils.failure_handling import persist_state_on_error

            persist_state_on_error(state, f"ATLAS year {atlas_year}")
            sys.exit(1)

    _sync_coupler(scenario, sim_state, last_synced=last_synced)
    return sim_state


def _activitysim_population_year(state: WorkflowState) -> int:
    forecast_year = resolve_forecast_year(state)
    if forecast_year is None:
        forecast_year = current_year(state)
    if forecast_year is None:
        raise RuntimeError(
            "WorkflowState.forecast_year or WorkflowState.year must be set before "
            "ActivitySim execution."
        )
    return int(forecast_year)


def run_activity_demand_phase(
    *,
    scenario: ScenarioWithCoupler,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState,
    year: int,
    iteration: int,
    last_synced: dict[str, ArtifactRef] | None = None,
) -> SimState:
    """Execute ActivitySim preprocess, run, and postprocess for one iteration."""
    population_year = _activitysim_population_year(state)
    formatted_print("ACTIVITY DEMAND MODEL")
    logger.info("[activity_demand] year=%s iteration=%s", year, iteration)

    prep_result, _ = execute_step(
        scenario=scenario,
        definition=activitysim_preprocess,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=population_year,
        iteration=iteration,
        phase="preprocess",
        sim_state=sim_state,
    )
    sim_state = replace(
        sim_state,
        demand=replace(
            sim_state.demand,
            land_use_csv=prep_result.outputs.get(ASIM_LAND_USE_IN),
            households_in=prep_result.outputs.get(ASIM_HOUSEHOLDS_IN),
            persons_in=prep_result.outputs.get(ASIM_PERSONS_IN),
            skims_omx_in=prep_result.outputs.get(
                ASIM_OMX_SKIMS, sim_state.demand.skims_omx_in
            ),
        ),
    )

    run_result, run_outputs = execute_step(
        scenario=scenario,
        definition=activitysim_run,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=population_year,
        iteration=iteration,
        phase="run",
        sim_state=sim_state,
    )
    zarr_ref = run_result.outputs.get(ZARR_SKIMS, run_outputs.zarr_skims)
    sim_state = replace(
        sim_state,
        demand=replace(
            sim_state.demand,
            households=run_result.outputs.get("households_asim_out"),
            persons=run_result.outputs.get("persons_asim_out"),
            tours=run_result.outputs.get("tours_asim_out"),
            trips=run_result.outputs.get("trips_asim_out"),
            plans=run_result.outputs.get("beam_plans_asim_out"),
            skims_zarr=zarr_ref
            if zarr_ref is not None
            else sim_state.demand.skims_zarr,
        ),
    )

    post_result, _ = execute_step(
        scenario=scenario,
        definition=activitysim_postprocess,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=population_year,
        iteration=iteration,
        phase="postprocess",
        sim_state=sim_state,
        prior_outputs=run_result.outputs,
    )
    next_h5 = post_result.outputs.get(USIM_DATASTORE_H5)
    updated_land_use = (
        replace(sim_state.land_use, current_h5=next_h5)
        if next_h5 is not None
        else sim_state.land_use
    )
    sim_state = replace(
        sim_state,
        land_use=updated_land_use,
        demand=replace(
            sim_state.demand,
            households=post_result.outputs.get(
                "households_asim_out", sim_state.demand.households
            ),
            persons=post_result.outputs.get(
                "persons_asim_out", sim_state.demand.persons
            ),
            tours=post_result.outputs.get("tours_asim_out", sim_state.demand.tours),
            trips=post_result.outputs.get("trips_asim_out", sim_state.demand.trips),
            plans=post_result.outputs.get(
                "beam_plans_asim_out", sim_state.demand.plans
            ),
        ),
    )
    _sync_coupler(scenario, sim_state, last_synced=last_synced)
    return sim_state


def _full_skim_run_schedule(settings: PilatesConfig) -> str:
    beam_cfg = settings.beam
    skim_cfg = beam_cfg.full_skim if beam_cfg is not None else None
    if skim_cfg is None:
        return "disabled"
    return skim_cfg.run_schedule


def _effective_supply_demand_iterations(
    settings: PilatesConfig,
    state: Optional[WorkflowState],
    iteration: int,
) -> int:
    if isinstance(state, WorkflowState):
        return state.effective_supply_demand_iterations(resumed_iteration=iteration)
    configured_iters = int(settings.run.supply_demand_iters)
    if (
        isinstance(settings.run, _HasRunModels)
        and settings.run.models is not None
        and settings.run.models.activity_demand is None
        and configured_iters > 1
    ):
        return max(1, iteration + 1)
    return configured_iters


def _should_run_full_skim(
    settings: PilatesConfig,
    iteration: int,
    *,
    state: Optional[WorkflowState] = None,
) -> bool:
    schedule = _full_skim_run_schedule(settings)
    if schedule in ("standalone", "after_each_iteration"):
        return True
    if schedule == "after_final_iteration":
        total_iters = _effective_supply_demand_iterations(settings, state, iteration)
        return iteration == total_iters - 1
    return False


def _run_beam_full_skim(
    *,
    scenario: ScenarioWithCoupler,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState,
    beam_year: int,
    iteration: int,
    prior_outputs: Mapping[str, ArtifactRef] | None = None,
    clear_zarr_on_omx: bool = False,
) -> SimState:
    skim_result, skim_outputs = execute_step(
        scenario=scenario,
        definition=beam_full_skim,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=beam_year,
        iteration=iteration,
        phase="full_skim",
        sim_state=sim_state,
        prior_outputs=prior_outputs,
    )
    new_skims_omx = skim_result.outputs.get(
        FINAL_SKIMS_OMX, skim_outputs.final_skims_omx
    )
    next_demand = (
        replace(sim_state.demand, skims_zarr=None)
        if clear_zarr_on_omx and new_skims_omx is not None
        else sim_state.demand
    )
    next_network = replace(
        sim_state.network,
        full_skims=skim_result.outputs.get(BEAM_FULL_SKIMS, skim_outputs.full_skims),
        skims_omx=(
            new_skims_omx if new_skims_omx is not None else sim_state.network.skims_omx
        ),
        skims_zarr=(
            None
            if clear_zarr_on_omx and new_skims_omx is not None
            else sim_state.network.skims_zarr
        ),
    )
    return replace(
        sim_state,
        demand=next_demand,
        network=next_network,
    )


def _maybe_fail_after_beam_run_for_canary(*, year: int, iteration: int) -> None:
    if os.environ.get(_FAIL_AFTER_BEAM_RUN_ENV) != "1":
        return
    message = (
        "Injected failure after completed beam_run for restart canary "
        f"({_FAIL_AFTER_BEAM_RUN_ENV}=1, year={year}, iteration={iteration}). "
        "Unset the environment variable before restarting from run_state.yaml."
    )
    logger.error(message)
    raise RuntimeError(message)


def run_traffic_assignment_phase(
    *,
    scenario: ScenarioWithCoupler,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState,
    year: int,
    iteration: int,
    last_synced: dict[str, ArtifactRef] | None = None,
) -> SimState:
    """Execute BEAM (or standalone full skim) for one iteration."""
    traffic_model = get_traffic_assignment_model(settings)
    if traffic_model is not None and traffic_model != "beam":
        raise NotImplementedError(
            f"Runtime traffic_assignment execution for model {traffic_model!r} "
            "is not yet wired in orchestrator."
        )

    formatted_print("TRAFFIC ASSIGNMENT MODEL")
    forecast_year = resolve_forecast_year(state)
    beam_year = int(forecast_year) if forecast_year is not None else year

    schedule = _full_skim_run_schedule(settings)
    if schedule == "standalone":
        if (
            sim_state.network.linkstats_warmstart is None
            and sim_state.network.linkstats is None
        ):
            resolved_ws = resolve_initial_linkstats_path(settings, workspace)
            if resolved_ws is not None:
                sim_state = replace(
                    sim_state,
                    network=replace(
                        sim_state.network, linkstats_warmstart=Path(resolved_ws)
                    ),
                )
        sim_state = _run_beam_full_skim(
            scenario=scenario,
            settings=settings,
            state=state,
            workspace=workspace,
            sim_state=sim_state,
            beam_year=beam_year,
            iteration=iteration,
            clear_zarr_on_omx=True,
        )
        _sync_coupler(scenario, sim_state, last_synced=last_synced)
        return sim_state

    prep_result, _ = execute_step(
        scenario=scenario,
        definition=beam_preprocess,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=beam_year,
        iteration=iteration,
        phase="preprocess",
        sim_state=sim_state,
    )
    sim_for_beam = replace(
        sim_state,
        network=replace(
            sim_state.network,
            linkstats_warmstart=prep_result.outputs.get(
                LINKSTATS_WARMSTART, sim_state.network.linkstats_warmstart
            ),
        ),
    )

    run_result, _ = execute_step(
        scenario=scenario,
        definition=beam_run,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=beam_year,
        iteration=iteration,
        phase="run",
        sim_state=sim_for_beam,
        prior_outputs=prep_result.outputs,
    )
    logger.info("[beam] completed native beam_run run_id=%s", run_result.run.id)
    _maybe_fail_after_beam_run_for_canary(year=beam_year, iteration=iteration)

    post_result, post_outputs = execute_step(
        scenario=scenario,
        definition=beam_postprocess,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="supply_demand",
        year=beam_year,
        iteration=iteration,
        phase="postprocess",
        sim_state=sim_for_beam,
        prior_outputs=run_result.outputs,
    )

    events_key = f"events_parquet_{beam_year}_{iteration}"
    events_pq = run_result.outputs.get(events_key)
    sim_state = replace(
        sim_state,
        network=replace(
            sim_state.network,
            plans=run_result.outputs.get(BEAM_PLANS_OUT),
            experienced_plans=run_result.outputs.get(BEAM_EXPERIENCED_PLANS_XML),
            network_csv=post_result.outputs.get(
                BEAM_NETWORK_FINAL,
                run_result.outputs.get(
                    BEAM_NETWORK_FINAL, sim_state.network.network_csv
                ),
            ),
            linkstats=post_result.outputs.get(
                LINKSTATS,
                run_result.outputs.get(LINKSTATS, sim_state.network.linkstats),
            ),
            skims_omx=post_result.outputs.get(
                FINAL_SKIMS_OMX,
                post_outputs.final_skims_omx or sim_state.network.skims_omx,
            ),
            skims_zarr=post_result.outputs.get(
                ZARR_SKIMS,
                post_outputs.zarr_skims or sim_state.network.skims_zarr,
            ),
            events_parquet=events_pq,
        ),
    )

    if _should_run_full_skim(settings, iteration, state=state):
        sim_state = _run_beam_full_skim(
            scenario=scenario,
            settings=settings,
            state=state,
            workspace=workspace,
            sim_state=sim_state,
            beam_year=beam_year,
            iteration=iteration,
            prior_outputs=prep_result.outputs,
        )

    _sync_coupler(scenario, sim_state, last_synced=last_synced)
    return sim_state


def _run_supply_demand_loop(
    *,
    scenario: ScenarioWithCoupler,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState,
    year: int,
    on_activity_demand_boundary: Optional[Callable[[int], None]] = None,
    on_iteration_boundary: Optional[Callable[[int], None]] = None,
    env: _OrchestratorEnv | None = None,
    last_synced: dict[str, ArtifactRef] | None = None,
) -> SimState:
    """Run the supply-demand loop (ActivitySim + optional pre/post-BEAM + BEAM)."""
    logger.info("[supply_demand] year=%s run_id=%s", year, cr.current_run_id())

    active_last_synced = env.last_synced if env is not None else last_synced
    configured_iters = settings.run.supply_demand_iters
    resumed_iteration = int(state.iteration or 0)
    total_iters = _effective_supply_demand_iterations(
        settings, state, resumed_iteration
    )

    if total_iters != configured_iters:
        logger.warning(
            "BEAM-only supply_demand_iters=%d. Clamping outer supply-demand "
            "iterations to %d because BEAM already manages its own internal "
            "iterations.",
            configured_iters,
            total_iters,
        )

    for it in range(state.iteration, total_iters):
        state.iteration = it
        formatted_print(f"SUPPLY/DEMAND ITERATION {it + 1}/{total_iters}")

        if get_activity_demand_model(settings) is not None:
            sim_state = run_activity_demand_phase(
                scenario=scenario,
                settings=settings,
                state=state,
                workspace=workspace,
                sim_state=sim_state,
                year=year,
                iteration=it,
                last_synced=active_last_synced,
            )
            if env is not None:
                env.maybe_on_activity_demand_boundary(year=year, iteration=it)
            elif on_activity_demand_boundary is not None:
                on_activity_demand_boundary(it)

        # Optional pre-BEAM extension point:
        # if settings.my_model is not None and settings.my_model.enabled:
        #     sim_state = run_my_model_pre_beam(...)

        if get_traffic_assignment_model(settings) is not None:
            sim_state = run_traffic_assignment_phase(
                scenario=scenario,
                settings=settings,
                state=state,
                workspace=workspace,
                sim_state=sim_state,
                year=year,
                iteration=it,
                last_synced=active_last_synced,
            )

        # Optional post-BEAM extension point:
        # if settings.my_model is not None and settings.my_model.enabled:
        #     sim_state = run_my_model_post_beam(...)

        if env is not None:
            env.maybe_on_outer_iteration_boundary(year=year, iteration=it)
        elif on_iteration_boundary is not None:
            on_iteration_boundary(it)

    return sim_state


def run_postprocessing_stage(
    *,
    scenario: ScenarioWithCoupler,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    sim_state: SimState,
    year: int,
    last_synced: dict[str, ArtifactRef] | None = None,
) -> SimState:
    """Execute global postprocessing and return updated ``SimState``."""
    post_year = resolve_forecast_year(state) or year
    logger.info("[postprocessing] year=%s run_id=%s", post_year, cr.current_run_id())
    post_result, post_outputs = execute_step(
        scenario=scenario,
        definition=postprocessing_step,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="postprocessing",
        year=post_year,
        phase="postprocess",
        sim_state=sim_state,
    )
    inexus_ref = post_result.outputs.get(
        "inexus_output", post_outputs.get("inexus_output")
    )
    if inexus_ref is not None:
        sim_state = replace(sim_state, inexus_outputs=inexus_ref)
    _sync_coupler(scenario, sim_state, last_synced=last_synced)
    return sim_state


def run_pilates(
    *,
    scenario: ScenarioWithCoupler,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    snapshot_manager: _SnapshotManagerLike | None = None,
    on_activity_demand_boundary: Optional[Callable[[int, int], None]] = None,
) -> SimState:
    """Run the full PILATES forecast-year loop over typed ``SimState``."""
    env = _OrchestratorEnv(
        scenario=scenario,
        settings=settings,
        state=state,
        workspace=workspace,
        snapshot_manager=snapshot_manager,
        on_activity_demand_boundary=on_activity_demand_boundary,
    )
    sim_state = bootstrap_initial_state(
        scenario=env.scenario,
        settings=env.settings,
        state=env.state,
        workspace=env.workspace,
        last_synced=env.last_synced,
    )

    for year in env.state:
        formatted_print(f"STARTING YEAR {year}")
        logger.info(
            "[launcher] starting year=%s run_id=%s",
            year,
            cr.current_run_id(),
        )
        if get_land_use_model(env.settings) is not None:
            sim_state = run_land_use_stage(
                scenario=env.scenario,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                sim_state=sim_state,
                year=year,
                last_synced=env.last_synced,
            )
            if env.snapshot_manager is not None:
                env.snapshot_manager.maybe_snapshot_interval(
                    reason=f"after_land_use_y{year}"
                )

        if get_vehicle_ownership_model(env.settings) is not None:
            formatted_print(
                f"VEHICLE OWNERSHIP MODEL (ATLAS) FOR YEAR {env.state.forecast_year}"
            )
            sim_state = run_vehicle_ownership_stage(
                scenario=env.scenario,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                sim_state=sim_state,
                year=year,
                last_synced=env.last_synced,
            )
            if env.snapshot_manager is not None:
                env.snapshot_manager.maybe_snapshot_interval(
                    reason=f"after_vehicle_ownership_y{year}"
                )

        if (
            get_activity_demand_model(env.settings) is not None
            or get_traffic_assignment_model(env.settings) is not None
        ):
            sim_state = _run_supply_demand_loop(
                scenario=env.scenario,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                sim_state=sim_state,
                year=year,
                env=env,
            )
            if env.snapshot_manager is not None:
                env.snapshot_manager.maybe_snapshot_interval(
                    reason=f"after_supply_demand_y{year}"
                )

        if env.settings.postprocessing is not None:
            formatted_print("POST-PROCESSING")
            sim_state = run_postprocessing_stage(
                scenario=env.scenario,
                settings=env.settings,
                state=env.state,
                workspace=env.workspace,
                sim_state=sim_state,
                year=year,
                last_synced=env.last_synced,
            )
            if env.snapshot_manager is not None:
                env.snapshot_manager.maybe_snapshot_interval(
                    reason=f"after_postprocessing_y{year}"
                )

        if env.snapshot_manager is not None:
            env.snapshot_manager.maybe_snapshot_interval(
                reason=f"year_boundary_y{year}"
            )

    return sim_state
