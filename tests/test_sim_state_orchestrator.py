"""Unit tests for typed ``SimState`` and imperative ``orchestrator`` behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from consist import Artifact

from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_FULL_SKIMS,
    BEAM_NETWORK_FINAL,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    FINAL_SKIMS_OMX,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_FORECAST_OUTPUT,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.orchestrator import (
    bootstrap_initial_state,
    run_traffic_assignment_phase,
    run_vehicle_ownership_stage,
)
from pilates.workflows.sim_state import (
    AtlasState,
    DemandState,
    LandUseState,
    NetworkState,
    SimState,
)


def _artifact(key: str, tmp_path: Path) -> Artifact:
    p = tmp_path / f"{key}.dat"
    p.write_text("x")
    return Artifact(key=key, uri=f"file://{p}", driver="other")


def test_sim_state_to_role_map_projects_canonical_coupler_keys(tmp_path: Path) -> None:
    state = SimState(
        land_use=LandUseState(
            current_h5=_artifact(USIM_DATASTORE_CURRENT_H5, tmp_path),
            prior_h5=_artifact(USIM_DATASTORE_BASE_H5, tmp_path),
            forecast_output=_artifact(USIM_FORECAST_OUTPUT, tmp_path),
            population_source=_artifact(USIM_POPULATION_SOURCE_H5, tmp_path),
        ),
        atlas=AtlasState(
            vehicles=_artifact("vehicles_2017", tmp_path),
            households_with_cars=_artifact("householdv_2017", tmp_path),
            hh_cars=_artifact(ATLAS_VEHICLES2_OUTPUT, tmp_path),
            rdata_bundle=_artifact("atlas_continuation_2017", tmp_path),
            last_simulated_year=2017,
        ),
        demand=DemandState(
            land_use_csv=_artifact(ASIM_LAND_USE_IN, tmp_path),
            households_in=_artifact(ASIM_HOUSEHOLDS_IN, tmp_path),
            persons_in=_artifact(ASIM_PERSONS_IN, tmp_path),
            skims_omx_in=_artifact(ASIM_OMX_SKIMS, tmp_path),
            households=_artifact("households_asim_out", tmp_path),
            persons=_artifact("persons_asim_out", tmp_path),
            tours=_artifact("tours_asim_out", tmp_path),
            trips=_artifact("trips_asim_out", tmp_path),
            plans=_artifact("beam_plans_asim_out", tmp_path),
        ),
        network=NetworkState(
            skims_omx=_artifact(FINAL_SKIMS_OMX, tmp_path),
            skims_zarr=_artifact(ZARR_SKIMS, tmp_path),
            linkstats=_artifact(LINKSTATS, tmp_path),
            linkstats_warmstart=_artifact(LINKSTATS_WARMSTART, tmp_path),
            plans=_artifact(BEAM_PLANS_OUT, tmp_path),
            experienced_plans=_artifact(BEAM_EXPERIENCED_PLANS_XML, tmp_path),
            events_parquet=_artifact("events_parquet", tmp_path),
            network_csv=_artifact(BEAM_NETWORK_FINAL, tmp_path),
            full_skims=_artifact(BEAM_FULL_SKIMS, tmp_path),
        ),
        inexus_outputs=_artifact("inexus_output", tmp_path),
        extras={"custom_pre_beam_out": tmp_path / "custom.parquet"},
    )

    roles = state.to_role_map()
    expected_keys = {
        USIM_DATASTORE_CURRENT_H5,
        USIM_DATASTORE_BASE_H5,
        USIM_FORECAST_OUTPUT,
        USIM_POPULATION_SOURCE_H5,
        ATLAS_VEHICLES2_OUTPUT,
        "vehicles_2017",
        "householdv_2017",
        "atlas_continuation_2017",
        ASIM_LAND_USE_IN,
        ASIM_HOUSEHOLDS_IN,
        ASIM_PERSONS_IN,
        ASIM_OMX_SKIMS,
        "households_asim_out",
        "persons_asim_out",
        "tours_asim_out",
        "trips_asim_out",
        "beam_plans_asim_out",
        FINAL_SKIMS_OMX,
        ZARR_SKIMS,
        LINKSTATS,
        LINKSTATS_WARMSTART,
        BEAM_PLANS_OUT,
        BEAM_EXPERIENCED_PLANS_XML,
        BEAM_FULL_SKIMS,
        "inexus_output",
        "custom_pre_beam_out",
    }
    assert set(roles.keys()) == expected_keys


def test_bootstrap_initial_state_seeds_warmstart_when_upstream_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    base_h5 = tmp_path / "base.h5"
    base_h5.write_text("h5")
    warmstart_linkstats = tmp_path / "warmstart.csv.gz"
    warmstart_linkstats.write_text("linkstats")
    init_skims = tmp_path / "skims.omx"
    init_skims.write_text("omx")

    monkeypatch.setattr(
        "pilates.workflows.orchestrator.resolve_urbansim_initial_datastore_source_path",
        lambda _settings: base_h5,
    )
    monkeypatch.setattr(
        "pilates.workflows.orchestrator.resolve_initial_linkstats_path",
        lambda _settings, _workspace: str(warmstart_linkstats),
    )
    monkeypatch.setattr(
        "pilates.activitysim.admission.resolve_bootstrap_activitysim_skims_path",
        lambda **_kwargs: init_skims,
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            start_year=2015,
            models=SimpleNamespace(
                land_use=None,
                vehicle_ownership=None,
                activity_demand="activitysim",
                travel="beam",
            ),
        ),
        urbansim=SimpleNamespace(),
        beam=SimpleNamespace(),
    )
    state = SimpleNamespace(
        year=2015,
        current_year=2015,
        iteration=0,
        is_start_year=lambda: True,
    )
    workspace = SimpleNamespace(full_path=tmp_path)
    coupler: dict[str, object] = {}
    scenario = SimpleNamespace(
        coupler=coupler,
        _first_step_started=False,
        add_input=lambda path, key: Path(path),
    )

    sim_state = bootstrap_initial_state(
        scenario=scenario,
        settings=settings,
        state=state,
        workspace=workspace,
    )

    assert sim_state.land_use.current_h5 == base_h5
    assert sim_state.land_use.prior_h5 == base_h5
    # With land_use disabled, population_source is seeded from base H5
    assert sim_state.land_use.population_source == base_h5
    assert sim_state.demand.skims_omx_in == init_skims
    assert sim_state.network.linkstats_warmstart == warmstart_linkstats
    assert coupler[USIM_POPULATION_SOURCE_H5] == base_h5
    assert coupler[LINKSTATS_WARMSTART] == warmstart_linkstats


def test_run_traffic_assignment_phase_preserves_it1_demand_and_full_events_parquet(
    monkeypatch, tmp_path: Path
) -> None:
    from dataclasses import replace
    from pilates.workflows.orchestrator import beam_postprocess, beam_run
    from pilates.workflows.steps.beam import _sim_state_beam_candidates

    plans_it0 = _artifact("plans_it0", tmp_path)
    plans_it1 = _artifact("plans_it1", tmp_path)
    prep_plans_it0 = _artifact("prep_plans_it0", tmp_path)
    full_events = _artifact("events_parquet_2020_0", tmp_path)
    split_events = _artifact("events_parquet_2020_0_type_PathTraversal", tmp_path)

    def _fake_execute_step(*, definition, **kwargs):
        if definition.name == "beam_preprocess":
            return (
                SimpleNamespace(outputs={BEAM_PLANS_IN: prep_plans_it0}),
                SimpleNamespace(prepared_inputs={BEAM_PLANS_IN: prep_plans_it0}),
            )
        if definition.name == "beam_run":
            return (
                SimpleNamespace(
                    run=SimpleNamespace(id="beam-run-0"),
                    outputs={"events_parquet_2020_0": full_events},
                ),
                SimpleNamespace(raw_outputs={}),
            )
        if definition.name == "beam_postprocess":
            return (
                SimpleNamespace(
                    outputs={
                        "events_parquet_2020_0_type_PathTraversal": split_events,
                    }
                ),
                SimpleNamespace(final_skims_omx=None, zarr_skims=None),
            )
        raise AssertionError(f"Unexpected step {definition.name}")

    monkeypatch.setattr(
        "pilates.workflows.orchestrator.execute_step", _fake_execute_step
    )
    monkeypatch.setattr(
        "pilates.workflows.orchestrator.beam_postprocess",
        replace(beam_postprocess, resolve_inputs=lambda **_kwargs: SimpleNamespace()),
    )
    monkeypatch.setattr(
        "pilates.workflows.orchestrator.beam_run",
        replace(beam_run, resolve_inputs=lambda **_kwargs: SimpleNamespace()),
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            supply_demand_iters=2,
            models=SimpleNamespace(travel="beam", activity_demand="activitysim"),
        ),
        beam=SimpleNamespace(full_skim=SimpleNamespace(run_schedule="never")),
    )
    state = SimpleNamespace(forecast_year=2020, year=2020, iteration=0)
    scenario = SimpleNamespace(coupler={})
    workspace = SimpleNamespace(full_path=tmp_path)

    sim_state = SimState(demand=DemandState(plans=plans_it0))
    after_it0 = run_traffic_assignment_phase(
        scenario=scenario,
        settings=settings,
        state=state,
        workspace=workspace,
        sim_state=sim_state,
        year=2020,
        iteration=0,
    )

    assert after_it0.network.events_parquet == full_events
    assert BEAM_PLANS_IN not in after_it0.extras

    for_it1 = replace(after_it0, demand=replace(after_it0.demand, plans=plans_it1))
    assert _sim_state_beam_candidates(for_it1)[BEAM_PLANS_IN] == plans_it1


def test_run_vehicle_ownership_stage_advances_population_source_on_intermediate_subyears(
    monkeypatch, tmp_path: Path
) -> None:
    pop_initial = _artifact("pop_init", tmp_path)
    pop_2010 = _artifact("pop_2010", tmp_path)
    pop_2012 = _artifact("pop_2012", tmp_path)
    seen_prior_h5: list[Artifact | Path | None] = []

    def _fake_execute_step(*, definition, year, sim_state, **kwargs):
        if definition.name == "atlas_preprocess":
            seen_prior_h5.append(sim_state.land_use.prior_h5)
            return SimpleNamespace(outputs={}), SimpleNamespace(prepared_inputs={})
        if definition.name == "atlas_run":
            return (
                SimpleNamespace(outputs={}),
                SimpleNamespace(
                    vehicles=None,
                    households_with_cars=None,
                    rdata_bundle=None,
                    raw_outputs={},
                    atlas_output_dir=tmp_path / f"atlas_{year}",
                ),
            )
        if definition.name == "atlas_postprocess":
            pop_ref = pop_2010 if year == 2010 else pop_2012
            return (
                SimpleNamespace(outputs={USIM_POPULATION_SOURCE_H5: pop_ref}),
                SimpleNamespace(processed_outputs={}, usim_datastore_h5=pop_ref),
            )
        raise AssertionError(f"Unexpected step {definition.name}")

    monkeypatch.setattr(
        "pilates.workflows.orchestrator.execute_step", _fake_execute_step
    )
    monkeypatch.setattr(
        "pilates.workflows.orchestrator._atlas_sub_years",
        lambda _state: [2010, 2012],
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(vehicle_ownership="atlas", land_use="urbansim")
        ),
        urbansim=SimpleNamespace(),
    )
    state = SimpleNamespace(
        forecast_year=2013,
        year=2010,
        start_year=2010,
        iteration=0,
        full_settings=settings,
    )
    scenario = SimpleNamespace(coupler={})
    workspace = SimpleNamespace(full_path=tmp_path)

    initial = SimState(
        land_use=LandUseState(
            current_h5=pop_initial, prior_h5=pop_initial, population_source=pop_initial
        )
    )
    updated = run_vehicle_ownership_stage(
        scenario=scenario,
        settings=settings,
        state=state,
        workspace=workspace,
        sim_state=initial,
        year=2010,
    )

    assert seen_prior_h5 == [pop_initial, pop_2010]
    assert updated.land_use.prior_h5 == pop_2012
    assert updated.land_use.population_source == pop_2012
