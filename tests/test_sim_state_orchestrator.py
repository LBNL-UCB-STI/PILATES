"""Unit tests for typed ``SimState`` and imperative ``orchestrator`` behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from consist import Artifact

from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_OUTPUT_DIR,
    ASIM_PERSONS_IN,
    ATLAS_OUTPUT_DIR,
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_FULL_SKIMS,
    BEAM_NETWORK_FINAL,
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
from pilates.workflows.orchestrator import bootstrap_initial_state
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
            output_dir=tmp_path / "atlas_out",
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
            output_dir=tmp_path / "asim_out",
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
        ATLAS_OUTPUT_DIR,
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
        ASIM_OUTPUT_DIR,
        FINAL_SKIMS_OMX,
        ZARR_SKIMS,
        LINKSTATS,
        LINKSTATS_WARMSTART,
        BEAM_PLANS_OUT,
        BEAM_EXPERIENCED_PLANS_XML,
        "events_parquet",
        BEAM_NETWORK_FINAL,
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
