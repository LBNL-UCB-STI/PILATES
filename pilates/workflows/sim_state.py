"""Typed simulation state threaded through the imperative PILATES orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from consist import Artifact

from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_FULL_SKIMS,
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

ArtifactRef = Artifact | Path


@dataclass(frozen=True, slots=True)
class LandUseState:
    """UrbanSim / base land-use H5 state (pre-Phase 3 HDF5 boundary)."""

    current_h5: ArtifactRef | None = None
    prior_h5: ArtifactRef | None = None
    merged_h5: ArtifactRef | None = None
    forecast_output: ArtifactRef | None = None
    population_source: ArtifactRef | None = None


@dataclass(frozen=True, slots=True)
class AtlasState:
    """ATLAS vehicle-ownership outputs and continuation artifacts."""

    vehicles: ArtifactRef | None = None
    households_with_cars: ArtifactRef | None = None
    hh_cars: ArtifactRef | None = None
    rdata_bundle: ArtifactRef | None = None
    last_simulated_year: int | None = None


@dataclass(frozen=True, slots=True)
class DemandState:
    """ActivitySim (or equivalent demand model) iteration inputs and outputs."""

    land_use_csv: ArtifactRef | None = None
    households_in: ArtifactRef | None = None
    persons_in: ArtifactRef | None = None
    skims_omx_in: ArtifactRef | None = None
    households: ArtifactRef | None = None
    persons: ArtifactRef | None = None
    tours: ArtifactRef | None = None
    trips: ArtifactRef | None = None
    plans: ArtifactRef | None = None
    skims_zarr: ArtifactRef | None = None


@dataclass(frozen=True, slots=True)
class NetworkState:
    """BEAM (or equivalent supply model) network & skim outputs."""

    skims_omx: ArtifactRef | None = None
    skims_zarr: ArtifactRef | None = None
    linkstats: ArtifactRef | None = None
    linkstats_warmstart: ArtifactRef | None = None
    plans: ArtifactRef | None = None
    experienced_plans: ArtifactRef | None = None
    events_parquet: ArtifactRef | None = None
    network_csv: ArtifactRef | None = None
    full_skims: ArtifactRef | None = None


@dataclass(frozen=True, slots=True)
class SimState:
    """Single source of truth threaded through ``run_pilates()``."""

    land_use: LandUseState = field(default_factory=LandUseState)
    atlas: AtlasState = field(default_factory=AtlasState)
    demand: DemandState = field(default_factory=DemandState)
    network: NetworkState = field(default_factory=NetworkState)
    inexus_outputs: ArtifactRef | None = None
    extras: Mapping[str, ArtifactRef] = field(default_factory=dict)

    def to_role_map(self) -> dict[str, ArtifactRef]:
        """Project non-None fields to canonical coupler keys for Consist lineage."""
        roles: dict[str, ArtifactRef] = {
            key: value for key, value in self.extras.items() if value is not None
        }

        if self.land_use.prior_h5 is not None:
            roles[USIM_DATASTORE_BASE_H5] = self.land_use.prior_h5
        if self.land_use.current_h5 is not None:
            roles[USIM_DATASTORE_CURRENT_H5] = self.land_use.current_h5
        if self.land_use.forecast_output is not None:
            roles[USIM_FORECAST_OUTPUT] = self.land_use.forecast_output
        if self.land_use.population_source is not None:
            roles[USIM_POPULATION_SOURCE_H5] = self.land_use.population_source

        if self.atlas.hh_cars is not None:
            roles[ATLAS_VEHICLES2_OUTPUT] = self.atlas.hh_cars
        if self.atlas.last_simulated_year is not None:
            yr = self.atlas.last_simulated_year
            if self.atlas.vehicles is not None:
                roles[f"vehicles_{yr}"] = self.atlas.vehicles
            if self.atlas.households_with_cars is not None:
                roles[f"householdv_{yr}"] = self.atlas.households_with_cars
            if self.atlas.rdata_bundle is not None:
                roles[f"atlas_continuation_{yr}"] = self.atlas.rdata_bundle

        if self.demand.land_use_csv is not None:
            roles[ASIM_LAND_USE_IN] = self.demand.land_use_csv
        if self.demand.households_in is not None:
            roles[ASIM_HOUSEHOLDS_IN] = self.demand.households_in
        if self.demand.persons_in is not None:
            roles[ASIM_PERSONS_IN] = self.demand.persons_in
        if self.demand.skims_omx_in is not None:
            roles[ASIM_OMX_SKIMS] = self.demand.skims_omx_in
        if self.demand.households is not None:
            roles["households_asim_out"] = self.demand.households
        if self.demand.persons is not None:
            roles["persons_asim_out"] = self.demand.persons
        if self.demand.tours is not None:
            roles["tours_asim_out"] = self.demand.tours
        if self.demand.trips is not None:
            roles["trips_asim_out"] = self.demand.trips
        if self.demand.plans is not None:
            roles["beam_plans_asim_out"] = self.demand.plans
        if self.demand.skims_zarr is not None:
            roles[ZARR_SKIMS] = self.demand.skims_zarr

        if self.network.skims_omx is not None:
            roles[FINAL_SKIMS_OMX] = self.network.skims_omx
        if self.network.skims_zarr is not None:
            roles[ZARR_SKIMS] = self.network.skims_zarr
        if self.network.linkstats is not None:
            roles[LINKSTATS] = self.network.linkstats
        if self.network.linkstats_warmstart is not None:
            roles[LINKSTATS_WARMSTART] = self.network.linkstats_warmstart
        if self.network.plans is not None:
            roles[BEAM_PLANS_OUT] = self.network.plans
        if self.network.experienced_plans is not None:
            roles[BEAM_EXPERIENCED_PLANS_XML] = self.network.experienced_plans
        if self.network.full_skims is not None:
            roles[BEAM_FULL_SKIMS] = self.network.full_skims

        if self.inexus_outputs is not None:
            roles["inexus_output"] = self.inexus_outputs

        return roles
