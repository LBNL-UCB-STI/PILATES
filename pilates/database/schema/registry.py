from __future__ import annotations

from typing import Dict, Optional, Type

from sqlmodel import SQLModel
from pilates.workflows.artifact_key_migrations import resolve_artifact_key

from pilates.database.schema.activitysim_schema import (
    AccessibilityAsimOut,
    BeamPlansAsimOut,
    HouseholdsAsimIn,
    HouseholdsAsimOut,
    LandUseAsimIn,
    LandUseAsimOut,
    JointTourParticipantsAsimOut,
    PersonsAsimIn,
    PersonsAsimOut,
    ToursAsimOut,
    TripsAsimOut,
)
from pilates.database.schema.atlas_schema import (
    AtlasBlocks,
    AtlasGrave,
    AtlasHousehold,
    HouseholdVAtlasOut,
    AtlasJobs,
    AtlasPersons,
    AtlasResidential,
    VehiclesAtlasOut,
)
from pilates.database.schema.beam_schema import (
    BeamEventsParquet,
    BeamPlansOut,
    BeamFinalVehicles,
    BeamEventsActEnd,
    BeamEventsActStart,
    BeamEventsArrival,
    BeamEventsDeparture,
    BeamLinkstats,
    BeamEventsLeavingParkingEvent,
    BeamEventsModeChoice,
    BeamEventsParkingEvent,
    BeamEventsPathTraversal,
    BeamEventsPersonCost,
    BeamEventsPersonEntersVehicle,
    BeamEventsPersonLeavesVehicle,
    BeamEventsReplanning,
    BeamEventsReserveRideHail,
    BeamEventsTeleportationEvent,
    BeamNetworkFinal,
    BeamPathTraversalLinks,
    BeamRouteHistory,
    HouseholdsBeamIn,
    PlansBeamIn,
    PersonsBeamIn,
    VehiclesBeamIn,
)
from pilates.database.schema.urbansim_schema import (
    ActivitysimPostprocessUsimHouseholdsUpdated,
    ActivitysimPostprocessUsimPersonsUpdated,
    AtlasPostprocessUsimHouseholdsUpdated,
    UrbansimPostprocessUsimBlocksTable,
    UrbansimPostprocessUsimGraveyardTable,
    UrbansimPostprocessUsimHouseholdsTable,
    UrbansimPostprocessUsimJobsTable,
    UrbansimPostprocessUsimPersonsTable,
    UrbansimPostprocessUsimResidentialUnitsTable,
    UrbansimPostprocessUsimTazZoneGeomsTable,
    UrbansimPostprocessUsimWorkLocationsTable,
)
from pilates.workflows.artifact_keys import BEAM_NETWORK_FINAL

# Central mapping of artifact keys to curated schema classes.
# Keep keys in sync with pilates/workflows/artifact_keys.py.
_SCHEMA_BY_KEY: Dict[str, Type[SQLModel]] = {
    "households_asim_in": HouseholdsAsimIn,
    "land_use_asim_in": LandUseAsimIn,
    "persons_asim_in": PersonsAsimIn,
    "households_asim_out": HouseholdsAsimOut,
    "persons_asim_out": PersonsAsimOut,
    "tours_asim_out": ToursAsimOut,
    "beam_plans_asim_out": BeamPlansAsimOut,
    "trips_asim_out": TripsAsimOut,
    "accessibility_asim_out": AccessibilityAsimOut,
    "land_use_asim_out": LandUseAsimOut,
    "joint_tour_participants_asim_out": JointTourParticipantsAsimOut,
    "activitysim_postprocess_usim_households_table_updated": ActivitysimPostprocessUsimHouseholdsUpdated,
    "activitysim_postprocess_usim_persons_table_updated": ActivitysimPostprocessUsimPersonsUpdated,
    "atlas_postprocess_usim_households_table_input": AtlasPostprocessUsimHouseholdsUpdated,
    "atlas_postprocess_usim_households_table_updated": AtlasPostprocessUsimHouseholdsUpdated,
    "urbansim_postprocess_usim_blocks_table_archived": UrbansimPostprocessUsimBlocksTable,
    "urbansim_postprocess_usim_blocks_table_updated": UrbansimPostprocessUsimBlocksTable,
    "urbansim_postprocess_usim_graveyard_table_archived": UrbansimPostprocessUsimGraveyardTable,
    "urbansim_postprocess_usim_graveyard_table_updated": UrbansimPostprocessUsimGraveyardTable,
    "urbansim_postprocess_usim_households_table_archived": UrbansimPostprocessUsimHouseholdsTable,
    "urbansim_postprocess_usim_households_table_updated": UrbansimPostprocessUsimHouseholdsTable,
    "urbansim_postprocess_usim_jobs_table_archived": UrbansimPostprocessUsimJobsTable,
    "urbansim_postprocess_usim_jobs_table_updated": UrbansimPostprocessUsimJobsTable,
    "urbansim_postprocess_usim_persons_table_archived": UrbansimPostprocessUsimPersonsTable,
    "urbansim_postprocess_usim_persons_table_updated": UrbansimPostprocessUsimPersonsTable,
    "urbansim_postprocess_usim_residential_units_table_archived": UrbansimPostprocessUsimResidentialUnitsTable,
    "urbansim_postprocess_usim_residential_units_table_updated": UrbansimPostprocessUsimResidentialUnitsTable,
    "urbansim_postprocess_usim_taz_zone_geoms_table_archived": UrbansimPostprocessUsimTazZoneGeomsTable,
    "urbansim_postprocess_usim_taz_zone_geoms_table_updated": UrbansimPostprocessUsimTazZoneGeomsTable,
    "urbansim_postprocess_usim_work_locations_table_archived": UrbansimPostprocessUsimWorkLocationsTable,
    "urbansim_postprocess_usim_work_locations_table_updated": UrbansimPostprocessUsimWorkLocationsTable,
    "households_beam_in": HouseholdsBeamIn,
    "persons_beam_in": PersonsBeamIn,
    "plans_beam_in": PlansBeamIn,
    "beam_plans_out": BeamPlansOut,
    BEAM_NETWORK_FINAL: BeamNetworkFinal,
    "linkstats": BeamLinkstats,
    "linkstats_warmstart": BeamLinkstats,
    "atlas_blocks_csv": AtlasBlocks,
    "atlas_grave_csv": AtlasGrave,
    "atlas_households_csv": AtlasHousehold,
    "atlas_jobs_csv": AtlasJobs,
    "atlas_persons_csv": AtlasPersons,
    "atlas_residential_csv": AtlasResidential,
    "atlas_vehicles2_input": VehiclesAtlasOut,
    "atlas_vehicles2_output": VehiclesAtlasOut,
    "vehicles_beam_in": VehiclesBeamIn,
}

_SCHEMA_BY_PREFIX: Dict[str, Type[SQLModel]] = {
    # BEAM linkstats artifacts are commonly iteration/version-scoped keys, e.g.
    # linkstats_parquet_2018_0_sub1 or linkstats_unmodified_parquet__y2018__i0__...
    "events_parquet_": BeamEventsParquet,
    "final_vehicles_": BeamFinalVehicles,
    "linkstats_": BeamLinkstats,
    "path_traversal_links_": BeamPathTraversalLinks,
    "route_history_": BeamRouteHistory,
    # ATLAS run outputs are year-scoped keys, e.g. householdv_2023(.csv)
    # and vehicles_2023(.csv).
    "householdv_": HouseholdVAtlasOut,
    "vehicles_": VehiclesAtlasOut,
}

_EVENTS_PARQUET_SPLIT_SCHEMA_BY_TYPE: Dict[str, Type[SQLModel]] = {
    "LeavingParkingEvent": BeamEventsLeavingParkingEvent,
    "ModeChoice": BeamEventsModeChoice,
    "ParkingEvent": BeamEventsParkingEvent,
    "PathTraversal": BeamEventsPathTraversal,
    "PersonCost": BeamEventsPersonCost,
    "PersonEntersVehicle": BeamEventsPersonEntersVehicle,
    "PersonLeavesVehicle": BeamEventsPersonLeavesVehicle,
    "Replanning": BeamEventsReplanning,
    "ReserveRideHail": BeamEventsReserveRideHail,
    "TeleportationEvent": BeamEventsTeleportationEvent,
    "actend": BeamEventsActEnd,
    "actstart": BeamEventsActStart,
    "arrival": BeamEventsArrival,
    "departure": BeamEventsDeparture,
}


def get_consist_schemas() -> list[Type[SQLModel]]:
    """Return unique schema classes to register with Consist Tracker."""
    unique = {schema for schema in _SCHEMA_BY_KEY.values()}
    unique.update(_SCHEMA_BY_PREFIX.values())
    unique.update(_EVENTS_PARQUET_SPLIT_SCHEMA_BY_TYPE.values())
    return sorted(unique, key=lambda cls: cls.__name__)


def get_schema_for_key(key: Optional[str]) -> Optional[Type[SQLModel]]:
    """Look up a curated schema for a given artifact key."""
    if not key:
        return None
    resolved_key = resolve_artifact_key(key)
    exact = _SCHEMA_BY_KEY.get(resolved_key)
    if exact is not None:
        return exact
    split_event_schema = _schema_for_events_parquet_split_key(resolved_key)
    if split_event_schema is not None:
        return split_event_schema
    for prefix, schema in _SCHEMA_BY_PREFIX.items():
        if resolved_key.startswith(prefix):
            return schema
    return None


def _schema_for_events_parquet_split_key(
    resolved_key: str,
) -> Optional[Type[SQLModel]]:
    if not resolved_key.startswith("events_parquet_") or "_type_" not in resolved_key:
        return None
    event_type = resolved_key.split("_type_", 1)[1]
    return _EVENTS_PARQUET_SPLIT_SCHEMA_BY_TYPE.get(event_type)
