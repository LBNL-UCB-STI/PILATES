from sqlalchemy import ForeignKeyConstraint

from pilates.database.schema.beam_schema import (
    BeamEventsPathTraversal,
    BeamEventsModeChoice,
    BeamPathTraversalLinks,
    HouseholdsBeamIn,
    PersonsBeamIn,
    PlansBeamIn,
    VehiclesBeamIn,
)
from pilates.database.schema.activitysim_schema import BeamPlansAsimOut


def _table_arg_fk_targets(model_cls):
    table_args = getattr(model_cls, "__table_args__", ())
    if isinstance(table_args, dict):
        return set()
    targets = set()
    for arg in table_args:
        if isinstance(arg, ForeignKeyConstraint):
            for element in arg.elements:
                targets.add(str(element.target_fullname))
    return targets


def _column_fk_targets(model_cls, field_name: str):
    column = model_cls.model_fields[field_name].sa_column
    return {str(fk.target_fullname) for fk in column.foreign_keys}


def test_households_beam_in_bridges_to_households_asim_out() -> None:
    targets = _column_fk_targets(HouseholdsBeamIn, "household_id")
    assert "HouseholdsAsimOut.household_id" in targets


def test_persons_beam_in_bridges_to_persons_asim_out() -> None:
    targets = _column_fk_targets(PersonsBeamIn, "person_id")
    assert "PersonsAsimOut.person_id" in targets


def test_vehicles_beam_in_bridges_to_vehicles_atlas_out() -> None:
    targets = _table_arg_fk_targets(VehiclesBeamIn)
    assert "VehiclesAtlasOut.household_id" in targets
    assert "VehiclesAtlasOut.vehicle_id" in targets
    assert "VehiclesAtlasOut.year" in targets
    assert _column_fk_targets(VehiclesBeamIn, "source_vehicle_id") == set()


def test_vehicles_beam_in_uses_string_vehicle_id_and_int_source_vehicle_id() -> None:
    vehicle_id_col = VehiclesBeamIn.model_fields["vehicle_id"].sa_column
    source_vehicle_id_col = VehiclesBeamIn.model_fields["source_vehicle_id"].sa_column
    assert vehicle_id_col.type.__class__.__name__ == "String"
    assert source_vehicle_id_col.type.__class__.__name__ == "BigInteger"


def test_plans_beam_in_trip_id_points_to_trips_asim_out() -> None:
    trip_col = PlansBeamIn.model_fields["trip_id"].sa_column
    fk_targets = {str(fk.target_fullname) for fk in trip_col.foreign_keys}
    assert "tripsAsimOut.trip_id" in fk_targets


def test_plans_beam_in_tour_id_points_to_tours_asim_out() -> None:
    tour_col = PlansBeamIn.model_fields["tour_id"].sa_column
    fk_targets = {str(fk.target_fullname) for fk in tour_col.foreign_keys}
    assert "ToursAsimOut.tour_id" in fk_targets


def test_beam_plans_asim_out_bridges_trip_and_tour_ids() -> None:
    trip_targets = {
        str(fk.target_fullname)
        for fk in BeamPlansAsimOut.model_fields["trip_id"].sa_column.foreign_keys
    }
    tour_targets = {
        str(fk.target_fullname)
        for fk in BeamPlansAsimOut.model_fields["tour_id"].sa_column.foreign_keys
    }
    assert "tripsAsimOut.trip_id" in trip_targets
    assert "ToursAsimOut.tour_id" in tour_targets


def test_beam_events_path_traversal_vehicle_is_not_hard_fk() -> None:
    vehicle_targets = _column_fk_targets(BeamEventsPathTraversal, "vehicle")
    assert vehicle_targets == set()


def test_beam_events_path_traversal_exposes_soft_vehicle_id_int() -> None:
    vehicle_id_int_col = BeamEventsPathTraversal.model_fields[
        "vehicle_id_int"
    ].sa_column
    assert vehicle_id_int_col.type.__class__.__name__ == "BigInteger"
    assert vehicle_id_int_col.foreign_keys == set()


def test_beam_events_mode_choice_trip_and_person_fks() -> None:
    trip_targets = _column_fk_targets(BeamEventsModeChoice, "tripid")
    person_targets = _column_fk_targets(BeamEventsModeChoice, "person")
    assert "tripsAsimOut.trip_id" in trip_targets
    assert "PersonsBeamIn.person_id" in person_targets


def test_beam_path_traversal_links_point_to_traversals_and_network() -> None:
    path_targets = _column_fk_targets(BeamPathTraversalLinks, "pathtraversaleventid")
    link_targets = _column_fk_targets(BeamPathTraversalLinks, "linkid")
    assert "BeamEventsPathTraversal.PathTraversalEventId" in path_targets
    assert "BeamNetworkFinal.linkId" in link_targets
