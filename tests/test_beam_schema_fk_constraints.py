from sqlalchemy import ForeignKeyConstraint

from pilates.database.schema.beam_schema import (
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
