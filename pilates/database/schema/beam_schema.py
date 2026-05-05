from __future__ import annotations

from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
)
from sqlmodel import Field, SQLModel
class HouseholdsBeamIn(SQLModel, table=True):
    __tablename__ = "HouseholdsBeamIn"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    household_id: Optional[int] = Field(
        default=None,
        description="Household identifier used to relate persons to households.",
        sa_column=Column(
            "household_id",
            BigInteger,
            ForeignKey("HouseholdsAsimOut.household_id"),
            nullable=True,
            index=True,
        ),
    )
    block_id: Optional[int] = Field(
        default=None,
        description="Census block identifier for household location.",
        sa_column=Column("block_id", BigInteger, nullable=True, index=True),
    )
    home_zone_id: Optional[int] = Field(
        default=None,
        description="Home TAZ for the household.",
        sa_column=Column("home_zone_id", BigInteger, nullable=True, index=True),
    )
    income: Optional[float] = Field(
        default=None, sa_column=Column("income", Float, nullable=True)
    )
    hhsize: Optional[int] = Field(
        default=None, sa_column=Column("hhsize", BigInteger, nullable=True)
    )
    hht: Optional[int] = Field(
        default=None, sa_column=Column("HHT", BigInteger, nullable=True)
    )
    cars: Optional[int] = Field(
        default=None, sa_column=Column("cars", BigInteger, nullable=True)
    )
    num_workers: Optional[float] = Field(
        default=None, sa_column=Column("num_workers", Float, nullable=True)
    )
    sample_rate: Optional[float] = Field(
        default=None, sa_column=Column("sample_rate", Float, nullable=True)
    )
    income_in_thousands: Optional[float] = Field(
        default=None,
        sa_column=Column("income_in_thousands", Float, nullable=True),
    )
    income_segment: Optional[int] = Field(
        default=None, sa_column=Column("income_segment", BigInteger, nullable=True)
    )
    median_value_of_time: Optional[float] = Field(
        default=None,
        sa_column=Column("median_value_of_time", Float, nullable=True),
    )
    hh_value_of_time: Optional[float] = Field(
        default=None, sa_column=Column("hh_value_of_time", Float, nullable=True)
    )
    num_non_workers: Optional[float] = Field(
        default=None, sa_column=Column("num_non_workers", Float, nullable=True)
    )
    num_drivers: Optional[int] = Field(
        default=None, sa_column=Column("num_drivers", BigInteger, nullable=True)
    )
    num_adults: Optional[int] = Field(
        default=None, sa_column=Column("num_adults", BigInteger, nullable=True)
    )
    num_children: Optional[int] = Field(
        default=None, sa_column=Column("num_children", BigInteger, nullable=True)
    )
    num_young_children: Optional[int] = Field(
        default=None,
        sa_column=Column("num_young_children", BigInteger, nullable=True),
    )
    num_children_5_to_15: Optional[int] = Field(
        default=None,
        sa_column=Column("num_children_5_to_15", BigInteger, nullable=True),
    )
    num_children_16_to_17: Optional[int] = Field(
        default=None,
        sa_column=Column("num_children_16_to_17", BigInteger, nullable=True),
    )
    num_college_age: Optional[int] = Field(
        default=None,
        sa_column=Column("num_college_age", BigInteger, nullable=True),
    )
    num_young_adults: Optional[int] = Field(
        default=None,
        sa_column=Column("num_young_adults", BigInteger, nullable=True),
    )
    non_family: Optional[bool] = Field(
        default=None, sa_column=Column("non_family", Boolean, nullable=True)
    )
    family: Optional[bool] = Field(
        default=None, sa_column=Column("family", Boolean, nullable=True)
    )
    home_is_urban: Optional[bool] = Field(
        default=None, sa_column=Column("home_is_urban", Boolean, nullable=True)
    )
    home_is_rural: Optional[bool] = Field(
        default=None, sa_column=Column("home_is_rural", Boolean, nullable=True)
    )
    hh_work_auto_savings_ratio: Optional[float] = Field(
        default=None,
        sa_column=Column("hh_work_auto_savings_ratio", Float, nullable=True),
    )
    num_under16_not_at_school: Optional[int] = Field(
        default=None,
        sa_column=Column("num_under16_not_at_school", BigInteger, nullable=True),
    )
    num_travel_active: Optional[int] = Field(
        default=None,
        sa_column=Column("num_travel_active", BigInteger, nullable=True),
    )
    num_travel_active_adults: Optional[int] = Field(
        default=None,
        sa_column=Column("num_travel_active_adults", BigInteger, nullable=True),
    )
    num_travel_active_preschoolers: Optional[int] = Field(
        default=None,
        sa_column=Column("num_travel_active_preschoolers", BigInteger, nullable=True),
    )
    num_travel_active_children: Optional[int] = Field(
        default=None,
        sa_column=Column("num_travel_active_children", BigInteger, nullable=True),
    )
    num_travel_active_non_preschoolers: Optional[int] = Field(
        default=None,
        sa_column=Column(
            "num_travel_active_non_preschoolers", BigInteger, nullable=True
        ),
    )
    participates_in_jtf_model: Optional[bool] = Field(
        default=None,
        sa_column=Column("participates_in_jtf_model", Boolean, nullable=True),
    )
    joint_tour_frequency: Optional[str] = Field(
        default=None, sa_column=Column("joint_tour_frequency", String, nullable=True)
    )
    num_hh_joint_tours: Optional[int] = Field(
        default=None,
        sa_column=Column("num_hh_joint_tours", BigInteger, nullable=True),
    )


class PersonsBeamIn(SQLModel, table=True):
    __tablename__ = "PersonsBeamIn"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    person_id: Optional[int] = Field(
        default=None,
        description="Person identifier unique within the input population.",
        sa_column=Column(
            "person_id",
            BigInteger,
            ForeignKey("PersonsAsimOut.person_id"),
            nullable=True,
            index=True,
        ),
    )
    household_id: Optional[int] = Field(
        default=None,
        description="Household identifier to link persons to households.",
        sa_column=Column(
            "household_id",
            BigInteger,
            ForeignKey("HouseholdsBeamIn.household_id"),
            nullable=True,
            index=True,
        ),
    )
    age: Optional[int] = Field(
        default=None, sa_column=Column("age", BigInteger, nullable=True)
    )
    pnum: Optional[int] = Field(
        default=None,
        description="Person number within household (PNUM).",
        sa_column=Column("PNUM", BigInteger, nullable=True),
    )
    sex: Optional[int] = Field(
        default=None, sa_column=Column("sex", BigInteger, nullable=True)
    )
    pemploy: Optional[int] = Field(
        default=None, sa_column=Column("pemploy", BigInteger, nullable=True)
    )
    pstudent: Optional[int] = Field(
        default=None, sa_column=Column("pstudent", BigInteger, nullable=True)
    )
    ptype: Optional[int] = Field(
        default=None, sa_column=Column("ptype", BigInteger, nullable=True)
    )
    home_x: Optional[float] = Field(
        default=None, sa_column=Column("home_x", Float, nullable=True)
    )
    home_y: Optional[float] = Field(
        default=None, sa_column=Column("home_y", Float, nullable=True)
    )
    age_16_to_19: Optional[bool] = Field(
        default=None, sa_column=Column("age_16_to_19", Boolean, nullable=True)
    )
    age_16_p: Optional[bool] = Field(
        default=None, sa_column=Column("age_16_p", Boolean, nullable=True)
    )
    adult: Optional[bool] = Field(
        default=None, sa_column=Column("adult", Boolean, nullable=True)
    )
    male: Optional[bool] = Field(
        default=None, sa_column=Column("male", Boolean, nullable=True)
    )
    female: Optional[bool] = Field(
        default=None, sa_column=Column("female", Boolean, nullable=True)
    )
    has_non_worker: Optional[bool] = Field(
        default=None, sa_column=Column("has_non_worker", Boolean, nullable=True)
    )
    has_retiree: Optional[bool] = Field(
        default=None, sa_column=Column("has_retiree", Boolean, nullable=True)
    )
    has_preschool_kid: Optional[bool] = Field(
        default=None, sa_column=Column("has_preschool_kid", Boolean, nullable=True)
    )
    has_driving_kid: Optional[bool] = Field(
        default=None, sa_column=Column("has_driving_kid", Boolean, nullable=True)
    )
    has_school_kid: Optional[bool] = Field(
        default=None, sa_column=Column("has_school_kid", Boolean, nullable=True)
    )
    has_full_time: Optional[bool] = Field(
        default=None, sa_column=Column("has_full_time", Boolean, nullable=True)
    )
    has_part_time: Optional[bool] = Field(
        default=None, sa_column=Column("has_part_time", Boolean, nullable=True)
    )
    has_university: Optional[bool] = Field(
        default=None, sa_column=Column("has_university", Boolean, nullable=True)
    )
    student_is_employed: Optional[bool] = Field(
        default=None,
        sa_column=Column("student_is_employed", Boolean, nullable=True),
    )
    nonstudent_to_school: Optional[bool] = Field(
        default=None,
        sa_column=Column("nonstudent_to_school", Boolean, nullable=True),
    )
    is_student: Optional[bool] = Field(
        default=None, sa_column=Column("is_student", Boolean, nullable=True)
    )
    is_gradeschool: Optional[bool] = Field(
        default=None, sa_column=Column("is_gradeschool", Boolean, nullable=True)
    )
    is_highschool: Optional[bool] = Field(
        default=None, sa_column=Column("is_highschool", Boolean, nullable=True)
    )
    is_university: Optional[bool] = Field(
        default=None, sa_column=Column("is_university", Boolean, nullable=True)
    )
    school_segment: Optional[int] = Field(
        default=None, sa_column=Column("school_segment", BigInteger, nullable=True)
    )
    is_worker: Optional[bool] = Field(
        default=None, sa_column=Column("is_worker", Boolean, nullable=True)
    )
    home_zone_id: Optional[int] = Field(
        default=None,
        description="Home TAZ for the person.",
        sa_column=Column("home_zone_id", BigInteger, nullable=True, index=True),
    )
    value_of_time: Optional[float] = Field(
        default=None, sa_column=Column("value_of_time", Float, nullable=True)
    )
    school_zone_id: Optional[int] = Field(
        default=None, sa_column=Column("school_zone_id", BigInteger, nullable=True)
    )
    workplace_zone_id: Optional[int] = Field(
        default=None, sa_column=Column("workplace_zone_id", BigInteger, nullable=True)
    )
    distance_to_work: Optional[float] = Field(
        default=None, sa_column=Column("distance_to_work", Float, nullable=True)
    )
    roundtrip_auto_time_to_work: Optional[float] = Field(
        default=None,
        sa_column=Column("roundtrip_auto_time_to_work", Float, nullable=True),
    )
    work_from_home: Optional[bool] = Field(
        default=None, sa_column=Column("work_from_home", Boolean, nullable=True)
    )
    transit_pass_ownership: Optional[int] = Field(
        default=None,
        sa_column=Column("transit_pass_ownership", BigInteger, nullable=True),
    )
    transit_pass_subsidy: Optional[float] = Field(
        default=None, sa_column=Column("transit_pass_subsidy", Float, nullable=True)
    )
    parking_cost: Optional[float] = Field(
        default=None, sa_column=Column("parking_cost", Float, nullable=True)
    )
    has_electric_bike: Optional[bool] = Field(
        default=None, sa_column=Column("has_electric_bike", Boolean, nullable=True)
    )
    has_carshare: Optional[bool] = Field(
        default=None, sa_column=Column("has_carshare", Boolean, nullable=True)
    )
    has_rideshare: Optional[bool] = Field(
        default=None, sa_column=Column("has_rideshare", Boolean, nullable=True)
    )
    free_parking_at_work: Optional[bool] = Field(
        default=None,
        sa_column=Column("free_parking_at_work", Boolean, nullable=True),
    )


class VehiclesBeamIn(SQLModel, table=True):
    __tablename__ = "VehiclesBeamIn"
    __table_args__ = (
        ForeignKeyConstraint(
            ["household_id", "source_vehicle_id", "year"],
            [
                "VehiclesAtlasOut.household_id",
                "VehiclesAtlasOut.vehicle_id",
                "VehiclesAtlasOut.year",
            ],
        ),
        {"extend_existing": True},
    )
    __abstract__ = True

    household_id: Optional[int] = Field(
        default=None,
        description="Household identifier associated with the vehicle.",
        sa_column=Column(
            "household_id",
            BigInteger,
            ForeignKey("HouseholdsBeamIn.household_id"),
            primary_key=True,
            nullable=True,
            index=True,
        ),
    )
    vehicle_id: Optional[str] = Field(
        default=None,
        description="BEAM-facing vehicle identifier.",
        sa_column=Column(
            "vehicle_id", String, primary_key=True, nullable=True, index=True
        ),
    )
    source_vehicle_id: Optional[int] = Field(
        default=None,
        description="Original ATLAS vehicle identifier preserved for lineage and foreign-key joins.",
        sa_column=Column("sourceVehicleId", BigInteger, nullable=True, index=True),
    )
    bodytype: Optional[str] = Field(
        default=None,
        description="Body type category for the vehicle.",
        sa_column=Column("bodytype", String, nullable=True),
    )
    pred_power: Optional[str] = Field(
        default=None,
        description="Predicted powertrain category.",
        sa_column=Column("pred_power", String, nullable=True),
    )
    ownlease: Optional[str] = Field(
        default=None,
        description="Ownership/lease status for the vehicle.",
        sa_column=Column("ownlease", String, nullable=True),
    )
    modelyear: Optional[int] = Field(
        default=None,
        description="Vehicle model year.",
        sa_column=Column("modelyear", BigInteger, nullable=True),
    )
    adopt_fuel: Optional[str] = Field(
        default=None,
        description="Adopted fuel type for the vehicle.",
        sa_column=Column("adopt_fuel", String, nullable=True),
    )
    adopt_veh: Optional[str] = Field(
        default=None,
        description="Vehicle technology adoption category.",
        sa_column=Column("adopt_veh", String, nullable=True),
    )
    acquire_year: Optional[str] = Field(
        default=None,
        description="Vehicle acquisition year.",
        sa_column=Column("acquire_year", String, nullable=True),
    )
    vehicle_tag: Optional[str] = Field(
        default=None,
        description="Vehicle tag/identifier label.",
        sa_column=Column("vehicle_tag", String, nullable=True),
    )
    year: Optional[int] = Field(
        default=None,
        description="Model year associated with the output record.",
        sa_column=Column("year", BigInteger, primary_key=True, nullable=True, index=True),
    )
    newhhflag: Optional[str] = Field(
        default=None,
        description="Indicator for newly created household in the model year.",
        sa_column=Column("newhhflag", String, nullable=True),
    )
    maindriver_id: Optional[int] = Field(
        default=None,
        description="Primary driver identifier for the vehicle.",
        sa_column=Column(
            "maindriver_id",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    vintage_category: Optional[str] = Field(
        default=None,
        description="Vehicle vintage/age category.",
        sa_column=Column("vintage_category", String, nullable=True),
    )
    vehicletypeid: Optional[str] = Field(
        default=None,
        description="BEAM vehicle type identifier composed by the ATLAS postprocessor.",
        sa_column=Column("vehicleTypeId", String, nullable=True, index=True),
    )


class PlansBeamIn(SQLModel, table=True):
    __tablename__ = "PlansBeamIn"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    # Same as BeamPlansAsimOut plus BEAM-remembered plan fields.
    tour_id: Optional[int] = Field(
        default=None,
        description="Tour identifier in ActivitySim outputs.",
        sa_column=Column(
            "tour_id",
            BigInteger,
            ForeignKey("ToursAsimOut.tour_id"),
            nullable=True,
            index=True,
        ),
    )
    trip_id: Optional[int] = Field(
        default=None,
        description="Trip identifier in ActivitySim outputs.",
        sa_column=Column(
            "trip_id",
            BigInteger,
            ForeignKey("tripsAsimOut.trip_id"),
            nullable=True,
            index=True,
        ),
    )
    person_id: Optional[int] = Field(
        default=None,
        description="Person identifier associated with the trip/tour.",
        sa_column=Column(
            "person_id",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    number_of_participants: Optional[float] = Field(
        default=None, sa_column=Column("number_of_participants", Float, nullable=True)
    )
    tour_mode: Optional[str] = Field(
        default=None,
        description="Primary tour mode for the tour.",
        sa_column=Column("tour_mode", String, nullable=True),
    )
    trip_mode: Optional[str] = Field(
        default=None,
        description="Mode used for the specific trip.",
        sa_column=Column("trip_mode", String, nullable=True),
    )
    planelementindex: Optional[int] = Field(
        default=None, sa_column=Column("PlanElementIndex", BigInteger, nullable=True)
    )
    activityelement: Optional[str] = Field(
        default=None,
        description="Plan element type (activity or leg) for BEAM.",
        sa_column=Column("ActivityElement", String, nullable=True),
    )
    activitytype: Optional[str] = Field(
        default=None,
        description="Activity type for BEAM plans.",
        sa_column=Column("ActivityType", String, nullable=True),
    )
    x: Optional[float] = Field(default=None, sa_column=Column("x", Float, nullable=True))
    y: Optional[float] = Field(default=None, sa_column=Column("y", Float, nullable=True))
    departure_time: Optional[float] = Field(
        default=None, sa_column=Column("departure_time", Float, nullable=True)
    )
    trip_dur_min: Optional[float] = Field(
        default=None, sa_column=Column("trip_dur_min", Float, nullable=True)
    )
    trip_cost_dollars: Optional[float] = Field(
        default=None, sa_column=Column("trip_cost_dollars", Float, nullable=True)
    )
    planindex: Optional[int] = Field(
        default=None,
        description="Plan index from remembered BEAM plans.",
        sa_column=Column("planindex", BigInteger, nullable=True),
    )
    planselected: Optional[bool] = Field(
        default=None,
        description="Whether the plan was selected in BEAM.",
        sa_column=Column("planselected", Boolean, nullable=True),
    )
    planscore: Optional[float] = Field(
        default=None,
        description="Score for the BEAM plan.",
        sa_column=Column("planscore", Float, nullable=True),
    )
    legtraveltime: Optional[float] = Field(
        default=None,
        description="Travel time for the BEAM leg.",
        sa_column=Column("legtraveltime", Float, nullable=True),
    )
    legroutetype: Optional[str] = Field(
        default=None,
        description="Route type for the BEAM leg.",
        sa_column=Column("legroutetype", String, nullable=True),
    )
    legroutestartlink: Optional[int] = Field(
        default=None,
        description="Start link for the BEAM leg route.",
        sa_column=Column(
            "legroutestartlink",
            BigInteger,
            ForeignKey("BeamNetworkFinal.linkId"),
            nullable=True,
            index=True,
        ),
    )
    legrouteendlink: Optional[int] = Field(
        default=None,
        description="End link for the BEAM leg route.",
        sa_column=Column(
            "legrouteendlink",
            BigInteger,
            ForeignKey("BeamNetworkFinal.linkId"),
            nullable=True,
            index=True,
        ),
    )
    legroutetraveltime: Optional[float] = Field(
        default=None,
        description="Route travel time for the BEAM leg.",
        sa_column=Column("legroutetraveltime", Float, nullable=True),
    )
    legroutedistance: Optional[float] = Field(
        default=None,
        description="Route distance for the BEAM leg.",
        sa_column=Column("legroutedistance", Float, nullable=True),
    )
    legroutelinks: Optional[str] = Field(
        default=None,
        description="Serialized list of route links for the BEAM leg.",
        sa_column=Column("legroutelinks", String, nullable=True),
    )


class BeamPlansOut(SQLModel, table=True):
    __tablename__ = "BeamPlansOut"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    trip_id: Optional[int] = Field(
        default=None,
        description="Trip identifier carried in BEAM plans output.",
        sa_column=Column("tripId", BigInteger, nullable=True, index=True),
    )
    person_id: Optional[int] = Field(
        default=None,
        description="Person identifier associated with the BEAM plan output row.",
        sa_column=Column(
            "personId",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    plan_index: Optional[int] = Field(default=None, sa_column=Column("planIndex", BigInteger, nullable=True))
    plan_score: Optional[float] = Field(default=None, sa_column=Column("planScore", Float, nullable=True))
    plan_selected: Optional[bool] = Field(default=None, sa_column=Column("planSelected", Boolean, nullable=True))
    plan_element_type: Optional[str] = Field(default=None, sa_column=Column("planElementType", String, nullable=True))
    plan_element_index: Optional[int] = Field(default=None, sa_column=Column("planElementIndex", BigInteger, nullable=True))
    activity_type: Optional[str] = Field(default=None, sa_column=Column("activityType", String, nullable=True))
    activity_location_x: Optional[float] = Field(default=None, sa_column=Column("activityLocationX", Float, nullable=True))
    activity_location_y: Optional[float] = Field(default=None, sa_column=Column("activityLocationY", Float, nullable=True))
    activity_end_time: Optional[float] = Field(default=None, sa_column=Column("activityEndTime", Float, nullable=True))
    leg_mode: Optional[str] = Field(default=None, sa_column=Column("legMode", String, nullable=True))
    leg_departure_time: Optional[float] = Field(default=None, sa_column=Column("legDepartureTime", Float, nullable=True))
    trip_dur_min: Optional[str] = Field(default=None, sa_column=Column("trip_dur_min", String, nullable=True))
    trip_cost_dollars: Optional[str] = Field(default=None, sa_column=Column("trip_cost_dollars", String, nullable=True))
    leg_travel_time: Optional[float] = Field(default=None, sa_column=Column("legTravelTime", Float, nullable=True))
    leg_route_type: Optional[str] = Field(default=None, sa_column=Column("legRouteType", String, nullable=True))
    leg_route_start_link: Optional[int] = Field(
        default=None,
        description="Start link identifier for the BEAM route leg.",
        sa_column=Column(
            "legRouteStartLink",
            BigInteger,
            ForeignKey("BeamNetworkFinal.linkId"),
            nullable=True,
            index=True,
        ),
    )
    leg_route_end_link: Optional[int] = Field(
        default=None,
        description="End link identifier for the BEAM route leg.",
        sa_column=Column(
            "legRouteEndLink",
            BigInteger,
            ForeignKey("BeamNetworkFinal.linkId"),
            nullable=True,
            index=True,
        ),
    )
    leg_route_travel_time: Optional[float] = Field(default=None, sa_column=Column("legRouteTravelTime", Float, nullable=True))
    leg_route_distance: Optional[float] = Field(default=None, sa_column=Column("legRouteDistance", Float, nullable=True))
    leg_route_links: Optional[str] = Field(default=None, sa_column=Column("legRouteLinks", String, nullable=True))


class BeamEventsParquet(SQLModel, table=True):
    __tablename__ = "BeamEventsParquet"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    capacity: Optional[int] = Field(default=None, sa_column=Column("capacity", Integer, nullable=True))
    tour_index: Optional[int] = Field(default=None, sa_column=Column("tourIndex", Integer, nullable=True))
    weight: Optional[str] = Field(default=None, sa_column=Column("weight", String, nullable=True))
    current_activity: Optional[str] = Field(default=None, sa_column=Column("currentActivity", String, nullable=True))
    from_stop_index: Optional[str] = Field(default=None, sa_column=Column("fromStopIndex", String, nullable=True))
    driver: Optional[str] = Field(default=None, sa_column=Column("driver", String, nullable=True))
    num_passengers: Optional[int] = Field(default=None, sa_column=Column("numPassengers", Integer, nullable=True))
    start_y: Optional[float] = Field(default=None, sa_column=Column("startY", Float, nullable=True))
    act_type: Optional[str] = Field(default=None, sa_column=Column("actType", String, nullable=True))
    toll_cost: Optional[str] = Field(default=None, sa_column=Column("tollCost", String, nullable=True))
    trip_id_event: Optional[str] = Field(default=None, sa_column=Column("tripId", String, nullable=True))
    charging_point_type: Optional[str] = Field(default=None, sa_column=Column("chargingPointType", String, nullable=True))
    location_y: Optional[float] = Field(default=None, sa_column=Column("locationY", Float, nullable=True))
    vehicle: Optional[str] = Field(default=None, sa_column=Column("vehicle", String, nullable=True))
    y: Optional[str] = Field(default=None, sa_column=Column("y", String, nullable=True))
    end_x: Optional[float] = Field(default=None, sa_column=Column("endX", Float, nullable=True))
    shift_status: Optional[str] = Field(default=None, sa_column=Column("shiftStatus", String, nullable=True))
    secondary_fuel_level: Optional[float] = Field(default=None, sa_column=Column("secondaryFuelLevel", Float, nullable=True))
    time: Optional[float] = Field(default=None, sa_column=Column("time", Float, nullable=True))
    link: Optional[int] = Field(
        default=None,
        description="Network link identifier carried by the BEAM event row when present.",
        sa_column=Column("link", Integer, ForeignKey("BeamNetworkFinal.linkId"), nullable=True, index=True),
    )
    personal_vehicle_available: Optional[bool] = Field(default=None, sa_column=Column("personalVehicleAvailable", Boolean, nullable=True))
    payload_ids: Optional[str] = Field(default=None, sa_column=Column("PayloadIds", String, nullable=True))
    primary_fuel_type: Optional[str] = Field(default=None, sa_column=Column("primaryFuelType", String, nullable=True))
    current_tour_mode: Optional[str] = Field(default=None, sa_column=Column("currentTourMode", String, nullable=True))
    link_travel_time: Optional[str] = Field(default=None, sa_column=Column("linkTravelTime", String, nullable=True))
    reason: Optional[str] = Field(default=None, sa_column=Column("reason", String, nullable=True))
    net_cost: Optional[str] = Field(default=None, sa_column=Column("netCost", String, nullable=True))
    fuel: Optional[float] = Field(default=None, sa_column=Column("fuel", Float, nullable=True))
    next_activity: Optional[str] = Field(default=None, sa_column=Column("nextActivity", String, nullable=True))
    depart_time: Optional[int] = Field(default=None, sa_column=Column("departTime", Integer, nullable=True))
    seating_capacity: Optional[int] = Field(default=None, sa_column=Column("seatingCapacity", Integer, nullable=True))
    parking_type: Optional[str] = Field(default=None, sa_column=Column("parkingType", String, nullable=True))
    location: Optional[int] = Field(default=None, sa_column=Column("location", Integer, nullable=True))
    secondary_fuel: Optional[float] = Field(default=None, sa_column=Column("secondaryFuel", Float, nullable=True))
    person: Optional[str] = Field(default=None, sa_column=Column("person", String, nullable=True))
    available_alternatives: Optional[str] = Field(default=None, sa_column=Column("availableAlternatives", String, nullable=True))
    departure_time: Optional[int] = Field(default=None, sa_column=Column("departureTime", Integer, nullable=True))
    secondary_fuel_type: Optional[str] = Field(default=None, sa_column=Column("secondaryFuelType", String, nullable=True))
    arrival_time: Optional[int] = Field(default=None, sa_column=Column("arrivalTime", Integer, nullable=True))
    cost: Optional[float] = Field(default=None, sa_column=Column("cost", Float, nullable=True))
    parking_taz: Optional[str] = Field(default=None, sa_column=Column("parkingTaz", String, nullable=True))
    toll_paid: Optional[float] = Field(default=None, sa_column=Column("tollPaid", Float, nullable=True))
    trip_id: Optional[str] = Field(default=None, sa_column=Column("trip_id", String, nullable=True))
    payloads: Optional[str] = Field(default=None, sa_column=Column("payloads", String, nullable=True))
    duration: Optional[float] = Field(default=None, sa_column=Column("duration", Float, nullable=True))
    expected_maximum_utility: Optional[float] = Field(default=None, sa_column=Column("expectedMaximumUtility", Float, nullable=True))
    leg_modes: Optional[str] = Field(default=None, sa_column=Column("legModes", String, nullable=True))
    length: Optional[float] = Field(default=None, sa_column=Column("length", Float, nullable=True))
    to_stop_index: Optional[str] = Field(default=None, sa_column=Column("toStopIndex", String, nullable=True))
    links: Optional[str] = Field(default=None, sa_column=Column("links", String, nullable=True))
    end_y: Optional[float] = Field(default=None, sa_column=Column("endY", Float, nullable=True))
    mode: Optional[str] = Field(default=None, sa_column=Column("mode", String, nullable=True))
    type: Optional[str] = Field(default=None, sa_column=Column("type", String, nullable=True))
    score: Optional[float] = Field(default=None, sa_column=Column("score", Float, nullable=True))
    start_x: Optional[float] = Field(default=None, sa_column=Column("startX", Float, nullable=True))
    facility: Optional[str] = Field(default=None, sa_column=Column("facility", String, nullable=True))
    location_x: Optional[float] = Field(default=None, sa_column=Column("locationX", Float, nullable=True))
    incentive: Optional[str] = Field(default=None, sa_column=Column("incentive", String, nullable=True))
    x: Optional[str] = Field(default=None, sa_column=Column("x", String, nullable=True))
    primary_fuel: Optional[float] = Field(default=None, sa_column=Column("primaryFuel", Float, nullable=True))
    riders: Optional[str] = Field(default=None, sa_column=Column("riders", String, nullable=True))
    current_trip_mode: Optional[str] = Field(default=None, sa_column=Column("currentTripMode", String, nullable=True))
    leg_vehicle_ids: Optional[str] = Field(default=None, sa_column=Column("legVehicleIds", String, nullable=True))
    payload_weight_in_kg: Optional[str] = Field(default=None, sa_column=Column("PayloadWeightInKg", String, nullable=True))
    require_wheelchair: Optional[str] = Field(default=None, sa_column=Column("requireWheelchair", String, nullable=True))
    pricing_model: Optional[str] = Field(default=None, sa_column=Column("pricingModel", String, nullable=True))
    vehicle_type: Optional[str] = Field(default=None, sa_column=Column("vehicleType", String, nullable=True))
    parking_zone_id: Optional[str] = Field(default=None, sa_column=Column("parkingZoneId", String, nullable=True))
    leg_mode: Optional[str] = Field(default=None, sa_column=Column("legMode", String, nullable=True))
    emissions: Optional[str] = Field(default=None, sa_column=Column("emissions", String, nullable=True))
    primary_fuel_level: Optional[float] = Field(default=None, sa_column=Column("primaryFuelLevel", Float, nullable=True))
    price: Optional[float] = Field(default=None, sa_column=Column("price", Float, nullable=True))


class BeamFinalVehicles(SQLModel, table=True):
    __tablename__ = "BeamFinalVehicles"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    vehicle_id: Optional[int] = Field(
        default=None,
        description="BEAM vehicle identifier in the final vehicles summary.",
        sa_column=Column("vehicleId", BigInteger, nullable=True, index=True),
    )
    vehicle_type_id: Optional[str] = Field(default=None, sa_column=Column("vehicleTypeId", String, nullable=True))
    state_of_charge: Optional[float] = Field(default=None, sa_column=Column("stateOfCharge", Float, nullable=True))
    household_id: Optional[int] = Field(
        default=None,
        description="Household identifier associated with the BEAM vehicle.",
        sa_column=Column(
            "householdId",
            BigInteger,
            ForeignKey("HouseholdsBeamIn.household_id"),
            nullable=True,
            index=True,
        ),
    )


class BeamRouteHistory(SQLModel, table=True):
    __tablename__ = "BeamRouteHistory"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    time_bin: Optional[int] = Field(default=None, sa_column=Column("timeBin", BigInteger, nullable=True))
    origin_link_id: Optional[int] = Field(
        default=None,
        description="Origin link identifier for the recorded route.",
        sa_column=Column(
            "originLinkId",
            BigInteger,
            ForeignKey("BeamNetworkFinal.linkId"),
            nullable=True,
            index=True,
        ),
    )
    dest_link_id: Optional[int] = Field(
        default=None,
        description="Destination link identifier for the recorded route.",
        sa_column=Column(
            "destLinkId",
            BigInteger,
            ForeignKey("BeamNetworkFinal.linkId"),
            nullable=True,
            index=True,
        ),
    )
    route: Optional[str] = Field(default=None, sa_column=Column("route", String, nullable=True))


class BeamNetworkFinal(SQLModel, table=True):
    __tablename__ = "BeamNetworkFinal"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    link_id: Optional[int] = Field(
        default=None,
        description="Unique BEAM network link identifier.",
        sa_column=Column("linkId", BigInteger, nullable=True, index=True),
    )
    link_length: Optional[float] = Field(
        default=None,
        description="Link length in meters.",
        sa_column=Column("linkLength", Float, nullable=True),
    )
    link_free_speed: Optional[float] = Field(
        default=None,
        description="Free-flow speed on the link.",
        sa_column=Column("linkFreeSpeed", Float, nullable=True),
    )
    link_capacity: Optional[float] = Field(
        default=None,
        description="Link capacity.",
        sa_column=Column("linkCapacity", Float, nullable=True),
    )
    number_of_lanes: Optional[float] = Field(
        default=None,
        description="Number of lanes on the link.",
        sa_column=Column("numberOfLanes", Float, nullable=True),
    )
    link_modes: Optional[str] = Field(
        default=None,
        description="Allowed modes on the link.",
        sa_column=Column("linkModes", String, nullable=True),
    )
    attribute_orig_id: Optional[float] = Field(
        default=None,
        description="Original ID attribute from the source network (if available).",
        sa_column=Column("attributeOrigId", Float, nullable=True),
    )
    attribute_orig_type: Optional[str] = Field(
        default=None,
        description="Original type attribute from the source network (if available).",
        sa_column=Column("attributeOrigType", String, nullable=True),
    )
    from_node_id: Optional[int] = Field(
        default=None,
        description="From-node identifier.",
        sa_column=Column("fromNodeId", BigInteger, nullable=True),
    )
    to_node_id: Optional[int] = Field(
        default=None,
        description="To-node identifier.",
        sa_column=Column("toNodeId", BigInteger, nullable=True),
    )
    from_location_x: Optional[float] = Field(
        default=None,
        description="X coordinate for the from-node.",
        sa_column=Column("fromLocationX", Float, nullable=True),
    )
    from_location_y: Optional[float] = Field(
        default=None,
        description="Y coordinate for the from-node.",
        sa_column=Column("fromLocationY", Float, nullable=True),
    )
    to_location_x: Optional[float] = Field(
        default=None,
        description="X coordinate for the to-node.",
        sa_column=Column("toLocationX", Float, nullable=True),
    )
    to_location_y: Optional[float] = Field(
        default=None,
        description="Y coordinate for the to-node.",
        sa_column=Column("toLocationY", Float, nullable=True),
    )


class BeamLinkstats(SQLModel, table=True):
    __tablename__ = "BeamLinkstats"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    link_id: Optional[int] = Field(
        default=None,
        description="BEAM link identifier aligned with BeamNetworkFinal.link_id.",
        sa_column=Column(
            "link",
            BigInteger,
            ForeignKey("BeamNetworkFinal.linkId"),
            nullable=True,
            index=True,
        ),
    )
    from_node_id: Optional[int] = Field(
        default=None,
        description="From-node identifier for the link segment.",
        sa_column=Column("from", BigInteger, nullable=True, index=True),
    )
    to_node_id: Optional[int] = Field(
        default=None,
        description="To-node identifier for the link segment.",
        sa_column=Column("to", BigInteger, nullable=True, index=True),
    )
    hour: Optional[int] = Field(
        default=None,
        description="Hour of day for this linkstats observation.",
        sa_column=Column("hour", Integer, nullable=True, index=True),
    )
    length: Optional[float] = Field(
        default=None,
        description="Link length in meters.",
        sa_column=Column("length", Float, nullable=True),
    )
    freespeed: Optional[float] = Field(
        default=None,
        description="Link free-flow speed.",
        sa_column=Column("freespeed", Float, nullable=True),
    )
    capacity: Optional[float] = Field(
        default=None,
        description="Link capacity used during assignment.",
        sa_column=Column("capacity", Float, nullable=True),
    )
    stat: Optional[str] = Field(
        default=None,
        description="Statistic type for the row (for example, average or sum).",
        sa_column=Column("stat", String, nullable=True),
    )
    volume: Optional[float] = Field(
        default=None,
        description="Total assigned volume on the link for the hour.",
        sa_column=Column("volume", Float, nullable=True),
    )
    volume_class_456_vocational: Optional[float] = Field(
        default=None,
        description="Assigned volume for class 4-6 vocational vehicles.",
        sa_column=Column("volume_Class456Vocational", Float, nullable=True),
    )
    volume_class_78_vocational: Optional[float] = Field(
        default=None,
        description="Assigned volume for class 7-8 vocational vehicles.",
        sa_column=Column("volume_Class78Vocational", Float, nullable=True),
    )
    volume_class_78_tractor: Optional[float] = Field(
        default=None,
        description="Assigned volume for class 7-8 tractor vehicles.",
        sa_column=Column("volume_Class78Tractor", Float, nullable=True),
    )
    traveltime: Optional[float] = Field(
        default=None,
        description="Observed or simulated travel time for the link-hour.",
        sa_column=Column("traveltime", Float, nullable=True),
    )


class BeamEventsLeavingParkingEvent(SQLModel, table=True):
    __tablename__ = "BeamEventsLeavingParkingEvent"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    leavingparkingeventeventid: Optional[int] = Field(
        default=None,
        description="Synthetic event identifier assigned during events split.",
        sa_column=Column(
            "LeavingParkingEventEventId",
            BigInteger,
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    driver: Optional[int] = Field(
        default=None,
        description="Optional driver person identifier when it maps to PersonsBeamIn.",
        sa_column=Column(
            "driver",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    chargingpointtype: Optional[str] = Field(
        default=None,
        description="Charging point type used at the parking location.",
        sa_column=Column("chargingPointType", String, nullable=True),
    )
    vehicle: Optional[str] = Field(
        default=None,
        description=(
            "Raw BEAM vehicle reference. This may be a stringified household "
            "vehicle id or a non-household fleet/transit identifier."
        ),
        sa_column=Column("vehicle", String, nullable=True, index=True),
    )
    vehicle_id_int: Optional[int] = Field(
        default=None,
        description=(
            "Optional normalized integer vehicle id parsed from the raw BEAM "
            "vehicle reference when the row refers to a household vehicle."
        ),
        sa_column=Column("vehicle_id_int", BigInteger, nullable=True, index=True),
    )
    time: Optional[float] = Field(
        default=None,
        description="Event time in simulation seconds.",
        sa_column=Column("time", Float, nullable=True, index=True),
    )
    parkingtype: Optional[str] = Field(
        default=None,
        description="Parking type classification.",
        sa_column=Column("parkingType", String, nullable=True),
    )
    cost: Optional[float] = Field(
        default=None,
        description="Parking cost charged for the event.",
        sa_column=Column("cost", Float, nullable=True),
    )
    parkingtaz: Optional[str] = Field(
        default=None,
        description="Parking TAZ identifier emitted by BEAM.",
        sa_column=Column("parkingTaz", String, nullable=True),
    )
    duration: Optional[float] = Field(
        default=None,
        description="Parking duration in seconds.",
        sa_column=Column("duration", Float, nullable=True),
    )
    links: Optional[str] = Field(
        default=None,
        description="Serialized route links associated with the event.",
        sa_column=Column("links", String, nullable=True),
    )
    type: Optional[str] = Field(
        default=None,
        description="BEAM event type label.",
        sa_column=Column("type", String, nullable=True),
    )
    score: Optional[float] = Field(
        default=None,
        description="Score emitted for this parking decision.",
        sa_column=Column("score", Float, nullable=True),
    )
    pricingmodel: Optional[str] = Field(
        default=None,
        description="Pricing model used to calculate parking cost.",
        sa_column=Column("pricingModel", String, nullable=True),
    )
    parkingzoneid: Optional[str] = Field(
        default=None,
        description="Parking zone identifier.",
        sa_column=Column("parkingZoneId", String, nullable=True),
    )
    emissions: Optional[str] = Field(
        default=None,
        description="Serialized emissions payload if present.",
        sa_column=Column("emissions", String, nullable=True),
    )


class BeamEventsModeChoice(SQLModel, table=True):
    __tablename__ = "BeamEventsModeChoice"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    modechoiceeventid: Optional[int] = Field(
        default=None,
        description="Synthetic event identifier assigned during events split.",
        sa_column=Column(
            "ModeChoiceEventId",
            BigInteger,
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    tourindex: Optional[int] = Field(
        default=None,
        description="Tour index in the executed plan.",
        sa_column=Column("tourIndex", Integer, nullable=True),
    )
    currentactivity: Optional[str] = Field(
        default=None,
        description="Current activity before mode choice.",
        sa_column=Column("currentActivity", String, nullable=True),
    )
    tripid: Optional[int] = Field(
        default=None,
        description="Trip identifier carried through BEAM eventing.",
        sa_column=Column(
            "tripId",
            BigInteger,
            ForeignKey("tripsAsimOut.trip_id"),
            nullable=True,
            index=True,
        ),
    )
    time: Optional[float] = Field(
        default=None,
        description="Event time in simulation seconds.",
        sa_column=Column("time", Float, nullable=True, index=True),
    )
    personalvehicleavailable: Optional[bool] = Field(
        default=None,
        description="Whether a personal vehicle was available to the traveler.",
        sa_column=Column("personalVehicleAvailable", Boolean, nullable=True),
    )
    currenttourmode: Optional[str] = Field(
        default=None,
        description="Current tour mode when the choice was made.",
        sa_column=Column("currentTourMode", String, nullable=True),
    )
    nextactivity: Optional[str] = Field(
        default=None,
        description="Next activity after this leg.",
        sa_column=Column("nextActivity", String, nullable=True),
    )
    location: Optional[int] = Field(
        default=None,
        description="Location identifier reported for the mode choice event.",
        sa_column=Column("location", Integer, nullable=True, index=True),
    )
    person: Optional[int] = Field(
        default=None,
        description="Optional person identifier when it maps to PersonsBeamIn.",
        sa_column=Column(
            "person",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    availablealternatives: Optional[str] = Field(
        default=None,
        description="Serialized available alternative set.",
        sa_column=Column("availableAlternatives", String, nullable=True),
    )
    legmodes: Optional[str] = Field(
        default=None,
        description="Serialized leg mode sequence.",
        sa_column=Column("legModes", String, nullable=True),
    )
    length: Optional[float] = Field(
        default=None,
        description="Trip length associated with the choice.",
        sa_column=Column("length", Float, nullable=True),
    )
    mode: Optional[str] = Field(
        default=None,
        description="Chosen mode.",
        sa_column=Column("mode", String, nullable=True),
    )
    type: Optional[str] = Field(
        default=None,
        description="BEAM event type label.",
        sa_column=Column("type", String, nullable=True),
    )
    legvehicleids: Optional[str] = Field(
        default=None,
        description=(
            "Serialized BEAM leg vehicle references. Entries may contain "
            "stringified household vehicle ids or other fleet/transit vehicle ids."
        ),
        sa_column=Column("legVehicleIds", String, nullable=True),
    )


class BeamEventsParkingEvent(SQLModel, table=True):
    __tablename__ = "BeamEventsParkingEvent"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    parkingeventeventid: Optional[int] = Field(
        default=None,
        description="Synthetic event identifier assigned during events split.",
        sa_column=Column(
            "ParkingEventEventId",
            BigInteger,
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    driver: Optional[int] = Field(
        default=None,
        description="Optional driver person identifier when it maps to PersonsBeamIn.",
        sa_column=Column(
            "driver",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    chargingpointtype: Optional[str] = Field(
        default=None,
        description="Charging point type used at the parking location.",
        sa_column=Column("chargingPointType", String, nullable=True),
    )
    locationy: Optional[float] = Field(
        default=None,
        description="Y coordinate of the parking location.",
        sa_column=Column("locationY", Float, nullable=True),
    )
    vehicle: Optional[str] = Field(
        default=None,
        description=(
            "Raw BEAM vehicle reference. This may be a stringified household "
            "vehicle id or a non-household fleet/transit identifier."
        ),
        sa_column=Column("vehicle", String, nullable=True, index=True),
    )
    vehicle_id_int: Optional[int] = Field(
        default=None,
        description=(
            "Optional normalized integer vehicle id parsed from the raw BEAM "
            "vehicle reference when the row refers to a household vehicle."
        ),
        sa_column=Column("vehicle_id_int", BigInteger, nullable=True, index=True),
    )
    time: Optional[float] = Field(
        default=None,
        description="Event time in simulation seconds.",
        sa_column=Column("time", Float, nullable=True, index=True),
    )
    parkingtype: Optional[str] = Field(
        default=None,
        description="Parking type classification.",
        sa_column=Column("parkingType", String, nullable=True),
    )
    cost: Optional[float] = Field(
        default=None,
        description="Parking cost charged for the event.",
        sa_column=Column("cost", Float, nullable=True),
    )
    parkingtaz: Optional[str] = Field(
        default=None,
        description="Parking TAZ identifier emitted by BEAM.",
        sa_column=Column("parkingTaz", String, nullable=True),
    )
    links: Optional[str] = Field(
        default=None,
        description="Serialized route links associated with the event.",
        sa_column=Column("links", String, nullable=True),
    )
    type: Optional[str] = Field(
        default=None,
        description="BEAM event type label.",
        sa_column=Column("type", String, nullable=True),
    )
    locationx: Optional[float] = Field(
        default=None,
        description="X coordinate of the parking location.",
        sa_column=Column("locationX", Float, nullable=True),
    )
    pricingmodel: Optional[str] = Field(
        default=None,
        description="Pricing model used to calculate parking cost.",
        sa_column=Column("pricingModel", String, nullable=True),
    )
    parkingzoneid: Optional[str] = Field(
        default=None,
        description="Parking zone identifier.",
        sa_column=Column("parkingZoneId", String, nullable=True),
    )


class BeamEventsPathTraversal(SQLModel, table=True):
    __tablename__ = "BeamEventsPathTraversal"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    pathtraversaleventid: Optional[int] = Field(
        default=None,
        description="Synthetic event identifier assigned during events split.",
        sa_column=Column(
            "PathTraversalEventId",
            BigInteger,
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    capacity: Optional[int] = Field(
        default=None,
        description="Vehicle capacity reported in the event.",
        sa_column=Column("capacity", Integer, nullable=True),
    )
    weight: Optional[str] = Field(
        default=None,
        description="Vehicle weight class payload from BEAM.",
        sa_column=Column("weight", String, nullable=True),
    )
    driver: Optional[int] = Field(
        default=None,
        description="Optional driver person identifier when it maps to PersonsBeamIn.",
        sa_column=Column(
            "driver",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    fromstopindex: Optional[str] = Field(
        default=None,
        description="From-stop index for transit traversals.",
        sa_column=Column("fromStopIndex", String, nullable=True),
    )
    numpassengers: Optional[int] = Field(
        default=None,
        description="Passenger count for the traversal.",
        sa_column=Column("numPassengers", Integer, nullable=True),
    )
    starty: Optional[float] = Field(
        default=None,
        description="Start Y coordinate.",
        sa_column=Column("startY", Float, nullable=True),
    )
    vehicle: Optional[str] = Field(
        default=None,
        description=(
            "Raw BEAM vehicle reference. This may be a stringified household "
            "vehicle id or a non-household fleet/transit identifier."
        ),
        sa_column=Column("vehicle", String, nullable=True, index=True),
    )
    vehicle_id_int: Optional[int] = Field(
        default=None,
        description=(
            "Optional normalized integer vehicle id parsed from the raw BEAM "
            "vehicle reference when the traversal refers to a household vehicle."
        ),
        sa_column=Column("vehicle_id_int", BigInteger, nullable=True, index=True),
    )
    endx: Optional[float] = Field(
        default=None,
        description="End X coordinate.",
        sa_column=Column("endX", Float, nullable=True),
    )
    secondaryfuellevel: Optional[float] = Field(
        default=None,
        description="Secondary fuel level after traversal.",
        sa_column=Column("secondaryFuelLevel", Float, nullable=True),
    )
    time: Optional[float] = Field(
        default=None,
        description="Event time in simulation seconds.",
        sa_column=Column("time", Float, nullable=True, index=True),
    )
    primaryfueltype: Optional[str] = Field(
        default=None,
        description="Primary fuel type for the vehicle.",
        sa_column=Column("primaryFuelType", String, nullable=True),
    )
    seatingcapacity: Optional[int] = Field(
        default=None,
        description="Seating capacity for the vehicle.",
        sa_column=Column("seatingCapacity", Integer, nullable=True),
    )
    secondaryfuel: Optional[float] = Field(
        default=None,
        description="Secondary fuel consumed on the traversal.",
        sa_column=Column("secondaryFuel", Float, nullable=True),
    )
    departuretime: Optional[int] = Field(
        default=None,
        description="Departure time in simulation seconds.",
        sa_column=Column("departureTime", Integer, nullable=True),
    )
    secondaryfueltype: Optional[str] = Field(
        default=None,
        description="Secondary fuel type for the vehicle.",
        sa_column=Column("secondaryFuelType", String, nullable=True),
    )
    arrivaltime: Optional[int] = Field(
        default=None,
        description="Arrival time in simulation seconds.",
        sa_column=Column("arrivalTime", Integer, nullable=True),
    )
    tollpaid: Optional[float] = Field(
        default=None,
        description="Toll paid during traversal.",
        sa_column=Column("tollPaid", Float, nullable=True),
    )
    payloads: Optional[str] = Field(
        default=None,
        description="Serialized payload description for freight events.",
        sa_column=Column("payloads", String, nullable=True),
    )
    length: Optional[float] = Field(
        default=None,
        description="Traversal length in meters.",
        sa_column=Column("length", Float, nullable=True),
    )
    tostopindex: Optional[str] = Field(
        default=None,
        description="To-stop index for transit traversals.",
        sa_column=Column("toStopIndex", String, nullable=True),
    )
    endy: Optional[float] = Field(
        default=None,
        description="End Y coordinate.",
        sa_column=Column("endY", Float, nullable=True),
    )
    mode: Optional[str] = Field(
        default=None,
        description="Traversal mode.",
        sa_column=Column("mode", String, nullable=True),
    )
    type: Optional[str] = Field(
        default=None,
        description="BEAM event type label.",
        sa_column=Column("type", String, nullable=True),
    )
    startx: Optional[float] = Field(
        default=None,
        description="Start X coordinate.",
        sa_column=Column("startX", Float, nullable=True),
    )
    primaryfuel: Optional[float] = Field(
        default=None,
        description="Primary fuel consumed on the traversal.",
        sa_column=Column("primaryFuel", Float, nullable=True),
    )
    riders: Optional[str] = Field(
        default=None,
        description="Serialized rider identifiers.",
        sa_column=Column("riders", String, nullable=True),
    )
    currenttripmode: Optional[str] = Field(
        default=None,
        description="Current trip mode at traversal time.",
        sa_column=Column("currentTripMode", String, nullable=True),
    )
    vehicletype: Optional[str] = Field(
        default=None,
        description="BEAM-reported vehicle type label for the traversal.",
        sa_column=Column("vehicleType", String, nullable=True),
    )
    emissions: Optional[str] = Field(
        default=None,
        description="Serialized emissions payload if present.",
        sa_column=Column("emissions", String, nullable=True),
    )
    primaryfuellevel: Optional[float] = Field(
        default=None,
        description="Primary fuel level after traversal.",
        sa_column=Column("primaryFuelLevel", Float, nullable=True),
    )


class BeamEventsPersonCost(SQLModel, table=True):
    __tablename__ = "BeamEventsPersonCost"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    personcosteventid: Optional[int] = Field(
        default=None,
        description="Synthetic event identifier assigned during events split.",
        sa_column=Column(
            "PersonCostEventId",
            BigInteger,
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    tollcost: Optional[str] = Field(
        default=None,
        description="Toll cost payload from BEAM.",
        sa_column=Column("tollCost", String, nullable=True),
    )
    time: Optional[float] = Field(
        default=None,
        description="Event time in simulation seconds.",
        sa_column=Column("time", Float, nullable=True, index=True),
    )
    netcost: Optional[str] = Field(
        default=None,
        description="Net cost payload from BEAM.",
        sa_column=Column("netCost", String, nullable=True),
    )
    person: Optional[int] = Field(
        default=None,
        description="Optional person identifier when it maps to PersonsBeamIn.",
        sa_column=Column(
            "person",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    mode: Optional[str] = Field(
        default=None,
        description="Mode associated with the cost event.",
        sa_column=Column("mode", String, nullable=True),
    )
    type: Optional[str] = Field(
        default=None,
        description="BEAM event type label.",
        sa_column=Column("type", String, nullable=True),
    )
    incentive: Optional[str] = Field(
        default=None,
        description="Incentive amount or identifier.",
        sa_column=Column("incentive", String, nullable=True),
    )


class BeamEventsPersonEntersVehicle(SQLModel, table=True):
    __tablename__ = "BeamEventsPersonEntersVehicle"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    personentersvehicleeventid: Optional[int] = Field(
        default=None,
        description="Synthetic event identifier assigned during events split.",
        sa_column=Column(
            "PersonEntersVehicleEventId",
            BigInteger,
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    vehicle: Optional[str] = Field(
        default=None,
        description=(
            "Raw BEAM vehicle reference. This may be a stringified household "
            "vehicle id or a non-household fleet/transit identifier."
        ),
        sa_column=Column("vehicle", String, nullable=True, index=True),
    )
    vehicle_id_int: Optional[int] = Field(
        default=None,
        description=(
            "Optional normalized integer vehicle id parsed from the raw BEAM "
            "vehicle reference when the row refers to a household vehicle."
        ),
        sa_column=Column("vehicle_id_int", BigInteger, nullable=True, index=True),
    )
    time: Optional[float] = Field(
        default=None,
        description="Event time in simulation seconds.",
        sa_column=Column("time", Float, nullable=True, index=True),
    )
    person: Optional[int] = Field(
        default=None,
        description="Optional person identifier when it maps to PersonsBeamIn.",
        sa_column=Column(
            "person",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    type: Optional[str] = Field(
        default=None,
        description="BEAM event type label.",
        sa_column=Column("type", String, nullable=True),
    )


class BeamEventsPersonLeavesVehicle(SQLModel, table=True):
    __tablename__ = "BeamEventsPersonLeavesVehicle"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    personleavesvehicleeventid: Optional[int] = Field(
        default=None,
        description="Synthetic event identifier assigned during events split.",
        sa_column=Column(
            "PersonLeavesVehicleEventId",
            BigInteger,
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    vehicle: Optional[str] = Field(
        default=None,
        description=(
            "Raw BEAM vehicle reference. This may be a stringified household "
            "vehicle id or a non-household fleet/transit identifier."
        ),
        sa_column=Column("vehicle", String, nullable=True, index=True),
    )
    vehicle_id_int: Optional[int] = Field(
        default=None,
        description=(
            "Optional normalized integer vehicle id parsed from the raw BEAM "
            "vehicle reference when the row refers to a household vehicle."
        ),
        sa_column=Column("vehicle_id_int", BigInteger, nullable=True, index=True),
    )
    time: Optional[float] = Field(
        default=None,
        description="Event time in simulation seconds.",
        sa_column=Column("time", Float, nullable=True, index=True),
    )
    person: Optional[int] = Field(
        default=None,
        description="Optional person identifier when it maps to PersonsBeamIn.",
        sa_column=Column(
            "person",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    type: Optional[str] = Field(
        default=None,
        description="BEAM event type label.",
        sa_column=Column("type", String, nullable=True),
    )


class BeamEventsReplanning(SQLModel, table=True):
    __tablename__ = "BeamEventsReplanning"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    replanningeventid: Optional[int] = Field(
        default=None,
        description="Synthetic event identifier assigned during events split.",
        sa_column=Column(
            "ReplanningEventId",
            BigInteger,
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    starty: Optional[float] = Field(
        default=None,
        description="Origin Y coordinate for replanning.",
        sa_column=Column("startY", Float, nullable=True),
    )
    endx: Optional[float] = Field(
        default=None,
        description="Destination X coordinate for replanning.",
        sa_column=Column("endX", Float, nullable=True),
    )
    time: Optional[float] = Field(
        default=None,
        description="Event time in simulation seconds.",
        sa_column=Column("time", Float, nullable=True, index=True),
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for triggering replanning.",
        sa_column=Column("reason", String, nullable=True),
    )
    person: Optional[int] = Field(
        default=None,
        description="Optional person identifier when it maps to PersonsBeamIn.",
        sa_column=Column(
            "person",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    endy: Optional[float] = Field(
        default=None,
        description="Destination Y coordinate for replanning.",
        sa_column=Column("endY", Float, nullable=True),
    )
    type: Optional[str] = Field(
        default=None,
        description="BEAM event type label.",
        sa_column=Column("type", String, nullable=True),
    )
    startx: Optional[float] = Field(
        default=None,
        description="Origin X coordinate for replanning.",
        sa_column=Column("startX", Float, nullable=True),
    )


class BeamEventsReserveRideHail(SQLModel, table=True):
    __tablename__ = "BeamEventsReserveRideHail"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    reserveridehaileventid: Optional[int] = Field(
        default=None,
        description="Synthetic event identifier assigned during events split.",
        sa_column=Column(
            "ReserveRideHailEventId",
            BigInteger,
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    starty: Optional[float] = Field(
        default=None,
        description="Origin Y coordinate for the reservation.",
        sa_column=Column("startY", Float, nullable=True),
    )
    endx: Optional[float] = Field(
        default=None,
        description="Destination X coordinate for the reservation.",
        sa_column=Column("endX", Float, nullable=True),
    )
    time: Optional[float] = Field(
        default=None,
        description="Event time in simulation seconds.",
        sa_column=Column("time", Float, nullable=True, index=True),
    )
    departtime: Optional[int] = Field(
        default=None,
        description="Requested departure time.",
        sa_column=Column("departTime", Integer, nullable=True),
    )
    person: Optional[int] = Field(
        default=None,
        description="Optional person identifier when it maps to PersonsBeamIn.",
        sa_column=Column(
            "person",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    endy: Optional[float] = Field(
        default=None,
        description="Destination Y coordinate for the reservation.",
        sa_column=Column("endY", Float, nullable=True),
    )
    type: Optional[str] = Field(
        default=None,
        description="BEAM event type label.",
        sa_column=Column("type", String, nullable=True),
    )
    startx: Optional[float] = Field(
        default=None,
        description="Origin X coordinate for the reservation.",
        sa_column=Column("startX", Float, nullable=True),
    )
    requirewheelchair: Optional[str] = Field(
        default=None,
        description="Wheelchair accessibility requirement indicator.",
        sa_column=Column("requireWheelchair", String, nullable=True),
    )


class BeamEventsTeleportationEvent(SQLModel, table=True):
    __tablename__ = "BeamEventsTeleportationEvent"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    teleportationeventeventid: Optional[int] = Field(
        default=None,
        description="Synthetic event identifier assigned during events split.",
        sa_column=Column(
            "TeleportationEventEventId",
            BigInteger,
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    starty: Optional[float] = Field(
        default=None,
        description="Origin Y coordinate for teleportation.",
        sa_column=Column("startY", Float, nullable=True),
    )
    endx: Optional[float] = Field(
        default=None,
        description="Destination X coordinate for teleportation.",
        sa_column=Column("endX", Float, nullable=True),
    )
    time: Optional[float] = Field(
        default=None,
        description="Event time in simulation seconds.",
        sa_column=Column("time", Float, nullable=True, index=True),
    )
    person: Optional[int] = Field(
        default=None,
        description="Optional person identifier when it maps to PersonsBeamIn.",
        sa_column=Column(
            "person",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    departuretime: Optional[int] = Field(
        default=None,
        description="Departure time in simulation seconds.",
        sa_column=Column("departureTime", Integer, nullable=True),
    )
    arrivaltime: Optional[int] = Field(
        default=None,
        description="Arrival time in simulation seconds.",
        sa_column=Column("arrivalTime", Integer, nullable=True),
    )
    endy: Optional[float] = Field(
        default=None,
        description="Destination Y coordinate for teleportation.",
        sa_column=Column("endY", Float, nullable=True),
    )
    type: Optional[str] = Field(
        default=None,
        description="BEAM event type label.",
        sa_column=Column("type", String, nullable=True),
    )
    startx: Optional[float] = Field(
        default=None,
        description="Origin X coordinate for teleportation.",
        sa_column=Column("startX", Float, nullable=True),
    )
    currenttripmode: Optional[str] = Field(
        default=None,
        description="Current trip mode associated with teleportation.",
        sa_column=Column("currentTripMode", String, nullable=True),
    )


class BeamEventsActEnd(SQLModel, table=True):
    __tablename__ = "BeamEventsActEnd"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    actendeventid: Optional[int] = Field(
        default=None,
        description="Synthetic event identifier assigned during events split.",
        sa_column=Column(
            "actendEventId",
            BigInteger,
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    acttype: Optional[str] = Field(
        default=None,
        description="Activity type that ended.",
        sa_column=Column("actType", String, nullable=True),
    )
    time: Optional[float] = Field(
        default=None,
        description="Event time in simulation seconds.",
        sa_column=Column("time", Float, nullable=True, index=True),
    )
    link: Optional[int] = Field(
        default=None,
        description="Network link where the activity ended.",
        sa_column=Column(
            "link",
            Integer,
            ForeignKey("BeamNetworkFinal.linkId"),
            nullable=True,
            index=True,
        ),
    )
    person: Optional[int] = Field(
        default=None,
        description="Optional person identifier when it maps to PersonsBeamIn.",
        sa_column=Column(
            "person",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    type: Optional[str] = Field(
        default=None,
        description="BEAM event type label.",
        sa_column=Column("type", String, nullable=True),
    )


class BeamEventsActStart(SQLModel, table=True):
    __tablename__ = "BeamEventsActStart"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    actstarteventid: Optional[int] = Field(
        default=None,
        description="Synthetic event identifier assigned during events split.",
        sa_column=Column(
            "actstartEventId",
            BigInteger,
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    acttype: Optional[str] = Field(
        default=None,
        description="Activity type that started.",
        sa_column=Column("actType", String, nullable=True),
    )
    time: Optional[float] = Field(
        default=None,
        description="Event time in simulation seconds.",
        sa_column=Column("time", Float, nullable=True, index=True),
    )
    link: Optional[int] = Field(
        default=None,
        description="Network link where the activity started.",
        sa_column=Column(
            "link",
            Integer,
            ForeignKey("BeamNetworkFinal.linkId"),
            nullable=True,
            index=True,
        ),
    )
    person: Optional[int] = Field(
        default=None,
        description="Optional person identifier when it maps to PersonsBeamIn.",
        sa_column=Column(
            "person",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    type: Optional[str] = Field(
        default=None,
        description="BEAM event type label.",
        sa_column=Column("type", String, nullable=True),
    )


class BeamEventsArrival(SQLModel, table=True):
    __tablename__ = "BeamEventsArrival"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    arrivaleventid: Optional[int] = Field(
        default=None,
        description="Synthetic event identifier assigned during events split.",
        sa_column=Column(
            "arrivalEventId",
            BigInteger,
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    time: Optional[float] = Field(
        default=None,
        description="Event time in simulation seconds.",
        sa_column=Column("time", Float, nullable=True, index=True),
    )
    link: Optional[int] = Field(
        default=None,
        description="Network link where the arrival occurred.",
        sa_column=Column(
            "link",
            Integer,
            ForeignKey("BeamNetworkFinal.linkId"),
            nullable=True,
            index=True,
        ),
    )
    person: Optional[int] = Field(
        default=None,
        description="Optional person identifier when it maps to PersonsBeamIn.",
        sa_column=Column(
            "person",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    type: Optional[str] = Field(
        default=None,
        description="BEAM event type label.",
        sa_column=Column("type", String, nullable=True),
    )
    legmode: Optional[str] = Field(
        default=None,
        description="Leg mode associated with the arrival.",
        sa_column=Column("legMode", String, nullable=True),
    )


class BeamEventsDeparture(SQLModel, table=True):
    __tablename__ = "BeamEventsDeparture"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    departureeventid: Optional[int] = Field(
        default=None,
        description="Synthetic event identifier assigned during events split.",
        sa_column=Column(
            "departureEventId",
            BigInteger,
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    time: Optional[float] = Field(
        default=None,
        description="Event time in simulation seconds.",
        sa_column=Column("time", Float, nullable=True, index=True),
    )
    link: Optional[int] = Field(
        default=None,
        description="Network link where the departure occurred.",
        sa_column=Column(
            "link",
            Integer,
            ForeignKey("BeamNetworkFinal.linkId"),
            nullable=True,
            index=True,
        ),
    )
    payloadids: Optional[str] = Field(
        default=None,
        description="Serialized payload identifiers for freight departures.",
        sa_column=Column("PayloadIds", String, nullable=True),
    )
    person: Optional[int] = Field(
        default=None,
        description="Optional person identifier when it maps to PersonsBeamIn.",
        sa_column=Column(
            "person",
            BigInteger,
            ForeignKey("PersonsBeamIn.person_id"),
            nullable=True,
            index=True,
        ),
    )
    trip_id: Optional[int] = Field(
        default=None,
        description="Trip identifier associated with the departure.",
        sa_column=Column(
            "trip_id",
            BigInteger,
            ForeignKey("tripsAsimOut.trip_id"),
            nullable=True,
            index=True,
        ),
    )
    type: Optional[str] = Field(
        default=None,
        description="BEAM event type label.",
        sa_column=Column("type", String, nullable=True),
    )
    payloadweightinkg: Optional[str] = Field(
        default=None,
        description="Serialized payload weight(s) in kilograms.",
        sa_column=Column("PayloadWeightInKg", String, nullable=True),
    )
    legmode: Optional[str] = Field(
        default=None,
        description="Leg mode associated with the departure.",
        sa_column=Column("legMode", String, nullable=True),
    )


class BeamPathTraversalLinks(SQLModel, table=True):
    __tablename__ = "BeamPathTraversalLinks"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    pathtraversaleventid: Optional[int] = Field(
        default=None,
        description="Reference to the parent PathTraversal split event row.",
        sa_column=Column(
            "PathTraversalEventId",
            BigInteger,
            ForeignKey("BeamEventsPathTraversal.PathTraversalEventId"),
            primary_key=True,
            nullable=False,
            index=True,
        ),
    )
    link_index: Optional[int] = Field(
        default=None,
        description="Ordinal position of the link within the traversal sequence.",
        sa_column=Column("link_index", BigInteger, primary_key=True, nullable=False),
    )
    linkid: Optional[int] = Field(
        default=None,
        description="Network link identifier traversed for this segment.",
        sa_column=Column(
            "linkId",
            BigInteger,
            ForeignKey("BeamNetworkFinal.linkId"),
            nullable=True,
            index=True,
        ),
    )
    traveltimeseconds: Optional[float] = Field(
        default=None,
        description="Travel time for this link segment in seconds.",
        sa_column=Column("travelTimeSeconds", Float, nullable=True),
    )
