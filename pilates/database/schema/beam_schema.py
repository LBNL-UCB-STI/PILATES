from __future__ import annotations

from typing import Optional

from sqlalchemy import BigInteger, Boolean, Column, Float, ForeignKey, String
from sqlmodel import Field, SQLModel


class PlansBeamIn(SQLModel, table=True):
    __tablename__ = "PlansBeamIn"
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    # Same as BeamPlansAsimOut plus BEAM-remembered plan fields.
    tour_id: Optional[int] = Field(
        default=None,
        description="Tour identifier in ActivitySim outputs.",
        sa_column=Column("tour_id", BigInteger, nullable=True, index=True),
    )
    trip_id: Optional[int] = Field(
        default=None,
        description="Trip identifier in ActivitySim outputs.",
        sa_column=Column("trip_id", BigInteger, nullable=True, index=True),
    )
    person_id: Optional[int] = Field(
        default=None,
        description="Person identifier associated with the trip/tour.",
        sa_column=Column(
            "person_id",
            BigInteger,
            ForeignKey("PersonsAsimOut.person_id"),
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
