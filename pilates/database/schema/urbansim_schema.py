from __future__ import annotations

from typing import Optional

from sqlalchemy import BigInteger, Column, Float, ForeignKey, Integer, String
from sqlmodel import Field, SQLModel


class ActivitysimPostprocessUsimHouseholdsUpdated(SQLModel, table=True):
    __tablename__ = 'ActivitysimPostprocessUsimHouseholdsUpdated'
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    household_id: Optional[int] = Field(
        default=None,
        description='UrbanSim household identifier updated by ActivitySim postprocessing.',
        sa_column=Column(
            'household_id',
            BigInteger,
            ForeignKey('HouseholdsAsimIn.household_id'),
            nullable=True,
            index=True,
        ),
    )
    tenure_mover: Optional[str] = Field(default=None, sa_column=Column('tenure_mover', String, nullable=True))
    block_id: Optional[str] = Field(
        default=None,
        description='Block identifier retained in the UrbanSim household table.',
        sa_column=Column(
            'block_id',
            String,
            ForeignKey('AtlasBlocks.block_id'),
            nullable=True,
            index=True,
        ),
    )
    workers: Optional[float] = Field(default=None, sa_column=Column('workers', Float, nullable=True))
    hh_size: Optional[str] = Field(default=None, sa_column=Column('hh_size', String, nullable=True))
    sf_detached: Optional[str] = Field(default=None, sa_column=Column('sf_detached', String, nullable=True))
    hh_seniors: Optional[str] = Field(default=None, sa_column=Column('hh_seniors', String, nullable=True))
    income: Optional[float] = Field(default=None, sa_column=Column('income', Float, nullable=True))
    seniors: Optional[float] = Field(default=None, sa_column=Column('seniors', Float, nullable=True))
    gt2: Optional[float] = Field(default=None, sa_column=Column('gt2', Float, nullable=True))
    hh_age_of_head: Optional[str] = Field(default=None, sa_column=Column('hh_age_of_head', String, nullable=True))
    hh_children: Optional[str] = Field(default=None, sa_column=Column('hh_children', String, nullable=True))
    hh_cars: Optional[str] = Field(default=None, sa_column=Column('hh_cars', String, nullable=True))
    hh_income: Optional[str] = Field(default=None, sa_column=Column('hh_income', String, nullable=True))
    hispanic_status_of_head: Optional[float] = Field(default=None, sa_column=Column('hispanic_status_of_head', Float, nullable=True))
    hh_race_of_head: Optional[str] = Field(default=None, sa_column=Column('hh_race_of_head', String, nullable=True))
    lcm_county_id: Optional[str] = Field(default=None, sa_column=Column('lcm_county_id', String, nullable=True))
    race_of_head: Optional[float] = Field(default=None, sa_column=Column('race_of_head', Float, nullable=True))
    recent_mover: Optional[str] = Field(default=None, sa_column=Column('recent_mover', String, nullable=True))
    serialno: Optional[int] = Field(default=None, sa_column=Column('serialno', BigInteger, nullable=True))
    tenure: Optional[str] = Field(default=None, sa_column=Column('tenure', String, nullable=True))
    cars: Optional[int] = Field(default=None, sa_column=Column('cars', BigInteger, nullable=True))
    hh_workers: Optional[str] = Field(default=None, sa_column=Column('hh_workers', String, nullable=True))
    gt55: Optional[float] = Field(default=None, sa_column=Column('gt55', Float, nullable=True))
    age_of_head: Optional[float] = Field(default=None, sa_column=Column('age_of_head', Float, nullable=True))
    hispanic_head: Optional[str] = Field(default=None, sa_column=Column('hispanic_head', String, nullable=True))
    hh_type: Optional[float] = Field(default=None, sa_column=Column('hh_type', Float, nullable=True))
    persons: Optional[int] = Field(default=None, sa_column=Column('persons', BigInteger, nullable=True))


class ActivitysimPostprocessUsimPersonsUpdated(SQLModel, table=True):
    __tablename__ = 'ActivitysimPostprocessUsimPersonsUpdated'
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    person_id: Optional[int] = Field(
        default=None,
        description='UrbanSim person identifier updated by ActivitySim postprocessing.',
        sa_column=Column(
            'person_id',
            BigInteger,
            ForeignKey('PersonsAsimOut.person_id'),
            nullable=True,
            index=True,
        ),
    )
    hispanic: Optional[float] = Field(default=None, sa_column=Column('hispanic', Float, nullable=True))
    person_sex: Optional[str] = Field(default=None, sa_column=Column('person_sex', String, nullable=True))
    age: Optional[float] = Field(default=None, sa_column=Column('age', Float, nullable=True))
    person_age: Optional[str] = Field(default=None, sa_column=Column('person_age', String, nullable=True))
    education_group: Optional[str] = Field(default=None, sa_column=Column('education_group', String, nullable=True))
    edu: Optional[float] = Field(default=None, sa_column=Column('edu', Float, nullable=True))
    workplace_taz: Optional[int] = Field(
        default=None,
        sa_column=Column(
            'workplace_taz',
            Integer,
            ForeignKey('LandUseAsimIn.TAZ'),
            nullable=True,
        ),
    )
    race: Optional[str] = Field(default=None, sa_column=Column('race', String, nullable=True))
    race_id: Optional[float] = Field(default=None, sa_column=Column('race_id', Float, nullable=True))
    earning: Optional[float] = Field(default=None, sa_column=Column('earning', Float, nullable=True))
    student: Optional[float] = Field(default=None, sa_column=Column('student', Float, nullable=True))
    age_group: Optional[str] = Field(default=None, sa_column=Column('age_group', String, nullable=True))
    p_hispanic: Optional[str] = Field(default=None, sa_column=Column('p_hispanic', String, nullable=True))
    work_block_id: Optional[str] = Field(
        default=None,
        description='Work block identifier retained in the UrbanSim person table.',
        sa_column=Column(
            'work_block_id',
            String,
            ForeignKey('AtlasBlocks.block_id'),
            nullable=True,
            index=True,
        ),
    )
    hispanic_1: Optional[float] = Field(default=None, sa_column=Column('hispanic.1', Float, nullable=True))
    school_zone_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            'school_zone_id',
            Integer,
            ForeignKey('LandUseAsimIn.TAZ'),
            nullable=True,
        ),
    )
    work_zone_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            'work_zone_id',
            Integer,
            ForeignKey('LandUseAsimIn.TAZ'),
            nullable=True,
        ),
    )
    household_id: Optional[float] = Field(
        default=None,
        description='Household identifier carried through the updated UrbanSim person table.',
        sa_column=Column('household_id', Float, nullable=True, index=True),
    )
    worker: Optional[float] = Field(default=None, sa_column=Column('worker', Float, nullable=True))
    school_block_id: Optional[str] = Field(
        default=None,
        description='School block identifier retained in the UrbanSim person table.',
        sa_column=Column(
            'school_block_id',
            String,
            ForeignKey('AtlasBlocks.block_id'),
            nullable=True,
            index=True,
        ),
    )
    school_id: Optional[str] = Field(default=None, sa_column=Column('school_id', String, nullable=True))
    sex: Optional[float] = Field(default=None, sa_column=Column('sex', Float, nullable=True))
    member_id: Optional[int] = Field(default=None, sa_column=Column('member_id', BigInteger, nullable=True))
    relate: Optional[float] = Field(default=None, sa_column=Column('relate', Float, nullable=True))
    work_at_home: Optional[float] = Field(default=None, sa_column=Column('work_at_home', Float, nullable=True))
    hours: Optional[float] = Field(default=None, sa_column=Column('hours', Float, nullable=True))
    mar: Optional[float] = Field(default=None, sa_column=Column('MAR', Float, nullable=True))
    school_taz: Optional[int] = Field(
        default=None,
        sa_column=Column(
            'school_taz',
            Integer,
            ForeignKey('LandUseAsimIn.TAZ'),
            nullable=True,
        ),
    )


class AtlasPostprocessUsimHouseholdsUpdated(
    ActivitysimPostprocessUsimHouseholdsUpdated, table=True
):
    __tablename__ = 'AtlasPostprocessUsimHouseholdsUpdated'
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    household_id: Optional[int] = Field(
        default=None,
        description='UrbanSim household identifier updated by ATLAS postprocessing.',
        sa_column=Column(
            'household_id',
            BigInteger,
            ForeignKey('AtlasHousehold.household_id'),
            nullable=True,
            index=True,
        ),
    )


class UrbansimPostprocessUsimBlocksTable(SQLModel, table=True):
    __tablename__ = 'UrbansimPostprocessUsimBlocksTable'
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    block_id: Optional[str] = Field(
        default=None,
        sa_column=Column('block_id', String, nullable=True, index=True),
    )
    state_id: Optional[str] = Field(default=None, sa_column=Column('state_id', String, nullable=True))
    block_group_id: Optional[str] = Field(default=None, sa_column=Column('block_group_id', String, nullable=True))
    zone_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'zone_id',
            String,
            ForeignKey('UrbansimPostprocessUsimTazZoneGeomsTable.zone_id'),
            nullable=True,
            index=True,
        ),
    )
    residential_unit_capacity: Optional[float] = Field(default=None, sa_column=Column('residential_unit_capacity', Float, nullable=True))
    tract_id: Optional[str] = Field(default=None, sa_column=Column('tract_id', String, nullable=True))
    sum_acres: Optional[float] = Field(default=None, sa_column=Column('sum_acres', Float, nullable=True))
    square_meters_land: Optional[int] = Field(default=None, sa_column=Column('square_meters_land', BigInteger, nullable=True))
    employment_capacity: Optional[float] = Field(default=None, sa_column=Column('employment_capacity', Float, nullable=True))
    mpo_id: Optional[str] = Field(default=None, sa_column=Column('MPO_ID', String, nullable=True))
    x: Optional[float] = Field(default=None, sa_column=Column('x', Float, nullable=True))
    taz_zone_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'taz_zone_id',
            String,
            ForeignKey('UrbansimPostprocessUsimTazZoneGeomsTable.zone_id'),
            nullable=True,
            index=True,
        ),
    )
    y: Optional[float] = Field(default=None, sa_column=Column('y', Float, nullable=True))
    county_id: Optional[str] = Field(default=None, sa_column=Column('county_id', String, nullable=True))
    cousub: Optional[str] = Field(default=None, sa_column=Column('cousub', String, nullable=True))
    node_id: Optional[int] = Field(default=None, sa_column=Column('node_id', BigInteger, nullable=True))


class UrbansimPostprocessUsimTazZoneGeomsTable(SQLModel, table=True):
    __tablename__ = 'UrbansimPostprocessUsimTazZoneGeomsTable'
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    zone_id: Optional[str] = Field(
        default=None,
        sa_column=Column('zone_id', String, nullable=True, index=True),
    )
    objectid: Optional[int] = Field(default=None, sa_column=Column('objectid', BigInteger, nullable=True))
    district: Optional[int] = Field(default=None, sa_column=Column('district', BigInteger, nullable=True))
    county: Optional[str] = Field(default=None, sa_column=Column('county', String, nullable=True))
    gacres: Optional[float] = Field(default=None, sa_column=Column('gacres', Float, nullable=True))
    shape_are: Optional[float] = Field(default=None, sa_column=Column('Shape__Are', Float, nullable=True))
    shape_len: Optional[float] = Field(default=None, sa_column=Column('Shape__Len', Float, nullable=True))
    geometry: Optional[str] = Field(default=None, sa_column=Column('geometry', String, nullable=True))


class UrbansimPostprocessUsimHouseholdsTable(SQLModel, table=True):
    __tablename__ = 'UrbansimPostprocessUsimHouseholdsTable'
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    household_id: Optional[float] = Field(
        default=None,
        sa_column=Column('household_id', Float, nullable=True, index=True),
    )
    hh_cars: Optional[str] = Field(default=None, sa_column=Column('hh_cars', String, nullable=True))
    cars: Optional[float] = Field(default=None, sa_column=Column('cars', Float, nullable=True))
    serialno: Optional[str] = Field(default=None, sa_column=Column('serialno', String, nullable=True))
    hh_workers: Optional[str] = Field(default=None, sa_column=Column('hh_workers', String, nullable=True))
    seniors: Optional[float] = Field(default=None, sa_column=Column('seniors', Float, nullable=True))
    workers: Optional[float] = Field(default=None, sa_column=Column('workers', Float, nullable=True))
    lcm_county_id: Optional[str] = Field(default=None, sa_column=Column('lcm_county_id', String, nullable=True))
    hh_age_of_head: Optional[str] = Field(default=None, sa_column=Column('hh_age_of_head', String, nullable=True))
    sf_detached: Optional[str] = Field(default=None, sa_column=Column('sf_detached', String, nullable=True))
    block_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'block_id',
            String,
            ForeignKey('UrbansimPostprocessUsimBlocksTable.block_id'),
            nullable=True,
            index=True,
        ),
    )
    gt2: Optional[float] = Field(default=None, sa_column=Column('gt2', Float, nullable=True))
    race_of_head: Optional[float] = Field(default=None, sa_column=Column('race_of_head', Float, nullable=True))
    age_of_head: Optional[float] = Field(default=None, sa_column=Column('age_of_head', Float, nullable=True))
    hh_children: Optional[str] = Field(default=None, sa_column=Column('hh_children', String, nullable=True))
    tenure: Optional[str] = Field(default=None, sa_column=Column('tenure', String, nullable=True))
    hh_size: Optional[str] = Field(default=None, sa_column=Column('hh_size', String, nullable=True))
    recent_mover: Optional[str] = Field(default=None, sa_column=Column('recent_mover', String, nullable=True))
    tenure_mover: Optional[str] = Field(default=None, sa_column=Column('tenure_mover', String, nullable=True))
    hispanic_head: Optional[str] = Field(default=None, sa_column=Column('hispanic_head', String, nullable=True))
    hh_income: Optional[str] = Field(default=None, sa_column=Column('hh_income', String, nullable=True))
    hh_seniors: Optional[str] = Field(default=None, sa_column=Column('hh_seniors', String, nullable=True))
    gt55: Optional[float] = Field(default=None, sa_column=Column('gt55', Float, nullable=True))
    income: Optional[float] = Field(default=None, sa_column=Column('income', Float, nullable=True))
    hh_race_of_head: Optional[str] = Field(default=None, sa_column=Column('hh_race_of_head', String, nullable=True))
    hispanic_status_of_head: Optional[float] = Field(default=None, sa_column=Column('hispanic_status_of_head', Float, nullable=True))
    hh_type: Optional[int] = Field(default=None, sa_column=Column('hh_type', BigInteger, nullable=True))
    persons: Optional[int] = Field(default=None, sa_column=Column('persons', BigInteger, nullable=True))


class UrbansimPostprocessUsimPersonsTable(SQLModel, table=True):
    __tablename__ = 'UrbansimPostprocessUsimPersonsTable'
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    person_id: Optional[int] = Field(
        default=None,
        sa_column=Column('person_id', BigInteger, nullable=True, index=True),
    )
    relate: Optional[int] = Field(default=None, sa_column=Column('relate', BigInteger, nullable=True))
    p_hispanic: Optional[str] = Field(default=None, sa_column=Column('p_hispanic', String, nullable=True))
    member_id: Optional[int] = Field(default=None, sa_column=Column('member_id', BigInteger, nullable=True))
    work_block_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'work_block_id',
            String,
            ForeignKey('UrbansimPostprocessUsimBlocksTable.block_id'),
            nullable=True,
            index=True,
        ),
    )
    school_zone_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'school_zone_id',
            String,
            ForeignKey('UrbansimPostprocessUsimTazZoneGeomsTable.zone_id'),
            nullable=True,
            index=True,
        ),
    )
    school_id: Optional[str] = Field(default=None, sa_column=Column('school_id', String, nullable=True))
    student: Optional[float] = Field(default=None, sa_column=Column('student', Float, nullable=True))
    work_zone_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'work_zone_id',
            String,
            ForeignKey('UrbansimPostprocessUsimTazZoneGeomsTable.zone_id'),
            nullable=True,
            index=True,
        ),
    )
    age_group: Optional[str] = Field(default=None, sa_column=Column('age_group', String, nullable=True))
    hispanic_1: Optional[float] = Field(default=None, sa_column=Column('hispanic.1', Float, nullable=True))
    education_group: Optional[str] = Field(default=None, sa_column=Column('education_group', String, nullable=True))
    workplace_taz: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'workplace_taz',
            String,
            ForeignKey('UrbansimPostprocessUsimTazZoneGeomsTable.zone_id'),
            nullable=True,
            index=True,
        ),
    )
    sex: Optional[float] = Field(default=None, sa_column=Column('sex', Float, nullable=True))
    person_age: Optional[str] = Field(default=None, sa_column=Column('person_age', String, nullable=True))
    school_block_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'school_block_id',
            String,
            ForeignKey('UrbansimPostprocessUsimBlocksTable.block_id'),
            nullable=True,
            index=True,
        ),
    )
    worker: Optional[float] = Field(default=None, sa_column=Column('worker', Float, nullable=True))
    earning: Optional[float] = Field(default=None, sa_column=Column('earning', Float, nullable=True))
    hours: Optional[float] = Field(default=None, sa_column=Column('hours', Float, nullable=True))
    edu: Optional[float] = Field(default=None, sa_column=Column('edu', Float, nullable=True))
    race_id: Optional[float] = Field(default=None, sa_column=Column('race_id', Float, nullable=True))
    hispanic: Optional[float] = Field(default=None, sa_column=Column('hispanic', Float, nullable=True))
    person_sex: Optional[str] = Field(default=None, sa_column=Column('person_sex', String, nullable=True))
    school_taz: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'school_taz',
            String,
            ForeignKey('UrbansimPostprocessUsimTazZoneGeomsTable.zone_id'),
            nullable=True,
            index=True,
        ),
    )
    age: Optional[float] = Field(default=None, sa_column=Column('age', Float, nullable=True))
    work_at_home: Optional[float] = Field(default=None, sa_column=Column('work_at_home', Float, nullable=True))
    race: Optional[str] = Field(default=None, sa_column=Column('race', String, nullable=True))
    household_id: Optional[float] = Field(
        default=None,
        sa_column=Column(
            'household_id',
            Float,
            ForeignKey('UrbansimPostprocessUsimHouseholdsTable.household_id'),
            nullable=True,
            index=True,
        ),
    )
    mar: Optional[int] = Field(default=None, sa_column=Column('MAR', BigInteger, nullable=True))


class UrbansimPostprocessUsimJobsTable(SQLModel, table=True):
    __tablename__ = 'UrbansimPostprocessUsimJobsTable'
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    job_id: Optional[int] = Field(
        default=None,
        sa_column=Column('job_id', BigInteger, nullable=True, index=True),
    )
    lcm_county_id: Optional[str] = Field(default=None, sa_column=Column('lcm_county_id', String, nullable=True))
    sector_id: Optional[str] = Field(default=None, sa_column=Column('sector_id', String, nullable=True))
    agg_sector: Optional[str] = Field(default=None, sa_column=Column('agg_sector', String, nullable=True))
    block_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'block_id',
            String,
            ForeignKey('UrbansimPostprocessUsimBlocksTable.block_id'),
            nullable=True,
            index=True,
        ),
    )


class UrbansimPostprocessUsimResidentialUnitsTable(SQLModel, table=True):
    __tablename__ = 'UrbansimPostprocessUsimResidentialUnitsTable'
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    unit_id: Optional[int] = Field(
        default=None,
        sa_column=Column('unit_id', BigInteger, nullable=True, index=True),
    )
    year_built: Optional[int] = Field(default=None, sa_column=Column('year_built', BigInteger, nullable=True))
    building_type_id: Optional[int] = Field(default=None, sa_column=Column('building_type_id', BigInteger, nullable=True))
    acs_13_rent: Optional[float] = Field(default=None, sa_column=Column('ACS_13_rent', Float, nullable=True))
    block_group_id: Optional[str] = Field(default=None, sa_column=Column('block_group_id', String, nullable=True))
    block_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'block_id',
            String,
            ForeignKey('UrbansimPostprocessUsimBlocksTable.block_id'),
            nullable=True,
            index=True,
        ),
    )
    lcm_county_id: Optional[str] = Field(default=None, sa_column=Column('lcm_county_id', String, nullable=True))
    acs_18_rent: Optional[float] = Field(default=None, sa_column=Column('ACS_18_rent', Float, nullable=True))
    acs_18_value: Optional[float] = Field(default=None, sa_column=Column('ACS_18_value', Float, nullable=True))
    acs_13_value: Optional[float] = Field(default=None, sa_column=Column('ACS_13_value', Float, nullable=True))


class UrbansimPostprocessUsimGraveyardTable(SQLModel, table=True):
    __tablename__ = 'UrbansimPostprocessUsimGraveyardTable'
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    person_id: Optional[int] = Field(
        default=None,
        sa_column=Column('person_id', BigInteger, nullable=True, index=True),
    )
    relate: Optional[float] = Field(default=None, sa_column=Column('relate', Float, nullable=True))
    p_hispanic: Optional[str] = Field(default=None, sa_column=Column('p_hispanic', String, nullable=True))
    work_block_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'work_block_id',
            String,
            ForeignKey('UrbansimPostprocessUsimBlocksTable.block_id'),
            nullable=True,
            index=True,
        ),
    )
    member_id: Optional[int] = Field(default=None, sa_column=Column('member_id', BigInteger, nullable=True))
    school_zone_id: Optional[int] = Field(default=None, sa_column=Column('school_zone_id', Integer, nullable=True))
    school_id: Optional[str] = Field(default=None, sa_column=Column('school_id', String, nullable=True))
    student: Optional[float] = Field(default=None, sa_column=Column('student', Float, nullable=True))
    work_zone_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'work_zone_id',
            String,
            ForeignKey('UrbansimPostprocessUsimTazZoneGeomsTable.zone_id'),
            nullable=True,
            index=True,
        ),
    )
    age_group: Optional[str] = Field(default=None, sa_column=Column('age_group', String, nullable=True))
    hispanic_1: Optional[float] = Field(default=None, sa_column=Column('hispanic.1', Float, nullable=True))
    education_group: Optional[str] = Field(default=None, sa_column=Column('education_group', String, nullable=True))
    workplace_taz: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'workplace_taz',
            String,
            ForeignKey('UrbansimPostprocessUsimTazZoneGeomsTable.zone_id'),
            nullable=True,
            index=True,
        ),
    )
    sex: Optional[float] = Field(default=None, sa_column=Column('sex', Float, nullable=True))
    person_age: Optional[str] = Field(default=None, sa_column=Column('person_age', String, nullable=True))
    school_block_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'school_block_id',
            String,
            ForeignKey('UrbansimPostprocessUsimBlocksTable.block_id'),
            nullable=True,
            index=True,
        ),
    )
    worker: Optional[float] = Field(default=None, sa_column=Column('worker', Float, nullable=True))
    earning: Optional[float] = Field(default=None, sa_column=Column('earning', Float, nullable=True))
    hours: Optional[float] = Field(default=None, sa_column=Column('hours', Float, nullable=True))
    edu: Optional[float] = Field(default=None, sa_column=Column('edu', Float, nullable=True))
    race_id: Optional[float] = Field(default=None, sa_column=Column('race_id', Float, nullable=True))
    hispanic: Optional[float] = Field(default=None, sa_column=Column('hispanic', Float, nullable=True))
    person_sex: Optional[str] = Field(default=None, sa_column=Column('person_sex', String, nullable=True))
    school_taz: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'school_taz',
            String,
            ForeignKey('UrbansimPostprocessUsimTazZoneGeomsTable.zone_id'),
            nullable=True,
            index=True,
        ),
    )
    age: Optional[float] = Field(default=None, sa_column=Column('age', Float, nullable=True))
    work_at_home: Optional[float] = Field(default=None, sa_column=Column('work_at_home', Float, nullable=True))
    race: Optional[str] = Field(default=None, sa_column=Column('race', String, nullable=True))
    household_id: Optional[float] = Field(
        default=None,
        sa_column=Column(
            'household_id',
            Float,
            ForeignKey('UrbansimPostprocessUsimHouseholdsTable.household_id'),
            nullable=True,
            index=True,
        ),
    )
    mar: Optional[float] = Field(default=None, sa_column=Column('MAR', Float, nullable=True))


class UrbansimPostprocessUsimWorkLocationsTable(SQLModel, table=True):
    __tablename__ = 'UrbansimPostprocessUsimWorkLocationsTable'
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    index: Optional[int] = Field(
        default=None,
        sa_column=Column('index', BigInteger, nullable=True, index=True),
    )
    work_block_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            'work_block_id',
            String,
            ForeignKey('UrbansimPostprocessUsimBlocksTable.block_id'),
            nullable=True,
            index=True,
        ),
    )
    person_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            'person_id',
            BigInteger,
            ForeignKey('UrbansimPostprocessUsimPersonsTable.person_id'),
            nullable=True,
            index=True,
        ),
    )
