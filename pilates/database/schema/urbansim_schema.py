from __future__ import annotations

from typing import Optional

from sqlalchemy import BigInteger, Column, Float, Integer, String
from sqlmodel import Field, SQLModel


class ActivitysimPostprocessUsimHouseholdsUpdated(SQLModel, table=True):
    __tablename__ = 'ActivitysimPostprocessUsimHouseholdsUpdated'
    __table_args__ = {"extend_existing": True}
    __abstract__ = True

    household_id: Optional[int] = Field(
        default=None,
        description='UrbanSim household identifier updated by ActivitySim postprocessing.',
        sa_column=Column('household_id', BigInteger, nullable=True, index=True),
    )
    tenure_mover: Optional[str] = Field(default=None, sa_column=Column('tenure_mover', String, nullable=True))
    block_id: Optional[str] = Field(
        default=None,
        description='Block identifier retained in the UrbanSim household table.',
        sa_column=Column('block_id', String, nullable=True, index=True),
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
        sa_column=Column('person_id', BigInteger, nullable=True, index=True),
    )
    hispanic: Optional[float] = Field(default=None, sa_column=Column('hispanic', Float, nullable=True))
    person_sex: Optional[str] = Field(default=None, sa_column=Column('person_sex', String, nullable=True))
    age: Optional[float] = Field(default=None, sa_column=Column('age', Float, nullable=True))
    person_age: Optional[str] = Field(default=None, sa_column=Column('person_age', String, nullable=True))
    education_group: Optional[str] = Field(default=None, sa_column=Column('education_group', String, nullable=True))
    edu: Optional[float] = Field(default=None, sa_column=Column('edu', Float, nullable=True))
    workplace_taz: Optional[int] = Field(default=None, sa_column=Column('workplace_taz', Integer, nullable=True))
    race: Optional[str] = Field(default=None, sa_column=Column('race', String, nullable=True))
    race_id: Optional[float] = Field(default=None, sa_column=Column('race_id', Float, nullable=True))
    earning: Optional[float] = Field(default=None, sa_column=Column('earning', Float, nullable=True))
    student: Optional[float] = Field(default=None, sa_column=Column('student', Float, nullable=True))
    age_group: Optional[str] = Field(default=None, sa_column=Column('age_group', String, nullable=True))
    p_hispanic: Optional[str] = Field(default=None, sa_column=Column('p_hispanic', String, nullable=True))
    work_block_id: Optional[str] = Field(
        default=None,
        description='Work block identifier retained in the UrbanSim person table.',
        sa_column=Column('work_block_id', String, nullable=True, index=True),
    )
    hispanic_1: Optional[float] = Field(default=None, sa_column=Column('hispanic.1', Float, nullable=True))
    school_zone_id: Optional[int] = Field(default=None, sa_column=Column('school_zone_id', Integer, nullable=True))
    work_zone_id: Optional[int] = Field(default=None, sa_column=Column('work_zone_id', Integer, nullable=True))
    household_id: Optional[float] = Field(
        default=None,
        description='Household identifier carried through the updated UrbanSim person table.',
        sa_column=Column('household_id', Float, nullable=True, index=True),
    )
    worker: Optional[float] = Field(default=None, sa_column=Column('worker', Float, nullable=True))
    school_block_id: Optional[str] = Field(
        default=None,
        description='School block identifier retained in the UrbanSim person table.',
        sa_column=Column('school_block_id', String, nullable=True, index=True),
    )
    school_id: Optional[str] = Field(default=None, sa_column=Column('school_id', String, nullable=True))
    sex: Optional[float] = Field(default=None, sa_column=Column('sex', Float, nullable=True))
    member_id: Optional[int] = Field(default=None, sa_column=Column('member_id', BigInteger, nullable=True))
    relate: Optional[float] = Field(default=None, sa_column=Column('relate', Float, nullable=True))
    work_at_home: Optional[float] = Field(default=None, sa_column=Column('work_at_home', Float, nullable=True))
    hours: Optional[float] = Field(default=None, sa_column=Column('hours', Float, nullable=True))
    mar: Optional[float] = Field(default=None, sa_column=Column('MAR', Float, nullable=True))
    school_taz: Optional[int] = Field(default=None, sa_column=Column('school_taz', Integer, nullable=True))
