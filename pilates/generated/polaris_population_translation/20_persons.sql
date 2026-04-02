-- Person translation.
-- `id` is normalized to a 0-based within-household index by auto-detecting
-- whether the source `member_id` starts at 1.

create or replace temp view polaris_persons as
with member_offset as (
  select member_id_offset from person_member_id_offset
)
select
  cast(p.person_id as bigint) as person,
  cast(coalesce(try_cast(p.member_id as bigint), 0) - member_offset.member_id_offset as integer) as id,
  coalesce(try_cast(p.school_block_id as bigint), 0) as school_location_id,
  0 as work_location_id,
  cast(p.age as integer) as age,
  cast(
    case
      when cast(p.worker as integer) = 1 then 1
      else 0
    end as integer
  ) as worker_class,
  cast(
    case
      when cast(p.edu as integer) between 16 and 17 then 16
      when cast(p.edu as integer) = 21 then 21
      else 0
    end as integer
  ) as education,
  cast(null as integer) as industry,
  cast(
    case
      when cast(p.worker as integer) = 1 then 1
      else 6
    end as integer
  ) as employment,
  cast(
    case
      when cast(p.sex as integer) = 1 then 1
      when cast(p.sex as integer) = 2 then 2
      else 0
    end as integer
  ) as gender,
  cast(p.earning as integer) as income,
  0 as journey_to_work_arrival_time,
  0 as journey_to_work_mode,
  0 as journey_to_work_travel_time,
  0 as journey_to_work_vehicle_occupancy,
  cast(
    case
      when cast(__PERSON_MAR_EXPR__ as integer) = 1 then 1
      when cast(__PERSON_MAR_EXPR__ as integer) = 5 then 5
      else 0
    end as integer
  ) as marital_status,
  cast(coalesce(try_cast(p.race_id as integer), 0) as integer) as race,
  cast(
    case
      when cast(p.student as integer) = 1 then 2
      else 1
    end as integer
  ) as school_enrollment,
  cast(
    case
      when cast(p.student as integer) <> 1 then 0
      when cast(p.age as integer) between 3 and 4 then 1
      when cast(p.age as integer) = 5 then 2
      when cast(p.age as integer) between 6 and 17 then cast(p.age as integer) - 4
      when cast(p.age as integer) >= 18 then 15
      else 0
    end as integer
  ) as school_grade_level,
  cast(p.hours as integer) as work_hours,
  cast(
    case
      when cast(p.work_at_home as integer) = 1 then 5
      else 0
    end as integer
  ) as telecommute_level,
  0 as transit_pass,
  cast(p.household_id as bigint) as household,
  0 as time_in_job,
  2 as disability,
  0 as escooter_use_level,
  cast(case when cast(p.age as integer) >= 18 then 1 else 0 end as integer) as is_long_term_chooser
from persons_post_atlas p
cross join member_offset;
