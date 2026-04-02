-- Replace these path literals before execution.
-- This file is the only place where source locations should need to change.

create or replace temp view households_post_atlas as
select * from read_parquet('__HOUSEHOLDS_POST_ATLAS_PARQUET__');

create or replace temp view persons_post_atlas as
select * from read_parquet('__PERSONS_POST_ATLAS_PARQUET__');

create or replace temp view vehicles_post_atlas as
select * from read_parquet('__VEHICLES_POST_ATLAS_PARQUET__');

create or replace temp view new_vehicles_ref as
select * from read_csv_auto('__ATLAS_NEW_VEHICLES_BIANNUAL_VALUES_CSV__');

create or replace temp view used_vehicles_ref as
select * from read_csv_auto('__ATLAS_USED_VEHICLES_CSV__');

create or replace temp view person_member_id_offset as
select
  case
    when min(try_cast(member_id as bigint)) = 1
      and max(case when try_cast(member_id as bigint) = 0 then 1 else 0 end) = 0
    then 1
    else 0
  end as member_id_offset
from persons_post_atlas;
