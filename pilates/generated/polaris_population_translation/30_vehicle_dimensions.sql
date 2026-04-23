-- Vehicle dimension tables derived from the observed vehicle stock plus
-- ATLAS adopt lookup inputs.

create or replace temp view vehicle_class_lookup as
select
  dense_rank() over (order by lower(coalesce(bodytype, 'unknown'))) as class_id,
  lower(coalesce(bodytype, 'unknown')) as bodytype_key,
  coalesce(bodytype, 'unknown') as class_type
from (
  select distinct cast(bodytype as varchar) as bodytype
  from vehicles_post_atlas
) classes;

create or replace temp view vehicle_class_ref as
select
  lower(cast(bodytype as varchar)) as bodytype_key,
  avg(try_cast(accel as double)) as avg_accel
from new_vehicles_ref
group by 1;

create or replace temp view vehicle_type_lookup as
select
  dense_rank() over (
    order by
      lower(coalesce(cast("vehicleTypeId" as varchar), '')),
      lower(coalesce(cast(bodytype as varchar), '')),
      lower(coalesce(cast(adopt_fuel as varchar), '')),
      cast(modelyear as bigint),
      lower(coalesce(cast(adopt_veh as varchar), ''))
  ) as type_id,
  cast("vehicleTypeId" as varchar) as vehicleTypeId,
  lower(cast(bodytype as varchar)) as bodytype_key,
  lower(cast(adopt_fuel as varchar)) as fueltype_key,
  cast(modelyear as bigint) as model_year,
  lower(coalesce(cast(adopt_veh as varchar), '')) as adopt_veh_key
from (
  select distinct
    "vehicleTypeId",
    bodytype,
    adopt_fuel,
    modelyear,
    adopt_veh
  from vehicles_post_atlas
) types;

create or replace temp view vehicle_type_ref as
select
  lower(cast(bodytype as varchar)) as bodytype_key,
  lower(cast(fueltype as varchar)) as fueltype_key,
  cast(model_year as bigint) as model_year,
  avg(try_cast(cpmile as double)) as avg_cpmile
from used_vehicles_ref
group by 1, 2, 3;

create or replace temp view vehicle_type_ref_new as
select
  lower(cast(bodytype as varchar)) as bodytype_key,
  lower(cast(fueltype as varchar)) as fueltype_key,
  avg(try_cast(cpmile as double)) as avg_cpmile
from new_vehicles_ref
group by 1, 2;
