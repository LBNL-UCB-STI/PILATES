-- Vehicle translation.

create or replace temp view polaris_vehicles as
select
  cast(v.vehicle_id as bigint) as vehicle_id,
  cast(v.household_id as bigint) as hhold,
  0 as parking,
  0 as L3_wtp,
  0 as L4_wtp,
  cast(t.type_id as integer) as type,
  cast(
    case
      when lower(coalesce(cast(v.ownlease as varchar), '')) = 'lease' then cast(v.household_id as bigint)
      when cast(v.household_id as bigint) < 0 then cast(v.household_id as bigint)
      else null
    end as bigint
  ) as fleet,
  0 as subtype
from vehicles_post_atlas v
left join vehicle_type_lookup t
  on cast(v."vehicleTypeId" as varchar) = t.vehicleTypeId
 and lower(cast(v.bodytype as varchar)) = t.bodytype_key
 and lower(cast(v.adopt_fuel as varchar)) = t.fueltype_key
 and cast(v.modelyear as bigint) = t.model_year
 and lower(coalesce(cast(v.adopt_veh as varchar), '')) = t.adopt_veh_key;
