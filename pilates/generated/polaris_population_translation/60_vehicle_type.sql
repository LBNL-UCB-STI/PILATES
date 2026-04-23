-- Vehicle type translation.
-- Replace `__EXPORT_YEAR__` with the actual export year before execution.

create or replace temp view polaris_vehicle_type as
select
  cast(t.type_id as integer) as type_id,
  cast(c.class_id as integer) as vehicle_class,
  cast(
    case
      when t.adopt_veh_key like '%l5%' or t.adopt_veh_key like '%cav%' then 1
      else 0
    end as integer
  ) as connectivity_type,
  cast(
    case
      when t.fueltype_key in ('conv', 'gasoline', 'gas') then 0
      when t.fueltype_key in ('phev', 'hybrid') then 2
      when t.fueltype_key in ('bev', 'electric') then 3
      when t.fueltype_key in ('fcev', 'hydrogen', 'h2') then 5
      else 0
    end as integer
  ) as powertrain_type,
  cast(
    case
      when t.adopt_veh_key like '%l5%' then 2
      when t.adopt_veh_key like '%av%' or t.adopt_veh_key like '%cav%' then 1
      else 0
    end as integer
  ) as automation_type,
  cast(
    case
      when t.fueltype_key in ('conv', 'gasoline', 'gas') then 0
      when t.fueltype_key = 'diesel' then 1
      when t.fueltype_key in ('fcev', 'hydrogen', 'h2') then 3
      when t.fueltype_key in ('bev', 'electric', 'phev', 'hybrid') then 4
      else 0
    end as integer
  ) as fuel_type,
  cast(
    case
      when __EXPORT_YEAR__ - t.model_year <= 5 then 0
      when __EXPORT_YEAR__ - t.model_year <= 10 then 1
      else 2
    end as integer
  ) as vintage_type,
  0 as ev_features_id,
  coalesce(used_ref.avg_cpmile, new_ref.avg_cpmile, 0.18) as operating_cost_per_mile
from vehicle_type_lookup t
left join vehicle_class_lookup c
  on t.bodytype_key = c.bodytype_key
left join vehicle_type_ref used_ref
  on t.bodytype_key = used_ref.bodytype_key
 and t.fueltype_key = used_ref.fueltype_key
 and t.model_year = used_ref.model_year
left join vehicle_type_ref_new new_ref
  on t.bodytype_key = new_ref.bodytype_key
 and t.fueltype_key = new_ref.fueltype_key;
