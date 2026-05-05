-- Vehicle class translation.
-- Length and EV ML class are best-effort lookup defaults for now.

create or replace temp view polaris_vehicle_class as
select
  cast(c.class_id as integer) as class_id,
  c.class_type as class_type,
  0 as capacity,
  cast(
    case c.bodytype_key
      when 'car' then 4.8
      when 'suv' then 5.1
      when 'truck' then 5.5
      when 'minvan' then 5.2
      else 5.0
    end as double
  ) as length,
  45.0 as max_speed,
  coalesce(r.avg_accel, 8.0) as max_accel,
  -4.5 as max_decel,
  cast(
    case c.bodytype_key
      when 'car' then 1
      when 'suv' then 2
      when 'truck' then 3
      when 'minvan' then 4
      else 0
    end as integer
  ) as ev_ml_class
from vehicle_class_lookup c
left join vehicle_class_ref r
  on c.bodytype_key = r.bodytype_key;
