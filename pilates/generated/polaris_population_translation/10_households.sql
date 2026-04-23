-- Household translation.
-- `type` intentionally mirrors the current ActivitySim preprocessor logic:
-- if `persons == 1`, map to 1, otherwise map to 4.

create or replace temp view polaris_households as
select
  cast(h.household_id as bigint) as household,
  cast(h.household_id as bigint) as hhold,
  try_cast(h.block_id as bigint) as location,
  cast(h.persons as integer) as persons,
  cast(h.workers as integer) as workers,
  cast(h.cars as integer) as vehicles,
  cast(case when cast(h.persons as bigint) = 1 then 1 else 4 end as integer) as type,
  cast(h.income as integer) as income,
  0 as bikes,
  cast(
    case
      when lower(coalesce(cast(h.sf_detached as varchar), '')) in ('yes', '1', 'true') then 1
      else 3
    end as integer
  ) as housing_unit_type,
  0 as ecom,
  0 as delRat,
  0 as dispose_veh,
  0 as time_in_home,
  0 as Has_Residential_Charging,
  0 as num_groceries,
  0 as num_meals
from households_post_atlas h;
