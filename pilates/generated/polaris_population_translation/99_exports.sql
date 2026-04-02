-- Example writes.
-- Replace `__OUT_DIR__` before execution.

copy (select * from polaris_households) to '__OUT_DIR__/household.parquet' (format parquet, codec zstd);
copy (select * from polaris_persons) to '__OUT_DIR__/person.parquet' (format parquet, codec zstd);
copy (select * from polaris_vehicles) to '__OUT_DIR__/vehicle.parquet' (format parquet, codec zstd);
copy (select * from polaris_vehicle_class) to '__OUT_DIR__/vehicle_class.parquet' (format parquet, codec zstd);
copy (select * from polaris_vehicle_type) to '__OUT_DIR__/vehicle_type.parquet' (format parquet, codec zstd);
