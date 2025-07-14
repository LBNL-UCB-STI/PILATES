#!/usr/bin/env python3

import argparse
import os
import logging
import numpy as np

try:
    import xarray as xr
except ImportError:
    raise ImportError("xarray is required to read Zarr skims.")
try:
    import openmatrix as omx
except ImportError:
    raise ImportError("openmatrix is required to write OMX files.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Zarr-format skims to OMX-format skims."
    )
    parser.add_argument("zarr_path", help="Path to input Zarr skims directory")
    parser.add_argument("omx_path", help="Path to output OMX file")
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="List of variable names to exclude"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("zarr_to_omx")

    zarr_path = args.zarr_path
    omx_path = args.omx_path
    exclude_tables = set(args.exclude)

    if not os.path.exists(zarr_path):
        logger.error(f"Zarr path does not exist: {zarr_path}")
        exit(1)

    if os.path.exists(omx_path):
        logger.info(f"Deleting existing OMX file: {omx_path}")
        os.remove(omx_path)

    logger.info(f"Opening Zarr skims: {zarr_path}")
    ds = xr.open_zarr(zarr_path)

    # Get zone IDs and time periods
    zone_ids = None
    if "original_zone_ids" in ds.attrs:
        zone_ids = np.array(ds.attrs["original_zone_ids"])
        logger.info(f"Using original zone IDs from Zarr attributes: {len(zone_ids)} zones")
    elif "otaz" in ds.coords:
        if ds["otaz"].attrs.get("preprocessed") != "zero-based-contiguous":
            zone_ids = ds.coords["otaz"].values
            logger.info(f"Using zone IDs from coordinates: {len(zone_ids)} zones")
        else:
            logger.error("Zarr uses zero-based zones but no original zone mapping found!")
            zone_ids = zone_order(settings, settings.get("start_year", 2015))
            logger.warning(f"Reconstructed zone IDs from settings: {len(zone_ids)} zones")
    else:
        logger.warning("No 'otaz' coordinate found in Zarr file.")

    time_periods = []
    if "time_period" in ds.coords:
        time_periods = [str(s) for s in ds.time_period.values]
        logger.info(f"Found {len(time_periods)} time periods: {time_periods}")
    else:
        logger.warning("No 'time_period' coordinate found in Zarr file.")

    logger.info(f"Creating OMX file: {omx_path}")
    omx_file = omx.open_file(omx_path, "w")

    # Add zone mapping if available
    if zone_ids is not None and len(zone_ids) > 0:
        try:
            zone_ids = np.array(zone_ids, dtype=int)
            omx_file.create_mapping("zone_id", zone_ids, overwrite=True)
            logger.info(f"Created 'zone_id' mapping in OMX file with {len(zone_ids)} zones.")
        except Exception as e:
            logger.error(f"Error creating zone mapping in OMX file: {e}.")

    scaled_measures = {"TOTIVT", "IVT", "WACC", "IWAIT", "XWAIT", "WAUX", "WEGR", "DTIM", "FERRYIVT", "KEYIVT", "FAR"}

    written_count = 0
    for key in ds.data_vars:
        if key in exclude_tables:
            logger.info(f"Skipping excluded variable: {key}")
            continue
        try:
            data_array = ds[key]
            data = data_array.values

            measure_name = key.split("_")[-1] if "_" in key else key
            needs_descaling = (measure_name in scaled_measures and not key.startswith(("TNC_", "RH_")))

            if data_array.ndim == 2:
                data_to_write = np.nan_to_num(data).astype(np.float32)
                if needs_descaling:
                    data_to_write = data_to_write / 100.0
                    logger.debug(f"Descaled 2D matrix '{key}' by 100x")
                omx_file[key] = data_to_write
                written_count += 1

            elif data_array.ndim == 3:
                if not time_periods or data_array.shape[-1] != len(time_periods):
                    logger.warning(
                        f"Period dimension mismatch for 3D variable '{key}': {data_array.shape[-1]} vs expected {len(time_periods)}. Skipping."
                    )
                    continue
                for t_idx, tp in enumerate(time_periods):
                    new_key = f"{key}__{tp}"
                    slice_data = data[:, :, t_idx]
                    data_to_write = np.nan_to_num(slice_data).astype(np.float32)
                    if needs_descaling:
                        data_to_write = data_to_write / 100.0
                        logger.debug(f"Descaled 3D slice '{new_key}' by 100x")
                    omx_file[new_key] = data_to_write
                    strsplit = key.rsplit("_", 1)
                    omx_file[new_key].attrs["measure"] = strsplit[-1]
                    omx_file[new_key].attrs["timePeriod"] = tp
                    if len(strsplit) == 2:
                        omx_file[new_key].attrs["mode"] = strsplit[0]
                    written_count += 1
            else:
                logger.warning(
                    f"Skipping variable '{key}' with unexpected dimension count: {data_array.ndim}"
                )
        except Exception as e:
            logger.error(f"Error writing variable '{key}' to OMX: {e}")

    omx_file.close()
    ds.close()
    logger.info(f"Finished writing {written_count} matrices to OMX file {omx_path}.")


if __name__ == "__main__":
    main()
