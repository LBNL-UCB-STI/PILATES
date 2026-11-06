import logging
import os
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import openmatrix as omx
import xarray as xr

import pilates.utils.zone_utils as zone_utils
from pilates.utils.settings_helper import get as get_setting

logger = logging.getLogger(__name__)

TNC_CONSOLIDATION_MAP = {
    "TNC_SINGLE": "RH_SOLO",
    "TNC_POOLED": "RH_SHARED",
}

_TNC_CONSOLIDATION_MEASURES = (
    "TRIPS",
    "FAILURES",
    "REJECTIONPROB",
    "IWAIT",
    "FAR",
    "DDIST",
    "TOTIVT",
)

_TNC_COMMON_MEASURES = (*_TNC_CONSOLIDATION_MEASURES, "DTIM")

_TNC_TRANSFER_MEASURES = (
    *_TNC_COMMON_MEASURES,
    "XWAIT",
    "WACC",
    "WAUX",
    "WEGR",
    "COST",
)

_GENERIC_OMX_MEASURES = (
    *_TNC_TRANSFER_MEASURES,
    "DIST",
    "TIME",
    "BTOLL",
    "VTOLL",
    "KEYIVT",
    "FERRYIVT",
)


def _normalize_beam_partial_time_period_coords(
    partial_skims_ds: xr.Dataset, expected_time_periods: List[str]
) -> xr.Dataset:
    """Convert BEAM's index+labels time_period coord into canonical string labels."""
    if "time_period" not in partial_skims_ds.coords:
        return partial_skims_ds

    time_coord = partial_skims_ds.coords["time_period"]
    current_values = [str(tp) for tp in time_coord.values]
    label_values = time_coord.attrs.get("labels")
    if label_values is None:
        return zone_utils.normalize_dimension_coords_for_zarr(partial_skims_ds)

    label_values = [str(tp) for tp in label_values]
    if current_values == expected_time_periods:
        return zone_utils.normalize_dimension_coords_for_zarr(partial_skims_ds)

    if label_values != expected_time_periods:
        logger.warning(
            "BEAM partial skims time_period labels %s do not match expected %s; "
            "leaving coords unchanged.",
            label_values,
            expected_time_periods,
        )
        return zone_utils.normalize_dimension_coords_for_zarr(partial_skims_ds)

    max_len = max(len(value) for value in label_values) if label_values else 1
    normalized_values = np.asarray(label_values, dtype=f"<U{max(1, max_len)}")
    normalized_attrs = {
        key: value for key, value in time_coord.attrs.items() if key != "labels"
    }
    logger.info(
        "Converting BEAM partial skims time_period coord from %s to canonical labels %s",
        current_values,
        label_values,
    )
    partial_skims_ds = partial_skims_ds.assign_coords(
        time_period=xr.DataArray(
            normalized_values,
            dims=time_coord.dims,
            attrs=normalized_attrs,
        )
    )
    return zone_utils.normalize_dimension_coords_for_zarr(partial_skims_ds)


def _copy_single_unobserved_mode_var(
    skims_ds: xr.Dataset,
    all_vars: List[str],
    var_name: str,
    from_mode: str,
    to_modes: List[str],
) -> None:
    if not var_name.startswith(from_mode + "_") or any(
        m in var_name for m in ["_TRIPS", "_FAILURES"]
    ):
        return
    if "TOLL" in var_name.split("_")[0]:
        return

    measure_name = var_name[len(from_mode) + 1 :]
    for to_mode in to_modes:
        target_var_name = f"{to_mode}_{measure_name}"
        if target_var_name not in all_vars:
            logger.debug(
                f"Target variable '{target_var_name}' not found in dataset. Skipping copy."
            )
            continue
        if skims_ds[var_name].shape == skims_ds[target_var_name].shape:
            skims_ds[target_var_name].data[:] = skims_ds[var_name].data[:]
            logger.info(f"Copied data from '{var_name}' to '{target_var_name}'")
        else:
            logger.warning(
                f"Shape mismatch when copying from '{var_name}' to '{target_var_name}'. Skipping."
            )


def copy_skims_for_unobserved_modes(mapping, skims_ds):
    """
    Copy skim data from one mode to others based on a mapping.
    Operates on the Zarr dataset directly.

    Parameters
    ----------
    mapping : dict
        Mapping from source mode to list of target modes (e.g., {"SOV": ["SOVTOLL", ...]}).
    skims_ds : xarray.Dataset
        The target Zarr dataset containing all skims.
    """
    logger.info("Copying skims for unobserved modes...")
    for fromMode, toModes in mapping.items():
        if isinstance(toModes, str):
            toModes = [toModes]

        all_vars = list(skims_ds.data_vars)
        for var_name in all_vars:
            _copy_single_unobserved_mode_var(
                skims_ds, all_vars, var_name, fromMode, toModes
            )

    logger.info("Completed copying skims for unobserved modes.")


def _parse_tnc_provider_mode_from_var(var_name: str) -> Optional[str]:
    if not (var_name.startswith("TNC_") and "_TRIPS" in var_name):
        return None
    parts = var_name.split("_")
    if len(parts) >= 3 and parts[2] not in _TNC_COMMON_MEASURES:
        return "_".join(parts[:3])
    if len(parts) >= 2 and parts[1] in ["SINGLE", "POOLED"]:
        measure_part = var_name.split("__")[0]
        for common_measure in _TNC_COMMON_MEASURES:
            if measure_part.endswith("_" + common_measure):
                return measure_part[: -len("_" + common_measure)]
        logger.debug(f"Could not parse TNC provider mode from variable: {var_name}")
    return None


def _resolve_tnc_modes_to_postprocess(
    skims_ds: xr.Dataset, use_rh_modes: bool
) -> List[str]:
    if use_rh_modes:
        modes_to_process = [
            target_mode
            for source_mode, target_mode in TNC_CONSOLIDATION_MAP.items()
            if f"{target_mode}_TRIPS" in skims_ds.data_vars
        ]
        logger.info(f"Post-processing consolidated RH modes: {modes_to_process}")
        return modes_to_process

    tnc_provider_modes = set()
    for var_name in list(skims_ds.data_vars):
        mode_name = _parse_tnc_provider_mode_from_var(var_name)
        if mode_name is not None:
            tnc_provider_modes.add(mode_name)

    modes_to_process = list(tnc_provider_modes)
    logger.info(f"Post-processing original TNC provider modes: {modes_to_process}")
    return modes_to_process


def _update_tnc_rejection_prob(
    mode: str,
    rejection_prob_da: xr.DataArray,
    completed_3d: np.ndarray,
    failed_3d: np.ndarray,
    time_periods: List[str],
) -> None:
    logger.debug(f"Calculating REJECTIONPROB for {mode}")
    total_trips_3d = completed_3d + failed_3d
    valid_ods_mask_3d = total_trips_3d > 0

    rejection_prob_3d = np.zeros_like(completed_3d, dtype=np.float32)
    np.divide(
        failed_3d,
        total_trips_3d,
        out=rejection_prob_3d,
        where=valid_ods_mask_3d,
    )

    completed_sum_by_origin_3d = completed_3d.sum(axis=1)
    failed_sum_by_origin_3d = failed_3d.sum(axis=1)
    total_trips_origin_3d = completed_sum_by_origin_3d + failed_sum_by_origin_3d
    valid_origins_mask_3d = total_trips_origin_3d > 0

    origin_probs_3d = np.zeros_like(completed_sum_by_origin_3d, dtype=np.float32)
    np.divide(
        failed_sum_by_origin_3d,
        total_trips_origin_3d,
        out=origin_probs_3d,
        where=valid_origins_mask_3d,
    )

    for tp_idx in range(len(time_periods)):
        valid_origins_tp = valid_origins_mask_3d[:, tp_idx]
        origin_probs_tp = origin_probs_3d[:, tp_idx]
        rejection_prob_da.data[valid_origins_tp, :, tp_idx] = origin_probs_tp[
            valid_origins_tp, None
        ]

    logger.debug(f"Updated REJECTIONPROB for {mode}")


def _update_tnc_iwait(
    mode: str,
    iwait_da: xr.DataArray,
    completed_3d: np.ndarray,
    time_periods: List[str],
) -> None:
    logger.debug(f"Processing IWAIT for {mode}")
    current_iwait_data_3d = np.nan_to_num(iwait_da.data)
    completed_3d_scaled = completed_3d

    sum_weighted_wait_3d = np.nansum(
        current_iwait_data_3d * completed_3d_scaled, axis=1
    )
    sum_completed_origin_3d = np.nansum(completed_3d_scaled, axis=1)

    weighted_mean_by_origin_3d = np.full_like(
        sum_completed_origin_3d, np.nan, dtype=np.float32
    )
    valid_origins_for_mean_3d = sum_completed_origin_3d != 0
    np.divide(
        sum_weighted_wait_3d,
        sum_completed_origin_3d,
        out=weighted_mean_by_origin_3d,
        where=valid_origins_for_mean_3d,
    )

    for tp_idx in range(len(time_periods)):
        completed_tp = completed_3d[:, :, tp_idx]
        valid_origins_tp = valid_origins_for_mean_3d[:, tp_idx]
        weighted_mean_tp = weighted_mean_by_origin_3d[:, tp_idx]

        mask_to_fill_with_average = (completed_tp == 0) & (valid_origins_tp[:, None])
        if mask_to_fill_with_average.any():
            origin_indices_to_fill = np.where(mask_to_fill_with_average)[0]
            iwait_da.data[mask_to_fill_with_average, tp_idx] = weighted_mean_tp[
                origin_indices_to_fill
            ]
            logger.debug(
                f"Filled {mask_to_fill_with_average.sum()} {mode} IWAIT values in {time_periods[tp_idx]} with origin weighted average."
            )

    logger.debug(f"Updated IWAIT for {mode}")


def _interpolate_tnc_measure_with_sov(
    mode: str,
    measure_label: str,
    sov_label: str,
    sov_var_name: str,
    target_da: Optional[xr.DataArray],
    sov_da: Optional[xr.DataArray],
    completed_3d: np.ndarray,
    min_ratio: float = 0.8,
) -> None:
    if target_da is None:
        return
    if sov_da is None:
        logger.warning(
            f"{mode} {measure_label} cannot be interpolated: {sov_var_name} is missing from skims."
        )
        return

    logger.debug(f"Processing {measure_label} for {mode}")
    current_data_3d = np.nan_to_num(target_da.data)
    sov_data_3d = np.nan_to_num(sov_da.data)

    mask_for_ratio_3d = (sov_data_3d > 0) & (completed_3d > 0) & (current_data_3d > 0)

    ratio = np.nan
    if mask_for_ratio_3d.any():
        weighted_mode_3d = (
            current_data_3d[mask_for_ratio_3d] * completed_3d[mask_for_ratio_3d]
        )
        weighted_sov_3d = (
            sov_data_3d[mask_for_ratio_3d] * completed_3d[mask_for_ratio_3d]
        )
        sum_weighted_sov = np.sum(weighted_sov_3d)
        if sum_weighted_sov > 0:
            ratio = np.sum(weighted_mode_3d) / sum_weighted_sov

        ratios_individual = (
            current_data_3d[mask_for_ratio_3d] / sov_data_3d[mask_for_ratio_3d]
        )
        logger.info(
            f"Observed {mode} {measure_label}/{sov_label} ratio of {ratio:2.3f} ({np.nanpercentile(ratios_individual, 10):2.3f} - {np.nanpercentile(ratios_individual, 90):2.3f}) for {mode}. "
            f"Interpolating {np.sum(~mask_for_ratio_3d):.0f} missing values."
        )
    else:
        logger.info(
            f"No data points available to calculate {mode} {measure_label}/{sov_label} ratio for {mode}."
        )

    if not np.isnan(ratio) and ratio < min_ratio:
        logger.warning(
            f"Calculated {mode} {measure_label}/{sov_label} ratio ({ratio:2.3f}) is below {min_ratio}. Setting ratio to {min_ratio}."
        )
        ratio = min_ratio

    mask_to_interpolate_3d = ~mask_for_ratio_3d
    if not np.isnan(ratio) and mask_to_interpolate_3d.any():
        mask_interpolation_valid_sov_3d = mask_to_interpolate_3d & (sov_data_3d > 0)
        target_da.data[mask_interpolation_valid_sov_3d] = (
            sov_data_3d[mask_interpolation_valid_sov_3d] * ratio
        )
        logger.debug(
            f"Interpolated {mask_interpolation_valid_sov_3d.sum()} {mode} {measure_label} values using {sov_var_name} ratio."
        )

    if np.isnan(target_da.data).any():
        nan_count = np.isnan(target_da.data).sum()
        target_da.data[np.isnan(target_da.data)] = 0.0
        logger.debug(
            f"Set {nan_count} remaining NaN {mode} {measure_label} values to 0."
        )

    logger.debug(f"Updated {measure_label} for {mode}")


def _update_tnc_far(
    mode: str,
    far_da: xr.DataArray,
    completed_3d: np.ndarray,
    failed_3d: np.ndarray,
) -> None:
    logger.debug(f"Processing FAR for {mode}")
    default_far = 0.0

    mask_zero_trips = (completed_3d + failed_3d) == 0
    mask_nan_far = np.isnan(far_da.data)
    mask_to_set = mask_zero_trips | mask_nan_far

    if mask_to_set.any():
        far_da.data[mask_to_set] = default_far
        logger.debug(
            f"Set {mask_to_set.sum()} {mode} FAR values to default {default_far}"
        )

    negative_far = far_da.data < 0
    if negative_far.any():
        logger.warning(
            f"Found {negative_far.sum()} negative FAR values for {mode}. Setting to 0."
        )
        far_da.data[negative_far] = 0.0

    logger.debug(f"Updated FAR for {mode}")


def _postprocess_single_tnc_mode_zarr(
    skims_ds: xr.Dataset,
    mode: str,
    time_periods: List[str],
    sov_dist_da: Optional[xr.DataArray],
    sov_time_da: Optional[xr.DataArray],
) -> None:
    logger.debug(f"Post-processing mode: {mode}")

    completed_key = f"{mode}_TRIPS"
    failed_key = f"{mode}_FAILURES"

    completed_da = skims_ds.get(completed_key)
    failed_da = skims_ds.get(failed_key)

    if completed_da is None or failed_da is None:
        logger.warning(
            f"Missing {completed_key} or {failed_key} for TNC/RH post-processing. Skipping {mode}."
        )
        return

    completed_3d = np.nan_to_num(completed_da.data)
    failed_3d = np.nan_to_num(failed_da.data)

    rejection_prob_da = skims_ds.get(f"{mode}_REJECTIONPROB")
    iwait_da = skims_ds.get(f"{mode}_IWAIT")
    ddist_da = skims_ds.get(f"{mode}_DDIST")
    totivt_da = skims_ds.get(f"{mode}_TOTIVT")
    far_da = skims_ds.get(f"{mode}_FAR")

    if rejection_prob_da is not None:
        _update_tnc_rejection_prob(
            mode, rejection_prob_da, completed_3d, failed_3d, time_periods
        )
    if iwait_da is not None:
        _update_tnc_iwait(mode, iwait_da, completed_3d, time_periods)

    _interpolate_tnc_measure_with_sov(
        mode=mode,
        measure_label="DDIST",
        sov_label="SOV DIST",
        sov_var_name="SOV_DIST",
        target_da=ddist_da,
        sov_da=sov_dist_da,
        completed_3d=completed_3d,
    )
    _interpolate_tnc_measure_with_sov(
        mode=mode,
        measure_label="TOTIVT",
        sov_label="SOV TIME",
        sov_var_name="SOV_TIME",
        target_da=totivt_da,
        sov_da=sov_time_da,
        completed_3d=completed_3d,
    )

    if far_da is not None:
        _update_tnc_far(mode, far_da, completed_3d, failed_3d)


def _postprocess_tnc_zarr(
    skims_ds,
    timePeriods,
    settings,
    completed_failed_dict,
    use_rh_modes=False,
):
    """
    Applies TNC/RH-specific post-processing rules (REJECTIONPROB, IWAIT filling,
    DDIST/TOTIVT interpolation, FAR assignment) to Zarr skims.
    Operates on either TNC_* or RH_* variables based on `use_rh_modes`.

    Parameters:
    -----------
    skims_ds : xarray.Dataset
        The target Zarr dataset containing all skims.
    timePeriods : list of str
        List of time period names.
    settings : dict
        Settings dictionary, needed for SOV keys and potential FAR value.
    completed_failed_dict : dict
        Dictionary containing completed and failed trip counts aggregated across
        all time periods for each mode path. Keyed by mode path, values are
        [completed_trips_3d_array, failed_trips_3d_array]. This dict should
        contain counts for the modes being processed (either TNC providers or RH).
    use_rh_modes : bool
        If True, process RH_* modes. If False, process TNC_* provider modes.
    """
    logger.info("Applying TNC/RH-specific post-processing...")
    modes_to_process = _resolve_tnc_modes_to_postprocess(skims_ds, use_rh_modes)
    if not modes_to_process:
        logger.info("No TNC/RH modes found in skims dataset to post-process.")
        return

    sov_dist_da = skims_ds.get("SOV_DIST")
    sov_time_da = skims_ds.get("SOV_TIME")
    if (sov_dist_da is None or sov_time_da is None) and any(
        mode.startswith("RH_") for mode in modes_to_process
    ):
        logger.warning(
            "SOV_DIST or SOV_TIME missing, DDIST/TOTIVT interpolation for RH modes will be skipped."
        )

    for mode in modes_to_process:
        _postprocess_single_tnc_mode_zarr(
            skims_ds, mode, timePeriods, sov_dist_da, sov_time_da
        )

    logger.info("Completed TNC/RH-specific post-processing.")


def _accumulate_all_completed_failed_trips_zarr(partial_skims_ds, timePeriods):
    """
    Accumulates completed and failed trip counts from all modes/providers
    in the partial Zarr dataset.

    Parameters:
    -----------
    partial_skims_ds : xarray.Dataset
        The Zarr dataset containing the partial skims from BEAM.
    timePeriods : list of str
        List of time period names.

    Returns:
    --------
    dict
        Dictionary where keys are mode paths and values are tuples:
        (completed_trips_3d_array, failed_trips_3d_array).
    """

    if "time_period" in partial_skims_ds.coords:
        zarr_time_periods = [str(tp) for tp in partial_skims_ds.time_period.values]
        if zarr_time_periods != timePeriods:
            logger.warning(
                f"Time periods mismatch: Zarr has {zarr_time_periods}, "
                f"expected {timePeriods}. Using Zarr periods."
            )

    logger.info(
        "Accumulating completed and failed trip counts from partial Zarr skims..."
    )
    completed_failed_dict = {}

    for var_name in partial_skims_ds.data_vars:
        is_trips = var_name.endswith("_TRIPS")
        is_failures = var_name.endswith("_FAILURES")

        if is_trips or is_failures:
            mode = (
                var_name[: -len("_TRIPS")]
                if is_trips
                else var_name[: -len("_FAILURES")]
            )

            if mode not in completed_failed_dict:
                shape = partial_skims_ds[var_name].shape
                completed_failed_dict[mode] = [
                    np.zeros(shape, dtype=np.float32),
                    np.zeros(shape, dtype=np.float32),
                ]

            try:
                data_3d = np.nan_to_num(partial_skims_ds[var_name].values)
                if is_trips:
                    completed_failed_dict[mode][0] = data_3d
                else:
                    completed_failed_dict[mode][1] = data_3d
            except Exception as e:
                logger.error(f"Error reading partial Zarr skim {var_name}: {e}")

    logger.info(
        f"Accumulated trip counts for {len(completed_failed_dict)} modes/providers."
    )
    return completed_failed_dict


def _resolve_omx_partial_skim_shapes(
    partial_skims: Any, time_periods: List[str]
) -> Tuple[Optional[Tuple[int, ...]], Optional[List[int]]]:
    try:
        sample_matrix = next(
            (
                m
                for m in partial_skims.list_matrices()
                if m.endswith("__" + time_periods[0])
            ),
            None,
        )
        if sample_matrix:
            out_array_shape_2d = partial_skims[sample_matrix].shape
            out_array_shape_3d = list(out_array_shape_2d) + [len(time_periods)]
            return out_array_shape_2d, out_array_shape_3d

        logger.warning(
            "No partial skim matrices found with period suffix. Cannot determine shape for trip counts."
        )
        return None, None
    except Exception as e:
        logger.error(
            f"Error determining partial skim shape: {e}. Cannot accumulate trip counts."
        )
        return None, None


def _accumulate_single_omx_trip_matrix(
    partial_skims: Any,
    omx_key: str,
    time_periods: List[str],
    out_array_shape_2d: Tuple[int, ...],
    out_array_shape_3d: List[int],
    completed_failed_dict: Dict[str, List[np.ndarray]],
) -> None:
    parts = omx_key.split("__")
    if len(parts) != 2:
        return

    measure_part, period = parts[0], parts[1]
    if period not in time_periods:
        return

    is_trips = measure_part.endswith("_TRIPS")
    is_failures = measure_part.endswith("_FAILURES")
    if not (is_trips or is_failures):
        return

    mode = (
        measure_part[: -len("_TRIPS")]
        if is_trips
        else measure_part[: -len("_FAILURES")]
    )
    if mode not in completed_failed_dict:
        completed_failed_dict[mode] = [
            np.zeros(out_array_shape_3d, dtype=np.float32),
            np.zeros(out_array_shape_3d, dtype=np.float32),
        ]

    try:
        tp_idx = time_periods.index(period)
        data_2d = np.nan_to_num(partial_skims[omx_key][:])
        if data_2d.shape != out_array_shape_2d:
            logger.warning(
                f"Shape mismatch for partial skim {omx_key}. Expected {out_array_shape_2d}, got {data_2d.shape}. Skipping accumulation."
            )
            return

        if is_trips:
            completed_failed_dict[mode][0][:, :, tp_idx] = data_2d
        else:
            completed_failed_dict[mode][1][:, :, tp_idx] = data_2d
    except ValueError:
        logger.warning(
            f"Period '{period}' from key '{omx_key}' not found in settings periods. Skipping."
        )
    except Exception as e:
        logger.error(
            f"Error reading partial skim {omx_key} for trip count accumulation: {e}"
        )


def _accumulate_all_completed_failed_trips(partialSkims, timePeriods):
    """
    Accumulates completed and failed trip counts from all modes/providers
    in the partial OMX skim file.

    Parameters:
    -----------
    partialSkims : omx.open_file object
        The OMX file containing the partial skims from BEAM.
    timePeriods : list of str
        List of time period names.

    Returns:
    --------
    dict
        Dictionary where keys are mode paths (e.g., "SOV", "TNC_SINGLE_UBER", "WLK_TRN_WLK")
        and values are tuples: (completed_trips_3d_array, failed_trips_3d_array).
    """
    logger.info("Accumulating completed and failed trip counts from partial skims...")
    out_array_shape_2d, out_array_shape_3d = _resolve_omx_partial_skim_shapes(
        partialSkims, timePeriods
    )
    if out_array_shape_2d is None or out_array_shape_3d is None:
        return {}

    completed_failed_dict: Dict[str, List[np.ndarray]] = {}
    for omx_key in partialSkims.list_matrices():
        _accumulate_single_omx_trip_matrix(
            partialSkims,
            omx_key,
            timePeriods,
            out_array_shape_2d,
            out_array_shape_3d,
            completed_failed_dict,
        )

    logger.info(
        f"Accumulated trip counts for {len(completed_failed_dict)} modes/providers."
    )
    return completed_failed_dict


def _transform_measure(
    input_vals, completed, failed, measure, path, transit_scale_factor=100.0
):
    """
    Transforms skim values based on measure type, path, and trip completion statistics.
    This is applied to a single 2D slice (one time period).

    Parameters:
    -----------
    input_vals : ndarray (2D)
        The new observed values for a single time period from the partial skim.
    completed : ndarray (2D)
        Count of completed trips for each OD pair for the current time period (2D slice).
    failed : ndarray (2D)
        Count of failed trips for each OD pair for the current time period (2D slice).
    measure : str
        The measure name (e.g., "IWAIT", "TOTIVT").
    path : str
        The mode path (e.g., "SOV", "WLK_TRN_WLK", "TNC_SINGLE_UBER", "RH_SOLO").

    Returns:
    --------
    tuple (mask, vals, to_cancel)
        mask: Boolean array (2D) indicating which cells to update with `vals`.
        vals: Values (1D, matching mask.sum()) to assign to the masked cells.
        to_cancel: Boolean array (2D) indicating cells to zero out due to high failure rate (for IVT/TOTIVT).
                   Returns None for other measures.
    """
    valid = ~np.isnan(input_vals)
    completed_mask = completed > 0
    basic_mask = valid & completed_mask
    negative = input_vals < 0
    if np.any(negative):
        logger.debug(
            f"Found {np.sum(negative)} negative values in input_vals for measure {measure} and path {path} in one period. Setting them to 0."
        )
        input_vals[negative] = 0.0

    measures_is_scaled_for_transit = {
        "TOTIVT",
        "IVT",
        "WACC",
        "IWAIT",
        "XWAIT",
        "WAUX",
        "WEGR",
        "DTIM",
        "FERRYIVT",
        "KEYIVT",
        "FAR",
    }
    is_tnc_or_rh = path.startswith("TNC_") or path.startswith("RH_")
    scaling = (
        transit_scale_factor
        if measure in measures_is_scaled_for_transit and not is_tnc_or_rh
        else 1.0
    )

    if measure in measures_is_scaled_for_transit and measure not in ["TOTIVT", "IVT"]:
        return basic_mask, input_vals[basic_mask] * scaling, None

    if measure in ["TOTIVT", "IVT"]:
        if is_tnc_or_rh:
            return basic_mask, input_vals[basic_mask], None
        to_cancel = (failed > 5) & (failed > (1 * completed))
        update_mask = basic_mask & ~to_cancel
        scaled_vals = input_vals[update_mask] * scaling
        return update_mask, scaled_vals, to_cancel

    if measure == "DIST" or measure not in ["REJECTIONPROB", "FAR"]:
        return basic_mask, input_vals[basic_mask], None

    logger.debug(
        f"Measure {measure} for path {path} will be handled in post-processing. No direct merge."
    )
    return np.zeros_like(input_vals, dtype=bool), np.array([]), None


def _merge_one_zarr_measure_from_zarr(
    partial_skims_ds,
    skims_ds,
    path,
    measure,
    timePeriods,
    completed_3d,
    failed_3d,
):
    """
    Merges a single measure's data from Zarr partial skims into the master Zarr dataset.
    """
    source_var_name = f"{path}_{measure}"
    target_var_name = source_var_name

    if source_var_name not in partial_skims_ds.data_vars:
        logger.debug(f"Variable {source_var_name} not found in partial Zarr skims.")
        return None, None

    if target_var_name not in skims_ds.data_vars:
        logger.warning(f"Target Zarr variable '{target_var_name}' not found. Ignoring")
        return None, None

    target_da = skims_ds[target_var_name]
    source_da = partial_skims_ds[source_var_name]

    input_vals_3d = source_da.values

    for tpIdx in range(input_vals_3d.shape[2]):
        input_vals_tp = input_vals_3d[:, :, tpIdx]
        completed_tp = completed_3d[:, :, tpIdx]
        failed_tp = failed_3d[:, :, tpIdx]

        mask_update, vals_update, to_cancel_tp = _transform_measure(
            input_vals_tp,
            completed_tp,
            failed_tp,
            measure,
            path,
        )

        current_slice = target_da.data[:, :, tpIdx]
        if mask_update.any():
            current_slice[mask_update] = vals_update

        if to_cancel_tp is not None and to_cancel_tp.any():
            cancellation_count = to_cancel_tp.sum()
            current_slice[to_cancel_tp] = 0.0
            logger.debug(f"  Canceled {cancellation_count} ODs for {source_var_name}")

        target_da.data[:, :, tpIdx] = current_slice

    return path, measure


def _ensure_target_zarr_var_from_sov_trips(
    skims_ds: xr.Dataset, target_var_name: str, warn_on_create: bool = False
) -> Optional[xr.DataArray]:
    if target_var_name not in skims_ds.data_vars:
        if warn_on_create:
            logger.warning(
                f"Target Zarr variable '{target_var_name}' not found. Creating with SOV_TRIPS shape."
            )
        else:
            logger.debug(
                f"Creating target variable '{target_var_name}' in Zarr dataset."
            )
        try:
            zarr_shape = skims_ds["SOV_TRIPS"].shape
            zarr_coords = skims_ds["SOV_TRIPS"].coords
            zarr_dims = skims_ds["SOV_TRIPS"].dims
            skims_ds[target_var_name] = xr.DataArray(
                np.zeros(zarr_shape, dtype=np.float32),
                coords=zarr_coords,
                dims=zarr_dims,
                name=target_var_name,
            )
        except KeyError:
            logger.error(
                f"Cannot create variable {target_var_name}: SOV_TRIPS not found to get shape/coords."
            )
            return None
    return skims_ds[target_var_name]


def _merge_single_omx_period_into_zarr_slice(
    partial_skims: Any,
    target_da: xr.DataArray,
    target_var_name: str,
    partial_key: str,
    path: str,
    measure: str,
    tp: str,
    tp_idx: int,
    completed_3d: np.ndarray,
    failed_3d: np.ndarray,
) -> None:
    if partial_key not in partial_skims:
        logger.debug(
            f"Partial skims missing key {partial_key}. Skipping merge for this period."
        )
        return

    try:
        input_vals_tp = partial_skims[partial_key][:]

        if target_da.ndim < 3 or target_da.shape[-1] <= tp_idx:
            logger.warning(
                f"Target Zarr variable {target_var_name} does not have enough time period dimensions for period {tp} (index {tp_idx}). Skipping merge for this period."
            )
            return

        if input_vals_tp.shape != target_da.shape[:2]:
            logger.warning(
                f"Shape mismatch for partial skim {partial_key}. Expected {target_da.shape[:2]}, got {input_vals_tp.shape}. Skipping merge for this period."
            )
            return

        completed_tp = completed_3d[:, :, tp_idx]
        failed_tp = failed_3d[:, :, tp_idx]

        mask_update, vals_update, to_cancel_tp = _transform_measure(
            input_vals_tp,
            completed_tp,
            failed_tp,
            measure,
            path,
        )

        current_slice = target_da.data[:, :, tp_idx]
        if mask_update.any():
            current_slice[mask_update] = vals_update

        if to_cancel_tp is not None and to_cancel_tp.any():
            cancellation_count = to_cancel_tp.sum()
            current_slice[to_cancel_tp] = 0.0
            logger.debug(f"  Canceled {cancellation_count} ODs for {partial_key}")

        target_da.data[:, :, tp_idx] = current_slice
    except Exception as e:
        logger.error(f"Error processing skim slice {partial_key}: {e}")


def _merge_one_zarr_measure(
    partialSkims,
    skims_ds,
    path,
    measure,
    timePeriods,
    completed_3d,
    failed_3d,
):
    """
    Merges a single measure's data for a given path and all time periods
    from OMX partial skims into a Zarr DataArray, applying measure-specific transformations.
    """
    target_var_name = f"{path}_{measure}"
    target_da = _ensure_target_zarr_var_from_sov_trips(
        skims_ds, target_var_name, warn_on_create=True
    )
    if target_da is None:
        return None, None

    for tpIdx, tp in enumerate(timePeriods):
        partial_key = f"{path}_{measure}__{tp}"
        _merge_single_omx_period_into_zarr_slice(
            partialSkims,
            target_da,
            target_var_name,
            partial_key,
            path,
            measure,
            tp,
            tpIdx,
            completed_3d,
            failed_3d,
        )

    return path, measure


def _merge_zarr_trip_counts(
    allSkims,
    path,
    completed,
    failed,
    previous_weight=0.9,
):
    key_completed = f"{path}_TRIPS"
    key_failed = f"{path}_FAILURES"
    any_observed = (completed + failed) > 0
    if key_completed in allSkims and key_failed in allSkims:
        prev_completed = np.nan_to_num(allSkims[key_completed].data)
        prev_failed = np.nan_to_num(allSkims[key_failed].data)
        logger.info(
            f"For {path} previously had {prev_completed[any_observed].sum():.0f} completed trips and {prev_failed[any_observed].sum():.0f} failed trips"
        )
        prev_completed = np.ceil(previous_weight * prev_completed + completed).astype(
            int
        )
        prev_failed = np.ceil(previous_weight * prev_failed + failed).astype(int)
        allSkims[key_completed].data[:] = prev_completed
        allSkims[key_failed].data[:] = prev_failed
        logger.info(
            f"Now we have {prev_completed[any_observed].sum():.0f} completed trips and {prev_failed[any_observed].sum():.0f} failed trips"
        )
    else:
        logger.warning(
            f"Skipping trip counts merge for {path} as {key_completed} or {key_failed} does not exist in target skims file"
        )


def _merge_zarr_trip_counts_slice(
    allSkims,
    path,
    completed_tp,
    failed_tp,
    tp_idx,
    previous_weight=0.9,
):
    key_completed = f"{path}_TRIPS"
    key_failed = f"{path}_FAILURES"
    if key_completed in allSkims and key_failed in allSkims:
        prev_completed_tp = np.nan_to_num(allSkims[key_completed].data[:, :, tp_idx])
        prev_failed_tp = np.nan_to_num(allSkims[key_failed].data[:, :, tp_idx])
        prev_completed_tp = np.ceil(
            previous_weight * prev_completed_tp + completed_tp
        ).astype(int)
        prev_failed_tp = np.ceil(previous_weight * prev_failed_tp + failed_tp).astype(
            int
        )
        allSkims[key_completed].data[:, :, tp_idx] = prev_completed_tp
        allSkims[key_failed].data[:, :, tp_idx] = prev_failed_tp
    else:
        logger.warning(
            f"Skipping trip counts merge for {path} at time index {tp_idx} as {key_completed} or {key_failed} does not exist in target skims file"
        )


def _update_specific_transit_mode_availability_slice(
    skims_ds: xr.Dataset,
    tp: str,
    tp_idx: int,
    mode: str,
    mask_significant_transit: np.ndarray,
    significant_count: int,
) -> None:
    mode_trips_da = skims_ds.get(f"{mode}_TRIPS")
    if mode_trips_da is None:
        logger.info(
            f"{tp:<6} | {mode:<12} | {significant_count:<18} | {'NO DATA':<18} | {'NO DATA':<10} | {'N/A':<10}"
        )
        return

    mode_trips = mode_trips_da.data[:, :, tp_idx]
    mask_mode_unused = mask_significant_transit & (mode_trips == 0)
    unused_count = np.sum(mask_mode_unused)

    totivt_da = skims_ds.get(f"{mode}_TOTIVT")
    changed_count = 0

    if totivt_da is not None and unused_count > 0:
        current_totivt_slice = totivt_da.data[:, :, tp_idx]
        mask_to_change = mask_mode_unused & (
            (current_totivt_slice > 0) | np.isnan(current_totivt_slice)
        )
        changed_count = np.sum(mask_to_change)
        if changed_count > 0:
            current_totivt_slice[mask_to_change] = 0.0
            totivt_da.data[:, :, tp_idx] = current_totivt_slice

    percent_changed = (
        (changed_count / significant_count) * 100 if significant_count > 0 else 0
    )
    logger.info(
        f"{tp:<6} | {mode:<12} | {significant_count:<18} | {unused_count:<18} | {changed_count:<10} | {percent_changed:<10.1f}"
    )


def _handle_transit_mode_availability(skims_ds, timePeriods):
    """
    Handles transit mode availability by checking specific transit modes.

    For OD pairs with 50+ successful general transit trips, any specific transit mode
    with 0 successful trips will be marked as unavailable (TOTIVT = 0).
    Only marks OD pairs where TOTIVT is currently > 0 or np.nan.
    Operates directly on the Zarr dataset.
    """
    general_transit_path = "WLK_TRN_WLK"
    specific_transit_modes = [
        "WLK_LOC_WLK",
        "WLK_HVY_WLK",
        "WLK_COM_WLK",
        "WLK_EXP_WLK",
    ]

    logger.info(f"{'=' * 80}")
    logger.info("Transit Mode Availability Analysis")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"{'Period':<6} | {'Mode':<12} | {'ODs w/50+ Transit':<18} | {'ODs w/0 Mode Trips':<18} | {'Changed':<10} | {'% Changed':<10}"
    )
    logger.info(
        f"{'-' * 6}-+-{'-' * 12}-+-{'-' * 18}-+-{'-' * 18}-+-{'-' * 10}-+-{'-' * 10}"
    )

    tp_to_idx = {tp: idx for idx, tp in enumerate(timePeriods)}
    general_trips_da = skims_ds.get(f"{general_transit_path}_TRIPS")

    for tp in timePeriods:
        tp_idx = tp_to_idx[tp]
        if general_trips_da is None:
            logger.info(
                f"{tp:<6} | {'ALL MODES':<12} | {'NO DATA':<18} | {'NO DATA':<18} | {'NO DATA':<10} | {'N/A':<10}"
            )
            continue

        general_transit_trips = general_trips_da.data[:, :, tp_idx]
        mask_significant_transit = general_transit_trips >= 50
        significant_count = np.sum(mask_significant_transit)

        if not np.any(mask_significant_transit):
            logger.info(
                f"{tp:<6} | {'ALL MODES':<12} | {0:<18} | {0:<18} | {0:<10} | {0:<10.1f}"
            )
            continue

        for mode in specific_transit_modes:
            _update_specific_transit_mode_availability_slice(
                skims_ds,
                tp,
                tp_idx,
                mode,
                mask_significant_transit,
                significant_count,
            )

    logger.info(f"{'=' * 80}")


def _parse_rightmost_measure(
    parts: List[str], valid_measures: Tuple[str, ...]
) -> Tuple[Optional[List[str]], Optional[str]]:
    reversed_parts = parts[::-1]
    for i in range(len(reversed_parts)):
        potential_measure = "_".join(reversed_parts[: i + 1][::-1])
        if potential_measure in valid_measures:
            prefix_parts = parts[: len(parts) - (i + 1)]
            return prefix_parts, potential_measure
    return None, None


def _parse_tnc_consolidation_omx_key(
    omx_key: str, time_periods: List[str]
) -> Optional[Tuple[str, str, str, str]]:
    parts = omx_key.split("__")
    if len(parts) != 2:
        return None

    measure_part, period = parts[0], parts[1]
    if not measure_part.startswith("TNC_") or period not in time_periods:
        return None

    measure_part_parts = measure_part.split("_")
    if len(measure_part_parts) < 3:
        logger.debug(
            f"Skipping TNC key with unexpected format (too few parts): {omx_key}"
        )
        return None

    base_mode_key = f"{measure_part_parts[0]}_{measure_part_parts[1]}"
    if base_mode_key not in TNC_CONSOLIDATION_MAP:
        logger.debug(f"Skipping TNC key with unknown base mode pattern: {omx_key}")
        return None

    provider_parts, measure = _parse_rightmost_measure(
        list(measure_part_parts[2:]), _TNC_CONSOLIDATION_MEASURES
    )
    if measure is None:
        logger.warning(
            f"Could not identify measure name in TNC key: {omx_key}. Skipping."
        )
        return None

    provider = "_".join(provider_parts) if provider_parts else "DEFAULT"
    return base_mode_key, measure, provider, period


def _group_tnc_consolidation_sources(
    all_partial_vars: List[str], time_periods: List[str]
) -> Dict[Tuple[str, str], List[Tuple[str, str, str]]]:
    grouped_sources: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}
    for omx_key in all_partial_vars:
        parsed = _parse_tnc_consolidation_omx_key(omx_key, time_periods)
        if parsed is None:
            continue
        base_mode_key, measure, provider, period = parsed
        grouped_sources.setdefault((base_mode_key, measure), []).append(
            (provider, period, omx_key)
        )
    logger.debug(f"Grouped TNC sources from OMX: {grouped_sources}")
    return grouped_sources


def _consolidate_tnc_omx_trip_counts_for_measure(
    partial_skims: Any,
    skims_ds: xr.Dataset,
    base_mode_key: str,
    measure: str,
    sources: List[Tuple[str, str, str]],
    time_periods: List[str],
    zarr_shape: Tuple[int, ...],
) -> str:
    target_mode = TNC_CONSOLIDATION_MAP[base_mode_key]
    target_var_name = f"{target_mode}_{measure}"
    target_da = _ensure_target_zarr_var_from_sov_trips(skims_ds, target_var_name)

    for tp_idx, period in enumerate(time_periods):
        consolidated_tp = np.zeros(zarr_shape[:2], dtype=np.float32)
        for _provider, period_key, omx_key in sources:
            if period_key != period:
                continue
            if omx_key not in partial_skims:
                logger.debug(f"Partial skim {omx_key} not found. Skipping.")
                continue
            try:
                input_vals_tp = np.nan_to_num(partial_skims[omx_key][:])
                if input_vals_tp.shape == zarr_shape[:2]:
                    consolidated_tp += input_vals_tp
                else:
                    logger.warning(
                        f"Shape mismatch for partial skim {omx_key}. Expected {zarr_shape[:2]}, got {input_vals_tp.shape}. Skipping sum."
                    )
            except Exception as e:
                logger.error(
                    f"Error reading partial skim {omx_key} for consolidation: {e}"
                )
        target_da.data[:, :, tp_idx] = consolidated_tp
    return target_mode


def _accumulate_weighted_provider_omx_slice(
    partial_skims: Any,
    omx_key: str,
    provider_mode_key: str,
    tp_idx: int,
    zarr_shape_2d: Tuple[int, ...],
    completed_failed_dict_providers: Dict[str, Any],
    sum_weighted_values_tp: np.ndarray,
    sum_weights_tp: np.ndarray,
) -> None:
    if omx_key not in partial_skims:
        logger.debug(f"Partial skim {omx_key} not found. Skipping.")
        return

    try:
        input_vals_tp = np.nan_to_num(partial_skims[omx_key][:])
        if provider_mode_key not in completed_failed_dict_providers:
            logger.debug(
                f"Completed trip data for provider mode {provider_mode_key} not found in provider dict. Skipping weighted average contribution for {omx_key}."
            )
            return

        provider_completed_3d, _ = completed_failed_dict_providers[provider_mode_key]
        completed_tp = np.nan_to_num(provider_completed_3d[:, :, tp_idx])
        if input_vals_tp.shape == zarr_shape_2d and completed_tp.shape == zarr_shape_2d:
            input_vals_tp[input_vals_tp < 0] = 0.0
            sum_weighted_values_tp += input_vals_tp * completed_tp
            sum_weights_tp += completed_tp
        else:
            logger.warning(
                f"Shape mismatch for partial skim {omx_key} or its trips data. Skipping weighted average contribution."
            )
    except Exception as e:
        logger.error(
            f"Error processing partial skim {omx_key} for weighted consolidation: {e}"
        )


def _consolidate_tnc_omx_weighted_measure(
    partial_skims: Any,
    skims_ds: xr.Dataset,
    base_mode_key: str,
    measure: str,
    sources: List[Tuple[str, str, str]],
    time_periods: List[str],
    zarr_shape: Tuple[int, ...],
    completed_failed_dict_providers: Dict[str, Any],
) -> None:
    target_mode = TNC_CONSOLIDATION_MAP[base_mode_key]
    target_var_name = f"{target_mode}_{measure}"
    target_da = _ensure_target_zarr_var_from_sov_trips(skims_ds, target_var_name)

    for tp_idx, period in enumerate(time_periods):
        sum_weighted_values_tp = np.zeros(zarr_shape[:2], dtype=np.float32)
        sum_weights_tp = np.zeros(zarr_shape[:2], dtype=np.float32)
        for provider, period_key, omx_key in sources:
            if period_key != period:
                continue
            provider_mode_key = f"{base_mode_key}_{provider}"
            _accumulate_weighted_provider_omx_slice(
                partial_skims,
                omx_key,
                provider_mode_key,
                tp_idx,
                zarr_shape[:2],
                completed_failed_dict_providers,
                sum_weighted_values_tp,
                sum_weights_tp,
            )

        consolidated_tp = np.zeros(zarr_shape[:2], dtype=np.float32)
        valid_weights_mask = sum_weights_tp != 0
        np.divide(
            sum_weighted_values_tp,
            sum_weights_tp,
            out=consolidated_tp,
            where=valid_weights_mask,
        )
        target_da.data[:, :, tp_idx] = consolidated_tp


def _get_validated_zarr_sov_trips_shape(
    skims_ds: xr.Dataset, time_periods: List[str], warn_on_mismatch: bool = True
) -> Optional[Tuple[int, ...]]:
    try:
        zarr_shape = skims_ds["SOV_TRIPS"].shape
        if warn_on_mismatch and zarr_shape[-1] != len(time_periods):
            logger.warning(
                f"Zarr time period dimension ({zarr_shape[-1]}) does not match settings periods ({len(time_periods)}). Using Zarr shape."
            )
        if zarr_shape[-1] < len(time_periods):
            logger.error(
                f"Zarr shape {zarr_shape} has fewer time periods than in settings ({len(time_periods)}). Cannot consolidate."
            )
            return None
        return zarr_shape
    except KeyError:
        logger.error(
            "Zarr dataset does not contain 'SOV_TRIPS'. Cannot determine shape and coordinates for new variables."
        )
        return None
    except Exception as e:
        logger.error(f"Error getting Zarr shape/coords: {e}. Cannot consolidate.")
        return None


def _consolidate_tnc_data_zarr(
    partialSkims,
    skims_ds,
    timePeriods,
    completed_failed_dict_providers,
    settings,
):
    """
    Consolidates skims from multiple TNC providers (e.g., TNC_SINGLE_UBER, TNC_SINGLE_LYFT)
    into single consolidated skims (e.g., RH_SOLO, RH_SHARED) within the Zarr dataset.
    Reads from partialSkims (OMX) and writes to skims_ds (Zarr).
    """
    logger.info("Starting TNC fleet consolidation from OMX to Zarr...")

    all_partial_vars = partialSkims.list_matrices()
    zarr_shape = _get_validated_zarr_sov_trips_shape(
        skims_ds, timePeriods, warn_on_mismatch=True
    )
    if zarr_shape is None:
        return {}

    grouped_sources = _group_tnc_consolidation_sources(all_partial_vars, timePeriods)
    consolidated_modes = set()

    for (base_mode_key, measure), sources in grouped_sources.items():
        if measure in ["TRIPS", "FAILURES"]:
            target_mode = _consolidate_tnc_omx_trip_counts_for_measure(
                partialSkims,
                skims_ds,
                base_mode_key,
                measure,
                sources,
                timePeriods,
                zarr_shape,
            )
            consolidated_modes.add(target_mode)

    for (base_mode_key, measure), sources in grouped_sources.items():
        if measure not in ["TRIPS", "FAILURES", "REJECTIONPROB"]:
            _consolidate_tnc_omx_weighted_measure(
                partialSkims,
                skims_ds,
                base_mode_key,
                measure,
                sources,
                timePeriods,
                zarr_shape,
                completed_failed_dict_providers,
            )

    logger.info("Completed TNC fleet consolidation from OMX to Zarr.")
    return consolidated_modes


def _parse_tnc_transfer_omx_key(
    omx_key: str, time_periods: List[str]
) -> Optional[Tuple[str, str, str]]:
    parts = omx_key.split("__")
    if len(parts) != 2:
        return None

    measure_part, period = parts[0], parts[1]
    if not measure_part.startswith("TNC_") or period not in time_periods:
        return None

    measure_part_parts = measure_part.split("_")
    if len(measure_part_parts) < 3:
        logger.debug(
            f"Skipping TNC key with unexpected format (too few parts): {omx_key}"
        )
        return None

    _, measure = _parse_rightmost_measure(
        list(measure_part_parts[2:]), _TNC_TRANSFER_MEASURES
    )
    if measure is None:
        logger.warning(
            f"Could not parse TNC provider mode or measure from key: {omx_key}. Skipping."
        )
        return None

    measure_parts_count = len(measure.split("_"))
    mode_parts = measure_part_parts[: len(measure_part_parts) - measure_parts_count]
    mode = "_".join(mode_parts)
    return mode, measure, period


def _ensure_tnc_transfer_target_var(
    skims_ds: xr.Dataset,
    target_var_name: str,
    mode: str,
    time_periods: List[str],
    zarr_shape: Tuple[int, ...],
    completed_failed_dict_providers: Dict[str, Any],
) -> xr.DataArray:
    if target_var_name not in skims_ds.data_vars:
        logger.debug(f"Creating target variable '{target_var_name}' in Zarr dataset.")
        try:
            provider_completed_shape = completed_failed_dict_providers[mode][0].shape
            if provider_completed_shape[:2] != zarr_shape[
                :2
            ] or provider_completed_shape[-1] < len(time_periods):
                logger.warning(
                    f"Provider completed counts shape {provider_completed_shape} for mode {mode} is incompatible with Zarr shape {zarr_shape} or settings periods {len(time_periods)}. Falling back to SOV_TRIPS shape."
                )
                raise KeyError
            current_zarr_shape = provider_completed_shape
        except KeyError:
            current_zarr_shape = zarr_shape
            logger.debug(
                f"Using default Zarr shape {current_zarr_shape} for mode {mode}"
            )

        skims_ds[target_var_name] = xr.DataArray(
            np.zeros(current_zarr_shape, dtype=np.float32),
            coords=skims_ds["SOV_TRIPS"].coords,
            dims=skims_ds["SOV_TRIPS"].dims,
            name=target_var_name,
        )
    return skims_ds[target_var_name]


def _transfer_tnc_provider_data_zarr(
    partialSkims,
    skims_ds,
    timePeriods,
    completed_failed_dict_providers,
    settings,
):
    """
    Transfers provider-specific TNC skims from partialSkims (OMX) to skims_ds (Zarr).
    Does NOT consolidate. This is for the case where consolidation is disabled.
    """
    logger.info("Starting TNC provider data transfer from OMX to Zarr...")

    all_partial_vars = partialSkims.list_matrices()
    tp_to_idx = {tp: idx for idx, tp in enumerate(timePeriods)}
    transferred_vars = set()

    grouped_sources: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for omx_key in all_partial_vars:
        parsed = _parse_tnc_transfer_omx_key(omx_key, timePeriods)
        if parsed is None:
            continue
        mode, measure, period = parsed
        grouped_sources.setdefault((mode, measure), []).append((period, omx_key))

    zarr_shape = _get_validated_zarr_sov_trips_shape(
        skims_ds, timePeriods, warn_on_mismatch=False
    )
    if zarr_shape is None:
        return []

    for (mode, measure), sources in grouped_sources.items():
        target_var_name = f"{mode}_{measure}"
        target_da = _ensure_tnc_transfer_target_var(
            skims_ds,
            target_var_name,
            mode,
            timePeriods,
            zarr_shape,
            completed_failed_dict_providers,
        )
        transferred_vars.add(target_var_name)

        completed_3d, failed_3d = completed_failed_dict_providers.get(
            mode, (None, None)
        )
        if completed_3d is None or failed_3d is None:
            logger.warning(
                f"Completed/failed trip data for mode {mode} not found in provider dict. Skipping merge for {target_var_name}."
            )
            continue

        for period, omx_key in sources:
            _merge_single_omx_period_into_zarr_slice(
                partialSkims,
                target_da,
                target_var_name,
                omx_key,
                mode,
                measure,
                period,
                tp_to_idx[period],
                completed_3d,
                failed_3d,
            )

    logger.info("Completed TNC provider data transfer from OMX to Zarr.")
    return list(transferred_vars)


def _consolidate_rh_measure_slice_zarr(
    partial_skims_ds: xr.Dataset,
    skims_ds: xr.Dataset,
    target_mode: str,
    target_var_name: str,
    measure: str,
    provider_modes: List[str],
    tp_idx: int,
    completed_failed_dict_all: Dict[str, Any],
) -> None:
    consolidated_data_tp = np.zeros(
        skims_ds[target_var_name].shape[:2], dtype=np.float32
    )
    total_weights_tp = np.zeros_like(consolidated_data_tp)

    for provider_mode in provider_modes:
        provider_var = f"{provider_mode}_{measure}"
        if provider_var in partial_skims_ds.data_vars:
            provider_data_tp = partial_skims_ds[provider_var].data[:, :, tp_idx]
            provider_completed_tp = completed_failed_dict_all[provider_mode][0][
                :, :, tp_idx
            ]

            mask = provider_completed_tp > 0
            consolidated_data_tp[mask] += (
                provider_data_tp[mask] * provider_completed_tp[mask]
            )
            total_weights_tp[mask] += provider_completed_tp[mask]

    mask = total_weights_tp > 0
    consolidated_data_tp[mask] = consolidated_data_tp[mask] / total_weights_tp[mask]

    completed_tp = skims_ds[f"{target_mode}_TRIPS"].data[:, :, tp_idx]
    failed_tp = skims_ds[f"{target_mode}_FAILURES"].data[:, :, tp_idx]

    mask_update, vals_update, to_cancel = _transform_measure(
        consolidated_data_tp,
        completed_tp,
        failed_tp,
        measure,
        target_mode,
    )

    if mask_update.any():
        skims_ds[target_var_name].data[:, :, tp_idx][mask_update] = vals_update

    if to_cancel is not None and to_cancel.any():
        skims_ds[target_var_name].data[:, :, tp_idx][to_cancel] = 0.0


def _consolidate_single_rh_target_mode_zarr(
    partial_skims_ds: xr.Dataset,
    skims_ds: xr.Dataset,
    target_mode: str,
    provider_modes: List[str],
    time_periods: List[str],
    completed_failed_dict_all: Dict[str, Any],
    previous_weight: float,
) -> None:
    logger.info(f"Consolidating {provider_modes} into {target_mode}")

    for tp_idx in range(len(time_periods)):
        consolidated_completed_tp = np.zeros(
            completed_failed_dict_all[provider_modes[0]][0].shape[:2],
            dtype=np.float32,
        )
        consolidated_failed_tp = np.zeros_like(consolidated_completed_tp)
        for provider_mode in provider_modes:
            completed, failed = completed_failed_dict_all[provider_mode]
            consolidated_completed_tp += completed[:, :, tp_idx]
            consolidated_failed_tp += failed[:, :, tp_idx]
        _merge_zarr_trip_counts_slice(
            skims_ds,
            target_mode,
            consolidated_completed_tp,
            consolidated_failed_tp,
            tp_idx,
            previous_weight=previous_weight,
        )

    first_provider = provider_modes[0]
    provider_measures = {
        var_name[len(first_provider) + 1 :]
        for var_name in partial_skims_ds.data_vars
        if var_name.startswith(f"{first_provider}_")
        and not (var_name.endswith("_TRIPS") or var_name.endswith("_FAILURES"))
    }

    for measure in provider_measures:
        target_var_name = f"{target_mode}_{measure}"
        if target_var_name not in skims_ds.data_vars:
            template_var = f"{first_provider}_{measure}"
            if template_var in partial_skims_ds.data_vars:
                skims_ds[target_var_name] = xr.DataArray(
                    np.zeros(partial_skims_ds[template_var].shape, dtype=np.float32),
                    coords=partial_skims_ds[template_var].coords,
                    dims=partial_skims_ds[template_var].dims,
                    name=target_var_name,
                )

        if target_var_name in skims_ds.data_vars:
            for tp_idx in range(len(time_periods)):
                _consolidate_rh_measure_slice_zarr(
                    partial_skims_ds,
                    skims_ds,
                    target_mode,
                    target_var_name,
                    measure,
                    provider_modes,
                    tp_idx,
                    completed_failed_dict_all,
                )


def _transfer_tnc_providers_from_zarr(
    partial_skims_ds: xr.Dataset,
    skims_ds: xr.Dataset,
    time_periods: List[str],
    completed_failed_dict_all: Dict[str, Any],
) -> Set[str]:
    logger.info("Transferring TNC provider modes directly from Zarr partial skims...")
    tnc_modes_processed: Set[str] = set()

    tnc_provider_modes = {
        var_name[: -len("_TRIPS")]
        for var_name in partial_skims_ds.data_vars
        if var_name.startswith("TNC_") and "_TRIPS" in var_name
    }

    for provider_mode in tnc_provider_modes:
        if provider_mode not in completed_failed_dict_all:
            continue

        completed, failed = completed_failed_dict_all[provider_mode]
        provider_measures = {
            var_name[len(provider_mode) + 1 :]
            for var_name in partial_skims_ds.data_vars
            if var_name.startswith(f"{provider_mode}_")
            and not (var_name.endswith("_TRIPS") or var_name.endswith("_FAILURES"))
        }

        for measure in provider_measures:
            _merge_one_zarr_measure_from_zarr(
                partial_skims_ds,
                skims_ds,
                provider_mode,
                measure,
                time_periods,
                completed,
                failed,
            )

        tnc_modes_processed.add(provider_mode)

    return tnc_modes_processed


def _process_all_tnc_logic_zarr(
    partial_skims_ds,
    skims_ds,
    timePeriods,
    settings,
    completed_failed_dict_all,
    previous_weight=0.9,
    get_setting_fn: Callable[..., Any] = get_setting,
):
    """
    Process TNC/RH logic for Zarr format partial skims.
    Either consolidates TNC providers into RH modes or transfers them directly.
    """
    consolidate_tnc_fleets = get_setting_fn(settings, "consolidate_tnc_fleets", True)
    tnc_modes_processed: Set[str] = set()
    rh_consolidation: Dict[str, List[str]] = {}

    if consolidate_tnc_fleets:
        logger.info("Consolidating TNC fleets into RH modes from Zarr partial skims...")
        for provider_mode, target_mode in TNC_CONSOLIDATION_MAP.items():
            if provider_mode in completed_failed_dict_all:
                rh_consolidation.setdefault(target_mode, []).append(provider_mode)
                tnc_modes_processed.add(provider_mode)

        for target_mode, provider_modes in rh_consolidation.items():
            _consolidate_single_rh_target_mode_zarr(
                partial_skims_ds,
                skims_ds,
                target_mode,
                provider_modes,
                timePeriods,
                completed_failed_dict_all,
                previous_weight,
            )
            tnc_modes_processed.add(target_mode)
    else:
        tnc_modes_processed = _transfer_tnc_providers_from_zarr(
            partial_skims_ds, skims_ds, timePeriods, completed_failed_dict_all
        )

    if tnc_modes_processed:
        completed_failed_for_postprocess = {
            mode: completed_failed_dict_all.get(mode, (None, None))
            for mode in tnc_modes_processed
            if mode in completed_failed_dict_all or mode in skims_ds.data_vars
        }

        if consolidate_tnc_fleets:
            for target_mode in rh_consolidation:
                if f"{target_mode}_TRIPS" in skims_ds.data_vars:
                    completed_failed_for_postprocess[target_mode] = (
                        np.nan_to_num(skims_ds[f"{target_mode}_TRIPS"].values),
                        np.nan_to_num(skims_ds[f"{target_mode}_FAILURES"].values),
                    )

        _postprocess_tnc_zarr(
            skims_ds,
            timePeriods,
            settings,
            completed_failed_for_postprocess,
            use_rh_modes=consolidate_tnc_fleets,
        )

    return tnc_modes_processed


def _extract_transferred_tnc_provider_mode(var_name: str) -> Optional[str]:
    parts = var_name.split("_")
    if len(parts) < 3 or parts[0] != "TNC":
        return None

    measure_part = var_name.rsplit("_", 1)[0]
    if measure_part.endswith("__") or "_" in var_name.split("__")[0]:
        measure_part = var_name.split("__")[0]

    measure_name_check = measure_part
    while "_" in measure_name_check:
        potential_mode = measure_name_check
        potential_measure_end = measure_name_check.rsplit("_", 1)[-1]
        if potential_measure_end.upper() in _TNC_CONSOLIDATION_MEASURES:
            mode_name = potential_mode.rsplit("_", 1)[0]
            return mode_name if mode_name.startswith("TNC_") else None
        measure_name_check = potential_mode.rsplit("_", 1)[0]
    return None


def _process_all_tnc_logic(
    partialSkims,
    skims_ds,
    timePeriods,
    settings,
    completed_failed_dict_all,
    get_setting_fn: Callable[..., Any] = get_setting,
):
    """
    Handles all TNC/RH related skim processing:
    - If consolidate_tnc_fleets is True: Consolidates provider data -> RH_* in Zarr, then post-processes RH_*.
    - If consolidate_tnc_fleets is False: Transfers provider data -> TNC_* in Zarr, then post-processes TNC_*.
    """
    consolidate_tnc_fleets = get_setting_fn(settings, "consolidate_tnc_fleets", True)
    processed_vars = set()
    tnc_modes_to_postprocess = set()
    completed_failed_dict_for_postprocess = {}

    if consolidate_tnc_fleets:
        logger.info("TNC fleet consolidation enabled.")
        consolidated_modes = _consolidate_tnc_data_zarr(
            partialSkims,
            skims_ds,
            timePeriods,
            completed_failed_dict_all,
            settings,
        )
        tnc_modes_to_postprocess.update(consolidated_modes)
        for target_mode in consolidated_modes:
            for measure in skims_ds.data_vars:
                if measure.startswith(f"{target_mode}_"):
                    processed_vars.add(measure)
    else:
        logger.info("TNC fleet consolidation disabled. Transferring provider data.")
        transferred_vars = _transfer_tnc_provider_data_zarr(
            partialSkims,
            skims_ds,
            timePeriods,
            completed_failed_dict_all,
            settings,
        )
        processed_vars.update(transferred_vars)

        for var_name in transferred_vars:
            mode_name = _extract_transferred_tnc_provider_mode(var_name)
            if mode_name is None:
                continue
            tnc_modes_to_postprocess.add(mode_name)
            if mode_name in completed_failed_dict_all:
                completed_failed_dict_for_postprocess[mode_name] = (
                    completed_failed_dict_all[mode_name]
                )
            else:
                logger.warning(
                    f"Completed/failed counts not found for provider mode {mode_name} in completed_failed_dict_all. Cannot post-process."
                )

        logger.info(
            f"Identified {len(tnc_modes_to_postprocess)} TNC provider modes for post-processing."
        )

    if tnc_modes_to_postprocess:
        _postprocess_tnc_zarr(
            skims_ds,
            timePeriods,
            settings,
            completed_failed_dict_for_postprocess,
            use_rh_modes=consolidate_tnc_fleets,
        )
    else:
        logger.info(
            "No TNC/RH modes to post-process after consolidation/transfer step."
        )

    return processed_vars


def _open_partial_beam_skims(
    current_skims_path: str,
    settings: Any,
    get_setting_fn: Callable[..., Any] = get_setting,
) -> Tuple[str, Optional[xr.Dataset], Optional[Any]]:
    if current_skims_path.endswith(".zarr"):
        partial_skims_ds = xr.open_zarr(current_skims_path)
        time_periods = get_setting_fn(settings, "shared.skims.periods")
        partial_skims_ds = _normalize_beam_partial_time_period_coords(
            partial_skims_ds, time_periods
        )
        logger.info(
            f"DEBUG: Partial (BEAM) skims otaz coords: {partial_skims_ds.coords['otaz'].values[:5]}...{partial_skims_ds.coords['otaz'].values[-5:]}"
        )
        if "time_period" in partial_skims_ds.coords:
            logger.debug(
                "Partial BEAM skims time_period coords for %s: %s",
                current_skims_path,
                list(partial_skims_ds.coords["time_period"].values),
            )
        return "zarr", partial_skims_ds, None

    partial_skims = omx.open_file(current_skims_path, mode="r")
    return "omx", None, partial_skims


def _load_and_prepare_master_skims_ds(
    all_skims_path: str,
    partial_skims_ds: Optional[xr.Dataset],
    canonical_order: List[Any],
    settings: Any,
    workspace: Any,
) -> Optional[xr.Dataset]:
    try:
        with xr.open_zarr(all_skims_path) as raw_skims_ds:
            skims_ds = raw_skims_ds.load().copy(deep=True)
        skims_ds, _ = zone_utils.normalize_0_based_and_flag_zarr_dataset(
            skims_ds, settings, workspace
        )
        logger.info(f"Opened Zarr skims file (read-only): {all_skims_path}")
        logger.info(
            f"DEBUG: Initial skims_ds otaz coords: {skims_ds.coords['otaz'].values[:5]}...{skims_ds.coords['otaz'].values[-5:]}"
        )

        if "original_zone_ids" not in skims_ds.attrs:
            if partial_skims_ds and "original_zone_ids" in partial_skims_ds.attrs:
                skims_ds.attrs["original_zone_ids"] = partial_skims_ds.attrs[
                    "original_zone_ids"
                ]
            else:
                skims_ds.attrs["original_zone_ids"] = list(canonical_order)
        return skims_ds
    except Exception as e:
        logger.error(f"Failed to open input Zarr skims file {all_skims_path}: {e}")
        if partial_skims_ds:
            partial_skims_ds.close()
        return None


def _reconstruct_and_postprocess_tnc_from_master_zarr(
    skims_ds: xr.Dataset,
    time_periods: List[str],
    settings: Any,
    consolidate_tnc_fleets: bool,
) -> None:
    logger.warning(
        "No new OMX skims found. Reconstructing TNC/RH trip counts from existing Zarr data for post-processing."
    )
    tnc_modes_for_postprocess = set()
    all_zarr_vars = list(skims_ds.data_vars)
    if consolidate_tnc_fleets:
        for target_mode in TNC_CONSOLIDATION_MAP.values():
            if f"{target_mode}_TRIPS" in all_zarr_vars:
                tnc_modes_for_postprocess.add(target_mode)
    else:
        for var_name in all_zarr_vars:
            if var_name.startswith("TNC_") and "_TRIPS" in var_name:
                mode = var_name.rsplit("_TRIPS", 1)[0]
                tnc_modes_for_postprocess.add(mode)

    completed_failed_dict_from_zarr = {}
    for mode in tnc_modes_for_postprocess:
        trips_key = f"{mode}_TRIPS"
        failures_key = f"{mode}_FAILURES"
        if trips_key in skims_ds.data_vars and failures_key in skims_ds.data_vars:
            completed_failed_dict_from_zarr[mode] = (
                np.nan_to_num(skims_ds[trips_key].data),
                np.nan_to_num(skims_ds[failures_key].data),
            )
    if tnc_modes_for_postprocess:
        _postprocess_tnc_zarr(
            skims_ds,
            time_periods,
            settings,
            completed_failed_dict_from_zarr,
            use_rh_modes=consolidate_tnc_fleets,
        )
    else:
        logger.info("No existing TNC/RH modes found in Zarr for post-processing.")


def _merge_non_tnc_measures_from_zarr(
    partial_skims_ds: xr.Dataset,
    skims_ds: xr.Dataset,
    time_periods: List[str],
    completed_failed_dict_all: Dict[str, Any],
) -> None:
    grouped_partial_sources = {}
    for var_name in partial_skims_ds.data_vars:
        parts = var_name.rsplit("_", 1)
        if len(parts) == 2:
            path, measure = parts
            if path.startswith("TNC_") or path.startswith("RH_"):
                continue
            if path not in completed_failed_dict_all:
                continue
            grouped_partial_sources[(path, measure)] = [var_name]

    for (path, measure), _var_names in grouped_partial_sources.items():
        completed_3d, failed_3d = completed_failed_dict_all[path]
        _merge_one_zarr_measure_from_zarr(
            partial_skims_ds,
            skims_ds,
            path,
            measure,
            time_periods,
            completed_3d,
            failed_3d,
        )


def _should_skip_omx_path_for_generic_merge(
    omx_key: str, path: str, consolidate_tnc_fleets: bool
) -> bool:
    if not (path.startswith("TNC_") or path.startswith("RH_")):
        return False

    if consolidate_tnc_fleets:
        is_provider_path = path.startswith("TNC_")
        is_consolidated_path = path in TNC_CONSOLIDATION_MAP.values()
        if is_provider_path or is_consolidated_path:
            logger.debug(
                f"Skipping OMX key {omx_key} for generic merge: Path {path} is TNC/RH and handled elsewhere (consolidation enabled)."
            )
            return True
    elif path.startswith("TNC_"):
        logger.debug(
            f"Skipping OMX key {omx_key} for generic merge: Path {path} is TNC and handled elsewhere (consolidation disabled)."
        )
        return True
    return False


def _group_non_tnc_omx_sources(
    partial_skims: Any,
    time_periods: List[str],
    consolidate_tnc_fleets: bool,
) -> Dict[Tuple[str, str], List[str]]:
    grouped_partial_sources: Dict[Tuple[str, str], List[str]] = {}
    for omx_key in partial_skims.list_matrices():
        parts = omx_key.split("__")
        if len(parts) != 2:
            continue
        measure_part, period = parts[0], parts[1]
        if period not in time_periods:
            continue

        path_parts, measure = _parse_rightmost_measure(
            measure_part.split("_"), _GENERIC_OMX_MEASURES
        )
        path = "_".join(path_parts) if path_parts else None
        if not path or measure is None:
            logger.debug(f"Could not parse path/measure from key {omx_key}. Skipping.")
            continue

        if _should_skip_omx_path_for_generic_merge(
            omx_key, path, consolidate_tnc_fleets
        ):
            continue

        grouped_partial_sources.setdefault((path, measure), []).append(omx_key)
    return grouped_partial_sources


def _merge_non_tnc_measures_from_omx(
    partial_skims: Any,
    skims_ds: xr.Dataset,
    time_periods: List[str],
    consolidate_tnc_fleets: bool,
    completed_failed_dict_all: Dict[str, Any],
) -> None:
    grouped_partial_sources = _group_non_tnc_omx_sources(
        partial_skims, time_periods, consolidate_tnc_fleets
    )
    logger.info(
        f"Merging {len(grouped_partial_sources)} non-TNC/non-RH (path, measure) groups from partial skims."
    )

    processed_non_tnc_vars = set()
    for (path, measure), _omx_keys in grouped_partial_sources.items():
        if path not in completed_failed_dict_all:
            logger.warning(
                f"Completed/failed trip data for path {path} not found. Skipping merge for measure {measure}."
            )
            continue

        completed_3d, failed_3d = completed_failed_dict_all[path]
        processed_path, processed_measure = _merge_one_zarr_measure(
            partial_skims,
            skims_ds,
            path,
            measure,
            time_periods,
            completed_3d,
            failed_3d,
        )
        if processed_path is not None:
            processed_non_tnc_vars.add(f"{processed_path}_{processed_measure}")

    logger.info(
        f"Completed merging {len(processed_non_tnc_vars)} non-TNC/non-RH variables."
    )


def _write_and_verify_merged_zarr_skims(
    skims_ds: xr.Dataset,
    target_skims_path: str,
    output_skims_path: Optional[str],
    canonical_order: List[Any],
) -> bool:
    logger.info(f"Started writing updated zarr skims to {target_skims_path}")
    try:
        expected_coords = np.arange(len(canonical_order))
        otaz_corrected = False
        dtaz_corrected = False
        if not np.array_equal(skims_ds.coords["otaz"].values, expected_coords):
            logger.warning(
                "Correcting 'otaz' coordinates to be 0-based before final save."
            )
            skims_ds = skims_ds.assign_coords(otaz=expected_coords)
            otaz_corrected = True
        if not np.array_equal(skims_ds.coords["dtaz"].values, expected_coords):
            logger.warning(
                "Correcting 'dtaz' coordinates to be 0-based before final save."
            )
            skims_ds = skims_ds.assign_coords(dtaz=expected_coords)
            dtaz_corrected = True

        skims_ds = zone_utils.normalize_dimension_coords_for_zarr(skims_ds)
        for var_name in skims_ds.variables:
            skims_ds[var_name].encoding = {}

        target_parent = os.path.dirname(target_skims_path)
        if target_parent:
            os.makedirs(target_parent, exist_ok=True)
        if output_skims_path is not None and os.path.exists(target_skims_path):
            shutil.rmtree(target_skims_path)

        skims_ds.to_zarr(target_skims_path, mode="w", consolidated=False, zarr_format=2)

        import zarr

        zgroup = zarr.open_group(target_skims_path, mode="a")
        if otaz_corrected and "otaz" in zgroup:
            zgroup["otaz"][:] = expected_coords
        if dtaz_corrected and "dtaz" in zgroup:
            zgroup["dtaz"][:] = expected_coords
        zgroup.attrs["ZARR_WRITE_TIME"] = time.time()
        if "original_zone_ids" in skims_ds.attrs:
            zgroup.attrs["original_zone_ids"] = skims_ds.attrs["original_zone_ids"]

        zarr.consolidate_metadata(target_skims_path)
        logger.info("Completed writing zarr skims successfully.")
        merge_successful = True
    except Exception as e:
        logger.error(f"FAILED to write updated zarr skims to {target_skims_path}: {e}")
        raise

    try:
        with xr.open_zarr(target_skims_path) as verified_ds:
            logger.info(
                f"DEBUG: VERIFICATION - Zarr file on disk ({target_skims_path}) otaz coords: {verified_ds.coords['otaz'].values[:5]}...{verified_ds.coords['otaz'].values[-5:]}"
            )
            if "time_period" in verified_ds.coords:
                logger.debug(
                    "Post-write time_period coords for %s: %s",
                    target_skims_path,
                    list(verified_ds.coords["time_period"].values),
                )
    except Exception as e:
        logger.error(
            f"DEBUG: VERIFICATION FAILED - Could not open or verify {target_skims_path}: {e}"
        )

    return merge_successful


def _apply_partial_beam_skim_updates(
    input_format: str,
    partial_skims_ds: Optional[xr.Dataset],
    partialSkims,
    skims_ds: xr.Dataset,
    timePeriods: List[str],
    settings,
    consolidate_tnc_fleets: bool,
    previous_weight: float,
    get_setting_fn: Callable[..., Any] = get_setting,
) -> None:
    completed_failed_dict_all = {}
    if input_format == "zarr" and partial_skims_ds:
        completed_failed_dict_all = _accumulate_all_completed_failed_trips_zarr(
            partial_skims_ds, timePeriods
        )
    elif input_format == "omx" and partialSkims:
        completed_failed_dict_all = _accumulate_all_completed_failed_trips(
            partialSkims, timePeriods
        )

    for path, (completed, failed) in completed_failed_dict_all.items():
        _merge_zarr_trip_counts(
            skims_ds, path, completed, failed, previous_weight=previous_weight
        )

    if partial_skims_ds or partialSkims:
        try:
            if input_format == "zarr":
                tnc_modes_processed = _process_all_tnc_logic_zarr(
                    partial_skims_ds,
                    skims_ds,
                    timePeriods,
                    settings,
                    completed_failed_dict_all,
                    previous_weight=previous_weight,
                    get_setting_fn=get_setting_fn,
                )
            else:
                tnc_modes_processed = _process_all_tnc_logic(
                    partialSkims,
                    skims_ds,
                    timePeriods,
                    settings,
                    completed_failed_dict_all,
                    get_setting_fn=get_setting_fn,
                )
            logger.info(f"Processed TNC modes: {tnc_modes_processed}")
            logger.info(
                f"DEBUG: skims_ds otaz coords after TNC logic: {skims_ds.coords['otaz'].values[:5]}...{skims_ds.coords['otaz'].values[-5:]}"
            )
        except Exception as e:
            logger.error(f"Failed to process TNC logic: {e}", exc_info=True)
    else:
        _reconstruct_and_postprocess_tnc_from_master_zarr(
            skims_ds, timePeriods, settings, consolidate_tnc_fleets
        )

    if input_format == "zarr" and partial_skims_ds:
        _merge_non_tnc_measures_from_zarr(
            partial_skims_ds, skims_ds, timePeriods, completed_failed_dict_all
        )
    elif input_format == "omx" and partialSkims:
        _merge_non_tnc_measures_from_omx(
            partialSkims,
            skims_ds,
            timePeriods,
            consolidate_tnc_fleets,
            completed_failed_dict_all,
        )


def _close_skim_datasets(partialSkims, partial_skims_ds, skims_ds) -> None:
    if partialSkims:
        partialSkims.close()
    if partial_skims_ds:
        partial_skims_ds.close()
    if skims_ds:
        skims_ds.close()


def _merge_beam_skims_to_zarr(
    all_skims_path,
    iteration_skims_path,
    beam_output_dir,
    settings,
    workspace,
    override=None,
    output_skims_path: Optional[str] = None,
    verify_skim_zone_order_fn: Optional[Callable[..., List[Any]]] = None,
    get_setting_fn: Callable[..., Any] = get_setting,
) -> Optional[str]:
    """
    Merges current BEAM OMX skims into a Zarr skims store in a single pass.
    Reads from ``all_skims_path`` strictly read-only and writes the merged
    result to ``output_skims_path`` (or ``all_skims_path`` when omitted).
    """
    target_skims_path = (
        output_skims_path if output_skims_path is not None else all_skims_path
    )
    logger.info(
        f"Starting merge of current BEAM skims from {all_skims_path} into Zarr at {target_skims_path}"
    )

    canonical_zones_df = zone_utils.load_canonical_zones(settings, workspace)
    canonical_order = canonical_zones_df.index.tolist()
    logger.info(f"DEBUG: Loaded canonical order for {len(canonical_order)} zones.")

    current_skims_path = (
        (iteration_skims_path if os.path.exists(iteration_skims_path) else None)
        if override is None
        else override
    )
    if current_skims_path is None or not os.path.exists(current_skims_path):
        logger.warning("No current skims found. Skipping merge.")
        return None

    if verify_skim_zone_order_fn is None:
        from pilates.beam.postprocessor import verify_skim_zone_order

        verify_skim_zone_order_fn = verify_skim_zone_order

    canonical_order = verify_skim_zone_order_fn(settings, current_skims_path, workspace)

    input_format, partial_skims_ds, partialSkims = _open_partial_beam_skims(
        current_skims_path, settings, get_setting_fn=get_setting_fn
    )

    skims_ds = _load_and_prepare_master_skims_ds(
        all_skims_path, partial_skims_ds, canonical_order, settings, workspace
    )
    if skims_ds is None:
        return None

    timePeriods = get_setting_fn(settings, "shared.skims.periods")
    consolidate_tnc_fleets = get_setting_fn(settings, "consolidate_tnc_fleets", True)
    previous_weight = get_setting_fn(settings, "beam.skim_previous_weight", 0.9)

    _apply_partial_beam_skim_updates(
        input_format,
        partial_skims_ds,
        partialSkims,
        skims_ds,
        timePeriods,
        settings,
        consolidate_tnc_fleets,
        previous_weight,
        get_setting_fn=get_setting_fn,
    )

    logger.info("Applying transit mode availability rules.")
    _handle_transit_mode_availability(skims_ds, timePeriods)

    logger.info("Copying skims for unobserved modes based on mapping.")
    mapping = get_setting_fn(
        settings,
        "beam.ridehail_path_map",
        {"SOV": ["SOVTOLL", "HOV2", "HOV2TOLL", "HOV3", "HOV3TOLL"]},
    )
    copy_skims_for_unobserved_modes(mapping, skims_ds)

    merge_successful = _write_and_verify_merged_zarr_skims(
        skims_ds, target_skims_path, output_skims_path, canonical_order
    )

    _close_skim_datasets(partialSkims, partial_skims_ds, skims_ds)

    if merge_successful:
        return target_skims_path
    logger.warning("No new OMX skims found, returning None.")
    return None
