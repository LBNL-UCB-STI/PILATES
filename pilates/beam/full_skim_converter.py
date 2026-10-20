"""Consist-tracked converter from raw BEAM FullSkimsCreatorApp OMX to ActivitySim OMX.

Transforms a raw OMX produced by BEAM's ``FullSkimsCreatorApp`` into a complete,
ActivitySim-compatible OMX skim file that plugs seamlessly into both ActivitySim
and PILATES ``BeamPostprocessor`` iterations:

1. **Intrazonal Diagonal Imputation**:
   Populates zero diagonals (`o == d`) on base distance and travel-time matrices
   (`SOV_DIST`, `SOV_TIME`, `SOVTOLL_DIST`, `SOVTOLL_TIME`, `WALK_DIST`,
   `WALK_TIME`, `BIKE_DIST`, `BIKE_TIME`) using half the mean of the `k` nearest
   positive off-diagonal neighbors per origin zone, preventing division-by-zero
   in ActivitySim (`DDIST / od_skims['DIST']`) and zero-duration intrazonal trips.
2. **Transit Scaling (`TRANSIT_SCALE_FACTOR = 100.0`)**:
   Multiplies transit time and drive-access distance measures (`TOTIVT`,
   `KEYIVT`, `FERRYIVT`, `IVT`, `IWAIT`, `XWAIT`, `WAIT`, `WACC`, `WAUX`,
   `WEGR`, `DTIM`, `DDIST`) by `100.0` while keeping `FAR` (already in cents),
   `BOARDS`, `TRIPS`, and `FAILURES` unscaled (`1.0x`).
3. **Transit `*_WAIT` Aliases**:
   Synthesizes `{prefix}_WAIT__{tp} = {prefix}_IWAIT__{tp} + {prefix}_XWAIT__{tp}`
   for all transit submodes.
4. **Express Bus (`EXP`) Unavailable**:
   Populates all `WLK_EXP_WLK_*`, `DRV_EXP_WLK_*`, and `WLK_EXP_DRV_*` matrices
   with `0.0` (`float32`), setting `walk_express_available = False` and
   `drive_express_available = False` in ActivitySim.
5. **General Walk-Transit (`WLK_TRN_WLK_*`) Synthesis**:
   Synthesizes `WLK_TRN_WLK_{IVT,TOTIVT,IWAIT,XWAIT,WAIT,WACC,WAUX,WEGR,FAR,BOARDS,TRIPS,FAILURES}`
   by selecting the walk-transit submode with minimum positive `TOTIVT` per OD pair.
6. **HOV & Toll Highway Copying (`HOV2`, `HOV3`, `HOV2TOLL`, `HOV3TOLL`)**:
   Copies `SOV_*` into `HOV2_*` and `HOV3_*`, and `SOVTOLL_*` into `HOV2TOLL_*`
   and `HOV3TOLL_*`.
7. **Ridehail (`RH_SOLO_*`, `RH_POOLED_*`, `RH_SHARED_*`) from `SOVTOLL`**:
   Sets solo ridehail travel time and distance equal to `SOVTOLL_TIME` and
   `SOVTOLL_DIST`, and sets pooled/shared ridehail travel time and distance to
   `SOVTOLL` multiplied by configurable multipliers (default `1.15x`).
8. **Static 2D Distance Matrices & Dual Zone Lookups**:
   Populates 2D `DIST`, `DISTBIKE`, `DISTWALK` and ensures both `/lookup/TAZ`
   and `/lookup/zone_id` exist.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from pydantic import BaseModel, Field

import consist as cr
from consist import Tracker

from pilates.beam.skim_provenance import (
    compute_omx_logical_matrices_sha256,
    read_skim_provenance_sidecar,
    skim_provenance_sidecar_path,
    write_skim_provenance_sidecar,
)

logger = logging.getLogger(__name__)

FULL_SKIM_CONVERTER_VERSION = "pilates.full_skim_converter.v1"
CONVERTED_ATTR_KEY = "pilates_full_skim_converted"

DEFAULT_TIME_PERIODS: tuple[str, ...] = ("EA", "AM", "MD", "PM", "EV")

TRANSIT_SUBMODES: tuple[str, ...] = ("COM", "HVY", "LOC", "LRF", "EXP")
TRANSIT_PREFIXES: tuple[str, ...] = tuple(
    f"{acc}_{sub}_{egr}"
    for acc, egr in (("WLK", "WLK"), ("DRV", "WLK"), ("WLK", "DRV"))
    for sub in TRANSIT_SUBMODES
) + ("WLK_TRN_WLK",)

SCALED_TRANSIT_MEASURES: frozenset[str] = frozenset(
    {
        "TOTIVT",
        "KEYIVT",
        "FERRYIVT",
        "IVT",
        "IWAIT",
        "XWAIT",
        "WAIT",
        "WACC",
        "WAUX",
        "WEGR",
        "DTIM",
        "DDIST",
    }
)

UNSCALED_TRANSIT_MEASURES: frozenset[str] = frozenset(
    {
        "FAR",
        "BOARDS",
        "TRIPS",
        "FAILURES",
    }
)

WLK_EXP_WLK_MEASURES: tuple[str, ...] = (
    "BOARDS",
    "FAR",
    "FERRYIVT",
    "IWAIT",
    "KEYIVT",
    "TOTIVT",
    "WAIT",
    "WACC",
    "WAUX",
    "WEGR",
    "XWAIT",
    "TRIPS",
    "FAILURES",
)

DRV_EXP_MEASURES: tuple[str, ...] = (
    "BOARDS",
    "DDIST",
    "DTIM",
    "FAR",
    "FERRYIVT",
    "IWAIT",
    "KEYIVT",
    "TOTIVT",
    "WAIT",
    "WACC",
    "WAUX",
    "WEGR",
    "XWAIT",
    "TRIPS",
    "FAILURES",
)


class FullSkimConversionConfig(BaseModel):
    """Configuration for converting raw BEAM full-skim OMX to ActivitySim OMX."""

    transit_scale_factor: float = Field(
        100.0,
        gt=0.0,
        description=(
            "Scale factor applied to transit time (minutes) and drive-access "
            "distance (miles) matrices to match ActivitySim TRANSIT_SCALE_FACTOR."
        ),
    )
    impute_intrazonal_from_nearest_neighbors: bool = Field(
        True,
        description=(
            "Impute zero diagonals (o == d) on base distance and travel time "
            "matrices using 0.5 * mean of the k nearest positive off-diagonal neighbors."
        ),
    )
    intrazonal_nearest_neighbors_k: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of nearest positive off-diagonal neighbors for intrazonal imputation.",
    )
    intrazonal_factor: float = Field(
        0.5,
        gt=0.0,
        le=1.0,
        description="Multiplier on nearest-neighbor mean for intrazonal diagonal imputation.",
    )
    copy_hov_from_sov: bool = Field(
        True,
        description=(
            "Populate HOV2_* and HOV3_* from SOV_*, and HOV2TOLL_* and "
            "HOV3TOLL_* from SOVTOLL_*."
        ),
    )
    disable_express_bus: bool = Field(
        True,
        description=(
            "Populate WLK_EXP_WLK_*, DRV_EXP_WLK_*, and WLK_EXP_DRV_* matrices "
            "with zeros so Express Bus is marked unavailable in ActivitySim."
        ),
    )
    synthesize_wlk_trn_wlk: bool = Field(
        True,
        description=(
            "Synthesize general walk-transit WLK_TRN_WLK_* matrices across "
            "available WLK_*_WLK submodes by selecting minimum positive TOTIVT."
        ),
    )
    synthesize_ridehail_from_sovtoll: bool = Field(
        True,
        description=(
            "Populate RH_SOLO_*, RH_POOLED_*, and RH_SHARED_* matrices from "
            "SOVTOLL_TIME and SOVTOLL_DIST."
        ),
    )
    rh_pooled_time_multiplier: float = Field(
        1.15,
        ge=1.0,
        description=(
            "Multiplier applied to SOVTOLL_TIME for pooled/shared ridehail "
            "in-vehicle travel time (RH_POOLED_TOTIVT and RH_SHARED_TOTIVT)."
        ),
    )
    rh_pooled_dist_multiplier: float = Field(
        1.15,
        ge=1.0,
        description=(
            "Multiplier applied to SOVTOLL_DIST for pooled/shared ridehail "
            "distance (RH_POOLED_DDIST and RH_SHARED_DDIST)."
        ),
    )
    rh_solo_wait_minutes: float = Field(
        6.0,
        ge=0.0,
        description="Default initial wait time in minutes for solo ridehail (RH_SOLO_WAIT).",
    )
    rh_pooled_wait_minutes: float = Field(
        8.0,
        ge=0.0,
        description="Default initial wait time in minutes for pooled ridehail (RH_POOLED_WAIT).",
    )
    rh_rejection_prob: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Default ridehail rejection probability (REJECTIONPROB).",
    )
    rh_solo_base_fare_cents: float = Field(
        220.0,
        ge=0.0,
        description="Base fare in cents for solo ridehail FAR matrix.",
    )
    rh_solo_per_mile_cents: float = Field(
        135.0,
        ge=0.0,
        description="Per-mile fare in cents for solo ridehail FAR matrix.",
    )
    rh_solo_per_minute_cents: float = Field(
        24.0,
        ge=0.0,
        description="Per-minute fare in cents for solo ridehail FAR matrix.",
    )
    rh_pooled_fare_discount: float = Field(
        0.70,
        gt=0.0,
        le=1.0,
        description="Multiplier on solo ridehail fare for pooled/shared ridehail FAR matrix.",
    )
    synthesize_static_dist_matrices: bool = Field(
        True,
        description="Populate static 2D DIST, DISTBIKE, and DISTWALK matrices.",
    )
    reference_dist_period: str = Field(
        "MD",
        description="Preferred time period used to derive static 2D distance matrices.",
    )
    gzip_compression_level: int = Field(
        1,
        ge=0,
        le=9,
        description="HDF5 gzip compression level for output OMX datasets (0 disables compression).",
    )
    force: bool = Field(
        False,
        description="Allow converting an OMX file that already carries converted metadata.",
    )

    def to_consist_facet(self) -> dict[str, Any]:
        return {
            "converter_version": FULL_SKIM_CONVERTER_VERSION,
            "transit_scale_factor": self.transit_scale_factor,
            "impute_intrazonal_from_nearest_neighbors": (
                self.impute_intrazonal_from_nearest_neighbors
            ),
            "intrazonal_nearest_neighbors_k": self.intrazonal_nearest_neighbors_k,
            "intrazonal_factor": self.intrazonal_factor,
            "copy_hov_from_sov": self.copy_hov_from_sov,
            "disable_express_bus": self.disable_express_bus,
            "synthesize_wlk_trn_wlk": self.synthesize_wlk_trn_wlk,
            "synthesize_ridehail_from_sovtoll": self.synthesize_ridehail_from_sovtoll,
            "rh_pooled_time_multiplier": self.rh_pooled_time_multiplier,
            "rh_pooled_dist_multiplier": self.rh_pooled_dist_multiplier,
            "rh_solo_wait_minutes": self.rh_solo_wait_minutes,
            "rh_pooled_wait_minutes": self.rh_pooled_wait_minutes,
            "rh_rejection_prob": self.rh_rejection_prob,
            "rh_solo_base_fare_cents": self.rh_solo_base_fare_cents,
            "rh_solo_per_mile_cents": self.rh_solo_per_mile_cents,
            "rh_solo_per_minute_cents": self.rh_solo_per_minute_cents,
            "rh_pooled_fare_discount": self.rh_pooled_fare_discount,
            "synthesize_static_dist_matrices": self.synthesize_static_dist_matrices,
            "reference_dist_period": self.reference_dist_period,
        }


class FullSkimConversionReport(BaseModel):
    """Summary report returned after converting a raw BEAM OMX file."""

    input_omx_path: Path
    output_omx_path: Path
    provenance_sidecar_path: Path | None = None
    zone_count: int
    input_matrix_count: int
    output_matrix_count: int
    scaled_transit_matrix_count: int
    intrazonal_imputed_matrix_count: int
    time_periods: list[str]
    config_facet: dict[str, Any]


def _split_matrix_name(name: str) -> tuple[str, str, str | None]:
    """Parse ``<PREFIX>_<MEASURE>__<PERIOD>`` into ``(prefix, measure, period)``."""
    period: str | None = None
    base = name
    if "__" in name:
        base, period = name.rsplit("__", 1)
    if "_" in base:
        prefix, measure = base.rsplit("_", 1)
        return prefix, measure, period
    return base, "", period


def _should_scale_transit_matrix(prefix: str, measure: str) -> bool:
    return prefix in TRANSIT_PREFIXES and measure in SCALED_TRANSIT_MEASURES


def _sanitize_non_negative_float32(values: np.ndarray) -> np.ndarray:
    """Convert to float32 and replace NaN, Inf, and negative values with 0.0."""
    arr = np.asarray(values, dtype=np.float32)
    return np.maximum(
        np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), np.float32(0.0)
    )


def _impute_intrazonal_diagonal(
    matrix: np.ndarray,
    *,
    k: int = 3,
    factor: float = 0.5,
    fallback_value: float = 0.25,
) -> tuple[np.ndarray, bool]:
    """Impute any non-positive diagonal entries from the ``k`` nearest positive off-diagonal values."""
    out = _sanitize_non_negative_float32(matrix)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        return out, False
    n = out.shape[0]
    if n == 0:
        return out, False

    diag = np.diag(out)
    zero_diag_mask = ~(diag > 0)
    if not np.any(zero_diag_mask):
        return out, False

    if n == 1:
        out[0, 0] = np.float32(fallback_value)
        return out, True

    off_diag_positive = (out > 0) & (~np.eye(n, dtype=bool))
    masked = np.where(off_diag_positive, out, np.inf)
    k_eff = min(k, n - 1)
    nearest_k = np.partition(masked, k_eff - 1, axis=1)[:, :k_eff]
    finite_mask = np.isfinite(nearest_k)
    counts = np.sum(finite_mask, axis=1)
    sums = np.sum(np.where(finite_mask, nearest_k, 0.0), axis=1)

    positive_all = out[off_diag_positive]
    global_fallback = (
        float(np.median(positive_all)) * factor
        if positive_all.size > 0
        else fallback_value
    )
    imputed_diag = np.where(
        counts > 0,
        factor * (sums / np.maximum(counts, 1)),
        global_fallback,
    ).astype(np.float32)
    merged_diag = np.where(zero_diag_mask, imputed_diag, diag).astype(np.float32)
    np.fill_diagonal(out, merged_diag)
    return out, True


def _write_matrix(
    data_group: h5py.Group,
    name: str,
    values: np.ndarray,
    *,
    compression_level: int = 1,
    attrs: Mapping[str, Any] | None = None,
) -> None:
    arr = _sanitize_non_negative_float32(values)
    if name in data_group:
        del data_group[name]
    kwargs: dict[str, Any] = {}
    if compression_level > 0:
        kwargs["compression"] = "gzip"
        kwargs["compression_opts"] = int(compression_level)
    ds = data_group.create_dataset(name, data=arr, dtype="f4", **kwargs)
    if attrs:
        for k, v in attrs.items():
            try:
                ds.attrs[k] = v
            except Exception:
                pass
    ds.attrs["TITLE"] = np.bytes_("")
    ds.attrs["CLASS"] = np.bytes_("ARRAY")
    ds.attrs["VERSION"] = np.bytes_("2.4")
    ds.attrs["FLAVOR"] = np.bytes_("numpy")


def _infer_periods(matrix_names: list[str]) -> list[str]:
    found: set[str] = set()
    for name in matrix_names:
        _, _, period = _split_matrix_name(name)
        if period:
            found.add(period)
    ordered = [p for p in DEFAULT_TIME_PERIODS if p in found]
    for p in sorted(found):
        if p not in ordered:
            ordered.append(p)
    return ordered or list(DEFAULT_TIME_PERIODS)


def is_omx_already_converted(omx_path: Path | str) -> bool:
    """Return True if ``omx_path`` already carries the converter root attribute."""
    path = Path(omx_path)
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as h5f:
            val = h5f.attrs.get(CONVERTED_ATTR_KEY)
            if isinstance(val, (bytes, bytearray)):
                return val.decode("utf-8", errors="replace").lower() == "true"
            if hasattr(val, "item") and callable(val.item):
                return bool(val.item())
            return bool(val)
    except Exception:
        return False


def convert_beam_full_skims_to_activitysim_omx(
    input_omx_path: Path | str,
    output_omx_path: Path | str,
    *,
    config: FullSkimConversionConfig | None = None,
    emit_provenance_sidecar: bool = True,
    producer_step: str = "beam_full_skim_omx_converter",
    run_id: str | None = None,
    scenario: str | None = None,
    region: str | None = None,
    year: int | None = None,
    iteration: int | None = None,
    tracker_db_path: Path | str | None = None,
) -> FullSkimConversionReport:
    """Convert a raw BEAM FullSkimsCreatorApp OMX into an ActivitySim-ready OMX.

    Safe to call when ``input_omx_path`` and ``output_omx_path`` resolve to the
    same path (writes to a temporary file first and atomically replaces the
    destination). Refuses to re-convert an already-converted OMX unless
    ``config.force=True``.
    """
    cfg = config or FullSkimConversionConfig()
    in_path = Path(input_omx_path)
    out_path = Path(output_omx_path)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input OMX file does not exist: {in_path}")

    if is_omx_already_converted(in_path) and not cfg.force:
        raise ValueError(
            f"Input OMX file {in_path} already has {CONVERTED_ATTR_KEY}=True. "
            "Refusing to re-scale transit skims; pass force=True to override."
        )

    # Read upstream provenance sidecar if available so metadata is preserved
    upstream_sidecar = skim_provenance_sidecar_path(in_path)
    upstream_prov: dict[str, Any] | None = None
    if upstream_sidecar.is_file():
        try:
            upstream_prov = read_skim_provenance_sidecar(upstream_sidecar)
        except Exception as exc:
            logger.debug(
                "Unable to read upstream skim provenance sidecar %s: %s",
                upstream_sidecar,
                exc,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out_path = out_path.parent / f".tmp_{out_path.name}.{os.getpid()}"
    if tmp_out_path.exists():
        tmp_out_path.unlink()

    intrazonal_prefixes = {
        ("SOV", "DIST"),
        ("SOV", "TIME"),
        ("SOVTOLL", "DIST"),
        ("SOVTOLL", "TIME"),
        ("WALK", "DIST"),
        ("WALK", "TIME"),
        ("BIKE", "DIST"),
        ("BIKE", "TIME"),
    }

    scaled_count = 0
    imputed_count = 0

    try:
        with h5py.File(in_path, "r") as src_h5, h5py.File(tmp_out_path, "w") as dst_h5:
            # 1. Copy root attributes and stamp conversion metadata
            for attr_key, attr_val in src_h5.attrs.items():
                dst_h5.attrs[attr_key] = attr_val

            src_data = src_h5.get("data") or src_h5.get("DATA")
            if src_data is None or not hasattr(src_data, "keys"):
                raise ValueError(f"Input OMX file {in_path} has no /data group.")

            input_matrix_names = sorted(str(k) for k in src_data.keys())
            if not input_matrix_names:
                raise ValueError(f"Input OMX file {in_path} has 0 matrices in /data.")

            first_shape = src_data[input_matrix_names[0]].shape
            if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
                raise ValueError(
                    f"Expected square 2D skim matrices in {in_path}, got shape {first_shape}."
                )
            zone_count = int(first_shape[0])
            time_periods = _infer_periods(input_matrix_names)

            if "OMX_VERSION" not in dst_h5.attrs:
                dst_h5.attrs["OMX_VERSION"] = np.bytes_("0.2")
            dst_h5.attrs["SHAPE"] = np.array([zone_count, zone_count], dtype=np.int32)
            dst_h5.attrs[CONVERTED_ATTR_KEY] = np.int32(1)
            dst_h5.attrs["pilates_conversion_version"] = FULL_SKIM_CONVERTER_VERSION
            dst_h5.attrs["pilates_transit_scale_factor"] = float(
                cfg.transit_scale_factor
            )
            dst_h5.attrs["pilates_rh_pooled_time_multiplier"] = float(
                cfg.rh_pooled_time_multiplier
            )
            dst_h5.attrs["pilates_rh_pooled_dist_multiplier"] = float(
                cfg.rh_pooled_dist_multiplier
            )
            dst_h5.attrs["pilates_express_bus_disabled"] = np.int32(
                1 if cfg.disable_express_bus else 0
            )

            # 2. Populate /lookup with TAZ, zone_id, and zone_number
            dst_lookup = dst_h5.create_group("lookup")
            src_lookup = src_h5.get("lookup") or src_h5.get("LOOKUP")
            zone_vector: np.ndarray | None = None
            if src_lookup is not None and hasattr(src_lookup, "keys"):
                for lkey in ("zone_number", "TAZ", "zone_id"):
                    if lkey in src_lookup:
                        zone_vector = np.asarray(src_lookup[lkey][...])
                        break
                if zone_vector is None:
                    for lkey in src_lookup.keys():
                        zone_vector = np.asarray(src_lookup[lkey][...])
                        break
            if zone_vector is None:
                zone_vector = np.arange(1, zone_count + 1, dtype=np.int64)
            dst_lookup.create_dataset("TAZ", data=zone_vector)
            dst_lookup.create_dataset("zone_id", data=zone_vector)
            dst_lookup.create_dataset("zone_number", data=zone_vector.astype(np.int32))

            dst_data = dst_h5.create_group("data")
            zero_matrix = np.zeros((zone_count, zone_count), dtype=np.float32)

            # Build reference highway distance and time matrices across periods
            # so any isolated NaN/zero cells in individual periods or unrouted
            # long-distance walk/bike cells (>3 miles) can be backfilled cleanly.
            ref_sov_dist = zero_matrix.copy()
            ref_sov_time = zero_matrix.copy()
            pref_periods = [cfg.reference_dist_period, "MD", "AM"] + list(time_periods)
            for p_cand in pref_periods:
                d_key = f"SOV_DIST__{p_cand}"
                if d_key in src_data:
                    cand_d = _sanitize_non_negative_float32(src_data[d_key][...])
                    ref_sov_dist = np.where(
                        ref_sov_dist > 0, ref_sov_dist, cand_d
                    ).astype(np.float32)
                t_key = f"SOV_TIME__{p_cand}"
                if t_key in src_data:
                    cand_t = _sanitize_non_negative_float32(src_data[t_key][...])
                    ref_sov_time = np.where(
                        ref_sov_time > 0, ref_sov_time, cand_t
                    ).astype(np.float32)
            if cfg.impute_intrazonal_from_nearest_neighbors:
                ref_sov_dist, _ = _impute_intrazonal_diagonal(
                    ref_sov_dist,
                    k=cfg.intrazonal_nearest_neighbors_k,
                    factor=cfg.intrazonal_factor,
                )
                ref_sov_time, _ = _impute_intrazonal_diagonal(
                    ref_sov_time,
                    k=cfg.intrazonal_nearest_neighbors_k,
                    factor=cfg.intrazonal_factor,
                )

            # 3. Copy existing matrices, applying NaN sanitization, highway/walk backfill,
            #    intrazonal imputation, and transit scaling.
            for name in input_matrix_names:
                prefix, measure, _period = _split_matrix_name(name)
                if cfg.disable_express_bus and prefix in (
                    "WLK_EXP_WLK",
                    "DRV_EXP_WLK",
                    "WLK_EXP_DRV",
                ):
                    # Express bus will be written as zeros in step 5
                    continue

                raw_arr = _sanitize_non_negative_float32(src_data[name][...])

                # Backfill any missing (NaN/0) off-diagonal cells in highway/walk/bike distance & time
                off_diag_zero = (raw_arr <= 0) & (~np.eye(zone_count, dtype=bool))
                if prefix in ("SOV", "SOVTOLL", "WALK", "BIKE") and measure == "DIST":
                    if np.any(off_diag_zero) and np.any(ref_sov_dist > 0):
                        raw_arr = np.where(off_diag_zero, ref_sov_dist, raw_arr).astype(
                            np.float32
                        )
                elif prefix in ("SOV", "SOVTOLL") and measure == "TIME":
                    if np.any(off_diag_zero) and np.any(ref_sov_time > 0):
                        raw_arr = np.where(off_diag_zero, ref_sov_time, raw_arr).astype(
                            np.float32
                        )
                elif prefix == "WALK" and measure == "TIME":
                    if np.any(off_diag_zero) and np.any(ref_sov_dist > 0):
                        raw_arr = np.where(
                            off_diag_zero,
                            ref_sov_dist * np.float32(20.0),
                            raw_arr,
                        ).astype(np.float32)
                elif prefix == "BIKE" and measure == "TIME":
                    if np.any(off_diag_zero) and np.any(ref_sov_dist > 0):
                        raw_arr = np.where(
                            off_diag_zero,
                            ref_sov_dist * np.float32(6.0),
                            raw_arr,
                        ).astype(np.float32)

                if (
                    cfg.impute_intrazonal_from_nearest_neighbors
                    and (prefix, measure) in intrazonal_prefixes
                ):
                    raw_arr, did_impute = _impute_intrazonal_diagonal(
                        raw_arr,
                        k=cfg.intrazonal_nearest_neighbors_k,
                        factor=cfg.intrazonal_factor,
                    )
                    if did_impute:
                        imputed_count += 1

                if _should_scale_transit_matrix(prefix, measure):
                    raw_arr = (raw_arr * np.float32(cfg.transit_scale_factor)).astype(
                        np.float32
                    )
                    scaled_count += 1

                _write_matrix(
                    dst_data,
                    name,
                    raw_arr,
                    compression_level=cfg.gzip_compression_level,
                    attrs=dict(src_data[name].attrs),
                )

            # 4. Synthesize *_WAIT__<tp> = IWAIT + XWAIT for non-EXP transit modes
            for prefix in TRANSIT_PREFIXES:
                if prefix in (
                    "WLK_EXP_WLK",
                    "DRV_EXP_WLK",
                    "WLK_EXP_DRV",
                    "WLK_TRN_WLK",
                ):
                    continue
                for tp in time_periods:
                    wait_name = f"{prefix}_WAIT__{tp}"
                    iwait_name = f"{prefix}_IWAIT__{tp}"
                    xwait_name = f"{prefix}_XWAIT__{tp}"
                    if wait_name not in dst_data and (
                        iwait_name in dst_data or xwait_name in dst_data
                    ):
                        iwait = (
                            np.asarray(dst_data[iwait_name][...], dtype=np.float32)
                            if iwait_name in dst_data
                            else zero_matrix
                        )
                        xwait = (
                            np.asarray(dst_data[xwait_name][...], dtype=np.float32)
                            if xwait_name in dst_data
                            else zero_matrix
                        )
                        _write_matrix(
                            dst_data,
                            wait_name,
                            iwait + xwait,
                            compression_level=cfg.gzip_compression_level,
                        )

            # 5. Express Bus (EXP) -> all zeros (unavailable)
            if cfg.disable_express_bus:
                for tp in time_periods:
                    for measure in WLK_EXP_WLK_MEASURES:
                        _write_matrix(
                            dst_data,
                            f"WLK_EXP_WLK_{measure}__{tp}",
                            zero_matrix,
                            compression_level=cfg.gzip_compression_level,
                        )
                    for prefix in ("DRV_EXP_WLK", "WLK_EXP_DRV"):
                        for measure in DRV_EXP_MEASURES:
                            _write_matrix(
                                dst_data,
                                f"{prefix}_{measure}__{tp}",
                                zero_matrix,
                                compression_level=cfg.gzip_compression_level,
                            )

            # 6. Synthesize WLK_TRN_WLK_* across available WLK_*_WLK modes
            if cfg.synthesize_wlk_trn_wlk:
                candidate_wlk_prefixes = [
                    "WLK_LOC_WLK",
                    "WLK_LRF_WLK",
                    "WLK_HVY_WLK",
                    "WLK_COM_WLK",
                ]
                if not cfg.disable_express_bus:
                    candidate_wlk_prefixes.append("WLK_EXP_WLK")

                for tp in time_periods:
                    active_prefixes = [
                        p
                        for p in candidate_wlk_prefixes
                        if f"{p}_TOTIVT__{tp}" in dst_data
                    ]
                    if not active_prefixes:
                        continue

                    totivt_stack = np.stack(
                        [
                            np.asarray(
                                dst_data[f"{p}_TOTIVT__{tp}"][...], dtype=np.float32
                            )
                            for p in active_prefixes
                        ],
                        axis=0,
                    )
                    masked_ivt = np.where(totivt_stack > 0, totivt_stack, np.inf)
                    best_idx = np.argmin(masked_ivt, axis=0)
                    has_valid = np.any(totivt_stack > 0, axis=0)

                    def _select_from_best(measure_suffix: str) -> np.ndarray:
                        stack = np.stack(
                            [
                                (
                                    np.asarray(
                                        dst_data[f"{p}_{measure_suffix}__{tp}"][...],
                                        dtype=np.float32,
                                    )
                                    if f"{p}_{measure_suffix}__{tp}" in dst_data
                                    else zero_matrix
                                )
                                for p in active_prefixes
                            ],
                            axis=0,
                        )
                        chosen = np.take_along_axis(
                            stack, best_idx[np.newaxis, :, :], axis=0
                        )[0]
                        return np.where(has_valid, chosen, 0.0).astype(np.float32)

                    best_totivt = _select_from_best("TOTIVT")
                    best_iwait = _select_from_best("IWAIT")
                    best_xwait = _select_from_best("XWAIT")
                    best_wacc = _select_from_best("WACC")
                    best_waux = _select_from_best("WAUX")
                    best_wegr = _select_from_best("WEGR")
                    best_far = _select_from_best("FAR")
                    best_boards = _select_from_best("BOARDS")

                    trips_sum = np.sum(
                        [
                            (
                                np.asarray(
                                    dst_data[f"{p}_TRIPS__{tp}"][...], dtype=np.float32
                                )
                                if f"{p}_TRIPS__{tp}" in dst_data
                                else zero_matrix
                            )
                            for p in active_prefixes
                        ],
                        axis=0,
                    ).astype(np.float32)
                    failures_min = _select_from_best("FAILURES")

                    trn_matrices = {
                        "IVT": best_totivt,
                        "TOTIVT": best_totivt,
                        "IWAIT": best_iwait,
                        "XWAIT": best_xwait,
                        "WAIT": (best_iwait + best_xwait).astype(np.float32),
                        "WACC": best_wacc,
                        "WAUX": best_waux,
                        "WEGR": best_wegr,
                        "FAR": best_far,
                        "BOARDS": best_boards,
                        "TRIPS": trips_sum,
                        "FAILURES": failures_min,
                    }
                    for measure, arr in trn_matrices.items():
                        _write_matrix(
                            dst_data,
                            f"WLK_TRN_WLK_{measure}__{tp}",
                            arr,
                            compression_level=cfg.gzip_compression_level,
                        )

            # 7. Copy HOV2 / HOV3 from SOV and HOV2TOLL / HOV3TOLL from SOVTOLL
            if cfg.copy_hov_from_sov:
                for tp in time_periods:
                    sov_time = (
                        np.asarray(dst_data[f"SOV_TIME__{tp}"][...], dtype=np.float32)
                        if f"SOV_TIME__{tp}" in dst_data
                        else zero_matrix
                    )
                    sov_dist = (
                        np.asarray(dst_data[f"SOV_DIST__{tp}"][...], dtype=np.float32)
                        if f"SOV_DIST__{tp}" in dst_data
                        else zero_matrix
                    )
                    sov_btoll = (
                        np.asarray(dst_data[f"SOV_BTOLL__{tp}"][...], dtype=np.float32)
                        if f"SOV_BTOLL__{tp}" in dst_data
                        else zero_matrix
                    )
                    sov_vtoll = (
                        np.asarray(dst_data[f"SOV_VTOLL__{tp}"][...], dtype=np.float32)
                        if f"SOV_VTOLL__{tp}" in dst_data
                        else zero_matrix
                    )
                    sov_trips = (
                        np.asarray(dst_data[f"SOV_TRIPS__{tp}"][...], dtype=np.float32)
                        if f"SOV_TRIPS__{tp}" in dst_data
                        else np.where(sov_time > 0, 1.0, 0.0).astype(np.float32)
                    )
                    sov_failures = (
                        np.asarray(
                            dst_data[f"SOV_FAILURES__{tp}"][...], dtype=np.float32
                        )
                        if f"SOV_FAILURES__{tp}" in dst_data
                        else zero_matrix
                    )

                    sovtoll_time = (
                        np.asarray(
                            dst_data[f"SOVTOLL_TIME__{tp}"][...], dtype=np.float32
                        )
                        if f"SOVTOLL_TIME__{tp}" in dst_data
                        else sov_time
                    )
                    sovtoll_dist = (
                        np.asarray(
                            dst_data[f"SOVTOLL_DIST__{tp}"][...], dtype=np.float32
                        )
                        if f"SOVTOLL_DIST__{tp}" in dst_data
                        else sov_dist
                    )
                    sovtoll_btoll = (
                        np.asarray(
                            dst_data[f"SOVTOLL_BTOLL__{tp}"][...], dtype=np.float32
                        )
                        if f"SOVTOLL_BTOLL__{tp}" in dst_data
                        else zero_matrix
                    )
                    sovtoll_vtoll = (
                        np.asarray(
                            dst_data[f"SOVTOLL_VTOLL__{tp}"][...], dtype=np.float32
                        )
                        if f"SOVTOLL_VTOLL__{tp}" in dst_data
                        else sov_btoll
                    )
                    sovtoll_trips = (
                        np.asarray(
                            dst_data[f"SOVTOLL_TRIPS__{tp}"][...], dtype=np.float32
                        )
                        if f"SOVTOLL_TRIPS__{tp}" in dst_data
                        else sov_trips
                    )
                    sovtoll_failures = (
                        np.asarray(
                            dst_data[f"SOVTOLL_FAILURES__{tp}"][...], dtype=np.float32
                        )
                        if f"SOVTOLL_FAILURES__{tp}" in dst_data
                        else sov_failures
                    )

                    if f"SOVTOLL_TRIPS__{tp}" not in dst_data:
                        _write_matrix(
                            dst_data,
                            f"SOVTOLL_TRIPS__{tp}",
                            sovtoll_trips,
                            compression_level=cfg.gzip_compression_level,
                        )
                    if f"SOVTOLL_FAILURES__{tp}" not in dst_data:
                        _write_matrix(
                            dst_data,
                            f"SOVTOLL_FAILURES__{tp}",
                            sovtoll_failures,
                            compression_level=cfg.gzip_compression_level,
                        )

                    for hov_prefix in ("HOV2", "HOV3"):
                        for measure, arr in (
                            ("TIME", sov_time),
                            ("DIST", sov_dist),
                            ("BTOLL", sov_btoll),
                            ("VTOLL", sov_vtoll),
                            ("TRIPS", sov_trips),
                            ("FAILURES", sov_failures),
                        ):
                            _write_matrix(
                                dst_data,
                                f"{hov_prefix}_{measure}__{tp}",
                                arr,
                                compression_level=cfg.gzip_compression_level,
                            )

                    for hovtoll_prefix in ("HOV2TOLL", "HOV3TOLL"):
                        for measure, arr in (
                            ("TIME", sovtoll_time),
                            ("DIST", sovtoll_dist),
                            ("BTOLL", sovtoll_btoll),
                            ("VTOLL", sovtoll_vtoll),
                            ("TRIPS", sovtoll_trips),
                            ("FAILURES", sovtoll_failures),
                        ):
                            _write_matrix(
                                dst_data,
                                f"{hovtoll_prefix}_{measure}__{tp}",
                                arr,
                                compression_level=cfg.gzip_compression_level,
                            )

            # 8. Synthesize Ridehail (RH_SOLO, RH_POOLED, RH_SHARED) from SOVTOLL
            if cfg.synthesize_ridehail_from_sovtoll:
                for tp in time_periods:
                    base_time = (
                        np.asarray(
                            dst_data[f"SOVTOLL_TIME__{tp}"][...], dtype=np.float32
                        )
                        if f"SOVTOLL_TIME__{tp}" in dst_data
                        else (
                            np.asarray(
                                dst_data[f"SOV_TIME__{tp}"][...], dtype=np.float32
                            )
                            if f"SOV_TIME__{tp}" in dst_data
                            else zero_matrix
                        )
                    )
                    base_dist = (
                        np.asarray(
                            dst_data[f"SOVTOLL_DIST__{tp}"][...], dtype=np.float32
                        )
                        if f"SOVTOLL_DIST__{tp}" in dst_data
                        else (
                            np.asarray(
                                dst_data[f"SOV_DIST__{tp}"][...], dtype=np.float32
                            )
                            if f"SOV_DIST__{tp}" in dst_data
                            else zero_matrix
                        )
                    )
                    base_trips = (
                        np.asarray(dst_data[f"SOV_TRIPS__{tp}"][...], dtype=np.float32)
                        if f"SOV_TRIPS__{tp}" in dst_data
                        else np.where(base_time > 0, 1.0, 0.0).astype(np.float32)
                    )

                    valid_od = (base_time > 0) | (base_dist > 0)
                    solo_wait = np.where(
                        valid_od, cfg.rh_solo_wait_minutes, 0.0
                    ).astype(np.float32)
                    pooled_wait = np.where(
                        valid_od, cfg.rh_pooled_wait_minutes, 0.0
                    ).astype(np.float32)
                    rejection_prob = np.full(
                        (zone_count, zone_count),
                        cfg.rh_rejection_prob,
                        dtype=np.float32,
                    )

                    solo_fare = np.where(
                        valid_od,
                        cfg.rh_solo_base_fare_cents
                        + cfg.rh_solo_per_mile_cents * base_dist
                        + cfg.rh_solo_per_minute_cents * base_time,
                        0.0,
                    ).astype(np.float32)
                    pooled_fare = (solo_fare * cfg.rh_pooled_fare_discount).astype(
                        np.float32
                    )

                    pooled_time = (base_time * cfg.rh_pooled_time_multiplier).astype(
                        np.float32
                    )
                    pooled_dist = (base_dist * cfg.rh_pooled_dist_multiplier).astype(
                        np.float32
                    )

                    solo_matrices = {
                        "TOTIVT": base_time,
                        "TIME": base_time,
                        "DDIST": base_dist,
                        "DIST": base_dist,
                        "WAIT": solo_wait,
                        "IWAIT": solo_wait,
                        "REJECTIONPROB": rejection_prob,
                        "FAR": solo_fare,
                        "TRIPS": base_trips,
                        "FAILURES": zero_matrix,
                    }
                    for measure, arr in solo_matrices.items():
                        _write_matrix(
                            dst_data,
                            f"RH_SOLO_{measure}__{tp}",
                            arr,
                            compression_level=cfg.gzip_compression_level,
                        )

                    pooled_matrices = {
                        "TOTIVT": pooled_time,
                        "TIME": pooled_time,
                        "DDIST": pooled_dist,
                        "DIST": pooled_dist,
                        "WAIT": pooled_wait,
                        "IWAIT": pooled_wait,
                        "REJECTIONPROB": rejection_prob,
                        "FAR": pooled_fare,
                        "TRIPS": base_trips,
                        "FAILURES": zero_matrix,
                    }
                    for rh_prefix in ("RH_POOLED", "RH_SHARED"):
                        for measure, arr in pooled_matrices.items():
                            _write_matrix(
                                dst_data,
                                f"{rh_prefix}_{measure}__{tp}",
                                arr,
                                compression_level=cfg.gzip_compression_level,
                            )

            # 9. Synthesize static 2D distance matrices (DIST, DISTBIKE, DISTWALK)
            if cfg.synthesize_static_dist_matrices:
                ref_tp = (
                    cfg.reference_dist_period
                    if cfg.reference_dist_period in time_periods
                    else (
                        "MD"
                        if "MD" in time_periods
                        else ("AM" if "AM" in time_periods else time_periods[0])
                    )
                )
                sov_dist_ref = (
                    np.asarray(dst_data[f"SOV_DIST__{ref_tp}"][...], dtype=np.float32)
                    if f"SOV_DIST__{ref_tp}" in dst_data
                    else zero_matrix
                )
                walk_dist_ref = (
                    np.asarray(dst_data[f"WALK_DIST__{ref_tp}"][...], dtype=np.float32)
                    if f"WALK_DIST__{ref_tp}" in dst_data
                    else sov_dist_ref
                )
                bike_dist_ref = (
                    np.asarray(dst_data[f"BIKE_DIST__{ref_tp}"][...], dtype=np.float32)
                    if f"BIKE_DIST__{ref_tp}" in dst_data
                    else walk_dist_ref
                )

                _write_matrix(
                    dst_data,
                    "DIST",
                    sov_dist_ref,
                    compression_level=cfg.gzip_compression_level,
                )
                _write_matrix(
                    dst_data,
                    "DISTWALK",
                    walk_dist_ref,
                    compression_level=cfg.gzip_compression_level,
                )
                _write_matrix(
                    dst_data,
                    "DISTBIKE",
                    bike_dist_ref,
                    compression_level=cfg.gzip_compression_level,
                )

            # 10. Compute canonical logical matrices SHA-256 and stamp root attribute
            logical_matrices_hex = compute_omx_logical_matrices_sha256(dst_h5)
            logical_matrices_identity = (
                f"sha256:logical_matrices:{logical_matrices_hex}"
            )
            dst_h5.attrs["logical_matrices_sha256"] = logical_matrices_identity

            output_matrix_count = len(dst_data.keys())

        tmp_out_path.replace(out_path)
    except Exception:
        if tmp_out_path.exists():
            tmp_out_path.unlink()
        raise

    sidecar_path: Path | None = None
    if emit_provenance_sidecar:
        upstream_producer = (
            upstream_prov.get("producer")
            if isinstance(upstream_prov, Mapping)
            and isinstance(upstream_prov.get("producer"), Mapping)
            else {}
        )
        resolved_run_id = run_id or (
            str(upstream_producer["run_id"])
            if upstream_producer.get("run_id") is not None
            else None
        )
        resolved_scenario = scenario or (
            str(upstream_producer["scenario"])
            if upstream_producer.get("scenario") is not None
            else None
        )
        resolved_region = region or (
            str(upstream_producer["region"])
            if upstream_producer.get("region") is not None
            else None
        )
        resolved_year = (
            year
            if year is not None
            else (
                int(upstream_producer["year"])
                if upstream_producer.get("year") is not None
                else None
            )
        )
        resolved_iteration = (
            iteration
            if iteration is not None
            else (
                int(upstream_producer["iteration"])
                if upstream_producer.get("iteration") is not None
                else None
            )
        )
        resolved_tracker_db = tracker_db_path or upstream_producer.get(
            "tracker_db_path"
        )

        extra_metadata: dict[str, Any] = {
            "pilates_full_skim_converted": True,
            "conversion_config": cfg.to_consist_facet(),
            "pilates_converted_at": datetime.now(timezone.utc).isoformat(),
            "logical_matrices_sha256": logical_matrices_identity,
            "input_matrix_count": len(input_matrix_names),
            "scaled_transit_matrix_count": scaled_count,
            "intrazonal_imputed_matrix_count": imputed_count,
        }
        if isinstance(upstream_prov, Mapping):
            extra_metadata["raw_beam_omx_identity"] = upstream_prov.get("identity")
            extra_metadata["raw_beam_producer"] = dict(upstream_producer)

        sidecar_path, _ = write_skim_provenance_sidecar(
            out_path,
            producer_step=producer_step,
            run_id=resolved_run_id,
            scenario=resolved_scenario,
            region=resolved_region,
            year=resolved_year,
            iteration=resolved_iteration,
            tracker_db_path=resolved_tracker_db,
            extra_metadata=extra_metadata,
        )

    logger.info(
        "[FullSkimConverter] Converted %s (%d matrices) -> %s (%d matrices, transit_scale=%s, rh_pooled_mult=%s, logical_sha256=%s)",
        in_path,
        len(input_matrix_names),
        out_path,
        output_matrix_count,
        cfg.transit_scale_factor,
        cfg.rh_pooled_time_multiplier,
        logical_matrices_identity,
    )

    return FullSkimConversionReport(
        input_omx_path=in_path,
        output_omx_path=out_path,
        provenance_sidecar_path=sidecar_path,
        zone_count=zone_count,
        input_matrix_count=len(input_matrix_names),
        output_matrix_count=output_matrix_count,
        scaled_transit_matrix_count=scaled_count,
        intrazonal_imputed_matrix_count=imputed_count,
        time_periods=time_periods,
        config_facet=cfg.to_consist_facet(),
    )


def run_tracked_full_skim_conversion(
    input_omx_path: Path | str,
    output_omx_path: Path | str,
    *,
    config: FullSkimConversionConfig | None = None,
    tracker_db_path: Path | str | None = None,
    run_dir: Path | str | None = None,
    run_id: str | None = None,
    scenario: str | None = None,
    region: str | None = None,
    year: int | None = None,
    iteration: int | None = None,
) -> FullSkimConversionReport:
    """Execute OMX conversion inside a Consist-tracked run with full artifact lineage."""
    cfg = config or FullSkimConversionConfig()
    in_path = Path(input_omx_path).resolve()
    out_path = Path(output_omx_path).resolve()

    resolved_run_dir = (
        Path(run_dir).resolve() if run_dir is not None else out_path.parent.resolve()
    )
    resolved_db_path = (
        Path(tracker_db_path).resolve()
        if tracker_db_path is not None
        else (resolved_run_dir / ".consist" / "provenance.duckdb")
    )
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    resolved_run_id = run_id or f"beam_full_skim_omx_converter_{timestamp}"

    tracker = Tracker(
        run_dir=resolved_run_dir,
        db_path=resolved_db_path,
    )
    with tracker.start_run(
        run_id=resolved_run_id,
        model="beam_full_skim_omx_converter",
        config=cfg.to_consist_facet(),
        year=year,
        iteration=iteration,
        tags=["skims", "activitysim", "omx_conversion"],
    ):
        cr.log_artifact(
            str(in_path),
            key="beam_full_skims",
            direction="input",
        )
        report = convert_beam_full_skims_to_activitysim_omx(
            in_path,
            out_path,
            config=cfg,
            emit_provenance_sidecar=True,
            producer_step="beam_full_skim_omx_converter",
            run_id=resolved_run_id,
            scenario=scenario,
            region=region,
            year=year,
            iteration=iteration,
            tracker_db_path=resolved_db_path,
        )
        sidecar_meta: dict[str, Any] = {}
        if (
            report.provenance_sidecar_path is not None
            and report.provenance_sidecar_path.is_file()
        ):
            prov_payload = read_skim_provenance_sidecar(report.provenance_sidecar_path)
            sidecar_meta["skim_provenance"] = prov_payload
            identity = prov_payload.get("identity")
            if isinstance(identity, str) and identity.startswith("sha256:file:"):
                sidecar_meta["admission_file_identity"] = identity

        cr.log_artifact(
            str(out_path),
            key="final_skims_omx",
            direction="output",
            **sidecar_meta,
        )
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a raw BEAM FullSkimsCreatorApp OMX file into an "
            "ActivitySim-compatible OMX file with Consist provenance tracking."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to raw BEAM FullSkimsCreatorApp .omx file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Path to write converted ActivitySim-ready .omx file.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Optional Consist DuckDB path (defaults to <output_dir>/.consist/provenance.duckdb).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional Consist run ID.",
    )
    parser.add_argument(
        "--transit-scale-factor",
        type=float,
        default=100.0,
        help="Transit scale factor (default: 100.0).",
    )
    parser.add_argument(
        "--rh-pooled-time-multiplier",
        type=float,
        default=1.15,
        help="Multiplier on SOVTOLL_TIME for pooled/shared ridehail time (default: 1.15).",
    )
    parser.add_argument(
        "--rh-pooled-dist-multiplier",
        type=float,
        default=1.15,
        help="Multiplier on SOVTOLL_DIST for pooled/shared ridehail distance (default: 1.15).",
    )
    parser.add_argument(
        "--rh-solo-wait-minutes",
        type=float,
        default=6.0,
        help="Default wait time in minutes for solo ridehail (default: 6.0).",
    )
    parser.add_argument(
        "--rh-pooled-wait-minutes",
        type=float,
        default=8.0,
        help="Default wait time in minutes for pooled ridehail (default: 8.0).",
    )
    parser.add_argument(
        "--rh-rejection-prob",
        type=float,
        default=0.0,
        help="Default ridehail rejection probability (default: 0.0).",
    )
    parser.add_argument(
        "--keep-express-bus",
        action="store_true",
        help="Do not zero out Express Bus (*_EXP_*) matrices.",
    )
    parser.add_argument(
        "--no-intrazonal-imputation",
        action="store_true",
        help="Disable nearest-neighbor intrazonal diagonal imputation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow re-converting an OMX file that already has converted metadata.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
    )
    args = _build_arg_parser().parse_args(argv)
    config = FullSkimConversionConfig(
        transit_scale_factor=args.transit_scale_factor,
        impute_intrazonal_from_nearest_neighbors=not args.no_intrazonal_imputation,
        disable_express_bus=not args.keep_express_bus,
        rh_pooled_time_multiplier=args.rh_pooled_time_multiplier,
        rh_pooled_dist_multiplier=args.rh_pooled_dist_multiplier,
        rh_solo_wait_minutes=args.rh_solo_wait_minutes,
        rh_pooled_wait_minutes=args.rh_pooled_wait_minutes,
        rh_rejection_prob=args.rh_rejection_prob,
        force=args.force,
    )
    report = run_tracked_full_skim_conversion(
        args.input,
        args.output,
        config=config,
        tracker_db_path=args.db_path,
        run_id=args.run_id,
    )
    print(
        f"Converted {report.input_omx_path} ({report.input_matrix_count} matrices) -> "
        f"{report.output_omx_path} ({report.output_matrix_count} matrices, "
        f"{report.zone_count} zones, {report.scaled_transit_matrix_count} transit matrices scaled, "
        f"{report.intrazonal_imputed_matrix_count} diagonals imputed)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
