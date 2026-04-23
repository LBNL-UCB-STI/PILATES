from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from pilates.activitysim.outputs import has_asim_run_marker
from pilates.beam.config_hocon import (
    BeamConfigHoconError,
    beam_config_env_overrides,
    beam_primary_config_path,
    resolve_beam_config_value,
)
from pilates.generic.records import FileRecord, RecordStore
from pilates.utils.beam_warmstart import resolve_initial_linkstats_path
from pilates.utils.coupler_helpers import resolve_existing_path
from pilates.utils.io import locate_beam_file
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
)

logger = logging.getLogger(__name__)


class BeamDataHelper:
    """
    Centralizes logic for reading, cleaning, and standardizing BEAM input data.
    """

    RENAMES = {
        "plans": {
            "tripId": "trip_id",
            "tripid": "trip_id",
            "personId": "person_id",
            "personid": "person_id",
            "vehicleId": "vehicle_id",
            "vehicleid": "vehicle_id",
            "legVehicleIds": "leg_vehicle_ids",
            "legvehicleids": "leg_vehicle_ids",
        },
        "households": {"VEHICL": "cars", "auto_ownership": "cars"},
        "persons": {},
    }

    DTYPES = {
        "plans": {
            "trip_id": pd.Int64Dtype(),
            "person_id": pd.Int64Dtype(),
            "planindex": pd.Int64Dtype(),
        },
        "households": {
            "household_id": pd.Int64Dtype(),
            "cars": pd.Int64Dtype(),
            "auto_ownership": pd.Int64Dtype(),
            "VEHICL": pd.Int64Dtype(),
        },
        "persons": {
            "household_id": pd.Int64Dtype(),
            "person_id": pd.Int64Dtype(),
            "age": pd.Int64Dtype(),
            "sex": pd.Int64Dtype(),
        },
    }

    PLAN_DEFAULTS = {
        "planindex": 0,
        "planselected": False,
        "planscore": 0.0,
        "legtraveltime": 0.0,
        "legroutetype": "",
        "legroutestartlink": 0,
        "legrouteendlink": 0,
        "legroutetraveltime": 0.0,
        "legroutedistance": 0.0,
        "legroutelinks": "",
    }

    @classmethod
    def read_and_clean(
        cls,
        file_path: str,
        table_type: str,
        file_format: str,
    ) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_format == "parquet":
            df = pd.read_parquet(file_path)
        elif file_format == "csv":
            dtype_map = cls.DTYPES.get(table_type, {})
            df = pd.read_csv(file_path, dtype=dtype_map)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        rename_map = cls.RENAMES.get(table_type, {})
        df = df.rename(columns=rename_map)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        if table_type == "plans":
            for col, default_val in cls.PLAN_DEFAULTS.items():
                if col not in df.columns:
                    df[col] = default_val

        index_map = {"households": "household_id", "persons": "person_id"}
        index_col = index_map.get(table_type)
        if index_col and index_col in df.columns:
            df.set_index(index_col, inplace=True, drop=True)

        return df


def copy_vehicles_from_atlas(
    *,
    workspace: Any,
    state: Any,
    resolve_beam_exchange_scenario_folder_fn: Callable[[Any], str],
    source_path: Optional[str] = None,
    preferred_format: Optional[str] = None,
) -> Optional[FileRecord]:
    beam_scenario_folder = resolve_beam_exchange_scenario_folder_fn(workspace)
    os.makedirs(beam_scenario_folder, exist_ok=True)
    beam_vehicles_format = _beam_vehicle_file_format(preferred_format)
    beam_vehicles_path = _beam_vehicle_path(
        beam_scenario_folder,
        file_format=beam_vehicles_format,
    )

    atlas_candidates = [source_path] if source_path else []
    if not atlas_candidates:
        atlas_output_data_dir = workspace.get_atlas_output_dir()
        atlas_candidates.extend(
            [
                os.path.join(atlas_output_data_dir, f"vehicles2_{state.forecast_year}.csv"),
                os.path.join(atlas_output_data_dir, f"vehicles2_{state.forecast_year - 1}.csv"),
            ]
        )

    atlas_vehicle_file_loc = None
    for candidate in atlas_candidates:
        if not candidate:
            continue
        resolved_candidate = resolve_existing_path(
            candidate,
            workspace=workspace,
            materialize_from_archive=True,
        )
        if resolved_candidate is not None and os.path.exists(resolved_candidate):
            atlas_vehicle_file_loc = resolved_candidate
            break

    if atlas_vehicle_file_loc is None:
        logger.warning(
            "ATLAS vehicles2 file not found for BEAM input: %s",
            atlas_candidates[0] if atlas_candidates else None,
        )
        return None

    logger.info(
        "Copying atlas vehicles2 file from %s to %s",
        atlas_vehicle_file_loc,
        beam_vehicles_path,
    )

    df = _read_vehicle_table(atlas_vehicle_file_loc)
    df = _normalize_beam_vehicle_columns(df)
    _write_vehicle_table(
        df,
        beam_vehicles_path,
        file_format=beam_vehicles_format,
    )
    return FileRecord(
        file_path=beam_vehicles_path,
        description="BEAM vehicles input derived from ATLAS vehicles2",
        short_name="vehicles_beam_in",
        year=getattr(state, "forecast_year", None),
        iteration=getattr(state, "current_inner_iter", None),
    )


def validate_population_consistency(
    *,
    workspace: Any,
    settings: Any,
    resolve_beam_exchange_scenario_folder_fn: Callable[[Any], str],
) -> None:
    beam_scenario_folder = resolve_beam_exchange_scenario_folder_fn(workspace)
    file_format = settings.activitysim.file_format if settings.activitysim else "csv"
    households_path = locate_beam_file(beam_scenario_folder, "households", file_format)
    vehicles_path = _find_existing_vehicle_path(
        beam_scenario_folder,
        preferred_format=file_format,
    )

    if (
        households_path is None
        or not os.path.exists(households_path)
        or vehicles_path is None
        or not os.path.exists(vehicles_path)
    ):
        return

    households = BeamDataHelper.read_and_clean(households_path, "households", file_format)
    vehicles = _read_vehicle_table(vehicles_path)
    report = summarize_population_consistency(households=households, vehicles=vehicles)
    category_report = summarize_vehicle_category_consistency(
        households=households,
        vehicles=vehicles,
        settings=settings,
        workspace=workspace,
    )

    if report["missing_vehicle_households"]:
        raise ValueError(
            "BEAM staged vehicles reference households that are absent from staged "
            "households. This can cause deterministic household vehicle assignment to "
            "misbehave. "
            f"missing_households={report['missing_vehicle_households']} "
            f"sample_missing_households={report['sample_missing_vehicle_households']} "
            f"households_path={households_path} vehicles_path={vehicles_path}"
        )

    if report["households_with_car_shortfall"]:
        raise ValueError(
            "BEAM staged households require more cars than are present in the staged "
            "vehicles file. This causes BEAM to fall back to sampled vehicles. "
            f"households_with_car_shortfall={report['households_with_car_shortfall']} "
            f"sample_household_shortfalls={report['sample_household_shortfalls']} "
            f"households_path={households_path} vehicles_path={vehicles_path}"
        )

    if report["households_with_vehicle_count_mismatch"]:
        logger.warning(
            "BEAM staged household/vehicle counts differ even though no shortfall was "
            "detected. sample_household_mismatches=%s households_path=%s vehicles_path=%s",
            report["sample_household_mismatches"],
            households_path,
            vehicles_path,
        )

    if category_report["status"] == "ok":
        if category_report["households_with_car_category_shortfall"]:
            logger.warning(
                "BEAM staged households have fewer staged Car-category vehicles than "
                "required by households.cars. This may trigger BEAM fallback sampling. "
                "households_with_car_category_shortfall=%s unmatched_vehicle_types=%s "
                "non_car_vehicle_rows=%s sample_household_car_shortfalls=%s "
                "sample_unmatched_vehicle_types=%s households_path=%s vehicles_path=%s "
                "vehicle_types_path=%s",
                category_report["households_with_car_category_shortfall"],
                category_report["unmatched_vehicle_types"],
                category_report["non_car_vehicle_rows"],
                category_report["sample_household_car_shortfalls"],
                category_report["sample_unmatched_vehicle_types"],
                households_path,
                vehicles_path,
                category_report["vehicle_types_path"],
            )
        else:
            logger.info(
                "Validated BEAM staged Car-category vehicles: households_with_car_category_shortfall=%s "
                "unmatched_vehicle_types=%s non_car_vehicle_rows=%s vehicle_types_path=%s",
                category_report["households_with_car_category_shortfall"],
                category_report["unmatched_vehicle_types"],
                category_report["non_car_vehicle_rows"],
                category_report["vehicle_types_path"],
            )
    elif category_report["status"] == "missing":
        logger.warning(
            "Skipping advisory BEAM Car-category vehicle validation because the staged "
            "vehicle types file could not be resolved. reason=%s",
            category_report["reason"],
        )
    elif category_report["status"] == "error":
        logger.warning(
            "Skipping advisory BEAM Car-category vehicle validation because vehicle "
            "types loading failed. reason=%s",
            category_report["reason"],
        )

    logger.info(
        "Validated BEAM staged household vehicles: households=%s vehicle_rows=%s "
        "duplicate_vehicle_ids=%s households_with_vehicle_count_mismatch=%s",
        report["total_households"],
        report["total_vehicle_rows"],
        report["duplicate_vehicle_ids"],
        report["households_with_vehicle_count_mismatch"],
    )


def _beam_vehicle_file_format(preferred_format: Optional[str]) -> str:
    return "parquet" if preferred_format == "parquet" else "csv.gz"


def _beam_vehicle_path(beam_scenario_folder: str, *, file_format: str) -> str:
    extension = "parquet" if file_format == "parquet" else "csv.gz"
    return os.path.join(beam_scenario_folder, f"vehicles.{extension}")


def _find_existing_vehicle_path(
    beam_scenario_folder: str,
    *,
    preferred_format: Optional[str],
) -> Optional[str]:
    candidate_formats = []
    normalized_preferred = _beam_vehicle_file_format(preferred_format)
    candidate_formats.append(normalized_preferred)
    for fallback in ("parquet", "csv.gz"):
        if fallback not in candidate_formats:
            candidate_formats.append(fallback)
    for candidate_format in candidate_formats:
        candidate_path = _beam_vehicle_path(
            beam_scenario_folder,
            file_format=candidate_format,
        )
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def _read_vehicle_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    compression = "gzip" if path.endswith(".gz") else None
    return pd.read_csv(path, compression=compression)


def _write_vehicle_table(df: pd.DataFrame, path: str, *, file_format: str) -> None:
    if file_format == "parquet":
        df.to_parquet(path, index=False)
        return
    df.to_csv(path, compression="gzip", index=False)


def _normalize_beam_vehicle_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    rename_map = {}
    if "vehicleId" not in normalized.columns and "vehicle_id" in normalized.columns:
        rename_map["vehicle_id"] = "vehicleId"
    if "householdId" not in normalized.columns and "household_id" in normalized.columns:
        rename_map["household_id"] = "householdId"
    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    required_columns = ("vehicleId", "vehicleTypeId", "householdId")
    missing_columns = [col for col in required_columns if col not in normalized.columns]
    if missing_columns:
        raise ValueError(
            "ATLAS vehicles2 input is missing required columns for BEAM staging: "
            f"{missing_columns}; available columns: {normalized.columns.tolist()}"
        )

    household_ids = _coerce_beam_vehicle_integer_column(
        normalized["householdId"],
        column_name="householdId",
    )
    source_vehicle_ids = _coerce_beam_vehicle_integer_column(
        normalized["vehicleId"],
        column_name="vehicleId",
    )
    duplicate_source_vehicle_mask = source_vehicle_ids.duplicated(keep=False)
    if duplicate_source_vehicle_mask.any():
        logger.warning(
            "ATLAS vehicles2 input used non-global vehicle_id values. PILATES "
            "is replacing BEAM vehicleId with a namespaced string identifier and "
            "preserving the original ATLAS vehicle_id in sourceVehicleId. "
            "duplicate_rows=%s duplicate_ids=%s",
            int(duplicate_source_vehicle_mask.sum()),
            int(source_vehicle_ids[duplicate_source_vehicle_mask].nunique()),
        )

    vehicle_ids = _synthesize_namespaced_vehicle_ids(
        household_ids=household_ids,
        source_vehicle_ids=source_vehicle_ids,
    )
    if vehicle_ids.duplicated().any():
        raise ValueError(
            "Failed to synthesize globally unique BEAM vehicle IDs from ATLAS "
            "household_id and vehicle_id."
        )

    normalized["vehicleId"] = vehicle_ids.astype(str)
    normalized["sourceVehicleId"] = source_vehicle_ids
    normalized["householdId"] = household_ids

    normalized["vehicleTypeId"] = normalized["vehicleTypeId"].astype(str)
    return normalized


def summarize_population_consistency(
    *,
    households: pd.DataFrame,
    vehicles: pd.DataFrame,
) -> Dict[str, Any]:
    household_ids = _coerce_index_to_int64(
        households.index,
        context="staged households index",
    )

    household_cars = (
        _coerce_beam_vehicle_integer_column(
            households["cars"] if "cars" in households.columns else pd.Series(0, index=households.index),
            column_name="cars",
        )
        .reindex(households.index, fill_value=0)
    )
    household_cars.index = household_ids.index

    vehicle_household_col = None
    for candidate in ("householdId", "household_id"):
        if candidate in vehicles.columns:
            vehicle_household_col = candidate
            break
    if vehicle_household_col is None:
        raise ValueError(
            "BEAM staged vehicles input is missing a household reference column. "
            f"Expected one of ['householdId', 'household_id'], found {vehicles.columns.tolist()}"
        )

    vehicle_household_ids = _coerce_beam_vehicle_integer_column(
        vehicles[vehicle_household_col],
        column_name=vehicle_household_col,
    )
    vehicle_counts = (
        pd.DataFrame(
            {
                "household_id": vehicle_household_ids,
                "vehicle_row_count": 1,
            }
        )
        .groupby("household_id", sort=False)["vehicle_row_count"]
        .sum()
        .astype("int64")
    )
    household_counts = (
        pd.DataFrame(
            {
                "household_id": household_ids.values,
                "cars": household_cars.values,
            }
        )
        .set_index("household_id")
        .join(vehicle_counts, how="left")
        .fillna({"vehicle_row_count": 0})
    )
    household_counts["vehicle_row_count"] = household_counts["vehicle_row_count"].astype(
        "int64"
    )

    mismatch_mask = household_counts["cars"] != household_counts["vehicle_row_count"]
    shortfall_mask = household_counts["cars"] > household_counts["vehicle_row_count"]
    missing_vehicle_households = pd.Index(vehicle_counts.index).difference(
        pd.Index(household_counts.index)
    )

    duplicate_vehicle_ids = 0
    if "vehicleId" in vehicles.columns:
        duplicate_vehicle_ids = int(
            vehicles["vehicleId"].astype(str).duplicated(keep=False).sum()
        )
    elif "vehicle_id" in vehicles.columns:
        duplicate_vehicle_ids = int(
            vehicles["vehicle_id"].astype(str).duplicated(keep=False).sum()
        )

    mismatch_examples = (
        household_counts.loc[mismatch_mask, ["cars", "vehicle_row_count"]]
        .head(10)
        .reset_index()
        .rename(columns={"index": "household_id"})
        .to_dict(orient="records")
    )
    shortfall_examples = (
        household_counts.loc[shortfall_mask, ["cars", "vehicle_row_count"]]
        .head(10)
        .reset_index()
        .rename(columns={"index": "household_id"})
        .to_dict(orient="records")
    )

    return {
        "total_households": int(len(household_counts)),
        "total_vehicle_rows": int(len(vehicles)),
        "duplicate_vehicle_ids": duplicate_vehicle_ids,
        "households_with_vehicle_count_mismatch": int(mismatch_mask.sum()),
        "households_with_car_shortfall": int(shortfall_mask.sum()),
        "missing_vehicle_households": int(len(missing_vehicle_households)),
        "sample_household_mismatches": mismatch_examples,
        "sample_household_shortfalls": shortfall_examples,
        "sample_missing_vehicle_households": missing_vehicle_households.tolist()[:10],
    }


def summarize_vehicle_category_consistency(
    *,
    households: pd.DataFrame,
    vehicles: pd.DataFrame,
    settings: Any,
    workspace: Any,
) -> Dict[str, Any]:
    vehicle_types_path = _resolve_beam_vehicle_types_path(settings=settings, workspace=workspace)
    if vehicle_types_path is None:
        return {
            "status": "missing",
            "reason": "vehicleTypesFilePath unavailable from staged BEAM config",
        }

    if not os.path.exists(vehicle_types_path):
        return {
            "status": "missing",
            "reason": f"resolved vehicle types file is missing: {vehicle_types_path}",
        }

    try:
        vehicle_type_categories = _load_vehicle_type_categories(vehicle_types_path)
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    household_ids = _coerce_index_to_int64(
        households.index,
        context="staged households index",
    )
    household_cars = (
        _coerce_beam_vehicle_integer_column(
            households["cars"] if "cars" in households.columns else pd.Series(0, index=households.index),
            column_name="cars",
        )
        .reindex(households.index, fill_value=0)
    )

    vehicle_household_col = None
    for candidate in ("householdId", "household_id"):
        if candidate in vehicles.columns:
            vehicle_household_col = candidate
            break
    if vehicle_household_col is None:
        raise ValueError(
            "BEAM staged vehicles input is missing a household reference column. "
            f"Expected one of ['householdId', 'household_id'], found {vehicles.columns.tolist()}"
        )

    vehicle_household_ids = _coerce_beam_vehicle_integer_column(
        vehicles[vehicle_household_col],
        column_name=vehicle_household_col,
    )
    vehicle_type_ids = vehicles["vehicleTypeId"].astype(str)
    categories = vehicle_type_ids.map(vehicle_type_categories)
    car_mask = categories == "Car"

    car_counts = (
        pd.DataFrame(
            {
                "household_id": vehicle_household_ids[car_mask],
                "car_vehicle_row_count": 1,
            }
        )
        .groupby("household_id", sort=False)["car_vehicle_row_count"]
        .sum()
        .astype("int64")
    )
    household_car_counts = (
        pd.DataFrame(
            {
                "household_id": household_ids.values,
                "cars": household_cars.values,
            }
        )
        .set_index("household_id")
        .join(car_counts, how="left")
        .fillna({"car_vehicle_row_count": 0})
    )
    household_car_counts["car_vehicle_row_count"] = household_car_counts[
        "car_vehicle_row_count"
    ].astype("int64")
    shortfall_mask = (
        household_car_counts["cars"] > household_car_counts["car_vehicle_row_count"]
    )
    shortfall_examples = (
        household_car_counts.loc[shortfall_mask, ["cars", "car_vehicle_row_count"]]
        .head(10)
        .reset_index()
        .rename(columns={"index": "household_id"})
        .to_dict(orient="records")
    )
    unmatched_vehicle_type_ids = pd.Index(vehicle_type_ids[categories.isna()].unique())

    return {
        "status": "ok",
        "vehicle_types_path": vehicle_types_path,
        "households_with_car_category_shortfall": int(shortfall_mask.sum()),
        "sample_household_car_shortfalls": shortfall_examples,
        "unmatched_vehicle_types": int(len(unmatched_vehicle_type_ids)),
        "sample_unmatched_vehicle_types": unmatched_vehicle_type_ids.tolist()[:10],
        "non_car_vehicle_rows": int((~car_mask & categories.notna()).sum()),
    }


def _coerce_beam_vehicle_integer_column(
    values: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or ((numeric % 1) != 0).any():
        raise ValueError(
            f"BEAM staging column '{column_name}' must be integer-valued."
        )
    return numeric.astype("int64")


def _coerce_index_to_int64(index: pd.Index, *, context: str) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(index, index=index), errors="coerce")
    invalid_mask = numeric.isna() | ((numeric % 1) != 0)
    if invalid_mask.any():
        raise ValueError(
            f"{context} must contain integer-valued household ids for BEAM validation."
        )
    return numeric.astype("int64")


def _synthesize_namespaced_vehicle_ids(
    *,
    household_ids: pd.Series,
    source_vehicle_ids: pd.Series,
) -> pd.Series:
    synthesis = pd.DataFrame(
        {
            "householdId": household_ids.astype("int64"),
            "sourceVehicleId": source_vehicle_ids.astype("int64"),
        }
    )
    duplicate_within_household = synthesis.groupby(
        ["householdId", "sourceVehicleId"], sort=False
    ).cumcount()
    base_ids = (
        "hh-"
        + synthesis["householdId"].astype(str)
        + "-veh-"
        + synthesis["sourceVehicleId"].astype(str)
    )
    duplicate_mask = duplicate_within_household > 0
    if duplicate_mask.any():
        base_ids = base_ids.where(
            ~duplicate_mask,
            base_ids + "-dup-" + (duplicate_within_household + 1).astype(str),
        )
    return base_ids.astype(str)


def _resolve_beam_vehicle_types_path(*, settings: Any, workspace: Any) -> Optional[str]:
    try:
        config_path = beam_primary_config_path(settings, workspace=workspace)
    except Exception:
        return None
    if not config_path.exists():
        return None
    try:
        resolved = resolve_beam_config_value(
            config_path,
            key="beam.agentsim.agents.vehicles.vehicleTypesFilePath",
            env_overrides=beam_config_env_overrides(settings, workspace=workspace),
        )
    except BeamConfigHoconError:
        return None
    if resolved is None:
        return None
    return os.fspath(resolved)


def _load_vehicle_type_categories(vehicle_types_path: str) -> Dict[str, str]:
    vehicle_types = pd.read_csv(vehicle_types_path)
    if "vehicleTypeId" not in vehicle_types.columns or "vehicleCategory" not in vehicle_types.columns:
        raise ValueError(
            "BEAM vehicle types file must contain vehicleTypeId and vehicleCategory columns."
        )
    deduped = (
        vehicle_types[["vehicleTypeId", "vehicleCategory"]]
        .dropna(subset=["vehicleTypeId"])
        .drop_duplicates(subset=["vehicleTypeId"], keep="last")
    )
    return dict(
        zip(
            deduped["vehicleTypeId"].astype(str),
            deduped["vehicleCategory"].astype(str),
        )
    )


def format_specific_output_records(
    *,
    file_stem: str,
    file_path: str,
    description_prefix: str,
    state: Any,
) -> List[FileRecord]:
    short_name_map = {
        "plans": BEAM_PLANS_IN,
        "households": BEAM_HOUSEHOLDS_IN,
        "persons": BEAM_PERSONS_IN,
    }
    short_name = short_name_map.get(file_stem, f"{file_stem}_beam_in")
    return [
        FileRecord(
            file_path=file_path,
            description=description_prefix,
            short_name=short_name,
            year=getattr(state, "current_year", None),
            iteration=getattr(state, "current_inner_iter", None),
        )
    ]


def copy_with_compression_asim_file_to_beam(
    *,
    asim_file_path: str,
    beam_file_name: str,
    file_format: str,
    beam_scenario_folder: str,
    state: Any,
) -> List[FileRecord]:
    beam_file_path = locate_beam_file(beam_scenario_folder, beam_file_name, file_format)
    logger.info(
        "Copying asim file %s to beam input scenario file %s",
        asim_file_path,
        beam_file_path,
    )

    if not os.path.exists(asim_file_path):
        logger.error("ActivitySim output file does not exist: %s", asim_file_path)
        return [
            FileRecord(
                file_path=beam_file_path,
                description=f"Missing BEAM input file: {beam_file_name}",
                short_name=f"{beam_file_name}_beam_in_missing",
                year=getattr(state, "current_year", None),
                iteration=getattr(state, "current_inner_iter", None),
            )
        ]

    table_type = "plans" if "plans" in beam_file_name else beam_file_name
    df = BeamDataHelper.read_and_clean(asim_file_path, table_type, file_format)

    if file_format == "parquet":
        df.to_parquet(beam_file_path, index=True)
    else:
        df.to_csv(beam_file_path, compression="gzip", index=False)

    return format_specific_output_records(
        file_stem=beam_file_name,
        file_path=beam_file_path,
        description_prefix=f"Copied from ActivitySim output: {beam_file_name}",
        state=state,
    )


def copy_initial_asim_files(
    *,
    asim_file_paths: Dict[str, Tuple[Optional[str], Optional[FileRecord]]],
    file_format: str,
    workspace: Any,
    resolve_beam_exchange_scenario_folder_fn: Callable[[Any], str],
    copy_with_compression_asim_file_to_beam_fn: Callable[..., List[FileRecord]],
) -> List[FileRecord]:
    record_list: List[FileRecord] = []
    asim_to_beam_mapping = [
        ("beam_plans", "plans"),
        ("households", "households"),
        ("persons", "persons"),
    ]
    beam_scenario_folder = resolve_beam_exchange_scenario_folder_fn(workspace)
    os.makedirs(beam_scenario_folder, exist_ok=True)

    for asim_name, beam_name in asim_to_beam_mapping:
        asim_file_path, _asim_file_record = asim_file_paths.get(asim_name, (None, None))
        if asim_file_path:
            records = copy_with_compression_asim_file_to_beam_fn(
                asim_file_path=asim_file_path,
                beam_file_name=beam_name,
                file_format=file_format,
                beam_scenario_folder=beam_scenario_folder,
            )
            if records:
                record_list.extend(records)
        else:
            logger.warning("ActivitySim output file not found: %s", asim_name)

    return record_list


def merge_replanned_asim_files(
    *,
    asim_file_paths: Dict[str, Tuple[Optional[str], Optional[FileRecord]]],
    file_format: str,
    workspace: Any,
    resolve_beam_exchange_scenario_folder_fn: Callable[[Any], str],
    format_specific_output_records_fn: Callable[..., List[FileRecord]],
) -> List[FileRecord]:
    logger.info("Merging asim outputs with existing beam input scenario files.")
    beam_scenario_folder = resolve_beam_exchange_scenario_folder_fn(workspace)
    os.makedirs(beam_scenario_folder, exist_ok=True)

    asim_plans_path, _asim_plans_rec = asim_file_paths.get("beam_plans", (None, None))
    asim_persons_path, _asim_persons_rec = asim_file_paths.get("persons", (None, None))
    asim_households_path, _asim_households_rec = asim_file_paths.get(
        "households", (None, None)
    )

    def get_data(path: str, table_type: str, source: str) -> pd.DataFrame:
        if path is None:
            raise FileNotFoundError(f"{source} file for table '{table_type}' not found.")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{source} file for table '{table_type}' not found at {path}."
            )
        return BeamDataHelper.read_and_clean(path, table_type, file_format)

    beam_plans_path = locate_beam_file(beam_scenario_folder, "plans", file_format)
    beam_persons_path = locate_beam_file(beam_scenario_folder, "persons", file_format)
    beam_households_path = locate_beam_file(
        beam_scenario_folder, "households", file_format
    )

    original_hh = get_data(beam_households_path, "households", "BEAM")
    original_per = get_data(beam_persons_path, "persons", "BEAM")
    original_plans = get_data(beam_plans_path, "plans", "BEAM")

    updated_hh = get_data(asim_households_path, "households", "ActivitySim")
    updated_per = get_data(asim_persons_path, "persons", "ActivitySim")
    updated_plans = get_data(asim_plans_path, "plans", "ActivitySim")

    updated_plans["planselected"] = True
    logger.info(
        "Replanned %d persons and %d households.",
        len(updated_per),
        len(updated_hh),
    )

    per_idx = updated_per.index
    hh_idx = updated_hh.index

    persons_final = pd.concat(
        [updated_per, original_per.loc[~original_per.index.isin(per_idx)]]
    )
    households_final = pd.concat(
        [updated_hh, original_hh.loc[~original_hh.index.isin(hh_idx)]]
    )
    plans_final = pd.concat(
        [updated_plans, original_plans.loc[~original_plans.person_id.isin(per_idx)]]
    )

    record_list: List[FileRecord] = []
    outputs = [
        (persons_final, "persons"),
        (households_final, "households"),
        (plans_final, "plans"),
    ]

    for df, name in outputs:
        path = locate_beam_file(beam_scenario_folder, name, file_format)
        if file_format == "parquet":
            df.to_parquet(path, index=True)
        else:
            df.to_csv(path, index=False, compression="gzip")

        record_list.extend(
            format_specific_output_records_fn(
                file_stem=name,
                file_path=path,
                file_format=file_format,
                description_prefix="Merged BEAM input file",
            )
        )

    return record_list


def copy_plans_from_asim(
    *,
    input_records: RecordStore,
    workspace: Any,
    state: Any,
    settings: Any,
    required_input_data: List[str],
    copy_initial_asim_files_fn: Callable[..., List[FileRecord]],
    merge_replanned_asim_files_fn: Callable[..., List[FileRecord]],
) -> RecordStore:
    if state.full_settings.activitysim is None:
        return RecordStore()

    logger.info("Attempting to copy final ASIM plans from input records.")
    file_format = settings.activitysim.file_format

    base_path = workspace.full_path

    asim_file_paths: Dict[str, Tuple[Optional[str], Optional[FileRecord]]] = {}
    for record in input_records.all_records():
        splt = record.short_name.rsplit("_", 2)
        shortened_name = splt[0] if len(splt) > 1 and str.isdigit(splt[1]) else record.short_name
        if shortened_name.endswith("_asim_out"):
            shortened_name = shortened_name.split("_asim_out")[0]
        if shortened_name == "plans":
            shortened_name = "beam_plans"
        if shortened_name in required_input_data:
            asim_file_paths[shortened_name] = (
                os.path.join(base_path, record.file_path),
                record,
            )
            logger.info(
                "Found ActivitySim output file %s: %s",
                record.short_name,
                record.file_path,
            )

    required_asim_base_names = ["households", "persons", "beam_plans"]
    asim_output_dir = workspace.get_asim_output_dir()

    allow_final_pipeline = has_asim_run_marker(
        asim_output_dir,
        state.current_year,
        state.current_inner_iter,
    )
    if not allow_final_pipeline:
        logger.warning(
            "ASim success marker not found for year %s iteration %s; "
            "skipping final_pipeline fallback for BEAM inputs.",
            state.current_year,
            state.current_inner_iter,
        )

    asim_output_iter_dir = os.path.join(
        asim_output_dir,
        f"year-{state.current_year}-iteration-{state.current_inner_iter}",
    )

    for base_name in required_asim_base_names:
        if base_name in asim_file_paths:
            continue
        expected_file_name = f"{base_name}.{file_format}"
        candidate_paths = [
            os.path.join(asim_output_iter_dir, expected_file_name),
            os.path.join(asim_output_iter_dir, base_name, f"final.{file_format}"),
        ]
        if allow_final_pipeline:
            candidate_paths.append(
                os.path.join(
                    asim_output_dir,
                    "final_pipeline",
                    base_name,
                    f"final.{file_format}",
                )
            )
        found_path = next((path for path in candidate_paths if os.path.exists(path)), None)

        if found_path:
            logger.warning(
                "ActivitySim output file '%s' (expected: %s) not found in input records. "
                "Falling back to filesystem at: %s",
                base_name,
                expected_file_name,
                found_path,
            )
            dummy_path = (
                os.path.relpath(found_path, base_path)
                if base_path and os.path.isabs(base_path)
                else found_path
            )
            dummy_record = FileRecord(
                file_path=dummy_path,
                short_name=base_name,
                description=(
                    "ActivitySim output file found via filesystem fallback "
                    f"({os.path.basename(found_path)})"
                ),
                year=state.current_year,
            )
            asim_file_paths[base_name] = (found_path, dummy_record)
        else:
            logger.warning(
                "Required ActivitySim output file '%s' (expected: %s) not found in input "
                "records AND not found on filesystem at any of: %s",
                base_name,
                expected_file_name,
                ", ".join(candidate_paths),
            )

    if state.current_inner_iter <= 0:
        record_list = copy_initial_asim_files_fn(
            asim_file_paths,
            file_format,
            workspace,
        )
    else:
        record_list = merge_replanned_asim_files_fn(
            asim_file_paths,
            file_format,
            workspace,
        )

    return RecordStore(recordList=[record for record in record_list if record is not None])


def handle_linkstats(
    *,
    workspace: Any,
    previous_beam_records: List[Any],
    store: RecordStore,
    state: Any,
    settings: Any,
) -> None:
    """
    Ensure a single explicit warm-start linkstats record is present.
    """

    def abs_from_record(rec: FileRecord) -> Optional[str]:
        if not getattr(rec, "file_path", None):
            return None
        if os.path.isabs(rec.file_path):
            return rec.file_path
        return os.path.abspath(os.path.join(str(workspace.full_path), rec.file_path))

    beam_output_linkstats = None
    csv_pattern = re.compile(r"^linkstats_\d+_\d+$")
    parquet_pattern = re.compile(r"^linkstats_parquet_\d+_\d+$")

    for rec in previous_beam_records or []:
        short_name = getattr(rec, "short_name", "") or ""
        if "_sub" in short_name:
            continue
        if csv_pattern.match(short_name):
            beam_output_linkstats = rec
            break

    if beam_output_linkstats is None:
        for rec in previous_beam_records or []:
            short_name = getattr(rec, "short_name", "") or ""
            if "_sub" in short_name:
                continue
            if parquet_pattern.match(short_name):
                beam_output_linkstats = rec
                break

    if beam_output_linkstats is None:
        for rec in previous_beam_records or []:
            short_name = getattr(rec, "short_name", "") or ""
            if short_name in {"linkstats", "linkstats_parquet"}:
                beam_output_linkstats = rec
                logger.warning(
                    "[NOT IDEAL] Using an unversioned `%s` record as warm-start input. "
                    "Prefer BEAM outputs logged as `linkstats_<year>_<inner_iter>` "
                    "or `linkstats_parquet_<year>_<inner_iter>` so lineage is unambiguous.",
                    short_name,
                )
                break

    warmstart_abs_path = None
    warmstart_source = None
    if beam_output_linkstats is not None:
        warmstart_abs_path = abs_from_record(beam_output_linkstats)
        warmstart_source = "previous_beam_output"

    if warmstart_abs_path is None:
        warmstart_abs_path = resolve_initial_linkstats_path(settings, workspace)
        warmstart_source = "initial_inputs"

    if not warmstart_abs_path or not os.path.exists(warmstart_abs_path):
        logger.warning(
            "[BEAM Preprocessor] Warm-start linkstats file not found (source=%s): %s",
            warmstart_source,
            warmstart_abs_path,
        )
        return

    warmstart_rel_path = os.path.relpath(warmstart_abs_path, str(workspace.full_path))
    warmstart_record = FileRecord(
        file_path=warmstart_rel_path,
        short_name="linkstats_warmstart",
        description=f"BEAM warm-start linkstats (source={warmstart_source})",
        year=getattr(state, "forecast_year", None),
        iteration=getattr(state, "current_inner_iter", None),
    )

    logger.info(
        "[BEAM Preprocessor] Using warm-start linkstats (source=%s): %s",
        warmstart_source,
        warmstart_rel_path,
    )
    store.add_record(warmstart_record)
