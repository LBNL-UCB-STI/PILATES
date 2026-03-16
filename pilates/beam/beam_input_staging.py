from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from pilates.activitysim.outputs import has_asim_run_marker
from pilates.generic.records import FileRecord, RecordStore
from pilates.utils.beam_warmstart import resolve_initial_linkstats_path
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
) -> Optional[FileRecord]:
    beam_scenario_folder = resolve_beam_exchange_scenario_folder_fn(workspace)
    os.makedirs(beam_scenario_folder, exist_ok=True)
    beam_vehicles_path = os.path.join(beam_scenario_folder, "vehicles.csv.gz")

    atlas_output_data_dir = workspace.get_atlas_output_dir()

    atlas_vehicle_file_loc = os.path.join(
        atlas_output_data_dir, f"vehicles2_{state.forecast_year}.csv"
    )
    if not os.path.exists(atlas_vehicle_file_loc):
        atlas_vehicle_file_loc = os.path.join(
            atlas_output_data_dir, f"vehicles2_{state.forecast_year - 1}.csv"
        )

    if not os.path.exists(atlas_vehicle_file_loc):
        logger.warning(
            "ATLAS vehicles2 file not found for BEAM input: %s",
            atlas_vehicle_file_loc,
        )
        return None

    logger.info(
        "Copying atlas vehicles2 file from %s to %s",
        atlas_vehicle_file_loc,
        beam_vehicles_path,
    )

    df = pd.read_csv(atlas_vehicle_file_loc)
    df.to_csv(beam_vehicles_path, compression="gzip", index=False)
    return FileRecord(
        file_path=beam_vehicles_path,
        description="BEAM vehicles input derived from ATLAS vehicles2",
        short_name="vehicles_beam_in",
        year=getattr(state, "forecast_year", None),
        iteration=getattr(state, "current_inner_iter", None),
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
