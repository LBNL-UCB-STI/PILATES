import logging
import re
import shutil
from pathlib import Path

import pandas as pd
import zipfile
import os
from typing import Tuple, Optional, Dict, Any

from pilates.config import PilatesConfig
from pilates.activitysim.outputs import ActivitySimPostprocessOutputs, ActivitySimRunOutputs
from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.records import FileRecord
from pilates.activitysim.outputs import (
    ASIM_REQUIRED_RUN_OUTPUT_KEYS,
    normalize_asim_output_key,
    has_asim_run_marker,
)
from pilates.activitysim.runner import (
    asim_required_run_output_paths,
    asim_runtime_zarr_path,
    asim_staged_input_paths,
)
from pilates.workflows.artifact_keys import USIM_DATASTORE_H5, ZARR_SKIMS
from pilates.workspace import Workspace
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


def _postprocess_output_stem(output_key: str) -> str:
    return re.sub(r"_asim_out$", "", output_key)


def _activitysim_iteration_output_paths(
    state: WorkflowState,
    workspace: Workspace,
) -> Dict[str, str]:
    year = getattr(state, "year", getattr(state, "current_year", None))
    if year is None:
        return {}
    iteration = getattr(state, "iteration", getattr(state, "current_inner_iter", 0))
    iteration_dir = (
        Path(workspace.get_asim_output_dir()) / f"year-{year}-iteration-{iteration}"
    )
    return {
        output_key: str(iteration_dir / f"{_postprocess_output_stem(output_key)}.parquet")
        for output_key in ASIM_REQUIRED_RUN_OUTPUT_KEYS
    }


def _activitysim_archived_input_paths(
    state: WorkflowState,
    workspace: Workspace,
) -> Dict[str, str]:
    year = getattr(state, "year", getattr(state, "current_year", None))
    if year is None:
        return {}
    iteration = getattr(state, "iteration", getattr(state, "current_inner_iter", 0))
    archived_inputs_dir = (
        Path(workspace.get_asim_output_dir()) / f"inputs-year-{year}-iteration-{iteration}"
    )
    return {
        "asim_input_households_csv_archived": str(archived_inputs_dir / "households.csv"),
        "asim_input_persons_csv_archived": str(archived_inputs_dir / "persons.csv"),
        "asim_input_land_use_csv_archived": str(archived_inputs_dir / "land_use.csv"),
        "asim_input_skims_omx_archived": str(archived_inputs_dir / "skims.omx"),
        "asim_input_skims_zarr_archived": str(archived_inputs_dir / "skims.zarr"),
    }


def _default_usim_datastore_output_path(
    settings: PilatesConfig,
    workspace: Workspace,
) -> Optional[str]:
    try:
        datastore_name = get_usim_datastore_fname(settings, io="input")
    except Exception:
        return None
    return os.path.join(workspace.get_usim_mutable_data_dir(), datastore_name)


def _load_asim_outputs(
    settings,
    workspace: Workspace,
    state: WorkflowState,
    fallback_dir: Optional[str] = None,
):
    prefix = settings.activitysim.output_tables["prefix"]
    output_tables = settings.activitysim.output_tables["tables"]
    asim_output_dict = {}
    asim_output_dir = workspace.get_asim_output_dir()
    allow_final_pipeline = has_asim_run_marker(
        asim_output_dir, state.current_year, state.current_inner_iter
    )
    if not allow_final_pipeline:
        logger.warning(
            "ASim success marker not found for year %s iteration %s; "
            "skipping final_pipeline outputs.",
            state.current_year,
            state.current_inner_iter,
        )
    for table_name in output_tables:
        file_format = settings.activitysim.file_format
        if file_format == "parquet":
            file_name = "%s%s.parquet" % (prefix, table_name)
            file_path = os.path.join(
                asim_output_dir, "final_pipeline", table_name, "final.parquet"
            )
            if allow_final_pipeline:
                try:
                    table = pd.read_parquet(file_path)
                except FileNotFoundError:
                    if fallback_dir:
                        fallback_path = os.path.join(
                            fallback_dir, f"{table_name}.parquet"
                        )
                        try:
                            table = pd.read_parquet(fallback_path)
                        except FileNotFoundError:
                            logger.warning(
                                "Parquet file not found: %s (fallback: %s)",
                                file_path,
                                fallback_path,
                            )
                            continue
                    else:
                        logger.warning("Parquet file not found: %s", file_path)
                        continue
            else:
                if fallback_dir:
                    fallback_path = os.path.join(fallback_dir, f"{table_name}.parquet")
                    try:
                        table = pd.read_parquet(fallback_path)
                    except FileNotFoundError:
                        logger.warning(
                            "Parquet file not found (final_pipeline ignored): %s",
                            fallback_path,
                        )
                        continue
                else:
                    logger.warning(
                        "Parquet file not found; final_pipeline ignored and no fallback provided: %s",
                        file_path,
                    )
                    continue
        else:
            file_name = "%s%s.csv" % (prefix, table_name)
            file_path = os.path.join(asim_output_dir, file_name)
            if table_name == "persons":
                index_col = "person_id"
            elif table_name == "households":
                index_col = "household_id"
            else:
                index_col = None
            try:
                table = pd.read_csv(file_path, index_col=index_col)
            except FileNotFoundError:
                logger.warning("CSV file not found: %s", file_path)
                continue

        if "block_id" in table.columns:
            table["block_id"] = table["block_id"].astype(str).str.zfill(15)
        if "lcm_county_id" in table.columns:
            table["lcm_county_id"] = table["lcm_county_id"].astype(str).str.zfill(5)

        asim_output_dict[table_name] = table

    return asim_output_dict


def get_usim_datastore_fname(settings, io, year=None):
    if io == "output":
        datastore_name = settings.urbansim.output_file_template.format(year=year)
    elif io == "input":
        region = settings.run.region
        region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
        usim_base_fname = settings.urbansim.input_file_template
        datastore_name = usim_base_fname.format(region_id=region_id)

    return datastore_name


def _normalized_h5_keys(store: pd.HDFStore) -> Dict[str, str]:
    return {key.strip("/"): key for key in store.keys()}


def _detect_h5_prefix(
    store: pd.HDFStore,
    required_tables,
    preferred_prefix=None,
) -> Optional[str]:
    normalized_keys = _normalized_h5_keys(store)
    candidate_prefixes = []
    if preferred_prefix not in (None, ""):
        candidate_prefixes.append(str(preferred_prefix))
    candidate_prefixes.append("")

    discovered_prefixes = []
    for key in normalized_keys:
        if "/" in key:
            discovered_prefixes.append(key.split("/", 1)[0])
    for candidate in discovered_prefixes:
        if candidate not in candidate_prefixes:
            candidate_prefixes.append(candidate)

    for candidate_prefix in candidate_prefixes:
        if all(
            (
                f"{candidate_prefix}/{table_name}"
                if candidate_prefix
                else table_name
            )
            in normalized_keys
            for table_name in required_tables
        ):
            return candidate_prefix
    return None


def _prepare_updated_tables(
    settings,
    state: WorkflowState,
    asim_output_dict,
    tables_updated_by_asim,
    population_source_store_path: str,
    prefix=None,
):
    """
    Combines ActivitySim and UrbanSim outputs for tables updated by
    ActivitySim (e.g. households and persons)
    """
    usim_output_store_path = population_source_store_path
    if not os.path.exists(usim_output_store_path):
        raise ValueError(
            "ActivitySim postprocess requires the bound UrbanSim population-source datastore, "
            f"but it does not exist: {usim_output_store_path}"
        )
    with pd.HDFStore(usim_output_store_path, mode="r") as prefix_store:
        resolved_prefix = _detect_h5_prefix(
            prefix_store,
            required_tables=tables_updated_by_asim,
            preferred_prefix=prefix,
        )
    if resolved_prefix is None:
        raise ValueError(
            "ActivitySim postprocess could not resolve required UrbanSim tables "
            f"{list(tables_updated_by_asim)} from bound population-source datastore "
            f"{usim_output_store_path}"
        )
    if resolved_prefix != prefix:
        logger.info(
            "Using UrbanSim datastore %s with table prefix %r for ActivitySim postprocess "
            "(requested prefix %r).",
            usim_output_store_path,
            resolved_prefix,
            prefix,
        )
    usim_output_store = pd.HDFStore(usim_output_store_path)
    normalized_usim_keys = _normalized_h5_keys(usim_output_store)

    def _resolve_h5_key(table_name: str) -> str:
        candidates = []
        if resolved_prefix not in (None, ""):
            candidates.append(f"{resolved_prefix}/{table_name}")
        candidates.append(table_name)
        for candidate in candidates:
            actual = normalized_usim_keys.get(candidate.strip("/"))
            if actual:
                return actual
        raise KeyError(
            "UrbanSim datastore {0} does not contain table '{1}'. Available keys: {2}".format(
                usim_output_store_path,
                table_name,
                sorted(normalized_usim_keys),
            )
        )

    # ensure we preserve all columns originally in the urbansim outputs
    required_cols = {}
    usim_tables = {}

    vehicle_ownership_model = (
        getattr(getattr(settings, "run", None), "models", None)
        and getattr(settings.run.models, "vehicle_ownership", None)
    )
    use_asim_auto_ownership = vehicle_ownership_model != "atlas"

    def _ensure_index(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
        if df.index.name == index_col:
            return df
        if index_col in df.columns:
            return df.set_index(index_col)
        return df

    def _get_usim_table(table_name: str) -> pd.DataFrame:
        if table_name not in usim_tables:
            h5_key = _resolve_h5_key(table_name)
            usim_tables[table_name] = usim_output_store[h5_key].copy()
        return usim_tables[table_name]

    for table_name in tables_updated_by_asim:
        h5_key = _resolve_h5_key(table_name)
        logger.info("Reading h5 table {0}".format(h5_key))
        required_cols[table_name] = list(usim_output_store[h5_key].columns)

    # This is the inverse process of asim_pre._update_persons_table()
    p_cols_to_include = required_cols["persons"]

    def _normalize_person_household_ids(
        persons_df: pd.DataFrame, target_dtype
    ) -> pd.DataFrame:
        if "household_id" not in persons_df.columns:
            return persons_df

        numeric = pd.to_numeric(persons_df["household_id"], errors="coerce")
        invalid_mask = numeric.isna()
        invalid_mask |= (numeric % 1 != 0).fillna(False)

        if invalid_mask.any():
            dropped = int(invalid_mask.sum())
            sample_person_ids = []
            if persons_df.index.name == "person_id":
                sample_person_ids = persons_df.index[invalid_mask].tolist()[:10]
            elif "person_id" in persons_df.columns:
                sample_person_ids = (
                    persons_df.loc[invalid_mask, "person_id"].tolist()[:10]
                )
            logger.warning(
                "Dropping %s ActivitySim persons rows with missing/invalid household_id "
                "before writing the updated UrbanSim persons table. Sample person_ids=%s",
                dropped,
                sample_person_ids,
            )
            persons_df = persons_df.loc[~invalid_mask].copy()
            numeric = numeric.loc[~invalid_mask]

        persons_df.loc[:, "household_id"] = numeric.astype(target_dtype)
        return persons_df

    def _set_from_source(df: pd.DataFrame, target: str, sources) -> bool:
        for source in sources:
            if source in df.columns:
                df[target] = df[source]
                return True
        return False

    def _prepare_asim_persons_overlay(
        persons: pd.DataFrame, usim_persons: pd.DataFrame
    ) -> pd.DataFrame:
        if persons.index.name == "person_id" or "person_id" in persons.columns:
            return _ensure_index(persons, "person_id")

        persons = persons.copy()
        if "member_id" not in persons.columns and "PNUM" in persons.columns:
            persons["member_id"] = persons["PNUM"]

        if not {"household_id", "member_id"} <= set(persons.columns):
            logger.warning(
                "ASim persons output lacks person_id and household/member identifiers; "
                "skipping ASim persons writeback overlays."
            )
            return pd.DataFrame(index=pd.Index([], name="person_id"))

        key_cols = ["household_id", "member_id"]
        persons[key_cols] = persons[key_cols].apply(
            pd.to_numeric, errors="coerce"
        ).astype("Int64")

        logger.warning(
            "ASim persons output is missing person_id; falling back to household_id/member_id "
            "alignment for work/school zone overlays. This alignment is weaker because "
            "member_id can change when preprocess filters or reorders household members."
        )

        usim_lookup = usim_persons.reset_index()[["person_id"] + key_cols].copy()
        usim_lookup[key_cols] = usim_lookup[key_cols].apply(
            pd.to_numeric, errors="coerce"
        ).astype("Int64")
        duplicate_mask = usim_lookup.duplicated(key_cols, keep=False)
        if duplicate_mask.any():
            logger.warning(
                "UrbanSim persons table has %s duplicate household/member pairs; "
                "dropping duplicates before ASim overlay alignment.",
                int(duplicate_mask.sum()),
            )
            usim_lookup = usim_lookup.loc[~duplicate_mask].copy()

        merged = persons.merge(usim_lookup, on=key_cols, how="left")
        missing_person_id = merged["person_id"].isna()
        if missing_person_id.any():
            logger.warning(
                "Dropping %s ASim persons rows that could not be aligned back to UrbanSim person_id "
                "via household_id/member_id.",
                int(missing_person_id.sum()),
            )
            merged = merged.loc[~missing_person_id].copy()

        merged["person_id"] = pd.to_numeric(
            merged["person_id"], errors="coerce"
        ).astype("Int64")
        merged = merged.loc[merged["person_id"].notna()].copy()
        merged["person_id"] = merged["person_id"].astype(int)
        return merged.set_index("person_id")

    if "persons" in asim_output_dict.keys():
        logger.info("Preparing persons table!")
        usim_persons = _ensure_index(_get_usim_table("persons"), "person_id")
        persons_overlay = _prepare_asim_persons_overlay(
            asim_output_dict["persons"], usim_persons
        )

        if not persons_overlay.empty:
            # Phase 1 migration keeps these aliases equivalent. Phase 2 should
            # switch this overlay logic to the canonical ActivitySim names
            # (`workplace_zone_id` / `school_zone_id`) and remove the fallbacks.
            work_zone_source = None
            if _set_from_source(
                persons_overlay, "work_zone_id", ["workplace_zone_id", "workplace_taz"]
            ):
                work_zone_source = "workplace_zone_id"
                if "workplace_zone_id" not in persons_overlay.columns:
                    work_zone_source = "workplace_taz"
            school_zone_source = None
            if "school_zone_id" in persons_overlay.columns:
                school_zone_source = "school_zone_id"
            elif _set_from_source(persons_overlay, "school_zone_id", ["school_taz"]):
                school_zone_source = "school_taz"
            if work_zone_source or school_zone_source:
                logger.debug(
                    "ASim persons zone mapping: work_zone_id <- %s, school_zone_id <- %s",
                    work_zone_source,
                    school_zone_source,
                )

        persons = usim_persons.copy()
        overlay_cols = [
            col
            for col in ("work_zone_id", "school_zone_id")
            if col in p_cols_to_include and col in persons_overlay.columns
        ]
        if overlay_cols:
            common_idx = persons.index.intersection(persons_overlay.index)
            for col in overlay_cols:
                persons.loc[common_idx, col] = persons_overlay.loc[common_idx, col]

        persons = _normalize_person_household_ids(
            persons, usim_persons["household_id"].dtype
        )
        asim_output_dict["persons"] = persons[p_cols_to_include]

    logger.info("Preparing households table!")
    hh_cols_to_include = required_cols["households"]
    if "households" in asim_output_dict.keys():
        usim_households = _ensure_index(_get_usim_table("households"), "household_id")
        households_overlay = _ensure_index(
            asim_output_dict["households"], "household_id"
        )
        households = usim_households.copy()
        if use_asim_auto_ownership and "cars" in required_cols["households"]:
            if "auto_ownership" in households_overlay.columns:
                common_idx = households.index.intersection(households_overlay.index)
                households.loc[common_idx, "cars"] = households_overlay.loc[
                    common_idx, "auto_ownership"
                ]
            else:
                logger.warning(
                    "ASim households output missing auto_ownership; preserving UrbanSim cars."
                )
        asim_output_dict["households"] = households[required_cols["households"]]
    else:
        logger.warning("Household table not found in ASim outputs!")
    for table_name in tables_updated_by_asim:
        h5_key = _resolve_h5_key(table_name)
        logger.info("Validating data schemas for table {0}.".format(table_name))

        # make sure all required columns are present
        if not all(
            [
                col in asim_output_dict[table_name].columns
                for col in required_cols[table_name]
            ]
        ):
            missing_columns = [
                col
                for col in required_cols[table_name]
                if col not in asim_output_dict[table_name].columns
            ]
            raise KeyError(
                "Not all required columns are in the {0} table! We're missing {1}".format(
                    table_name,
                    missing_columns,
                )
            )
        # make sure data types match
        else:
            dtypes = usim_output_store[h5_key].dtypes.to_dict()
            for col in required_cols[table_name]:
                if asim_output_dict[table_name][col].dtype != dtypes[col]:
                    asim_output_dict[table_name][col] = (
                        asim_output_dict[table_name]
                        .loc[~asim_output_dict[table_name][col].isna(), col]
                        .fillna(0)
                        .astype(dtypes[col])
                    )
    usim_output_store.close()

    # specific dtype required conversions
    asim_output_dict["households"]["block_id"] = asim_output_dict["households"][
        "block_id"
    ].astype(str)

    return asim_output_dict


def create_beam_input_data(
    settings,
    state: WorkflowState,
    workspace: Workspace,
    asim_output_dict,
    source_file_paths: list,
) -> Tuple[str, Optional[FileRecord]]:
    forecast_year = state.forecast_year
    asim_output_data_dir = workspace.get_asim_output_dir()
    archive_name = "asim_outputs_{0}.zip".format(forecast_year)
    outpath = os.path.join(asim_output_data_dir, archive_name)
    logger.info("Merging results back into UrbanSim format and storing as .zip!")

    with zipfile.ZipFile(outpath, "w") as csv_zip:
        # copy asim outputs into archive
        for table_name in asim_output_dict.keys():
            logger.info("Zipping {0} asim table to output archive!".format(table_name))
            csv_zip.writestr(table_name + ".csv", asim_output_dict[table_name].to_csv())
    logger.info("Done creating .zip archive!")

    output_record = FileRecord(
        file_path=outpath,
        year=forecast_year,
        description="Zipped ActivitySim outputs for BEAM",
        short_name="asim_outputs_zip",
    )
    return outpath, output_record


def create_usim_input_data(
    settings,
    state: WorkflowState,
    asim_output_dict,
    tables_updated_by_asim,
    asim_source_paths: list,
    current_input_store_path: str,
    population_source_store_path: Optional[str],
) -> Tuple[str, Optional[FileRecord]]:
    """
    Creates UrbanSim input data for the next iteration.

    Populates an .h5 datastore from three sources in order: 1. ActivitySim
    outputs; 2. UrbanSim outputs; 3. UrbanSim inputs. The three sources will
    have tables in common, so care must be taken to use only the most
    recently updated version of each table. In a given iteration, ActivitySim
    runs last, thus UrbanSim outputs are only passed on if they weren't found
    in the ActivitySim outputs. Likewise, UrbanSim *inputs* are only passed
    on to the next iteration if they were not found in the UrbanSim *outputs*.
    """
    forecast_year = state.forecast_year
    input_store_path = current_input_store_path
    input_datastore_name = os.path.basename(input_store_path)
    archive_fname = "input_data_for_{0}_outputs.h5".format(forecast_year)
    archive_path = os.path.join(os.path.dirname(input_store_path), archive_fname)

    fallback_to_current_input = not (
        population_source_store_path and os.path.exists(population_source_store_path)
    )
    source_store_path = (
        input_store_path if fallback_to_current_input else population_source_store_path
    )
    source_store_label = (
        "current UrbanSim input datastore"
        if fallback_to_current_input
        else "UrbanSim population-source datastore"
    )

    if os.path.exists(input_store_path):
        logger.info(
            "Moving urbansim inputs from the previous iteration to {0}".format(
                archive_fname
            )
        )
        os.rename(input_store_path, archive_path)
        if fallback_to_current_input or (
            source_store_path
            and os.path.abspath(source_store_path) == os.path.abspath(input_store_path)
        ):
            source_store_path = archive_path
    elif not os.path.exists(archive_path):
        logger.warning(
            "No input data found at {0} or {1}. Cannot create next iteration inputs.".format(
                input_store_path, archive_path
            )
        )
        return None, None

    source_prefix = None
    if os.path.exists(source_store_path):
        with pd.HDFStore(source_store_path, mode="r") as source_store:
            source_prefix = _detect_h5_prefix(
                source_store,
                required_tables=["households"],
                preferred_prefix=forecast_year,
            )

    if source_prefix is None:
        logger.warning(
            "No UrbanSim source data found at %s. Cannot create next iteration inputs.",
            source_store_path,
        )
        return None, None

    if fallback_to_current_input:
        logger.info(
            "Falling back to %s for ActivitySim postprocess because forecast output is missing: %s",
            source_store_label,
            source_store_path,
        )

    # 3. Previous UrbanSim output/current data
    usim_output_datastore_name = os.path.basename(source_store_path)
    usim_output_store_path = source_store_path
    if not os.path.exists(usim_output_store_path):
        logger.warning(
            "No UrbanSim source data found at {0}. Cannot create next iteration inputs.".format(
                usim_output_store_path
            )
        )
        return None, None

    logger.info("ActivitySim output tables: %s", list(asim_output_dict.keys()))

    # load last iter UrbanSim input data
    og_input_store = pd.HDFStore(archive_path, mode="r")

    # load last iter UrbanSim output/current data
    same_source_as_archive = os.path.abspath(usim_output_store_path) == os.path.abspath(
        archive_path
    )
    usim_output_store = (
        og_input_store
        if same_source_as_archive
        else pd.HDFStore(usim_output_store_path, mode="r")
    )
    table_prefix_year = str(source_prefix)

    logger.info(
        "Merging results back into UrbanSim format and storing as .h5 using %s (%s).",
        usim_output_store_path,
        source_store_label,
    )

    # instantiate empty .h5 store (e.g. custom_mpo_321487234_model_data.h5)
    new_input_store = pd.HDFStore(input_store_path)
    assert len(new_input_store.keys()) == 0

    # Keep track of which tables have already been added (i.e. updated)
    updated_tables = []

    # 1. copy ASIM OUTPUTS into new input data store
    logger.info(
        "Copying ActivitySim outputs to the new Urbansim input store at {0}! "
        "Tables: {1}".format(input_store_path, tables_updated_by_asim)
    )
    for table_name in tables_updated_by_asim:
        logger.info("   Moving {0}".format(table_name))
        new_input_store["/" + table_name] = asim_output_dict[table_name]
        updated_tables.append(table_name)

    # 2. copy USIM OUTPUTS into new input data store if not present already
    logger.info(
        (
            "Passing last set of UrbanSim outputs through to the new "
            "Urbansim input store! {0} -> {1}".format(
                usim_output_datastore_name, input_store_path
            )
        )
    )
    for h5_key in usim_output_store.keys():
        table_name = h5_key.split("/")[-1]
        if table_name not in updated_tables:
            if os.path.join("/", table_prefix_year, table_name) == h5_key:
                new_input_store[table_name] = usim_output_store[h5_key]
                updated_tables.append(table_name)
                logger.info(
                    (
                        "    Passing static UrbanSim table {0} through to the new Urbansim "
                        "input store!"
                    ).format(table_name)
                )
            else:
                logger.info(
                    "    Skipping key {0} because it is not formatted correctly".format(
                        h5_key
                    )
                )
        else:
            logger.info(
                "    Skipping {0} because it was updated by Asim".format(h5_key)
            )

    # 3. copy USIM INPUTS into new input data store if not present already

    for h5_key in og_input_store.keys():
        table_name = h5_key.split("/")[-1]
        if table_name not in updated_tables:
            new_input_store[table_name] = og_input_store[h5_key]
            logger.info(
                (
                    "Passing original UrbanSim table {0} through to the new Urbansim "
                    "input store!"
                ).format(table_name)
            )

    og_input_store.close()
    new_input_store.close()
    if not same_source_as_archive:
        usim_output_store.close()
    logger.info("Closing all open h5 files")

    output_record = FileRecord(
        file_path=input_store_path,
        year=forecast_year,
        description="New UrbanSim input data for next iteration",
        short_name=USIM_DATASTORE_H5,
    )

    return input_store_path, output_record


def update_usim_inputs_after_warm_start(
    settings,
    state: WorkflowState,
    workspace: Workspace,
    model_run_hash: Optional[str] = None,
):
    """
    TODO: Combine this method with create_usim_input_data() above
    """
    # load usim data
    usim_data_dir = workspace.get_usim_mutable_data_dir()
    datastore_name = get_usim_datastore_fname(settings, io="input")
    input_store_path = os.path.join(usim_data_dir, datastore_name)
    if not os.path.exists(input_store_path):
        raise ValueError("No input data found at {0}".format(input_store_path))

    # load warm start data
    warm_start_dir = workspace.get_asim_output_dir()
    warm_start_persons_path = os.path.join(warm_start_dir, "warm_start_persons.csv")
    warm_start_households_path = os.path.join(
        warm_start_dir, "warm_start_households.csv"
    )

    # --- Perform Transformation ---
    usim_datastore = pd.HDFStore(input_store_path)
    p = usim_datastore["persons"]
    hh = usim_datastore["households"]

    warm_start_persons = pd.read_csv(
        warm_start_persons_path,
        index_col="person_id",
        dtype={"workplace_taz": str, "school_taz": str},
    )
    warm_start_households = pd.read_csv(
        warm_start_households_path,
        index_col="household_id",
    )

    # replace persons and households with warm start data
    assert p.shape[0] == warm_start_persons.shape[0]
    assert hh.shape[0] == warm_start_households.shape[0]

    p.loc[:, "work_zone_id"] = warm_start_persons.loc[:, "workplace_taz"].reindex(
        p.index
    )
    p.loc[:, "school_zone_id"] = warm_start_persons.loc[:, "school_taz"].reindex(
        p.index
    )
    hh.loc[:, "cars"] = warm_start_households["auto_ownership"].reindex(hh.index)

    usim_datastore["persons"] = p
    usim_datastore["households"] = hh

    usim_datastore.close()

    return


class ActivitysimPostprocessor(GenericPostprocessor):
    """
    ActivitySim-specific postprocessor that consolidates all postprocessing steps.
    """

    @staticmethod
    def declared_expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this postprocessor expects without
        disk checks.
        """
        del settings
        inputs: Dict[str, Any] = {
            "asim_output_dir": workspace.get_asim_output_dir(),
            **asim_staged_input_paths(workspace),
            **asim_required_run_output_paths(workspace),
        }
        inputs[ZARR_SKIMS] = asim_runtime_zarr_path(workspace)
        return inputs

    @staticmethod
    def runtime_expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare runtime expected inputs, including filesystem presence checks.
        """
        inputs = ActivitysimPostprocessor.declared_expected_inputs(
            settings, state, workspace
        )
        asim_output_dir = inputs.get("asim_output_dir")
        inputs["asim_output_dir"] = (
            asim_output_dir if asim_output_dir and os.path.exists(asim_output_dir) else None
        )
        zarr_path = inputs.get(ZARR_SKIMS)
        inputs[ZARR_SKIMS] = (
            zarr_path if zarr_path and os.path.exists(zarr_path) else None
        )
        return inputs

    @staticmethod
    def expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        return ActivitysimPostprocessor.runtime_expected_inputs(
            settings, state, workspace
        )

    @staticmethod
    def expected_outputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the output paths/artifacts this postprocessor produces.

        Notes
        -----
        Output keys
            - ``asim_output_dir``: ActivitySim output directory retained after
              postprocessing. This coarse directory contract remains because
              archived inputs and per-iteration outputs share stable topology
              underneath it.
            - ``usim_datastore_h5``: Updated UrbanSim datastore (H5) written
              for downstream UrbanSim/ATLAS steps when land use writeback is enabled.
            - required ``*_asim_out`` keys: Canonical postprocessed parquet
              outputs archived under the year/iteration directory.
            - ``asim_input_*_archived``: Archived source inputs retained for
              provenance-aware recovery.
        Related docs
            - See `pilates/activitysim/inputs.py` for the corresponding input
              descriptions used by ActivitySim and downstream models.
        """
        outputs: Dict[str, Any] = {
            "asim_output_dir": workspace.get_asim_output_dir(),
            USIM_DATASTORE_H5: _default_usim_datastore_output_path(settings, workspace),
        }
        outputs.update(_activitysim_iteration_output_paths(state, workspace))
        outputs.update(_activitysim_archived_input_paths(state, workspace))
        return outputs

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        super().__init__(model_name, state)

    def postprocess(
        self,
        raw_outputs: ActivitySimRunOutputs,
        workspace: Workspace,
        model_run_hash: Optional[str] = None,
        population_source_h5_path: Optional[str] = None,
        current_input_h5_path: Optional[str] = None,
    ) -> ActivitySimPostprocessOutputs:
        if not isinstance(raw_outputs, ActivitySimRunOutputs):
            raise TypeError(
                "ActivitysimPostprocessor.postprocess expects ActivitySimRunOutputs"
            )
        self.state.set_sub_stage_progress("postprocessor")
        postprocess_kwargs: Dict[str, Any] = {}
        if population_source_h5_path is not None:
            postprocess_kwargs["population_source_h5_path"] = population_source_h5_path
        if current_input_h5_path is not None:
            postprocess_kwargs["current_input_h5_path"] = current_input_h5_path
        return self._postprocess(
            raw_outputs,
            workspace,
            model_run_hash,
            **postprocess_kwargs,
        )

    def _postprocess(
        self,
        raw_outputs: ActivitySimRunOutputs,
        workspace: Workspace,
        model_run_hash: Optional[str] = None,
        population_source_h5_path: Optional[str] = None,
        current_input_h5_path: Optional[str] = None,
    ) -> ActivitySimPostprocessOutputs:
        """
        Consolidates all postprocessing steps for ActivitySim.
        This involves taking the raw outputs from the ActivitySim run,
        processing them, and creating the necessary inputs for the next
        models in the workflow (e.g., UrbanSim, BEAM).

        Args:
            raw_outputs (ActivitySimRunOutputs): The raw outputs from the model run.
            workspace (Workspace): The workspace object for path management.
            model_run_hash (Optional[str]): The unique hash for this postprocessor run.

        Returns:
            ActivitySimPostprocessOutputs: Postprocessed output data.
        """
        settings = self.state.full_settings
        year = self.state.year
        forecast_year = self.state.forecast_year
        replanning_iteration_number = self.state.current_inner_iter
        logger.info(
            "Running ActivitySim postprocessor for year %s, forecast year %s",
            year,
            forecast_year,
        )

        iteration_folder_name = "year-{0}-iteration-{1}".format(
            year, replanning_iteration_number
        )

        iteration_folder_path = os.path.join(
            workspace.get_asim_output_dir(), iteration_folder_name
        )
        if not os.path.exists(os.path.abspath(iteration_folder_path)):
            os.makedirs(iteration_folder_path, exist_ok=True)

        # Archive ActivitySim inputs for this iteration
        # This ensures Consist can find input files at stable paths for hybrid views
        inputs_folder_name = "inputs-year-{0}-iteration-{1}".format(
            year, replanning_iteration_number
        )
        inputs_folder_path = os.path.join(
            workspace.get_asim_output_dir(), inputs_folder_name
        )
        if not os.path.exists(os.path.abspath(inputs_folder_path)):
            os.makedirs(inputs_folder_path, exist_ok=True)

        processed_outputs: Dict[str, str] = {}
        processed_output_hashes: Dict[str, str] = {}
        usim_datastore_h5: Optional[str] = None
        usim_datastore_key: Optional[str] = None

        def _build_content_hash_map() -> Dict[str, str]:
            hash_map: Dict[str, str] = {}
            for short_name, path in raw_outputs.raw_outputs.items():
                record_hash = raw_outputs.raw_output_hashes.get(short_name)
                if record_hash:
                    hash_map[os.path.abspath(str(path))] = record_hash
            source_input_paths = raw_outputs.source_input_paths
            source_input_hashes = raw_outputs.source_input_hashes
            if not source_input_paths and not source_input_hashes:
                logger.warning(
                    "ActivitySim postprocess raw outputs are missing source input metadata; "
                    "archived input content hashes may be unavailable."
                )
            for short_name, source_path in source_input_paths.items():
                record_hash = source_input_hashes.get(short_name)
                if not record_hash or not source_path:
                    continue
                hash_map[os.path.abspath(str(source_path))] = record_hash
            return hash_map

        content_hash_map = _build_content_hash_map()

        def _resolve_content_hash(source_path: str) -> Optional[str]:
            if not source_path:
                return None
            return content_hash_map.get(os.path.abspath(source_path))

        # Archive input files from activitysim/data/
        asim_data_dir = workspace.get_asim_mutable_data_dir()
        input_files_to_archive = [
            "households.csv",
            "persons.csv",
            "land_use.csv",
        ]
        zarr_used_as_input = bool(raw_outputs.source_input_paths.get(ZARR_SKIMS))
        if not zarr_used_as_input:
            input_files_to_archive.append("skims.omx")

        for input_file in input_files_to_archive:
            source_path = os.path.join(asim_data_dir, input_file)
            if os.path.exists(source_path):
                target_path = os.path.join(inputs_folder_path, input_file)
                shutil.copy(source_path, target_path)
                content_hash = _resolve_content_hash(source_path)

                archived_key = f"asim_input_{input_file.replace('.', '_')}_archived"
                processed_outputs[archived_key] = target_path
                if content_hash:
                    processed_output_hashes[archived_key] = content_hash
                logger.info(f"Archived ActivitySim input: {input_file}")
            else:
                logger.debug(f"Input file not found, skipping archive: {source_path}")

        # Archive skims.zarr from activitysim/output/cache/
        zarr_source_path = asim_runtime_zarr_path(workspace)
        if os.path.exists(zarr_source_path):
            zarr_target_path = os.path.join(inputs_folder_path, "skims.zarr")
            if os.path.exists(zarr_target_path):
                if os.path.isdir(zarr_target_path):
                    shutil.rmtree(zarr_target_path)
                else:
                    os.remove(zarr_target_path)
            if os.path.isdir(zarr_source_path):
                shutil.copytree(zarr_source_path, zarr_target_path)
            else:
                shutil.copy2(zarr_source_path, zarr_target_path)
            content_hash = _resolve_content_hash(zarr_source_path)

            processed_outputs["asim_input_skims_zarr_archived"] = zarr_target_path
            if content_hash:
                processed_output_hashes["asim_input_skims_zarr_archived"] = content_hash
            logger.info("Archived ActivitySim input: skims.zarr")
        else:
            logger.debug(f"Zarr skims not found, skipping archive: {zarr_source_path}")

        if self.state.is_enabled(WorkflowState.Stage.land_use):
            if not population_source_h5_path:
                raise ValueError(
                    "ActivitySim postprocess requires population_source_h5_path when land use is enabled."
                )
            if not current_input_h5_path:
                raise ValueError(
                    "ActivitySim postprocess requires current_input_h5_path when land use is enabled."
                )

            # 1. Load raw ActivitySim outputs from files
            # The raw_outputs RecordStore contains the paths to these files.
            # TODO: update this to only grad tables_updated_by_asim
            asim_output_dict = _load_asim_outputs(
                settings,
                workspace,
                self.state,
                fallback_dir=iteration_folder_path,
            )

            # The raw output files are implicitly the source for all derived products in this post-processing step.
            source_file_paths = [str(path) for path in raw_outputs.raw_outputs.values()]

            # 2. Prepare tables for integration with UrbanSim
            tables_updated_by_asim = ["households", "persons"]
            asim_output_dict = _prepare_updated_tables(
                settings,
                self.state,
                asim_output_dict,
                tables_updated_by_asim,
                population_source_store_path=population_source_h5_path,
                prefix=forecast_year,
            )

            # 3. Create UrbanSim input data for the next iteration
            # This function will handle its own provenance logging.
            next_usim_input_path, usim_record = create_usim_input_data(
                settings,
                self.state,
                asim_output_dict,
                tables_updated_by_asim,
                source_file_paths,
                current_input_store_path=current_input_h5_path,
                population_source_store_path=population_source_h5_path,
            )
            if usim_record:
                usim_datastore_h5 = next_usim_input_path
                usim_datastore_key = getattr(usim_record, "short_name", None)

        # Record raw outputs as inputs to this post-processing run
        for short_name, path in raw_outputs.raw_outputs.items():
            source = os.path.abspath(str(path))
            clean_name = re.sub(r"_asim_out_temp$", "", short_name)
            output_key = normalize_asim_output_key(clean_name)
            target = os.path.join(
                iteration_folder_path,
                clean_name + ".parquet",
            )
            content_hash = raw_outputs.raw_output_hashes.get(short_name)
            if os.path.abspath(source) == os.path.abspath(target):
                processed_outputs[output_key] = target
                if content_hash:
                    processed_output_hashes[output_key] = content_hash
                continue
            if os.path.exists(target):
                logger.debug("ASim output already archived: %s", target)
                processed_outputs[output_key] = target
                if content_hash is None:
                    content_hash = _resolve_content_hash(source) or _resolve_content_hash(
                        target
                    )
                if content_hash:
                    processed_output_hashes[output_key] = content_hash
                continue
            if not os.path.exists(source):
                logger.debug("ASim output missing, skipping move: %s", source)
                continue
            shutil.move(source, target)
            if content_hash is None:
                content_hash = _resolve_content_hash(source)
            processed_outputs[output_key] = target
            if content_hash:
                processed_output_hashes[output_key] = content_hash

        return ActivitySimPostprocessOutputs(
            usim_datastore_h5=Path(usim_datastore_h5) if usim_datastore_h5 else None,
            asim_output_dir=Path(workspace.get_asim_output_dir()),
            processed_outputs={
                key: Path(path) for key, path in processed_outputs.items()
            },
            processed_output_hashes=processed_output_hashes,
            usim_datastore_key=usim_datastore_key,
        )
