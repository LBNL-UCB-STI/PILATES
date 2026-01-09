import logging
import re
import shutil

import pandas as pd
import zipfile
import os
from typing import Tuple, Optional

from pilates.generic.postprocessor import GenericPostprocessor
from pilates.generic.records import RecordStore, FileRecord
from pilates.utils.io import read_datastore
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker

logger = logging.getLogger(__name__)


def _load_asim_outputs(settings, workspace: Workspace):
    prefix = settings.activitysim.output_tables["prefix"]
    output_tables = settings.activitysim.output_tables["tables"]
    asim_output_dict = {}
    asim_output_dir = workspace.get_asim_output_dir()
    for table_name in output_tables:
        file_format = settings.activitysim.file_format
        if file_format == "parquet":
            file_name = "%s%s.parquet" % (prefix, table_name)
            file_path = os.path.join(
                asim_output_dir, "final_pipeline", table_name, "final.parquet"
            )
            try:
                table = pd.read_parquet(file_path)
            except FileNotFoundError:
                logger.warning("Parquet file not found: %s", file_path)
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


def _prepare_updated_tables(
    settings,
    state: WorkflowState,
    workspace: Workspace,
    asim_output_dict,
    tables_updated_by_asim,
    prefix=None,
):
    """
    Combines ActivitySim and UrbanSim outputs for tables updated by
    ActivitySim (e.g. households and persons)
    """

    data_dir = workspace.get_usim_mutable_data_dir()

    # e.g. model_data_2012.h5
    usim_output_store_name = get_usim_datastore_fname(
        settings, io="output", year=state.forecast_year
    )
    usim_output_store_path = os.path.join(data_dir, usim_output_store_name)
    if not os.path.exists(usim_output_store_path):
        raise ValueError(
            "No output data store found at {0}".format(usim_output_store_path)
        )
    usim_output_store = pd.HDFStore(usim_output_store_path)

    # ensure we preserve all columns originally in the urbansim outputs
    required_cols = {}
    for table_name in tables_updated_by_asim:
        h5_key = table_name
        if prefix:
            h5_key = os.path.join(str(prefix), h5_key)
        logger.info("Reading h5 table {0}".format(h5_key))
        required_cols[table_name] = list(usim_output_store[h5_key].columns)

    # This is the inverse process of asim_pre._update_persons_table()
    p_cols_to_include = required_cols["persons"]
    p_cols_to_replace = ["work_zone_id", "school_zone_id"]
    p_names_dict = {
        "PNUM": "member_id",
        "workplace_taz": "work_zone_id",
        "school_taz": "school_zone_id",
    }
    if "persons" in asim_output_dict.keys():
        logger.info("Preparing persons table!")
        for col in p_cols_to_replace:
            # Double check that work_zone_id and school_zone_id are included
            # bc these aren't native columns to UrbanSim but should be there
            # if "warm start" activities were generated
            if col not in required_cols["persons"]:
                p_cols_to_include.append(col)
            if col in asim_output_dict["persons"].columns:
                del asim_output_dict["persons"][col]
        for fromCol, toCol in p_names_dict.items():
            if (toCol in p_cols_to_include) & (
                fromCol in asim_output_dict["persons"].columns
            ):
                asim_output_dict["persons"].loc[:, toCol] = (
                    asim_output_dict["persons"].loc[:, fromCol].copy()
                )
        asim_output_dict["persons"] = asim_output_dict["persons"][p_cols_to_include]

    logger.info("Preparing households table!")
    # This is the inverse process of asim_pre._update_households_table()
    # no new columns to persist, just convert column names
    hh_names_dict = {
        "hhsize": "persons",
        "num_workers": "workers",
        "auto_ownership": "cars",
    }
    hh_cols_to_replace = ["cars"]
    hh_cols_to_include = required_cols["households"]
    if "households" in asim_output_dict.keys():
        for col in hh_cols_to_replace:
            if col not in required_cols["households"]:
                hh_cols_to_include.append(col)
            if col in asim_output_dict["households"].columns:
                del asim_output_dict["households"][col]
        asim_output_dict["households"].rename(columns=hh_names_dict, inplace=True)
        # only preserve original usim columns
        asim_output_dict["households"] = asim_output_dict["households"][
            required_cols["households"]
        ]
    else:
        logger.warning("Household table not found in ASim outputs!")
    for table_name in tables_updated_by_asim:
        h5_key = table_name
        if prefix:
            h5_key = os.path.join(str(prefix), h5_key)
        logger.info("Validating data schemas for table {0}.".format(table_name))

        # make sure all required columns are present
        if not all(
            [
                col in asim_output_dict[table_name].columns
                for col in required_cols[table_name]
            ]
        ):
            raise KeyError(
                "Not all required columns are in the {0} table! We're missing".format(
                    table_name,
                    [
                        col
                        for col in required_cols[table_name]
                        if col not in asim_output_dict[table_name].columns
                    ],
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
    provenance_tracker: "FileProvenanceTracker",
    asim_output_dict,
    source_file_paths: list,
    model_run_hash: str,
) -> Tuple[str, Optional[FileRecord]]:
    forecast_year = state.forecast_year
    asim_output_data_dir = workspace.get_asim_output_dir()
    archive_name = "asim_outputs_{0}.zip".format(forecast_year)
    outpath = os.path.join(asim_output_data_dir, archive_name)
    logger.info("Merging results back into UrbanSim format and storing as .zip!")

    # A virtual model name for this step to distinguish it in provenance
    model_name = "activitysim_postprocessor"

    # Record the source files (raw asim outputs) as inputs to this step
    for source_path in source_file_paths:
        provenance_tracker.record_input_file(
            model_name,
            source_path,
            description="Raw ActivitySim output for BEAM zip creation",
            model_run_id=model_run_hash,
        )

    with zipfile.ZipFile(outpath, "w") as csv_zip:
        # copy asim outputs into archive
        for table_name in asim_output_dict.keys():
            logger.info("Zipping {0} asim table to output archive!".format(table_name))
            csv_zip.writestr(table_name + ".csv", asim_output_dict[table_name].to_csv())
    logger.info("Done creating .zip archive!")

    # Record the created zip file as an output
    output_record = provenance_tracker.record_output_file(
        model_name,
        outpath,
        year=forecast_year,
        description="Zipped ActivitySim outputs for BEAM",
        model_run_id=model_run_hash,
        source_file_paths=source_file_paths,
        state=state,
    )
    return outpath, output_record


def create_usim_input_data(
    settings,
    state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: "FileProvenanceTracker",
    asim_output_dict,
    tables_updated_by_asim,
    asim_source_paths: list,
    model_run_hash: str,
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
    input_year = state.year
    forecast_year = state.forecast_year
    model_name = "activitysim_postprocessor"  # Virtual model name

    # parse settings
    data_dir = workspace.get_usim_mutable_data_dir()

    # --- Record Inputs ---
    # 1. Raw ActivitySim outputs (passed via asim_source_paths)
    for source_path in asim_source_paths:
        provenance_tracker.record_input_file(
            model_name,
            source_path,
            description="Raw ActivitySim output for next UrbanSim input creation",
            model_run_id=model_run_hash,
        )

    # Move UrbanSim input store (e.g. custom_mpo_193482435_model_data.h5)
    # to archive (e.g. input_data_for_2015_outputs.h5) because otherwise
    # it will be overwritten in the next step.
    input_datastore_name = get_usim_datastore_fname(settings, io="input")
    input_store_path = os.path.join(data_dir, input_datastore_name)
    archive_fname = "input_data_for_{0}_outputs.h5".format(forecast_year)
    archive_path = input_store_path.replace(input_datastore_name, archive_fname)

    if os.path.exists(input_store_path):
        logger.info(
            "Moving urbansim inputs from the previous iteration to {0}".format(
                archive_fname
            )
        )
        os.rename(input_store_path, archive_path)
    elif not os.path.exists(archive_path):
        logger.warning(
            "No input data found at {0} or {1}. Cannot create next iteration inputs.".format(
                input_store_path, archive_path
            )
        )
        return None, None

    # 2. Previous UrbanSim input data (now archived)
    provenance_tracker.record_input_file(
        model_name,
        archive_path,
        description=f"Previous UrbanSim input data (for year {input_year})",
        model_run_id=model_run_hash,
    )

    # 3. Previous UrbanSim output data
    usim_output_datastore_name = get_usim_datastore_fname(
        settings, "output", forecast_year
    )
    usim_output_store_path = os.path.join(data_dir, usim_output_datastore_name)
    if not os.path.exists(usim_output_store_path):
        logger.warning(
            "No UrbanSim output data found at {0}. Cannot create next iteration inputs.".format(
                usim_output_store_path
            )
        )
        return None, None
    provenance_tracker.record_input_file(
        model_name,
        usim_output_store_path,
        description=f"Previous UrbanSim output data (for year {forecast_year})",
        model_run_id=model_run_hash,
    )

    # --- Perform Transformation ---
    logger.info("ActivitySim output tables: %s", list(asim_output_dict.keys()))

    # load last iter UrbanSim input data
    og_input_store = pd.HDFStore(archive_path)

    # load last iter UrbanSim output data
    usim_output_store, table_prefix_year = read_datastore(
        settings, forecast_year, mutable_data_dir=workspace.get_usim_mutable_data_dir()
    )

    logger.info("Merging results back into UrbanSim format and storing as .h5!")

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
    usim_output_store.close()
    logger.info("Closing all open h5 files")

    # --- Record Output ---
    all_source_paths = asim_source_paths + [
        provenance_tracker.get_relative_path(archive_path),
        provenance_tracker.get_relative_path(usim_output_store_path),
    ]
    output_record = provenance_tracker.record_output_file(
        model_name,
        input_store_path,  # The path to the newly created H5 file
        year=forecast_year,
        description="New UrbanSim input data for next iteration",
        model_run_id=model_run_hash,
        source_file_paths=all_source_paths,
    )

    return input_store_path, output_record


def update_usim_inputs_after_warm_start(
    settings,
    state: WorkflowState,
    workspace: Workspace,
    provenance_tracker: "FileProvenanceTracker",
    model_run_hash: Optional[str] = None,
):
    """
    TODO: Combine this method with create_usim_input_data() above
    """
    model_name = "activitysim_warm_start_postprocessor"  # Virtual model name

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

    # --- Record Inputs ---
    provenance_tracker.record_input_file(
        model_name,
        input_store_path,
        description="UrbanSim input H5 before warm start update",
        model_run_id=model_run_hash,
    )
    provenance_tracker.record_input_file(
        model_name,
        warm_start_persons_path,
        description="Warm start persons data",
        model_run_id=model_run_id,
    )
    provenance_tracker.record_input_file(
        model_name,
        warm_start_households_path,
        description="Warm start households data",
        model_run_id=model_run_id,
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

    # --- Record Output ---
    source_files = [
        input_store_path,
        warm_start_persons_path,
        warm_start_households_path,
    ]
    provenance_tracker.record_output_file(
        model_name,
        input_store_path,  # The modified H5 file
        year=state.year,
        description="UrbanSim input H5 after warm start update",
        model_run_id=model_run_id,
        source_file_paths=source_files,
    )

    return


class ActivitysimPostprocessor(GenericPostprocessor):
    """
    ActivitySim-specific postprocessor that consolidates all postprocessing steps.
    """

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: FileProvenanceTracker,
        major_stage: Optional["WorkflowState.Stage"] = None,
    ):
        super().__init__(model_name, state, provenance_tracker, major_stage)

    def _postprocess(
        self,
        raw_outputs: RecordStore,
        workspace: Workspace,
        model_run_hash: Optional[str] = None,
    ) -> RecordStore:
        """
        Consolidates all postprocessing steps for ActivitySim.
        This involves taking the raw outputs from the ActivitySim run,
        processing them, and creating the necessary inputs for the next
        models in the workflow (e.g., UrbanSim, BEAM).

        Args:
            raw_outputs (RecordStore): The raw outputs from the model run.
            workspace (Workspace): The workspace object for path management.
            model_run_hash (Optional[str]): The unique hash for this postprocessor run.

        Returns:
            RecordStore: Postprocessed output data.
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

        processed_records = []

        # Archive input files from activitysim/data/
        asim_data_dir = workspace.get_asim_mutable_data_dir()
        input_files_to_archive = [
            "households.csv",
            "persons.csv",
            "land_use.csv",
            "skims.omx",
        ]

        for input_file in input_files_to_archive:
            source_path = os.path.join(asim_data_dir, input_file)
            if os.path.exists(source_path):
                target_path = os.path.join(inputs_folder_path, input_file)
                shutil.copy(source_path, target_path)

                if self.provenance_tracker:
                    archived_record = self.provenance_tracker.record_output_file(
                        model="activitysim_postprocessor",
                        file_path=target_path,
                        year=self.state.current_year,
                        description=f"Archived ActivitySim input: {input_file}",
                        short_name=f"asim_input_{input_file.replace('.', '_')}_archived",
                        source_file_paths=[source_path],
                        state=self.state,
                    )
                    if archived_record:
                        archived_record.file_path = (
                            self.provenance_tracker.get_path_relative_to_workspace_root(
                                target_path
                            )
                        )
                        archived_record.iteration = self.state.current_inner_iter
                        processed_records.append(archived_record)
                logger.info(f"Archived ActivitySim input: {input_file}")
            else:
                logger.debug(f"Input file not found, skipping archive: {source_path}")

        # Archive skims.zarr from activitysim/output/cache/
        zarr_source_path = os.path.join(
            workspace.get_asim_output_dir(), "cache", "skims.zarr"
        )
        if os.path.exists(zarr_source_path):
            zarr_target_path = os.path.join(inputs_folder_path, "skims.zarr")
            if os.path.exists(zarr_target_path):
                shutil.rmtree(zarr_target_path)
            shutil.copytree(zarr_source_path, zarr_target_path)

            if self.provenance_tracker:
                zarr_record = self.provenance_tracker.record_output_file(
                    model="activitysim_postprocessor",
                    file_path=zarr_target_path,
                    year=self.state.current_year,
                    description="Archived ActivitySim input: skims.zarr (snapshot)",
                    short_name=f"asim_input_skims_zarr_archived",
                    source_file_paths=[zarr_source_path],
                    state=self.state,
                )
                if zarr_record:
                    zarr_record.file_path = (
                        self.provenance_tracker.get_path_relative_to_workspace_root(
                            zarr_target_path
                        )
                    )
                    zarr_record.iteration = self.state.current_inner_iter
                    processed_records.append(zarr_record)
            logger.info("Archived ActivitySim input: skims.zarr")
        else:
            logger.debug(f"Zarr skims not found, skipping archive: {zarr_source_path}")

        # Record raw outputs as inputs to this post-processing run
        for record in raw_outputs.all_records():
            if hasattr(record, "file_path"):
                source = record.get_absolute_path(base_path=workspace.full_path)
                clean_name = re.sub(r"_asim_out_temp$", "", record.short_name)
                target = os.path.join(
                    iteration_folder_path,
                    clean_name + ".parquet",
                )
                if self.provenance_tracker:
                    moved_record = self.provenance_tracker.move_file(
                        record,
                        source,
                        target,
                        model="activitysim_postprocessor",
                        state=self.state,
                    )
                    moved_record.file_path = (
                        self.provenance_tracker.get_path_relative_to_workspace_root(
                            target
                        )
                    )
                    moved_record.iteration = self.state.current_inner_iter
                    processed_records.append(moved_record)
                else:
                    shutil.move(source, target)

        if self.state.is_enabled(WorkflowState.Stage.land_use):

            # 1. Load raw ActivitySim outputs from files
            # The raw_outputs RecordStore contains the paths to these files.
            # TODO: update this to only grad tables_updated_by_asim
            asim_output_dict = _load_asim_outputs(settings, workspace)

            # The raw output files are implicitly the source for all derived products in this post-processing step.
            source_file_paths = [
                getattr(r, "file_path")
                for r in raw_outputs.all_records()
                if hasattr(r, "file_path")
            ]

            # 2. Prepare tables for integration with UrbanSim
            tables_updated_by_asim = ["households", "persons"]
            asim_output_dict = _prepare_updated_tables(
                settings,
                self.state,
                workspace,
                asim_output_dict,
                tables_updated_by_asim,
                prefix=forecast_year,
            )

            # 3. Create UrbanSim input data for the next iteration
            # This function will handle its own provenance logging.
            next_usim_input_path, usim_record = create_usim_input_data(
                settings,
                self.state,
                workspace,
                self.provenance_tracker,
                asim_output_dict,
                tables_updated_by_asim,
                source_file_paths,
                model_run_hash,
            )
            if usim_record:
                processed_records.append(usim_record)

        # Return a new RecordStore with the paths to the newly created/processed files.
        processed_store = RecordStore(recordList=processed_records)
        return processed_store
