import logging
import os
import shutil
from pathlib import Path
import glob
from typing import Tuple

import numpy as np
import pandas as pd

from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore
from workflow_state import WorkflowState
from pilates.workspace import Workspace
from pilates.utils.provenance import FileProvenanceTracker

logger = logging.getLogger(__name__)


def _get_usim_datastore_fname(settings, io, year=None):
    # reference: asim postprocessor
    if io == "output":
        datastore_name = settings["usim_formattable_output_file_name"].format(year=year)
    elif io == "input":
        region = settings["region"]
        region_id = settings["region_to_region_id"][region]
        usim_base_fname = settings["usim_formattable_input_file_name"]
        datastore_name = usim_base_fname.format(region_id=region_id)

    return datastore_name


class AtlasPreprocessor(GenericPreprocessor):
    def __init__(self):
        super().__init__()
        self.required_input_data = ["usim_data"]

    def copy_data_to_mutable_location(
        self,
        settings,
        output_dir,
        provenance_tracker,
    ) -> Tuple[RecordStore, RecordStore]:
        """
        No input files for atlas currently
        """
        return RecordStore(), RecordStore()

    def preprocess(
        self,
        state: WorkflowState,
        workspace: Workspace,
        provenance_tracker: FileProvenanceTracker,
    ) -> RecordStore:
        settings = state.full_settings
        # set where to find urbansim output
        urbansim_output_path = workspace.get_usim_mutable_data_dir()
        if state.is_start_year():
            # if warm start, read custom_mpo h5
                    urbansim_output_fname = _get_usim_datastore_fname(settings, io="input")
        else:
            # if in main loop, read urbansim-generated h5
            urbansim_output_fname = _get_usim_datastore_fname(
                settings, io="output", year=state.forecast_year
            )
        urbansim_output = os.path.join(urbansim_output_path, urbansim_output_fname)

        # set where to put atlas csv inputs (processed from urbansim outputs)
        atlas_input_path = os.path.join(
            workspace.get_atlas_mutable_input_dir(),
            "year{}".format(state.year),
        )

        # if atlas input path does not exist, create one
        if not os.path.exists(atlas_input_path):
            os.makedirs(atlas_input_path)
            logger.info("ATLAS Input Path Created for Year {}".format(state.year))

        if state.year != state.start_year:
            old_input_path = os.path.join(
                workspace.get_atlas_mutable_input_dir(),
                "year{}".format(state.start_year),
            )
            for f in glob.glob(os.path.join(old_input_path, "*.RData")):
                if os.path.exists(os.path.join(atlas_input_path, Path(f).name)):
                    logger.info(
                        "Not file {0} to atlas input  {1} b/c it exists".format(
                            f, os.path.join(atlas_input_path, Path(f).name)
                        )
                    )
                else:
                    logger.info(
                        "Moving file {0} to atlas input for year {1}".format(f, state.year)
                    )
                    shutil.copyfile(f, os.path.join(atlas_input_path, Path(f).name))

        # read urbansim h5 outputs
        with pd.HDFStore(urbansim_output, mode="r") as data:
            if not state.is_start_year():
                try:
                    # prepare households atlas input
                    households = data["/{}/households".format(state.year)]
                    households.to_csv("{}/households.csv".format(atlas_input_path))

                    # prepare blocks atlas input
                    blocks = data["/{}/blocks".format(state.year)]
                    blocks.to_csv("{}/blocks.csv".format(atlas_input_path))

                    # prepare persons atlas input
                    persons = data["/{}/persons".format(state.year)]
                    persons.to_csv("{}/persons.csv".format(atlas_input_path))

                    # prepare dead persons atlas input (RIP)
                    persons = data["/{}/graveyard".format(state.year)]
                    persons.to_csv("{}/grave.csv".format(atlas_input_path))

                    # prepare residential unit atlas input
                    residential_units = data["/{}/residential_units".format(state.year)]
                    residential_units.to_csv("{}/residential.csv".format(atlas_input_path))

                    # prepare jobs atlas input
                    jobs = data["/{}/jobs".format(state.year)]
                    jobs.to_csv("{}/jobs.csv".format(atlas_input_path))

                    logger.info(
                        "Preparing ATLAS Year {} Input from Urbansim Output".format(
                            state.year
                        )
                    )

                except:
                    logger.error(
                        "Urbansim Year {} Output Was Not Loaded Correctly by ATLAS".format(
                            state.year
                        )
                    )

            else:
                try:
                    # prepare households atlas input
                    households = data["/households"]
                    households.to_csv("{}/households.csv".format(atlas_input_path))

                    # prepare blocks atlas input
                    blocks = data["/blocks"]
                    blocks.to_csv("{}/blocks.csv".format(atlas_input_path))

                    # prepare persons atlas input
                    persons = data["/persons"]
                    persons.to_csv("{}/persons.csv".format(atlas_input_path))

                    # prepare residential unit atlas input
                    residential_units = data["/residential_units"]
                    residential_units.to_csv("{}/residential.csv".format(atlas_input_path))

                    # prepare jobs atlas input
                    jobs = data["/jobs"]
                    jobs.to_csv("{}/jobs.csv".format(atlas_input_path))

                    logger.info(
                        "Preparing ATLAS Year {} Input from Urbansim Output".format(
                            state.year
                        )
                    )

                except:
                    logger.error(
                        "Urbansim Year {} Output Was Not Loaded Correctly by ATLAS".format(
                            state.year
                        )
                    )

        model_run_hash = provenance_tracker.start_model_run(
            "atlas_preprocessor",
            state.current_year,
            description="ATLAS preprocessing",
        )

        input_records = workspace.input_data.get("atlas", RecordStore())
        output_records = RecordStore()

        provenance_tracker.complete_model_run(
            run_hash=model_run_hash, output_records=output_records.all_records()
        )
        return RecordStore(recordList=output_records.all_records())
