import logging
import os
import psutil
import sys
from typing import Tuple, List

from pilates.generic.runner import GenericRunner
from pilates.generic.records import RecordStore, ModelRunInfo, Record
from pilates.beam.postprocessor import (
    find_produced_od_skims,
    find_produced_origin_skims,
    find_produced_linkstats,
    find_latest_beam_iteration,
    find_produced_plans,
)
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker

logger = logging.getLogger(__name__)


def find_not_taken_dir_name(dir_name):
    for x in range(1, 99999):
        testing_name = f"{dir_name}_{x}"
        if not os.path.exists(testing_name):
            return testing_name
    raise RuntimeError(f"Cannot find an appropriate not taken directory for {dir_name}")


def rename_beam_output_directory(
    beam_output_dir, settings, year, replanning_iteration_number=0
) -> str:
    iteration_output_directory, _ = find_latest_beam_iteration(beam_output_dir)
    beam_run_output_dir = os.path.join(*iteration_output_directory.split(os.sep)[:-2])
    new_iteration_output_directory = os.path.join(
        beam_output_dir,
        settings["region"],
        "year-{0}-iteration-{1}".format(year, replanning_iteration_number),
    )
    if os.path.exists(new_iteration_output_directory):
        os.rename(
            new_iteration_output_directory,
            find_not_taken_dir_name(new_iteration_output_directory),
        )
    try:
        os.rename(beam_run_output_dir, new_iteration_output_directory)
    except FileNotFoundError:
        logger.warning(
            "Files {0} not found. Adding a slash".format(beam_run_output_dir)
        )
        os.rename("/" + str(beam_run_output_dir), new_iteration_output_directory)
    return new_iteration_output_directory




class BeamRunner(GenericRunner):
    """
    Runner for the BEAM model.
    """

    def __init__(self, model_name: str, state: "WorkflowState", provenance_tracker: FileProvenanceTracker):
        super().__init__(model_name, state, provenance_tracker)

    def gather_outputs(self, beam_local_output_folder: str, skimFormat: str = "omx") -> List[Record]:
        od_skims_path = find_produced_od_skims(beam_local_output_folder, skimFormat)
        origin_skims_path = find_produced_origin_skims(beam_local_output_folder)
        linkstats_path = find_produced_linkstats(beam_local_output_folder)
        plans_out_path = find_produced_plans(beam_local_output_folder)

        output_records = []
        if od_skims_path and os.path.exists(od_skims_path):
            output_rec = self.provenance_tracker.record_output_file(
                self.model_name,
                od_skims_path,
                year=self.state.forecast_year,
                short_name="raw_od_skims",
                model_run_id=self.provenance_tracker.run_info.model_runs.get(self.model_name).unique_id,
            )
            if output_rec:
                output_records.append(output_rec)
            else:
                logger.warning(
                    f"[BEAM Runner] Could not record output file: {od_skims_path}"
                )

        if origin_skims_path and os.path.exists(origin_skims_path):
            output_rec = self.provenance_tracker.record_output_file(
                self.model_name,
                origin_skims_path,
                year=self.state.forecast_year,
                short_name="raw_origin_skims",
                model_run_id=self.provenance_tracker.run_info.model_runs.get(self.model_name).unique_id,
            )
            if output_rec:
                output_records.append(output_rec)
            else:
                logger.warning(
                    f"[BEAM Runner] Could not record output file: {origin_skims_path}"
                )

        if linkstats_path and os.path.exists(linkstats_path):
            output_rec = self.provenance_tracker.record_output_file(
                self.model_name,
                linkstats_path,
                year=self.state.forecast_year,
                short_name="linkstats",
                model_run_id=self.provenance_tracker.run_info.model_runs.get(self.model_name).unique_id,
            )
            if output_rec:
                output_records.append(output_rec)
            else:
                logger.warning(
                    f"[BEAM Runner] Could not record output file: {linkstats_path}"
                )

        if plans_out_path and os.path.exists(plans_out_path):
            output_rec = self.provenance_tracker.record_output_file(
                self.model_name,
                plans_out_path,
                year=self.state.forecast_year,
                short_name="beam_plans_out",
                model_run_id=self.provenance_tracker.run_info.model_runs.get(self.model_name).unique_id,
            )
            if output_rec:
                output_records.append(output_rec)
            else:
                logger.warning(
                    f"[BEAM Runner] Could not record output file: {plans_out_path}"
                )
        return output_records

    def run(
        self,
        store: RecordStore,
        workspace: Workspace,
    ) -> Tuple[RecordStore, ModelRunInfo]:
        """
        Executes a BEAM model run.
        """
        settings = self.state.full_settings
        client = None  # Initialize client to None
        if settings.get("container_manager") == "docker":
            try:
                client = self.initialize_docker_client(settings)
            except Exception as e:
                logger.error(f"Failed to initialize Docker client: {e}")

        # 1. PARSE SETTINGS
        travel_model_image = settings[f"{settings['container_manager']}_images"]["beam"]
        beam_config = settings["beam_config"]
        region = settings["region"]
        path_to_beam_config = f"/app/input/{region}/{beam_config}"

        abs_beam_input = workspace.get_beam_mutable_data_dir()
        abs_beam_output = workspace.get_beam_output_dir()

        beam_memory = settings.get(
            "beam_memory",
            str(int(psutil.virtual_memory().total / (1024.0**3)) - 2) + "g",
        )

        # 2. RUN BEAM
        logger.info(
            "[BEAM Runner] Starting BEAM container, input: %s, output: %s, config: %s",
            abs_beam_input,
            abs_beam_output,
            beam_config,
        )

        # Record BEAM run start
        beam_run_hash = self.provenance_tracker.start_model_run(
            self.model_name,
            self.state.current_year,
            self.state.current_inner_iter,
            description="BEAM run",
            inputs=store,
        )

        # beam_data_repo = provenance_tracker.run_info.repo_records["beam"][0]

        # provenance_tracker.record_input_record(beam_data_repo)

        # for record in store:
        #     if isinstance(record, FileRecord):
        #         provenance_tracker.record_input_record(record)

        success = self.run_container(
            client=client,
            settings=settings,
            image=travel_model_image,
            volumes={
                abs_beam_input: {"bind": "/app/input", "mode": "rw"},
                abs_beam_output: {"bind": "/app/output", "mode": "rw"},
            },
            command=f"--config={path_to_beam_config}",
            model_name=self.model_name,
            working_dir="/app",
            environment={"JAVA_OPTS": (f"-Xmx{beam_memory}")},
        )

        if not success:
            logger.error("[BEAM Runner] BEAM run failed.")
            self.provenance_tracker.complete_model_run(beam_run_hash, status="failed")
            sys.exit(1)

        try:
            new_path = rename_beam_output_directory(
                workspace.get_beam_output_dir(),
                settings,
                self.state.current_year,
                self.state.current_inner_iter,
            )
        except Exception as e:
            new_path = workspace.get_beam_output_dir()
            logger.error("Whoops!")

        # 3. ASSEMBLE OUTPUTS
        skims_fname = settings["skims_fname"]
        if skims_fname.endswith(".csv.gz"):
            skimFormat = "csv.gz"
        elif skims_fname.endswith(".omx"):
            skimFormat = "omx"
        else:
            # BEAM outputs an OMX file that is then merged into the Zarr store
            skimFormat = "omx"
            logger.info(
                "[BEAM Runner] Defaulting to 'omx' skim format for finding BEAM outputs."
            )


        output_records = self.gather_outputs(new_path, skimFormat)

        # Record BEAM run completion now that outputs are recorded
        self.provenance_tracker.complete_model_run(
            beam_run_hash, status="completed", output_records=output_records
        )

        output_store = RecordStore(recordList=output_records)

        run_info = self.provenance_tracker.run_info.model_runs.get(beam_run_hash)

        logger.info(
            f"[BEAM Runner] BEAM run complete. Output records: {len(output_records)}"
        )
        return output_store, run_info
