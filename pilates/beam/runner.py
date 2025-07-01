import logging
import os
import psutil
import sys
from typing import Tuple

from pilates.generic.runner import GenericRunner
from pilates.generic.records import RecordStore, ModelRunInfo, OutputRecord
from pilates.beam.postprocessor import find_produced_od_skims, find_produced_origin_skims
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


class BeamRunner(GenericRunner):
    """
    Runner for the BEAM model.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)

    def run(self, store: RecordStore, state: WorkflowState) -> Tuple[RecordStore, ModelRunInfo]:
        """
        Executes a BEAM model run.
        """
        settings = state.settings
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
        run_path = state.full_path
        beam_local_mutable_data_folder = os.path.join(
            run_path, settings["beam_local_mutable_data_folder"]
        )
        abs_beam_input = os.path.abspath(str(beam_local_mutable_data_folder))
        beam_local_output_folder = os.path.join(
            run_path, settings["beam_local_output_folder"]
        )
        abs_beam_output = os.path.abspath(str(beam_local_output_folder))
        beam_memory = settings.get(
            "beam_memory",
            str(int(psutil.virtual_memory().total / (1024.0**3)) - 2) + "g",
        )

        # 2. RUN BEAM
        logger.info(
            "Starting BEAM container, input: %s, output: %s, config: %s",
            abs_beam_input,
            abs_beam_output,
            beam_config,
        )

        # Record BEAM run start
        beam_run_hash = state.record_model_start()

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

        # Record BEAM run completion
        state.record_model_completion(
            beam_run_hash, status="completed" if success else "failed"
        )
        if not success:
            logger.error("BEAM run failed.")
            sys.exit(1)

        # 3. ASSEMBLE OUTPUTS
        skims_fname = settings["skims_fname"]
        if skims_fname.endswith(".csv.gz"):
            skimFormat = "csv.gz"
        elif skims_fname.endswith(".omx"):
            skimFormat = "omx"
        else:
            # BEAM outputs an OMX file that is then merged into the Zarr store
            skimFormat = "omx"
            logger.info("Defaulting to 'omx' skim format for finding BEAM outputs.")


        # Find raw BEAM outputs
        od_skims_path = find_produced_od_skims(beam_local_output_folder, skimFormat)
        origin_skims_path = find_produced_origin_skims(beam_local_output_folder)

        output_records = []
        if od_skims_path and os.path.exists(od_skims_path):
            output_records.append(
                OutputRecord(
                    file_path=od_skims_path,
                    output_type="raw_od_skims",
                    model_run_id=beam_run_hash,
                    year=state.forecast_year
                )
            )
        if origin_skims_path and os.path.exists(origin_skims_path):
            output_records.append(
                OutputRecord(
                    file_path=origin_skims_path,
                    output_type="raw_origin_skims",
                    model_run_id=beam_run_hash,
                    year=state.forecast_year
                )
            )
        
        output_store = RecordStore(recordList=output_records)

        run_info = state.provenance_tracker.run_info.model_runs.get(beam_run_hash)

        return output_store, run_info
