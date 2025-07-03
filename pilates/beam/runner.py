import logging
import os
import psutil
import sys
from typing import Tuple

from pilates.generic.runner import GenericRunner
from pilates.generic.records import RecordStore, ModelRunInfo, FileRecord
from pilates.beam.postprocessor import (
    find_produced_od_skims,
    find_produced_origin_skims,
)
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker

logger = logging.getLogger(__name__)


class BeamRunner(GenericRunner):
    """
    Runner for the BEAM model.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)

    def run(
        self,
        store: RecordStore,
        state: WorkflowState,
        workspace: Workspace,
        provenance_tracker: "FileProvenanceTracker",
    ) -> Tuple[RecordStore, ModelRunInfo]:
        """
        Executes a BEAM model run.
        """
        settings = state.full_settings
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
        beam_run_hash = provenance_tracker.start_model_run(
            self.model_name,
            state.current_year,
            state.current_inner_iter,
            description="BEAM run",
        )

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
            provenance_tracker.complete_model_run(beam_run_hash, status="failed")
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
            logger.info(
                "[BEAM Runner] Defaulting to 'omx' skim format for finding BEAM outputs."
            )

        # Find raw BEAM outputs
        beam_local_output_folder = workspace.get_beam_output_dir()
        od_skims_path = find_produced_od_skims(beam_local_output_folder, skimFormat)
        origin_skims_path = find_produced_origin_skims(beam_local_output_folder)

        output_records = []
        if od_skims_path and os.path.exists(od_skims_path):
            output_rec = provenance_tracker.record_output_file(
                self.model_name,
                od_skims_path,
                year=state.forecast_year,
                description="raw_od_skims",
                model_run_id=beam_run_hash,
            )
            if output_rec:
                output_records.append(output_rec)
            else:
                logger.warning(
                    f"[BEAM Runner] Could not record output file: {od_skims_path}"
                )

        if origin_skims_path and os.path.exists(origin_skims_path):
            output_rec = provenance_tracker.record_output_file(
                self.model_name,
                origin_skims_path,
                year=state.forecast_year,
                description="raw_origin_skims",
                model_run_id=beam_run_hash,
            )
            if output_rec:
                output_records.append(output_rec)
            else:
                logger.warning(
                    f"[BEAM Runner] Could not record output file: {origin_skims_path}"
                )

        # Record BEAM run completion now that outputs are recorded
        provenance_tracker.complete_model_run(beam_run_hash, status="completed")

        output_store = RecordStore(recordList=output_records)

        run_info = provenance_tracker.run_info.model_runs.get(beam_run_hash)

        logger.info(
            f"[BEAM Runner] BEAM run complete. Output records: {len(output_records)}"
        )
        return output_store, run_info
