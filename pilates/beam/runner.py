import logging
import os
from datetime import datetime

import sys
from typing import List, Optional

from pilates.config import PilatesConfig
from pilates.generic.runner import GenericRunner
from pilates.generic.records import RecordStore, FileRecord
from pilates.beam.postprocessor import (
    find_latest_beam_iteration,
    find_beam_iterations,
    find_iteration_file,
)
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker
from pilates.utils.settings_helper import get as get_setting

logger = logging.getLogger(__name__)


def find_not_taken_dir_name(dir_name):
    for x in range(1, 99999):
        testing_name = f"{dir_name}_{x}"
        if not os.path.exists(testing_name):
            return testing_name
    raise RuntimeError(f"Cannot find an appropriate not taken directory for {dir_name}")


def rename_beam_output_directory(
    beam_output_dir, settings, year, replanning_iteration_number=0
) -> (str, str):
    iteration_output_directory, _ = find_latest_beam_iteration(beam_output_dir)
    beam_run_output_dir = os.path.dirname(os.path.dirname(iteration_output_directory))
    new_iteration_output_directory = os.path.join(
        beam_output_dir,
        get_setting(settings, "run.region"),
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
    return beam_run_output_dir, new_iteration_output_directory


class BeamRunner(GenericRunner):
    """
    Runner for the BEAM model.
    """

    @staticmethod
    def expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> dict:
        return {
            "beam_mutable_data_dir": workspace.get_beam_mutable_data_dir(),
            "zarr_skims": os.path.join(
                workspace.get_asim_output_dir(), "cache", "skims.zarr"
            ),
        }

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: FileProvenanceTracker,
        major_stage: Optional["WorkflowState.Stage"] = None,
    ):
        super().__init__(model_name, state, provenance_tracker, major_stage)

    def gather_outputs(
        self,
        beam_local_output_folder: str,
        skimFormat: str = "omx",
    ) -> List[FileRecord]:
        files_to_get = {
            "raw_od_skims": ("skimsActivitySimOD_current", ".omx"),
            # Zarr OD skims naming differs across BEAM builds/configs:
            # - `{it}.skimsActivitySimOD_current.zarr` (observed in production)
            # - `{it}.activitySimODSkims_current.zarr` (observed in some builds)
            # We handle both below.
            "raw_od_skims_zarr": (None, ".zarr"),
            "raw_origin_skims": ("skimsRidehail", ".csv.gz"),
            "linkstats": ("linkstats", ".csv.gz"),
            "beam_plans_out": ("plans", ".csv.gz"),
            "events": ("events", ".csv.gz"),
            "events_parquet": ("events", ".parquet"),
        }

        output_records = []
        paths, last_iter = find_beam_iterations(beam_local_output_folder)
        for it, path in paths.items():
            for short_name, (file_name, extension) in files_to_get.items():
                if short_name == "raw_od_skims_zarr":
                    full_path = (
                        find_iteration_file(path, it, "skimsActivitySimOD_current", ".zarr")
                        or find_iteration_file(path, it, "activitySimODSkims_current", ".zarr")
                    )
                else:
                    full_path = find_iteration_file(path, it, file_name, extension)
                if full_path:
                    if it == last_iter:
                        dataset_name = f"{short_name}_{self.state.forecast_year}_{self.state.iteration}"
                    else:
                        dataset_name = f"{short_name}_{self.state.forecast_year}_{self.state.iteration}_sub{it}"
                    output_records.append(
                        FileRecord(
                            file_path=full_path,
                            year=self.state.forecast_year,
                            short_name=dataset_name,
                            description=f"BEAM output artifact: {dataset_name}",
                        )
                    )

        return output_records

    def get_beam_docker_vols(self, settings: PilatesConfig, workspace: Workspace):
        region = settings.run.region
        beam_local_input_folder = os.path.join(
            workspace.get_beam_mutable_data_dir(), region
        )
        beam_local_output_folder = os.path.join(workspace.get_beam_output_dir(), region)
        asim_local_output_folder = workspace.get_asim_output_dir()

        return {
            beam_local_input_folder: {"bind": "/app/input", "mode": "rw"},
            beam_local_output_folder: {"bind": "/app/output", "mode": "rw"},
            asim_local_output_folder: {
                "bind": "/app/activitysim-output",
                "mode": "rw",
            },
        }

    def _run(
        self,
        store: RecordStore,
        workspace: Workspace,
    ) -> RecordStore:
        settings = self.state.full_settings
        region = settings.run.region
        beam_memory = settings.beam.memory

        client = None  # Handled by Consist

        travel_model, travel_model_image = self.get_model_and_image(
            settings, "travel_model"
        )
        beam_config = settings.beam.config
        path_to_beam_config = f"/app/input/{region}/{beam_config}"

        abs_beam_input = workspace.get_beam_mutable_data_dir()
        abs_beam_output = workspace.get_beam_output_dir()

        # Make sure there's a temp dir for the JVM to use
        os.makedirs(os.path.join(abs_beam_output, "tmp"), exist_ok=True)

        logger.info(
            "[BEAM Runner] Starting BEAM container, input: %s, output: %s, config: %s",
            abs_beam_input,
            abs_beam_output,
            beam_config,
        )
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        java_opts = (
            # Memory settings (match Xms and Xmx)
            f"-Xms{beam_memory} "
            f"-Xmx{beam_memory} "
            # G1GC with more aggressive settings
            "-XX:+UseG1GC "
            "-XX:G1HeapRegionSize=32M "
            # More GC threads for burst capacity
            "-XX:ParallelGCThreads=36 "
            "-XX:ConcGCThreads=9 "
            "-XX:+UseNUMA "
            "-XX:+AlwaysPreTouch "
            # Flexible young gen - WIDE range for adaptability
            "-XX:+UnlockExperimentalVMOptions "
            "-XX:G1NewSizePercent=40 "  # Min 72 GB young gen
            "-XX:G1MaxNewSizePercent=60 "  # Max 108 GB young gen
            "-XX:MaxTenuringThreshold=6 "  # Objects die faster in young gen
            "-XX:SurvivorRatio=6 "  # 12.5% survivors (helps transit burst)
            "-XX:MaxGCPauseMillis=5000 "  # Accept 10s pauses for throughput
            "-XX:G1MixedGCCountTarget=12 "  # Spread old gen work
            # Conservative mixed GC - spread work over more cycles
            "-XX:G1MixedGCLiveThresholdPercent=65 "  # More conservative (was 50)
            # Earlier concurrent marking to avoid surprises
            "-XX:InitiatingHeapOccupancyPercent=30 "
            # More evacuation buffer for large populations
            "-XX:G1ReservePercent=15 "  # I (27GB reserve)
            # Less aggressive old gen collection
            "-XX:G1OldCSetRegionThresholdPercent=10 "  # Reduce from 15
            # GC logging
            f"-Xlog:gc*:file=/app/output/gc_{timestamp}.log:time,uptime,level,tags "
            f"-Xlog:gc+heap=debug:file=/app/output/heap-detail_{timestamp}.log "
            "-Djava.io.tmpdir=/app/output/tmp "
            "-Djna.tmpdir=/app/output/tmp"
        )

        # RecordStore may include non-file records (e.g., RepoRecord) when Consist-backed.
        # Consist container input lineage expects filesystem paths, so filter to FileRecord instances.
        from pilates.generic.records import FileRecord

        input_paths = [
            r.file_path for r in store.all_records() if isinstance(r, FileRecord)
        ]

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
            environment={"JAVA_OPTS": java_opts},
            input_artifacts=input_paths,
            output_paths=[abs_beam_output],
            lineage_mode="none",
        )

        if not success:
            raise RuntimeError("BEAM run failed after container execution.")

        output_path_for_gather: str
        try:
            old_path, new_path = rename_beam_output_directory(
                workspace.get_beam_output_dir(),
                settings,
                self.state.current_year,
                self.state.current_inner_iter,
            )
            output_path_for_gather = new_path
        except Exception as e:
            logger.error(
                f"Failed to rename BEAM output directory: {e}. Proceeding without rename."
            )
            output_path_for_gather = workspace.get_beam_output_dir()

        # ASSEMBLE OUTPUTS
        skims_fname = get_setting(settings, "shared.skims.fname")
        if skims_fname.endswith(".csv.gz"):
            skimFormat = "csv.gz"
        elif skims_fname.endswith(".omx"):
            skimFormat = "omx"
        else:
            skimFormat = "omx"
            logger.info(
                "[BEAM Runner] Defaulting to 'omx' skim format for finding BEAM outputs."
            )

        output_records = self.gather_outputs(output_path_for_gather, skimFormat)
        output_store = RecordStore(recordList=output_records)

        logger.info(
            f"[BEAM Runner] BEAM run complete. Output records: {len(output_records)}"
        )

        return output_store
