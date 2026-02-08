import logging
import os
from datetime import datetime

import sys
from typing import List, Optional, Dict, Any

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
from pilates.utils.settings_helper import get as get_setting

logger = logging.getLogger(__name__)


def _calculate_optimal_parallelism(cpu_ratio: float = 0.8) -> int:
    """
    Calculate optimal parallelism for skim generation based on available resources.

    Parameters
    ----------
    cpu_ratio : float
        Ratio of CPU cores to use (0.0-1.0). Default is 0.8 (80%).

    Returns
    -------
    int
        Recommended number of parallel threads
    """
    import multiprocessing
    import psutil

    # Get available CPU cores
    cpu_count = multiprocessing.cpu_count()

    # Get available memory in GB
    memory_gb = psutil.virtual_memory().available / (1024 ** 3)

    # Calculate based on CPU using the specified ratio
    cpu_based = max(1, int(cpu_count * cpu_ratio))

    # Calculate based on memory (assume ~2 GB per thread)
    memory_based = max(1, int(memory_gb / 2))

    # Take the minimum to avoid oversubscription
    optimal = min(cpu_based, memory_based)

    # Cap at reasonable maximum (diminishing returns beyond this)
    optimal = min(optimal, 128)

    # Ensure minimum of 4 threads (unless severely resource constrained)
    optimal = max(optimal, min(4, cpu_count))

    logger.info(
        f"Auto-calculated parallelism: {optimal} ({cpu_ratio:.1%} of {cpu_count} cores, "
        f"Memory: {memory_gb:.1f} GB available)"
    )

    return optimal


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
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this runner expects from the workflow.
        """
        zarr_path = None
        if getattr(settings, "activitysim", None) is not None:
            candidate = os.path.join(workspace.get_asim_output_dir(), "cache", "skims.zarr")
            if os.path.exists(candidate):
                zarr_path = candidate
        return {
            "beam_mutable_data_dir": workspace.get_beam_mutable_data_dir(),
            "zarr_skims": zarr_path,
        }

    @staticmethod
    def expected_outputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the output paths/artifacts this runner produces.

        Notes
        -----
        Output keys
            - ``beam_output_dir``: BEAM output directory for the run.
        Related docs
            - See `pilates/beam/inputs.py` for the corresponding input
              descriptions used by BEAM and downstream models.
        """
        return {"beam_output_dir": workspace.get_beam_output_dir()}

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        major_stage: Optional["WorkflowState.Stage"] = None,
    ):
        super().__init__(model_name, state, major_stage)

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
            "linkstats_unmodified": ("linkstats_unmodified", ".csv.gz"),
            "linkstats_parquet": ("linkstats", ".parquet"),
            "linkstats_unmodified_parquet": ("linkstats_unmodified", ".parquet"),
            "beam_plans_out": ("plans", ".csv.gz"),
            "beam_plans_xml": ("plans", ".xml.gz"),
            "beam_experienced_plans_xml": ("experienced_plans", ".xml.gz"),
            "beam_experienced_plans_scores": ("experienced_plans_scores", ".txt.gz"),
            "events": ("events", ".csv.gz"),
            "events_parquet": ("events", ".parquet"),
            "legs": ("legs", ".csv.gz"),
            "route_history": ("routeHistory", ".csv.gz"),
            "final_vehicles": ("final_vehicles", ".csv.gz"),
            "skims_taz": ("skimsTAZ", ".csv.gz"),
            "skims_taz_agg": ("skimsTAZ_Aggregated", ".csv.gz"),
            "skims_od": ("skimsOD", ".csv.gz"),
            "skims_od_agg": ("skimsOD_Aggregated", ".csv.gz"),
            "skims_od_vehicle_type": ("skimsODVehicleType", ".csv.gz"),
            "skims_od_vehicle_type_agg": ("skimsODVehicleType_Aggregated", ".csv.gz"),
            "skims_emissions": ("skimsEmissions", ".csv.gz"),
            "skims_emissions_agg": ("skimsEmissions_Aggregated", ".csv.gz"),
            "skims_ridehail_agg": ("skimsRidehail_Aggregated", ".csv.gz"),
            "skims_parking": ("skimsParking", ".csv.gz"),
            "skims_parking_agg": ("skimsParking_Aggregated", ".csv.gz"),
            "skims_transit_crowding": ("skimsTransitCrowding", ".csv.gz"),
            "skims_transit_crowding_agg": (
                "skimsTransitCrowding_Aggregated",
                ".csv.gz",
            ),
            "skims_freight": ("skimsFreight", ".csv.gz"),
            "skims_freight_agg": ("skimsFreight_Aggregated", ".csv.gz"),
            "skims_travel_time_obs_sim": (
                "skimsTravelTimeObservedVsSimulated",
                ".csv.gz",
            ),
            "skims_travel_time_obs_sim_agg": (
                "skimsTravelTimeObservedVsSimulated_Aggregated",
                ".csv.gz",
            ),
        }
        top_level_files = {
            "beam_plans_final": ("plans", ".csv.gz"),
            "beam_vehicles_final": ("vehicles", ".csv.gz"),
            "beam_households_final": ("households", ".csv.gz"),
            "beam_persons_final": ("output_persons", ".csv.gz"),
            "beam_population_final": ("population", ".csv.gz"),
            "beam_network_final": ("network", ".csv.gz"),
            "beam_output_plans_xml": ("output_plans", ".xml.gz"),
            "beam_output_experienced_plans_xml": (
                "output_experienced_plans",
                ".xml.gz",
            ),
            "beam_output_vehicles_xml": ("output_vehicles", ".xml.gz"),
            "beam_output_households_xml": ("output_households", ".xml.gz"),
            "beam_output_facilities_xml": ("output_facilities", ".xml.gz"),
            "beam_output_network_xml": ("output_network", ".xml.gz"),
            "beam_output_counts_xml": ("output_counts", ".xml.gz"),
        }

        output_records = []
        paths, last_iter = find_beam_iterations(beam_local_output_folder)
        for it, path in paths.items():
            for short_name, (file_name, extension) in files_to_get.items():
                if short_name == "raw_od_skims_zarr":
                    full_path = find_iteration_file(
                        path, it, "skimsActivitySimOD_current", ".zarr"
                    ) or find_iteration_file(
                        path, it, "activitySimODSkims_current", ".zarr"
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

        for short_name, (file_name, extension) in top_level_files.items():
            full_path = os.path.join(
                beam_local_output_folder, f"{file_name}{extension}"
            )
            if not os.path.exists(full_path):
                continue
            dataset_name = (
                f"{short_name}_{self.state.forecast_year}_{self.state.iteration}"
            )
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
        vols = {
            beam_local_input_folder: {"bind": "/app/input", "mode": "rw"},
            beam_local_output_folder: {"bind": "/app/output", "mode": "rw"},
        }
        if settings.activitysim is not None:
            asim_local_output_folder = workspace.get_asim_output_dir()
            vols[asim_local_output_folder] = {
                "bind": "/app/activitysim-output",
                "mode": "rw",
            }
        return vols

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

        # Determine if skim-only mode is enabled via configuration
        environment = {"JAVA_OPTS": java_opts}
        command = f"--config={path_to_beam_config}"
        beam_cfg = getattr(settings, "beam", None)
        if beam_cfg and beam_cfg.skim_only and beam_cfg.skim_only.enabled:
            # Override main class for skim-only mode
            environment["BEAM_MAIN_CLASS"] = "scripts.BackgroundSkimsCreatorApp"
            skim_cfg = beam_cfg.skim_only

            # Calculate parallelism based on CPU ratio (default 0.8 = 80% if not specified)
            cpu_ratio = (
                skim_cfg.parallelism_thread_ratio
                if skim_cfg.parallelism_thread_ratio is not None
                else 0.8
            )
            parallelism = _calculate_optimal_parallelism(cpu_ratio)

            # Build command arguments for the BackgroundSkimsCreatorApp
            output_path = os.path.join(abs_beam_output, skim_cfg.output_filename)
            cmd_parts = [
                f"--configPath={path_to_beam_config}",
                f"--output={output_path}",
                f"--parallelism={parallelism}",
                f"--routerType={skim_cfg.router_type}",
                f"--skimsGeoType={skim_cfg.skims_geo_type}",
                f"--skimsKind={skim_cfg.skims_kind}",
            ]

            # Format peak hours as comma-separated list
            peak_hours_str = ",".join(str(h) for h in skim_cfg.peak_hours)
            cmd_parts.append(f"--peakHours={peak_hours_str}")

            # Format modes to build as comma-separated list of enabled modes
            enabled_modes = [mode for mode, enabled in skim_cfg.modes_to_build.items() if enabled]
            if enabled_modes:
                modes_str = ",".join(enabled_modes)
                cmd_parts.append(f"--modesToBuild={modes_str}")

            if skim_cfg.linkstats_file:
                # Prepend the region directory so the path resolves inside the container
                region = get_setting(settings, "run.region")
                # Build an absolute container‑side path: /app/input/<region>/<linkstats_file>
                linkstats_path = os.path.join("/app/input", region, skim_cfg.linkstats_file)
                cmd_parts.append(f"--linkstatsPath={linkstats_path}")
            command = " ".join(cmd_parts)

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
            command=command,
            model_name=self.model_name,
            working_dir="/app",
            environment=environment,
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
