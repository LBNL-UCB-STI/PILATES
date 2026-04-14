import logging
import os
from datetime import datetime
from pathlib import Path

from typing import List, Optional, Dict, Any, Mapping

from pilates.config import PilatesConfig
from pilates.beam.outputs import (
    BeamFullSkimOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
)
from pilates.generic.runner import GenericRunner
from pilates.generic.records import RecordStore, FileRecord
from pilates.beam.postprocessor import (
    find_latest_beam_iteration,
    find_beam_iterations,
    find_iteration_file,
)
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.artifact_keys import (
    BEAM_CONFIG_FILE,
    BEAM_FULL_SKIMS,
    BEAM_MUTABLE_DATA_DIR,
    BEAM_NETWORK_FINAL,
    BEAM_OUTPUT_DIR,
)
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.settings_helper import get as get_setting
from pilates.activitysim.runner import asim_runtime_zarr_path

logger = logging.getLogger(__name__)


def _artifact_content_hash(value: Any) -> Optional[str]:
    if value is None:
        return None
    for attr_name in ("content_hash", "hash"):
        content_hash = getattr(value, attr_name, None)
        if content_hash:
            return str(content_hash)
    return None


def _append_artifact_mapping_records(
    record_store: RecordStore,
    artifact_mapping: Optional[Mapping[str, Any]],
    *,
    description_prefix: str,
) -> None:
    if not artifact_mapping:
        return
    for key, value in artifact_mapping.items():
        path = artifact_to_path(value, None)
        if path is None and isinstance(value, (str, os.PathLike)):
            path = os.fspath(value)
        if not path:
            continue
        record_store.add_record(
            FileRecord(
                file_path=str(path),
                short_name=key,
                description=f"{description_prefix}: {key}",
                content_hash=_artifact_content_hash(value),
            )
        )


def _calculate_optimal_parallelism(cpu_ratio: float = 0.8) -> int:
    """
    Calculate a conservative thread count for FullSkimsCreatorApp.
    """
    cpu_count = os.cpu_count() or 1
    cpu_based = max(1, int(cpu_count * cpu_ratio))

    memory_based = cpu_based
    try:
        import psutil  # type: ignore

        memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        memory_based = max(1, int(memory_gb / 2))
    except Exception:
        pass

    optimal = min(cpu_based, memory_based, 128)
    optimal = max(1, min(optimal, cpu_count))
    logger.info(
        "Calculated full-skim parallelism=%s from cpu_ratio=%.2f", optimal, cpu_ratio
    )
    return optimal


def _map_host_path_to_container(
    host_path: str, abs_beam_input: str, abs_beam_output: str
) -> str:
    """
    Map host paths into the BEAM container namespace.
    """
    if host_path.startswith(abs_beam_output):
        rel = os.path.relpath(host_path, abs_beam_output)
        return os.path.join("/app/output", rel)
    if host_path.startswith(abs_beam_input):
        rel = os.path.relpath(host_path, abs_beam_input)
        return os.path.join("/app/input", rel)
    return host_path


def _select_latest_linkstats_path(
    records: RecordStore,
    abs_beam_input: str,
    abs_beam_output: str,
) -> Optional[str]:
    """
    Select the newest linkstats artifact and map it to a container path.
    """
    if records is None:
        return None

    def _linkstats_rank(short_name: str) -> Optional[tuple[int, int, int]]:
        if "_sub" in short_name:
            return None
        if short_name == "linkstats":
            return (0, 0, 1)
        if short_name == "linkstats_parquet":
            return (0, 0, 0)

        for prefix, format_priority in (
            ("linkstats_", 1),
            ("linkstats_parquet_", 0),
        ):
            if not short_name.startswith(prefix):
                continue
            tail = short_name[len(prefix) :]
            parts = tail.split("_")
            if len(parts) != 2:
                continue
            try:
                year = int(parts[0])
                iteration = int(parts[1])
            except ValueError:
                continue
            return (year, iteration, format_priority)
        return None

    best_record: Optional[FileRecord] = None
    best_rank: Optional[tuple[int, int, int]] = None
    for record in records.all_records():
        short_name = getattr(record, "short_name", "") or ""
        rank = _linkstats_rank(short_name)
        if rank is None:
            continue
        if best_rank is None or rank >= best_rank:
            best_rank = rank
            best_record = record

    if best_record is None:
        return None

    path = best_record.file_path
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return _map_host_path_to_container(path, abs_beam_input, abs_beam_output)


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


def find_iteration_phys_sim_linkstats_parquet(
    iteration_path: str, iteration: int
) -> List[tuple[int, str]]:
    """
    Discover BEAM sub-iteration unmodified linkstats parquet files.

    Matches files like:
      {iteration}.linkstats_unmodified_physSimIter1.parquet
      {iteration}.linkstats_unmodified_physSimIter2.parquet
    """
    prefix = f"{iteration}.linkstats_unmodified_physSimIter"
    suffix = ".parquet"
    found: List[tuple[int, str]] = []

    try:
        filenames = os.listdir(iteration_path)
    except OSError:
        return found

    for filename in filenames:
        if not filename.startswith(prefix) or not filename.endswith(suffix):
            continue
        iter_token = filename[len(prefix) : -len(suffix)]
        try:
            phys_sim_iter = int(iter_token)
        except ValueError:
            continue
        found.append((phys_sim_iter, os.path.join(iteration_path, filename)))

    found.sort(key=lambda item: item[0])
    return found


class BeamRunner(GenericRunner):
    """
    Runner for the BEAM model.
    """

    @staticmethod
    def declared_expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this runner expects without disk checks.
        """
        zarr_path = None
        if getattr(settings, "activitysim", None) is not None:
            zarr_path = asim_runtime_zarr_path(workspace)
        return {
            "beam_mutable_data_dir": workspace.get_beam_mutable_data_dir(),
            "zarr_skims": zarr_path,
        }

    @staticmethod
    def runtime_expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare runtime expected inputs, including filesystem presence checks.
        """
        zarr_path = None
        if getattr(settings, "activitysim", None) is not None:
            candidate = asim_runtime_zarr_path(workspace)
            if os.path.exists(candidate):
                zarr_path = candidate
        return {
            "beam_mutable_data_dir": workspace.get_beam_mutable_data_dir(),
            "zarr_skims": zarr_path,
        }

    @staticmethod
    def expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        return BeamRunner.runtime_expected_inputs(settings, state, workspace)

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
    ):
        super().__init__(model_name, state)

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
            BEAM_NETWORK_FINAL: ("network", ".csv.gz"),
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

            for phys_sim_iter, full_path in find_iteration_phys_sim_linkstats_parquet(
                path, it
            ):
                facet = {
                    "artifact_family": "linkstats_unmodified_phys_sim_iter_parquet",
                    "year": self.state.forecast_year,
                    "iteration": self.state.iteration,
                    "phys_sim_iteration": phys_sim_iter,
                    # Keep sub-iteration facet on promoted final artifacts too.
                    # This preserves existing key semantics while improving facet-only analysis.
                    "beam_sub_iteration": it,
                }
                if it == last_iter:
                    dataset_name = (
                        "linkstats_unmodified_parquet__"
                        f"y{self.state.forecast_year}__i{self.state.iteration}"
                        f"__phys_sim_iter{phys_sim_iter}"
                    )
                else:
                    dataset_name = (
                        "linkstats_unmodified_parquet__"
                        f"y{self.state.forecast_year}__i{self.state.iteration}"
                        f"__phys_sim_iter{phys_sim_iter}__beam_sub_iter{it}"
                    )
                output_records.append(
                    FileRecord(
                        file_path=full_path,
                        year=self.state.forecast_year,
                        sub_iteration=None if it == last_iter else it,
                        short_name=dataset_name,
                        description=f"BEAM output artifact: {dataset_name}",
                        metadata={
                            "facet": facet,
                            "facet_schema_version": "v1",
                            "facet_index": True,
                        },
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
        if getattr(settings, "activitysim", None) is not None:
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
            "-XX:MaxGCPauseMillis=3000 "  # Accept 3s pauses for throughput
            # Conservative mixed GC - spread work over more cycles
            # Earlier concurrent marking to avoid surprises
            "-XX:InitiatingHeapOccupancyPercent=20 "
            # More evacuation buffer for large populations
            "-XX:G1ReservePercent=20 "  # I (27GB reserve)
            "-XX:+ParallelRefProcEnabled "
            # Less aggressive old gen collection
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

    def run(
        self,
        inputs: BeamPreprocessOutputs,
        workspace: Workspace,
        *,
        extra_inputs: Optional[Mapping[str, Any]] = None,
    ) -> BeamRunOutputs:
        """
        Run BEAM from typed preprocess outputs and return typed run outputs.
        """
        if not isinstance(inputs, BeamPreprocessOutputs):
            raise TypeError("BeamRunner.run expects BeamPreprocessOutputs")
        self.state.set_sub_stage_progress("runner")
        input_store = RecordStore(
            recordList=[
                FileRecord(
                    file_path=str(path),
                    short_name=short_name,
                    description=description,
                )
                for short_name, path, description in inputs._iter_record_items()
            ]
        )
        _append_artifact_mapping_records(
            input_store,
            extra_inputs,
            description_prefix="BEAM run extra input",
        )
        output_store = self._run(input_store, workspace)
        raw_outputs: Dict[str, Path] = {}
        for key, value in output_store.to_mapping().items():
            path = artifact_to_path(value, workspace)
            if path is None:
                continue
            raw_outputs[key] = Path(path)
        return BeamRunOutputs(
            beam_output_dir=Path(workspace.get_beam_output_dir()),
            raw_outputs=raw_outputs,
        )


class BeamFullSkimRunner(GenericRunner):
    """
    Runner for BEAM FullSkimsCreatorApp as a dedicated workflow step.
    """

    @staticmethod
    def expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this runner expects without disk checks.
        """
        return {
            BEAM_CONFIG_FILE: (
                Path(workspace.get_beam_mutable_data_dir())
                / settings.run.region
                / settings.beam.config
            ),
            BEAM_MUTABLE_DATA_DIR: workspace.get_beam_mutable_data_dir(),
            BEAM_OUTPUT_DIR: workspace.get_beam_output_dir(),
        }

    @staticmethod
    def expected_outputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the output paths/artifacts this runner produces.
        """
        year = getattr(state, "current_year", None)
        if year is None:
            year = getattr(state, "year", None)
        iteration = getattr(state, "current_inner_iter", None)
        if iteration is None:
            iteration = getattr(state, "iteration", 0)
        if year is None:
            return {}
        return {
            BEAM_FULL_SKIMS: (
                Path(workspace.get_beam_output_dir())
                / settings.run.region
                / f"year-{int(year)}-iteration-{int(iteration)}"
                / "skimsODFull.csv.gz"
            )
        }

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        super().__init__(model_name, state)

    def _run(
        self,
        store: RecordStore,
        workspace: Workspace,
    ) -> RecordStore:
        settings = self.state.full_settings
        region = settings.run.region
        beam_memory = settings.beam.memory

        beam_cfg = getattr(settings, "beam", None)
        skim_cfg = getattr(beam_cfg, "full_skim", None) if beam_cfg else None
        if skim_cfg is None:
            raise RuntimeError("BEAM full skim requested but beam.full_skim is not set.")
        if getattr(skim_cfg, "run_schedule", "disabled") == "disabled":
            raise RuntimeError(
                "BEAM full skim requested but beam.full_skim.run_schedule is disabled."
            )

        abs_beam_input = workspace.get_beam_mutable_data_dir()
        abs_beam_output = workspace.get_beam_output_dir()

        output_dir = os.path.join(
            abs_beam_output,
            region,
            f"year-{self.state.current_year}-iteration-{self.state.current_inner_iter}",
        )
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(abs_beam_output, "tmp"), exist_ok=True)

        _, travel_model_image = self.get_model_and_image(settings, "travel_model")
        beam_config = settings.beam.config
        path_to_beam_config = f"/app/input/{region}/{beam_config}"

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        java_opts = (
            f"-Xms{beam_memory} "
            f"-Xmx{beam_memory} "
            "-XX:+UseG1GC "
            "-XX:G1HeapRegionSize=32M "
            "-XX:ParallelGCThreads=36 "
            "-XX:ConcGCThreads=9 "
            "-XX:+UseNUMA "
            "-XX:+AlwaysPreTouch "
            "-XX:+UnlockExperimentalVMOptions "
            "-XX:G1NewSizePercent=40 "
            "-XX:G1MaxNewSizePercent=60 "
            "-XX:MaxTenuringThreshold=6 "
            "-XX:SurvivorRatio=6 "
            "-XX:MaxGCPauseMillis=5000 "
            "-XX:G1MixedGCCountTarget=12 "
            "-XX:G1MixedGCLiveThresholdPercent=65 "
            "-XX:InitiatingHeapOccupancyPercent=30 "
            "-XX:G1ReservePercent=15 "
            "-XX:G1OldCSetRegionThresholdPercent=10 "
            f"-Xlog:gc*:file=/app/output/gc_{timestamp}.log:time,uptime,level,tags "
            f"-Xlog:gc+heap=debug:file=/app/output/heap-detail_{timestamp}.log "
            "-Djava.io.tmpdir=/app/output/tmp "
            "-Djna.tmpdir=/app/output/tmp"
        )

        cpu_ratio = (
            skim_cfg.parallelism_thread_ratio
            if skim_cfg.parallelism_thread_ratio is not None
            else 0.8
        )
        parallelism = _calculate_optimal_parallelism(cpu_ratio)

        host_output_file = os.path.join(output_dir, "skimsODFull.csv.gz")
        container_output_file = _map_host_path_to_container(
            host_output_file, abs_beam_input, abs_beam_output
        )

        cmd_parts = [
            f"--configPath={path_to_beam_config}",
            f"--output={container_output_file}",
            f"--parallelism={parallelism}",
            f"--routerType={skim_cfg.router_type}",
            f"--skimsGeoType={skim_cfg.skims_geo_type}",
            f"--skimsKind={skim_cfg.skims_kind}",
            f"--peakHours={','.join(str(hour) for hour in skim_cfg.peak_hours)}",
        ]

        enabled_modes = [
            mode for mode, enabled in skim_cfg.modes_to_build.items() if enabled
        ]
        if enabled_modes:
            cmd_parts.append(f"--modesToBuild={','.join(enabled_modes)}")

        linkstats_path = _select_latest_linkstats_path(
            store, abs_beam_input, abs_beam_output
        )
        if linkstats_path is None:
            router_dir = getattr(settings.beam, "router_directory", None)
            if router_dir:
                init_parquet = os.path.join(
                    abs_beam_input, region, router_dir, "init.linkstats.parquet"
                )
                init_csv = os.path.join(
                    abs_beam_input, region, router_dir, "init.linkstats.csv.gz"
                )
                if os.path.exists(init_parquet):
                    linkstats_path = _map_host_path_to_container(
                        init_parquet, abs_beam_input, abs_beam_output
                    )
                elif os.path.exists(init_csv):
                    linkstats_path = _map_host_path_to_container(
                        init_csv, abs_beam_input, abs_beam_output
                    )
        if linkstats_path is not None:
            cmd_parts.append(f"--linkstatsPath={linkstats_path}")

        input_paths = [
            record.file_path
            for record in store.all_records()
            if isinstance(record, FileRecord)
        ]

        success = self.run_container(
            client=None,
            settings=settings,
            image=travel_model_image,
            volumes={
                abs_beam_input: {"bind": "/app/input", "mode": "rw"},
                abs_beam_output: {"bind": "/app/output", "mode": "rw"},
            },
            command=" ".join(cmd_parts),
            model_name=self.model_name,
            working_dir="/app",
            environment={
                "JAVA_OPTS": java_opts,
                "BEAM_MAIN_CLASS": "scripts.FullSkimsCreatorApp",
            },
            input_artifacts=input_paths,
            output_paths=[abs_beam_output],
            lineage_mode="none",
        )
        if not success:
            raise RuntimeError("BEAM full skim run failed after container execution.")
        if not os.path.exists(host_output_file):
            raise RuntimeError(
                f"BEAM full skim completed but output was not found: {host_output_file}"
            )

        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=host_output_file,
                    year=self.state.forecast_year,
                    short_name=BEAM_FULL_SKIMS,
                    description="BEAM full-skim background skims output",
                )
            ]
        )

    def run(
        self,
        inputs: BeamPreprocessOutputs,
        workspace: Workspace,
        *,
        previous_beam_outputs: Optional[Mapping[str, Any]] = None,
    ) -> BeamFullSkimOutputs:
        """
        Run FullSkimsCreatorApp from typed preprocess outputs and optional warm-start artifacts.
        """
        if not isinstance(inputs, BeamPreprocessOutputs):
            raise TypeError("BeamFullSkimRunner.run expects BeamPreprocessOutputs")
        self.state.set_sub_stage_progress("runner")
        input_store = RecordStore(
            recordList=[
                FileRecord(
                    file_path=str(path),
                    short_name=short_name,
                    description=description,
                )
                for short_name, path, description in inputs._iter_record_items()
            ]
        )
        _append_artifact_mapping_records(
            input_store,
            previous_beam_outputs,
            description_prefix="BEAM full-skim warm-start input",
        )
        output_store = self._run(input_store, workspace)
        full_skims_path = artifact_to_path(
            output_store.to_mapping().get(BEAM_FULL_SKIMS), workspace
        )
        if full_skims_path is None:
            raise ValueError("Missing beam_full_skims in run outputs.")
        return BeamFullSkimOutputs(full_skims=Path(full_skims_path))
