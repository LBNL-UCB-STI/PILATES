import logging
import os
from pathlib import Path
from typing import Tuple, Optional

from pilates.config import PilatesConfig
from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.generic.runner import GenericRunner
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker
from pilates.utils.snapshot_manager import SnapshotManager
from pilates.utils.duckdb_manager import DuckDBManager
from pilates.utils.zone_utils import ensure_0_based_and_flag_zarr_skims

logger = logging.getLogger(__name__)


class ActivitysimRunner(GenericRunner):
    """
    Runner for ActivitySim model.
    """

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: FileProvenanceTracker,
        major_stage: Optional["WorkflowState.Stage"] = None,
    ):
        super().__init__(model_name, state, provenance_tracker, major_stage)
        self.required_input_files = [
            "persons_asim_in",
            "households_asim_in",
            "land_use_asim_in",
            "omx_skims",
            "zarr_skims",
            "asim_geoms",
            "asim_configs",
        ]

    @staticmethod
    def get_base_asim_cmd(
        settings: PilatesConfig, household_sample_size=None, num_processes=None
    ):
        formattable_asim_cmd = settings.activitysim.command_template
        if not household_sample_size:
            household_sample_size = settings.activitysim.household_sample_size
        num_processes = num_processes or settings.activitysim.num_processes
        chunk_size = settings.activitysim.chunk_size  # default no chunking
        base_asim_cmd = formattable_asim_cmd.format(
            household_sample_size, num_processes, chunk_size
        )
        return base_asim_cmd

    @staticmethod
    def get_asim_additional_args(settings: PilatesConfig, asim_docker_vols, compile):
        additional_args = []
        if settings.activitysim.file_format == "parquet":
            additional_args.append("--persist-sharrow-cache")
            for local, d in asim_docker_vols.items():
                if "data" in d["bind"]:
                    additional_args.append("-d")
                    additional_args.append(d["bind"])
                elif "output" in d["bind"]:
                    additional_args.append("-o")
                    additional_args.append(d["bind"])
                elif "compile" in d["bind"]:
                    if compile:
                        additional_args.append("-c")
                        additional_args.append(d["bind"])
                elif "configs" in d["bind"]:
                    additional_args.append("-c")
                    additional_args.append(d["bind"])
        return additional_args

    @staticmethod
    def get_asim_docker_vols(settings: PilatesConfig, working_dir=None):
        region = settings.run.region
        asim_subdir = settings.activitysim.region_mappings["region_to_subdir"][region]
        asim_remote_workdir = os.path.join("/activitysim", asim_subdir)
        if working_dir is not None:
            asim_local_mutable_data_folder = os.path.abspath(
                os.path.join(
                    working_dir, settings.activitysim.local_mutable_data_folder
                )
            )
            asim_local_output_folder = os.path.abspath(
                os.path.join(working_dir, settings.activitysim.local_output_folder)
            )
            asim_local_configs_folder = os.path.abspath(
                os.path.join(
                    working_dir,
                    settings.activitysim.local_mutable_configs_folder,
                    settings.activitysim.main_configs_dir,
                )
            )
            asim_local_configs_compile_folder = os.path.abspath(
                os.path.join(
                    working_dir,
                    settings.activitysim.local_mutable_configs_folder,
                    "configs_sh_compile",
                )
            )
        else:
            asim_local_mutable_data_folder = os.path.abspath(
                settings.activitysim.local_mutable_data_folder
            )
            asim_local_output_folder = os.path.abspath(
                settings.activitysim.local_output_folder
            )
            asim_local_configs_folder = os.path.abspath(
                os.path.join(
                    settings.activitysim.local_configs_folder, region, "configs"
                )
            )
            asim_local_configs_compile_folder = os.path.abspath(
                os.path.join(
                    settings.activitysim.local_configs_folder,
                    region,
                    "configs_sh_compile",
                )
            )
        asim_remote_input_folder = os.path.join(asim_remote_workdir, "data")
        asim_remote_output_folder = os.path.join(asim_remote_workdir, "output")
        asim_remote_configs_folder = os.path.join(asim_remote_workdir, "configs")
        asim_remote_configs_compile_folder = os.path.join(
            asim_remote_workdir, "configs_sh_compile"
        )
        asim_docker_vols = {
            asim_local_mutable_data_folder: {
                "bind": asim_remote_input_folder,
                "mode": "rw",
            },
            asim_local_output_folder: {"bind": asim_remote_output_folder, "mode": "rw"},
            asim_local_configs_compile_folder: {
                "bind": asim_remote_configs_compile_folder,
                "mode": "rw",
            },
            asim_local_configs_folder: {
                "bind": asim_remote_configs_folder,
                "mode": "rw",
            },
        }
        return asim_docker_vols

    def _parse_year_iteration_from_short_name(self, short_name: str) -> Tuple[int, int]:
        parts = short_name.split("_")
        if len(parts) >= 3 and parts[0] == "zarr" and parts[1] == "skims":
            try:
                year = int(parts[2])
                iteration = int(parts[3])
                return year, iteration
            except ValueError:
                pass
        return 0, 0  # Default or error case

    def _run(
        self,
        store: RecordStore,
        workspace: Workspace,
    ) -> Tuple[RecordStore, ModelRunInfo]:
        """
        Do the model run

        Args:
            store (RecordStore): The input data generated by the preprocessor.
            workspace (Workspace): The workspace object for path management.

        Returns:
            tuple: A tuple containing:
                - data (RecordStore): The raw output files that have been prepared to run the model
                - model_run_info (ModelRunInfo): Information about the model run
        """
        settings = self.state.full_settings
        region = settings.run.region
        asim_subdir = settings.activitysim.region_mappings["region_to_subdir"][region]
        asim_workdir = os.path.join("activitysim", asim_subdir)

        # self.setup_container_cache_dirs(settings) # Handled by Consist

        # Get from your config
        output_directory = (
            settings.run.output_directory
        )  # "/global/scratch/users/$USER/pilates-output"

        # Expand $USER if needed
        output_directory = os.path.expandvars(output_directory)

        # Create shared cache and tmp at the base pilates-output level
        shared_cache_dir = os.path.join(output_directory, "shared_cache")
        shared_tmp_dir = os.path.join(output_directory, "tmp")

        # Create them
        os.makedirs(os.path.join(shared_cache_dir, "numba"), exist_ok=True)
        os.makedirs(shared_tmp_dir, exist_ok=True)

        client = None  # Handled by Consist

        asim_docker_vols = self.get_asim_docker_vols(
            settings, working_dir=workspace.full_path
        )

        asim_docker_vols.update(
            {
                shared_tmp_dir: {"bind": "/tmp", "mode": "rw"},
                shared_cache_dir: {"bind": "/app/numba_cache", "mode": "rw"},
            }
        )

        activity_demand_model, activity_demand_image = self.get_model_and_image(
            settings, "activity_demand_model"
        )

        asim_compile_run_hash = self.provenance_tracker.run_info.get_latest_model_run(
            "activitysim"
        )

        filtered_store = RecordStore(
            recordList=[
                rec
                for rec in store.all_records()
                if rec.short_name in self.required_input_files
            ]
        )

        all_skims_path = os.path.join(
            workspace.get_asim_output_dir(), "cache", "skims.zarr"
        )

        asim_local_output_folder = os.path.abspath(
            os.path.join(workspace.full_path, settings.activitysim.local_output_folder)
        )

        os.makedirs(
            os.path.join(asim_local_output_folder, "cache", "numba"), exist_ok=True
        )

        compiled_asim_this_year = False

        if not self.state.asim_compiled:

            compiled_asim_this_year = True

            asim_cmd = self.get_base_asim_cmd(
                settings, household_sample_size=2500, num_processes=1
            )

            additional_args = self.get_asim_additional_args(
                settings, asim_docker_vols, True
            )

            success = self.run_container(
                client=client,
                settings=settings,
                image=activity_demand_image,
                volumes=asim_docker_vols,
                command=asim_cmd,
                model_name="activitysim",
                working_dir=asim_workdir,
                args=additional_args,
                environment={
                    "NUMBA_CACHE_DIR": "/app/numba_cache/numba",
                    "XDG_CACHE_HOME": "/app/numba_cache",
                },
                provenance_tracker=self.provenance_tracker,
                output_paths=[all_skims_path],
                lineage_mode="none",
            )

            output_records = []

            # Record output files for this compilation run
            if os.path.exists(all_skims_path):
                from pilates.generic.records import FileRecord

                zarr_skims_rec = FileRecord(
                    file_path=all_skims_path,
                    models=["activitysim"],
                    year=self.state.current_year,
                    description="Zarr skims initialized from omx.",
                    short_name=f"zarr_skims_{self.state.current_year}_-1",
                )
                output_records.append(zarr_skims_rec)
                logger.info(f"Using zarr skims from ASIM compilation: {all_skims_path}")

            # Create initial zarr version snapshot after compilation
            if success and os.path.exists(all_skims_path):
                try:
                    logger.info(
                        "Creating initial zarr version snapshot after ActivitySim compilation..."
                    )
                    database_path_str = settings.shared.database.path
                    if database_path_str:
                        db_manager = DuckDBManager(database_path_str)

                        archive_root_str = settings.shared.database.shapshot_path
                        archive_root_path = (
                            Path(archive_root_str) if archive_root_str else None
                        )

                        snapshot_manager = SnapshotManager(
                            db_manager, archive_root_path=archive_root_path
                        )

                        snapshot_id = snapshot_manager.create_snapshot(
                            run_id=self.provenance_tracker.run_info.run_id,
                            year=self.state.current_year,
                            iteration=-1,
                            sub_iteration=None,
                            model="activitysim",
                            snapshot_type="initialization",
                            source_path=Path(all_skims_path),
                            artifact_format="zarr",
                            provenance_tracker=self.provenance_tracker,
                        )
                        logger.info(f"Created initial zarr snapshot: {snapshot_id}")
                    else:
                        logger.debug(
                            "Database path not configured, skipping zarr snapshot creation"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to create initial zarr snapshot: {e}. Continuing without snapshot.",
                        exc_info=True,
                    )

            logger.info("ASIM Compilation success: {0}".format(success))
            if not success:
                raise RuntimeError("ASim Compilation failed")
            else:
                try:
                    ensure_0_based_and_flag_zarr_skims(
                        all_skims_path, settings, workspace
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to correct and flag initial Zarr skims after compilation: {e}",
                        exc_info=True,
                    )
                    raise RuntimeError(
                        "Failed to correct initial Zarr skims, cannot proceed."
                    ) from e
            self.state.compile_asim()  # Update state to mark as compiled
        else:
            if "beam" in self.provenance_tracker.run_info.models_used:
                last_beam_post_records = self.provenance_tracker.run_info.get_latest_model_run_output_records(
                    "beam_postprocessor"
                )

                zarr_skims_recs = sorted(
                    [
                        r
                        for r in last_beam_post_records
                        if r.short_name.startswith("zarr_skims")
                    ],
                    key=lambda r: self._parse_year_iteration_from_short_name(
                        r.short_name
                    ),
                )
                if zarr_skims_recs:
                    zarr_skims_rec = zarr_skims_recs[-1]
                    logger.info(
                        f"Using zarr skims from last BEAM postprocessor run: {zarr_skims_rec.file_path}"
                    )
                else:
                    logger.warning(
                        "No zarr skims found as inputs to the previous ASIM run. OMX skims will be used."
                    )
                    zarr_skims_rec = None
            else:
                last_asim_run_hash = (
                    self.provenance_tracker.run_info.get_latest_model_run("activitysim")
                )
                if last_asim_run_hash:
                    last_asim_run_input_hashes = (
                        self.provenance_tracker.run_info.model_runs[
                            last_asim_run_hash
                        ].input_record_hashes
                    )
                    last_asim_run_input_records = [
                        self.provenance_tracker.run_info.file_records.get(h)
                        for h in last_asim_run_input_hashes
                    ]
                    # Filter and sort to find the most recent versioned zarr_skims
                    zarr_skims_recs = sorted(
                        [
                            r
                            for r in last_asim_run_input_records
                            if r and r.short_name.startswith("zarr_skims")
                        ],
                        key=lambda r: self._parse_year_iteration_from_short_name(
                            r.short_name
                        ),
                    )
                    if zarr_skims_recs:
                        zarr_skims_rec = zarr_skims_recs[-1]
                        logger.info(
                            f"Using zarr skims that were inputs to the previous ASIM run: {zarr_skims_rec.file_path}"
                        )
                    else:
                        logger.warning(
                            "No zarr skims found as inputs to the previous ASIM run. OMX skims will be used."
                        )
                        zarr_skims_rec = None
                else:
                    logger.warning(
                        "No previous ASIM run found. OMX skims will be used."
                    )
                    zarr_skims_rec = None

        if zarr_skims_rec:
            filtered_store.remove_record_type("omx_skims")
            filtered_store.add_record(zarr_skims_rec)
        else:
            logger.warning(
                "No ASIM skims cache found at: {0}. OMX skims will be used.".format(
                    all_skims_path
                )
            )

        asim_cmd = self.get_base_asim_cmd(settings)

        additional_args = self.get_asim_additional_args(
            settings, asim_docker_vols, False
        )

        success = self.run_container(
            client=client,
            settings=settings,
            image=activity_demand_image,
            volumes=asim_docker_vols,
            command=asim_cmd,
            model_name="activitysim",
            working_dir=asim_workdir,
            args=additional_args,
            environment={
                "NUMBA_CACHE_DIR": "/app/numba_cache/numba",
                "XDG_CACHE_HOME": "/app/numba_cache",
            },
            provenance_tracker=self.provenance_tracker,
            output_paths=[workspace.get_asim_output_dir()],
            lineage_mode="none",
        )

        run_info = None
        if not success:
            logger.error(
                "ASIM run failed for year {0} iteration {1}".format(
                    self.state.current_year, self.state.current_inner_iter
                )
            )
            return RecordStore(), run_info

        # Assemble outputs: find the expected output files and return as a RecordStore
        output_dir = os.path.join(workspace.get_asim_output_dir(), "final_pipeline")
        output_records = []
        if os.path.exists(output_dir):
            for fname in os.listdir(output_dir):
                fpath = os.path.join(output_dir, fname, "final.parquet")
                if os.path.isfile(fpath):
                    # Record output files for this full run
                    from pilates.generic.records import FileRecord

                    output_rec = FileRecord(
                        file_path=fpath,
                        models=["activitysim"],
                        year=self.state.forecast_year,
                        description=f"ActivitySim output file: {fname}",
                        short_name=fname + "_asim_out_temp",
                        iteration=self.state.current_inner_iter,
                    )
                    output_records.append(output_rec)

        # Prepare runtime metadata for main run
        runtime_metadata = {
            "container_command": asim_cmd,
            "runtime_parameters": {
                "household_sample_size": settings.activitysim.household_sample_size,
                "num_processes": settings.activitysim.num_processes,
                "chunk_size": settings.activitysim.chunk_size,
                "additional_args": additional_args,
            },
            "container_image": activity_demand_image,
            "container_manager": settings.infrastructure.container_manager,
            "working_directory": asim_workdir,
        }

        output_store = RecordStore(recordList=output_records)

        # Provide minimal ModelRunInfo for downstream consumers.
        if run_info is None:
            run_info = ModelRunInfo(
                model="activitysim",
                year=self.state.forecast_year,
                iteration=self.state.current_inner_iter,
                description="ActivitySim run",
                status="completed" if success else "failed",
                metadata=runtime_metadata,
            )

        return output_store, run_info
