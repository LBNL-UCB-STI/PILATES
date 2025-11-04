import logging
import multiprocessing
import os
from typing import Tuple, Optional

from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.generic.runner import GenericRunner
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker
from pilates.utils.zarr_versioning import VersionedZarrStore
from pilates.utils.settings_helper import get as get_setting

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
    def get_base_asim_cmd(settings, household_sample_size=None, num_processes=None):
        formattable_asim_cmd = settings["asim_formattable_command"]
        if not household_sample_size:
            household_sample_size = settings.get("household_sample_size", 0)
        num_processes = num_processes or settings.get(
            "num_processes", multiprocessing.cpu_count() - 1
        )
        chunk_size = settings.get("chunk_size", 0)  # default no chunking
        base_asim_cmd = formattable_asim_cmd.format(
            household_sample_size, num_processes, chunk_size
        )
        return base_asim_cmd

    @staticmethod
    def get_asim_additional_args(settings, asim_docker_vols, compile):
        additional_args = []
        if settings.get("file_format", "parquet") == "parquet":
            additional_args.append("--persist-sharrow-cache")
            for local, d in asim_docker_vols.items():
                if "data" in d["bind"]:
                    additional_args.append("-d")
                    additional_args.append('"{0}"'.format(d["bind"]))
                elif "output" in d["bind"]:
                    additional_args.append("-o")
                    additional_args.append('"{0}"'.format(d["bind"]))
                elif "compile" in d["bind"]:
                    if compile:
                        additional_args.append("-c")
                        additional_args.append('"{0}"'.format(d["bind"]))
                elif "configs" in d["bind"]:
                    additional_args.append("-c")
                    additional_args.append('"{0}"'.format(d["bind"]))
        return additional_args

    @staticmethod
    def get_asim_docker_vols(settings, working_dir=None):
        region = get_setting(settings, "run.region")
        asim_subdir = settings["region_to_asim_subdir"][region]
        asim_remote_workdir = os.path.join("/activitysim", asim_subdir)
        if working_dir is not None:
            asim_local_mutable_data_folder = os.path.abspath(
                os.path.join(working_dir, settings["asim_local_mutable_data_folder"])
            )
            asim_local_output_folder = os.path.abspath(
                os.path.join(working_dir, settings["asim_local_output_folder"])
            )
            asim_local_configs_folder = os.path.abspath(
                os.path.join(
                    working_dir,
                    settings["asim_local_mutable_configs_folder"],
                    settings.get("asim_main_configs_dir", "configs"),
                )
            )
            asim_local_configs_compile_folder = os.path.abspath(
                os.path.join(
                    working_dir,
                    settings["asim_local_mutable_configs_folder"],
                    "configs_sh_compile",
                )
            )
        else:
            asim_local_mutable_data_folder = os.path.abspath(
                settings["asim_local_mutable_data_folder"]
            )
            asim_local_output_folder = os.path.abspath(
                settings["asim_local_output_folder"]
            )
            asim_local_configs_folder = os.path.abspath(
                os.path.join(settings["asim_local_configs_folder"], region, "configs")
            )
            asim_local_configs_compile_folder = os.path.abspath(
                os.path.join(
                    settings["asim_local_configs_folder"], region, "configs_sh_compile"
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
        region = get_setting(settings, "run.region")
        asim_subdir = settings["region_to_asim_subdir"][region]
        asim_workdir = os.path.join("activitysim", asim_subdir)

        # Get from your config
        output_directory = settings['output_directory']  # "/global/scratch/users/$USER/pilates-output"

        # Expand $USER if needed
        output_directory = os.path.expandvars(output_directory)

        # Create shared cache and tmp at the base pilates-output level
        shared_cache_dir = os.path.join(output_directory, "shared_cache")
        shared_tmp_dir = os.path.join(output_directory, "tmp")

        # Create them
        os.makedirs(os.path.join(shared_cache_dir, "numba"), exist_ok=True)
        os.makedirs(shared_tmp_dir, exist_ok=True)

        # start docker client
        client = None  # Initialize client to None
        if get_setting(settings, "infrastructure.container_manager") == "docker":
            try:
                client = self.initialize_docker_client(settings)
            except Exception as e:
                logger.error(f"Failed to initialize Docker client: {e}")
                # Decide how to handle failure - maybe exit?
                # For now, log and continue, assuming Singularity might be used or stubs.
                # If no client and no singularity, container runs will fail later.

        asim_docker_vols = self.get_asim_docker_vols(
            settings, working_dir=workspace.full_path
        )

        asim_docker_vols.update({
            shared_tmp_dir: {"bind": "/tmp", "mode": "rw"},
            shared_cache_dir: {"bind": "/app/numba_cache", "mode": "rw"},
        })

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
            os.path.join(workspace.full_path, settings["asim_local_output_folder"])
        )

        os.makedirs(os.path.join(asim_local_output_folder, "cache", "numba"), exist_ok=True)

        compiled_asim_this_year = False

        # Record ActivitySim run start (Compilation if needed)
        if not self.state.asim_compiled:

            compiled_asim_this_year = True

            asim_cmd = self.get_base_asim_cmd(
                settings, household_sample_size=2500, num_processes=1
            )

            additional_args = self.get_asim_additional_args(
                settings, asim_docker_vols, True
            )

            asim_compile_run_hash = self.provenance_tracker.start_model_run(
                model=activity_demand_model,
                year=self.state.current_year,
                iteration=-1,
                description="asim compilation",
                inputs=filtered_store,
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
                }
            )

            output_records = []

            zarr_skims_rec = self.provenance_tracker.record_output_file(
                "activitysim",
                all_skims_path,
                model_run_id=asim_compile_run_hash,
                description="Zarr skims initialized from omx.",
                short_name="zarr_skims",
            )
            if zarr_skims_rec:
                output_records.append(zarr_skims_rec)
                logger.info(
                    f"Using zarr skims from ASIM compilation: {zarr_skims_rec.file_path}"
                )

            # Create initial zarr version snapshot after compilation
            if success and os.path.exists(all_skims_path):
                try:
                    logger.info("Creating initial zarr version snapshot after ActivitySim compilation...")

                    # Get database path from settings
                    database_path = settings.get("database", {}).get("path")
                    if database_path:
                        # Initialize zarr version manager
                        zarr_base_path = os.path.dirname(database_path)
                        zarr_manager = VersionedZarrStore(zarr_base_path)

                        # Create initialization snapshot (iteration -1)
                        snapshot_id = zarr_manager.create_snapshot_from_initialization(
                            run_id=self.provenance_tracker.run_info.run_id,
                            year=self.state.current_year,
                            source_zarr_path=all_skims_path,
                            provenance_tracker=self.provenance_tracker,
                        )
                        logger.info(f"Created initial zarr snapshot: {snapshot_id}")
                    else:
                        logger.debug("Database path not configured, skipping zarr snapshot creation")
                except Exception as e:
                    logger.warning(f"Failed to create initial zarr snapshot: {e}. Continuing without snapshot.")

            # Prepare runtime metadata for compilation run
            compile_metadata = {
                "container_command": asim_cmd,
                "runtime_parameters": {
                    "household_sample_size": 2500,
                    "num_processes": 1,
                    "chunk_size": settings.get("chunk_size", 0),
                    "additional_args": additional_args,
                },
                "container_image": activity_demand_image,
                "container_manager": get_setting(settings, "infrastructure.container_manager", "docker"),
                "working_directory": asim_workdir,
            }

            self.provenance_tracker.complete_model_run(
                asim_compile_run_hash,
                status="completed" if success else "failed",
                output_records=output_records,
                metadata=compile_metadata,
            )

            logger.info("ASIM Compilation success: {0}".format(success))
            if not success:
                raise RuntimeError("ASim Compilation failed")
            self.state.compile_asim()  # Update state to mark as compiled
        else:
            if "beam" in self.provenance_tracker.run_info.models_used:
                last_beam_post_records = self.provenance_tracker.run_info.get_latest_model_run_output_records(
                    "beam_postprocessor"
                )
                zarr_skims_rec = next(
                    r for r in last_beam_post_records if r.short_name == "zarr_skims"
                )
                logger.info(
                    f"Using zarr skims from last BEAM postprocessor run: {zarr_skims_rec.file_path}"
                )
            else:
                last_asim_run_hash = (
                    self.provenance_tracker.run_info.get_latest_model_run("activitysim")
                )
                last_asim_run_input_hashes = (
                    self.provenance_tracker.run_info.model_runs[
                        last_asim_run_hash
                    ].input_record_hashes
                )
                last_asim_run_input_records = [
                    self.provenance_tracker.run_info.file_records.get(h)
                    for h in last_asim_run_input_hashes
                ]
                zarr_skims_rec = next(
                    r
                    for r in last_asim_run_input_records
                    if r.short_name == "zarr_skims"
                )
                logger.info(
                    f"Using zarr skims that were inputs to the previous ASIM run: {zarr_skims_rec.file_path}"
                )

        if zarr_skims_rec:
            filtered_store.remove_record_type("omx_skims")
            filtered_store.add_record(zarr_skims_rec)
        else:
            logger.warning(
                "No ASIM skims cache found at: {0}. OMX skims will be used.".format(
                    all_skims_path
                )
            )

        new_asim_run_hash = self.provenance_tracker.start_model_run(
            model=activity_demand_model,
            year=self.state.current_year,
            iteration=self.state.current_inner_iter,
            description="asim full run",
            inputs=filtered_store,
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
            }
        )

        run_info = self.provenance_tracker.run_info.model_runs.get(new_asim_run_hash)
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
                    # Record as output file in provenance and collect FileRecord
                    output_rec = self.provenance_tracker.record_output_file(
                        "activitysim",
                        fpath,
                        year=self.state.forecast_year,
                        description=f"ActivitySim output file: {fname}",
                        short_name=fname + "_asim_out",
                        model_run_id=new_asim_run_hash,
                        state=self.state,
                    )
                    if output_rec:
                        output_records.append(output_rec)

        # Prepare runtime metadata for main run
        runtime_metadata = {
            "container_command": asim_cmd,
            "runtime_parameters": {
                "household_sample_size": settings.get("household_sample_size", 0),
                "num_processes": settings.get("num_processes", multiprocessing.cpu_count() - 1),
                "chunk_size": settings.get("chunk_size", 0),
                "additional_args": additional_args,
            },
            "container_image": activity_demand_image,
            "container_manager": get_setting(settings, "infrastructure.container_manager", "docker"),
            "working_directory": asim_workdir,
        }

        # Record ActivitySim run completion (Main run)
        self.provenance_tracker.complete_model_run(
            new_asim_run_hash,
            status="completed" if success else "failed",
            output_records=output_records,
            metadata=runtime_metadata,
        )

        output_store = RecordStore(recordList=output_records)

        # Get the ModelRunInfo for this run
        if run_info is None:
            # Fallback: create a minimal ModelRunInfo
            run_info = ModelRunInfo(
                model="activitysim",
                year=self.state.forecast_year,
                iteration=self.state.current_inner_iter,
                description="ActivitySim run",
                status="completed",
            )

        return output_store, run_info
