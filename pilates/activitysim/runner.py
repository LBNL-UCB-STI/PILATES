import logging
import multiprocessing
import os
from typing import Dict, Optional, List, Tuple

from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.generic.runner import GenericRunner
from workflow_state import WorkflowState

logger = logging.getLogger(__name__)


class ActivitysimRunner(GenericRunner):
    """
    Runner for ActivitySim model.
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)

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
        region = settings["region"]
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
            asim_local_output_folder = os.path.abspath(settings["asim_local_output_folder"])
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
            asim_local_configs_folder: {"bind": asim_remote_configs_folder, "mode": "rw"},
        }
        return asim_docker_vols

    def run(self, store: RecordStore, state: WorkflowState) -> Tuple[RecordStore, ModelRunInfo]:
        """
        Do the model run

        Args:
            store (RecordStore): The input data generated by the preprocessor.
            state (WorkflowState): The workflow state.

        Returns:
            tuple: A tuple containing:
                - data (RecordStore): The raw output files that have been prepared to run the model
                - model_run_info (ModelRunInfo): Information about the model run
        """

        region = state.settings["region"]
        asim_subdir = state.settings["region_to_asim_subdir"][region]
        asim_workdir = os.path.join("activitysim", asim_subdir)

        # start docker client
        client = None  # Initialize client to None
        if state.settings.get("container_manager") == "docker":
            try:
                client = self.initialize_docker_client(state.settings)
            except Exception as e:
                logger.error(f"Failed to initialize Docker client: {e}")
                # Decide how to handle failure - maybe exit?
                # For now, log and continue, assuming Singularity might be used or stubs.
                # If no client and no singularity, container runs will fail later.

        asim_docker_vols = self.get_asim_docker_vols(state.settings)
        activity_demand_model, activity_demand_image = self.get_model_and_image(
            state.settings, "activity_demand_model"
        )

        # Record ActivitySim run start (Compilation if needed)
        if not state.asim_compiled:

            asim_cmd = self.get_base_asim_cmd(
                state.settings, household_sample_size=2500, num_processes=1
            )

            additional_args = self.get_asim_additional_args(state.settings, asim_docker_vols, True)

            asim_compile_run_hash = state.record_model_start(
                message="asim compilation"
            )

            success = self.run_container(
                client=client,
                settings=state.settings,
                image=activity_demand_image,
                volumes=asim_docker_vols,
                command=asim_cmd,
                model_name="activitysim",
                working_dir=asim_workdir,
                args=additional_args,
            )

            state.record_model_completion(
                asim_compile_run_hash, status="completed" if success else "failed"
            )

            logger.info("ASIM Compilation success: {0}".format(success))
            if not success:
                raise RuntimeError("ASim Compilation failed")
            state.compile_asim()  # Update state to mark as compiled

            new_asim_run_hash = state.record_model_init(
                activity_demand_model,
                year=state.forecast_year,
                iteration=state.current_inner_iter,
                message="asim full run"
            )
            asim_main_run_hash = state.record_model_start(
                run_inputs_to_duplicate=new_asim_run_hash
            )
        else:
            asim_main_run_hash = state.record_model_start()
            asim_cmd = self.get_base_asim_cmd(
                state.settings, num_processes=1
            )

        asim_cmd = self.get_base_asim_cmd(state.settings)

        additional_args = self.get_asim_additional_args(state.settings, asim_docker_vols, False)

        success = self.run_container(
            client=client,
            settings=state.settings,
            image=activity_demand_image,
            volumes=asim_docker_vols,
            command=asim_cmd,
            model_name="activitysim",
            working_dir=asim_workdir,
            args=additional_args,
        )

        # Record ActivitySim run completion (Main run)
        state.record_model_completion(
            asim_main_run_hash, status="completed" if success else "failed"
        )

        # Assemble outputs: find the expected output files and return as a RecordStore
        # For ActivitySim, typical outputs are in the asim_local_output_folder
        output_dir = os.path.join(state.full_path, state.settings["asim_local_output_folder"])
        output_files = []
        if os.path.exists(output_dir):
            for fname in os.listdir(output_dir):
                fpath = os.path.join(output_dir, fname)
                if os.path.isfile(fpath) and (fname.endswith(".csv") or fname.endswith(".parquet")):
                    # Record as output file in provenance and collect OutputRecord
                    state.record_output_file("activitysim", fpath, year=state.forecast_year, description="ActivitySim output file", model_run_id=asim_main_run_hash)
                    from pilates.generic.records import OutputRecord
                    output_files.append(OutputRecord(file_path=fpath, output_type="ActivitySim output", model_run_id=asim_main_run_hash, year=state.forecast_year))
        output_store = RecordStore(recordList=output_files)

        # Get the ModelRunInfo for this run
        run_info = None
        if hasattr(state, "provenance_tracker") and state.provenance_tracker:
            run_info = state.provenance_tracker.run_info.model_runs.get(asim_main_run_hash)
        if run_info is None:
            # Fallback: create a minimal ModelRunInfo
            from pilates.generic.records import ModelRunInfo
            run_info = ModelRunInfo(model="activitysim", year=state.forecast_year, iteration=state.current_inner_iter, description="ActivitySim run", status="completed" if success else "failed")

        return output_store, run_info
