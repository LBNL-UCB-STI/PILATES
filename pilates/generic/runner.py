import abc
import logging
import os
import subprocess

from pilates.config import PilatesConfig
from pilates.generic.model import Model

try:
    import docker
except ImportError:
    print("Warning: Unable to import Docker Module")
from abc import ABC
from typing import Optional, Tuple

from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.utils.provenance import FileProvenanceTracker
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.container_utils import to_singularity_volumes, to_singularity_env
from pilates.utils.settings_helper import get as get_setting

logger = logging.getLogger(__name__)


class GenericRunner(ABC, Model):
    """
    A generic runner class that can be extended for specific models.
    """

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: FileProvenanceTracker,
        major_stage: Optional["WorkflowState.Stage"] = None,  # new
    ):
        super().__init__(model_name, state, provenance_tracker, major_stage)  # new
        self.required_input_files = []
        self.required_output_files = []

    def setup_container_cache_dirs(self, settings: PilatesConfig):
        """
        Set up Apptainer/Singularity cache directories.

        Uses local node storage (/local) for cache when available for faster
        extraction, while keeping outputs on scratch for large file I/O.

        Args:
            settings: PilatesConfig object containing run configuration
        """
        # Try to use fast local storage for cache (in order of preference)
        cache_base = None
        local_options = [
            "/local",  # Prioritize /local (853GB available)
            os.environ.get("TMPDIR", ""),
            "/tmp",  # Last resort - only 7.4GB
        ]

        for option in local_options:
            if option and os.path.exists(option) and os.access(option, os.W_OK):
                # Check if there's enough space (require at least 20GB free)
                try:
                    stat = os.statvfs(option)
                    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                    if free_gb >= 20:
                        cache_base = option
                        logger.info(
                            f"[{self.model_name}] Using local storage for cache: {cache_base} ({free_gb:.1f}GB free)"
                        )
                        break
                    else:
                        logger.debug(
                            f"[{self.model_name}] Skipping {option} - only {free_gb:.1f}GB free"
                        )
                except Exception as e:
                    logger.debug(
                        f"[{self.model_name}] Could not check space on {option}: {e}"
                    )
                    continue

        if not cache_base:
            # Fall back to scratch if no local storage available
            output_base = os.path.expandvars(settings.run.output_directory)
            cache_base = output_base
            logger.warning(
                f"[{self.model_name}] No suitable local storage found, using scratch for cache (may be slow)"
            )

        # Define cache directory paths
        apptainer_cache = os.path.join(cache_base, ".apptainer", "cache")
        apptainer_tmp = os.path.join(cache_base, ".apptainer", "tmp")
        singularity_cache = os.path.join(cache_base, ".singularity", "cache")
        singularity_tmp = os.path.join(cache_base, ".singularity", "tmp")

        # Set environment variables for Singularity/Apptainer
        os.environ["APPTAINER_CACHEDIR"] = apptainer_cache
        os.environ["APPTAINER_TMPDIR"] = apptainer_tmp
        os.environ["SINGULARITY_CACHEDIR"] = singularity_cache
        os.environ["SINGULARITY_TMPDIR"] = singularity_tmp

        # Create directories
        for cache_dir in [
            apptainer_cache,
            apptainer_tmp,
            singularity_cache,
            singularity_tmp,
        ]:
            os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"[{self.model_name}] Set Apptainer cache to: {apptainer_cache}")
        logger.info(f"[{self.model_name}] Set Apptainer tmp to: {apptainer_tmp}")

    def check_required_input_files(self, inputStore: RecordStore) -> bool:
        # TODO: Implement a check for required input files. Requires Record.simple_name
        return True

    @staticmethod
    def get_model_and_image(settings: PilatesConfig, model_type: str):
        manager = settings.infrastructure.container_manager
        if manager == "docker":
            image_names = settings.infrastructure.docker_images
        elif manager == "singularity":
            image_names = settings.infrastructure.singularity_images
        else:
            raise ValueError(
                "Container Manager not specified (container_manager param in settings.yaml)"
            )

        # Map legacy model_type keys to their new paths under run.models
        model_name_map = {
            "land_use_model": settings.run.models.land_use,
            "travel_model": settings.run.models.travel,
            "activity_demand_model": settings.run.models.activity_demand,
            "vehicle_ownership_model": settings.run.models.vehicle_ownership,
        }

        model_name = model_name_map.get(model_type)

        # Fallback for custom or non-standard model types passed in tests
        if model_name is None:
            model_name = getattr(settings.run.models, model_type, None)

        if not model_name:
            optional_models = ["vehicle_ownership_model"]
            if model_type in optional_models:
                return None, None
            else:
                raise ValueError(f"No model {model_type} specified in settings.")

        image_name = image_names.get(model_name)
        if not image_name:
            raise ValueError(
                f"No {manager} image specified for model '{model_name}' (model type: {model_type}). Check settings for '{manager}_images'."
            )

        return model_name, image_name

    def run(
        self,
        store: RecordStore,
        workspace: Workspace,
    ) -> Tuple[RecordStore, ModelRunInfo]:
        """
        Do the model run

        Args:
            store (RecordStore): The input data generated by the preprocessor.
            workspace (Workspace): The workspace.

        Returns:
            tuple: A tuple containing:
                - data (RecordStore): The raw output files that have been prepared to run the model
                - model_run_info (ModelRunInfo): Information about the model run
        """
        if (
            self.state.current_major_stage == self.major_stage
            and self.state.sub_stage_progress == "postprocessor"
        ):
            logger.info("Skipping runner, loading outputs from provenance.")
            runner_run = self.provenance_tracker.get_latest_completed_model_run(
                self.model_name, self.state.current_year, self.state.current_inner_iter
            )
            if runner_run:
                raw_outputs = RecordStore.from_file_records(
                    runner_run.output_record_hashes,
                    self.provenance_tracker.run_info.file_records,
                )
                run_info = runner_run
                return raw_outputs, run_info
            else:
                logger.warning(
                    "Could not find completed runner run in provenance, re-running runner."
                )
                self.state.set_sub_stage_progress("runner")
                return self._run(store, workspace)
        else:
            self.state.set_sub_stage_progress("runner")
            return self._run(store, workspace)

    @abc.abstractmethod
    def _run(
        self,
        store: RecordStore,
        workspace: Workspace,
    ) -> Tuple[RecordStore, ModelRunInfo]:
        """
        Do the model run.

        Note: Subclasses should apply @provenance_logging decorator (from
        pilates.generic.model) if they want automatic provenance tracking.
        Some runners (like ActivitySim) handle provenance manually because
        they may run multiple containers.

        Args:
            store (RecordStore): The input data generated by the preprocessor.
            workspace (Workspace): The workspace.

        Returns:
            tuple: A tuple containing:
                - data (RecordStore): The raw output files that have been prepared to run the model
                - model_run_info (ModelRunInfo): Information about the model run
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def run_container(
        client,
        settings: PilatesConfig,
        image: str,
        volumes: dict,
        command: str,
        model_name: str,
        working_dir=None,
        environment=None,
        args=None,
    ) -> bool:
        """
        Executes container using docker or singularity

        Args:
            client: the docker client. If it's provided then docker is used, otherwise singularity is used
            settings: settings to get docker configuration
            image: the image to run
            volumes: a dictionary describing volume binding
            command: the command to run
            model_name: name of the model, used for stubs
            working_dir: the working directory inside the container
            environment: a dictionary that contains environment variables
            args: additional arguments to the command

        Returns:
            bool: True if the container/stub ran successfully (exit code 0), False otherwise.
        """
        if client:  # Docker client is available
            docker_stdout = get_setting(
                settings, "infrastructure.docker_config.stdout", False
            )
            logger.info("Running docker container: %s, command: %s", image, command)
            run_kwargs = {
                "volumes": volumes,
                "command": command,
                "stdout": docker_stdout,
                "stderr": True,  # Always capture stderr
                "detach": True,  # Run in detached mode to stream logs
            }
            if working_dir:
                run_kwargs["working_dir"] = working_dir
            if environment:
                run_kwargs["environment"] = environment
            if args:
                # Append args to command string for docker
                full_command = command + " " + " ".join(args)
                run_kwargs["command"] = full_command
                logger.info("Full docker command: %s", full_command)

            container = None
            try:
                container = client.containers.run(image, **run_kwargs)
                # Stream logs
                for line in container.logs(
                    stream=True, stderr=True, stdout=docker_stdout
                ):
                    # Decode bytes and print
                    try:
                        print(line.decode("utf-8").strip())
                    except UnicodeDecodeError:
                        print(line.strip())  # Print raw bytes if decoding fails

                # Wait for container to finish and get exit code
                result = container.wait()
                exit_code = result["StatusCode"]
                logger.info(
                    f"Docker container {image} finished with exit code: {exit_code}"
                )
                return exit_code == 0

            except Exception as e:
                logger.error(f"Unexpected error running docker container {image}: {e}")
                return False
            finally:
                if container:
                    try:
                        container.remove()
                        logger.debug(f"Removed container {container.id}")
                    except Exception as e:
                        logger.warning(
                            f"Could not remove container {container.id}: {e}"
                        )

        else:  # Singularity
            import os

            for local_folder in volumes:
                # Ensure local directories exist for singularity binds
                os.makedirs(local_folder, exist_ok=True)

            singularity_volumes = to_singularity_volumes(volumes)
            # Construct the singularity command
            proc = (
                ["singularity", "run", "--cleanenv", "--writable-tmpfs"]
                + (["--env", to_singularity_env(environment)] if environment else [])
                + (["--pwd", working_dir] if working_dir else [])
                + ["-B", singularity_volumes, image]
                + (args if args else [])
                + command.split()
            )  # Split command string into list

            logger.info("Running singularity command: %s", " ".join(proc))

            # Check if using stubs
            if (
                settings.run.use_stubs
            ):  # Pass the full command string as config_name to the stub
                stub_cmd = [
                    "python",
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "tests",
                        "stubs",
                        "run_stub.py",
                    ),  # Use relative path for stub
                    "--model_name",
                    model_name,
                    "--cwd",
                    os.getcwd(),  # Pass current working directory of run.py
                    "--config_name",
                    " ".join(command),  # Pass the original command string
                ]
                logger.info(
                    f"Using stub for {model_name} ({image}). Running stub command: {' '.join(stub_cmd)}"
                )
                try:
                    result = subprocess.run(
                        stub_cmd, check=True, capture_output=True, text=True
                    )
                    print("Stub stdout:\n", result.stdout)
                    print("Stub stderr:\n", result.stderr)
                    logger.info(f"Stub for {model_name} finished successfully.")
                except subprocess.CalledProcessError as e:
                    logger.error(
                        f"Stub for {model_name} failed with exit code {e.returncode}."
                    )
                    print("Stub stdout:\n", e.stdout)
                    print("Stub stderr:\n", e.stderr)
                except FileNotFoundError:
                    logger.error(f"Stub script not found at {stub_cmd[2]}.")
                except Exception as e:
                    logger.error(f"Unexpected error running stub for {model_name}: {e}")
                finally:
                    return True

            else:  # Run actual singularity container
                try:
                    # Use subprocess.run to execute singularity command
                    result = subprocess.run(
                        proc, check=False
                    )  # Don't raise exception on non-zero exit code
                    logger.info(
                        f"Singularity container {image} finished with exit code: {result.returncode}"
                    )
                    return result.returncode == 0
                except FileNotFoundError:
                    logger.error(
                        "Singularity command not found. Is Singularity installed and in your PATH?"
                    )
                    return False
                except Exception as e:
                    logger.error(
                        f"Unexpected error running singularity container {image}: {e}"
                    )
                    return False

    @staticmethod
    def initialize_docker_client(settings):
        land_use_model = get_setting(settings, "run.models.land_use", False)
        vehicle_ownership_model = get_setting(
            settings, "run.models.vehicle_ownership", False
        )  ## ATLAS
        activity_demand_model = get_setting(
            settings, "run.models.activity_demand", False
        )
        travel_model = get_setting(settings, "run.models.travel", False)
        models = [
            land_use_model,
            vehicle_ownership_model,
            activity_demand_model,
            travel_model,
        ]
        image_names = get_setting(settings, "infrastructure.docker_images")
        pull_latest = get_setting(
            settings, "infrastructure.docker_config.pull_latest", False
        )

        client = docker.from_env()
        if pull_latest:
            logger.info("Pulling from docker...")
            for model in models:
                if model:
                    image = image_names.get(model)  # Use .get for safety
                    if image is not None:
                        print("Pulling latest image for {0}".format(image))
                        try:
                            client.images.pull(image)
                        except Exception as e:
                            logger.error(f"Error pulling docker image {image}: {e}")

        return client
