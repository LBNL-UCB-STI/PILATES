import os
import logging
from typing import Dict

from pilates.activitysim import preprocessor as asim_pre
from pilates.generic.records import RecordStore
from pilates.urbansim import preprocessor as usim_pre
from pilates.beam import preprocessor as beam_pre
from pilates.atlas import preprocessor as atlas_pre
from pilates.utils.beam import get_beam_source_dir
from pilates.utils.provenance import FileProvenanceTracker

logger = logging.getLogger(__name__)


class Workspace:
    """
    Manages the file system workspace for a Pilates run.
    This includes creating directories, copying initial data, and providing
    canonical paths to model data directories.
    """

    def __init__(
        self,
        settings: dict,
        output_path: str,
        folder_name: str,
        provenance_tracker: "FileProvenanceTracker",
    ):
        self.settings = settings
        self.output_path = output_path
        self.folder_name = folder_name
        self.provenance_tracker = provenance_tracker
        self.input_data: Dict[str, RecordStore] = {}
        self.output_data: Dict[str, RecordStore] = {}
        self._setup_directories()

    @property
    def full_path(self) -> str:
        """Returns the full path to the root of the run-specific workspace."""
        if self.output_path is None:
            return os.getcwd()
        elif self.folder_name:
            return os.path.join(self.output_path, self.folder_name)
        else:
            return self.output_path

    def _setup_directories(self):
        """Creates output directory structure."""
        if not self.output_path:
            logger.info("No output_directory specified. Running in-place.")
            return

        os.makedirs(self.full_path, exist_ok=True)
        logger.info(f"Workspace initialized at: {self.full_path}")

    def _copy_initial_data(self):
        """
        Orchestrates the copying of all necessary initial data from source locations
        to the mutable workspace directory.
        """
        settings = self.settings
        base_folder_path = self.full_path
        have_not_copied_usim_data = True

        # BEAM
        if settings.get("travel_model") == "beam":
            input_dir = self.get_beam_mutable_data_dir()
            os.makedirs(input_dir, exist_ok=True)
            beam_preprocessor = beam_pre.BeamPreprocessor()
            input_store, output_store = beam_preprocessor.copy_data_to_mutable_location(
                settings, input_dir, self.provenance_tracker
            )
            self.input_data["beam"] = input_store
            self.output_data["beam"] = output_store
            os.makedirs(self.get_beam_output_dir(), exist_ok=True)
            # self._record_initial_repo_files("beam", get_beam_source_dir(settings))

        # Other models
        for model_key in [
            "activity_demand_model",
            "vehicle_ownership_model",
            "land_use_model",
        ]:
            model_name = settings.get(model_key)
            if not model_name:
                continue

            # UrbanSim data copy (once)
            if model_name == "urbansim" or (
                model_name == "activitysim" and have_not_copied_usim_data
            ):
                output_dir = self.get_usim_mutable_data_dir()
                os.makedirs(output_dir, exist_ok=True)
                usim_preprocessor = usim_pre.UrbansimPreprocessor()
                input_store, output_store = usim_preprocessor.copy_data_to_mutable_location(
                    settings, output_dir, self.provenance_tracker
                )
                if model_name in self.input_data:
                    self.input_data[model_name] += input_store
                else:
                    self.input_data[model_name] = input_store
                if model_name in self.output_data:
                    self.output_data[model_name] += output_store
                else:
                    self.output_data[model_name] = output_store
                have_not_copied_usim_data = False

            # Atlas data copy
            if model_name == "atlas":
                input_dir = self.get_atlas_mutable_input_dir()
                os.makedirs(input_dir, exist_ok=True)
                atlas_pre.copy_data_to_mutable_location(settings, input_dir)
                os.makedirs(self.get_atlas_output_dir(), exist_ok=True)
                self._record_initial_files(
                    "atlas", settings.get("atlas_host_input_folder"), "Atlas input file"
                )

            # ActivitySim config copy
            if model_name == "activitysim":
                asim_preprocessor = asim_pre.ActivitysimPreprocessor()
                input_store, output_store = asim_preprocessor.copy_data_to_mutable_location(
                    settings, base_folder_path, self.provenance_tracker
                )
                os.makedirs(self.get_asim_output_dir(), exist_ok=True)
                if model_name in self.input_data:
                    self.input_data[model_name] += input_store
                else:
                    self.input_data[model_name] = input_store
                if model_name in self.output_data:
                    self.output_data[model_name] += output_store
                else:
                    self.output_data[model_name] = output_store
                # asim_config_dir = os.path.join(
                #     settings.get("asim_local_configs_folder"), settings.get("region")
                # )
                # self._record_initial_repo_files(
                #     "activitysim",
                #     asim_config_dir,
                #     "ActivitySim configuration repository",
                # )

    def _record_initial_repo_files(self, model_name, repo_path, description=""):
        if not repo_path or not os.path.exists(repo_path):
            logger.warning(
                f"Initial data path for {model_name} does not exist: {repo_path}"
            )
            return

        if self.provenance_tracker.is_git_repo(repo_path):
            git_hash = self.provenance_tracker.get_git_hash(repo_path)
            self.provenance_tracker.record_repo_input(
                model_name, repo_path, description, git_hash
            )
        else:
            self._record_initial_files(model_name, repo_path, description)

    def _record_initial_files(self, model_name, source_dir, description):
        if not source_dir or not os.path.exists(source_dir):
            logger.warning(
                f"Initial data path for {model_name} does not exist: {source_dir}"
            )
            return
        for root, _, files in os.walk(source_dir):
            for file in files:
                input_path = os.path.join(root, file)
                if os.path.isfile(input_path):
                    self.provenance_tracker.record_input_file(
                        model_name, input_path, description=description
                    )

    # Path getter methods
    def get_usim_mutable_data_dir(self) -> str:
        return os.path.join(
            self.full_path, self.settings["usim_local_mutable_data_folder"]
        )

    def get_asim_mutable_data_dir(self) -> str:
        return os.path.join(
            self.full_path, self.settings["asim_local_mutable_data_folder"]
        )

    def get_asim_output_dir(self) -> str:
        return os.path.join(self.full_path, self.settings["asim_local_output_folder"])

    def get_beam_mutable_data_dir(self) -> str:
        return os.path.join(
            self.full_path, self.settings["beam_local_mutable_data_folder"]
        )

    def get_beam_output_dir(self) -> str:
        return os.path.join(self.full_path, self.settings["beam_local_output_folder"])

    def get_atlas_mutable_input_dir(self) -> str:
        return os.path.join(
            self.full_path, self.settings["atlas_host_mutable_input_folder"]
        )

    def get_atlas_output_dir(self) -> str:
        return os.path.join(self.full_path, self.settings["atlas_host_output_folder"])
