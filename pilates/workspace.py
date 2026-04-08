import os
import logging
from typing import Dict

from pilates.config import PilatesConfig
from pilates.generic.records import RecordStore
from pilates.utils.settings_helper import get as get_setting

logger = logging.getLogger(__name__)


class Workspace:
    """
    Manages the file system workspace for a Pilates run.
    This includes creating directories, copying initial data, and providing
    canonical paths to model data directories.
    """

    def __init__(
        self,
        settings: PilatesConfig,
        output_path: str,
        folder_name: str,
    ):
        self.settings = settings
        self.output_path = output_path
        self.folder_name = folder_name
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

    # Path getter methods
    def get_usim_mutable_data_dir(self) -> str:
        return os.path.join(
            self.full_path,
            get_setting(self.settings, "urbansim.local_mutable_data_folder"),
        )

    def get_asim_mutable_data_dir(self) -> str:
        return os.path.join(
            self.full_path,
            get_setting(self.settings, "activitysim.local_mutable_data_folder"),
        )

    def get_asim_mutable_configs_dir(self) -> str:
        return os.path.join(
            self.full_path,
            get_setting(self.settings, "activitysim.local_mutable_configs_folder"),
        )

    def get_asim_output_dir(self) -> str:
        return os.path.join(
            self.full_path,
            get_setting(self.settings, "activitysim.local_output_folder"),
        )

    def get_beam_mutable_data_dir(self) -> str:
        return os.path.join(
            self.full_path, get_setting(self.settings, "beam.local_mutable_data_folder")
        )

    def get_beam_output_dir(self) -> str:
        return os.path.join(
            self.full_path, get_setting(self.settings, "beam.local_output_folder")
        )

    def get_atlas_mutable_input_dir(self) -> str:
        return os.path.join(
            self.full_path,
            get_setting(self.settings, "atlas.host_mutable_input_folder"),
        )

    def get_atlas_output_dir(self) -> str:
        return os.path.join(
            self.full_path, get_setting(self.settings, "atlas.host_output_folder")
        )
