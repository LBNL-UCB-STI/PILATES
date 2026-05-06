import os
import logging
from typing import Dict, Optional

from pilates.config import PilatesConfig
from pilates.generic.records import RecordStore

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
        self._asim_mutable_data_dir_override: Optional[str] = None
        self._asim_runtime_cache_dir_override: Optional[str] = None
        self._beam_mutable_data_dir_override: Optional[str] = None
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
        urbansim_settings = self.settings.urbansim
        if urbansim_settings is None:
            raise RuntimeError(
                "UrbanSim config is required for UrbanSim workspace paths."
            )
        return os.path.join(
            self.full_path,
            urbansim_settings.local_mutable_data_folder,
        )

    def get_asim_mutable_data_dir(self) -> str:
        if self._asim_mutable_data_dir_override:
            return self._asim_mutable_data_dir_override
        activitysim_settings = self.settings.activitysim
        if activitysim_settings is None:
            raise RuntimeError(
                "ActivitySim config is required for ActivitySim workspace paths."
            )
        return os.path.join(
            self.full_path,
            activitysim_settings.local_mutable_data_folder,
        )

    def get_asim_mutable_configs_dir(self) -> str:
        activitysim_settings = self.settings.activitysim
        if activitysim_settings is None:
            raise RuntimeError(
                "ActivitySim config is required for ActivitySim workspace paths."
            )
        return os.path.join(
            self.full_path,
            activitysim_settings.local_mutable_configs_folder,
        )

    def get_asim_output_dir(self) -> str:
        activitysim_settings = self.settings.activitysim
        if activitysim_settings is None:
            raise RuntimeError(
                "ActivitySim config is required for ActivitySim workspace paths."
            )
        return os.path.join(
            self.full_path,
            activitysim_settings.local_output_folder,
        )

    def get_asim_runtime_cache_dir(self) -> str:
        if self._asim_runtime_cache_dir_override:
            return self._asim_runtime_cache_dir_override
        return os.path.join(self.get_asim_output_dir(), "cache")

    def get_beam_mutable_data_dir(self) -> str:
        if self._beam_mutable_data_dir_override:
            return self._beam_mutable_data_dir_override
        beam_settings = self.settings.beam
        if beam_settings is None:
            raise RuntimeError("BEAM config is required for BEAM workspace paths.")
        return os.path.join(
            self.full_path,
            beam_settings.local_mutable_data_folder,
        )

    def get_beam_output_dir(self) -> str:
        beam_settings = self.settings.beam
        if beam_settings is None:
            raise RuntimeError("BEAM config is required for BEAM workspace paths.")
        return os.path.join(
            self.full_path,
            beam_settings.local_output_folder,
        )

    def get_atlas_mutable_input_dir(self) -> str:
        atlas_settings = self.settings.atlas
        if atlas_settings is None:
            raise RuntimeError("ATLAS config is required for ATLAS workspace paths.")
        return os.path.join(
            self.full_path,
            atlas_settings.host_mutable_input_folder,
        )

    def get_atlas_output_dir(self) -> str:
        atlas_settings = self.settings.atlas
        if atlas_settings is None:
            raise RuntimeError("ATLAS config is required for ATLAS workspace paths.")
        return os.path.join(
            self.full_path,
            atlas_settings.host_output_folder,
        )

    def set_asim_mutable_data_dir_override(self, path: Optional[str]) -> None:
        self._asim_mutable_data_dir_override = os.path.abspath(path) if path else None

    def set_asim_runtime_cache_dir_override(self, path: Optional[str]) -> None:
        self._asim_runtime_cache_dir_override = os.path.abspath(path) if path else None

    def set_beam_mutable_data_dir_override(self, path: Optional[str]) -> None:
        self._beam_mutable_data_dir_override = os.path.abspath(path) if path else None
