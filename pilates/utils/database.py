"""
Abstract database manager for PILATES data storage and retrieval.

This module provides the base interface for database operations to support
the centralized data store functionality, enabling storage and querying of
model inputs/outputs with complete provenance tracking.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pilates.generic.records import PilatesRunInfo

logger = logging.getLogger(__name__)


class DatabaseManager(ABC):
    """
    Abstract base class for database operations in PILATES.

    Provides interface for storing and retrieving PILATES run data,
    including configuration snapshots, file records, model runs,
    and OpenLineage event metadata.
    """

    def __init__(self, database_path: str, **kwargs):
        """
        Initialize the database manager.

        Args:
            database_path: Path to the database file or connection string
            **kwargs: Additional database-specific configuration options
        """
        self.database_path = database_path
        self.config = kwargs

    @abstractmethod
    def initialize_database(self) -> bool:
        """
        Initialize the database, creating tables if they don't exist.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def upload_run_data(self, run_info: PilatesRunInfo) -> bool:
        """
        Upload complete run data to the database.

        Args:
            run_info: Complete PILATES run information from run_info.json

        Returns:
            bool: True if upload successful, False otherwise
        """
        pass

    @abstractmethod
    def get_run_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve run information by run ID.

        Args:
            run_id: PILATES run ID

        Returns:
            Dict containing run information, or None if not found
        """
        pass

    @abstractmethod
    def get_runs_by_config_hash(self, config_hash: str) -> List[Dict[str, Any]]:
        """
        Find runs with matching configuration hash.

        Args:
            config_hash: Configuration content hash

        Returns:
            List of runs with matching configuration
        """
        pass

    @abstractmethod
    def get_dataset_by_openlineage_id(
        self, openlineage_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve dataset information by OpenLineage ID.

        Args:
            openlineage_id: OpenLineage dataset ID

        Returns:
            Dict containing dataset information, or None if not found
        """
        pass

    @abstractmethod
    def check_dataset_exists(self, openlineage_id: str) -> bool:
        """
        Check if a dataset exists by OpenLineage ID.

        Args:
            openlineage_id: OpenLineage dataset ID

        Returns:
            bool: True if dataset exists, False otherwise
        """
        pass

    @abstractmethod
    def get_model_runs_by_config(
        self, model_name: str, config_hash: str
    ) -> List[Dict[str, Any]]:
        """
        Find model runs with matching model and configuration.

        Args:
            model_name: Name of the model (e.g., 'beam', 'activitysim')
            config_hash: Model-specific configuration hash

        Returns:
            List of matching model runs
        """
        pass

    @abstractmethod
    def close(self):
        """Close database connection and clean up resources."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class DatabaseUploadError(Exception):
    """Exception raised when database upload operations fail."""

    pass


class DatabaseQueryError(Exception):
    """Exception raised when database query operations fail."""

    pass
