"""
This module provides utilities for provenance tracking in the Pilates framework. It includes classes and functions
to manage run contexts, track provenance information, and interact with file-based and repository-based data.

Classes:
- PilatesRunInfo: Represents metadata and provenance information for a single Pilates run.
- RunContext: Manages the context for a single model run, including its unique ID and methods for recording provenance.
- ProvenanceTracker: Base class for tracking provenance information.
- FileProvenanceTracker: Extends ProvenanceTracker with file I/O, hashing, and OS/git logic.

Functions:
- find_project_root: Searches upwards from a given path for a directory containing specific marker directories/files.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

import pandas as pd
try:
    from openlineage.client import set_producer, OpenLineageClient
    from openlineage.client.facet import (
        DocumentationJobFacet,
        SourceCodeLocationJobFacet,
    )
    from openlineage.client.run import RunEvent, Run, Job, RunState
    from openlineage.client.transport.http import HttpTransport, HttpConfig
    from openlineage.client.transport.file import FileTransport, FileConfig
    from openlineage.client.transport.composite import (
        CompositeTransport,
        CompositeConfig,
    )

    _OPENLINEAGE_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - depends on optional extra
    set_producer = None  # type: ignore[assignment]
    OpenLineageClient = object  # type: ignore[assignment]
    DocumentationJobFacet = object  # type: ignore[assignment]
    SourceCodeLocationJobFacet = object  # type: ignore[assignment]
    RunEvent = object  # type: ignore[assignment]
    Run = object  # type: ignore[assignment]
    Job = object  # type: ignore[assignment]
    RunState = object  # type: ignore[assignment]
    HttpTransport = object  # type: ignore[assignment]
    HttpConfig = object  # type: ignore[assignment]
    FileTransport = object  # type: ignore[assignment]
    FileConfig = object  # type: ignore[assignment]
    CompositeTransport = object  # type: ignore[assignment]
    CompositeConfig = object  # type: ignore[assignment]
    _OPENLINEAGE_AVAILABLE = False

from pilates.config import PilatesConfig
from pilates.utils.schema_inference import get_schema_from_file

if _OPENLINEAGE_AVAILABLE and set_producer is not None:
    set_producer("https://github.com/LBNL-UCB-STI/PILATES")


def _require_openlineage() -> None:
    if not _OPENLINEAGE_AVAILABLE:
        raise RuntimeError(
            "OpenLineage is required for OpenLineageTracker. "
            "Install the optional dependency (e.g., `pip install openlineage-python`)."
        )

from pilates.generic.records import (
    FileRecord,
    RepoRecord,
    ModelRunInfo,
    RecordStore,
    Record,
    PilatesRunInfo,
    OpenLineageEventMetadata,
    H5TableRecord,
    H5FileRecord,
)

from pilates.generic.execution_context import ExecutionContext
from pilates.utils.config_snapshot import ConfigSnapshotManager

logger = logging.getLogger(__name__)


def find_project_root(start_path=None, markers=("pilates", ".git")):
    """
    Search upwards from start_path for a directory containing one of the marker directories/files.
    Returns the absolute path to the project root, or None if not found.
    """
    if start_path is None:
        start_path = os.getcwd()
    current = os.path.abspath(start_path)
    while True:
        for marker in markers:
            if os.path.exists(os.path.join(current, marker)):
                return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


class ProvenanceTracker:
    """
    Pure data model for provenance tracking (no file I/O, OS, or git logic).
    """

    def __init__(self, run_id: str, output_path: str = None, folder_name: str = None):
        """
        Initializes the ProvenanceTracker.

        Args:
            run_id (str): A unique identifier for the current run.
            output_path (str, optional): The base directory for storing run-related outputs.
            folder_name (str, optional): An optional subfolder name within `output_path`
                                         to organize run data.
        """
        self.run_id = run_id
        self.output_path = output_path
        self.folder_name = folder_name
        self.run_info = PilatesRunInfo(
            run_id=run_id, created_at=datetime.now().isoformat()
        )
        self.current_model_run_id = None

    def _save_run_info(self):
        """Hook for persisting the in-memory run information.

        The base implementation does not perform any I/O. Subclasses that
        maintain a file-backed run_info should override this method to write
        `self.run_info` to persistent storage. Calling the base method will log
        a warning to make missing persistence explicit.
        """
        # Base class does nothing (or could log, or raise NotImplementedError)
        logger.warning("No save_run_info implemented in base ProvenanceTracker class.")
        pass

    def current_model_run(self) -> Optional[ModelRunInfo]:
        """
        Returns the current model run info if available, otherwise None
        """
        if (
            self.current_model_run_id
            and self.current_model_run_id in self.run_info.model_runs
        ):
            return self.run_info.model_runs[self.current_model_run_id]
        else:
            return None

    def _normalize_model_name(self, model: str) -> str:
        """Normalize a model name to a canonical form.

        Currently this normalizes to lowercase when a string is present. The
        helper centralizes normalization so it can be extended (e.g. remove
        whitespace or prefixes) in one place.
        """
        return model.lower() if model else model

    def initialize_from_settings(self, settings: PilatesConfig):
        """
        Initializes the tracker with settings from a PilatesConfig object.

        This populates the `run_info` with global run parameters like start/end year
        and lists the models used based on the provided settings.

        Args:
            settings (PilatesConfig): The Pilates configuration object.
        """
        self.run_info.start_year = settings.run.start_year
        self.run_info.end_year = settings.run.end_year
        self.run_info.settings_hash = None  # FileProvenanceTracker will set this
        models_used = []
        if settings.run.models.land_use:
            models_used.append(settings.run.models.land_use)
        if settings.run.models.vehicle_ownership:
            models_used.append(settings.run.models.vehicle_ownership)
        if settings.run.models.activity_demand:
            models_used.append(settings.run.models.activity_demand)
        if settings.run.models.travel:
            models_used.append(settings.run.models.travel)
        self.run_info.models_used = list(set(models_used))

    def record_repo_input(
        self,
        model: str,
        repo_path: str,
        description: str = None,
        git_hash: str = None,
    ):
        """
        Records a repository as an input to a specific model run.

        This method adds a `RepoRecord` to the `run_info.repo_records` for the given model.

        Args:
            model (str): The name of the model consuming this repository.
            repo_path (str): The file system path to the repository.
            description (str, optional): A human-readable description of the repository.
            git_hash (str, optional): The Git commit hash of the repository. If None,
                                      the `FileProvenanceTracker` subclass will attempt to
                                      derive it or use a path hash.
        """
        model = self._normalize_model_name(model)
        if model not in self.run_info.repo_records:
            self.run_info.repo_records[model] = []
        repo_record = RepoRecord(
            repo_path=repo_path,
            description=description,
            unique_id=git_hash,
        )
        self.run_info.repo_records[model].append(repo_record)
        current_model_run = self.current_model_run()
        if current_model_run:
            current_model_run.input_record_hashes.append(repo_record.unique_id)

    def get_run_info(self) -> Dict[str, Any]:
        """Return a copy of the current in-memory run information.

        The returned object is a shallow copy of the underlying data model
        (PilatesRunInfo). Subclasses may provide serialized representations
        via their own `get_run_info` implementations.
        """
        return self.run_info.copy()

    def get_model_summary(self) -> Dict[str, Any]:
        """Produce a high-level summary of models, inputs, outputs, and status.

        Returns a dictionary with aggregate counts and categorization useful
        for quick inspection or reporting. The structure includes counts of
        input/output files per model and counts of runs by model and status.
        """
        summary = {
            "total_model_runs": len(self.run_info.model_runs),
            "models_used": self.run_info.model_runs,
            "input_files_by_model": {},
            "output_files_by_model": {},
            "run_status": {},
        }
        input_counts = {}
        output_counts = {}
        for fr in self.run_info.file_records.values():
            for model in fr.models:
                if fr.producing_run_id:
                    output_counts[model] = output_counts.get(model, 0) + 1
                if fr.consuming_run_ids:
                    input_counts[model] = input_counts.get(model, 0) + 1

        summary["input_files_by_model"] = input_counts
        summary["output_files_by_model"] = output_counts

        for run_id in self.run_info.model_runs:
            run = self.run_info.model_runs[run_id]
            model = run.model
            status = run.status
            if model not in summary["run_status"]:
                summary["run_status"][model] = {}
            summary["run_status"][model][status] = (
                summary["run_status"][model].get(status, 0) + 1
            )
        return summary

    def start_model_run(
        self,
        model: str,
        year: int = None,
        iteration: int = None,
        description: str = None,
        inputs: RecordStore = RecordStore(),
        **kwargs,
    ) -> str:
        """Start tracking a new model run.

        Creates a ModelRunInfo record, registers it in `self.run_info.model_runs`,
        and returns the generated run identifier.

        Args:
            model: The model name.
            year: Optional year associated with the run.
            iteration: Optional iteration index for supply-demand loops.
            description: Human-friendly description of this run.
            inputs: A RecordStore containing input records to attach to the run.

        Returns:
            The unique run id string assigned to this model run.
        """
        for record in inputs.all_records():
            if record.unique_id not in self.run_info.file_records:
                self.run_info.file_records[record.unique_id] = record

        model_run_id = (
            f"{model}_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        )
        run_record = ModelRunInfo(
            unique_id=model_run_id,
            model=model,
            year=year,
            iteration=iteration,
            description=description,
            created_at=datetime.now().isoformat(),
            status="running",
            input_record_hashes=inputs.all_unique_ids(),
        )
        self.current_model_run_id = model_run_id
        self.run_info.model_runs[model_run_id] = run_record
        self._save_run_info()
        return model_run_id

    def complete_model_run(
        self,
        run_hash: str,
        status: str = "completed",
        output_records: List[Union[FileRecord, RepoRecord]] = None,
        metadata: dict = None,
    ):
        """Mark a model run as complete and attach its outputs.

        Updates timestamps and status for the model run identified by
        `run_hash`. Any provided output records will be added to the
        model run's outputs if not already present.

        Args:
            run_hash: The unique id of the model run to complete.
            status: Final status (e.g., 'completed' or 'failed').
            output_records: Optional list of FileRecord/RepoRecord objects
                produced by the run.
            metadata: Optional dict with runtime execution metadata (container command, parameters, etc.).
        """
        if output_records is None:
            output_records = []
        if run_hash in self.run_info.model_runs:
            self.run_info.model_runs[run_hash].completed_at = datetime.now().isoformat()
            self.run_info.model_runs[run_hash].status = status
            if metadata:
                self.run_info.model_runs[run_hash].metadata.update(metadata)
            for dataset in output_records:
                if isinstance(dataset, (FileRecord, H5FileRecord)):
                    if dataset.unique_id not in self.run_info.file_records:
                        self.run_info.file_records[dataset.unique_id] = dataset
                    # Ensure the record is linked to this run as a producer
                    dataset.producing_run_id = run_hash
                    if dataset.models is None:
                        dataset.models = []
                    if self.run_info.model_runs[run_hash].model not in dataset.models:
                        dataset.models.append(self.run_info.model_runs[run_hash].model)
                elif isinstance(dataset, RepoRecord):
                    if dataset.unique_id not in self.run_info.repo_records:
                        self.run_info.repo_records[dataset.unique_id] = dataset

                if isinstance(dataset, Record):
                    if (
                        dataset.unique_id
                        not in self.run_info.model_runs[run_hash].output_record_hashes
                    ):
                        self.run_info.model_runs[run_hash].output_record_hashes.append(
                            dataset.unique_id
                        )
            self._save_run_info()
        else:
            logger.error(f"Model run hash {run_hash} not found to complete.")


class FileProvenanceTracker(ProvenanceTracker):
    """
    File-backed ProvenanceTracker: adds file I/O, hashing, and OS/git logic.
    """

    def __init__(self, run_id: str, output_path: str, folder_name: str = None):
        """
        Initializes the FileProvenanceTracker.

        Args:
            run_id (str): A unique identifier for the current run.
            output_path (str): The base directory for storing run-related outputs.
                               This path must be absolute or will be converted to absolute.
            folder_name (str, optional): An optional subfolder name within `output_path`
                                         to organize run data.
        """
        super().__init__(run_id, output_path, folder_name)
        self.output_path = os.path.abspath(output_path) if output_path else None
        self.folder_name = folder_name
        self.workspace_root = (
            os.path.join(self.output_path, self.folder_name)
            if self.folder_name and self.output_path
            else self.output_path
        )
        self.run_info_path = self._get_run_info_path()
        self.run_info = self._initialize_run_info()
        logger.info(f"FileProvenanceTracker initialized for run ID: {self.run_id}")

    def get_latest_completed_model_run(
        self, model_name: str, year: int, iteration: int
    ) -> Optional[ModelRunInfo]:
        """
        Find the latest completed model run for a given model name and year.

        Args:
            model_name (str): The name of the model.
            year (int): The year of the model run.
            iteration (int): The iteration of the model run.

        Returns:
            Optional[ModelRunInfo]: The latest completed model run, or None if not found.
        """
        latest_run = None
        latest_time = None

        for run in self.run_info.model_runs.values():
            if (
                run.model == model_name
                and run.year == year
                and run.iteration == iteration
                and run.status == "completed"
            ):
                completed_at = datetime.fromisoformat(run.completed_at)
                if latest_time is None or completed_at > latest_time:
                    latest_time = completed_at
                    latest_run = run

        return latest_run

    def _get_run_info_path(self) -> str:
        """Return the filesystem path where run_info.json should be stored.

        If a `folder_name` was provided during initialization the run_info
        file is placed under that subdirectory; otherwise it is placed
        directly in `self.output_path`.
        """
        if self.folder_name:
            return os.path.join(self.output_path, self.folder_name, "run_info.json")
        else:
            return os.path.join(self.output_path, "run_info.json")

    def _load_metadata(self, file_path: str) -> Dict[str, Any]:
        """Load an adjacent .metadata.json file for a given data file if present.

        Many datasets in this project ship with a sidecar `<filename>.metadata.json`
        containing human-readable metadata; this helper attempts to load that
        JSON file and returns an empty dict if it is missing or invalid.
        """
        metadata_file = os.path.join(
            os.path.dirname(file_path), f"{os.path.basename(file_path)}.metadata.json"
        )
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load metadata from {metadata_file}: {e}")
        return {}

    def is_git_repo(self, path: str) -> bool:
        """Check whether a given path is the root of a Git repository.

        This is a lightweight check that simply verifies the existence of a
        `.git` directory at the given path. It does not attempt to run git
        commands or validate repository health.
        """
        abs_path = os.path.abspath(path)
        git_dir = os.path.join(abs_path, ".git")
        is_repo = os.path.exists(git_dir)
        logger.debug(f"Checking if path {abs_path} is a git repo: {is_repo}")
        return is_repo

    def get_git_hash(self, repo_path: str = None) -> Optional[str]:
        """Attempt to retrieve the current Git commit hash (HEAD) for a repo.

        Runs `git rev-parse HEAD` in `repo_path` (or the package directory when
        not supplied). Returns the hash string or None if git is unavailable or
        the command fails.
        """
        try:
            abs_repo_path = (
                os.path.abspath(repo_path) if repo_path else os.path.dirname(__file__)
            )
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=abs_repo_path,
                timeout=5,
            )
            logger.debug(
                f"Git hash for repo at {abs_repo_path}: {result.stdout.strip()}"
            )
            return result.stdout.strip()
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ) as e:
            logger.warning(f"Could not determine git hash for repo at {repo_path}: {e}")
            return None

    def _validate_file_path(self, file_path: str) -> Optional[str]:
        """Verify that a file path exists and return its absolute form.

        Returns the absolute path when the file exists. If the provided path is
        falsy or the file does not exist, logs a warning and returns None.
        """
        if not file_path:
            logger.warning("Empty file path provided for validation")
            return None
        abs_path = os.path.abspath(file_path)
        logger.debug(f"Validating file path: {abs_path}")
        if not os.path.exists(abs_path):
            logger.warning(f"File does not exist: {abs_path}")
            return None
        return abs_path

    def _get_validated_paths(
        self, file_path: str, skip_missing: bool
    ) -> Tuple[Optional[str], Optional[str]]:
        """Validate `file_path` and return a tuple (absolute_or_original, relative).

        If the file does not exist and `skip_missing` is True this returns
        (None, None). Otherwise it returns the absolute path when available or
        the original path, together with a relative path computed against the
        tracker's output base.
        """
        abs_path = self._validate_file_path(file_path)
        if not abs_path and skip_missing:
            logger.debug(f"Skipping missing file: {file_path}")
            return None, None
        path_to_use = abs_path or file_path
        relative_path = self.get_path_relative_to_workspace_root(path_to_use)
        return path_to_use, relative_path

    def _calculate_directory_hash(
        self, abs_dir_path: str, state: Optional[ExecutionContext] = None
    ) -> Optional[str]:
        """
        Calculates a SHA-256 hash of a directory based on its file and subdirectory names,
        and the size and modification time of each file. It optionally incorporates additional
        state information.

        Args:
            abs_dir_path (str): The path to the directory to hash.
            state (Optional[ExecutionContext]): Optional state metadata to include in the hash.

        Returns:
            Optional[str]: The calculated hash, or None if the path is invalid.
        """

        sha256_hash = hashlib.sha256()
        # Use os.walk to traverse the directory tree
        for root, dirs, files in os.walk(abs_dir_path):
            # Sort directory and file names to ensure a consistent hash
            dirs.sort()
            files.sort()

            for dir_name in dirs:
                # Update hash with subdirectory names
                sha256_hash.update(dir_name.encode())

            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    # Get file size and modification time
                    file_size = os.path.getsize(file_path)
                    mod_time = os.path.getmtime(file_path)

                    # Update hash with file metadata
                    sha256_hash.update(file_name.encode())
                    sha256_hash.update(str(file_size).encode())
                    sha256_hash.update(str(mod_time).encode())
                except OSError:
                    # Ignore files that can't be accessed
                    continue

        # Optionally include the additional state information
        if state:
            sha256_hash.update(str(state.current_major_stage).encode())
            sha256_hash.update(str(state.current_year).encode())
            sha256_hash.update(str(state.current_inner_iter).encode())

        return sha256_hash.hexdigest()

    def _calculate_path_hash(
        self, file_path: str, state: Optional[ExecutionContext] = None
    ) -> Optional[str]:
        """
        Calculates the SHA-256 hash of a file, depending on its contents and its location. It also optionally
        incorporates additional state information, such as the year and iteration for which it was created.

        Args:
            file_path (str): The path to the file whose hash is to be calculated.
            state (Optional[ExecutionContext]): An optional state object containing additional metadata
                (e.g., current stage, year, and iteration) to include in the hash.

        Returns:
            Optional[str]: The calculated SHA-256 hash as a hexadecimal string, or None if the file
            path is invalid or an error occurs during hashing.

        Raises:
            Warning: Logs a warning if the file cannot be read or hashed due to an IOError or OSError.
        """
        # Validate the file path and ensure it exists
        abs_file_path = self._validate_file_path(file_path)
        if not abs_file_path:
            return None
        if os.path.isfile(abs_file_path):
            return self._calculate_file_hash(abs_file_path, state)
        elif os.path.isdir(abs_file_path):
            return self._calculate_directory_hash(abs_file_path, state)
        else:
            logger.error(f"Could not calculate hash for {abs_file_path}")
            return None

    def _calculate_file_hash(
        self, abs_file_path: str, state: Optional[ExecutionContext] = None
    ) -> Optional[str]:
        """
        Calculates the SHA-256 hash of a file, depending on its contents and its location. It also optionally
        incorporates additional state information, such as the year and iteration for which it was created.

        Args:
            abs_file_path (str): The path to the file whose hash is to be calculated.
            state (Optional[ExecutionContext]): An optional state object containing additional metadata
                (e.g., current stage, year, and iteration) to include in the hash.

        Returns:
            Optional[str]: The calculated SHA-256 hash as a hexadecimal string, or None if the file
            path is invalid or an error occurs during hashing.

        Raises:
            Warning: Logs a warning if the file cannot be read or hashed due to an IOError or OSError.
        """
        try:
            sha256_hash = hashlib.sha256()
            sha256_hash.update(abs_file_path.encode())
            with open(abs_file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            if state:
                sha256_hash.update(str(state.current_major_stage).encode())
                sha256_hash.update(str(state.current_year).encode())
                sha256_hash.update(str(state.current_inner_iter).encode())
            return sha256_hash.hexdigest()
        except (IOError, OSError) as e:
            logger.warning(f"Could not calculate hash for {abs_file_path}: {e}")
            return None

    def _calculate_settings_hash(self, settings: PilatesConfig) -> str:
        """Return a stable SHA-256 hash of the settings dict.

        The settings are serialized with sorted keys so semantically equivalent
        dicts map to the same hash regardless of insertion order.
        """
        settings_str = settings.model_dump_json()
        return hashlib.sha256(settings_str.encode("utf-8")).hexdigest()

    def get_path_relative_to_workspace_root(self, file_path: str) -> str:
        """Compute a path for storing in run_info relative to the workspace root."""
        abs_path = os.path.abspath(file_path)
        if self.workspace_root:
            try:
                # Return path relative to workspace root
                return os.path.relpath(abs_path, self.workspace_root)
            except ValueError:
                # Fallback to absolute path if on a different drive (e.g., Windows)
                logger.warning(
                    f"Could not create relative path for {abs_path} relative to {self.workspace_root}. Storing absolute path."
                )
                return abs_path
        else:
            # Fallback to absolute path if workspace root isn't set
            logger.warning("Workspace root not set. Storing absolute path.")
            return abs_path

    def initialize_from_settings(self, settings: PilatesConfig):
        super().initialize_from_settings(settings)
        self.run_info.settings_hash = self._calculate_settings_hash(settings)
        self._save_run_info()
        logger.info("FileProvenanceTracker initialized with settings.")

    def record_repo_input(
        self,
        model: str,
        repo_path: str,
        short_name: str = None,
        description: str = None,
        git_hash: str = None,
    ) -> RepoRecord:
        """Record a repository (or directory) as an input to the run.

        The method calculates a unique id for the repository (using a path
        hash when `git_hash` is not provided), stores a RepoRecord in
        `self.run_info.repo_records`, and returns it.
        """
        if git_hash is None:
            git_hash = self._calculate_path_hash(repo_path)
        model = self._normalize_model_name(model)
        abs_path = self._validate_file_path(repo_path)
        if not abs_path:
            logger.warning(f"Skipping missing repository for {model}: {repo_path}")
            return
        relative_path = self.get_path_relative_to_workspace_root(abs_path)

        repo_record = RepoRecord(
            unique_id=git_hash,
            repo_path=relative_path,
            accessed_at=datetime.now().isoformat(),
            description=description,
            short_name=short_name,
        )
        self.run_info.repo_records[git_hash] = repo_record
        self._save_run_info()
        logger.debug(
            f"Recorded repository input for {model}: {relative_path} (exists: {abs_path is not None})"
        )
        return repo_record

    def record_repo_output(
        self,
        model: str,
        repo_path: str,
        short_name: str = None,
        description: str = None,
        git_hash: str = None,
    ) -> RepoRecord:
        """
        Record a repository (or directory) as an output of the current model run.

        This mirrors `record_repo_input` for API parity with the Consist adapter and
        to support dual-storage workflows that materialize derived repos/dirs.
        """
        if git_hash is None:
            git_hash = self._calculate_path_hash(repo_path)
        model = self._normalize_model_name(model)
        abs_path = self._validate_file_path(repo_path)
        if not abs_path:
            logger.warning(f"Skipping missing repository output for {model}: {repo_path}")
            return
        relative_path = self.get_path_relative_to_workspace_root(abs_path)

        repo_record = RepoRecord(
            unique_id=git_hash,
            repo_path=relative_path,
            accessed_at=datetime.now().isoformat(),
            description=description,
            short_name=short_name,
        )
        self.run_info.repo_records[git_hash] = repo_record

        current_model_run = self.current_model_run()
        if current_model_run:
            current_model_run.output_record_hashes.append(repo_record.unique_id)

        self._save_run_info()
        logger.debug(
            f"Recorded repository output for {model}: {relative_path} (exists: {abs_path is not None})"
        )
        return repo_record

    def _create_h5_table_records(
        self,
        h5_file_path: str,
        h5_file_record: "H5FileRecord",
    ) -> List["H5TableRecord"]:
        """
        Creates H5TableRecord objects for all tables found within a given HDF5 file.

        These records capture metadata and a unique hash for each individual table
        within the H5 container, linking them back to the parent `H5FileRecord`.

        Args:
            h5_file_path (str): The absolute path to the HDF5 file on disk.
            h5_file_record (H5FileRecord): The parent H5FileRecord for the container file.

        Returns:
            List[H5TableRecord]: A list of `H5TableRecord` objects, one for each table
                                 found in the HDF5 file.
        """
        table_records = []
        try:
            file_mtime = os.path.getmtime(h5_file_path)
            with pd.HDFStore(h5_file_path, mode="r") as store:
                for table_name in store.keys():
                    try:
                        df_sample = store.select(table_name, stop=1)
                        try:
                            storer = store.get_storer(table_name)
                            nrows = storer.nrows if storer else 0
                        except Exception:
                            nrows = 0

                        fingerprint = (
                            f"{table_name}:"
                            f"{nrows}:"
                            f"{len(df_sample.columns)}:"
                            f"{file_mtime}"
                        )
                        table_hash = hashlib.sha256(fingerprint.encode()).hexdigest()

                        schema = [
                            {"name": col, "type": str(dtype)}
                            for col, dtype in df_sample.dtypes.items()
                        ]

                        table_record = H5TableRecord(
                            unique_id=table_hash,
                            h5_file_unique_id=h5_file_record.unique_id,
                            table_name=table_name,
                            file_path=f"{h5_file_record.file_path}{table_name}",
                            created_at=datetime.now().isoformat(),
                            short_name=f"{h5_file_record.short_name}{table_name.replace('/', '_')}",
                            description=f"Table '{table_name}' from H5 file '{h5_file_record.short_name}'",
                            models=h5_file_record.models,
                            schema=schema,
                            metadata={
                                "table_name": table_name,
                                "source_h5_file": h5_file_record.file_path,
                                "nrows": nrows,
                                "ncols": len(df_sample.columns),
                                "file_modified": file_mtime,
                            },
                            source_file_paths=h5_file_record.source_file_paths,
                        )
                        table_records.append(table_record)
                    except Exception as e:
                        logger.warning(
                            f"Could not read table '{table_name}' from H5 file {h5_file_path}: {e}"
                        )
        except Exception as e:
            logger.error(f"Failed to create H5TableRecords for {h5_file_path}: {e}")
        return table_records

    def _get_or_create_file_record(
        self,
        file_path: str,
        skip_missing: bool = True,
        description: Optional[str] = None,
        short_name: Optional[str] = None,
        state: Optional[ExecutionContext] = None,
        source_file_paths: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Union["FileRecord", "H5FileRecord"]]:
        """
        Retrieves an existing `FileRecord` or `H5FileRecord` for a given file path,
        or creates a new one if it doesn't exist.

        This function handles both regular files and HDF5 container files,
        automatically generating `H5TableRecord`s for HDF5 files.

        Args:
            file_path (str): The path to the file (or directory for Zarr).
            skip_missing (bool, optional): If True, returns None if the file does not exist.
                                          Defaults to True.
            description (Optional[str], optional): A human-readable description for the file.
            short_name (Optional[str], optional): A short, unique name for the file.
                                                  If None, it's derived from the file path.
            state (Optional[ExecutionContext], optional): Execution context to include in the hash
                                                        (e.g., year, iteration).
            source_file_paths (Optional[List[str]], optional): List of paths to source files
                                                                 that produced this file.
            metadata (Optional[Dict[str, Any]], optional): Additional metadata to associate with the record.

        Returns:
            Optional[Union[FileRecord, H5FileRecord]]: The created or retrieved `FileRecord`
                                                     (or `H5FileRecord`), or None if the
                                                     file is missing and `skip_missing` is True.
        """
        path_to_use, relative_path = self._get_validated_paths(file_path, skip_missing)
        if not path_to_use:
            return None

        if path_to_use.endswith((".h5", ".hdf5")):
            file_hash = self._calculate_file_hash(path_to_use, state)
            if not file_hash:
                logger.warning(f"Could not calculate hash for H5 file {file_path}")
                return None

            if file_hash in self.run_info.file_records:
                existing_record = self.run_info.file_records[file_hash]
                if metadata:
                    existing_record.metadata.update(metadata)
                return existing_record

            combined_metadata = self._load_metadata(path_to_use)
            if metadata:
                combined_metadata.update(metadata)

            h5_file_record = H5FileRecord(
                unique_id=file_hash,
                file_path=relative_path,
                created_at=datetime.now().isoformat(),
                short_name=short_name
                or os.path.splitext(os.path.basename(file_path))[0],
                metadata=combined_metadata,
                description=description,
                year=state.current_year if state else None,
                models=[],
                source_file_paths=source_file_paths or [],
            )
            table_records = self._create_h5_table_records(
                file_path,
                h5_file_record,
            )
            for table_record in table_records:
                self.run_info.file_records[table_record.unique_id] = table_record
            h5_file_record.table_record_ids = [tr.unique_id for tr in table_records]
            self.run_info.file_records[file_hash] = h5_file_record
            return h5_file_record
        else:
            if os.path.isdir(path_to_use):
                file_hash = self._calculate_path_hash(path_to_use, state)
            else:
                file_hash = self._calculate_file_hash(path_to_use, state)
            if not file_hash:
                logger.warning(
                    f"Could not calculate hash for {file_path}, cannot create record."
                )
                return None

            if file_hash in self.run_info.file_records:
                existing_record = self.run_info.file_records[file_hash]
                if metadata:
                    existing_record.metadata.update(metadata)
                return existing_record

            combined_metadata = self._load_metadata(path_to_use)
            if metadata:
                combined_metadata.update(metadata)
            schema = get_schema_from_file(path_to_use)

            file_record = FileRecord(
                unique_id=file_hash,
                file_path=relative_path,
                created_at=datetime.now().isoformat(),
                short_name=short_name,
                metadata=combined_metadata,
                description=description,
                year=state.current_year if state else None,
                schema=schema,
                source_file_paths=source_file_paths or [],
            )
            self.run_info.file_records[file_hash] = file_record
            return file_record

    def record_h5_input_container(
        self, model: str, file_path: str, **kwargs
    ) -> Optional[H5FileRecord]:
        """Records an HDF5 file as an input container.

        This is a convenience wrapper around `record_input_file` that ensures the created
        record is an H5FileRecord, which includes records for all its internal tables.

        Args:
            model (str): The name of the model consuming this input.
            file_path (str): The path to the HDF5 file.
            **kwargs: Additional arguments passed to `record_input_file`.

        Returns:
            Optional[H5FileRecord]: The created or retrieved H5FileRecord, or None.
        """
        record = self.record_input_file(model, file_path, **kwargs)
        if record and not isinstance(record, H5FileRecord):
            logger.error(
                f"Recorded H5 container for {file_path}, but it was not an H5FileRecord."
            )
            return None
        return record

    def record_h5_table_input(
        self,
        model_name: str,
        h5_container_record: H5FileRecord,
        table_name: str,
        model_run_id: str,
        **kwargs,
    ) -> Optional[H5TableRecord]:
        """Records a specific table from an HDF5 container as a model input.

        Finds the table record within the container and links it as an input to the specified model run.

        Args:
            model_name (str): The name of the model consuming this table.
            h5_container_record (H5FileRecord): The record for the parent HDF5 file.
            table_name (str): The name/path of the table within the HDF5 file.
            model_run_id (str): The ID of the model run that is consuming this table.
            **kwargs: Additional arguments (not currently used).

        Returns:
            Optional[H5TableRecord]: The found and recorded H5TableRecord, or None.
        """
        if not h5_container_record:
            logger.error("No H5 container record provided to record table input.")
            return None
        for table_id in h5_container_record.table_record_ids:
            if table_id in self.run_info.file_records:
                table_record = self.run_info.file_records[table_id]
                if table_record.table_name == table_name:
                    self.record_input_record(table_record, model_run_id)
                    logger.info(
                        f"Recorded H5 table '{table_name}' as input to run {model_run_id}"
                    )
                    return table_record
        logger.warning(
            f"Could not find table '{table_name}' in container {h5_container_record.short_name} to record as input."
        )
        return None

    def record_h5_table_output(
        self,
        model_name: str,
        h5_container_record: H5FileRecord,
        table_name: str,
        input_records: List[Record],
        model_run_id: str,
        **kwargs,
    ) -> Optional[H5TableRecord]:
        """Records a newly created or updated HDF5 table as a model output.

        Creates a new H5TableRecord, links it to its input records, and registers it
        as an output of the specified model run.

        Args:
            model_name (str): The name of the model producing this table.
            h5_container_record (H5FileRecord): The record for the parent HDF5 file.
            table_name (str): The name/path of the table within the HDF5 file.
            input_records (List[Record]): A list of input records that produced this output table.
            model_run_id (str): The ID of the model run that produced this table.
            **kwargs: Additional keyword arguments to pass to the H5TableRecord constructor.

        Returns:
            Optional[H5TableRecord]: The newly created H5TableRecord.
        """
        # Create a new unique ID for the output table based on its inputs and timestamp
        input_hashes = "".join(sorted([rec.unique_id for rec in input_records if rec]))
        fingerprint = f"{h5_container_record.unique_id}:{table_name}:{input_hashes}:{datetime.now().isoformat()}"
        table_hash = hashlib.sha256(fingerprint.encode()).hexdigest()

        output_table_record = H5TableRecord(
            unique_id=table_hash,
            h5_file_unique_id=h5_container_record.unique_id,
            table_name=table_name,
            file_path=f"{h5_container_record.file_path}{table_name}",
            created_at=datetime.now().isoformat(),
            models=[model_name],
            source_file_paths=[rec.file_path for rec in input_records if rec],
            producing_run_id=model_run_id,
            **kwargs,
        )
        self.run_info.file_records[table_hash] = output_table_record
        if model_run_id in self.run_info.model_runs:
            self.run_info.model_runs[model_run_id].output_record_hashes.append(
                table_hash
            )
        logger.info(
            f"Recorded new H5 table output '{table_name}' with hash {table_hash}"
        )
        return output_table_record

    def record_h5_output_container(
        self,
        model: str,
        file_path: str,
        table_records: List[H5TableRecord] = None,
        **kwargs,
    ) -> Optional[H5FileRecord]:
        """Records an HDF5 file as an output container.

        This is a convenience wrapper around `record_output_file` that ensures the created
        record is an H5FileRecord, which includes records for all its internal tables.

        Args:
            model (str): The name of the model producing this output.
            file_path (str): The path to the HDF5 file.
            table_records (List[H5TableRecord], optional): A list of pre-created table records.
                If not provided, they will be generated from the file.
            **kwargs: Additional arguments passed to `record_output_file`.

        Returns:
            Optional[H5FileRecord]: The created or retrieved H5FileRecord, or None.
        """
        input_records = kwargs.pop("input_records", [])
        record = self.record_output_file_with_inputs(
            model, file_path, input_records, **kwargs
        )

        if record and not isinstance(record, H5FileRecord):
            logger.error(
                f"Recorded H5 container for {file_path}, but it was not an H5FileRecord."
            )
            return None
        if record and table_records:
            record.table_record_ids = [tr.unique_id for tr in table_records]
            for tr in table_records:
                if tr.unique_id not in self.run_info.file_records:
                    self.run_info.file_records[tr.unique_id] = tr
        return record

    def rename_directory(self, old_directory_name, new_directory_name):
        """
        Renames a directory in the run_info.file_records.
        This is useful when the directory structure changes but the file records need to be preserved.
        """
        for record in self.run_info.file_records.values():
            if record.file_path.startswith(old_directory_name):
                new_path = record.file_path.replace(
                    old_directory_name, new_directory_name, 1
                )
                record.file_path = new_path
                logger.info(
                    f"Renamed file path from {old_directory_name} to {new_directory_name} for record {record.unique_id}"
                )
        self._save_run_info()

    def move_file(
        self,
        record: Record,
        source_path: str,
        destination_path: str,
        model: str,
        state: Optional[ExecutionContext] = None,
    ) -> Optional[FileRecord]:
        """Move a file on disk and update provenance records accordingly.

        Copies the file from `source_path` to `destination_path`, marks the
        original record as no longer existing, and records the destination as an
        output of the current model run. Only file records are supported; for
        repo moves callers must handle git operations manually.
        """
        if isinstance(record, FileRecord):
            self.record_input_file(
                model=self._normalize_model_name(model),
                file_path=source_path,
                model_run_id=self.current_model_run_id,
                source_run_id=record.producing_run_id,
                state=state,
            )
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copyfile(source_path, destination_path)
            record.exists = False
            self.run_info.file_records[record.unique_id] = record
            output_record = self.record_output_file(
                model=self._normalize_model_name(model),
                file_path=destination_path,
                description=record.description,
                short_name=re.sub(r"_temp$", "", record.short_name),
                model_run_id=self.current_model_run_id,
                state=state,
                source_file_paths=[record.file_path],
            )
            return output_record
        else:
            raise NotImplementedError("You have to move git repos manually")

    def record_input_record(self, record: Record, model_run_id: str = None):
        """Attach an existing `Record` as an input to the given model run.

        If `model_run_id` is omitted the currently active model run is used.
        This simply appends the record's unique id to the model run's
        `input_record_hashes` list.
        """
        if model_run_id is None:
            model_run_id = self.current_model_run_id
        if model_run_id in self.run_info.model_runs:
            self.run_info.model_runs[model_run_id].input_record_hashes.append(
                record.unique_id
            )

    def record_input_from_previous_output(
        self,
        model: str,
        file_path: str,
        producing_model: str = None,
        description: str = None,
        model_run_id: str = None,
        short_name: str = None,
        state: Optional[ExecutionContext] = None,
    ) -> Optional[FileRecord]:
        """
        Automatically finds the previous output record and links it, making cross-model
        linkages explicit and enabling validation.

        Args:
            model: Name of the current model consuming this file
            file_path: Path to the input file
            producing_model: Optional name of model that produced this file (for validation)
            description: Human-readable description
            model_run_id: ID of the current model run
            short_name: Short identifier for this file
            state: Current workflow state

        Returns:
            FileRecord for the input, reusing existing record if found

        Example:
            >>> # ActivitySim reads H5 that ATLAS modified
            >>> asim_h5_record = tracker.record_input_from_previous_output(
            ...     "activitysim_preprocessor",
            ...     usim_h5_file,
            ...     producing_model="atlas_postprocessor",  # Optional validation
            ...     model_run_id=asim_pre_hash,
            ... )
            >>> # Automatically finds ATLAS's output record and links it!
        """
        model = self._normalize_model_name(model)

        # Try to find existing file record
        file_hash = self._calculate_file_hash(file_path)

        if file_hash in self.run_info.file_records:
            existing_record = self.run_info.file_records[file_hash]

            # Validate producing model if specified
            if producing_model:
                producing_model_normalized = self._normalize_model_name(producing_model)
                if producing_model_normalized not in existing_record.models:
                    logger.warning(
                        f"File {existing_record.short_name} was expected to be produced by "
                        f"{producing_model}, but it was produced by {existing_record.models}. "
                        f"This may indicate an unexpected data flow."
                    )

            # Add current model as consumer
            if model not in existing_record.models:
                existing_record.models.append(model)

            # Link to current model run
            if model_run_id and model_run_id in self.run_info.model_runs:
                run_info = self.run_info.model_runs[model_run_id]
                if file_hash not in run_info.input_record_hashes:
                    run_info.input_record_hashes.append(file_hash)

            self._save_run_info()

            logger.info(
                f"Linked existing file {existing_record.short_name} as input to {model}"
            )

            return existing_record
        else:
            # Fall back to normal input recording
            logger.warning(
                f"File {file_path} not found in previous outputs. "
                f"Recording as new input. Did you forget to track it as output from {producing_model}?"
            )
            return self.record_input_file(
                model=model,
                file_path=file_path,
                description=description,
                model_run_id=model_run_id,
                short_name=short_name,
                state=state,
            )

    def record_input_file(
        self,
        model: str,
        file_path: str,
        source_run_id: str = None,
        description: str = None,
        short_name: str = None,
        source_file_paths: List[str] = None,
        skip_missing: bool = True,
        model_run_id: str = None,
        state: Optional[ExecutionContext] = None,
        context: Optional[ExecutionContext] = None,
    ) -> Optional[FileRecord]:
        """
        Record an input file with optional execution context.

        Parameters
        ----------
        model : str
            Name of the model consuming this input
        file_path : str
            Path to the input file
        source_run_id : str, optional
            ID of the run that produced this file
        description : str, optional
            Human-readable description of the file
        short_name : str, optional
            Short identifier for the file
        source_file_paths : List[str], optional
            Paths to source files that produced this file
        skip_missing : bool, default=True
            If True, skip files that don't exist rather than raising an error
        model_run_id : str, optional
            ID of the model run consuming this input
        state : ExecutionContext, optional
            **DEPRECATED**: Use 'context' instead. Execution context providing
            year, stage, and iteration metadata.
        context : ExecutionContext, optional
            Execution context providing year, stage, and iteration metadata.
            Any object with current_year, current_major_stage, and current_inner_iter
            attributes works (including WorkflowState).

        Returns
        -------
        FileRecord or None
            Record of the input file, or None if file doesn't exist and skip_missing=True
        """
        # Support both 'state' and 'context' parameters during transition
        # context takes precedence if both provided
        ctx = context if context is not None else state
        model = self._normalize_model_name(model)
        file_record = self._get_or_create_file_record(
            file_path,
            skip_missing,
            description,
            short_name=short_name,
            state=ctx,
            source_file_paths=source_file_paths,
        )
        if not file_record:
            return None

        if model and model not in file_record.models:
            file_record.models.append(model)
        if description:
            file_record.description = description
        if source_file_paths:
            file_record.source_file_paths = [
                self.get_path_relative_to_workspace_root(p) for p in source_file_paths
            ]
        if source_run_id:
            file_record.producing_run_id = source_run_id

        run_id_to_consume = model_run_id or self.current_model_run_id
        if run_id_to_consume:
            if run_id_to_consume not in file_record.consuming_run_ids:
                file_record.consuming_run_ids.append(run_id_to_consume)

            if run_id_to_consume in self.run_info.model_runs:
                run_info = self.run_info.model_runs[run_id_to_consume]
                if file_record.unique_id not in run_info.input_record_hashes:
                    run_info.input_record_hashes.append(file_record.unique_id)

        self._save_run_info()
        return file_record

    def record_output_file(
        self,
        model: str,
        file_path: str,
        year: int = None,
        description: str = None,
        skip_missing: bool = True,
        model_run_id: str = None,
        short_name: str = None,
        source_file_paths: list = None,
        state: Optional[ExecutionContext] = None,
        context: Optional[ExecutionContext] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[FileRecord]:
        """
        Record an output file with optional execution context.

        Parameters
        ----------
        model : str
            Name of the model producing this output
        file_path : str
            Path to the output file
        year : int, optional
            Year associated with this output (overrides context.current_year if provided)
        description : str, optional
            Human-readable description of the file
        skip_missing : bool, default=True
            If True, skip files that don't exist rather than raising an error
        model_run_id : str, optional
            ID of the model run producing this output
        short_name : str, optional
            Short identifier for the file
        source_file_paths : list, optional
            Paths to source files that produced this output
        state : ExecutionContext, optional
            **DEPRECATED**: Use 'context' instead. Execution context providing
            year, stage, and iteration metadata.
        context : ExecutionContext, optional
            Execution context providing year, stage, and iteration metadata.
            Any object with current_year, current_major_stage, and current_inner_iter
            attributes works (including WorkflowState).
        metadata : dict, optional
            Arbitrary key/value metadata to merge into the record.

        Returns
        -------
        FileRecord or None
            Record of the output file, or None if file doesn't exist and skip_missing=True
        """
        # Support both 'state' and 'context' parameters during transition
        ctx = context if context is not None else state
        model = self._normalize_model_name(model)
        file_record = self._get_or_create_file_record(
            file_path,
            skip_missing,
            description,
            short_name,
            state=ctx,
            source_file_paths=source_file_paths,
            metadata=metadata,
        )
        if not file_record:
            return None

        if model and model not in file_record.models:
            file_record.models.append(model)
        if description:
            file_record.description = description
        if year:
            file_record.year = year

        if source_file_paths:
            file_record.source_file_paths = [
                self.get_path_relative_to_workspace_root(p) for p in source_file_paths
            ]

        run_id_producing = model_run_id or self.current_model_run_id
        if run_id_producing:
            file_record.producing_run_id = run_id_producing
            if run_id_producing in self.run_info.model_runs:
                run_info = self.run_info.model_runs[run_id_producing]
                if file_record.unique_id not in run_info.output_record_hashes:
                    run_info.output_record_hashes.append(file_record.unique_id)

        self._save_run_info()
        return file_record

    def record_output_file_with_inputs(
        self,
        model: str,
        file_path: str,
        input_records: List[Optional[FileRecord]],
        **kwargs,
    ) -> Optional[FileRecord]:
        """
        This helper reduces boilerplate by automatically extracting file paths from input FileRecords.
        It also accepts any other keyword arguments and passes them through to `record_output_file`.

        Args:
            model: Name of the model producing this output
            file_path: Path to the output file
            input_records: List of FileRecords that were inputs to this transformation.
                          None values are automatically filtered out.
            **kwargs: Additional keyword arguments to pass to `record_output_file`
                      (e.g., year, description, short_name, model_run_id, state, updated_children).

        Returns:
            FileRecord for the output, or None if file doesn't exist and skip_missing=True

        Example:
            >>> usim_output_record = tracker.record_output_file_with_inputs(
            ...     "atlas_postprocessor",
            ...     usim_h5_file,
            ...     input_records=[usim_input_record, atlas_hh_input_record],
            ...     year=2017,
            ...     description="UrbanSim H5 after ATLAS vehicle update",
            ...     short_name="usim_h5_updated",
            ...     model_run_id=model_run_hash,
            ...     state=state,
            ...     updated_children=['/2017/households']
            ... )
        """
        # Extract file paths from input records, filtering out None values
        source_file_paths = [rec.file_path for rec in input_records if rec is not None]

        # Add extracted source paths to kwargs, overriding if present
        kwargs["source_file_paths"] = source_file_paths

        return self.record_output_file(
            model=model,
            file_path=file_path,
            **kwargs,
        )

    def _save_run_info(self, data_to_save: PilatesRunInfo = None):
        """Persist the tracker's `run_info` dataclass to JSON on disk.

        If `data_to_save` is provided it will be saved instead of the current
        `self.run_info`. Errors while writing the file are logged as errors.
        """
        import dataclasses

        data = data_to_save if data_to_save is not None else self.run_info
        os.makedirs(os.path.dirname(self.run_info_path), exist_ok=True)
        try:
            with open(self.run_info_path, "w") as f:
                json.dump(dataclasses.asdict(data), f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Could not save run_info.json to {self.run_info_path}: {e}")

    def get_run_info(self) -> dict:
        """Return the persisted run_info as a dict when available.

        This attempts to read the on-disk run_info.json and falls back to the
        in-memory dataclass representation if the file is missing or invalid.
        """
        import dataclasses

        if os.path.exists(self.run_info_path):
            try:
                with open(self.run_info_path, "r") as f:
                    data = json.load(f)
                    return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not reload run_info.json for get_run_info: {e}")
                return dataclasses.asdict(self.run_info)
        return dataclasses.asdict(self.run_info)

    def validate_provenance_chain(self) -> Dict[str, List[str]]:
        """
        PHASE 1 IMPROVEMENT #3: Validate the completeness of provenance tracking.

        Checks for common provenance issues:
        - Outputs without source_file_paths (incomplete lineage)
        - Model runs without inputs or outputs
        - Orphaned file records (not referenced by any model run)
        - Broken source_file_paths references

        Returns:
            Dict with 'warnings' and 'errors' keys containing validation issues.
            Warnings indicate potential issues that may be acceptable.
            Errors indicate definite problems that should be fixed.

        Example:
            >>> issues = tracker.validate_provenance_chain()
            >>> if issues['errors']:
            ...     logger.error(f"Provenance errors: {issues['errors']}")
            >>> if issues['warnings']:
            ...     logger.warning(f"Provenance warnings: {issues['warnings']}")
        """
        issues = {"warnings": [], "errors": []}

        # Check 1: All outputs should have source_file_paths (except initial inputs)
        for file_hash, file_record in self.run_info.file_records.items():
            # Check if this file is an output of any model run
            is_output = any(
                file_hash in run.output_record_hashes
                for run in self.run_info.model_runs.values()
            )
            # Check if this file is ONLY an output (not also an input)
            is_input = any(
                file_hash in run.input_record_hashes
                for run in self.run_info.model_runs.values()
            )

            # If it's an output but never used as input, it should have sources
            # (unless it's a pure initial input that's also marked as output, which is rare)
            if is_output and not file_record.source_file_paths:
                # Don't warn about files that are ALSO inputs (likely initial data)
                if not is_input:
                    issues["warnings"].append(
                        f"Output file '{file_record.short_name}' ({file_record.file_path}) "
                        f"has no source_file_paths. Lineage may be incomplete."
                    )

        # Check 2: All model runs should have inputs and outputs
        for run_hash, model_run in self.run_info.model_runs.items():
            if not model_run.input_record_hashes:
                issues["warnings"].append(
                    f"Model run '{model_run.model}' (year {model_run.year}, iter {model_run.iteration}) "
                    f"has no input records. This may indicate incomplete tracking."
                )
            if not model_run.output_record_hashes:
                issues["warnings"].append(
                    f"Model run '{model_run.model}' (year {model_run.year}, iter {model_run.iteration}) "
                    f"has no output records. This may indicate incomplete tracking."
                )

        # Check 3: Orphaned file records (not referenced by any model run)
        all_referenced_hashes = set()
        for model_run in self.run_info.model_runs.values():
            all_referenced_hashes.update(model_run.input_record_hashes)
            all_referenced_hashes.update(model_run.output_record_hashes)

        orphaned = set(self.run_info.file_records.keys()) - all_referenced_hashes
        if orphaned:
            for file_hash in orphaned:
                record = self.run_info.file_records[file_hash]
                issues["warnings"].append(
                    f"File '{record.short_name}' ({record.file_path}) "
                    f"is not referenced by any model run. It may have been recorded but never used."
                )

        # Check 4: Broken source_file_paths references
        for file_hash, file_record in self.run_info.file_records.items():
            if file_record.source_file_paths:
                for source_path in file_record.source_file_paths:
                    # Check if source exists in file_records
                    # Need to check both absolute and relative paths
                    source_found = any(
                        rec.file_path == source_path
                        or self.get_path_relative_to_workspace_root(rec.file_path)
                        == source_path
                        for rec in self.run_info.file_records.values()
                    )
                    if not source_found:
                        issues["errors"].append(
                            f"File '{file_record.short_name}' references source '{source_path}' "
                            f"which is not in file_records. This indicates a broken provenance chain."
                        )

        # Check 5: Model runs with status='failed' should be flagged
        failed_runs = [
            f"{run.model} (year {run.year})"
            for run in self.run_info.model_runs.values()
            if run.status == "failed"
        ]
        if failed_runs:
            issues["warnings"].append(
                f"Found {len(failed_runs)} failed model runs: {', '.join(failed_runs)}"
            )

        return issues

    def _initialize_run_info(self) -> PilatesRunInfo:
        """Load an existing run_info.json or create an initial PilatesRunInfo.

        If a `run_info.json` exists at `self.run_info_path` it is loaded and
        converted into a `PilatesRunInfo` object. Otherwise a fresh
        `PilatesRunInfo` is created, written to disk, and returned.
        """
        if os.path.exists(self.run_info_path):
            try:
                with open(self.run_info_path, "r") as f:
                    data = json.load(f)

                    # Deserialize file_records from dicts to actual FileRecord/H5FileRecord/H5TableRecord objects
                    deserialized_file_records = {}
                    for uid, record_dict in data.get("file_records", {}).items():
                        if "h5_file_unique_id" in record_dict:
                            deserialized_file_records[uid] = H5TableRecord(
                                **record_dict
                            )
                        elif "table_record_ids" in record_dict:
                            deserialized_file_records[uid] = H5FileRecord(**record_dict)
                        else:
                            deserialized_file_records[uid] = FileRecord(**record_dict)

                    # Deserialize model_runs from dicts to ModelRunInfo objects
                    deserialized_model_runs = {}
                    for uid, run_dict in data.get("model_runs", {}).items():
                        deserialized_model_runs[uid] = ModelRunInfo(**run_dict)

                    run_info = PilatesRunInfo(
                        run_id=data.get("run_id"),
                        created_at=data.get("created_at"),
                        start_year=data.get("start_year"),
                        end_year=data.get("end_year"),
                        models_used=data.get("models_used", []),
                        settings_hash=data.get("settings_hash"),
                        code_version=data.get("code_version"),
                        hostname=data.get("hostname"),
                        file_records=deserialized_file_records,
                        repo_records=data.get("repo_records", {}),
                        model_runs=deserialized_model_runs,
                        config_snapshot=data.get("config_snapshot"),
                        openlineage_event_metadata=data.get(
                            "openlineage_event_metadata", []
                        ),
                    )
                    return run_info
            except Exception as e:  # Catch broader exception during deserialization
                logger.warning(
                    f"Could not load existing run_info.json due to deserialization error: {e}. Creating new one."
                )
        run_info = PilatesRunInfo(
            run_id=self.run_id,
            created_at=datetime.now().isoformat(),
            start_year=None,
            end_year=None,
            models_used=[],
            settings_hash=None,
            code_version=self.get_git_hash(),
            hostname=os.uname().nodename if hasattr(os, "uname") else "unknown",
            file_records={},
            repo_records={},
            model_runs={},
        )
        self._save_run_info(run_info)
        return run_info


class OpenLineageTracker(FileProvenanceTracker):
    """
    Extends FileProvenanceTracker to generate an OpenLineage event log
    using the official OpenLineage transport interface.
    """

    def __init__(
        self,
        run_id: str,
        output_path: str,
        folder_name: str = None,
        use_file: bool = True,
        use_marquez: bool = False,
        marquez_url: str = "http://localhost:5002",
        add_year_to_job_name: bool = True,
        add_iteration_to_job_name: bool = True,
    ):
        """
        Initializes the OpenLineageTracker instance.

        Args:
            run_id (str): Unique identifier for the run.
            output_path (str): Path to the output directory where logs will be stored.
            folder_name (str, optional): Name of the folder within the output path for storing logs.
            use_file (bool): Whether to write events to a local file (default True).
            use_marquez (bool): Whether to send events to Marquez (default False).
            marquez_url (str): URL of the Marquez server.
        """
        _require_openlineage()
        super().__init__(run_id, output_path, folder_name)
        self.namespace = "default"
        self.client = None
        self.add_year_to_job_name = add_year_to_job_name
        self.add_iteration_to_job_name = add_iteration_to_job_name

        # Initialize configuration snapshot manager
        workspace_path = (
            os.path.join(output_path, folder_name) if folder_name else output_path
        )
        self.config_manager = ConfigSnapshotManager(workspace_path)

        # This list will hold configuration dictionaries
        transports_config = []

        # 1. Build a list of transport configurations
        if use_file and self.output_path:
            log_path = os.path.join(
                self.output_path, self.folder_name or "", "openlineage.jsonl"
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            if os.path.exists(log_path):
                os.remove(log_path)

            transports_config.append(
                {"type": "file", "log_file_path": log_path, "append": True}
            )

        if use_marquez:
            transports_config.append({"type": "http", "url": marquez_url})

        # 2. Instantiate the transport(s) and the client
        if transports_config:
            final_transport = None
            if len(transports_config) > 1:
                # For multiple transports, use CompositeTransport with CompositeConfig
                composite_config = CompositeConfig(transports=transports_config)
                final_transport = CompositeTransport(config=composite_config)
            else:
                # For a single transport, instantiate it directly
                config = transports_config[0]
                if config["type"] == "file":
                    final_transport = FileTransport(
                        config=FileConfig(
                            log_file_path=config["log_file_path"],
                            append=config.get("append", True),
                        )
                    )
                elif config["type"] == "http":
                    final_transport = HttpTransport(
                        config=HttpConfig(url=config["url"])
                    )

            if final_transport:
                self.client = OpenLineageClient(transport=final_transport)

    def initialize_from_settings(self, settings: PilatesConfig):
        """Override to capture config snapshot during initialization."""
        # Call parent method to set basic fields
        super().initialize_from_settings(settings)

        # Create and store config snapshot
        config_snapshot = self.config_manager.create_config_snapshot(settings)
        self.run_info.config_snapshot = config_snapshot
        self._save_run_info()

        logger.info(
            f"Created config snapshot {config_snapshot['snapshot_id']} "
            f"with hash {config_snapshot['config_content_hash'][:8]}"
        )

    def _emit_event(self, event: RunEvent, model_run_id: str = None):
        """
        Emits an OpenLineage event using the configured client and transport.
        Also captures essential event metadata in run_info for database upload.
        """
        # Capture essential metadata from the event
        event_metadata = OpenLineageEventMetadata(
            event_time=event.eventTime,
            event_type=event.eventType.value,  # START, COMPLETE, FAIL
            run_uuid=event.run.runId,
            job_name=event.job.name,
            model_run_id=model_run_id or "unknown",
        )
        self.run_info.openlineage_event_metadata.append(event_metadata)
        self._save_run_info()

        if self.client:
            try:
                self.client.emit(event)
            except Exception as e:
                logger.error(f"Could not emit OpenLineage event: {e}")
        else:
            logger.warning("No OpenLineage transport configured; event was not sent.")

    def _get_job_facets(
        self, description: str, model_run_id: str = None, git_hash: str = None
    ):
        """
        Generates job facets for the OpenLineage event.

        Args:
            description (str): Description of the job.
            git_hash (str, optional): Git hash of the repository.

        Returns:
            dict: A dictionary containing job facets.
        """
        facets = {"documentation": DocumentationJobFacet(description=description)}
        if model_run_id:
            facets["customModelRunId"] = {
                "_producer": "pilates",
                "model_run_id": model_run_id,
            }
        if git_hash:
            repo_path = find_project_root() or os.getcwd()
            facets["sourceCodeLocation"] = SourceCodeLocationJobFacet(
                type="git", url=repo_path, repo=repo_path, tag=git_hash
            )
        return facets

    def _format_name(
        self, name: str, year: Optional[int], iteration: Optional[int]
    ) -> str:
        """Format a job name optionally appending year/iteration suffixes.

        This keeps OpenLineage job names stable while allowing disambiguation
        by year and iteration when the tracker is configured to do so.
        """
        if self.add_year_to_job_name and year:
            name += f"_{year}"
            if self.add_iteration_to_job_name and iteration:
                name += f"_{iteration}"
        return name

    def move_file(
        self,
        record: Record,
        source_path: str,
        destination_path: str,
        model: str,
        state: Optional[ExecutionContext] = None,
    ) -> Optional[FileRecord]:
        """
        Moves a file on disk and updates provenance records accordingly,
        extending the parent method to include year and iteration in the short name.

        Copies the file from `source_path` to `destination_path`, marks the
        original record as no longer existing, and records the destination as an
        output of the current model run. Only file records are supported; for
        repo moves callers must handle git operations manually.

        Args:
            record (Record): The provenance record of the file to move.
            source_path (str): The original path of the file.
            destination_path (str): The new path for the file.
            model (str): The name of the model performing the move.
            state (Optional[ExecutionContext], optional): Current execution context
                                                        containing year and iteration.

        Returns:
            Optional[FileRecord]: The updated `FileRecord` for the moved file, or None if the
                                  original record was not a `FileRecord`.
        """
        output_record = super().move_file(
            record, source_path, destination_path, model, state
        )

        if state:
            # If state is provided, update the output record with state information
            output_record.short_name = (
                output_record.short_name
                + f"_{state.current_year}_{state.current_inner_iter}"
            )
        return output_record

    def start_model_run(
        self,
        model: str,
        year: int = None,
        iteration: int = None,
        description: str = None,
        inputs: RecordStore = RecordStore(),
        **kwargs,
    ) -> str:
        """
        Starts tracking a new model run and emits an OpenLineage START event.

        This method extends the parent's `start_model_run` by also generating
        and emitting an OpenLineage event to signal the beginning of a job execution.

        Args:
            model (str): The name of the model.
            year (int, optional): Optional year associated with the run.
            iteration (int, optional): Optional iteration index for supply-demand loops.
            description (str, optional): Human-friendly description of this run.
            inputs (RecordStore, optional): A `RecordStore` containing input records to attach to the run.

        Returns:
            str: The unique run ID string assigned to this model run.
        """
        model_run_id = super().start_model_run(
            model, year, iteration, description, inputs
        )
        run_id_uuid = self.run_info.model_runs[model_run_id].openlineage_id

        input_datasets = []
        current_run_info = self.run_info.model_runs.get(model_run_id)
        if current_run_info:
            for record_hash in current_run_info.input_record_hashes:
                if record_hash in self.run_info.file_records:
                    file_record = self.run_info.file_records[record_hash]
                    dataset = file_record.toInputDataset(self.namespace)
                    input_datasets.append(dataset)
        input_names = [dataset.name for dataset in input_datasets]

        event = RunEvent(
            eventType=RunState.START,
            eventTime=datetime.now().isoformat(),
            run=Run(runId=run_id_uuid),
            job=Job(
                namespace=self.namespace,
                name=self._format_name(model, year, iteration),
                facets=self._get_job_facets(
                    description or f"Pilates model: {model}", model_run_id=model_run_id
                ),
            ),
            inputs=input_datasets,
            producer="https://github.com/LBNL-UCB-STI/PILATES",
        )
        self._emit_event(event, model_run_id)
        return model_run_id

    def complete_model_run(
        self,
        run_hash: str,
        status: str = "completed",
        output_records: List[Union[FileRecord, RepoRecord]] = None,
        metadata: dict = None,
    ):
        """
        Completes a model run and emits an OpenLineage COMPLETE or FAIL event.

        This method extends the parent's `complete_model_run` by also generating
        and emitting an OpenLineage event to signal the completion (or failure)
        of a job execution.

        Args:
            run_hash (str): The unique ID of the model run to complete.
            status (str, optional): Final status (e.g., 'completed' or 'failed').
                                    Defaults to "completed".
            output_records (List[Union[FileRecord, RepoRecord]], optional): Optional list
                                                                             of `FileRecord`
                                                                             or `RepoRecord`
                                                                             objects produced by the run.
            metadata (dict, optional): Optional dictionary with runtime execution metadata.
        """
        if output_records is None:
            output_records = []

        output_names = [dataset.short_name for dataset in output_records]

        model_run_info = self.run_info.model_runs.get(run_hash)
        year = None
        iteration = None
        if model_run_info:
            model_name = model_run_info.model
            run_id_uuid = model_run_info.openlineage_id
            year = model_run_info.year
            iteration = model_run_info.iteration
            for record_hash in model_run_info.output_record_hashes:
                if record_hash in self.run_info.file_records:
                    file_record = self.run_info.file_records[record_hash]
                    if file_record.short_name not in output_names:
                        output_records.append(file_record)
                elif record_hash in self.run_info.repo_records:
                    repo_record = self.run_info.repo_records[record_hash]
                    if repo_record.short_name not in output_names:
                        output_records.append(repo_record)

            output_datasets = [
                record.toOutputDataset(self.namespace) for record in output_records
            ]

            event_type = RunState.COMPLETE if status == "completed" else RunState.FAIL
            event = RunEvent(
                eventType=event_type,
                eventTime=datetime.now().isoformat(),
                run=Run(runId=run_id_uuid),
                job=Job(
                    namespace=self.namespace,
                    name=self._format_name(model_name, year, iteration),
                    facets=self._get_job_facets(
                        model_run_info.description or f"Pilates model: {model_name}",
                        model_run_id=run_hash,
                    ),
                ),
                outputs=output_datasets,
                producer="https://github.com/LBNL-UCB-STI/PILATES",
            )
            self._emit_event(event, run_hash)
        super().complete_model_run(run_hash, status, output_records, metadata)
