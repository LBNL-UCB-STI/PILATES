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
import shutil
import subprocess
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import attr
from openlineage.client import set_producer
from openlineage.client.facet import DocumentationJobFacet, SourceCodeLocationJobFacet
from openlineage.client.run import RunEvent, Run, Job

set_producer("https://github.com/LBNL-UCB-STI/PILATES")

from pilates.generic.records import (
    FileRecord,
    RepoRecord,
    ModelRunInfo,
    RecordStore,
    Record,
    PilatesRunInfo,
)

from workflow_state import WorkflowState

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
        self.run_id = run_id
        self.output_path = output_path
        self.folder_name = folder_name
        self.run_info = PilatesRunInfo(
            run_id=run_id, created_at=datetime.now().isoformat()
        )
        self.current_model_run_id = None

    def _save_run_info(self):
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
        return model.lower() if model else model

    def initialize_from_settings(self, settings: Dict[str, Any]):
        self.run_info.start_year = settings.get("start_year")
        self.run_info.end_year = settings.get("end_year")
        self.run_info.settings_hash = None  # FileProvenanceTracker will set this
        models_used = []
        model_keys = [
            "land_use_model",
            "vehicle_ownership_model",
            "activity_demand_model",
            "travel_model",
        ]
        for key in model_keys:
            if settings.get(key):
                models_used.append(settings[key])
        self.run_info.models_used = list(set(models_used))

    def record_repo_input(
        self,
        model: str,
        repo_path: str,
        description: str = None,
        git_hash: str = None,
    ):
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
        return self.run_info.copy()

    def get_model_summary(self) -> Dict[str, Any]:
        summary = {
            "total_model_runs": len(self.run_info.model_runs),
            "models_used": self.run_info.model_runs,
            "input_files_by_model": {},
            "output_files_by_model": {},
            "run_status": {},
        }
        # This needs to be updated to reflect the new data structure
        # For now, we can count files based on the models that touched them
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
    ) -> str:
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
    ):
        if output_records is None:
            output_records = []
        if run_hash in self.run_info.model_runs:
            self.run_info.model_runs[run_hash].completed_at = datetime.now().isoformat()
            self.run_info.model_runs[run_hash].status = status
            for dataset in output_records:
                if isinstance(dataset, Record):
                    if (
                        dataset.unique_id
                        not in self.run_info.model_runs[run_hash].output_record_hashes
                    ):
                        logger.info(
                            f"Adding dataset {dataset.short_name} to model run {run_hash} outputs, despite it "
                            f"not being flagged in the main model run."
                        )
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
        super().__init__(run_id, output_path, folder_name)
        self.output_path = os.path.abspath(output_path) if output_path else None
        self.folder_name = folder_name
        self.run_info_path = self._get_run_info_path()
        self.run_info = self._initialize_run_info()
        logger.info(f"FileProvenanceTracker initialized for run ID: {self.run_id}")

    def _get_run_info_path(self) -> str:
        if self.folder_name:
            return os.path.join(self.output_path, self.folder_name, "run_info.json")
        else:
            return os.path.join(self.output_path, "run_info.json")

    def _load_metadata(self, file_path: str) -> Dict[str, Any]:
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
        abs_path = os.path.abspath(path)
        git_dir = os.path.join(abs_path, ".git")
        is_repo = os.path.exists(git_dir)
        logger.debug(f"Checking if path {abs_path} is a git repo: {is_repo}")
        return is_repo

    def get_git_hash(self, repo_path: str = None) -> Optional[str]:
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
    ) -> (Optional[str], Optional[str]):
        abs_path = self._validate_file_path(file_path)
        if not abs_path and skip_missing:
            logger.debug(f"Skipping missing file: {file_path}")
            return None, None
        path_to_use = abs_path or file_path
        relative_path = self._get_relative_path(path_to_use)
        return path_to_use, relative_path

    def _calculate_directory_hash(
            self, abs_dir_path: str, state: Optional[WorkflowState] = None
    ) -> Optional[str]:
        """
        Calculates a SHA-256 hash of a directory based on its file and subdirectory names,
        and the size and modification time of each file. It optionally incorporates additional
        state information.

        Args:
            abs_dir_path (str): The path to the directory to hash.
            state (Optional[WorkflowState]): Optional state metadata to include in the hash.

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
        self, file_path: str, state: Optional[WorkflowState] = None
    ) -> Optional[str]:
        """
        Calculates the SHA-256 hash of a file, depending on its contents and its location. It also optionally
        incorporates additional state information, such as the year and iteration for which it was created.

        Args:
            file_path (str): The path to the file whose hash is to be calculated.
            state (Optional[WorkflowState]): An optional state object containing additional metadata
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
            logger.error(f"Could not calculate hash for {abs_file_path}: {e}")
            return None

    def _calculate_file_hash(
        self, abs_file_path: str, state: Optional[WorkflowState] = None
    ) -> Optional[str]:
        """
        Calculates the SHA-256 hash of a file, depending on its contents and its location. It also optionally
        incorporates additional state information, such as the year and iteration for which it was created.

        Args:
            abs_file_path (str): The path to the file whose hash is to be calculated.
            state (Optional[WorkflowState]): An optional state object containing additional metadata
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

    def _calculate_settings_hash(self, settings: Dict[str, Any]) -> str:
        settings_str = json.dumps(settings, sort_keys=True, default=str)
        return hashlib.sha256(settings_str.encode("utf-8")).hexdigest()

    def _get_relative_path(self, file_path: str) -> str:
        abs_path = os.path.abspath(file_path)
        base_path = self.output_path or os.getcwd()
        try:
            return os.path.relpath(abs_path, base_path)
        except ValueError:
            logger.warning(
                f"Could not create relative path for {abs_path} relative to {base_path}"
            )
            return abs_path

    def initialize_from_settings(self, settings: Dict[str, Any]):
        self.run_info.start_year = settings.get("start_year")
        self.run_info.end_year = settings.get("end_year")
        self.run_info.settings_hash = self._calculate_settings_hash(settings)
        models_used = []
        model_keys = [
            "land_use_model",
            "vehicle_ownership_model",
            "activity_demand_model",
            "travel_model",
        ]
        for key in model_keys:
            if settings.get(key):
                models_used.append(settings[key])
        self.run_info.models_used = list(set(models_used))
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
        if git_hash is None:
            git_hash = self._calculate_path_hash(repo_path)
        model = self._normalize_model_name(model)
        abs_path = self._validate_file_path(repo_path)
        if not abs_path:
            logger.warning(f"Skipping missing repository for {model}: {repo_path}")
            return
        relative_path = self._get_relative_path(abs_path)

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

    def _get_or_create_file_record(
        self,
        file_path: str,
        skip_missing: bool = True,
        description: Optional[str] = None,
        short_name: Optional[str] = None,
        state: Optional[WorkflowState] = None,
    ) -> Optional[FileRecord]:
        path_to_use, relative_path = self._get_validated_paths(file_path, skip_missing)
        if not path_to_use:
            return None

        file_hash = self._calculate_file_hash(path_to_use, state)
        if not file_hash:
            logger.warning(
                f"Could not calculate hash for {file_path}, cannot create record."
            )
            return None

        if file_hash in self.run_info.file_records:
            return self.run_info.file_records[file_hash]

        metadata = self._load_metadata(path_to_use)
        file_record = FileRecord(
            unique_id=file_hash,
            file_path=relative_path,
            created_at=datetime.now().isoformat(),
            short_name=short_name,
            metadata=metadata,
            description=description,
        )
        self.run_info.file_records[file_hash] = file_record
        return file_record

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
        state: Optional[WorkflowState] = None,
    ) -> Optional[FileRecord]:
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
                short_name=record.short_name,
                model_run_id=self.current_model_run_id,
                state=state,
            )
            return output_record
        else:
            raise NotImplementedError("You have to move git repos manually")

    def record_input_record(self, record: Record, model_run_id: str = None):
        if model_run_id is None:
            model_run_id = self.current_model_run_id
        self.run_info.model_runs[model_run_id].input_record_hashes.append(
            record.unique_id
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
        state: Optional[WorkflowState] = None,
    ) -> Optional[FileRecord]:
        model = self._normalize_model_name(model)
        file_record = self._get_or_create_file_record(
            file_path, skip_missing, description, short_name=short_name, state=state
        )
        if not file_record:
            return None

        if model and model not in file_record.models:
            file_record.models.append(model)
        if description:
            file_record.description = description
        if source_file_paths:
            file_record.source_file_paths = [
                self._get_relative_path(p) for p in source_file_paths
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
        state: Optional[WorkflowState] = None,
    ) -> Optional[FileRecord]:
        model = self._normalize_model_name(model)
        file_record = self._get_or_create_file_record(
            file_path, skip_missing, description, short_name, state=state
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
                self._get_relative_path(p) for p in source_file_paths
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

    def _save_run_info(self, data_to_save: PilatesRunInfo = None):
        import dataclasses

        data = data_to_save if data_to_save is not None else self.run_info
        os.makedirs(os.path.dirname(self.run_info_path), exist_ok=True)
        try:
            with open(self.run_info_path, "w") as f:
                json.dump(dataclasses.asdict(data), f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Could not save run_info.json to {self.run_info_path}: {e}")

    def get_run_info(self) -> dict:
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

    def _initialize_run_info(self) -> PilatesRunInfo:

        if os.path.exists(self.run_info_path):
            try:
                with open(self.run_info_path, "r") as f:
                    data = json.load(f)
                    # Convert dicts/lists to dataclasses for all fields
                    run_info = PilatesRunInfo(
                        run_id=data.get("run_id"),
                        created_at=data.get("created_at"),
                        start_year=data.get("start_year"),
                        end_year=data.get("end_year"),
                        models_used=data.get("models_used", []),
                        settings_hash=data.get("settings_hash"),
                        code_version=data.get("code_version"),
                        hostname=data.get("hostname"),
                        file_records=data.get("file_records", {}),
                        repo_records=data.get("repo_records", {}),
                        model_runs=data.get("model_runs", {}),
                    )
                    return run_info
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"Could not load existing run_info.json: {e}. Creating new one."
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
    in a file named `openlineage.jsonl`.
    """

    def __init__(
        self,
        run_id: str,
        output_path: str,
        folder_name: str = None,
        use_file: bool = True,
        use_marquez: bool = True,
        marquez_url: str = "http://localhost:5002",
    ):
        """
        Initializes the OpenLineageTracker instance.

        Args:
            run_id (str): Unique identifier for the run.
            output_path (str): Path to the output directory where logs will be stored.
            folder_name (str, optional): Name of the folder within the output path for storing logs.
            use_file (bool): Whether to write events to a local file (default True).
            use_marquez (bool): Whether to send events to Marquez (default False).
            marquez_url (str): URL of the Marquez server (default "http://localhost:5000").

        Note:
            To use Marquez, ensure the OpenLineage client is installed and configured correctly.
            `docker-compose up -d`
            `python run.py --settings settings.yaml`
            `docker-compose down`
        """
        super().__init__(run_id, output_path, folder_name)
        self.namespace = "default"
        self.use_file = use_file
        self.use_marquez = use_marquez

        # Setup file logging if enabled
        self.log_path = None
        if self.use_file and self.output_path:
            self.log_path = os.path.join(
                self.output_path, self.folder_name or "", "openlineage.jsonl"
            )
            if os.path.exists(self.log_path):
                os.remove(self.log_path)

        # Setup Marquez client if enabled
        self.marquez_client = None
        if self.use_marquez:
            from openlineage.client import OpenLineageClient

            self.marquez_client = OpenLineageClient(url=marquez_url)

    def _emit_event(self, event: RunEvent):
        """
        Emits an OpenLineage event to configured destinations.

        Args:
            event (RunEvent): The OpenLineage event to emit.
        """
        # Emit to file if enabled
        if self.use_file and self.log_path:
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(attr.asdict(event)) + "\n")
            except IOError as e:
                logger.error(f"Could not write to OpenLineage log file: {e}")

        # Emit to Marquez if enabled
        if self.use_marquez and self.marquez_client:
            try:
                self.marquez_client.emit(event)
            except Exception as e:
                logger.error(f"Could not send event to Marquez: {e}")

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

    def move_file(
        self,
        record: Record,
        source_path: str,
        destination_path: str,
        model: str,
        state: Optional[WorkflowState] = None,
    ) -> Optional[FileRecord]:
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
    ) -> str:
        """Start a model run and emit OpenLineage event."""
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
        for record in inputs.records.values():
            dataset = record.toInputDataset(self.namespace)
            if dataset.name not in input_names:
                input_datasets.append(dataset)
                logger.info(f"Adding input dataset {dataset.name} to run inputs.")
            else:
                logger.warning(
                    f"Input dataset {dataset.name} already exists in the run inputs, skipping duplicate."
                )

        event = RunEvent(
            eventType="START",
            eventTime=datetime.now().isoformat(),
            run=Run(runId=run_id_uuid),
            job=Job(
                namespace=self.namespace,
                name=model,
                facets=self._get_job_facets(
                    description or f"Pilates model: {model}", model_run_id=model_run_id
                ),
            ),
            inputs=input_datasets,
            producer="https://github.com/LBNL-UCB-STI/PILATES",
        )
        self._emit_event(event)
        return model_run_id

    def complete_model_run(
        self,
        run_hash: str,
        status: str = "completed",
        output_records: List[Union[FileRecord, RepoRecord]] = None,
    ):
        """Complete a model run and emit OpenLineage event."""
        if output_records is None:
            output_records = []

        output_names = [dataset.short_name for dataset in output_records]

        model_run_info = self.run_info.model_runs.get(run_hash)
        if model_run_info:
            model_name = model_run_info.model
            run_id_uuid = model_run_info.openlineage_id
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

            event_type = "COMPLETE" if status == "completed" else "FAIL"
            event = RunEvent(
                eventType=event_type,
                eventTime=datetime.now().isoformat(),
                run=Run(runId=run_id_uuid),
                job=Job(
                    namespace=self.namespace,
                    name=model_name,
                    facets=self._get_job_facets(
                        model_run_info.description or f"Pilates model: {model_name}",
                        model_run_id=run_hash,
                    ),
                ),
                outputs=output_datasets,
                producer="https://github.com/LBNL-UCB-STI/PILATES",
            )
            self._emit_event(event)
        super().complete_model_run(run_hash, status, output_records)
