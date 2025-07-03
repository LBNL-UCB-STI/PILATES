import uuid
import logging
import json
import os
import subprocess
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

from pilates.generic.records import FileRecord, RepoRecord, ModelRunInfo, RecordStore
from pilates.utils.git_utils import is_git_repo, get_git_hash
from pilates.utils.file_utils import (
    _validate_file_path,
    _get_relative_path,
    _calculate_file_hash,
    _load_metadata,
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


# Re-export for backward compatibility
is_git_repo = is_git_repo
get_git_hash = get_git_hash
_validate_file_path = _validate_file_path
_get_relative_path = _get_relative_path
_calculate_file_hash = _calculate_file_hash
_load_metadata = _load_metadata


@dataclass
class PilatesRunInfo:
    run_id: str
    created_at: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    models_used: list = field(default_factory=list)
    settings_hash: Optional[str] = None
    code_version: Optional[str] = None
    hostname: Optional[str] = None
    file_records: Dict[str, "FileRecord"] = field(default_factory=dict)
    repo_records: Dict[str, List[RepoRecord]] = field(default_factory=dict)
    model_runs: Dict[str, ModelRunInfo] = field(default_factory=dict)


class RunContext:
    """
    Manages the context for a single model run, including its unique ID
    and methods for recording provenance information.
    (Note: This class seems intended for database interaction and is kept separate
    from the file-based ProvenanceTracker for now.)
    """

    def __init__(
        self,
        run_id: str = None,
        parameters: dict = None,
        code_version: str = None,
        hostname: str = None,
    ):
        """
        Initializes the RunContext.

        Args:
            run_id (str, optional): A pre-defined run ID. If None, a new UUID is generated.
            parameters (dict, optional): Dictionary of run parameters.
            code_version (str, optional): Identifier for the code version (e.g., git hash).
            hostname (str, optional): Hostname where the run is executed.
        """
        self.run_id = run_id if run_id else str(uuid.uuid4())
        self.start_time = None
        self.end_time = None
        self.status = "initialized"
        self.parameters = parameters
        self.code_version = code_version
        self.hostname = hostname
        # logger.info(f"RunContext initialized with ID: {self.run_id}") # Avoid logging here to prevent double logging with ProvenanceTracker

    def record_run_start(self):
        """Records the start time and updates the status."""
        self.start_time = datetime.now()
        self.status = "running"
        # logger.info(f"Run {self.run_id} started at {self.start_time}") # Avoid logging here
        # TODO: Add database interaction to record run start in ModelRuns table

    def record_run_end(self, status: str = "completed"):
        """Records the end time and final status."""
        self.end_time = datetime.now()
        self.status = status
        # logger.info(f"Run {self.run_id} ended at {self.end_time} with status: {self.status}") # Avoid logging here
        # TODO: Add database interaction to update run end time and status in ModelRuns table

    def record_input(
        self, source_run_id: str, file_path: str, input_type: str = "unknown"
    ):
        """
        Records an input file consumed by the current run.

        Args:
            source_run_id (str): The run ID that produced this input file.
            file_path (str): The path to the input file.
            input_type (str, optional): A description of the input (e.g., 'ActivitySim Plans').
        """
        # logger.info(f"Run {self.run_id} consumed input: Type='{input_type}', Path='{file_path}', SourceRun='{source_run_id}'") # Avoid logging here
        # TODO: Add database interaction to record input in ModelInputs table.
        # This might involve looking up the source_output_id from ModelOutputs
        # based on source_run_id and file_path.

    def record_output(self, output_type: str, file_path: str):
        """
        Records an output file produced by the current run.

        Args:
            output_type (str): A description of the output (e.g., 'BEAM Plans GZ').
            file_path (str): The path where the output file was saved.
        """
        # logger.info(f"Run {self.run_id} produced output: Type='{output_type}', Path='{file_path}'") # Avoid logging here
        # TODO: Add database interaction to record output in ModelOutputs table.
        # The output_id would be generated here or in the database.


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
            git_hash=git_hash,
        )
        self.run_info.repo_records[model].append(repo_record)
        current_model_run = self.current_model_run()
        if current_model_run:
            current_model_run.input_record_hashes.append(repo_record.git_hash)

    def get_run_info(self) -> Dict[str, Any]:
        return self.run_info.copy()

    def get_model_summary(self) -> Dict[str, Any]:
        summary = {
            "total_model_runs": len(self.run_info.get("model_runs", [])),
            "models_used": self.run_info.get("models_used", []),
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

        for run in self.run_info.model_runs:
            model = run.model
            status = run.status
            if model not in summary["run_status"]:
                summary["run_status"][model] = {}
            summary["run_status"][model][status] = (
                summary["run_status"][model].get(status, 0) + 1
            )
        return summary


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

    def _calculate_file_hash(
        self, file_path: str, state: Optional[WorkflowState] = None
    ) -> Optional[str]:
        abs_file_path = self._validate_file_path(file_path)
        if not abs_file_path:
            return None
        try:
            sha256_hash = hashlib.sha256()
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
            status="running",
            input_record_hashes=inputs.all_unique_ids(),
        )
        self.current_model_run_id = model_run_id
        self.run_info.model_runs[model_run_id] = run_record
        self._save_run_info()
        return model_run_id

    def complete_model_run(self, run_hash: str, status: str = "completed"):
        if run_hash in self.run_info.model_runs:
            self.run_info.model_runs[run_hash].completed_at = datetime.now().isoformat()
            self.run_info.model_runs[run_hash].status = status
            self._save_run_info()
        else:
            logger.error(f"Model run hash {run_hash} not found to complete.")

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
        description: str = None,
        git_hash: str = None,
    ):
        model = self._normalize_model_name(model)
        abs_path = self._validate_file_path(repo_path)
        if not abs_path:
            logger.warning(f"Skipping missing repository for {model}: {repo_path}")
            return
        relative_path = self._get_relative_path(abs_path)
        if model not in self.run_info.repo_records:
            self.run_info.repo_records[model] = []

        repo_record = RepoRecord(
            unique_id=git_hash,
            repo_path=relative_path,
            accessed_at=datetime.now().isoformat(),
            description=description,
        )
        self.run_info.repo_records[model].append(repo_record)
        self._save_run_info()
        logger.debug(
            f"Recorded repository input for {model}: {relative_path} (exists: {abs_path is not None})"
        )

    def _get_or_create_file_record(
        self,
        file_path: str,
        skip_missing: bool = True,
        description: Optional[str] = None,
        short_name: Optional[str] = None,
        state: Optional[WorkflowState] = None
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

    def record_input_file(
        self,
        model: str,
        file_path: str,
        source_run_id: str = None,
        description: str = None,
        source_file_paths: List[str] = None,
        skip_missing: bool = True,
        model_run_id: str = None,
        state: Optional[WorkflowState] = None
    ) -> Optional[FileRecord]:
        model = self._normalize_model_name(model)
        file_record = self._get_or_create_file_record(
            file_path, skip_missing, description, state=state
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
        import dataclasses

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


# Backward compatibility: ProvenanceTracker = FileProvenanceTracker
ProvenanceTracker = FileProvenanceTracker
