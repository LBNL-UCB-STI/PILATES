import uuid
import logging
import hashlib
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


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
    Enhanced provenance tracking that maintains comprehensive run_info.json files.
    """

    def __init__(self, run_id: str, output_path: str, folder_name: str = None):
        self.run_id = run_id
        self.output_path = os.path.abspath(output_path) if output_path else None
        self.folder_name = folder_name
        self.run_info_path = self._get_run_info_path()
        self.run_info = self._initialize_run_info()
        logger.info(f"ProvenanceTracker initialized for run ID: {self.run_id}")

    def _get_run_info_path(self) -> str:
        """Get the path to the run_info.json file."""
        if self.folder_name:
            return os.path.join(self.output_path, self.folder_name, "run_info.json")
        else:
            # Fallback if no folder_name, though folder_name is expected with output_path
            return os.path.join(self.output_path, "run_info.json")

    def _load_metadata(self, file_path: str) -> Dict[str, Any]:
        """Load metadata from a JSON file located in the same directory as the file."""
        metadata_file = os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path)}.metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load metadata from {metadata_file}: {e}")
        return {}

    def is_git_repo(self, path: str) -> bool:
        """Check if a directory is a git repository (accepts .git as file or directory)."""
        abs_path = os.path.abspath(path)
        git_dir = os.path.join(abs_path, '.git')
        is_repo = os.path.exists(git_dir)
        logger.debug(f"Checking if path {abs_path} is a git repo: {is_repo}")
        return is_repo

    def get_git_hash(self, repo_path: str = None) -> Optional[str]:
        """Get the current git commit hash."""
        try:
            abs_repo_path = os.path.abspath(repo_path) if repo_path else os.path.dirname(__file__)
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=abs_repo_path,
                timeout=5,
            )
            logger.debug(f"Git hash for repo at {abs_repo_path}: {result.stdout.strip()}")
            return result.stdout.strip()
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ) as e:
            logger.warning(f"Could not determine git hash for repo at {repo_path}: {e}")
            return None

    def _validate_file_path(self, file_path: str) -> Optional[str]:
        """
        Validate and normalize file path.

        Args:
            file_path: Path to validate

        Returns:
            Absolute path if file exists, None otherwise
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

    def _get_validated_paths(self, file_path: str, skip_missing: bool) -> (Optional[str], Optional[str]):
        """
        Validate file path and get both absolute and relative paths.

        Args:
            file_path: Path to validate
            skip_missing: Whether to skip missing files

        Returns:
            Tuple of absolute path and relative path if valid, otherwise (None, None)
        """
        abs_path = self._validate_file_path(file_path)
        if not abs_path and skip_missing:
            logger.debug(f"Skipping missing file: {file_path}")
            return None, None

        # Use original path if validation failed but skip_missing is False
        path_to_use = abs_path or file_path
        relative_path = self._get_relative_path(path_to_use)
        return path_to_use, relative_path

    def _calculate_file_hash(self, file_path: str) -> Optional[str]:
        """Calculate SHA256 hash of a file with improved error handling."""
        abs_file_path = self._validate_file_path(file_path)
        if not abs_file_path:
            return None

        try:
            sha256_hash = hashlib.sha256()
            with open(abs_file_path, "rb") as f:
                # Read and update hash string value in blocks of 4K
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except (IOError, OSError) as e:
            logger.warning(f"Could not calculate hash for {abs_file_path}: {e}")
            return None
        """Calculate SHA256 hash of a file with improved error handling."""
        abs_file_path = self._validate_file_path(file_path)
        if not abs_file_path:
            return None

        try:
            sha256_hash = hashlib.sha256()
            with open(abs_file_path, "rb") as f:
                # Read and update hash string value in blocks of 4K
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except (IOError, OSError) as e:
            logger.warning(f"Could not calculate hash for {abs_file_path}: {e}")
            return None

    def _calculate_settings_hash(self, settings: Dict[str, Any]) -> str:
        """Calculate SHA256 hash of settings dictionary."""
        # Create a stable string representation of settings by sorting keys
        # Use default=str to handle non-serializable types if any
        settings_str = json.dumps(settings, sort_keys=True, default=str)
        return hashlib.sha256(settings_str.encode("utf-8")).hexdigest()

    def _get_relative_path(self, file_path: str) -> str:
        """Get path relative to output directory for consistent storage."""
        abs_path = os.path.abspath(file_path)
        base_path = self.output_path or os.getcwd()
        try:
            # Ensure the path is relative to the base directory
            return os.path.relpath(abs_path, base_path)
        except ValueError:
            # Can happen on Windows with different drives
            logger.warning(
                f"Could not create relative path for {abs_path} relative to {base_path}"
            )
            return abs_path

    def initialize_from_settings(self, settings: Dict[str, Any]):
        """Initialize run info from settings."""
        self.run_info.update(
            {
                "start_year": settings.get("start_year"),
                "end_year": settings.get("end_year"),
                "settings_hash": self._calculate_settings_hash(settings),
            }
        )

        # Determine models used from settings
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

        self.run_info["models_used"] = list(set(models_used))  # Remove duplicates
        self._save_run_info()
        logger.info("ProvenanceTracker initialized with settings.")

    def record_repo_input(
        self,
        model: str,
        repo_path: str,
        description: str = None,
        git_hash: str = None,
    ):
        """
        Record a repository input for a model.

        Args:
            model: Name of the model using this input
            repo_path: Path to the repository
            description: Description of the repository
            git_hash: Git hash of the repository
        """
        abs_path = self._validate_file_path(repo_path)
        if not abs_path:
            logger.warning(f"Skipping missing repository for {model}: {repo_path}")
            return

        relative_path = self._get_relative_path(abs_path)

        if model not in self.run_info["inputs"]["repos"]:
            self.run_info["inputs"]["repos"][model] = []

        repo_record = {
            "repo_path": relative_path,
            "git_hash": git_hash,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "exists": relative_path is not None,
        }

        self.run_info["inputs"]["repos"][model].append(repo_record)
        self._save_run_info()
        logger.debug(
            f"Recorded repository input for {model}: {relative_path} (exists: {abs_path is not None})"
        )
    def record_input_file(
        self,
        model: str,
        file_path: str,
        source_run_id: str = None,
        description: str = None,
        source_file_paths: List[str] = None,  # New parameter for source file paths
        skip_missing: bool = True,
    ):
        """
        Record an input file for a model with validation.

        Args:
            model: Name of the model using this input
            file_path: Path to the input file
            source_run_id: Run ID that produced this file (if known)
            description: Description of the input file
            skip_missing: If True, skip recording missing files; if False, record anyway
        """
        # Load metadata and apply it to the input record
        metadata = self._load_metadata(file_path)
        path_to_use, relative_path = self._get_validated_paths(file_path, skip_missing)
        if not path_to_use:
            return

        if model not in self.run_info["inputs"]["files"]:
            self.run_info["inputs"]["files"][model] = []

        input_record = {
            "run_id": [self.run_id],  # Associate the current run_id
            "file_path": relative_path,
            "source_run_id": source_run_id,
            "source_file_paths": [],  # New field to store source file paths
            "file_hash": self._calculate_file_hash(path_to_use) if path_to_use else None,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "exists": path_to_use is not None,
        }

        # Integrate metadata into the main file's record
        if metadata:
            input_record.update(metadata)


        if source_file_paths:
            input_record["source_file_paths"] = [
                self._get_relative_path(path) for path in source_file_paths
            ]
            # Lookup hash for source files if they have been logged
            source_hashes = []
            source_run_ids = []
            for path in source_file_paths:
                found = False
                for model_inputs in self.run_info["inputs"]["files"].values():
                    for record in model_inputs:
                        if record["file_path"] == self._get_relative_path(path):
                            source_hashes.append(record["file_hash"])
                            source_run_ids.append(record.get("source_run_id"))
                            found = True
                            break
                    if found:
                        break
            input_record["source_file_hashes"] = source_hashes
            input_record["source_run_id"] = source_run_ids

        self.run_info["inputs"]["files"][model].append(input_record)
        self._save_run_info()
        logger.debug(
            f"Recorded input for {model}: {input_record['file_path']} (exists: {path_to_use is not None})"
        )

    def update_file_path(self, model: str, old_path: str, new_path: str):
        """
        Update the file path for a recorded file when it is archived or moved.

        Args:
            model: Name of the model associated with the file.
            old_path: The old file path.
            new_path: The new file path.
        """
        relative_old_path = self._get_relative_path(old_path)
        relative_new_path = self._get_relative_path(new_path)

        # Update input file paths
        if model in self.run_info["inputs"]:
            for input_record in self.run_info["inputs"][model]:
                if input_record["file_path"] == relative_old_path:
                    input_record["file_path"] = relative_new_path

        # Update output file paths
        if model in self.run_info["outputs"]:
            for output_record in self.run_info["outputs"][model]:
                if output_record["file_path"] == relative_old_path:
                    output_record["file_path"] = relative_new_path

        self._save_run_info()

    def record_output_file(
        self,
        model: str,
        file_path: str,
        year: int = None,
        description: str = None,
        skip_missing: bool = True,
    ):
        """
        Record an output file for a model with validation.

        Args:
            model: Name of the model that produced this output
            file_path: Path to the output file
            year: Simulation year the output corresponds to
            description: Description of the output file
            skip_missing: If True, skip recording missing files; if False, record anyway
        """
        path_to_use, relative_path = self._get_validated_paths(file_path, skip_missing)
        if not path_to_use:
            return

        if model not in self.run_info["outputs"]:
            self.run_info["outputs"][model] = []

        output_record = {
            "run_id": self.run_id,
            "file_path": relative_path,
            "file_hash": self._calculate_file_hash(path_to_use) if path_to_use else None,
            "created_at": datetime.now().isoformat(),
            "year": year,
            "description": description,
            "exists": path_to_use is not None,
        }

        self.run_info["outputs"][model].append(output_record)
        self._save_run_info()
        logger.debug(
            f"Recorded output for {model}: {relative_path} (exists: {path_to_use is not None})"
        )

    def record_directory_inputs(
        self,
        model: str,
        input_dir: str,
        pattern: str = "*",
        description: str = None,
        recursive: bool = False,
    ):
        """
        Record all files in a directory matching a pattern as inputs.

        Args:
            model: Name of the model using these inputs
            input_dir: Directory containing input files
            pattern: File pattern to match (default: all files)
            description: Base description for the files
            recursive: Whether to search recursively
        """
        import glob

        if self.is_git_repo(input_dir):
            repo_name = os.path.basename(input_dir)
            git_hash = self.get_git_hash(input_dir)
            logger.info(f"Recording git repo {repo_name} with hash {git_hash} as input")
            self.record_input_file(model, input_dir, description=f"Git repo {repo_name} at {git_hash}")
            return

        if not os.path.exists(input_dir):
            logger.warning(f"Input directory does not exist: {input_dir}")
            return

        search_pattern = (
            os.path.join(input_dir, "**", pattern)
            if recursive
            else os.path.join(input_dir, pattern)
        )
        files = glob.glob(search_pattern, recursive=recursive)

        for file_path in files:
            if os.path.isfile(file_path):
                file_desc = (
                    f"{description} - {os.path.basename(file_path)}"
                    if description
                    else os.path.basename(file_path)
                )
                self.record_input_file(model, file_path, description=file_desc)

        logger.info(f"Recorded {len(files)} input files for {model} from {input_dir}")

    def record_directory_outputs(
        self,
        model: str,
        output_dir: str,
        pattern: str = "*",
        description: str = None,
        year: int = None,
        recursive: bool = False,
    ):
        """
        Record all files in a directory matching a pattern as outputs.

        Args:
            model: Name of the model that produced these outputs
            output_dir: Directory containing output files
            pattern: File pattern to match (default: all files)
            description: Base description for the files
            year: Simulation year the outputs correspond to
            recursive: Whether to search recursively
        """
        import glob

        if self.is_git_repo(output_dir):
            repo_name = os.path.basename(output_dir)
            git_hash = self.get_git_hash(output_dir)
            logger.info(f"Recording git repo {repo_name} with hash {git_hash} as output")
            self.record_output_file(model, output_dir, year=year, description=f"Git repo {repo_name} at {git_hash}")
            return

        if not os.path.exists(output_dir):
            logger.warning(f"Output directory does not exist: {output_dir}")
            return

        search_pattern = (
            os.path.join(output_dir, "**", pattern)
            if recursive
            else os.path.join(output_dir, pattern)
        )
        files = glob.glob(search_pattern, recursive=recursive)

        for file_path in files:
            if os.path.isfile(file_path):
                file_desc = (
                    f"{description} - {os.path.basename(file_path)}"
                    if description
                    else os.path.basename(file_path)
                )
                self.record_output_file(
                    model, file_path, year=year, description=file_desc
                )

        logger.info(f"Recorded {len(files)} output files for {model} from {output_dir}")

    def record_model_io_batch(
        self,
        model: str,
        inputs: List[str] = None,
        outputs: List[str] = None,
        year: int = None,
        input_descriptions: List[str] = None,
        output_descriptions: List[str] = None,
    ):
        """
        Record multiple input and output files for a model in batch.

        Args:
            model: Name of the model
            inputs: List of input file paths
            outputs: List of output file paths
            year: Simulation year for outputs
            input_descriptions: Descriptions for input files (same length as inputs)
            output_descriptions: Descriptions for output files (same length as outputs)
        """
        if inputs:
            input_descs = input_descriptions or [None] * len(inputs)
            for file_path, desc in zip(inputs, input_descs):
                self.record_input_file(model, file_path, description=desc)

        if outputs:
            output_descs = output_descriptions or [None] * len(outputs)
            for file_path, desc in zip(outputs, output_descs):
                self.record_output_file(model, file_path, year=year, description=desc)

        logger.info(
            f"Batch recorded {len(inputs or [])} inputs and {len(outputs or [])} outputs for {model}"
        )

    def start_model_run(
        self,
        model: str,
        year: int = None,
        iteration: int = None,
        description: str = None,
    ) -> int:
        """Record the start of a model run."""
        run_id = f"{model}_{datetime.now().strftime('%Y-%m-%d')}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting model run with ID: {run_id}")

        run_id = f"{model}_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting model run with ID: {run_id}")

        run_record = {
            "run_id": run_id,
            "run_id": run_id,
            "model": model,
            "year": year,
            "iteration": iteration,
            "description": description,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "status": "running",
        }

        self.run_info["model_runs"].append(run_record)
        self._save_run_info()
        logger.debug(f"Started model run: {model} (year={year}, iteration={iteration})")
        return len(self.run_info["model_runs"]) - 1  # Return index for completion

    def complete_model_run(self, run_index: int, status: str = "completed"):
        """Record the completion of a model run."""
        if 0 <= run_index < len(self.run_info["model_runs"]):
            self.run_info["model_runs"][run_index].update(
                {"completed_at": datetime.now().isoformat(), "status": status}
            )
            self._save_run_info()

            model = self.run_info["model_runs"][run_index]["model"]
            logger.debug(f"Completed model run: {model} with status {status}")
        else:
            logger.warning(
                f"Attempted to complete model run with invalid index: {run_index}"
            )

    def _save_run_info(self, data_to_save: Dict[str, Any] = None):
        """Save the run_info to JSON file."""
        data = data_to_save if data_to_save is not None else self.run_info
        os.makedirs(os.path.dirname(self.run_info_path), exist_ok=True)

        try:
            with open(self.run_info_path, "w") as f:
                # Use default=str to handle datetime objects if any sneak in
                json.dump(data, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Could not save run_info.json to {self.run_info_path}: {e}")

    def get_run_info(self) -> Dict[str, Any]:
        """Get the current run info."""
        # Reload from file to ensure it's the latest state
        if os.path.exists(self.run_info_path):
            try:
                with open(self.run_info_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not reload run_info.json for get_run_info: {e}")
                return self.run_info.copy()  # Return in-memory copy if reload fails
        return self.run_info.copy()  # Return in-memory copy if file doesn't exist

    def _initialize_run_info(self) -> Dict[str, Any]:
        """Initialize or load existing run_info structure."""
        if os.path.exists(self.run_info_path):
            try:
                with open(self.run_info_path, "r") as f:
                    run_info = json.load(f)
                    # Ensure required keys exist for backward compatibility if needed
                    run_info.setdefault("inputs", {"files": {}, "repos": {}})
                    run_info.setdefault("outputs", {})
                    run_info.setdefault("model_runs", [])
                    run_info.setdefault("models_used", [])
                    run_info.setdefault("settings_hash", None)
                    run_info.setdefault("code_version", self.get_git_hash())
                    run_info.setdefault(
                        "hostname",
                        os.uname().nodename if hasattr(os, "uname") else "unknown",
                    )
                    return run_info
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"Could not load existing run_info.json: {e}. Creating new one."
                )

        # Create a new structure
        new_run_info = {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "start_year": None,
            "end_year": None,
            "models_used": [],
            "settings_hash": None,
            "code_version": self.get_git_hash(),
            "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "inputs": {"files": {}, "repos": {}},
            "outputs": {},
            "model_runs": [],
        }
        self._save_run_info(new_run_info)  # Save immediately on creation
        return new_run_info

    # Helper methods for postprocessing scripts (optional, but good for utility)
    def find_input_by_pattern(self, model: str, pattern: str) -> List[Dict[str, Any]]:
        """Find input files matching a pattern for a specific model."""
        import fnmatch

        if model not in self.run_info.get("inputs", {}):
            return []

        matching_inputs = []
        for input_record in self.run_info["inputs"][model]:
            if fnmatch.fnmatch(input_record["file_path"], pattern):
                matching_inputs.append(input_record)

        return matching_inputs

    def find_output_by_pattern(self, model: str, pattern: str) -> List[Dict[str, Any]]:
        """Find output files matching a pattern for a specific model."""
        import fnmatch

        if model not in self.run_info.get("outputs", {}):
            return []

        matching_outputs = []
        for output_record in self.run_info["outputs"][model]:
            if fnmatch.fnmatch(output_record["file_path"], pattern):
                matching_outputs.append(output_record)

        return matching_outputs

    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of all model runs and their I/O."""
        summary = {
            "total_model_runs": len(self.run_info.get("model_runs", [])),
            "models_used": self.run_info.get("models_used", []),
            "input_files_by_model": {},
            "output_files_by_model": {},
            "run_status": {},
        }

        # Count inputs and outputs by model
        for model, inputs in self.run_info.get("inputs", {}).items():
            summary["input_files_by_model"][model] = len(inputs)

        for model, outputs in self.run_info.get("outputs", {}).items():
            summary["output_files_by_model"][model] = len(outputs)

        # Summarize run statuses
        for run in self.run_info.get("model_runs", []):
            model = run["model"]
            status = run.get("status", "unknown")
            if model not in summary["run_status"]:
                summary["run_status"][model] = {}
            summary["run_status"][model][status] = (
                summary["run_status"][model].get(status, 0) + 1
            )

        return summary
