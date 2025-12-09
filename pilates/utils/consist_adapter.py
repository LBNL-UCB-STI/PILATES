"""
pilates/utils/consist_adapter.py

Consist Adapter for PILATES Provenance Tracking.
Provides compatibility between PILATES and Consist, allowing gradual migration.
Includes support for 'Attach Mode' when running inside Consist Scenarios.
"""

import hashlib
import logging
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from consist import Tracker, Artifact, Run

from pilates.config import PilatesConfig
from pilates.generic.records import (
    FileRecord,
    RepoRecord,
    ModelRunInfo,
    RecordStore,
    PilatesRunInfo,
    H5TableRecord,
    H5FileRecord,
)
from pilates.generic.execution_context import ExecutionContext

logger = logging.getLogger(__name__)


def _inject_workflow_context(state: Optional[ExecutionContext]) -> Dict[str, Any]:
    """
    Extract PILATES workflow context into Consist run metadata.
    """
    if state is None:
        return {}

    # Handle Enum serialization for stage
    stage = getattr(state, "current_major_stage", None)
    if stage is not None:
        if hasattr(stage, "name"):
            stage = stage.name
        elif not isinstance(stage, (str, int, float, bool)):
            stage = str(stage)

    return {
        "pilates_year": getattr(state, "current_year", None),
        "pilates_stage": stage,
        "pilates_iteration": getattr(state, "current_inner_iter", None),
    }


class ConsistProvenanceTracker:
    """
    Adapter that provides FileProvenanceTracker interface backed by Consist.
    """

    def __init__(
        self,
        run_id: str,
        output_path: str,
        folder_name: str = None,
        db_path: str = None,
        mounts: Dict[str, str] = None,
        tracker: Optional[Tracker] = None  # Added for Attach Mode
    ):
        self.run_id = run_id
        self.output_path = os.path.abspath(output_path) if output_path else None
        self.folder_name = folder_name

        if self.folder_name:
            self.run_info_path = os.path.join(
                self.output_path, self.folder_name, "run_info.json"
            )
        elif self.output_path:
            self.run_info_path = os.path.join(self.output_path, "run_info.json")
        else:
            self.run_info_path = ""

        self.workspace_root = (
            os.path.join(self.output_path, self.folder_name)
            if self.folder_name and self.output_path
            else self.output_path
        )

        # Internal flag to track if we own the run lifecycle (Attach Mode)
        self._attached_mode = False

        if tracker:
            self._tracker = tracker
        else:
            # DEFINE MOUNTS (Standalone Mode)
            mounts = mounts or {}
            if self.workspace_root:
                mounts["workspace"] = self.workspace_root

            self._tracker = Tracker(
                run_dir=Path(self.workspace_root or "."),
                db_path=db_path,
                mounts=mounts,
                project_root=str(Path(self.workspace_root or ".")),
                hashing_strategy="fast",
            )

        # Track H5 containers for parent-child linking
        self._h5_containers: Dict[str, Artifact] = {}

        # Current model run context
        self._current_model_run_id: Optional[str] = None
        self._current_run: Optional[Any] = None

        # In-memory run info for compatibility
        self.run_info = PilatesRunInfo(
            run_id=run_id,
            created_at=datetime.now().isoformat(),
        )
        self._save_run_info()

        logger.info(f"ConsistProvenanceTracker initialized for run ID: {self.run_id}")

    def _save_run_info(self):
        """Persist the run_info to JSON for backward compatibility."""
        import json
        import dataclasses

        if not self.output_path:
            return

        os.makedirs(os.path.dirname(self.run_info_path), exist_ok=True)
        try:
            with open(self.run_info_path, "w") as f:
                json.dump(dataclasses.asdict(self.run_info), f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Could not save run_info.json to {self.run_info_path}: {e}")

    @property
    def data_manager(self) -> "ConsistDataManager":
        if not hasattr(self, "_data_manager"):
            self._data_manager = ConsistDataManager(self._tracker)
        return self._data_manager

    @property
    def current_model_run_id(self) -> Optional[str]:
        return self._current_model_run_id

    def _normalize_model_name(self, model: str) -> str:
        return model.lower() if model else model

    def initialize_from_settings(self, settings: PilatesConfig):
        self.run_info.start_year = settings.run.start_year
        self.run_info.end_year = settings.run.end_year
        self.run_info.settings_hash = self._calculate_settings_hash(settings)
        self._save_run_info()

        models_used = []
        if settings.run.models.land_use: models_used.append(settings.run.models.land_use)
        if settings.run.models.vehicle_ownership: models_used.append(settings.run.models.vehicle_ownership)
        if settings.run.models.activity_demand: models_used.append(settings.run.models.activity_demand)
        if settings.run.models.travel: models_used.append(settings.run.models.travel)
        self.run_info.models_used = list(set(models_used))

    def _calculate_settings_hash(self, settings: PilatesConfig) -> str:
        settings_str = settings.model_dump_json()
        return hashlib.sha256(settings_str.encode("utf-8")).hexdigest()

    def start_model_run(
        self,
        model: str,
        year: int = None,
        iteration: int = None,
        description: str = None,
        inputs: RecordStore = None,
        state: Optional[ExecutionContext] = None,
        cache_mode: str = "reuse",
        **kwargs,
    ) -> str:
        """
        Start tracking a new model run.
        If a Consist run is already active (via scenario.step), attaches to it.
        """
        model = self._normalize_model_name(model)

        # 1. Check for Active Run (Attach Mode)
        if self._tracker.current_consist is not None:
            self._attached_mode = True
            current_id = self._tracker.current_consist.run.id
            logger.info(f"ConsistAdapter: Attaching to active run '{current_id}' for '{model}'")

            # Log inputs immediately since we are active
            if inputs:
                for rec in inputs.all_records():
                    if isinstance(rec, (FileRecord, H5FileRecord)):
                        self.record_input_file(model, rec.file_path, description=getattr(rec, 'description', None))

            self._current_model_run_id = current_id
            return current_id

        # 2. Start New Run (Legacy/Standalone Mode)
        self._attached_mode = False

        if inputs is None:
            inputs = RecordStore()

        model_run_id = (
            f"{model}_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        )

        config = {
            "model": model,
            "year": year,
            "iteration": iteration,
        }

        pilates_meta = _inject_workflow_context(state)

        consist_inputs = []
        for record in inputs.all_records():
            if isinstance(record, (FileRecord, H5FileRecord)):
                abs_path = self._resolve_record_path(record)
                if abs_path and os.path.exists(abs_path):
                    consist_inputs.append(abs_path)

        self._current_run = self._tracker.begin_run(
            run_id=model_run_id,
            model=model,
            config=config,
            inputs=consist_inputs,
            tags=None,
            description=description,
            cache_mode=cache_mode,
            year=year,
            iteration=iteration,
            **pilates_meta,
        )
        self._current_model_run_id = model_run_id

        # Create ModelRunInfo for compatibility
        run_record = ModelRunInfo(
            unique_id=model_run_id,
            model=model,
            year=year,
            iteration=iteration,
            description=description,
            created_at=datetime.now().isoformat(),
            status="running",
            input_record_hashes=inputs.all_unique_ids() if inputs else [],
        )
        self.run_info.model_runs[model_run_id] = run_record
        self._save_run_info()

        if self._tracker.is_cached:
            logger.info(f"⚡️ Consist Cache Hit for {model}. Hydrating workspace...")
            self._hydrate_outputs()
        else:
            logger.info(f"Started Consist run: {model_run_id}")
        return model_run_id

    def _hydrate_outputs(self):
        cached_run = self._tracker.current_consist.cached_run
        for artifact in self._tracker.current_consist.outputs:
            source_path = self._tracker.resolve_historical_path(artifact, cached_run)
            dest_path = Path(self._tracker.resolve_uri(artifact.uri))

            if not source_path.exists():
                continue
            if source_path == dest_path:
                continue

            if source_path.is_dir():
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(source_path, dest_path)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)

            artifact.abs_path = str(dest_path)

    def complete_model_run(
        self,
        run_hash: str,
        status: str = "completed",
        output_records: List[Union[FileRecord, RepoRecord]] = None,
        metadata: dict = None,
    ):
        """
        Mark a model run as complete.
        If in Attach Mode, only logs outputs; does NOT close the run.
        """
        if output_records is None:
            output_records = []

        # Log outputs to Consist
        for record in output_records:
            if isinstance(record, (FileRecord, H5FileRecord)):
                self._log_record_as_artifact(record, direction="output")

        # Update compatibility run info
        if run_hash in self.run_info.model_runs:
            self.run_info.model_runs[run_hash].completed_at = datetime.now().isoformat()
            self.run_info.model_runs[run_hash].status = status
            if metadata:
                self.run_info.model_runs[run_hash].metadata.update(metadata)

            for record in output_records:
                if hasattr(record, "unique_id") and record.unique_id:
                    if (
                        record.unique_id
                        not in self.run_info.model_runs[run_hash].output_record_hashes
                    ):
                        self.run_info.model_runs[run_hash].output_record_hashes.append(
                            record.unique_id
                        )
                if hasattr(record, "producing_run_id"):
                    record.producing_run_id = run_hash

        self._save_run_info()

        # Handle Attach Mode
        if self._attached_mode:
            logger.debug(f"ConsistAdapter: Detaching from run '{run_hash}' (lifecycle managed externally)")
            self._attached_mode = False
            return

        # End Consist run (Legacy Mode)
        try:
            if status == "failed":
                self._tracker.end_run(status="failed")
            else:
                self._tracker.end_run(status="completed")
        except Exception as e:
            logger.warning(f"Error ending Consist run: {e}")
        self._current_run = None
        self._current_model_run_id = None
        logger.info(f"Completed Consist run: {run_hash} with status: {status}")

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
        ctx = context if context is not None else state
        model = self._normalize_model_name(model)
        abs_path = os.path.abspath(file_path)

        if not os.path.exists(abs_path):
            if skip_missing: return None
            else: logger.warning(f"Input file does not exist: {file_path}")

        key = short_name or Path(file_path).stem
        input_meta = {}
        if source_run_id: input_meta["source_run_id"] = source_run_id
        if source_file_paths: input_meta["source_file_paths"] = source_file_paths
        if model_run_id: input_meta["model_run_id"] = model_run_id

        artifact = self._tracker.log_input(
            path=abs_path,
            key=key,
            model=model,
            description=description,
            **input_meta,
            **_inject_workflow_context(ctx),
        )

        file_record = self._artifact_to_file_record(artifact, model, ctx)
        file_record.models = [model]
        if description: file_record.description = description

        self.run_info.file_records[file_record.unique_id] = file_record
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
        ctx = context if context is not None else state
        model = self._normalize_model_name(model)
        abs_path = os.path.abspath(file_path)

        if not os.path.exists(abs_path):
            if skip_missing: return None
            else: logger.warning(f"Output file does not exist: {file_path}")

        key = short_name or Path(file_path).stem
        meta = metadata or {}
        if source_file_paths: meta["source_file_paths"] = source_file_paths
        if model_run_id: meta["model_run_id"] = model_run_id

        artifact = self._tracker.log_output(
            path=abs_path,
            key=key,
            model=model,
            description=description,
            year=year,
            **meta,
            **_inject_workflow_context(ctx),
        )

        file_record = self._artifact_to_file_record(artifact, model, ctx)
        file_record.models = [model]
        if description: file_record.description = description
        if year: file_record.year = year

        self.run_info.file_records[file_record.unique_id] = file_record
        self._save_run_info()
        return file_record

    def record_output_file_with_inputs(
        self, model: str, file_path: str, input_records: List[Optional[FileRecord]], **kwargs
    ) -> Optional[FileRecord]:
        source_file_paths = []
        if input_records:
            for rec in input_records:
                if rec is not None and hasattr(rec, "file_path"):
                    source_file_paths.append(rec.file_path)
        kwargs["source_file_paths"] = source_file_paths
        return self.record_output_file(model=model, file_path=file_path, **kwargs)

    def record_input_record(self, record: Any, model_run_id: str = None):
        if record is None: return
        if model_run_id is None: model_run_id = self.current_model_run_id

        if model_run_id and model_run_id in self.run_info.model_runs:
            if hasattr(record, "unique_id"):
                if record.unique_id not in self.run_info.model_runs[model_run_id].input_record_hashes:
                    self.run_info.model_runs[model_run_id].input_record_hashes.append(record.unique_id)
            if hasattr(record, "unique_id") and record.unique_id not in self.run_info.file_records:
                self.run_info.file_records[record.unique_id] = record

        self._save_run_info()

        if hasattr(record, "file_path") and record.file_path:
            abs_path = self._resolve_record_path(record)
            if abs_path and os.path.exists(abs_path):
                self._tracker.log_input(
                    path=abs_path,
                    key=getattr(record, "short_name", None),
                    description=getattr(record, "description", None),
                    pilates_unique_id=getattr(record, "unique_id", None)
                )

    def rename_directory(self, old_directory_name: str, new_directory_name: str):
        updated_count = 0
        for record in self.run_info.file_records.values():
            if record.file_path and record.file_path.startswith(old_directory_name):
                new_path = record.file_path.replace(old_directory_name, new_directory_name, 1)
                record.file_path = new_path
                updated_count += 1
        if updated_count > 0:
            logger.info(f"Renamed {updated_count} records in run_info from {old_directory_name} to {new_directory_name}")
            self._save_run_info()

    def move_file(
            self, record: Any, source_path: str, destination_path: str, model: str, state: Optional[ExecutionContext] = None,
    ) -> Optional[FileRecord]:
        model = self._normalize_model_name(model)
        producing_run_id = getattr(record, "producing_run_id", None)

        self.record_input_file(
            model=model,
            file_path=source_path,
            source_run_id=producing_run_id,
            description=getattr(record, "description", None),
            state=state,
            skip_missing=False
        )

        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        try:
            if source_path != destination_path:
                shutil.move(source_path, destination_path)
        except OSError as e:
            logger.error(f"Failed to move file from {source_path} to {destination_path}: {e}")
            raise

        short_name = getattr(record, "short_name", Path(destination_path).stem)
        if short_name:
            short_name = re.sub(r"_asim_out_temp$", "", short_name)
            short_name = re.sub(r"_temp$", "", short_name)

        output_record = self.record_output_file(
            model=model,
            file_path=destination_path,
            description=getattr(record, "description", None),
            short_name=short_name,
            state=state,
            source_file_paths=[source_path]
        )
        return output_record

    def record_h5_input_container(self, model: str, file_path: str, **kwargs) -> Optional[H5FileRecord]:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            logger.warning(f"H5 input container does not exist: {file_path}")
            return None

        key = kwargs.get("short_name") or Path(file_path).stem
        artifact, table_artifacts = self._tracker.log_h5_container(
            path=abs_path, key=key, direction="input", discover_tables=True, model=model
        )
        self._h5_containers[abs_path] = artifact
        h5_record = self._artifact_to_h5_file_record(artifact, model)
        h5_record.table_record_ids = [str(t.id) for t in table_artifacts]
        self.run_info.file_records[h5_record.unique_id] = h5_record
        self._save_run_info()
        return h5_record

    def record_h5_output_container(
        self, model: str, file_path: str, table_records: List[H5TableRecord] = None, **kwargs,
    ) -> Optional[H5FileRecord]:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            logger.warning(f"H5 output container does not exist: {file_path}")
            return None

        key = kwargs.get("short_name") or Path(file_path).stem
        artifact, table_artifacts = self._tracker.log_h5_container(
            path=abs_path, key=key, direction="output", discover_tables=True, model=model,
        )
        self._h5_containers[abs_path] = artifact
        h5_record = self._artifact_to_h5_file_record(artifact, model)
        h5_record.table_record_ids = [str(t.id) for t in table_artifacts]
        self.run_info.file_records[h5_record.unique_id] = h5_record
        self._save_run_info()
        return h5_record

    def record_repo_input(
        self, model: str, repo_path: str, short_name: str = None, description: str = None, git_hash: str = None,
    ) -> Optional[RepoRecord]:
        model = self._normalize_model_name(model)
        abs_path = os.path.abspath(repo_path)
        if not os.path.exists(abs_path):
            logger.warning(f"Repository does not exist: {repo_path}")
            return None

        if git_hash is None: git_hash = self.get_git_hash(abs_path)

        artifact = self._tracker.log_artifact(
            path=abs_path,
            key=short_name or Path(repo_path).name,
            direction="input",
            driver="git",
            git_hash=git_hash,
            description=description,
            model=model,
        )

        relative_path = self.get_path_relative_to_workspace_root(abs_path)
        repo_record = RepoRecord(
            unique_id=git_hash or str(artifact.id),
            repo_path=relative_path,
            accessed_at=datetime.now().isoformat(),
            description=description,
            short_name=short_name,
        )
        self.run_info.repo_records[repo_record.unique_id] = repo_record
        self._save_run_info()
        return repo_record

    def get_path_relative_to_workspace_root(self, file_path: str) -> str:
        abs_path = os.path.abspath(file_path)
        if self.workspace_root:
            try: return os.path.relpath(abs_path, self.workspace_root)
            except ValueError: return abs_path
        return abs_path

    def get_run_info(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self.run_info)

    def current_model_run(self) -> Optional[ModelRunInfo]:
        if self._current_model_run_id and self._current_model_run_id in self.run_info.model_runs:
            return self.run_info.model_runs[self._current_model_run_id]
        return None

    def get_latest_completed_model_run(self, model_name: str, year: int, iteration: int) -> Optional[ModelRunInfo]:
        model_name = self._normalize_model_name(model_name)
        latest_run = None
        latest_time = None
        for run in self.run_info.model_runs.values():
            if run.model == model_name and run.year == year and run.iteration == iteration and run.status == "completed":
                completed_at = datetime.fromisoformat(run.completed_at)
                if latest_time is None or completed_at > latest_time:
                    latest_time = completed_at
                    latest_run = run
        return latest_run

    def _resolve_record_path(self, record: Union[FileRecord, H5FileRecord]) -> Optional[str]:
        if not record.file_path: return None
        if os.path.isabs(record.file_path): return record.file_path
        if self.workspace_root: return os.path.join(self.workspace_root, record.file_path)
        return os.path.abspath(record.file_path)

    def _log_record_as_artifact(self, record: Union[FileRecord, H5FileRecord], direction: str) -> Optional[Artifact]:
        abs_path = self._resolve_record_path(record)
        if not abs_path or not os.path.exists(abs_path): return None

        driver = None
        meta = {}
        if isinstance(record, H5FileRecord):
            driver = "h5"
            meta["is_container"] = True
        elif isinstance(record, H5TableRecord):
            driver = "h5_table"
            meta["parent_id"] = record.h5_file_unique_id
            meta["table_path"] = record.table_name

        return self._tracker.log_artifact(
            path=abs_path,
            key=record.short_name or Path(abs_path).stem,
            direction=direction,
            driver=driver,
            **meta,
        )

    def _artifact_to_file_record(self, artifact: Artifact, model: str, ctx: Optional[ExecutionContext] = None) -> FileRecord:
        abs_path = artifact.abs_path or self._tracker.resolve_uri(artifact.uri)
        relative_path = self.get_path_relative_to_workspace_root(abs_path)
        meta = dict(artifact.meta) if artifact.meta else {}
        if artifact.hash: meta["file_hash"] = artifact.hash

        return FileRecord(
            unique_id=str(artifact.id),
            file_path=relative_path,
            created_at=artifact.created_at_iso or datetime.now().isoformat(),
            short_name=artifact.key,
            year=ctx.current_year if ctx else None,
            models=[model],
            metadata=meta,
        )

    def _artifact_to_h5_file_record(self, artifact: Artifact, model: str) -> H5FileRecord:
        abs_path = artifact.abs_path or self._tracker.resolve_uri(artifact.uri)
        relative_path = self.get_path_relative_to_workspace_root(abs_path)
        return H5FileRecord(
            unique_id=str(artifact.id),
            file_path=relative_path,
            created_at=artifact.created_at_iso or datetime.now().isoformat(),
            short_name=artifact.key,
            models=[model],
            metadata=dict(artifact.meta) if artifact.meta else {},
            table_record_ids=[],
        )

    def get_git_hash(self, repo_path: str) -> Optional[str]:
        import subprocess
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=repo_path, timeout=5)
            return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Could not get git hash for {repo_path}: {e}")
            return None

    # --- Schema Mapping Conversions (Phase 5.3.1) ---
    def recordToArtifact(self, record) -> Optional[Artifact]:
        if isinstance(record, H5FileRecord):
            return self._h5_file_record_to_artifact(record)
        elif isinstance(record, H5TableRecord):
            return self._h5_table_record_to_artifact(record)
        elif isinstance(record, FileRecord):
            return self._file_record_to_artifact(record)
        else:
            logger.warning(f"Unrecognized record type {type(record)}")
            return None

    @staticmethod
    def _file_record_to_artifact(record: FileRecord) -> Artifact:
        import uuid as uuid_mod
        path = record.file_path or ""
        if path.endswith(".parquet"): driver = "parquet"
        elif path.endswith(".csv"): driver = "csv"
        elif path.endswith((".h5", ".hdf5")): driver = "h5"
        elif path.endswith(".zarr"): driver = "zarr"
        else: driver = "auto"

        meta = dict(record.metadata) if record.metadata else {}
        meta["models"] = record.models
        meta["year"] = record.year
        meta["iteration"] = record.iteration
        meta["description"] = record.description
        meta["source_file_paths"] = record.source_file_paths
        meta["producing_run_id"] = record.producing_run_id
        meta["pilates_unique_id"] = record.unique_id

        try: artifact_id = uuid_mod.UUID(record.unique_id)
        except (ValueError, TypeError): artifact_id = uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, record.unique_id or "")

        return Artifact(
            id=artifact_id,
            key=record.short_name or Path(path).stem,
            uri=path,
            driver=driver,
            hash=record.metadata.get("file_hash") if record.metadata else None,
            meta={k: v for k, v in meta.items() if v is not None},
        )

    @staticmethod
    def _h5_file_record_to_artifact(record: H5FileRecord) -> Artifact:
        import uuid as uuid_mod
        meta = dict(record.metadata) if record.metadata else {}
        meta["is_container"] = True
        meta["table_ids"] = record.table_record_ids
        meta["models"] = record.models
        meta["year"] = record.year
        meta["iteration"] = record.iteration
        meta["pilates_unique_id"] = record.unique_id

        try: artifact_id = uuid_mod.UUID(record.unique_id)
        except (ValueError, TypeError): artifact_id = uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, record.unique_id or "")

        return Artifact(
            id=artifact_id,
            key=record.short_name or Path(record.file_path).stem,
            uri=record.file_path,
            driver="h5",
            meta={k: v for k, v in meta.items() if v is not None},
        )

    @staticmethod
    def _h5_table_record_to_artifact(record: H5TableRecord, parent_artifact_id: str = None) -> Artifact:
        import uuid as uuid_mod
        meta = dict(record.metadata) if record.metadata else {}
        meta["table_path"] = record.table_name
        meta["parent_id"] = record.h5_file_unique_id
        if parent_artifact_id: meta["parent_artifact_id"] = parent_artifact_id
        meta["models"] = record.models
        meta["year"] = record.year
        meta["pilates_unique_id"] = record.unique_id

        try: artifact_id = uuid_mod.UUID(record.unique_id)
        except (ValueError, TypeError): artifact_id = uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, record.unique_id or "")

        uri = f"{record.file_path}#{record.table_name}"
        return Artifact(
            id=artifact_id,
            key=record.short_name or record.table_name,
            uri=uri,
            driver="h5_table",
            meta={k: v for k, v in meta.items() if v is not None},
        )

    @staticmethod
    def _model_run_to_consist_run(model_run: ModelRunInfo) -> Run:
        started_at = datetime.fromisoformat(model_run.created_at) if model_run.created_at else datetime.now()
        ended_at = datetime.fromisoformat(model_run.completed_at) if model_run.completed_at else None

        meta = dict(model_run.metadata) if model_run.metadata else {}
        meta["input_record_hashes"] = model_run.input_record_hashes
        meta["output_record_hashes"] = model_run.output_record_hashes
        meta["pilates_unique_id"] = model_run.unique_id

        return Run(
            id=model_run.unique_id,
            model_name=model_run.model,
            year=model_run.year,
            iteration=model_run.iteration,
            description=model_run.description,
            status=model_run.status,
            started_at=started_at,
            ended_at=ended_at,
            meta={k: v for k, v in meta.items() if v is not None},
        )

    @staticmethod
    def _pilates_run_to_consist_run(run_info: PilatesRunInfo) -> Run:
        import socket
        started_at = datetime.fromisoformat(run_info.created_at) if run_info.created_at else datetime.now()

        meta = {
            "start_year": run_info.start_year,
            "end_year": run_info.end_year,
            "hostname": run_info.hostname or socket.gethostname(),
            "is_pilates_run": True,
        }
        if run_info.config_snapshot: meta["config_snapshot"] = run_info.config_snapshot

        return Run(
            id=run_info.run_id,
            model_name="pilates",
            config_hash=run_info.settings_hash,
            git_hash=run_info.code_version,
            tags=run_info.models_used or [],
            started_at=started_at,
            meta={k: v for k, v in meta.items() if v is not None},
        )

    @staticmethod
    def _consist_run_to_model_run_info(run: Run) -> ModelRunInfo:
        return ModelRunInfo(
            unique_id=run.id,
            model=run.model_name,
            year=run.year,
            iteration=run.iteration,
            description=run.description,
            status=run.status,
            created_at=run.started_at.isoformat() if run.started_at else None,
            completed_at=run.ended_at.isoformat() if run.ended_at else None,
            input_record_hashes=(run.meta.get("input_record_hashes", []) if run.meta else []),
            output_record_hashes=(run.meta.get("output_record_hashes", []) if run.meta else []),
            metadata=dict(run.meta) if run.meta else {},
        )

    def register_openlineage_hooks(self, openlineage_tracker):
        @self._tracker.on_run_start
        def emit_start(run: Run): logger.debug(f"Consist run started: {run.id}")
        @self._tracker.on_run_complete
        def emit_complete(run: Run, outputs: List[Artifact]): logger.debug(f"Consist run completed: {run.id} with {len(outputs)} outputs")
        @self._tracker.on_run_failed
        def emit_failed(run: Run, error: Exception): logger.debug(f"Consist run failed: {run.id} with error: {error}")

    def add_openlineage_event(self, *args, **kwargs): pass
    def save(self): pass


class ConsistDataManager:
    def __init__(self, tracker: Tracker, db_path: str = None):
        self.tracker = tracker
        self.db_path = db_path or tracker.db_path
        self._engine = tracker.engine

    def store_dataframe(
        self, table_name: str, df, run_id: str, year: int, iteration: int = None, model: str = None, **kwargs,
    ) -> bool:
        try:
            artifact = self.tracker.log_artifact(
                path=f"memory://{table_name}",
                key=table_name,
                direction="output",
                driver="dataframe",
                year=year,
                iteration=iteration,
                model=model or "pilates",
                run_id=run_id,
            )
            self.tracker.ingest(artifact, data=df)
            logger.info(f"Ingested {len(df)} rows to {table_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to ingest {table_name}: {e}")
            return False

    def store_urbansim_raw_data(self, table_name: str, df, run_id: str, year: int, iteration: int = None, **kwargs) -> bool:
        return self.store_dataframe(table_name=f"urbansim_{table_name}_raw", df=df, run_id=run_id, year=year, iteration=iteration, model="urbansim", **kwargs)

    def store_activitysim_data(self, table_name: str, df, run_id: str, year: int, iteration: int = None, **kwargs) -> bool:
        return self.store_dataframe(table_name=f"activitysim_{table_name}", df=df, run_id=run_id, year=year, iteration=iteration, model="activitysim", **kwargs)

    def query(self, sql: str):
        import pandas as pd
        return pd.read_sql(sql, self._engine)