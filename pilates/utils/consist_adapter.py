"""
pilates/utils/consist_adapter.py

Consist Adapter for PILATES Provenance Tracking.
Refactored Phase 6.2: Lightweight wrapper around an active Consist Tracker context.
"""

import logging
import os
import shutil
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from consist import Tracker, Artifact

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
    Lightweight Adapter that bridges legacy PILATES instrumentation
    to an active Consist Tracker context.

    This adapter assumes that the Run Lifecycle is managed externally
    (via 'tracker.scenario' and 'scenario.step' in run.py).
    """

    def __init__(
        self,
        run_id: str,
        output_path: str,
        tracker: Optional[Tracker] = None,
        folder_name: str = None,
        # Legacy args ignored/optional now
        db_path: str = None,
        mounts: Dict[str, str] = None,
    ):
        if not tracker:
            raise ValueError("ConsistProvenanceTracker requires a valid 'tracker' instance.")

        self.run_id = run_id
        self.output_path = output_path
        self.folder_name = folder_name
        self._tracker = tracker

        # Track H5 containers for legacy parent-child linking if needed locally
        self._h5_containers: Dict[str, Artifact] = {}

        # Legacy In-memory storage for getters (get_latest_model_run, etc.)
        # Populated passively as we log to Consist.
        self.run_info = PilatesRunInfo(
            run_id=run_id,
            created_at=datetime.now().isoformat(),
        )

        logger.info(f"ConsistProvenanceTracker initialized (Wrapper Mode).")

    @property
    def data_manager(self) -> "ConsistDataManager":
        if not hasattr(self, "_data_manager"):
            self._data_manager = ConsistDataManager(self._tracker)
        return self._data_manager

    @property
    def current_model_run_id(self) -> Optional[str]:
        """Returns the ID of the currently active Consist run."""
        if self._tracker.current_consist:
            return self._tracker.current_consist.run.id
        return None

    def _normalize_model_name(self, model: str) -> str:
        return model.lower() if model else model

    def initialize_from_settings(self, settings: PilatesConfig):
        """Populate legacy run_info header data."""
        self.run_info.start_year = settings.run.start_year
        self.run_info.end_year = settings.run.end_year

        models_used = []
        if settings.run.models.land_use: models_used.append(settings.run.models.land_use)
        if settings.run.models.vehicle_ownership: models_used.append(settings.run.models.vehicle_ownership)
        if settings.run.models.activity_demand: models_used.append(settings.run.models.activity_demand)
        if settings.run.models.travel: models_used.append(settings.run.models.travel)
        self.run_info.models_used = list(set(models_used))

    def start_model_run(
        self,
        model: str,
        year: int = None,
        iteration: int = None,
        description: str = None,
        inputs: RecordStore = None,
        state: Optional[ExecutionContext] = None,
        cache_mode: str = "reuse", # Ignored here, managed by scenario.step
        **kwargs,
    ) -> str:
        """
        Adapts legacy start call to the active Consist context.

        Verifies that a Consist run is active (started by run.py scenario.step).
        Logs inputs and metadata to that active run.
        """
        if not self._tracker.current_consist:
            raise RuntimeError(
                f"ConsistAdapter Error: No active Consist run found when starting '{model}'. "
                "Ensure this code is executed within a 'with scenario.step(...):' block."
            )

        active_run = self._tracker.current_consist.run
        logger.debug(f"ConsistAdapter: '{model}' running within active step '{active_run.id}'")

        # 1. Log Metadata to the active run
        meta_updates = {
            "pilates_model_alias": model,
            "description": description,
            **_inject_workflow_context(state)
        }
        # Only set if not already set by step() context
        if year is not None: meta_updates["year"] = year
        if iteration is not None: meta_updates["iteration"] = iteration

        self._tracker.log_meta(**meta_updates)

        # 2. Log Inputs
        if inputs:
            for rec in inputs.all_records():
                # We skip re-logging if it's already an artifact in the current run,
                # but log_artifact is idempotent anyway.
                if hasattr(rec, 'file_path') and rec.file_path:
                    # Resolve path: FileRecord usually stores relative paths
                    abs_path = self._resolve_record_path(rec)
                    if abs_path and os.path.exists(abs_path):
                        self._tracker.log_input(
                            path=abs_path,
                            key=rec.short_name,
                            description=rec.description,
                            pilates_unique_id=rec.unique_id
                        )

        # 3. Update Legacy InMemory State for getters
        run_record = ModelRunInfo(
            unique_id=active_run.id,
            model=model,
            year=year,
            iteration=iteration,
            description=description,
            created_at=datetime.now().isoformat(),
            status="running",
            input_record_hashes=inputs.all_unique_ids() if inputs else [],
        )
        self.run_info.model_runs[active_run.id] = run_record

        return active_run.id

    def complete_model_run(
        self,
        run_hash: str,
        status: str = "completed",
        output_records: List[Union[FileRecord, RepoRecord]] = None,
        metadata: dict = None,
    ):
        """
        Adapts legacy complete call.
        Lifecycle is managed by the context manager in run.py, so this is primarily
        for logging legacy completion status to memory and ensuring outputs are linked.
        """
        # Ensure all passed outputs are definitely logged, but avoid double-logging
        # for records already captured in the active Consist run.
        existing_output_uris = set()
        if getattr(self._tracker, "current_consist", None):
            existing_output_uris = {
                a.uri for a in (self._tracker.current_consist.outputs or [])
            }

        if output_records:
            for record in output_records:
                if not isinstance(record, (FileRecord, H5FileRecord)):
                    continue
                abs_path = self._resolve_record_path(record)
                if abs_path:
                    uri = self._tracker.fs.virtualize_path(abs_path)
                    if uri in existing_output_uris:
                        continue
                self.log_record_as_artifact(record, direction="output")

        # Update Legacy InMemory State
        if run_hash in self.run_info.model_runs:
            self.run_info.model_runs[run_hash].completed_at = datetime.now().isoformat()
            self.run_info.model_runs[run_hash].status = status
            if metadata:
                self.run_info.model_runs[run_hash].metadata.update(metadata)

        logger.debug(f"ConsistAdapter: Legacy completion signal for '{run_hash}' (status: {status}). Lifecycle delegated to context.")

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
        Wrapper: Logs input to Consist, returns legacy FileRecord.
        """
        ctx = context if context is not None else state
        abs_path = os.path.abspath(file_path)

        if not os.path.exists(abs_path):
            if skip_missing: return None
            else: logger.warning(f"Input file does not exist: {file_path}")

        key = short_name or Path(file_path).stem

        # 1. Log to Consist
        artifact = self._tracker.log_input(
            path=abs_path,
            key=key,
            description=description,
            source_run_id=source_run_id, # Meta field
            **_inject_workflow_context(ctx),
        )

        # 2. Return Legacy Object
        file_record = self._artifact_to_file_record(artifact, model, ctx)

        # Update legacy memory
        self.run_info.file_records[file_record.unique_id] = file_record
        return file_record

    # ---------------------------------------------------------------------
    # Init-artifact helpers (Consist-only mode)
    # ---------------------------------------------------------------------

    def get_init_output_artifacts(
        self,
        keys: List[str],
        init_tag: str = "init",
        scenario_id: Optional[str] = None,
    ) -> Dict[str, Artifact]:
        """
        Fetch selected output artifacts from the initialization step in the current scenario.

        This is a convenience for preprocessors/runners that want to treat specific
        initialization outputs as inputs without re-tagging the entire init store.

        Args:
            keys: Output artifact keys to retrieve from the init run.
            init_tag: Tag used to identify the init step (default: "init").
            scenario_id: Override scenario ID; by default uses active run parent.

        Returns:
            Dict mapping key -> Artifact for keys found. Missing keys are omitted.

        Raises:
            RuntimeError if no active Consist run or no init run found.
        """
        if not self._tracker.current_consist:
            raise RuntimeError(
                "Cannot fetch init artifacts outside an active Consist step."
            )

        if scenario_id is None:
            scenario_id = self._tracker.current_consist.run.parent_run_id

        init_runs = self._tracker.find_runs(parent_id=scenario_id, tags=[init_tag])
        if not init_runs:
            raise RuntimeError(
                f"No initialization run found for scenario '{scenario_id}' with tag '{init_tag}'."
            )

        init_run_id = init_runs[0].id
        init_artifacts = self._tracker.get_artifacts_for_run(init_run_id)

        found: Dict[str, Artifact] = {}
        for key in keys:
            art = init_artifacts.outputs.get(key)
            if art:
                # Ensure abs_path is populated for convenience.
                try:
                    art.abs_path = self._tracker.resolve_uri(art.uri)
                except Exception:
                    pass
                found[key] = art

        return found

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
        Wrapper: Logs output to Consist, returns legacy FileRecord.
        """
        ctx = context if context is not None else state
        abs_path = os.path.abspath(file_path)

        if not os.path.exists(abs_path):
            if skip_missing: return None
            else: logger.warning(f"Output file does not exist: {file_path}")

        key = short_name or Path(file_path).stem
        meta = metadata or {}
        if source_file_paths: meta["source_file_paths"] = source_file_paths

        # 1. Log to Consist
        artifact = self._tracker.log_output(
            path=abs_path,
            key=key,
            description=description,
            year=year,
            **meta,
            **_inject_workflow_context(ctx),
        )

        # 2. Return Legacy Object
        file_record = self._artifact_to_file_record(artifact, model, ctx)

        # Update legacy memory
        self.run_info.file_records[file_record.unique_id] = file_record
        return file_record

    def record_output_file_with_inputs(
        self, model: str, file_path: str, input_records: List[Optional[FileRecord]], **kwargs
    ) -> Optional[FileRecord]:
        """Helper to chain source paths."""
        source_file_paths = []
        if input_records:
            for rec in input_records:
                if rec is not None and hasattr(rec, "file_path"):
                    source_file_paths.append(rec.file_path)
        kwargs["source_file_paths"] = source_file_paths
        return self.record_output_file(model=model, file_path=file_path, **kwargs)

    def record_input_record(self, record: Any, model_run_id: str = None):
        """Used by some preprocessors to register existing records as inputs."""
        if record is None: return

        # Just ensure it's logged to Consist
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
        """Legacy helper for when PILATES moves folders around."""
        # Update legacy in-memory records so getters don't break
        updated_count = 0
        for record in self.run_info.file_records.values():
            if record.file_path and record.file_path.startswith(old_directory_name):
                new_path = record.file_path.replace(old_directory_name, new_directory_name, 1)
                record.file_path = new_path
                updated_count += 1
        if updated_count > 0:
            logger.debug(f"Renamed {updated_count} legacy records from {old_directory_name} to {new_directory_name}")

    def move_file(
            self, record: Any, source_path: str, destination_path: str, model: str, state: Optional[ExecutionContext] = None,
    ) -> Optional[FileRecord]:
        """
        Moves a file and records the lineage (Input -> Output).
        """
        model = self._normalize_model_name(model)

        # Avoid logging the raw ActivitySim parquet pipeline outputs as separate inputs.
        # These files are commonly named `final.parquet`, which collapses many distinct
        # artifacts under the same key ("final") and creates very noisy Consist graphs.
        # We still record lineage via `source_file_paths` on the output artifact.
        should_log_source_as_input = True
        try:
            short_name = getattr(record, "short_name", None)
            if (
                short_name
                and str(short_name).endswith("_asim_out_temp")
                and Path(source_path).name == "final.parquet"
            ):
                should_log_source_as_input = False
        except Exception:
            should_log_source_as_input = True

        if should_log_source_as_input:
            self.record_input_file(
                model=model,
                file_path=source_path,
                description=getattr(record, "description", None),
                state=state,
                skip_missing=False,
            )

        # Perform Move
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

        # Log Destination as Output
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
        if not os.path.exists(abs_path): return None

        key = kwargs.get("short_name") or Path(file_path).stem

        # Log to Consist
        artifact, table_artifacts = self._tracker.log_h5_container(
            path=abs_path, key=key, direction="input", discover_tables=True, model=model
        )

        # Convert to Legacy
        self._h5_containers[abs_path] = artifact
        h5_record = self._artifact_to_h5_file_record(artifact, model)
        h5_record.table_record_ids = [str(t.id) for t in table_artifacts]

        self.run_info.file_records[h5_record.unique_id] = h5_record
        return h5_record

    def record_h5_output_container(
        self, model: str, file_path: str, table_records: List[H5TableRecord] = None, **kwargs,
    ) -> Optional[H5FileRecord]:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path): return None

        key = kwargs.get("short_name") or Path(file_path).stem

        # Log to Consist
        artifact, table_artifacts = self._tracker.log_h5_container(
            path=abs_path, key=key, direction="output", discover_tables=True, model=model,
        )

        # Convert to Legacy
        self._h5_containers[abs_path] = artifact
        h5_record = self._artifact_to_h5_file_record(artifact, model)
        h5_record.table_record_ids = [str(t.id) for t in table_artifacts]

        self.run_info.file_records[h5_record.unique_id] = h5_record
        return h5_record

    def record_repo_input(
            self, model: str, repo_path: str, short_name: str = None, description: str = None, git_hash: str = None,
    ) -> Optional[RepoRecord]:
        # Inputs are true Git repos
        return self._record_repo(
            model, repo_path, "input",
            driver="git",
            short_name=short_name, description=description, git_hash=git_hash
        )

    def record_repo_output(
            self, model: str, repo_path: str, short_name: str = None, description: str = None, git_hash: str = None,
    ) -> Optional[RepoRecord]:
        # Outputs are just directories (the .git folder was removed during copy)
        # We pass the git_hash as metadata so we know what version it CAME from,
        # but the driver should be "directory" (or None for auto).
        return self._record_repo(
            model, repo_path, "output",
            driver="directory",
            short_name=short_name, description=description, git_hash=git_hash
        )

    def _record_repo(
            self, model: str, repo_path: str, direction: str, driver: str = None,
            short_name: str = None, description: str = None, git_hash: str = None,
    ) -> Optional[RepoRecord]:
        model = self._normalize_model_name(model)
        abs_path = os.path.abspath(repo_path)
        if not os.path.exists(abs_path): return None

        # Calculate hash if missing (only works if it's actually a git repo)
        if git_hash is None and driver == "git":
            git_hash = self.get_git_hash(abs_path)

        # Log to Consist
        # Note: If driver="directory", Consist will compute a content hash of the files.
        # This is actually BETTER for the output, because it verifies the copy succeeded.
        artifact = self._tracker.log_artifact(
            path=abs_path,
            key=short_name or Path(repo_path).name,
            direction=direction,
            driver=driver,
            # We store the git_hash in meta for the output, so we don't lose the provenance link
            meta={"git_hash": git_hash} if git_hash else {},
            description=description,
            model=model,
        )

        # Convert to Legacy
        relative_path = self.get_path_relative_to_workspace_root(abs_path)
        repo_record = RepoRecord(
            # For inputs, use git_hash. For outputs, use the artifact ID or content hash.
            unique_id=git_hash or str(artifact.id),
            repo_path=relative_path,
            accessed_at=datetime.now().isoformat(),
            description=description,
            short_name=short_name,
        )
        self.run_info.repo_records[repo_record.unique_id] = repo_record
        return repo_record


    def get_path_relative_to_workspace_root(self, file_path: str) -> str:
        # Use tracker run_dir as workspace root
        root = str(self._tracker.run_dir)
        abs_path = os.path.abspath(file_path)
        try: return os.path.relpath(abs_path, root)
        except ValueError: return abs_path

    # --- Getters for Legacy Compatibility ---

    def get_run_info(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self.run_info)

    def current_model_run(self) -> Optional[ModelRunInfo]:
        """Return the ModelRunInfo corresponding to the active Consist run."""
        cid = self.current_model_run_id
        if cid and cid in self.run_info.model_runs:
            return self.run_info.model_runs[cid]
        return None

    def get_latest_completed_model_run(self, model_name: str, year: int, iteration: int) -> Optional[ModelRunInfo]:
        # Fallback to internal memory cache
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

    # --- Internals ---

    def _resolve_record_path(self, record: Union[FileRecord, H5FileRecord]) -> Optional[str]:
        if not record.file_path: return None
        if os.path.isabs(record.file_path): return record.file_path
        # Use tracker logic to resolve
        return self._tracker.fs.resolve_uri(record.file_path)

    def log_record_as_artifact(self, record: Union[FileRecord, H5FileRecord], direction: str) -> Optional[Artifact]:
        """Helper for batch logging records that might not have been logged yet."""
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

    def log_record_store(self, record_store: RecordStore, direction: str) -> None:
        """
        Batch log all records in a RecordStore as Consist artifacts.

        Args:
            record_store: RecordStore containing FileRecords to log
            direction: "input" or "output"
        """
        for record in record_store.all_records():
            self.log_record_as_artifact(record, direction)

    def _artifact_to_file_record(self, artifact: Artifact, model: str, ctx: Optional[ExecutionContext] = None) -> FileRecord:
        """Constructs a FileRecord from a Consist Artifact to satisfy legacy APIs."""
        abs_path = artifact.abs_path or self._tracker.resolve_uri(artifact.uri)
        relative_path = self.get_path_relative_to_workspace_root(abs_path)
        meta = dict(artifact.meta) if artifact.meta else {}
        if artifact.hash: meta["file_hash"] = artifact.hash

        # FIX: Explicitly extract fields needed by FileRecord constructor from metadata
        description = meta.get("description")
        source_file_paths = meta.get("source_file_paths", [])

        # If year not in context, try to find it in meta
        year = ctx.current_year if ctx else meta.get("year")

        return FileRecord(
            unique_id=str(artifact.id),
            file_path=relative_path,
            created_at=artifact.created_at_iso or datetime.now().isoformat(),
            short_name=artifact.key,
            year=year,
            models=[model],
            # Explicitly mapping fields that were missing in previous implementation
            description=description,
            producing_run_id=artifact.run_id,
            source_file_paths=source_file_paths,
            metadata=meta,
        )

    def _artifact_to_h5_file_record(self, artifact: Artifact, model: str) -> H5FileRecord:
        abs_path = artifact.abs_path or self._tracker.resolve_uri(artifact.uri)
        relative_path = self.get_path_relative_to_workspace_root(abs_path)
        meta = dict(artifact.meta) if artifact.meta else {}

        return H5FileRecord(
            unique_id=str(artifact.id),
            file_path=relative_path,
            created_at=artifact.created_at_iso or datetime.now().isoformat(),
            short_name=artifact.key,
            models=[model],
            metadata=meta,
            table_record_ids=[],
            # Map legacy fields from meta
            description=meta.get("description"),
            producing_run_id=artifact.run_id,
            source_file_paths=meta.get("source_file_paths", []),
            year=meta.get("year")
        )

    def get_git_hash(self, repo_path: str) -> Optional[str]:
        # Delegate to Consist IdentityManager logic if possible, or keep this shim
        try:
            return self._tracker.identity.get_code_version()
        except:
            return "unknown"

    # --- Hooks / Misc ---
    def register_openlineage_hooks(self, openlineage_tracker): pass
    def add_openlineage_event(self, *args, **kwargs): pass
    def save(self): pass
    # recordToArtifact & converters removed for brevity as they are for post-run analysis tools


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
