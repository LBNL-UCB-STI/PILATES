from __future__ import annotations

import functools
import logging
from typing import Optional, TYPE_CHECKING

from pilates.utils.consist_adapter import ConsistProvenanceTracker
from pilates.utils.duckdb_manager import DuckDBManager

if TYPE_CHECKING:
    from workflow_state import WorkflowState
    from pilates.utils.provenance import FileProvenanceTracker
    from pilates.generic.records import RecordStore, ModelRunInfo


logger = logging.getLogger(__name__)


def provenance_logging(func):
    """
    Thin PILATES-specific provenance bridge.

    Consist-only mode assumptions:
    - Run lifecycle and caching are owned by Consist Scenario/Step contexts
      (i.e., this decorator must execute inside `with scenario.step(...):`).
    - PILATES continues to pass `RecordStore` inputs/outputs between stages.

    Responsibilities:
    1. Extract a `RecordStore` from args/kwargs and log it as inputs.
    2. Attach PILATES workflow metadata to the active Consist run.
    3. Execute the wrapped method.
    4. Log any returned `RecordStore` as outputs.
    5. Update legacy in-memory `run_info` for PILATES compatibility.

    Non-responsibilities (handled elsewhere):
    - Starting/ending Consist runs.
    - Cache hydration / auto-skip (stubbed for later).
    - Legacy DuckDB provenance upserts.
    """

    @functools.wraps(func)
    def wrapper(self: "Model", *args, **kwargs):  # Removed specific 'store' arg
        model_run_id = None
        output_store_from_func = None
        run_info_from_func = None

        # Determine method name to provide better description
        method_name = func.__name__.replace("_", " ").strip()
        description = f"{self.model_name} {method_name} run"

        # Dynamically determine the 'inputs' RecordStore from args/kwargs
        input_record_store = None
        # Check positional arguments first
        from pilates.generic.records import RecordStore

        for arg in args:
            if isinstance(arg, RecordStore):
                input_record_store = arg
                break
        # If not found in positional, check keyword arguments known to be RecordStores
        if not input_record_store:
            if "store" in kwargs and isinstance(kwargs["store"], RecordStore):
                input_record_store = kwargs["store"]
            elif "previous_records" in kwargs and isinstance(
                kwargs["previous_records"], RecordStore
            ):
                input_record_store = kwargs["previous_records"]
            elif "raw_outputs" in kwargs and isinstance(
                kwargs["raw_outputs"], RecordStore
            ):
                input_record_store = kwargs["raw_outputs"]

        try:
            if not self.provenance_tracker:
                logger.warning(
                    f"[{self.model_name}] No provenance tracker configured; executing without provenance."
                )
                return func(self, *args, **kwargs)

            is_consist_backed = hasattr(self.provenance_tracker, "_tracker")
            if is_consist_backed:
                tracker = self.provenance_tracker._tracker
                if not getattr(tracker, "current_consist", None):
                    raise RuntimeError(
                        f"[{self.model_name}] Consist-backed provenance requires an active step run. "
                        "Ensure this method is called within `with scenario.step(...):`."
                    )

                # Best-effort cache hit visibility; we still execute user code.
                if getattr(tracker, "is_cached", False):
                    cached_run = getattr(tracker.current_consist, "cached_run", None)
                    cache_source = cached_run.id if cached_run else None
                    msg = (
                        f"[{self.model_name}] Consist cache hit for step '{tracker.current_consist.run.id}'"
                    )
                    if cache_source:
                        msg += f" (source={cache_source})"
                    msg += "; proceeding with execution (hydration not yet implemented)."
                    logger.info(msg)

            # 1. Bridge into active run: log inputs + metadata, update in-memory run_info
            model_run_id = self.provenance_tracker.start_model_run(
                self.model_name,
                self.state.current_year,
                self.state.current_inner_iter,
                description=description,
                inputs=input_record_store,  # Use the extracted input RecordStore
                state=self.state,
            )
            # 2. Legacy DB logging only when not Consist-backed (should disappear over time).
            if not is_consist_backed:
                self.start()

            # 3. Execute the actual model logic
            func_result = func(self, *args, **kwargs)  # Pass all original args/kwargs

            # Handle different return types
            if (
                isinstance(func_result, tuple)
                and len(func_result) == 2
                and isinstance(func_result[0], RecordStore)
            ):
                output_store_from_func, run_info_from_func = func_result
            elif isinstance(func_result, RecordStore):
                output_store_from_func = func_result
                run_info_from_func = self.provenance_tracker.run_info.model_runs.get(
                    model_run_id
                )  # Get run_info for metadata
            else:
                # If the function returns something else, it's not a standard RecordStore output.
                # We'll log completion but won't have specific output records or run_info metadata
                logger.warning(
                    f"Decorated method {func.__name__} returned unexpected type: {type(func_result)}. Expected RecordStore or Tuple[RecordStore, ModelRunInfo]. Provenance output records/metadata might be incomplete."
                )
                output_store_from_func = (
                    RecordStore()
                )  # Default to empty for completion logging
                run_info_from_func = self.provenance_tracker.run_info.model_runs.get(
                    model_run_id
                )  # Get run_info for metadata

            # NEW: Add output records to the main file_records dictionary
            if output_store_from_func:
                for record in output_store_from_func.all_records():
                    if (
                        record.unique_id
                        not in self.provenance_tracker.run_info.file_records
                    ):
                        self.provenance_tracker.run_info.file_records[
                            record.unique_id
                        ] = record

            # 4. Mark the run as completed in memory
            self.provenance_tracker.complete_model_run(
                model_run_id,
                status="completed",
                output_records=(
                    output_store_from_func.all_records()
                    if output_store_from_func
                    else []
                ),
                metadata=run_info_from_func.metadata if run_info_from_func else {},
            )

            # Return the original result of the function
            return func_result

        except Exception as e:
            logger.error(
                f"[{self.model_name}] {method_name} failed with an exception: {e}",
                exc_info=True,
            )
            if model_run_id:
                # Mark run as failed in memory
                self.provenance_tracker.complete_model_run(
                    model_run_id, status="failed"
                )
            # Re-raise the exception to stop the workflow
            raise e

        finally:
            # 5. Persist legacy DB state only when not Consist-backed.
            if model_run_id and self.provenance_tracker and not hasattr(
                self.provenance_tracker, "_tracker"
            ):
                logger.debug(
                    f"Persisting final state for legacy model run {model_run_id}."
                )
                self.stop()

    return wrapper


class Model:
    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: Optional[ConsistProvenanceTracker],
        major_stage: Optional["WorkflowState.Stage"] = None,  # new
    ):
        self.model_name = model_name
        self.state = state
        self.provenance_tracker = provenance_tracker
        self.major_stage = major_stage  # new

    def update_state(self, state: "WorkflowState"):
        self.state = state

    def start(self):
        """
        If the database is enabled, this method logs the start of the model
        component's execution to the database. It should be called after
        provenance_tracker.start_model_run().
        """
        if not self.provenance_tracker or not self.provenance_tracker.run_info:
            return

        if hasattr(self.provenance_tracker, "_tracker"):
            logger.info("Consist should handle its own tagging now")
            return

        database_path_str = self.state.full_settings.shared.database.path
        if not database_path_str:
            return

        # Assumes the provenance tracker has a concept of the "current" model run
        model_run_id = getattr(self.provenance_tracker, "current_model_run_id", None)
        if not model_run_id:
            logger.debug(
                "No active model run in provenance tracker; skipping DB start log."
            )
            return

        try:
            with DuckDBManager(database_path_str) as db_manager:
                conn = db_manager._get_connection()
                conn.begin()

                # Get the newly started model run record
                model_run = self.provenance_tracker.run_info.model_runs.get(
                    model_run_id
                )
                if not model_run:
                    return

                logger.debug(f"Logging start of model run {model_run_id} to database.")

                # 1. Upsert the model run itself (status should be 'running')
                db_manager.upsert_model_run(
                    conn, model_run, self.provenance_tracker.run_info.run_id
                )

                # 2. Upsert its input file records
                for record_hash in model_run.input_record_hashes:
                    file_record = self.provenance_tracker.run_info.file_records.get(
                        record_hash
                    )
                    if file_record:
                        db_manager.upsert_file_record(
                            conn, file_record, self.provenance_tracker.run_info.run_id
                        )

                conn.commit()

        except Exception as e:
            logger.warning(
                f"Failed to log model start for {model_run_id} to database: {e}",
                exc_info=True,
            )

    def stop(self):
        """
        If the database is enabled, this method logs the completion of the
        model component's execution. It should be called after
        provenance_tracker.complete_model_run().
        """
        if not self.provenance_tracker or not self.provenance_tracker.run_info:
            return

        if hasattr(self.provenance_tracker, "_tracker"):
            logger.info("Consist should handle its own tagging now")
            return

        database_path_str = self.state.full_settings.shared.database.path
        if not database_path_str:
            return

        # The model run should have been completed in the tracker, but we can still get its ID.
        model_run_id = getattr(self.provenance_tracker, "current_model_run_id", None)
        if not model_run_id:
            logger.debug(
                "No active model run to stop in provenance tracker; skipping DB stop log."
            )
            return

        try:
            with DuckDBManager(database_path_str) as db_manager:
                conn = db_manager._get_connection()
                conn.begin()

                # Get the now-completed model run record
                model_run = self.provenance_tracker.run_info.model_runs.get(
                    model_run_id
                )
                if not model_run:
                    return

                logger.debug(
                    f"Logging completion of model run {model_run_id} to database."
                )

                # 1. Upsert the model run itself (status should be 'completed' or 'failed')
                db_manager.upsert_model_run(
                    conn, model_run, self.provenance_tracker.run_info.run_id
                )

                # 2. Upsert its output file records
                for record_hash in model_run.output_record_hashes:
                    file_record = self.provenance_tracker.run_info.file_records.get(
                        record_hash
                    )
                    if file_record:
                        db_manager.upsert_file_record(
                            conn, file_record, self.provenance_tracker.run_info.run_id
                        )

                conn.commit()

        except Exception as e:
            logger.warning(
                f"Failed to log model completion for {model_run_id} to database: {e}",
                exc_info=True,
            )
