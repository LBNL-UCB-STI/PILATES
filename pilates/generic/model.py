from __future__ import annotations

import functools
import logging
from typing import Optional, TYPE_CHECKING

from pilates.utils import consist_runtime as cr

if TYPE_CHECKING:
    from workflow_state import WorkflowState
    from pilates.generic.records import RecordStore


logger = logging.getLogger(__name__)


_H5_INTERNAL_TOKENS = ("_axis", "_block", "_level", "_label")


def _normalize_h5_table_name(name: str) -> str:
    if name.startswith("/"):
        return name
    return f"/{name}"


def _h5_table_filter_from_list(tables_used):
    normalized = {_normalize_h5_table_name(name) for name in tables_used if name}

    def _filter(table_name: str) -> bool:
        if any(tok in table_name for tok in _H5_INTERNAL_TOKENS):
            return False
        return table_name in normalized

    return _filter


def _log_record_store(record_store: "RecordStore", *, direction: str) -> None:
    from pilates.generic.records import RecordStore as _RecordStore

    if not isinstance(record_store, _RecordStore):
        return

    bulk_mapping = {}
    metadata_by_key = {}
    facets_by_key = {}
    facet_schema_versions_by_key = {}
    facet_index_enabled = False
    for record in record_store.all_records():
        key = getattr(record, "short_name", None) or getattr(record, "unique_id", None)
        if not key:
            continue

        tables_used = getattr(record, "h5_tables_used", None)
        if tables_used:
            path = getattr(record, "file_path", None) or getattr(
                record, "repo_path", None
            )
        else:
            path = getattr(record, "container_uri", None) or getattr(record, "uri", None)
            if not path:
                path = getattr(record, "file_path", None) or getattr(
                    record, "repo_path", None
                )
        if not path:
            continue

        description = getattr(record, "description", None)
        meta = dict(getattr(record, "metadata", None) or {})
        if tables_used:
            table_filter = _h5_table_filter_from_list(tables_used)
            cr.log_h5_container(
                path,
                key=key,
                direction=direction,
                table_filter=table_filter,
                description=description,
                **meta,
            )
        else:
            bulk_mapping[key] = path
            facet = meta.pop("facet", None)
            facet_schema_version = meta.pop("facet_schema_version", None)
            facet_index = bool(meta.pop("facet_index", False))

            if description and "description" not in meta:
                meta["description"] = description
            if meta:
                metadata_by_key[key] = meta
            if facet is not None:
                facets_by_key[key] = facet
            if facet_schema_version is not None:
                facet_schema_versions_by_key[key] = facet_schema_version
            if facet_index:
                facet_index_enabled = True

    if bulk_mapping:
        kwargs = {"direction": direction}
        if metadata_by_key:
            kwargs["metadata_by_key"] = metadata_by_key
        if facets_by_key:
            kwargs["facets_by_key"] = facets_by_key
        if facet_schema_versions_by_key:
            kwargs["facet_schema_versions_by_key"] = facet_schema_versions_by_key
        if facet_index_enabled:
            kwargs["facet_index"] = True
        cr.log_artifacts(bulk_mapping, **kwargs)


def provenance_logging(func):
    """
    Thin PILATES-specific provenance bridge.

    Consist-only mode assumptions:
    - Run lifecycle and caching are owned by Consist Scenario/Step contexts.
    - PILATES continues to pass `RecordStore` inputs/outputs between stages.

    Responsibilities:

    1. Extract a `RecordStore` from args/kwargs and log it as inputs.
    2. Attach PILATES workflow metadata to the active Consist run.
    3. Execute the wrapped method.
    4. Log any returned `RecordStore` as outputs.

    Non-responsibilities (handled elsewhere):

    - Starting/ending Consist runs.
    - Cache hydration / auto-skip (stubbed for later).
    - Legacy DuckDB provenance upserts.
    """

    @functools.wraps(func)
    def wrapper(self: "Model", *args, **kwargs):  # Removed specific 'store' arg
        # Determine method name to provide better description
        method_name = func.__name__.replace("_", " ").strip()
        description = f"{self.model_name} {method_name} run"

        # Dynamically determine the 'inputs' RecordStore from args/kwargs
        input_record_store = None
        from pilates.generic.records import RecordStore

        # Check positional arguments first
        for arg in args:
            if isinstance(arg, RecordStore):
                input_record_store = arg
                break
        # If not found in positional, check keyword arguments known to be RecordStores
        if not input_record_store:
            for name in ("input_store", "store", "previous_records", "raw_outputs"):
                candidate = kwargs.get(name)
                if isinstance(candidate, RecordStore):
                    input_record_store = candidate
                    break

        if not hasattr(self, "state") or self.state is None:
            raise RuntimeError(
                f"[{self.model_name}] Model.state is not initialized. "
                "Ensure Model was instantiated with a valid WorkflowState."
            )

        if cr.consist_available(self.state.full_settings) and cr.current_run() is None:
            raise RuntimeError(
                f"[{self.model_name}] Consist enabled but no active run context. "
                "Ensure this method is called within `scenario.run(...)` or `scenario.trace(...)`."
            )

        if cr.current_run():
            cr.log_meta(
                pilates_model=self.model_name,
                pilates_description=description,
                pilates_year=self.state.current_year,
                pilates_forecast_year=self.state.forecast_year,
                pilates_iteration=self.state.current_inner_iter,
                pilates_stage=(
                    self.state.current_major_stage.name
                    if self.state.current_major_stage
                    else None
                ),
                pilates_sub_stage=(
                    self.state.current_sub_stage.name
                    if self.state.current_sub_stage
                    else None
                ),
            )
            if input_record_store:
                _log_record_store(input_record_store, direction="input")

        func_result = func(self, *args, **kwargs)

        if cr.current_run():
            if isinstance(func_result, RecordStore):
                _log_record_store(func_result, direction="output")
            elif func_result is not None:
                logger.warning(
                    f"Decorated method {func.__name__} returned unexpected type: "
                    f"{type(func_result)}. Expected RecordStore."
                )

        return func_result

    return wrapper


class Model:
    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        major_stage: Optional["WorkflowState.Stage"] = None,  # new
    ):
        self.model_name = model_name
        self.state = state
        self.major_stage = major_stage  # new

    def update_state(self, state: "WorkflowState"):
        self.state = state
