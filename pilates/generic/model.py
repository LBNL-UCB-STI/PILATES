from __future__ import annotations

import functools
import logging
from typing import Optional, TYPE_CHECKING

from pilates.utils import consist_runtime as cr

if TYPE_CHECKING:
    from workflow_state import WorkflowState
    from pilates.generic.records import RecordStore


logger = logging.getLogger(__name__)


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
                cr.log_artifacts(input_record_store.to_mapping(), direction="input")

        func_result = func(self, *args, **kwargs)

        if cr.current_run():
            if isinstance(func_result, RecordStore):
                cr.log_artifacts(func_result.to_mapping(), direction="output")
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
