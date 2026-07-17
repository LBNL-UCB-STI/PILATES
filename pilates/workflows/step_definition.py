"""PILATES definitions for one native Consist step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, TypeVar

from consist import CacheOptions, ExecutionOptions

from pilates.workflows.output_projection import TypedOutputProjector
from pilates.workflows.resolved_inputs import ResolvedStepInputs

OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True)
class StepDefinition(Generic[OutputT]):
    """One executable PILATES definition backed by a decorated Consist callable."""

    name: str
    function: Callable[..., None]
    resolve_inputs: Callable[..., ResolvedStepInputs]
    project_outputs: TypedOutputProjector[OutputT]
    output_paths: Callable[..., Mapping[str, Any]] | None = None
    execution_options: Callable[..., ExecutionOptions] | None = None
    cache_options: Callable[..., CacheOptions] | None = None
