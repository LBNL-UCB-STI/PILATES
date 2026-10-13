"""PILATES definitions for one native Consist step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, Mapping, TypeVar

from consist import CacheOptions, ExecutionOptions, OutputSet

from pilates.workflows.output_projection import TypedOutputProjector
from pilates.workflows.resolved_inputs import ResolvedStepInputs

OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True)
class ConfigContract:
    """The canonical configuration form a native executable will eventually use."""

    kind: Literal["adapter", "payload"]
    adapter_name: str | None = None

    @classmethod
    def adapter(cls, name: str) -> ConfigContract:
        """Declare a named Consist configuration-adapter contract."""

        return cls(kind="adapter", adapter_name=name)

    @classmethod
    def payload(cls) -> ConfigContract:
        """Declare an allow-listed scalar configuration payload contract."""

        return cls(kind="payload")

    def __post_init__(self) -> None:
        if self.kind == "adapter" and not self.adapter_name:
            raise ValueError("adapter configuration contracts require an adapter name")
        if self.kind == "payload" and self.adapter_name is not None:
            raise ValueError("payload configuration contracts cannot name an adapter")


@dataclass(frozen=True, slots=True)
class InputContract:
    """Cache-eligibility classification owned by one native step definition."""

    status: Literal["incomplete", "complete"]
    reason: str | None = None
    config_contract: ConfigContract | None = None

    def __post_init__(self) -> None:
        if self.status == "incomplete" and not self.reason:
            raise ValueError("incomplete input contracts require a reason")
        if self.status == "complete" and self.reason is not None:
            raise ValueError("complete input contracts cannot include a reason")
        if self.status == "complete" and self.config_contract is None:
            raise ValueError(
                "complete input contracts require a configuration contract"
            )


@dataclass(frozen=True, slots=True)
class StepDefinition(Generic[OutputT]):
    """One executable PILATES definition backed by a decorated Consist callable."""

    name: str
    function: Callable[..., None]
    resolve_inputs: Callable[..., ResolvedStepInputs]
    project_outputs: TypedOutputProjector[OutputT]
    input_contract: InputContract
    output_paths: Callable[..., Mapping[str, Any]] | None = None
    output_sets: Callable[..., Mapping[str, OutputSet]] | None = None
    execution_options: Callable[..., ExecutionOptions] | None = None
    cache_options: Callable[..., CacheOptions] | None = None
    preflight_identity: bool = False
    archive_outputs: bool = True
