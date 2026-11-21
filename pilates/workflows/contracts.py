"""Leaf workflow step contracts and catalog specification types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Mapping,
    Protocol,
    TypeVar,
)

from consist import Artifact, CacheOptions, ExecutionOptions, OutputSet

OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True)
class StepCallInputs:
    """Explicit typed inputs and staging destinations for one step execution."""

    inputs: Mapping[str, Artifact | Path] = field(default_factory=dict)
    logical_destinations: Mapping[str, Path] = field(default_factory=dict)
    required_keys: tuple[str, ...] = ()
    optional_keys: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def require_complete(self, step_name: str) -> None:
        """Raise ``RuntimeError`` when any required input key is missing."""
        missing = [key for key in self.required_keys if self.inputs.get(key) is None]
        if missing:
            raise RuntimeError(
                f"{step_name} is missing required input roles: "
                + ", ".join(sorted(missing))
            )


class TypedOutputProjector(Protocol, Generic[OutputT]):
    """Project persisted Consist output links into validated PILATES outputs."""

    def __call__(
        self,
        outputs: Mapping[str, Artifact],
        *,
        settings: Any,
        state: Any,
        workspace: Any,
        step_inputs: StepCallInputs | None = None,
    ) -> OutputT: ...


def require_output(
    outputs: Mapping[str, Artifact], *, step_name: str, key: str
) -> Artifact:
    """Return one required persisted output with actionable failure context."""
    try:
        return outputs[key]
    except KeyError as error:
        raise RuntimeError(
            f"{step_name} is missing required output {key!r}; "
            f"available keys: {', '.join(sorted(outputs)) or '<none>'}"
        ) from error


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
    resolve_inputs: Callable[..., StepCallInputs]
    project_outputs: TypedOutputProjector[OutputT]
    input_contract: InputContract
    outputs: Callable[..., Mapping[str, Any]] | None = None
    output_paths: Callable[..., Mapping[str, Any]] | None = None
    output_sets: Callable[..., Mapping[str, OutputSet]] | None = None
    execution_options: Callable[..., ExecutionOptions] | None = None
    cache_options: Callable[..., CacheOptions] | None = None
    optional_output_keys: tuple[str, ...] = ()
    preflight_identity: bool = False
    archive_outputs: bool = True

    def __post_init__(self) -> None:
        if self.outputs is not None and self.output_paths is not None:
            raise ValueError(
                f"StepDefinition {self.name!r} cannot specify both outputs and output_paths."
            )


@dataclass(frozen=True)
class WorkflowStepProvenanceSpec:
    builder_key: str
