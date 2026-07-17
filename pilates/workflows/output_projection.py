"""Pure projection from persisted Consist outputs to PILATES output values."""

from __future__ import annotations

from typing import Any, Generic, Mapping, Protocol, TypeVar

from consist.models.artifact import Artifact

OutputT = TypeVar("OutputT")


class TypedOutputProjector(Protocol, Generic[OutputT]):
    """Project persisted Consist output links into validated PILATES outputs."""

    def __call__(
        self,
        outputs: Mapping[str, Artifact],
        *,
        settings: Any,
        state: Any,
        workspace: Any,
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
