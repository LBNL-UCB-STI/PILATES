"""Immutable semantic input decisions for native Consist step execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from consist import BindingResult


@dataclass(frozen=True, slots=True)
class ResolvedStepInputs:
    """One completed PILATES semantic input decision for one invocation."""

    step_name: str
    binding: BindingResult
    required_roles: tuple[str, ...] = ()
    optional_roles: tuple[str, ...] = ()
    source_by_role: Mapping[str, str] = field(default_factory=dict)
    selected_key_by_role: Mapping[str, str] = field(default_factory=dict)
    logical_destinations: Mapping[str, Path] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Defensively freeze all selected inputs and diagnostic maps."""

        object.__setattr__(self, "required_roles", tuple(self.required_roles))
        object.__setattr__(self, "optional_roles", tuple(self.optional_roles))
        input_keys = self.binding.input_keys
        optional_input_keys = self.binding.optional_input_keys
        object.__setattr__(
            self,
            "binding",
            BindingResult(
                inputs=(
                    MappingProxyType(dict(self.binding.inputs))
                    if self.binding.inputs is not None
                    else None
                ),
                input_keys=(
                    input_keys
                    if isinstance(input_keys, str)
                    else tuple(input_keys or ())
                ),
                optional_input_keys=(
                    optional_input_keys
                    if isinstance(optional_input_keys, str)
                    else tuple(optional_input_keys or ())
                ),
                metadata=(
                    MappingProxyType(dict(self.binding.metadata))
                    if self.binding.metadata is not None
                    else None
                ),
            ),
        )

        object.__setattr__(
            self, "source_by_role", MappingProxyType(dict(self.source_by_role))
        )
        object.__setattr__(
            self,
            "selected_key_by_role",
            MappingProxyType(dict(self.selected_key_by_role)),
        )
        object.__setattr__(
            self,
            "logical_destinations",
            MappingProxyType(dict(self.logical_destinations)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def selected_roles(self) -> tuple[str, ...]:
        inputs = tuple((self.binding.inputs or {}).keys())
        required = self.binding.input_keys
        optional = self.binding.optional_input_keys
        required_roles = (
            (required,) if isinstance(required, str) else tuple(required or ())
        )
        optional_roles = (
            (optional,) if isinstance(optional, str) else tuple(optional or ())
        )
        return tuple(dict.fromkeys((*inputs, *required_roles, *optional_roles)))

    def require_complete(self) -> None:
        missing = [
            role
            for role in self.required_roles
            if self.source_by_role.get(role) in {None, "missing"}
        ]
        if missing:
            raise RuntimeError(
                f"{self.step_name} is missing required input roles: "
                + ", ".join(sorted(missing))
            )
