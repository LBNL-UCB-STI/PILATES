from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.coupler_helpers import resolve_input_precedence


@dataclass(frozen=True)
class ResolvedStepInputs:
    """
    Canonical representation of resolved step inputs.

    Attributes
    ----------
    inputs : dict
        Explicit input mapping to pass via ``StepRef.inputs``.
    input_keys : list
        Coupler keys to pass via ``StepRef.input_keys``.
    source_by_key : dict
        Resolution source per key: ``explicit`` / ``coupler`` / ``fallback`` / ``missing``.
    coupler_key_by_key : dict
        Actual coupler key used for each resolved coupler-backed input.
    missing_required : list
        Required keys that could not be resolved from any source.
    """

    inputs: Dict[str, Any]
    input_keys: list[str]
    source_by_key: Dict[str, str]
    coupler_key_by_key: Dict[str, str]
    missing_required: list[str]

    def stepref_inputs(self) -> Optional[Dict[str, Any]]:
        return self.inputs or None

    def stepref_input_keys(self) -> Optional[list[str]]:
        return self.input_keys or None


def _resolve_single_key(
    *,
    key: str,
    coupler: Optional[CouplerProtocol],
    explicit_inputs: Optional[Mapping[str, Any]],
    fallback_inputs: Optional[Mapping[str, Any]],
) -> tuple[str, Any, Optional[str]]:
    """
    Resolve one key using canonical precedence:
    explicit input -> coupler key -> fallback input.
    """
    return resolve_input_precedence(
        key=key,
        coupler=coupler,
        explicit_inputs=explicit_inputs,
        fallback_inputs=fallback_inputs,
    )


def resolve_step_inputs(
    *,
    keys: Iterable[str],
    coupler: Optional[CouplerProtocol] = None,
    explicit_inputs: Optional[Mapping[str, Any]] = None,
    fallback_inputs: Optional[Mapping[str, Any]] = None,
    required_keys: Optional[Sequence[str]] = None,
) -> ResolvedStepInputs:
    """
    Resolve a set of keys into ``inputs`` and ``input_keys`` for a step.
    """
    resolved_inputs: Dict[str, Any] = {}
    resolved_input_keys: list[str] = []
    source_by_key: Dict[str, str] = {}
    coupler_key_by_key: Dict[str, str] = {}

    for key in keys:
        source, value, coupler_key = _resolve_single_key(
            key=key,
            coupler=coupler,
            explicit_inputs=explicit_inputs,
            fallback_inputs=fallback_inputs,
        )
        source_by_key[key] = source
        if source == "coupler":
            selected_key = coupler_key or key
            # Keep the canonical workflow key on StepRef.input_keys.
            # Namespace-aware aliases remain internal lookup details and are
            # tracked separately via coupler_key_by_key for later lazy access.
            resolved_input_keys.append(key)
            coupler_key_by_key[key] = selected_key
        elif source in {"explicit", "fallback"}:
            resolved_inputs[key] = value

    required = list(required_keys or [])
    missing_required = [key for key in required if source_by_key.get(key) == "missing"]
    return ResolvedStepInputs(
        inputs=resolved_inputs,
        input_keys=resolved_input_keys,
        source_by_key=source_by_key,
        coupler_key_by_key=coupler_key_by_key,
        missing_required=missing_required,
    )


def resolve_preferred_step_input(
    *,
    preferred_keys: Sequence[str],
    coupler: Optional[CouplerProtocol] = None,
    explicit_inputs: Optional[Mapping[str, Any]] = None,
    fallback_inputs: Optional[Mapping[str, Any]] = None,
    required: bool = False,
) -> ResolvedStepInputs:
    """
    Resolve at most one key from an ordered preference list.

    For each candidate key, per-key precedence remains:
    explicit input -> coupler key -> fallback input.
    """
    for key in preferred_keys:
        resolved = resolve_step_inputs(
            keys=[key],
            coupler=coupler,
            explicit_inputs=explicit_inputs,
            fallback_inputs=fallback_inputs,
            required_keys=None,
        )
        if resolved.source_by_key.get(key) != "missing":
            return resolved

    missing_required = [preferred_keys[0]] if required and preferred_keys else []
    return ResolvedStepInputs(
        inputs={},
        input_keys=[],
        source_by_key={key: "missing" for key in preferred_keys},
        coupler_key_by_key={},
        missing_required=missing_required,
    )


def first_resolved_key(
    resolved: ResolvedStepInputs,
    candidate_keys: Sequence[str],
) -> Optional[str]:
    """
    Return the first non-missing key from ``candidate_keys``.
    """
    for key in candidate_keys:
        source = resolved.source_by_key.get(key)
        if source is not None and source != "missing":
            return key
    return None


def resolved_value_for_key(
    *,
    resolved: ResolvedStepInputs,
    key: str,
    coupler: Optional[CouplerProtocol] = None,
) -> Any:
    """
    Return the resolved value for ``key`` from a prior resolution result.

    ``ResolvedStepInputs.inputs`` only stores explicit/fallback values. Coupler
    values are fetched lazily from ``coupler`` to preserve source semantics.
    """
    source = resolved.source_by_key.get(key)
    if source == "coupler":
        get_value = getattr(coupler, "get", None) if coupler is not None else None
        if callable(get_value):
            selected_key = resolved.coupler_key_by_key.get(key, key)
            value = get_value(selected_key)
            if value is not None:
                return value
            if selected_key != key:
                return get_value(key)
        return None
    if source in {"explicit", "fallback"}:
        return resolved.inputs.get(key)
    return None
