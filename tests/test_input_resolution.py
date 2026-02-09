"""
Narrative contract tests for workflow input resolution.

These tests document the canonical model used throughout workflow stage assembly:

1. Resolve each candidate key with strict precedence:
   explicit input -> coupler value -> fallback input.
2. Separate the result into:
   - ``inputs`` for explicit/fallback concrete paths
   - ``input_keys`` for coupler-backed references
3. Report missing required keys deterministically.

Why this matters:
- Stage builders can share one predictable resolution model.
- Cache identity and execution behavior stay stable across refactors.
- Developers can reason about data provenance without reading each stage module.
"""

from pilates.workflows.input_resolution import (
    first_resolved_key,
    resolve_preferred_step_input,
    resolve_step_inputs,
    resolved_value_for_key,
)


class _CouplerStub:
    def __init__(self, values):
        self._values = dict(values)

    def get(self, key, default=None):
        return self._values.get(key, default)


def test_resolve_step_inputs_prefers_explicit_over_coupler_and_fallback():
    """Explicit values always win for a given key."""
    coupler = _CouplerStub({"foo": "from-coupler"})
    resolved = resolve_step_inputs(
        keys=["foo"],
        coupler=coupler,
        explicit_inputs={"foo": "from-explicit"},
        fallback_inputs={"foo": "from-fallback"},
    )
    assert resolved.inputs == {"foo": "from-explicit"}
    assert resolved.input_keys == []
    assert resolved.source_by_key["foo"] == "explicit"


def test_resolve_step_inputs_prefers_coupler_over_fallback():
    """Coupler-backed values win when no explicit value is provided."""
    coupler = _CouplerStub({"foo": "from-coupler"})
    resolved = resolve_step_inputs(
        keys=["foo"],
        coupler=coupler,
        fallback_inputs={"foo": "from-fallback"},
    )
    assert resolved.inputs == {}
    assert resolved.input_keys == ["foo"]
    assert resolved.source_by_key["foo"] == "coupler"


def test_resolve_step_inputs_uses_fallback_when_no_explicit_or_coupler():
    """Fallback is used only when explicit/coupler sources are unavailable."""
    resolved = resolve_step_inputs(
        keys=["foo"],
        explicit_inputs={"foo": None},
        fallback_inputs={"foo": "from-fallback"},
    )
    assert resolved.inputs == {"foo": "from-fallback"}
    assert resolved.input_keys == []
    assert resolved.source_by_key["foo"] == "fallback"


def test_resolve_step_inputs_reports_missing_required():
    """Required keys are surfaced in ``missing_required`` for clear stage errors."""
    resolved = resolve_step_inputs(
        keys=["foo"],
        explicit_inputs={"foo": None},
        fallback_inputs={"foo": None},
        required_keys=["foo"],
    )
    assert resolved.inputs == {}
    assert resolved.input_keys == []
    assert resolved.source_by_key["foo"] == "missing"
    assert resolved.missing_required == ["foo"]


def test_resolve_preferred_step_input_selects_first_available_key():
    """Preferred-key resolution picks the first candidate that can be resolved."""
    coupler = _CouplerStub({"second": "from-coupler"})
    resolved = resolve_preferred_step_input(
        preferred_keys=["first", "second", "third"],
        coupler=coupler,
    )
    assert resolved.inputs == {}
    assert resolved.input_keys == ["second"]
    assert resolved.source_by_key["second"] == "coupler"


def test_resolve_preferred_step_input_respects_per_key_precedence():
    """Preferred order does not bypass per-key precedence rules."""
    coupler = _CouplerStub({"second": "from-coupler"})
    resolved = resolve_preferred_step_input(
        preferred_keys=["first", "second"],
        coupler=coupler,
        explicit_inputs={"second": "from-explicit"},
    )
    assert resolved.inputs == {"second": "from-explicit"}
    assert resolved.input_keys == []
    assert resolved.source_by_key["second"] == "explicit"


def test_first_resolved_key_returns_first_non_missing_candidate():
    """Helper returns the first resolved candidate and skips missing entries."""
    coupler = _CouplerStub({"second": "from-coupler"})
    resolved = resolve_preferred_step_input(
        preferred_keys=["first", "second", "third"],
        coupler=coupler,
    )
    assert first_resolved_key(resolved, ["first", "second", "third"]) == "second"


def test_resolved_value_for_key_fetches_coupler_values():
    """Helper rehydrates coupler-backed values from the coupler when needed."""
    coupler = _CouplerStub({"foo": "from-coupler"})
    resolved = resolve_step_inputs(
        keys=["foo"],
        coupler=coupler,
    )
    assert resolved_value_for_key(resolved=resolved, key="foo", coupler=coupler) == "from-coupler"
