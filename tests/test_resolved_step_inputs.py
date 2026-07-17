from pathlib import Path

import pytest
from consist import BindingResult

from pilates.workflows.resolved_inputs import ResolvedStepInputs


def test_selected_roles_are_deterministic_and_include_binding_contract_roles() -> None:
    resolved = ResolvedStepInputs(
        step_name="example",
        binding=BindingResult(
            inputs={"explicit": "value"},
            input_keys=["required", "explicit"],
            optional_input_keys=["optional"],
        ),
    )

    assert resolved.selected_roles() == ("explicit", "required", "optional")


def test_require_complete_fails_closed_with_step_and_roles() -> None:
    resolved = ResolvedStepInputs(
        step_name="beam_postprocess",
        binding=BindingResult(inputs={}),
        required_roles=("events", "skims"),
        source_by_role={"events": "coupler", "skims": "missing"},
        logical_destinations={"events": Path("/tmp/events")},
    )

    with pytest.raises(RuntimeError, match="beam_postprocess.*skims"):
        resolved.require_complete()


def test_resolved_role_diagnostics_cannot_be_mutated_after_selection() -> None:
    source_by_role = {"events": "coupler"}
    resolved = ResolvedStepInputs(
        step_name="beam_postprocess",
        binding=BindingResult(inputs={}),
        source_by_role=source_by_role,
    )

    source_by_role["events"] = "history"

    assert resolved.source_by_role == {"events": "coupler"}
    with pytest.raises(TypeError):
        resolved.source_by_role["events"] = "history"  # type: ignore[index]


def test_resolved_binding_cannot_be_mutated_after_selection() -> None:
    inputs = {"explicit": "selected"}
    input_keys = ["required"]
    resolved = ResolvedStepInputs(
        step_name="example",
        binding=BindingResult(inputs=inputs, input_keys=input_keys),
    )

    inputs["explicit"] = "changed"
    input_keys.append("later")

    assert resolved.binding.inputs == {"explicit": "selected"}
    assert resolved.selected_roles() == ("explicit", "required")
    with pytest.raises(TypeError):
        resolved.binding.inputs["explicit"] = "changed"  # type: ignore[index]


def test_resolved_required_and_optional_roles_cannot_be_mutated_after_selection() -> None:
    required_roles = ["skims"]
    optional_roles = ["warmstart"]
    resolved = ResolvedStepInputs(
        step_name="beam_postprocess",
        binding=BindingResult(inputs={}),
        required_roles=required_roles,
        optional_roles=optional_roles,
        source_by_role={"skims": "missing"},
    )

    required_roles.clear()
    optional_roles.clear()

    assert resolved.required_roles == ("skims",)
    assert resolved.optional_roles == ("warmstart",)
    with pytest.raises(RuntimeError, match="beam_postprocess.*skims"):
        resolved.require_complete()
