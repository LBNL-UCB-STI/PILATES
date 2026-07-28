"""Classification contract for the generic binding fallback registry."""

from pilates.workflows.binding import (
    FALLBACK_PROVIDER_INVENTORY,
    _FALLBACK_PROVIDERS,
)


def test_every_registered_generic_fallback_provider_has_one_inventory_entry() -> None:
    identifiers = tuple(entry.identifier for entry in FALLBACK_PROVIDER_INVENTORY)

    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) == set(_FALLBACK_PROVIDERS)


def test_generic_fallback_inventory_declares_policy_and_exit_for_every_provider() -> (
    None
):
    allowed_classes = {
        "bootstrap",
        "recovery",
        "format_selection",
        "legacy_compatibility",
    }
    allowed_end_states = {"retain", "replace_with_producer_handoff", "delete"}
    expected_policy = {
        "urbansim_inputs_for_year": (
            "legacy_compatibility",
            "replace_with_producer_handoff",
        ),
        "activitysim_input_datastore": (
            "bootstrap",
            "replace_with_producer_handoff",
        ),
        "activitysim_population_source": (
            "legacy_compatibility",
            "replace_with_producer_handoff",
        ),
    }

    for entry in FALLBACK_PROVIDER_INVENTORY:
        assert entry.consuming_steps
        assert entry.semantic_roles
        assert entry.trigger
        assert entry.candidate_order
        assert entry.identity_source
        assert entry.policy_class in allowed_classes
        assert entry.intended_end_state in allowed_end_states
        assert entry.focused_tests

    assert {
        entry.identifier: (entry.policy_class, entry.intended_end_state)
        for entry in FALLBACK_PROVIDER_INVENTORY
    } == expected_policy
