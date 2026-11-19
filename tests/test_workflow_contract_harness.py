"""Tests for the production-like workflow contract test harness."""

from types import SimpleNamespace

from tests.workflow_contract_harness import CouplerStub, FakeScenario


def test_fake_scenario_injects_requested_consist_context():
    """Match Consist's context-injection contract for wrapped steps."""

    received = {}

    def step(*, _consist_ctx=None):
        received["context"] = _consist_ctx

    scenario = FakeScenario(CouplerStub())

    scenario.run(
        fn=step,
        execution_options=SimpleNamespace(
            runtime_kwargs={},
            inject_context="_consist_ctx",
        ),
    )

    assert received["context"] is not None
    assert received["context"].canonicalization is None
