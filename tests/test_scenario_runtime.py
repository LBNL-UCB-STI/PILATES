from __future__ import annotations

from types import SimpleNamespace

from consist import define_step

from pilates.runtime.scenario_runtime import ScenarioParentLinkProxy


def test_parent_link_proxy_does_not_add_model_override_to_preflight_identity() -> None:
    """A preflight identity already owns the callable model metadata."""

    @define_step(model="urbansim_postprocess")
    def postprocess() -> None:
        return None

    seen: dict[str, object] = {}

    class _Scenario:
        def run(self, **kwargs: object) -> SimpleNamespace:
            seen.update(kwargs)
            return SimpleNamespace(run=SimpleNamespace(id="run-id"))

    identity = SimpleNamespace()
    proxy = ScenarioParentLinkProxy(_Scenario())

    proxy.run(fn=postprocess, step_identity=identity, year=2017, iteration=0)

    assert seen["step_identity"] is identity
    assert "model" not in seen
