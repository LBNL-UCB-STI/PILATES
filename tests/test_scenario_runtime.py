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


def test_activitysim_postprocess_does_not_replace_activitysim_run_producer() -> None:
    """Only the ActivitySim execution phase may seed BEAM's parent producer."""

    @define_step(model="activitysim")
    def activitysim_step() -> None:
        return None

    class _Scenario:
        def __init__(self) -> None:
            self.run_ids = iter(("activitysim-run-id", "activitysim-postprocess-id"))

        def run(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(run=SimpleNamespace(id=next(self.run_ids)))

    proxy = ScenarioParentLinkProxy(_Scenario())

    proxy.run(fn=activitysim_step, year=2018, iteration=0, phase="run")
    proxy.run(fn=activitysim_step, year=2018, iteration=0, phase="postprocess")

    assert proxy._activitysim_run_ids == {(2018, 0): "activitysim-run-id"}


def test_parent_link_proxy_records_causal_parent_without_overwriting_scenario_parent() -> (
    None
):
    """Causal lineage is stored via tags/facets while leaving scenario containment intact."""

    @define_step(model="activitysim")
    def activitysim_step() -> None:
        return None

    @define_step(model="beam")
    def beam_step() -> None:
        return None

    captured_kwargs: list[dict[str, object]] = []

    class _Scenario:
        def __init__(self) -> None:
            self.run_ids = iter(("asim-run-1", "beam-run-1", "asim-run-2"))

        def run(self, **kwargs: object) -> SimpleNamespace:
            captured_kwargs.append(dict(kwargs))
            return SimpleNamespace(run=SimpleNamespace(id=next(self.run_ids)))

    proxy = ScenarioParentLinkProxy(_Scenario())

    # Step 1: ActivitySim iteration 0 (no causal parent expected)
    proxy.run(
        fn=activitysim_step,
        year=2018,
        iteration=0,
        phase="run",
        tags=["initial_step"],
        facet={"step_key": "asim_0"},
    )
    asim1_call = captured_kwargs[0]
    assert "parent_run_id" not in asim1_call
    assert asim1_call.get("tags") == ["initial_step"]
    assert asim1_call.get("facet") == {"step_key": "asim_0"}

    # Step 2: BEAM iteration 0 (causal parent should be asim-run-1)
    proxy.run(
        fn=beam_step,
        year=2018,
        iteration=0,
        tags=["sim_step"],
        facet={"substage": "beam_0"},
    )
    beam_call = captured_kwargs[1]
    assert "parent_run_id" not in beam_call
    assert "causal_parent:asim-run-1" in (beam_call.get("tags") or [])
    assert (beam_call.get("tags") or []).count("sim_step") == 1
    assert isinstance(beam_call.get("facet"), dict)
    assert beam_call["facet"].get("substage") == "beam_0"
    assert beam_call["facet"].get("causal_parent_run_id") == "asim-run-1"

    # Step 3: ActivitySim iteration 1 (causal parent should be beam-run-1)
    proxy.run(
        fn=activitysim_step,
        year=2018,
        iteration=1,
        phase="run",
    )
    asim2_call = captured_kwargs[2]
    assert "parent_run_id" not in asim2_call
    assert "causal_parent:beam-run-1" in (asim2_call.get("tags") or [])
    assert isinstance(asim2_call.get("facet"), dict)
    assert asim2_call["facet"].get("causal_parent_run_id") == "beam-run-1"
