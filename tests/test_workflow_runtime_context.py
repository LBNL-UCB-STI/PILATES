from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pilates.runtime.context import (
    WorkflowRuntimeContext,
    ensure_workflow_runtime_context,
)
from pilates.workflows.stages.postprocessing import run_postprocessing_stage
from pilates.workflows.stages.supply_demand import run_supply_demand_stage
from pilates.workflows.stages.supply_demand_activity import (
    ActivityDemandPhaseOutputs,
)
from pilates.workflows.stages.supply_demand_beam import (
    TrafficAssignmentPhaseOutputs,
)


def test_workflow_runtime_context_builds_surface_when_missing(monkeypatch) -> None:
    settings = SimpleNamespace()
    state = SimpleNamespace()
    workspace = SimpleNamespace()
    built_surface = SimpleNamespace(profile=SimpleNamespace())
    captured = {}

    def _fake_build_enabled_workflow_surface(received_settings, *, state):
        captured["settings"] = received_settings
        captured["state"] = state
        return built_surface

    monkeypatch.setattr(
        "pilates.workflows.surface.build_enabled_workflow_surface",
        _fake_build_enabled_workflow_surface,
    )

    context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
    )

    assert context.surface is built_surface
    assert captured == {"settings": settings, "state": state}


def test_workflow_runtime_context_reuses_explicit_surface() -> None:
    settings = SimpleNamespace()
    state = SimpleNamespace()
    workspace = SimpleNamespace()
    surface = SimpleNamespace(profile=SimpleNamespace())

    context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )

    assert context.surface is surface
    assert ensure_workflow_runtime_context(context=context) is context


def test_run_supply_demand_stage_passes_runtime_context_to_phase_helpers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeStage:
        supply_demand_loop = "supply_demand_loop"
        activity_demand = "activity_demand"
        traffic_assignment = "traffic_assignment"
        land_use = "land_use"

    class _FakeState:
        Stage = _FakeStage
        iteration = 0
        current_major_stage = _FakeStage.supply_demand_loop
        year = 2018
        forecast_year = 2018

        def __iter__(self):
            return iter([2018])

        def should_run(self, major_stage, iteration=None, sub_stage=None):
            if major_stage != self.Stage.supply_demand_loop:
                return False
            if sub_stage is None:
                return True
            return sub_stage in {
                self.Stage.activity_demand,
                self.Stage.traffic_assignment,
            }

        def is_enabled(self, _stage):
            return False

        def complete_step(self, *args, **kwargs):
            return None

    settings = SimpleNamespace(
        run=SimpleNamespace(
            supply_demand_iters=1,
            models=SimpleNamespace(activity_demand=object()),
        )
    )
    state = _FakeState()
    workspace = SimpleNamespace(full_path=str(tmp_path))
    surface = SimpleNamespace(profile=SimpleNamespace())
    context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
    seen = {}

    monkeypatch.setattr(
        "pilates.workflows.stages.supply_demand._run_activity_demand_phase",
        lambda **kwargs: (
            seen.setdefault("activity_context", kwargs["context"]),
            ActivityDemandPhaseOutputs(activity_demand_outputs={"plans": "ok"}),
        )[1],
    )
    monkeypatch.setattr(
        "pilates.workflows.stages.supply_demand._run_traffic_assignment_phase",
        lambda **kwargs: (
            seen.setdefault("beam_context", kwargs["context"]),
            TrafficAssignmentPhaseOutputs(previous_beam_outputs={"linkstats": "ok"}),
        )[1],
    )
    monkeypatch.setattr(
        "pilates.workflows.stages.supply_demand.flush_archive_queue",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "pilates.workflows.stages.supply_demand.archive_copy_now",
        lambda *args, **kwargs: None,
    )

    run_supply_demand_stage(
        scenario=SimpleNamespace(),
        coupler=SimpleNamespace(),
        year=2018,
        usim_inputs={},
        build_manifest_path=lambda _workspace, _year, iteration: (
            tmp_path / f"iter-{iteration}.yaml"
        ),
        context=context,
    )

    assert seen["activity_context"] is context
    assert seen["beam_context"] is context


def test_run_postprocessing_stage_uses_runtime_context(
    monkeypatch, tmp_path: Path
) -> None:
    settings = SimpleNamespace()
    state = SimpleNamespace(iteration=0)
    workspace = SimpleNamespace(full_path=str(tmp_path))
    surface = SimpleNamespace(profile=SimpleNamespace())
    context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
    captured = {}

    monkeypatch.setattr(
        "pilates.workflows.stages.postprocessing.make_postprocessing_step",
        lambda: "postprocess-step",
    )
    monkeypatch.setattr(
        "pilates.workflows.stages.postprocessing.run_workflow",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "pilates.workflows.stages.postprocessing.flush_archive_queue",
        lambda *args, **kwargs: None,
    )

    run_postprocessing_stage(
        scenario="scenario",
        coupler="coupler",
        year=2018,
        context=context,
    )

    assert captured["settings"] is settings
    assert captured["state"] is state
    assert captured["workspace"] is workspace
