from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


from pilates.runtime.context import WorkflowRuntimeContext
from pilates.workspace import Workspace
from pilates.workflows.orchestrator import (
    _run_supply_demand_loop as run_supply_demand_stage,
    run_postprocessing_stage,
)
from pilates.workflows.sim_state import SimState
from tests.conftest import (
    make_conforming_settings,
    make_conforming_workflow_runtime_context,
    make_conforming_workspace,
)


def test_workflow_runtime_context_builds_from_parts() -> None:
    settings = SimpleNamespace()
    state = SimpleNamespace()
    cb = lambda _msg: None
    workspace = SimpleNamespace(on_archive_commit=cb, expected_prefix_steps=4)

    context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
    )

    assert context.settings is settings
    assert context.state is state
    assert context.workspace is workspace
    assert context.on_archive_commit is cb
    assert context.expected_prefix_steps == 4


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
        is_restart_run = False

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
            models=SimpleNamespace(activity_demand=object(), travel="beam"),
        )
    )
    state = _FakeState()
    workspace = make_conforming_workspace(tmp_path)
    context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    seen = {}

    monkeypatch.setattr(
        "pilates.workflows.orchestrator.run_activity_demand_phase",
        lambda **kwargs: (
            seen.setdefault("activity_settings", kwargs["settings"]),
            seen.setdefault("activity_state", kwargs["state"]),
            seen.setdefault("activity_workspace", kwargs["workspace"]),
            kwargs["sim_state"],
        )[3],
    )
    monkeypatch.setattr(
        "pilates.workflows.orchestrator.run_traffic_assignment_phase",
        lambda **kwargs: (
            seen.setdefault("beam_settings", kwargs["settings"]),
            seen.setdefault("beam_state", kwargs["state"]),
            seen.setdefault("beam_workspace", kwargs["workspace"]),
            kwargs["sim_state"],
        )[3],
    )
    run_supply_demand_stage(
        scenario=SimpleNamespace(),
        settings=context.settings,
        state=context.state,
        workspace=context.workspace,
        sim_state=SimState(),
        year=2018,
    )

    assert seen["activity_settings"] is settings
    assert seen["activity_state"] is state
    assert seen["activity_workspace"] is workspace
    assert seen["beam_settings"] is settings
    assert seen["beam_state"] is state
    assert seen["beam_workspace"] is workspace


def test_supply_demand_snapshots_after_activitysim_before_beam(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A completed ActivitySim handoff is durable before BEAM begins."""

    class _FakeStage:
        supply_demand_loop = "supply_demand_loop"
        activity_demand = "activity_demand"
        traffic_assignment = "traffic_assignment"

    class _FakeState:
        Stage = _FakeStage
        iteration = 0
        current_major_stage = _FakeStage.supply_demand_loop
        year = 2018
        forecast_year = 2018
        is_restart_run = False

        def should_run(self, major_stage, iteration=None, sub_stage=None):
            return major_stage == self.Stage.supply_demand_loop

        def complete_step(self, *_args, **_kwargs):
            return None

    context = WorkflowRuntimeContext.from_parts(
        settings=SimpleNamespace(
            run=SimpleNamespace(
                supply_demand_iters=1,
                models=SimpleNamespace(activity_demand="activitysim", travel="beam"),
            )
        ),
        state=_FakeState(),
        workspace=make_conforming_workspace(tmp_path),
    )
    events: list[str] = []
    monkeypatch.setattr(
        "pilates.workflows.orchestrator.run_activity_demand_phase",
        lambda **kwargs: (events.append("activitysim"), kwargs["sim_state"])[1],
    )
    monkeypatch.setattr(
        "pilates.workflows.orchestrator.run_traffic_assignment_phase",
        lambda **kwargs: (events.append("beam"), kwargs["sim_state"])[1],
    )

    run_supply_demand_stage(
        scenario=SimpleNamespace(),
        settings=context.settings,
        state=context.state,
        workspace=context.workspace,
        sim_state=SimState(),
        year=2018,
        on_activity_demand_boundary=lambda _iteration: events.append("snapshot"),
    )

    assert events[:3] == ["activitysim", "snapshot", "beam"]


def test_run_postprocessing_stage_uses_runtime_context(
    monkeypatch, tmp_path: Path
) -> None:
    settings = SimpleNamespace()
    state = SimpleNamespace(iteration=0)
    workspace = make_conforming_workspace(tmp_path)
    context = WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    captured = {}

    monkeypatch.setattr(
        "pilates.workflows.orchestrator.execute_step",
        lambda **kwargs: (
            captured.update(kwargs),
            (
                SimpleNamespace(outputs={}),
                {},
            ),
        )[1],
    )

    run_postprocessing_stage(
        scenario="scenario",
        settings=context.settings,
        state=context.state,
        workspace=context.workspace,
        sim_state=SimState(),
        year=2018,
    )

    assert captured["settings"] is settings
    assert captured["state"] is state
    assert captured["workspace"] is workspace


def test_conforming_workspace_and_context_factory(tmp_path: Path) -> None:
    """Validate that conforming factories produce strongly typed conforming instances."""
    settings = make_conforming_settings(tmp_path)
    workspace = make_conforming_workspace(
        tmp_path,
        settings=settings,
        beam_mutable_data_dir=tmp_path / "custom_beam",
    )
    assert isinstance(workspace, Workspace)
    assert workspace.full_path == str(tmp_path / "run")
    assert workspace.get_beam_mutable_data_dir() == str(tmp_path / "custom_beam")
    assert workspace.archive_run_dir is None
    assert workspace.expected_prefix_steps == 0
    assert workspace.on_archive_commit is None

    context = make_conforming_workflow_runtime_context(
        tmp_path,
        settings=settings,
        workspace=workspace,
    )
    assert isinstance(context, WorkflowRuntimeContext)
    assert context.workspace is workspace
    assert context.settings is settings
