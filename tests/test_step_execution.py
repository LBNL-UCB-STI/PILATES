from pathlib import Path
from types import SimpleNamespace

from consist import BindingResult, ExecutionOptions, define_step
from consist.core.tracker import Tracker

from pilates.workflows.output_projection import require_output
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.step_definition import StepDefinition
from pilates.workflows.step_execution import execute_step


def test_execute_step_resolves_once_runs_once_and_projects_run_outputs() -> None:
    calls: list[object] = []
    binding = BindingResult(inputs={"source": "selected"})
    resolved = ResolvedStepInputs(step_name="example", binding=binding)

    @define_step(model="example", outputs=["result"])
    def example(*, settings, state, workspace) -> None:
        return None

    def resolve_inputs(**kwargs):
        calls.append(("resolve", kwargs["coupler"]))
        return resolved

    def project(outputs, *, settings, state, workspace, resolved_inputs):
        calls.append(("project", outputs, resolved_inputs))
        return outputs["result"]

    definition = StepDefinition(
        name="example",
        function=example,
        resolve_inputs=resolve_inputs,
        project_outputs=project,
        execution_options=lambda **_: ExecutionOptions(
            input_binding="paths",
            input_paths={"source": "/workspace/selected"},
            input_materialization="requested",
            input_materialization_mode="copy",
            runtime_kwargs={"provider_value": "preserved"},
            load_inputs=True,
        ),
    )
    scenario = SimpleNamespace(coupler="coupler")

    def run(**kwargs):
        calls.append(("run", kwargs))
        return SimpleNamespace(outputs={"result": "persisted"})

    scenario.run = run
    _, projected = execute_step(
        scenario=scenario,
        definition=definition,
        settings="settings",
        state="state",
        workspace="workspace",
        stage="stage",
        year=2035,
        iteration=1,
        phase="run",
        runtime_kwargs={"extra": "value"},
    )

    assert projected == "persisted"
    assert [call[0] for call in calls] == ["resolve", "run", "project"]
    run_kwargs = calls[1][1]
    assert run_kwargs["binding"] is resolved.binding
    assert run_kwargs["execution_options"].input_paths == {
        "source": "/workspace/selected"
    }
    assert run_kwargs["execution_options"].load_inputs is True
    assert run_kwargs["execution_options"].runtime_kwargs == {
        "settings": "settings",
        "state": "state",
        "workspace": "workspace",
        "extra": "value",
        "provider_value": "preserved",
    }


def test_execute_step_uses_supplied_resolved_inputs_without_resolving() -> None:
    binding = BindingResult(inputs={})
    supplied = ResolvedStepInputs(step_name="example", binding=binding)

    @define_step(model="example", outputs=["result"])
    def example(*, settings, state, workspace) -> None:
        return None

    definition = StepDefinition(
        name="example",
        function=example,
        resolve_inputs=lambda **_: (_ for _ in ()).throw(AssertionError("resolver")),
        project_outputs=lambda outputs, **_: outputs["result"],
    )
    scenario = SimpleNamespace(coupler="coupler")
    scenario.run = lambda **_: SimpleNamespace(outputs={"result": "persisted"})

    _, projected = execute_step(
        scenario=scenario,
        definition=definition,
        settings="settings",
        state="state",
        workspace="workspace",
        stage="stage",
        year=None,
        iteration=None,
        phase=None,
        resolved_inputs=supplied,
    )

    assert projected == "persisted"


def test_execute_step_passes_resolved_inputs_to_output_path_provider() -> None:
    supplied = ResolvedStepInputs(step_name="example", binding=BindingResult(inputs={}))

    @define_step(model="example", outputs=["result"])
    def example(*, settings, state, workspace) -> None:
        return None

    definition = StepDefinition(
        name="example",
        function=example,
        resolve_inputs=lambda **_: supplied,
        project_outputs=lambda outputs, **_: outputs["result"],
        output_paths=lambda *, resolved_inputs, **_: {
            "result": f"/outputs/{resolved_inputs.step_name}"
        },
    )
    scenario = SimpleNamespace(coupler="coupler")
    seen: dict[str, object] = {}
    scenario.run = lambda **kwargs: (
        seen.update(kwargs) or SimpleNamespace(outputs={"result": "persisted"})
    )

    execute_step(
        scenario=scenario,
        definition=definition,
        settings="settings",
        state="state",
        workspace="workspace",
        stage="stage",
        year=None,
        iteration=None,
        phase=None,
        resolved_inputs=supplied,
    )

    assert seen["output_paths"] == {"result": "/outputs/example"}


def test_execute_step_projects_persisted_outputs_identically_for_miss_and_hit() -> None:
    binding = BindingResult(inputs={})
    supplied = ResolvedStepInputs(step_name="example", binding=binding)

    @define_step(model="example", outputs=["result"])
    def example(*, settings, state, workspace) -> None:
        return None

    projector_inputs: list[dict[str, str]] = []
    definition = StepDefinition(
        name="example",
        function=example,
        resolve_inputs=lambda **_: supplied,
        project_outputs=lambda outputs, **_: (
            projector_inputs.append(outputs)
            or require_output(outputs, step_name="example", key="result")
        ),
    )
    scenario = SimpleNamespace(coupler="coupler")
    scenario.run = lambda **_: SimpleNamespace(
        outputs={"result": "persisted"}, cache_hit=False
    )
    _, miss_output = execute_step(
        scenario=scenario,
        definition=definition,
        settings="settings",
        state="state",
        workspace="workspace",
        stage="stage",
        year=None,
        iteration=None,
        phase=None,
        resolved_inputs=supplied,
    )
    scenario.run = lambda **_: SimpleNamespace(
        outputs={"result": "persisted"}, cache_hit=True
    )
    _, hit_output = execute_step(
        scenario=scenario,
        definition=definition,
        settings="settings",
        state="state",
        workspace="workspace",
        stage="stage",
        year=None,
        iteration=None,
        phase=None,
        resolved_inputs=supplied,
    )

    assert miss_output == hit_output == "persisted"
    assert projector_inputs == [{"result": "persisted"}, {"result": "persisted"}]


def test_require_output_reports_step_missing_key_and_available_keys() -> None:
    try:
        require_output({"available": "artifact"}, step_name="example", key="result")
    except RuntimeError as error:
        message = str(error)
    else:  # pragma: no cover - establishes the helper's required failure mode.
        raise AssertionError("require_output accepted a missing output")

    assert "example" in message
    assert "result" in message
    assert "available" in message


def test_execute_step_evaluates_dynamic_metadata_once(tmp_path: Path) -> None:
    metadata_calls: list[tuple[int | None, object]] = []

    def dynamic_model(context):
        metadata_calls.append((context.year, context.runtime_settings))
        return "example"

    @define_step(model=dynamic_model)
    def example(*, settings, state, workspace) -> None:
        return None

    definition = StepDefinition(
        name="example",
        function=example,
        resolve_inputs=lambda **_: ResolvedStepInputs(
            step_name="example", binding=BindingResult(inputs={})
        ),
        project_outputs=lambda outputs, **_: outputs,
    )
    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )

    with tracker.scenario("metadata-once") as scenario:
        _, projected = execute_step(
            scenario=scenario,
            definition=definition,
            settings="settings",
            state="state",
            workspace="workspace",
            stage="stage",
            year=2035,
            iteration=1,
            phase="run",
        )

    assert projected == {}
    assert metadata_calls == [(2035, "settings")]
