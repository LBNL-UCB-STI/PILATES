from dataclasses import replace
from pathlib import Path
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from consist import (
    BindingResult,
    CacheOptions,
    ExecutionOptions,
    OutputArtifactSpec,
    OutputSet,
    RunResult,
    define_step,
)
from consist.core.tracker import Tracker
from consist.models.artifact import Artifact
from consist.models.run import Run

from pilates.workflows.binding import build_resolved_binding
from pilates.workflows.output_projection import require_output
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.step_definition import (
    ConfigContract,
    InputContract,
    StepDefinition,
)
from pilates.workflows.step_execution import execute_step
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.steps import urbansim_atlas
from pilates.workflows.steps.urbansim_atlas import URBANSIM_POSTPROCESS, URBANSIM_RUN


_EXAMPLE_INPUT_CONTRACT = InputContract(
    status="incomplete",
    reason="test-only executable has no closed workspace contract",
    config_contract=ConfigContract.payload(),
)


@pytest.mark.parametrize("cache_hit", [False, True], ids=["miss", "cache-hit"])
def test_execute_step_forwards_output_sets_and_projects_archived_outputs(
    monkeypatch, tmp_path: Path, cache_hit: bool
) -> None:
    """A completed child run publishes its archived artifacts before projection."""

    class Coupler:
        def __init__(self) -> None:
            self.values: dict[str, object] = {}

        def set_from_artifact(self, key: str, artifact: object) -> None:
            self.values[key] = artifact

    class Tracker:
        def __init__(self, archived_outputs: dict[str, object]) -> None:
            self.archived_outputs = archived_outputs
            self.archive_calls: list[tuple[str, str]] = []

        def archive_run_outputs(
            self, run_id: str, archive_root: str, *, mode: str
        ) -> SimpleNamespace:
            assert mode == "copy"
            self.archive_calls.append((run_id, archive_root))
            return SimpleNamespace(outputs=self.archived_outputs)

    @define_step(model="example", outputs=["bundle"])
    def example(*, settings, state, workspace) -> None:
        return None

    output_sets = {"bundle": OutputSet(root="bundle", include="*.csv")}
    run_id = "child-run-id"
    run = Run(
        id=run_id,
        model_name="example",
        config_hash=None,
        git_hash=None,
    )
    original_outputs = {
        "bundle": Artifact(
            key="bundle", container_uri="workspace://bundle", driver="other"
        )
    }
    archived_outputs = {
        "bundle": Artifact(
            key="bundle", container_uri="workspace://archived/bundle", driver="other"
        )
    }
    run_result = RunResult(run=run, outputs=original_outputs, cache_hit=cache_hit)
    coupler = Coupler()
    scenario = SimpleNamespace(coupler=coupler, tracker=Tracker(archived_outputs))
    seen: dict[str, object] = {}
    scenario.run = lambda **kwargs: seen.update(kwargs) or run_result
    definition = StepDefinition(
        name="example",
        function=example,
        resolve_inputs=lambda **_: ResolvedStepInputs(
            step_name="example", binding=BindingResult(inputs={})
        ),
        output_sets=lambda **_: output_sets,
        project_outputs=lambda outputs, **_: SimpleNamespace(
            received_outputs=outputs,
            coupler_bundle=coupler.values["bundle"],
        ),
        input_contract=_EXAMPLE_INPUT_CONTRACT,
    )
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")

    result, projected = execute_step(
        scenario=scenario,
        definition=definition,
        settings="settings",
        state="state",
        workspace="workspace",
        stage="stage",
        year=None,
        iteration=None,
        phase=None,
    )

    assert seen["output_sets"] == output_sets
    assert scenario.tracker.archive_calls == [
        (run_id, str(archive_root / "consist-recovery" / run_id))
    ]
    assert result.outputs == archived_outputs
    assert result.cache_hit is cache_hit
    assert projected.received_outputs == archived_outputs
    assert projected.coupler_bundle is archived_outputs["bundle"]


def test_execute_step_skips_archival_when_normalized_roots_are_equal(
    monkeypatch, tmp_path: Path
) -> None:
    class Tracker:
        def __init__(self) -> None:
            self.archive_calls: list[tuple[str, str]] = []

        def archive_run_outputs(
            self, run_id: str, archive_root: str, *, mode: str
        ) -> SimpleNamespace:
            self.archive_calls.append((run_id, archive_root))
            raise AssertionError("equal local and archive roots must not archive")

    @define_step(model="example", outputs=["result"])
    def example(*, settings, state, workspace) -> None:
        return None

    run = Run(
        id="same-root-run",
        model_name="example",
        config_hash=None,
        git_hash=None,
    )
    outputs = {
        "result": Artifact(
            key="result", container_uri="workspace://result", driver="other"
        )
    }
    result = RunResult(run=run, outputs=outputs)
    tracker = Tracker()
    scenario = SimpleNamespace(
        coupler=SimpleNamespace(update=lambda _: None), tracker=tracker
    )
    scenario.run = lambda **_: result
    definition = StepDefinition(
        name="example",
        function=example,
        resolve_inputs=lambda **_: ResolvedStepInputs(
            step_name="example", binding=BindingResult(inputs={})
        ),
        project_outputs=lambda received_outputs, **_: received_outputs,
        input_contract=_EXAMPLE_INPUT_CONTRACT,
    )
    shared_root = tmp_path / "shared"
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(shared_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(shared_root / "."))
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "true")

    received_result, projected = execute_step(
        scenario=scenario,
        definition=definition,
        settings="settings",
        state="state",
        workspace="workspace",
        stage="stage",
        year=None,
        iteration=None,
        phase=None,
    )

    assert tracker.archive_calls == []
    assert received_result is result
    assert projected == outputs


def test_execute_step_keeps_explicit_staging_outputs_out_of_recovery_archive(
    monkeypatch, tmp_path: Path
) -> None:
    """A staging step can retain its local handoff without archive-copying it."""

    class Tracker:
        def archive_run_outputs(self, *_args, **_kwargs) -> SimpleNamespace:
            raise AssertionError("staging outputs must not enter recovery archival")

    @define_step(model="example", outputs=["staging_root"])
    def example(*, settings, state, workspace) -> None:
        return None

    result = RunResult(
        run=Run(
            id="staging-run",
            model_name="example",
            config_hash=None,
            git_hash=None,
        ),
        outputs={
            "staging_root": Artifact(
                key="staging_root",
                container_uri="workspace://staging",
                driver="unknown",
            )
        },
    )
    scenario = SimpleNamespace(
        coupler=SimpleNamespace(update=lambda _: None),
        tracker=Tracker(),
    )
    scenario.run = lambda **_: result
    definition = StepDefinition(
        name="staging",
        function=example,
        resolve_inputs=lambda **_: ResolvedStepInputs(
            step_name="staging", binding=BindingResult(inputs={})
        ),
        project_outputs=lambda received_outputs, **_: received_outputs,
        input_contract=_EXAMPLE_INPUT_CONTRACT,
        archive_outputs=False,
    )
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(tmp_path / "archive"))
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")

    received_result, projected = execute_step(
        scenario=scenario,
        definition=definition,
        settings="settings",
        state="state",
        workspace="workspace",
        stage="stage",
        year=None,
        iteration=None,
        phase=None,
    )

    assert received_result is result
    assert projected == result.outputs


def test_execute_step_carries_report_only_contract_without_changing_options(
    monkeypatch, tmp_path: Path
) -> None:
    """Passive contracts preserve existing cache and execution option behavior."""

    @define_step(model="example", outputs=["result"])
    def example(*, settings, state, workspace) -> None:
        return None

    resolved = ResolvedStepInputs(step_name="example", binding=BindingResult(inputs={}))
    cache_options = CacheOptions(cache_hydration="outputs-requested")
    execution_options = ExecutionOptions(
        input_binding="paths",
        input_paths={"selected": "/workspace/selected"},
    )
    result = RunResult(
        run=Run(
            id="example-run",
            model_name="example",
            config_hash=None,
            git_hash=None,
        ),
        outputs={"result": Artifact(key="result", container_uri="workspace://result")},
    )
    seen: dict[str, object] = {}
    scenario = SimpleNamespace(
        coupler=SimpleNamespace(update=lambda _: None),
        tracker=SimpleNamespace(),
    )
    scenario.run = lambda **kwargs: seen.update(kwargs) or result
    definition = StepDefinition(
        name="example",
        function=example,
        resolve_inputs=lambda **_: resolved,
        project_outputs=lambda outputs, **_: outputs,
        execution_options=lambda **_: execution_options,
        cache_options=lambda **_: cache_options,
        input_contract=InputContract(
            status="incomplete",
            reason="example runner has not closed its workspace reads",
            config_contract=ConfigContract.payload(),
        ),
        archive_outputs=False,
    )
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "0")

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
    )

    supplied_options = seen["execution_options"]
    assert isinstance(supplied_options, ExecutionOptions)
    assert supplied_options.input_paths == {"selected": "/workspace/selected"}
    assert seen["cache_options"] == cache_options
    assert projected == result.outputs


def test_urbansim_run_archives_only_scalar_forecast_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The active UrbanSim run uses central archival without publishing staging."""

    class Coupler:
        def __init__(self) -> None:
            self.values: dict[str, object] = {}

        def set_from_artifact(self, key: str, artifact: object) -> None:
            self.values[key] = artifact

    class ArchiveTracker:
        def __init__(self, archived_outputs: dict[str, Artifact]) -> None:
            self.archived_outputs = archived_outputs
            self.archive_calls: list[tuple[str, str]] = []

        def archive_run_outputs(
            self, run_id: str, archive_root: str, *, mode: str
        ) -> SimpleNamespace:
            assert mode == "copy"
            self.archive_calls.append((run_id, archive_root))
            return SimpleNamespace(outputs=self.archived_outputs)

    local_root = tmp_path / "local"
    archive_root = tmp_path / "archive"
    workspace_root = tmp_path / "workspace"
    forecast_h5 = workspace_root / "urbansim" / "data" / "output_2030.h5"
    forecast_h5.parent.mkdir(parents=True)
    forecast_h5.write_bytes(b"forecast\n")
    input_h5 = tmp_path / "input_001.h5"
    input_h5.write_bytes(b"input\n")

    run_id = "urbansim-run-id"
    run = Run(
        id=run_id,
        model_name="urbansim_run",
        config_hash=None,
        git_hash=None,
    )
    outputs = {
        key: Artifact(key=key, container_uri=str(forecast_h5), driver="other")
        for key in (USIM_DATASTORE_H5, USIM_FORECAST_OUTPUT)
    }
    archived_outputs = {
        key: Artifact(
            key=key,
            container_uri=f"workspace://archived/{forecast_h5.name}",
            driver="other",
        )
        for key in outputs
    }
    coupler = Coupler()
    tracker = ArchiveTracker(archived_outputs)
    scenario = SimpleNamespace(coupler=coupler, tracker=tracker)
    scenario.run = lambda **_kwargs: RunResult(run=run, outputs=outputs)
    settings = SimpleNamespace(
        urbansim=SimpleNamespace(output_file_template="output_{year}.h5")
    )
    state = SimpleNamespace(forecast_year=2030)
    workspace = SimpleNamespace(
        full_path=str(workspace_root),
        get_usim_mutable_data_dir=lambda: str(forecast_h5.parent),
    )
    resolved = ResolvedStepInputs(
        step_name="urbansim_run",
        binding=BindingResult(inputs={USIM_DATASTORE_H5: input_h5}),
        required_roles=(USIM_DATASTORE_H5,),
        source_by_role={USIM_DATASTORE_H5: "coupler"},
        metadata={
            "urbansim_launch_context": urbansim_atlas.UrbanSimLaunchContext(
                mutable_data_dir=forecast_h5.parent,
                output_datastore=forecast_h5,
            )
        },
    )
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")

    result, projected = execute_step(
        scenario=scenario,
        definition=URBANSIM_RUN,
        settings=settings,
        state=state,
        workspace=workspace,
        stage="land_use",
        year=2030,
        iteration=0,
        phase="run",
        resolved_inputs=resolved,
    )

    assert URBANSIM_RUN.archive_outputs is True
    assert tracker.archive_calls == [
        (run_id, str(archive_root / "consist-recovery" / run_id))
    ]
    assert result.outputs == archived_outputs
    assert projected.usim_datastore_h5 == forecast_h5
    assert set(coupler.values) == {USIM_DATASTORE_H5, USIM_FORECAST_OUTPUT}
    assert "usim_mutable_data_dir" not in coupler.values


def test_execute_step_archives_and_hydrates_nested_output_set_from_recovery_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Central archival preserves nested scalar and directory OutputSet members."""

    local_root = tmp_path / "local"
    archive_root = tmp_path / "archive"
    tracker = Tracker(
        run_dir=local_root / "consist-runs",
        db_path=str(local_root / "provenance.duckdb"),
        hashing_strategy="full",
    )
    captured: dict[str, object] = {}
    original_archive = tracker.archive_run_outputs

    def capture_archive(*args: object, **kwargs: object) -> object:
        archived = original_archive(*args, **kwargs)
        captured["archived"] = archived
        return archived

    monkeypatch.setattr(tracker, "archive_run_outputs", capture_archive)
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")

    @define_step(model="archive_probe")
    def archive_probe(*, settings, state, workspace, ctx) -> None:
        del settings, state, workspace
        bundle_root = ctx.run_dir / "bundle"
        bundle_root.mkdir(parents=True)
        (bundle_root / "summary.csv").write_text("metric,value\ncount,1\n")
        zarr_root = bundle_root / "skims.zarr"
        (zarr_root / "nested").mkdir(parents=True)
        (zarr_root / ".zgroup").write_text("{}\n")
        (zarr_root / "nested" / "0.0").write_bytes(b"skim")

    definition = StepDefinition(
        name="archive_probe",
        function=archive_probe,
        resolve_inputs=lambda **_: ResolvedStepInputs(
            step_name="archive_probe", binding=BindingResult(inputs={})
        ),
        project_outputs=lambda outputs, **_: outputs,
        output_paths=lambda **_: {
            "summary": "bundle/summary.csv",
            "skims": OutputArtifactSpec(
                path="bundle/skims.zarr",
                meta={"directory_artifact": True},
            ),
        },
        output_sets=lambda **_: {
            "bundle": OutputSet(root="bundle", include="**/*", recursive=True)
        },
        execution_options=lambda **_: ExecutionOptions(inject_context="ctx"),
        input_contract=_EXAMPLE_INPUT_CONTRACT,
    )

    with tracker.scenario("archive-probe") as scenario:
        result, projected = execute_step(
            scenario=scenario,
            definition=definition,
            settings=object(),
            state=object(),
            workspace=object(),
            stage="test",
            year=2030,
            iteration=0,
            phase="run",
        )

    archived = captured["archived"]
    recovery_root = archive_root / "consist-recovery" / str(result.run.id)
    assert archived.paths["bundle"] == archived["bundle"]
    assert archived.paths["summary"] == archived["bundle"] / "summary.csv"
    assert archived.paths["skims"] == archived["bundle"] / "skims.zarr"
    assert archived.paths["bundle"].is_relative_to(recovery_root)
    assert (archived.paths["skims"] / "nested" / "0.0").read_bytes() == b"skim"
    assert projected == result.outputs == archived.outputs
    assert scenario.coupler.get("bundle") == archived.outputs["bundle"]
    assert all(
        artifact.recovery_roots == [str(recovery_root.resolve())]
        for artifact in archived.outputs.values()
    )

    parent = archived.outputs["bundle"]
    members = tracker.get_child_artifacts(parent)
    assert {member.meta["output_set_relative_path"] for member in members} == {
        "summary.csv",
        "skims.zarr/.zgroup",
        "skims.zarr/nested/0.0",
    }
    manifest = tracker.get_artifact(parent.meta["manifest_artifact_id"])
    assert manifest is not None
    assert all(
        artifact.recovery_roots == [str(recovery_root.resolve())]
        for artifact in [parent, *members, manifest]
    )

    shutil.rmtree(tracker.run_artifact_dir(result.run))
    assert not tracker.run_artifact_dir(result.run).exists()
    hydrated = tracker.hydrate_run_outputs(
        result.run.id,
        target_root=tracker.run_dir / "hydrated",
        on_missing="raise",
    )
    assert hydrated.outputs["bundle"].resolvable is True
    assert hydrated.outputs["bundle"].status == "materialized_from_filesystem"
    assert (hydrated.outputs["skims"].path / "nested" / "0.0").read_bytes() == b"skim"


def test_urbansim_population_snapshot_archives_from_managed_workspace_mount(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """UrbanSim postprocess publishes its immutable population snapshot centrally."""

    local_root = tmp_path / "local"
    archive_root = tmp_path / "archive"
    workspace_root = tmp_path / "workspace"
    datastore = tmp_path / "source" / "output_2030.h5"
    datastore.parent.mkdir(parents=True)
    with pd.HDFStore(datastore, mode="w") as store:
        for table in ("households", "persons", "jobs", "blocks"):
            store.put(table, pd.DataFrame({"value": [1]}))

    tracker = Tracker(
        run_dir=local_root / "consist-runs",
        db_path=str(local_root / "provenance.duckdb"),
        hashing_strategy="full",
        mounts={"workspace": str(workspace_root)},
    )
    with tracker.start_run("seed", "test"):
        artifact = tracker.log_artifact(
            datastore,
            key=USIM_DATASTORE_H5,
            direction="input",
        )

    settings = SimpleNamespace(
        run=SimpleNamespace(region="test"),
        urbansim=SimpleNamespace(
            command_template="urbansim {0}",
            input_file_template="input_{region_id}.h5",
            input_file_template_year=None,
            output_file_template="output_{year}.h5",
            region_id="001",
            region_mappings={"region_to_region_id": {"test": "001"}},
            admission=None,
        ),
    )
    state = SimpleNamespace(
        year=2030,
        current_year=2030,
        forecast_year=2030,
        set_sub_stage_progress=lambda _stage: None,
    )
    workspace = SimpleNamespace(
        full_path=str(workspace_root),
        get_usim_mutable_data_dir=lambda: str(workspace_root / "urbansim" / "data"),
    )

    class _Postprocessor:
        def __init__(self, *_args: object) -> None:
            pass

        def postprocess(self, *_args: object, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(urbansim_atlas, "UrbansimPostprocessor", _Postprocessor)
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    population_source = (
        workspace_root / "urbansim" / "data" / "output_2030_population_source.h5"
    )
    definition = replace(
        URBANSIM_POSTPROCESS,
        output_paths=lambda **_kwargs: {USIM_POPULATION_SOURCE_H5: population_source},
        project_outputs=lambda outputs, **_kwargs: outputs,
    )

    with tracker.scenario("urbansim") as scenario:
        scenario.coupler.set_from_artifact(USIM_DATASTORE_H5, artifact)
        result, _ = execute_step(
            scenario=scenario,
            definition=definition,
            settings=settings,
            state=state,
            workspace=workspace,
            stage="land_use",
            year=state.year,
            iteration=0,
            phase="postprocess",
        )

    source = Path(result.outputs[USIM_POPULATION_SOURCE_H5].path)
    assert source == population_source
    assert result.outputs[USIM_POPULATION_SOURCE_H5].recovery_roots == [
        str((archive_root / "consist-recovery" / str(result.run.id)).resolve())
    ]
    source.unlink()
    tracker.hydrate_run_outputs(
        result.run.id,
        target_root=workspace_root,
        keys=[USIM_POPULATION_SOURCE_H5],
        preserve_existing=False,
        on_missing="raise",
    )
    assert source.exists()


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
        input_contract=_EXAMPLE_INPUT_CONTRACT,
    )
    scenario = SimpleNamespace(coupler="coupler", tracker=None)

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
    assert calls[2][2].input_contract is _EXAMPLE_INPUT_CONTRACT


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
        input_contract=_EXAMPLE_INPUT_CONTRACT,
    )
    scenario = SimpleNamespace(coupler="coupler", tracker=None)
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


def test_execute_step_forwards_a_strict_binding_unchanged(tmp_path: Path) -> None:
    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        hashing_strategy="full",
    )
    source = tmp_path / "source.txt"
    source.write_text("selected\n", encoding="utf-8")
    with tracker.start_run("seed", "test"):
        artifact = tracker.log_artifact(source, key="payload", direction="input")

    @define_step(model="example", outputs=["result"])
    def example(payload: Path, *, settings, state, workspace) -> None:
        del payload, settings, state, workspace

    binding = build_resolved_binding(
        step_name="example",
        function=example,
        selected_artifacts={"payload": artifact},
        logical_destinations={"payload": Path("inputs/payload.txt")},
        selection_diagnostics={},
        step_identity=SimpleNamespace(
            name="example__y2030__i1__phase_test",
            step_contract_identity="sha256:step-v1:" + "0" * 64,
        ),
    )
    supplied = ResolvedStepInputs(step_name="example", binding=binding)
    definition = StepDefinition(
        name="example",
        function=example,
        resolve_inputs=lambda **_: (_ for _ in ()).throw(AssertionError("resolver")),
        project_outputs=lambda outputs, **_: outputs["result"],
        input_contract=_EXAMPLE_INPUT_CONTRACT,
    )
    seen: dict[str, object] = {}
    scenario = SimpleNamespace(coupler="coupler", tracker=None)
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

    assert seen["binding"] is binding


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
        input_contract=_EXAMPLE_INPUT_CONTRACT,
    )
    scenario = SimpleNamespace(coupler="coupler", tracker=None)
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
        input_contract=_EXAMPLE_INPUT_CONTRACT,
    )
    scenario = SimpleNamespace(coupler="coupler", tracker=None)
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
        input_contract=_EXAMPLE_INPUT_CONTRACT,
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
