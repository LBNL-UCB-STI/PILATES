from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from consist import ExecutionOptions, ResolvedBinding, Tracker, resolve_step_contract

from pilates.workflows.artifact_keys import (
    ATLAS_OUTPUT_DIR,
    ATLAS_VEHICLES2_OUTPUT,
    USIM_DATASTORE_H5,
    USIM_MUTABLE_DATA_DIR,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.atlas_state import AtlasSubState
from pilates.workflows.steps import urbansim_atlas
from pilates.urbansim.preprocessor import _stage_declared_urbansim_datastore
from pilates.workflows.steps.urbansim_atlas import (
    ATLAS_POSTPROCESS,
    ATLAS_PREPROCESS,
    ATLAS_RUN,
    URBANSIM_POSTPROCESS,
    URBANSIM_PREPROCESS,
    URBANSIM_RUN,
)
from pilates.workflows.step_execution import execute_step
from workflow_state import WorkflowState


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(region="test"),
        urbansim=SimpleNamespace(
            command_template="urbansim {0}",
            input_file_template="input_{region_id}.h5",
            input_file_template_year=None,
            output_file_template="output_{year}.h5",
            region_id="001",
            region_mappings={"region_to_region_id": {"test": "001"}},
        ),
        atlas=SimpleNamespace(),
    )


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        year=2030,
        current_year=2030,
        forecast_year=2030,
        start_year=2020,
        is_start_year=lambda: False,
    )


def _workspace(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        full_path=str(tmp_path),
        get_usim_mutable_data_dir=lambda: str(tmp_path / "urbansim" / "data"),
        get_atlas_mutable_input_dir=lambda: str(tmp_path / "atlas" / "input"),
        get_atlas_output_dir=lambda: str(tmp_path / "atlas" / "output"),
    )


def test_native_urbansim_atlas_definitions_resolve_to_canonical_contracts(
    tmp_path: Path,
) -> None:
    definitions = (
        URBANSIM_PREPROCESS,
        URBANSIM_RUN,
        URBANSIM_POSTPROCESS,
        ATLAS_PREPROCESS,
        ATLAS_RUN,
        ATLAS_POSTPROCESS,
    )
    settings = _settings()
    state = _state()
    workspace = _workspace(tmp_path)

    for definition in definitions:
        contract = resolve_step_contract(
            definition.function,
            year=2030,
            iteration=1,
            phase=definition.name.rsplit("_", 1)[1],
            runtime_kwargs={
                "settings": settings,
                "state": state,
                "workspace": workspace,
            },
        )

        assert contract.name.startswith(f"{definition.name}__")
        assert contract.model == definition.name
        assert contract.input_binding == "paths"
        assert contract.output_paths == definition.output_paths(
            settings=settings,
            state=state,
            workspace=workspace,
        )


def test_native_output_path_providers_preserve_typed_downstream_contracts(
    tmp_path: Path,
) -> None:
    settings = _settings()
    state = _state()
    workspace = _workspace(tmp_path)

    urbansim_preprocess_paths = URBANSIM_PREPROCESS.output_paths(
        settings=settings, state=state, workspace=workspace
    )
    urbansim_run_paths = URBANSIM_RUN.output_paths(
        settings=settings, state=state, workspace=workspace
    )
    atlas_postprocess_paths = ATLAS_POSTPROCESS.output_paths(
        settings=settings, state=state, workspace=workspace
    )

    assert set(urbansim_preprocess_paths) >= {
        USIM_MUTABLE_DATA_DIR,
        USIM_DATASTORE_H5,
    }
    assert set(urbansim_run_paths) == {USIM_DATASTORE_H5}
    assert set(atlas_postprocess_paths) >= {
        ATLAS_OUTPUT_DIR,
        USIM_POPULATION_SOURCE_H5,
        ATLAS_VEHICLES2_OUTPUT,
    }


def test_atlas_postprocess_output_uses_population_source_snapshot_at_start_year(
    tmp_path: Path,
) -> None:
    settings = _settings()
    state = _state()
    state.is_start_year = lambda: True
    workspace = _workspace(tmp_path)

    output_paths = ATLAS_POSTPROCESS.output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
    )

    assert Path(output_paths[USIM_POPULATION_SOURCE_H5]) == (
        Path(workspace.get_usim_mutable_data_dir()) / "output_2030_population_source.h5"
    )


def test_native_atlas_postprocess_publishes_exact_year_population_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    state = _state()
    settings = _settings()
    datastore = Path(workspace.get_usim_mutable_data_dir()) / "output_2030.h5"
    datastore.parent.mkdir(parents=True)
    with pd.HDFStore(datastore, mode="w") as store:
        for table_name in ("households", "persons", "jobs", "blocks"):
            store.put(f"/{table_name}", pd.DataFrame({"value": [1]}))

    class _AtlasPostprocessor:
        def __init__(self, *_args: object) -> None:
            pass

        def postprocess(
            self, _outputs: object, _workspace: object, **_kwargs: object
        ) -> None:
            pass

    monkeypatch.setattr(urbansim_atlas, "AtlasPostprocessor", _AtlasPostprocessor)

    urbansim_atlas._native_atlas_postprocess(
        Path(workspace.get_atlas_output_dir()),
        datastore,
        settings=settings,
        state=state,
        workspace=workspace,
    )

    population_source = (
        Path(workspace.get_usim_mutable_data_dir()) / "output_2030_population_source.h5"
    )
    assert population_source.exists()
    with pd.HDFStore(population_source, mode="r") as store:
        for table_name in ("households", "persons", "jobs", "blocks"):
            assert f"/2030/{table_name}" in store


def test_atlas_interval_start_postprocess_materializes_bound_h5_at_subyear_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tracker_root = tmp_path / "consist-runs"
    tracker = Tracker(
        run_dir=tracker_root,
        db_path=str(tmp_path / "provenance.duckdb"),
        hashing_strategy="full",
    )
    source_h5 = tracker_root / "fixtures" / "bound.h5"
    source_h5.parent.mkdir(parents=True)
    with pd.HDFStore(source_h5, mode="w") as store:
        for table_name in ("households", "persons", "jobs", "blocks"):
            store.put(f"/{table_name}", pd.DataFrame({"value": [1]}))
    source_atlas_output = tracker_root / "fixtures" / "atlas-output"
    source_atlas_output.mkdir()
    with tracker.start_run("seed", "test"):
        h5_artifact = tracker.log_artifact(
            source_h5,
            key=USIM_DATASTORE_H5,
            direction="input",
        )
        atlas_output_artifact = tracker.log_artifact(
            source_atlas_output,
            key=ATLAS_OUTPUT_DIR,
            direction="input",
        )

    settings = _settings()
    parent_state = SimpleNamespace(
        year=2023,
        current_year=2023,
        forecast_year=2029,
        start_year=2017,
        full_settings=settings,
        set_sub_stage_progress=lambda _value: None,
    )
    state = AtlasSubState(cast(WorkflowState, parent_state), 2023)
    workspace = _workspace(tracker_root / "workspace")
    output_h5 = Path(workspace.get_usim_mutable_data_dir()) / "output_2023.h5"
    population_h5 = (
        Path(workspace.get_usim_mutable_data_dir()) / "output_2023_population_source.h5"
    )
    stale_input = Path(workspace.get_usim_mutable_data_dir()) / "input_001.h5"
    output_h5.parent.mkdir(parents=True)
    output_h5.write_bytes(b"stale output")
    stale_input.write_bytes(b"stale input")
    vehicles2 = Path(workspace.get_atlas_output_dir()) / "vehicles2_2023.csv"
    seen: dict[str, Path | bytes] = {}

    class _Postprocessor:
        def __init__(self, *_args: object) -> None:
            pass

        def postprocess(
            self,
            _outputs: object,
            _workspace: object,
            *,
            usim_datastore_h5: Path,
        ) -> None:
            seen["path"] = usim_datastore_h5
            with pd.HDFStore(usim_datastore_h5, mode="w") as store:
                for table_name in ("households", "persons", "jobs", "blocks"):
                    store.put(f"/{table_name}", pd.DataFrame({"value": [2]}))
            vehicles2.parent.mkdir(parents=True, exist_ok=True)
            vehicles2.write_text("vehicle_id\n1\n", encoding="utf-8")

    monkeypatch.setattr(urbansim_atlas, "AtlasPostprocessor", _Postprocessor)
    definition = replace(
        ATLAS_POSTPROCESS,
        output_paths=lambda **_kwargs: {
            USIM_POPULATION_SOURCE_H5: population_h5,
            ATLAS_VEHICLES2_OUTPUT: vehicles2,
        },
        project_outputs=lambda outputs, **_kwargs: outputs,
    )

    with tracker.scenario("atlas-interval-start") as scenario:
        scenario.coupler.set_from_artifact(USIM_DATASTORE_H5, h5_artifact)
        scenario.coupler.set_from_artifact(ATLAS_OUTPUT_DIR, atlas_output_artifact)
        result, projected = execute_step(
            scenario=scenario,
            definition=definition,
            settings=settings,
            state=state,
            workspace=workspace,
            stage="vehicle_ownership",
            year=state.year,
            iteration=1,
            phase="postprocess",
        )

    assert seen == {"path": output_h5}
    with pd.HDFStore(output_h5, mode="r") as store:
        assert store["/households"].iloc[0, 0] == 2
    assert stale_input.read_bytes() == b"stale input"
    assert result.output_path(USIM_POPULATION_SOURCE_H5) == population_h5
    with pd.HDFStore(population_h5, mode="r") as store:
        assert "/2023/households" in store
    assert projected == result.outputs


def test_urbansim_preprocess_stages_the_bound_datastore_over_workspace_state(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    bound_snapshot = tmp_path / "consist-snapshot" / "input_001.h5"
    bound_snapshot.parent.mkdir()
    bound_snapshot.write_bytes(b"bound snapshot")
    stale_workspace_copy = Path(workspace.get_usim_mutable_data_dir()) / "input_001.h5"
    stale_workspace_copy.parent.mkdir(parents=True)
    stale_workspace_copy.write_bytes(b"stale workspace copy")

    staged = _stage_declared_urbansim_datastore(
        settings=_settings(),
        workspace=workspace,
        usim_datastore_h5=bound_snapshot,
    )

    assert staged == stale_workspace_copy
    assert staged.read_bytes() == b"bound snapshot"


def test_native_resolver_selects_one_coupler_artifact_at_its_model_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "urbansim-data"
    source.parent.mkdir()
    source.write_text("h5 bytes\n", encoding="utf-8")

    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        hashing_strategy="full",
    )
    with tracker.start_run("seed", "test"):
        artifact = tracker.log_artifact(
            source,
            key=USIM_DATASTORE_H5,
            direction="input",
        )

    class _Coupler:
        def get(self, key: str):
            return {USIM_DATASTORE_H5: artifact}.get(key)

    workspace = _workspace(tmp_path)
    settings = _settings()
    state = _state()
    with tracker.scenario("urbansim") as scenario:
        identity = scenario.resolve_step_identity(
            URBANSIM_POSTPROCESS.function,
            year=state.year,
            iteration=1,
            phase="postprocess",
            stage="land_use",
            execution_options=ExecutionOptions(
                input_binding="paths",
                runtime_kwargs={
                    "settings": settings,
                    "state": state,
                    "workspace": workspace,
                },
            ),
        )
        resolved = URBANSIM_POSTPROCESS.resolve_inputs(
            settings=settings,
            state=state,
            workspace=workspace,
            coupler=_Coupler(),
            step_identity=identity,
        )

    assert isinstance(resolved.binding, ResolvedBinding)
    assert resolved.binding.step_name == identity.name
    assert resolved.binding.step_contract_identity == identity.step_contract_identity
    assert (
        resolved.binding.inputs[USIM_DATASTORE_H5].artifact.artifact_id == artifact.id
    )
    assert resolved.binding.inputs[USIM_DATASTORE_H5].destination == Path(
        "inputs/usim_datastore_h5"
    )
    assert resolved.source_by_role == {USIM_DATASTORE_H5: "coupler"}
    assert resolved.logical_destinations == {
        USIM_DATASTORE_H5: Path(workspace.get_usim_mutable_data_dir())
        / "output_2030.h5"
    }


def test_urbansim_postprocess_consumes_the_strict_snapshot_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source" / "urbansim-data.h5"
    source.parent.mkdir()
    source.write_text("tracked h5 bytes\n", encoding="utf-8")
    output = tmp_path / "workspace" / "outputs" / "processed.h5"
    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        hashing_strategy="full",
    )
    with tracker.start_run("seed", "test"):
        artifact = tracker.log_artifact(
            source,
            key=USIM_DATASTORE_H5,
            direction="input",
        )

    seen: dict[str, Path] = {}

    class _Postprocessor:
        def __init__(self, *_args: object) -> None:
            pass

        def postprocess(self, outputs: object, _workspace: object) -> None:
            snapshot = outputs.usim_datastore_h5
            seen["snapshot"] = snapshot
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(snapshot.read_bytes())

    monkeypatch.setattr(urbansim_atlas, "UrbansimPostprocessor", _Postprocessor)
    definition = replace(
        URBANSIM_POSTPROCESS,
        output_paths=lambda **_kwargs: {USIM_DATASTORE_H5: output},
        project_outputs=lambda outputs, **_kwargs: outputs,
    )
    settings = _settings()
    state = _state()
    workspace = _workspace(tmp_path / "workspace")

    with tracker.scenario("urbansim") as scenario:
        scenario.coupler.set_from_artifact(USIM_DATASTORE_H5, artifact)
        result, projected = execute_step(
            scenario=scenario,
            definition=definition,
            settings=settings,
            state=state,
            workspace=workspace,
            stage="land_use",
            year=state.year,
            iteration=1,
            phase="postprocess",
        )

    snapshot = seen["snapshot"]
    assert snapshot != source
    assert snapshot.read_text(encoding="utf-8") == "tracked h5 bytes\n"
    assert snapshot.is_relative_to(tmp_path / "consist-runs" / ".resolved-bindings")
    assert output.read_bytes() == source.read_bytes()
    assert projected == result.outputs


def test_each_eligible_urbansim_atlas_resolver_freezes_preflight_identity(
    tmp_path: Path,
) -> None:
    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        hashing_strategy="full",
    )
    roles = (USIM_DATASTORE_H5, "atlas_mutable_input_dir", ATLAS_OUTPUT_DIR)
    artifacts = {}
    with tracker.start_run("seed", "test"):
        for role in roles:
            source = tmp_path / "source" / role
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"{role}\n", encoding="utf-8")
            artifacts[role] = tracker.log_artifact(
                source,
                key=role,
                direction="input",
            )

    class _Coupler:
        def get(self, key: str):
            return artifacts.get(key)

    settings = _settings()
    state = _state()
    workspace = _workspace(tmp_path)
    eligible_definitions = (URBANSIM_POSTPROCESS,)

    with tracker.scenario("urbansim-atlas") as scenario:
        for definition in eligible_definitions:
            assert definition.preflight_identity is True
            identity = scenario.resolve_step_identity(
                definition.function,
                year=state.year,
                iteration=1,
                phase=definition.name.rsplit("_", 1)[1],
                stage="land_use",
                execution_options=ExecutionOptions(
                    input_binding="paths",
                    runtime_kwargs={
                        "settings": settings,
                        "state": state,
                        "workspace": workspace,
                    },
                ),
            )
            resolved = definition.resolve_inputs(
                settings=settings,
                state=state,
                workspace=workspace,
                coupler=_Coupler(),
                step_identity=identity,
            )

            assert isinstance(resolved.binding, ResolvedBinding)
            assert resolved.binding.step_name == identity.name
            assert (
                resolved.binding.step_contract_identity
                == identity.step_contract_identity
            )
            assert all(
                not input.destination.is_absolute()
                for input in resolved.binding.inputs.values()
                if input.destination is not None
            )

    assert URBANSIM_RUN.preflight_identity is False


def test_native_callables_forward_resolved_paths_to_model_adapters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    state = _state()
    settings = _settings()
    snapshots = {
        "usim_datastore_h5": tmp_path / "snapshots" / "usim.h5",
        "atlas_mutable_input_dir": tmp_path / "snapshots" / "atlas-input",
        ATLAS_OUTPUT_DIR: tmp_path / "snapshots" / "atlas-output",
    }
    snapshots["usim_datastore_h5"].parent.mkdir(parents=True)
    with pd.HDFStore(snapshots["usim_datastore_h5"], mode="w") as store:
        for table_name in ("households", "persons", "jobs", "blocks"):
            store.put(f"/{table_name}", pd.DataFrame({"value": [1]}))
    calls: dict[str, object] = {}

    class _UrbanSimPreprocessor:
        def __init__(self, *_args: object) -> None:
            pass

        def preprocess(self, _workspace: object, **kwargs: object) -> None:
            calls["urbansim_preprocess"] = kwargs

    class _AtlasPreprocessor:
        def __init__(self, *_args: object) -> None:
            pass

        def preprocess(self, _workspace: object, **kwargs: object) -> None:
            calls["atlas_preprocess"] = kwargs

    class _AtlasRunner:
        def __init__(self, *_args: object) -> None:
            pass

        def run(self, inputs: object, _workspace: object) -> None:
            calls["atlas_run"] = inputs

    class _AtlasPostprocessor:
        def __init__(self, *_args: object) -> None:
            pass

        def postprocess(
            self, _outputs: object, _workspace: object, **kwargs: object
        ) -> None:
            calls["atlas_postprocess"] = kwargs

    monkeypatch.setattr(urbansim_atlas, "UrbansimPreprocessor", _UrbanSimPreprocessor)
    monkeypatch.setattr(urbansim_atlas, "AtlasPreprocessor", _AtlasPreprocessor)
    monkeypatch.setattr(urbansim_atlas, "AtlasRunner", _AtlasRunner)
    monkeypatch.setattr(urbansim_atlas, "AtlasPostprocessor", _AtlasPostprocessor)

    urbansim_atlas._native_urbansim_preprocess(
        snapshots["usim_datastore_h5"],
        settings=settings,
        state=state,
        workspace=workspace,
    )
    urbansim_atlas._native_atlas_preprocess(
        snapshots["usim_datastore_h5"],
        settings=settings,
        state=state,
        workspace=workspace,
    )
    urbansim_atlas._native_atlas_run(
        snapshots["atlas_mutable_input_dir"],
        settings=settings,
        state=state,
        workspace=workspace,
    )
    urbansim_atlas._native_atlas_postprocess(
        snapshots[ATLAS_OUTPUT_DIR],
        snapshots["usim_datastore_h5"],
        settings=settings,
        state=state,
        workspace=workspace,
    )

    # The bound datastore path is an execution input, not a hint for an
    # adapter-owned workspace lookup.
    assert calls["urbansim_preprocess"] == {
        "usim_datastore_h5": snapshots["usim_datastore_h5"],
        "final_skims_omx": None,
        "allow_workspace_skim_fallback": True,
    }
    assert calls["atlas_preprocess"] == {
        "usim_datastore_h5": snapshots["usim_datastore_h5"],
        "final_skims_omx": None,
        "allow_workspace_skim_fallback": True,
    }
    assert (
        getattr(calls["atlas_run"], "atlas_mutable_input_dir")
        == snapshots["atlas_mutable_input_dir"]
    )
    assert calls["atlas_postprocess"] == {
        "usim_datastore_h5": snapshots["usim_datastore_h5"]
    }


def test_native_atlas_postprocess_projector_uses_persisted_outputs_only(
    monkeypatch, tmp_path: Path
) -> None:
    atlas_output_dir = tmp_path / "atlas-output"
    atlas_output_dir.mkdir()
    usim_h5 = tmp_path / "population.h5"
    usim_h5.touch()
    vehicles2 = atlas_output_dir / "vehicles2_2030.csv"
    vehicles2.write_text("vehicle_id\n1\n", encoding="utf-8")
    declared_outputs = {
        ATLAS_OUTPUT_DIR: atlas_output_dir,
        USIM_POPULATION_SOURCE_H5: usim_h5,
        ATLAS_VEHICLES2_OUTPUT: vehicles2,
    }
    monkeypatch.setattr(
        urbansim_atlas,
        "_atlas_postprocess_native_output_paths",
        lambda **_kwargs: declared_outputs,
    )

    projected = ATLAS_POSTPROCESS.project_outputs(
        {
            ATLAS_OUTPUT_DIR: SimpleNamespace(path=atlas_output_dir),
            USIM_POPULATION_SOURCE_H5: SimpleNamespace(path=usim_h5),
            ATLAS_VEHICLES2_OUTPUT: SimpleNamespace(path=vehicles2),
        },
        settings=_settings(),
        state=_state(),
        workspace=_workspace(tmp_path),
    )

    assert projected.atlas_output_dir == atlas_output_dir
    assert projected.usim_datastore_h5 == usim_h5
    assert projected.processed_outputs == {ATLAS_VEHICLES2_OUTPUT: vehicles2}


def test_native_atlas_projector_rejects_source_mount_when_destination_missing(
    monkeypatch, tmp_path: Path
) -> None:
    atlas_output_dir = tmp_path / "current" / "atlas-output"
    atlas_output_dir.mkdir(parents=True)
    usim_h5 = tmp_path / "current" / "population.h5"
    usim_h5.touch()
    declared_vehicles2 = atlas_output_dir / "vehicles2_2030.csv"
    source_mount = tmp_path / "source-mount" / "vehicles2_2030.csv"
    source_mount.parent.mkdir()
    source_mount.write_text("historical vehicle output\n", encoding="utf-8")
    declared_outputs = {
        ATLAS_OUTPUT_DIR: atlas_output_dir,
        USIM_POPULATION_SOURCE_H5: usim_h5,
        ATLAS_VEHICLES2_OUTPUT: declared_vehicles2,
    }
    monkeypatch.setattr(
        urbansim_atlas,
        "_atlas_postprocess_native_output_paths",
        lambda **_kwargs: declared_outputs,
    )

    with pytest.raises(
        RuntimeError,
        match="atlas_postprocess output 'atlas_vehicles2_output' is missing at declared destination",
    ):
        ATLAS_POSTPROCESS.project_outputs(
            {
                ATLAS_OUTPUT_DIR: SimpleNamespace(path=atlas_output_dir),
                USIM_POPULATION_SOURCE_H5: SimpleNamespace(path=usim_h5),
                ATLAS_VEHICLES2_OUTPUT: SimpleNamespace(
                    path=source_mount,
                    container_uri="workspace://current/atlas-output/vehicles2_2030.csv",
                ),
            },
            settings=_settings(),
            state=_state(),
            workspace=_workspace(tmp_path),
        )
