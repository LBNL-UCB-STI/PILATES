from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
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
from pilates.workflows import step_consist_meta
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
        atlas=SimpleNamespace(beamac=0),
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
    atlas_preprocess_paths = ATLAS_PREPROCESS.output_paths(
        settings=settings, state=state, workspace=workspace
    )
    atlas_postprocess_paths = ATLAS_POSTPROCESS.output_paths(
        settings=settings, state=state, workspace=workspace
    )

    assert set(urbansim_preprocess_paths) >= {
        USIM_MUTABLE_DATA_DIR,
        USIM_DATASTORE_H5,
    }
    assert URBANSIM_PREPROCESS.archive_outputs is False
    assert set(urbansim_run_paths) == {USIM_DATASTORE_H5}
    assert set(atlas_postprocess_paths) == {
        USIM_POPULATION_SOURCE_H5,
        ATLAS_VEHICLES2_OUTPUT,
    }
    assert set(atlas_preprocess_paths) == {
        "atlas_households_csv",
        "atlas_blocks_csv",
        "atlas_persons_csv",
        "atlas_residential_csv",
        "atlas_jobs_csv",
        "atlas_grave_csv",
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


def test_urbansim_postprocess_declares_population_source_snapshot(
    tmp_path: Path,
) -> None:
    outputs = urbansim_atlas._urbansim_postprocess_native_output_paths(
        settings=_settings(), state=_state(), workspace=_workspace(tmp_path)
    )

    assert outputs[USIM_POPULATION_SOURCE_H5] == (
        tmp_path / "urbansim" / "data" / "output_2030_population_source.h5"
    )


def test_urbansim_postprocess_rejects_snapshot_missing_required_root_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An incomplete source H5 must not be published as a population snapshot."""

    workspace = _workspace(tmp_path)
    datastore = Path(workspace.get_usim_mutable_data_dir()) / "output_2030.h5"
    datastore.parent.mkdir(parents=True)
    with pd.HDFStore(datastore, mode="w") as store:
        for table_name in ("households", "persons", "jobs"):
            store.put(f"/{table_name}", pd.DataFrame({"value": [1]}))

    class _Postprocessor:
        def __init__(self, *_args: object) -> None:
            pass

        def postprocess(self, *_args: object, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(urbansim_atlas, "UrbansimPostprocessor", _Postprocessor)

    with pytest.raises(RuntimeError, match="missing root tables"):
        urbansim_atlas._native_urbansim_postprocess(
            datastore,
            settings=_settings(),
            state=_state(),
            workspace=workspace,
        )


def test_atlas_preprocess_declares_explicit_prepared_csv_outputs(
    tmp_path: Path,
) -> None:
    settings = _settings()
    state = _state()
    workspace = _workspace(tmp_path)

    assert ATLAS_PREPROCESS.output_sets is None
    assert set(
        ATLAS_PREPROCESS.output_paths(
            settings=settings, state=state, workspace=workspace
        )
    ) == {
        "atlas_households_csv",
        "atlas_blocks_csv",
        "atlas_persons_csv",
        "atlas_residential_csv",
        "atlas_jobs_csv",
        "atlas_grave_csv",
    }


def test_atlas_preprocess_includes_accessibility_output_when_beamac_enabled(
    tmp_path: Path,
) -> None:
    settings = _settings()
    settings.atlas.beamac = 1

    output_paths = ATLAS_PREPROCESS.output_paths(
        settings=settings, state=_state(), workspace=_workspace(tmp_path)
    )

    assert output_paths["atlas_accessibility_csv"] == (
        tmp_path / "atlas" / "input" / "year2030" / "accessibility_2030_tract.csv"
    )


def test_atlas_run_declares_scalar_raw_outputs_and_one_continuation_set(
    tmp_path: Path,
) -> None:
    settings = _settings()
    state = _state()
    workspace = _workspace(tmp_path)

    assert set(
        ATLAS_RUN.output_paths(settings=settings, state=state, workspace=workspace)
    ) == {"householdv_2030", "vehicles_2030"}
    assert set(
        ATLAS_RUN.output_sets(settings=settings, state=state, workspace=workspace)
    ) == {"atlas_continuation_2030"}


def test_atlas_prepared_csvs_and_continuation_set_archive_and_hydrate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Central archival restores ATLAS prepared scalars and continuation as declared."""

    local_root = tmp_path / "local"
    archive_root = tmp_path / "archive"
    workspace_root = tmp_path / "workspace"
    workspace = _workspace(workspace_root)
    settings = _settings()
    state = _state()
    tracker = Tracker(
        run_dir=local_root / "consist-runs",
        db_path=str(local_root / "provenance.duckdb"),
        hashing_strategy="full",
        mounts={"workspace": workspace.full_path},
    )
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")

    source_h5 = tmp_path / "source" / "output_2030.h5"
    source_h5.parent.mkdir(parents=True)
    source_h5.write_bytes(b"UrbanSim source\n")

    class _Preprocessor:
        def __init__(self, *_args: object) -> None:
            pass

        @staticmethod
        def expected_inputs(
            _settings: object, _state: object, run_workspace: object
        ) -> dict[str, Path]:
            return {
                USIM_DATASTORE_H5: Path(run_workspace.get_usim_mutable_data_dir())
                / "output_2030.h5"
            }

        def preprocess(self, run_workspace: object, **_kwargs: object) -> None:
            year_root = Path(run_workspace.get_atlas_mutable_input_dir()) / "year2030"
            year_root.mkdir(parents=True)
            for filename in (
                "households.csv",
                "blocks.csv",
                "persons.csv",
                "residential.csv",
                "jobs.csv",
                "grave.csv",
            ):
                (year_root / filename).write_text("id\n1\n", encoding="utf-8")

    class _Runner:
        def __init__(self, *_args: object) -> None:
            pass

        def run(self, _inputs: object, run_workspace: object) -> None:
            output_root = Path(run_workspace.get_atlas_output_dir())
            output_root.mkdir(parents=True)
            (output_root / "householdv_2030.csv").write_text("household_id\n1\n")
            (output_root / "vehicles_2030.csv").write_text("vehicle_id\n1\n")
            year_root = Path(run_workspace.get_atlas_mutable_input_dir()) / "year2030"
            (year_root / "vehicles_output.RData").write_text("vehicles\n")
            (year_root / "households_output.RData").write_text("households\n")

    monkeypatch.setattr(urbansim_atlas, "AtlasPreprocessor", _Preprocessor)
    monkeypatch.setattr(urbansim_atlas, "AtlasRunner", _Runner)

    with tracker.scenario("atlas-output-recovery") as scenario:
        with tracker.start_run("seed-usim", "test"):
            source_artifact = tracker.log_artifact(
                source_h5, key=USIM_DATASTORE_H5, direction="input"
            )
        scenario.coupler.set_from_artifact(USIM_DATASTORE_H5, source_artifact)
        prepared_result, _ = execute_step(
            scenario=scenario,
            definition=ATLAS_PREPROCESS,
            settings=settings,
            state=state,
            workspace=workspace,
            stage="vehicle_ownership",
            year=2030,
            iteration=1,
            phase="preprocess",
        )
        continuation_result, _ = execute_step(
            scenario=scenario,
            definition=ATLAS_RUN,
            settings=settings,
            state=state,
            workspace=workspace,
            stage="vehicle_ownership",
            year=2030,
            iteration=1,
            phase="run",
        )

    prepared_keys = tuple(
        ATLAS_PREPROCESS.output_paths(
            settings=settings, state=state, workspace=workspace
        )
    )
    recovery_root = archive_root / "consist-recovery"
    assert all(
        prepared_result.outputs[key].recovery_roots
        == [str((recovery_root / str(prepared_result.run.id)).resolve())]
        for key in prepared_keys
    )
    assert continuation_result.outputs["atlas_continuation_2030"].recovery_roots == [
        str((recovery_root / str(continuation_result.run.id)).resolve())
    ]

    year_root = Path(workspace.get_atlas_mutable_input_dir()) / "year2030"
    output_root = Path(workspace.get_atlas_output_dir())
    shutil.rmtree(year_root)
    shutil.rmtree(output_root)
    assert not year_root.exists()
    assert not output_root.exists()

    hydrated_continuation = tracker.hydrate_run_outputs(
        continuation_result.run.id,
        target_root=workspace_root,
        keys=["atlas_continuation_2030"],
        preserve_existing=False,
        on_missing="raise",
    )
    hydrated_prepared = tracker.hydrate_run_outputs(
        prepared_result.run.id,
        target_root=workspace_root,
        keys=prepared_keys,
        preserve_existing=False,
        on_missing="raise",
    )

    assert hydrated_prepared.paths == {
        key: ATLAS_PREPROCESS.output_paths(
            settings=settings, state=state, workspace=workspace
        )[key]
        for key in prepared_keys
    }
    assert hydrated_continuation.paths["atlas_continuation_2030"] == year_root
    assert all(
        hydrated_prepared[key].status == "materialized_from_filesystem"
        for key in prepared_keys
    )
    assert (
        hydrated_continuation["atlas_continuation_2030"].status
        == "materialized_from_filesystem"
    )
    assert (year_root / "households.csv").read_text(encoding="utf-8") == "id\n1\n"
    assert (year_root / "vehicles_output.RData").read_text(
        encoding="utf-8"
    ) == "vehicles\n"
    assert (year_root / "households_output.RData").read_text(
        encoding="utf-8"
    ) == "households\n"


def test_atlas_run_identity_includes_exact_selected_static_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    existing_input = tmp_path / "existing.txt"
    selected_source = tmp_path / "atlas-static" / "modeaccessibility.csv"
    existing_input.write_text("existing\n", encoding="utf-8")
    selected_source.parent.mkdir()
    selected_source.write_text("selected static input\n", encoding="utf-8")

    class _Context:
        def get_runtime(self, name: str, default: object = None) -> object:
            return {"settings": _settings(), "state": _state()}.get(name, default)

    monkeypatch.setattr(
        step_consist_meta,
        "build_step_consist_kwargs",
        lambda **_kwargs: {"identity_inputs": [("existing", existing_input)]},
    )
    monkeypatch.setattr(
        step_consist_meta,
        "selected_atlas_static_input_sources",
        lambda _settings: (("modeaccessibility.csv", selected_source),),
        raising=False,
    )

    identity_inputs = step_consist_meta.consist_step_meta("atlas_run")[
        "identity_inputs"
    ](_Context())

    assert identity_inputs == [
        ("existing", existing_input),
        ("atlas_static/modeaccessibility.csv", selected_source),
    ]


def test_atlas_preprocess_projects_declared_csv_paths(
    tmp_path: Path,
) -> None:
    settings = _settings()
    state = _state()
    workspace = _workspace(tmp_path)
    year_root = Path(workspace.get_atlas_mutable_input_dir()) / "year2030"
    year_root.mkdir(parents=True)
    for filename in (
        "households.csv",
        "blocks.csv",
        "persons.csv",
        "residential.csv",
        "jobs.csv",
        "grave.csv",
    ):
        (year_root / filename).write_text("id\n1\n", encoding="utf-8")

    projected = ATLAS_PREPROCESS.project_outputs(
        {
            key: object()
            for key in ATLAS_PREPROCESS.output_paths(
                settings=settings, state=state, workspace=workspace
            )
        },
        settings=settings,
        state=state,
        workspace=workspace,
    )

    assert projected.prepared_inputs["atlas_households_csv"] == (
        year_root / "households.csv"
    )
    assert projected.prepared_inputs["atlas_grave_csv"] == year_root / "grave.csv"


def test_native_atlas_run_hands_scalar_outputs_to_postprocess_without_directory_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings()
    state = _state()
    workspace = _workspace(tmp_path / "workspace")
    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        hashing_strategy="full",
        mounts={"workspace": workspace.full_path},
    )
    prepared_root = Path(workspace.get_atlas_mutable_input_dir()) / "year2030"
    prepared_root.mkdir(parents=True)
    for filename in (
        "households.csv",
        "blocks.csv",
        "persons.csv",
        "residential.csv",
        "jobs.csv",
        "grave.csv",
    ):
        (prepared_root / filename).write_text("id\n1\n", encoding="utf-8")
    source_h5 = tmp_path / "source" / "output_2030.h5"
    source_h5.parent.mkdir()
    with pd.HDFStore(source_h5, mode="w") as store:
        for table_name in ("households", "persons", "jobs", "blocks"):
            store.put(f"/{table_name}", pd.DataFrame({"value": [1]}))

    class _Runner:
        def __init__(self, *_args: object) -> None:
            pass

        def run(self, _inputs: object, run_workspace: object) -> None:
            output_root = Path(run_workspace.get_atlas_output_dir())
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "householdv_2030.csv").write_text("household_id\n1\n")
            (output_root / "vehicles_2030.csv").write_text("vehicle_id\n1\n")
            continuation_root = (
                Path(run_workspace.get_atlas_mutable_input_dir()) / "year2030"
            )
            continuation_root.mkdir(parents=True, exist_ok=True)
            (continuation_root / "vehicles_output.RData").write_text("vehicles")
            (continuation_root / "households_output.RData").write_text("households")

    class _Postprocessor:
        def __init__(self, *_args: object) -> None:
            pass

        @staticmethod
        def expected_outputs(
            _settings: object, _state: object, run_workspace: object
        ) -> dict[str, Path]:
            output_root = Path(run_workspace.get_atlas_output_dir())
            return {
                ATLAS_OUTPUT_DIR: output_root,
                USIM_POPULATION_SOURCE_H5: Path(
                    run_workspace.get_usim_mutable_data_dir()
                )
                / "output_2030_population_source.h5",
                ATLAS_VEHICLES2_OUTPUT: output_root / "vehicles2_2030.csv",
            }

        def postprocess(
            self, outputs: object, run_workspace: object, **_kwargs: object
        ) -> None:
            assert outputs.atlas_output_dir == Path(
                run_workspace.get_atlas_output_dir()
            )
            assert outputs.raw_outputs == {
                "householdv_2030": Path(run_workspace.get_atlas_output_dir())
                / "householdv_2030.csv",
                "vehicles_2030": Path(run_workspace.get_atlas_output_dir())
                / "vehicles_2030.csv",
            }
            (
                Path(run_workspace.get_atlas_output_dir()) / "vehicles2_2030.csv"
            ).write_text("vehicle_id\n1\n", encoding="utf-8")

    monkeypatch.setattr(urbansim_atlas, "AtlasRunner", _Runner)
    monkeypatch.setattr(urbansim_atlas, "AtlasPostprocessor", _Postprocessor)
    monkeypatch.setattr(
        urbansim_atlas,
        "ensure_usim_population_year_table_aliases",
        lambda **_kwargs: {},
    )
    with tracker.scenario("atlas-scalar-handoff") as scenario:
        with tracker.start_run("seed-usim", "test"):
            h5_artifact = tracker.log_artifact(
                source_h5, key=USIM_DATASTORE_H5, direction="input"
            )
        scenario.coupler.set_from_artifact(USIM_DATASTORE_H5, h5_artifact)

        prepared_inputs = ATLAS_PREPROCESS.output_paths(
            settings=settings,
            state=state,
            workspace=workspace,
        )

        def native_run_producer() -> None:
            urbansim_atlas._native_atlas_run(
                **prepared_inputs,
                settings=settings,
                state=state,
                workspace=workspace,
            )

        run_result = tracker.run(
            name="atlas-run-native",
            fn=native_run_producer,
            output_paths=ATLAS_RUN.output_paths(
                settings=settings,
                state=state,
                workspace=workspace,
            ),
        )
        scenario.coupler.update(run_result.outputs)
        assert scenario.coupler.get(ATLAS_OUTPUT_DIR) is None

        _, projected = execute_step(
            scenario=scenario,
            definition=ATLAS_POSTPROCESS,
            settings=settings,
            state=state,
            workspace=workspace,
            stage="vehicle_ownership",
            year=2030,
            iteration=1,
            phase="postprocess",
        )

    assert projected.processed_outputs[ATLAS_VEHICLES2_OUTPUT] == (
        Path(workspace.get_atlas_output_dir()) / "vehicles2_2030.csv"
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

    atlas_output_dir = Path(workspace.get_atlas_output_dir())
    atlas_output_dir.mkdir(parents=True)
    householdv = atlas_output_dir / "householdv_2030.csv"
    vehicles = atlas_output_dir / "vehicles_2030.csv"
    householdv.write_text("household_id\n1\n", encoding="utf-8")
    vehicles.write_text("vehicle_id\n1\n", encoding="utf-8")
    urbansim_atlas._native_atlas_postprocess(
        householdv,
        vehicles,
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
    source_householdv = tracker_root / "fixtures" / "householdv_2023.csv"
    source_vehicles = tracker_root / "fixtures" / "vehicles_2023.csv"
    source_householdv.write_text("household_id\n1\n", encoding="utf-8")
    source_vehicles.write_text("vehicle_id\n1\n", encoding="utf-8")
    with tracker.start_run("seed", "test"):
        h5_artifact = tracker.log_artifact(
            source_h5,
            key=USIM_DATASTORE_H5,
            direction="input",
        )
        householdv_artifact = tracker.log_artifact(
            source_householdv,
            key="householdv_2023",
            direction="input",
        )
        vehicles_artifact = tracker.log_artifact(
            source_vehicles,
            key="vehicles_2023",
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
        scenario.coupler.set_from_artifact("householdv_2023", householdv_artifact)
        scenario.coupler.set_from_artifact("vehicles_2023", vehicles_artifact)
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
    monkeypatch.setattr(
        urbansim_atlas,
        "ensure_usim_population_year_table_aliases",
        lambda **_kwargs: {},
    )
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
        ATLAS_OUTPUT_DIR: tmp_path / "snapshots" / "atlas-output",
    }
    snapshots.update(
        ATLAS_PREPROCESS.output_paths(
            settings=settings,
            state=state,
            workspace=workspace,
        )
    )
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
        **{
            key: snapshots[key]
            for key in ATLAS_PREPROCESS.output_paths(
                settings=settings,
                state=state,
                workspace=workspace,
            )
        },
        settings=settings,
        state=state,
        workspace=workspace,
    )
    snapshots["householdv_2030"] = (
        Path(workspace.get_atlas_output_dir()) / "householdv_2030.csv"
    )
    snapshots["vehicles_2030"] = (
        Path(workspace.get_atlas_output_dir()) / "vehicles_2030.csv"
    )
    snapshots["householdv_2030"].parent.mkdir(parents=True, exist_ok=True)
    snapshots["householdv_2030"].write_text("household_id\n1\n", encoding="utf-8")
    snapshots["vehicles_2030"].write_text("vehicle_id\n1\n", encoding="utf-8")
    urbansim_atlas._native_atlas_postprocess(
        snapshots["householdv_2030"],
        snapshots["vehicles_2030"],
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
    assert getattr(calls["atlas_run"], "atlas_mutable_input_dir") == Path(
        workspace.get_atlas_mutable_input_dir()
    )
    assert calls["atlas_postprocess"] == {
        "usim_datastore_h5": snapshots["usim_datastore_h5"]
    }


def test_native_atlas_postprocess_projector_uses_persisted_outputs_only(
    monkeypatch, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    atlas_output_dir = Path(workspace.get_atlas_output_dir())
    atlas_output_dir.mkdir(parents=True)
    usim_h5 = tmp_path / "population.h5"
    usim_h5.touch()
    vehicles2 = atlas_output_dir / "vehicles2_2030.csv"
    vehicles2.write_text("vehicle_id\n1\n", encoding="utf-8")
    declared_outputs = {
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
            USIM_POPULATION_SOURCE_H5: SimpleNamespace(path=usim_h5),
            ATLAS_VEHICLES2_OUTPUT: SimpleNamespace(path=vehicles2),
        },
        settings=_settings(),
        state=_state(),
        workspace=workspace,
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
