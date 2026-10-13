from pathlib import Path
from types import SimpleNamespace
import inspect
import shutil

import pytest
from consist import ExecutionOptions, Tracker, resolve_step_contract

from pilates.atlas.preprocessor import AtlasPreprocessor
from pilates.atlas.runner import AtlasLaunchContext
import pilates.atlas.preprocessor as atlas_preprocessor_module
from pilates.runtime import restart as restart_runtime
from pilates.workflows import binding as workflow_binding
from pilates.workflows.atlas_state import AtlasSubState
from pilates.workflows.steps import urbansim_atlas
from pilates.workflows.steps.urbansim_atlas import ATLAS_PREPROCESS, ATLAS_RUN
from pilates.workflows.artifact_keys import USIM_DATASTORE_H5
from pilates.workflows.stages.land_use import run_land_use_stage
from pilates.workflows.stages.vehicle_ownership import run_vehicle_ownership_stage
from workflow_state import WorkflowState


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_copy_data_to_mutable_location_uses_fallback_source_for_missing_required_files(
    tmp_path, monkeypatch
):
    primary_source = tmp_path / "primary_atlas_input"
    project_root = tmp_path / "project_root"
    fallback_source = project_root / "pilates" / "atlas" / "atlas_input"
    output_dir = tmp_path / "output"

    required_relpaths = (
        "psid_names.Rdat",
        "adopt/zev_mandate/new_vehicles_biannual_values_2021.csv",
    )
    _touch(fallback_source / required_relpaths[0])
    _touch(fallback_source / required_relpaths[1], content="Year,value\n2021,1\n")

    # Primary source intentionally does not contain required files.
    primary_source.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        atlas_preprocessor_module,
        "atlas_static_input_relpaths",
        lambda _settings: required_relpaths,
    )
    monkeypatch.setattr(
        atlas_preprocessor_module,
        "find_project_root",
        lambda start_path=None: str(project_root),
    )

    preprocessor = AtlasPreprocessor.__new__(AtlasPreprocessor)
    settings = {
        "atlas": {
            "host_input_folder": str(primary_source),
            "scenario": "zev_mandate",
            "adscen": "zev_mandate",
        }
    }

    _inputs, outputs = preprocessor.copy_data_to_mutable_location(
        settings=settings,
        output_dir=str(output_dir),
    )

    output_paths = {
        Path(record.file_path).relative_to(output_dir).as_posix()
        for record in outputs.all_records()
    }
    assert set(required_relpaths) <= output_paths

    inputs_by_key = {record.short_name: record for record in _inputs.all_records()}
    psid_record = inputs_by_key["psid_names"]
    assert psid_record.metadata["atlas_static_input"] is True
    assert psid_record.metadata["atlas_relpath"] == "psid_names.Rdat"
    assert psid_record.metadata["atlas_source_origin"] == "fallback"
    assert psid_record.metadata["atlas_input_group"] == "global"

    adopt_record = inputs_by_key["adopt/zev_mandate/new_vehicles_biannual_values_2021"]
    assert adopt_record.metadata["atlas_static_input"] is True
    assert adopt_record.metadata["atlas_source_origin"] == "fallback"
    assert adopt_record.metadata["atlas_input_group"] == "adopt"
    assert adopt_record.metadata["atlas_scenario"] == "zev_mandate"
    assert adopt_record.metadata["atlas_input_year"] == 2021
    assert adopt_record.metadata["profile_file_schema"] is True


def test_copy_data_to_mutable_location_raises_when_required_static_file_missing(
    tmp_path, monkeypatch
):
    primary_source = tmp_path / "primary_atlas_input"
    project_root = tmp_path / "project_root"
    fallback_source = project_root / "pilates" / "atlas" / "atlas_input"
    output_dir = tmp_path / "output"

    required_relpaths = ("adopt/zev_mandate/new_vehicles_biannual_values_2021.csv",)
    primary_source.mkdir(parents=True, exist_ok=True)
    fallback_source.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        atlas_preprocessor_module,
        "atlas_static_input_relpaths",
        lambda _settings: required_relpaths,
    )
    monkeypatch.setattr(
        atlas_preprocessor_module,
        "find_project_root",
        lambda start_path=None: str(project_root),
    )

    preprocessor = AtlasPreprocessor.__new__(AtlasPreprocessor)
    settings = {
        "atlas": {
            "host_input_folder": str(primary_source),
            "scenario": "zev_mandate",
            "adscen": "zev_mandate",
        }
    }

    with pytest.raises(RuntimeError, match="Missing required ATLAS static input files"):
        preprocessor.copy_data_to_mutable_location(
            settings=settings,
            output_dir=str(output_dir),
        )


def test_restart_required_atlas_input_years_uses_previous_subyear_not_start_year_minus_two():
    assert atlas_preprocessor_module._restart_required_atlas_input_years(
        start_year=2017,
        atlas_year=2023,
    ) == [2017, 2021]


def test_restart_local_artifact_policy_has_no_atlas_mirror_rule():
    assert all(
        rule.name != "atlas_restart_inputs"
        for rule in workflow_binding.restart_required_local_artifact_policy()
    )


def test_restart_parent_forecast_rejects_subyear_before_scheduler_boundary() -> None:
    state = SimpleNamespace(_year_schedule=(2017, 2023, 2030))

    with pytest.raises(RuntimeError, match="outside WorkflowState._year_schedule"):
        restart_runtime._atlas_parent_forecast_year_for_subyear(
            state=state,
            atlas_subyear=2016,
        )


def test_vehicle_ownership_stage_does_not_archive_atlas_members_directly():
    """Native producer archival is centralized in ``execute_step``."""

    source = inspect.getsource(run_vehicle_ownership_stage)

    assert "archive_copy_now" not in source
    assert "flush_archive_queue" not in source


def test_land_use_stage_does_not_archive_urbansim_members_directly():
    """UrbanSim producer archival is centralized in ``execute_step``."""

    source = inspect.getsource(run_land_use_stage)

    assert "archive_copy_now" not in source
    assert "flush_archive_queue" not in source


def test_restart_selects_archived_native_atlas_producers_from_prior_interval_facet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restart selects the persisted native parent interval, then hydrates it."""

    archive_run_dir = tmp_path / "archive-run"
    producer_root = tmp_path / "producer-workspace"
    current_run_dir = tmp_path / "current-run"
    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        hashing_strategy="full",
        mounts={"workspace": str(producer_root), "restart": str(current_run_dir)},
    )
    workspace = SimpleNamespace(
        full_path=str(producer_root),
        get_atlas_mutable_input_dir=lambda: str(
            producer_root / "atlas" / "atlas_input"
        ),
        get_atlas_output_dir=lambda: str(producer_root / "atlas" / "output"),
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(vehicle_ownership="atlas"), cache_epoch=1
        ),
        atlas=SimpleNamespace(beamac=0),
    )

    producer_parent = WorkflowState(
        2017,
        2030,
        6,
        True,
        True,
        False,
        False,
        year=2017,
        major_stage=WorkflowState.Stage.vehicle_ownership_model,
        inner_iter=0,
        sub_stage=None,
        file_loc=None,
        asim_compiled=False,
        full_settings={},
    )
    producer_parent._set_year_schedule((2017, 2023, 2030))
    producer_parent.forecast_year = 2023
    producer_parent.current_inner_iter = 0
    producer_state = AtlasSubState(producer_parent, 2021)

    class _Preprocessor:
        def __init__(self, *_args: object) -> None:
            pass

        def preprocess(self, run_workspace: object, **_kwargs: object) -> None:
            root = Path(run_workspace.get_atlas_mutable_input_dir()) / "year2021"
            root.mkdir(parents=True)
            for name in (
                "households.csv",
                "blocks.csv",
                "persons.csv",
                "residential.csv",
                "jobs.csv",
                "grave.csv",
            ):
                (root / name).write_text("id\n1\n", encoding="utf-8")

    class _Runner:
        def __init__(self, *_args: object) -> None:
            pass

        def run(self, _inputs: object, launch_context: AtlasLaunchContext) -> None:
            output_dir = launch_context.output_root
            output_dir.mkdir(parents=True)
            (output_dir / "householdv_2021.csv").write_text("id\n1\n")
            (output_dir / "vehicles_2021.csv").write_text("id\n1\n")
            root = launch_context.input_root / "year2021"
            (root / "vehicles_output.RData").write_text("vehicles\n")
            (root / "households_output.RData").write_text("households\n")

    monkeypatch.setattr(urbansim_atlas, "AtlasPreprocessor", _Preprocessor)
    monkeypatch.setattr(urbansim_atlas, "AtlasRunner", _Runner)
    source_h5 = tmp_path / "source" / "output_2023.h5"
    source_h5.parent.mkdir(parents=True)
    source_h5.write_bytes(b"source")
    runtime = {"settings": settings, "state": producer_state, "workspace": workspace}
    native_facet = resolve_step_contract(
        ATLAS_PREPROCESS.function,
        year=2021,
        iteration=0,
        phase="preprocess",
        stage="vehicle_ownership",
        runtime_kwargs=runtime,
    ).facet
    assert native_facet == {
        "atlas_subyear": 2021,
        "main_forecast_year": 2023,
        "native_input_contract": {
            "status": "incomplete",
            "reason": "conditional workspace skim fallback remains available",
            "configuration": {"available": True, "kind": "payload"},
        },
    }

    with tracker.scenario("native-atlas-restart") as scenario:
        preprocess_result = scenario.run(
            fn=ATLAS_PREPROCESS.function,
            run_id="archive-run__atlas-preprocess-2021",
            inputs={USIM_DATASTORE_H5: source_h5},
            year=2021,
            iteration=0,
            stage="vehicle_ownership",
            phase="preprocess",
            facet=native_facet,
            output_paths=ATLAS_PREPROCESS.output_paths(
                settings=settings, state=producer_state, workspace=workspace
            ),
            execution_options=ExecutionOptions(
                input_binding="paths", runtime_kwargs=runtime
            ),
        )
        run_inputs = {
            key: path
            for key, path in ATLAS_PREPROCESS.output_paths(
                settings=settings, state=producer_state, workspace=workspace
            ).items()
        }
        run_result = scenario.run(
            fn=ATLAS_RUN.function,
            run_id="archive-run__atlas-run-2021",
            inputs=run_inputs,
            year=2021,
            iteration=0,
            stage="vehicle_ownership",
            phase="run",
            facet=native_facet,
            output_paths=ATLAS_RUN.output_paths(
                settings=settings, state=producer_state, workspace=workspace
            ),
            output_sets=ATLAS_RUN.output_sets(
                settings=settings, state=producer_state, workspace=workspace
            ),
            execution_options=ExecutionOptions(
                input_binding="paths",
                runtime_kwargs={
                    **runtime,
                    "atlas_launch_context": AtlasLaunchContext(
                        input_root=Path(workspace.get_atlas_mutable_input_dir()),
                        output_root=Path(workspace.get_atlas_output_dir()),
                    ),
                },
            ),
        )

    for result in (preprocess_result, run_result):
        tracker.archive_run_outputs(
            result.run.id,
            str(archive_run_dir / "consist-recovery" / result.run.id),
            mode="copy",
        )
    shutil.rmtree(producer_root)

    restart_state = WorkflowState(
        2017,
        2030,
        6,
        True,
        True,
        False,
        False,
        year=2023,
        major_stage=WorkflowState.Stage.vehicle_ownership_model,
        inner_iter=0,
        sub_stage=None,
        file_loc=None,
        asim_compiled=False,
        full_settings={},
    )
    restart_state._set_year_schedule((2017, 2023, 2030))
    restart_state.forecast_year = 2030
    restart_state.current_major_stage = WorkflowState.Stage.vehicle_ownership_model
    restart_state.current_inner_iter = 0
    restart_state.run_info_path = str(archive_run_dir / "run_state.yaml")
    current_workspace = SimpleNamespace(
        full_path=str(current_run_dir),
        get_atlas_mutable_input_dir=lambda: str(
            current_run_dir / "atlas" / "atlas_input"
        ),
    )
    monkeypatch.setattr(
        restart_runtime,
        "_open_archive_tracker",
        lambda _settings, **_kwargs: tracker,
    )

    restart_runtime.hydrate_restart_atlas_continuation_inputs(
        settings=settings,
        state=restart_state,
        workspace=current_workspace,
        workflow_stage=WorkflowState.Stage,
    )

    restored_root = current_run_dir / "atlas" / "atlas_input" / "year2021"
    for name in (
        "households.csv",
        "blocks.csv",
        "persons.csv",
        "residential.csv",
        "jobs.csv",
        "grave.csv",
    ):
        assert (restored_root / name).read_text() == "id\n1\n"
    assert (restored_root / "vehicles_output.RData").read_text() == "vehicles\n"
    assert (restored_root / "households_output.RData").read_text() == "households\n"


def test_restart_rejects_ambiguous_selected_atlas_run(tmp_path, monkeypatch):
    class ArchiveTracker:
        def find_matching_runs(self, **kwargs):
            if kwargs["model"] == "atlas_preprocess":
                return [SimpleNamespace(id="preprocess-run")]
            return [SimpleNamespace(id="run-a"), SimpleNamespace(id="run-b")]

    monkeypatch.setattr(
        restart_runtime,
        "_open_archive_tracker",
        lambda _settings, **_kwargs: ArchiveTracker(),
        raising=False,
    )
    workspace = SimpleNamespace(
        get_atlas_mutable_input_dir=lambda: str(
            tmp_path / "current" / "atlas" / "atlas_input"
        )
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(vehicle_ownership="atlas"), cache_epoch=2
        ),
        atlas=SimpleNamespace(beamac=0),
    )
    state = SimpleNamespace(
        start_year=2017,
        year=2023,
        current_year=2023,
        _year_schedule=(2017, 2023, 2030),
        iteration=0,
        run_info_path=str(tmp_path / "archive" / "run_state.yaml"),
        current_major_stage=WorkflowState.Stage.vehicle_ownership_model,
    )

    with pytest.raises(RuntimeError, match="exactly one completed atlas_run"):
        restart_runtime.hydrate_restart_atlas_continuation_inputs(
            settings=settings,
            state=state,
            workspace=workspace,
            workflow_stage=WorkflowState.Stage,
        )


def test_restart_selection_rejects_cross_scope_and_epoch_runs(tmp_path):
    tracker = Tracker(
        run_dir=tmp_path / "archive-run",
        db_path=str(tmp_path / "provenance.duckdb"),
        hashing_strategy="full",
    )
    for run_id, cache_epoch in (
        ("archive-run__atlas-preprocess", 2),
        ("archive-run__stale-epoch", 1),
        ("other-run__atlas-preprocess", 2),
    ):
        with tracker.start_run(
            run_id,
            "atlas_preprocess",
            year=2021,
            iteration=0,
            stage="atlas",
            phase="preprocess",
            cache_epoch=cache_epoch,
        ):
            pass

    selected = restart_runtime._select_exact_completed_restart_run(
        tracker=tracker,
        target={
            "model": "atlas_preprocess",
            "status": "completed",
            "year": 2021,
            "iteration": 0,
            "stage": "atlas",
            "phase": "preprocess",
            "cache_epoch": 2,
            "run_scope": "archive-run",
        },
        step_name="atlas_preprocess",
    )

    assert selected.id == "archive-run__atlas-preprocess"
