from types import SimpleNamespace
import json
import os
from pathlib import Path
import re

import pytest
from consist.types import CacheOptions

import run as run_module
from pilates.generic.records import FileRecord, RecordStore
from pilates.utils import consist_db_snapshot as snapshot_module
from workflow_state import WorkflowState


class DummyWorkspace:
    def __init__(self, full_path="/tmp/bootstrap", settings=None):
        self.full_path = full_path
        self.settings = settings
        self.input_data = {}
        self.output_data = {}

    def get_usim_mutable_data_dir(self):
        return os.path.join(self.full_path, "urbansim", "data")

    def get_asim_mutable_data_dir(self):
        return os.path.join(self.full_path, "activitysim", "data")

    def get_asim_mutable_configs_dir(self):
        return os.path.join(self.full_path, "activitysim", "configs")

    def get_asim_output_dir(self):
        return os.path.join(self.full_path, "activitysim", "output")

    def get_atlas_mutable_input_dir(self):
        return os.path.join(self.full_path, "atlas", "atlas_input")

    def get_beam_mutable_data_dir(self):
        return os.path.join(self.full_path, "beam", "input")


class DummyInitialization:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, _settings, workspace):
        rec_in = RecordStore(
            recordList=[
                FileRecord(
                    unique_id="in1",
                    short_name="bootstrap_in",
                    file_path="/tmp/source",
                    metadata={"model": "beam", "bootstrap_direction": "input"},
                )
            ]
        )
        rec_out = RecordStore(
            recordList=[
                FileRecord(
                    unique_id="out1",
                    short_name="bootstrap_out",
                    file_path="/tmp/dest",
                    metadata={"model": "beam", "bootstrap_direction": "output"},
                )
            ]
        )
        combined = RecordStore()
        combined += rec_in
        combined += rec_out
        return combined


class DummyTracker:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses[len(self.calls) - 1]
        if response["execute_fn"] and kwargs.get("fn") is not None:
            kwargs["fn"]()
        return SimpleNamespace(
            cache_hit=response["cache_hit"],
            run=SimpleNamespace(id=response["run_id"]),
        )


class DummySnapshotTracker:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def snapshot_db(self, dest_path: str, checkpoint: bool = True):
        self.calls.append({"dest_path": dest_path, "checkpoint": checkpoint})
        if self.fail:
            raise RuntimeError("simulated snapshot failure")
        output_path = Path(dest_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("db", encoding="utf-8")
        metadata = {
            "run_id": "run-test",
            "snapshot_ts_utc": "2026-02-24T00:00:00Z",
        }
        (output_path.parent / snapshot_module.snapshot_meta_filename(output_path.name)).write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )


def _settings(cache_enabled=True):
    return SimpleNamespace(run=SimpleNamespace(bootstrap_cache_enabled=cache_enabled))


def _state():
    return SimpleNamespace(start_year=2017)


def test_run_bootstrap_phase_cache_miss_executes_once(monkeypatch):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[
            {"cache_hit": False, "execute_fn": True, "run_id": "bootstrap_probe"}
        ]
    )
    workspace = DummyWorkspace()

    result = run_module.run_bootstrap_phase(
        tracker=tracker,
        settings=_settings(cache_enabled=True),
        state=_state(),
        workspace=workspace,
        scenario_id="seattle-baseline",
        seed=12345,
    )

    assert len(tracker.calls) == 1
    first_call = tracker.calls[0]
    assert "scenario_id:seattle-baseline" in first_call["tags"]
    assert "seed:12345" in first_call["tags"]
    assert "year:2017" in first_call["tags"]
    assert "iteration:0" in first_call["tags"]
    assert "model:initialization" in first_call["tags"]
    assert first_call["facet"]["scenario_id"] == "seattle-baseline"
    assert first_call["facet"]["seed"] == 12345
    assert first_call["facet"]["year"] == 2017
    assert first_call["facet"]["iteration"] == 0
    assert first_call["facet"]["model"] == "initialization"
    assert result["bootstrap_cache_hit"] is False
    assert result["staged_artifact_summary"]["copied_records_total"] == 2
    assert result["manifest_reference"] == {"probe_run_id": "bootstrap_probe"}


def test_run_bootstrap_phase_cache_hit_materializes_with_overwrite(monkeypatch):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[
            {"cache_hit": True, "execute_fn": False, "run_id": "bootstrap_probe"},
            {
                "cache_hit": False,
                "execute_fn": True,
                "run_id": "bootstrap_materialize",
            },
        ]
    )
    workspace = DummyWorkspace()

    result = run_module.run_bootstrap_phase(
        tracker=tracker,
        settings=_settings(cache_enabled=True),
        state=_state(),
        workspace=workspace,
        scenario_id="seattle-baseline",
        seed=12345,
    )

    assert len(tracker.calls) == 2
    assert result["bootstrap_cache_hit"] is True
    overwrite_options = tracker.calls[1]["cache_options"]
    assert isinstance(overwrite_options, CacheOptions)
    assert overwrite_options.cache_mode == "overwrite"
    assert result["staged_artifact_summary"]["copied_records_total"] == 2
    assert result["manifest_reference"] == {
        "probe_run_id": "bootstrap_probe",
        "materialization_run_id": "bootstrap_materialize",
    }


def test_run_bootstrap_phase_cache_disabled_uses_cache_off(monkeypatch):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[
            {"cache_hit": False, "execute_fn": True, "run_id": "bootstrap_off"}
        ]
    )
    workspace = DummyWorkspace()

    result = run_module.run_bootstrap_phase(
        tracker=tracker,
        settings=_settings(cache_enabled=False),
        state=_state(),
        workspace=workspace,
        scenario_id="seattle-baseline",
        seed=None,
    )

    assert len(tracker.calls) == 1
    first_call = tracker.calls[0]
    assert "scenario_id:seattle-baseline" in first_call["tags"]
    assert all(not tag.startswith("seed:") for tag in first_call["tags"])
    assert first_call["facet"]["scenario_id"] == "seattle-baseline"
    assert "seed" not in first_call["facet"]
    cache_options = tracker.calls[0]["cache_options"]
    assert isinstance(cache_options, CacheOptions)
    assert cache_options.cache_mode == "off"
    assert result["bootstrap_cache_hit"] is False
    assert result["manifest_reference"] == {"probe_run_id": "bootstrap_off"}


def test_run_bootstrap_phase_archives_restart_critical_bootstrap_artifacts(monkeypatch, tmp_path):
    class BootstrapInit:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, _settings, workspace):
            os.makedirs(workspace.get_usim_mutable_data_dir(), exist_ok=True)
            os.makedirs(workspace.get_asim_mutable_data_dir(), exist_ok=True)
            os.makedirs(workspace.get_asim_mutable_configs_dir(), exist_ok=True)
            beam_config_path = Path(
                workspace.get_beam_mutable_data_dir(), "test", "beam.conf"
            )
            beam_config_path.parent.mkdir(parents=True, exist_ok=True)
            beam_config_path.write_text("beam")
            Path(workspace.get_usim_mutable_data_dir(), "usim_000.h5").write_text("h5")
            Path(workspace.get_asim_mutable_data_dir(), "households.csv").write_text("hh")
            Path(workspace.get_asim_mutable_configs_dir(), "configs").mkdir(parents=True, exist_ok=True)
            rec = RecordStore(
                recordList=[
                    FileRecord(unique_id="out1", short_name="bootstrap_out", file_path="/tmp/dest")
                ]
            )
            return rec

    monkeypatch.setattr(run_module, "Initialization", BootstrapInit)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    enqueued = []
    flushed = []
    monkeypatch.setattr(
        run_module,
        "enqueue_archive_copy",
        lambda *, key, path, workspace=None: enqueued.append((key, path)),
    )
    monkeypatch.setattr(
        run_module,
        "flush_archive_queue",
        lambda timeout=None, fail_on_timeout=False: flushed.append((timeout, fail_on_timeout)) or True,
    )

    tracker = DummyTracker(
        responses=[{"cache_hit": False, "execute_fn": True, "run_id": "bootstrap_probe"}]
    )
    workspace = DummyWorkspace(str(tmp_path / "workspace"))
    settings = SimpleNamespace(
        run=SimpleNamespace(
            bootstrap_cache_enabled=True,
            region="test",
            models=SimpleNamespace(activity_demand="activitysim", traffic_assignment="beam"),
        ),
        activitysim=SimpleNamespace(main_configs_dir="configs"),
        beam=SimpleNamespace(config="beam.conf"),
        urbansim=SimpleNamespace(
            region_mappings={"region_to_region_id": {"test": "000"}},
            input_file_template="usim_{region_id}.h5",
        ),
    )
    state = SimpleNamespace(start_year=2017)

    run_module.run_bootstrap_phase(
        tracker=tracker,
        settings=settings,
        state=state,
        workspace=workspace,
        scenario_id="test-scenario",
        seed=None,
    )

    assert ("urbansim_bootstrap_data_root", workspace.get_usim_mutable_data_dir()) in enqueued
    assert ("bootstrap_usim_datastore_base_h5", os.path.join(workspace.get_usim_mutable_data_dir(), "usim_000.h5")) in enqueued
    assert ("activitysim_bootstrap_data_root", workspace.get_asim_mutable_data_dir()) in enqueued
    assert ("activitysim_bootstrap_configs_root", workspace.get_asim_mutable_configs_dir()) in enqueued
    assert ("beam_mutable_data_dir", workspace.get_beam_mutable_data_dir()) in enqueued
    assert flushed == [(300, True)]


def test_bootstrap_output_invariant_accepts_valid_result():
    run_module._assert_bootstrap_output_invariant(
        {
            "bootstrap_cache_hit": False,
            "manifest_reference": {"probe_run_id": "bootstrap_probe"},
            "staged_artifact_summary": {"copied_records_total": 2},
        }
    )


def test_bootstrap_output_invariant_accepts_zero_copied_records_for_valid_summary():
    run_module._assert_bootstrap_output_invariant(
        {
            "bootstrap_cache_hit": False,
            "manifest_reference": {"probe_run_id": "bootstrap_probe"},
            "staged_artifact_summary": {"copied_records_total": 0},
        }
    )


@pytest.mark.parametrize(
    "invalid_result",
    [
        None,
        {},
        {"staged_artifact_summary": {}},
    ],
)
def test_bootstrap_output_invariant_rejects_invalid_or_empty_result(invalid_result):
    with pytest.raises(RuntimeError, match="Bootstrap initialization invariant failed"):
        run_module._assert_bootstrap_output_invariant(invalid_result)


def _restart_settings():
    return SimpleNamespace(
        run=SimpleNamespace(
            region="test",
            models=SimpleNamespace(
                activity_demand="activitysim",
                vehicle_ownership="atlas",
                traffic_assignment="beam",
            ),
        ),
        atlas=SimpleNamespace(scenario="baseline"),
        activitysim=SimpleNamespace(main_configs_dir="configs"),
        beam=SimpleNamespace(config="beam.conf"),
        urbansim=SimpleNamespace(
            region_mappings={"region_to_region_id": {"test": "000"}},
            input_file_template="usim_{region_id}.h5",
            output_file_template="usim_{year}.h5",
        ),
    )


def _resume_doctor_check_names(caplog) -> set:
    names = set()
    for record in caplog.records:
        if "[ResumeDoctor] check=" not in record.message:
            continue
        match = re.search(r"check=([^\s]+)", record.message)
        if match:
            names.add(match.group(1))
    return names


def test_resume_doctor_ready_summary_logs_expected_checks(tmp_path, caplog):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    workspace = DummyWorkspace(str(local_run_dir))
    state = SimpleNamespace(current_year=2018, current_inner_iter=0, data_initialized=True)

    archive_state_path = archive_run_dir / "run_state.yaml"
    local_state_path = local_run_dir / "run_state.yaml"
    archive_state_path.parent.mkdir(parents=True, exist_ok=True)
    local_state_path.parent.mkdir(parents=True, exist_ok=True)
    archive_state_path.write_text("archive-state", encoding="utf-8")
    local_state_path.write_text("local-state", encoding="utf-8")

    local_consist_db_path = local_run_dir / ".consist" / "provenance.duckdb"
    local_consist_db_path.parent.mkdir(parents=True, exist_ok=True)
    local_consist_db_path.write_text("db", encoding="utf-8")

    snapshot_db_path = (
        snapshot_module.snapshot_latest_dir(str(archive_run_dir))
        / local_consist_db_path.name
    )
    snapshot_db_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_db_path.write_text("snapshot-db", encoding="utf-8")

    local_manifest = run_module.build_manifest_path(workspace=workspace, year=2018, iteration=0)
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    local_manifest.write_text("manifest", encoding="utf-8")
    archive_manifest = run_module._map_local_path_to_archive(
        local_path=str(local_manifest),
        local_run_dir=str(local_run_dir),
        archive_run_dir=str(archive_run_dir),
    )
    assert archive_manifest is not None
    Path(archive_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(archive_manifest).write_text("archive-manifest", encoding="utf-8")

    with caplog.at_level("INFO"):
        run_module._run_resume_doctor_diagnostics(
            state=state,
            workspace=workspace,
            local_run_dir=str(local_run_dir),
            archive_run_dir=str(archive_run_dir),
            archive_state_path=str(archive_state_path),
            local_state_path=str(local_state_path),
            local_consist_db_path=str(local_consist_db_path),
            restart_missing_artifacts_initial=[],
            restart_missing_artifacts_after_rehydrate=[],
        )

    check_messages = [
        record.message
        for record in caplog.records
        if "[ResumeDoctor] check=" in record.message
    ]
    assert check_messages
    assert all(message.startswith("[ResumeDoctor] check=") for message in check_messages)
    assert _resume_doctor_check_names(caplog) >= {
        "archive_run_state",
        "local_run_state_mirror",
        "local_consist_db",
        "archive_latest_consist_db_snapshot",
        "required_restart_local_artifacts",
        "supply_demand_manifest_local",
        "supply_demand_manifest_archive_mapped",
    }
    assert "[ResumeDoctor] check=supply_demand_manifest_local status=ok" in caplog.text
    assert (
        "[ResumeDoctor] check=supply_demand_manifest_archive_mapped status=ok"
        in caplog.text
    )
    assert "[ResumeDoctor] summary status=ready reason=all_checks_ok" in caplog.text


def test_resume_doctor_degraded_summary_reports_missing_checks_and_manifest_checks(
    tmp_path, caplog
):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    workspace = DummyWorkspace(str(local_run_dir))
    state = SimpleNamespace(current_year=2022, current_inner_iter=1, data_initialized=True)
    missing = [{"key": "activitysim_settings_yaml", "path": "missing", "reason": "test"}]

    with caplog.at_level("INFO"):
        run_module._run_resume_doctor_diagnostics(
            state=state,
            workspace=workspace,
            local_run_dir=str(local_run_dir),
            archive_run_dir=str(archive_run_dir),
            archive_state_path=str(archive_run_dir / "run_state.yaml"),
            local_state_path=str(local_run_dir / "run_state.yaml"),
            local_consist_db_path=str(local_run_dir / ".consist" / "provenance.duckdb"),
            restart_missing_artifacts_initial=missing,
            restart_missing_artifacts_after_rehydrate=missing,
        )

    check_names = _resume_doctor_check_names(caplog)
    assert check_names >= {
        "archive_run_state",
        "local_run_state_mirror",
        "local_consist_db",
        "archive_latest_consist_db_snapshot",
        "required_restart_local_artifacts",
        "supply_demand_manifest_local",
        "supply_demand_manifest_archive_mapped",
    }
    assert "[ResumeDoctor] check=supply_demand_manifest_local status=missing" in caplog.text
    assert (
        "[ResumeDoctor] check=supply_demand_manifest_archive_mapped status=missing"
        in caplog.text
    )
    assert "[ResumeDoctor] summary status=degraded reason=missing_checks:" in caplog.text


def test_format_restart_command_uses_config_and_archive_state():
    settings = SimpleNamespace(settings_file="scenarios/settings-seattle.yaml")

    command = run_module._format_restart_command(
        settings=settings,
        archive_state_path="/tmp/pilates run/run_state.yaml",
    )

    assert (
        command
        == "python run.py -c scenarios/settings-seattle.yaml -S '/tmp/pilates run/run_state.yaml'"
    )


def test_main_logs_restart_instructions_on_failure(tmp_path, monkeypatch, caplog):
    class WorkspaceStub:
        def __init__(self, _settings, local_root: str, folder_name: str):
            self.full_path = os.path.join(local_root, folder_name)
            os.makedirs(self.full_path, exist_ok=True)

    class SnapshotStub:
        def final_snapshot(self):
            return True

    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(tmp_path / "archive-root"),
            local_workspace_root=str(tmp_path / "local-root"),
            enable_archive_copy=False,
            output_run_name="restart-hint-test",
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
        settings_file="scenarios/settings-seattle.yaml",
    )
    state = SimpleNamespace(
        run_info_path=None,
        data_initialized=False,
    )
    state.set_run_info_path = lambda path: setattr(state, "run_info_path", path)
    state.set_data_initialized = lambda initialized: setattr(
        state, "data_initialized", initialized
    )

    monkeypatch.setattr(run_module, "parse_args_and_settings", lambda: settings)
    monkeypatch.setattr(run_module.WorkflowState, "from_settings", lambda _s: state)
    monkeypatch.setattr(run_module, "_log_local_storage_info", lambda: None)
    monkeypatch.setattr(
        run_module,
        "resolve_consist_db_paths",
        lambda **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        run_module,
        "restore_local_consist_db_from_snapshot",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module,
        "seed_local_consist_db_from_shared",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(run_module, "_resolve_cache_epoch", lambda _settings: 1)
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(run_module.cr, "create_tracker", lambda **_kwargs: object())
    monkeypatch.setattr(run_module, "ConsistDbSnapshotManager", lambda **_kwargs: SnapshotStub())
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
    monkeypatch.setattr(
        run_module,
        "build_restart_bundle_manifest",
        lambda **_kwargs: {"artifacts": []},
    )
    monkeypatch.setattr(
        run_module,
        "write_restart_bundle_manifest",
        lambda **_kwargs: str(tmp_path / "archive-root" / "manifest.yaml"),
    )
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)

    def _fail_bootstrap(**_kwargs):
        raise RuntimeError("simulated bootstrap failure")

    monkeypatch.setattr(run_module, "run_bootstrap_phase", _fail_bootstrap)

    run_module._RUN_FAILURE_CONTEXT.clear()
    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="simulated bootstrap failure"):
            run_module.main()
        run_module._log_restart_instructions_on_failure()

    assert "Run failed. Restart command:" in caplog.text
    assert "python run.py -c scenarios/settings-seattle.yaml -S " in caplog.text
    assert "run_state.yaml" in caplog.text


def test_restart_preflight_detects_missing_local_workspace_artifacts(tmp_path):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    state = SimpleNamespace()

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )

    assert {item["key"] for item in missing} == {
        "usim_datastore_base_h5",
        "omx_skims",
        "hh_size",
        "income_rates",
        "relmap",
        "schools",
        "school_districts",
        "activitysim_config_settings_yaml_configs",
        "activitysim_config_settings_yaml_configs_extended",
        "activitysim_config_settings_yaml_configs_mp",
        "activitysim_config_settings_yaml_configs_sh_compile",
    }


def test_restart_preflight_skips_activitysim_locals_outside_supply_demand_stage(tmp_path):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    state = SimpleNamespace(current_major_stage=WorkflowState.Stage.vehicle_ownership_model)

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )

    keys = {item["key"] for item in missing}
    assert "usim_datastore_base_h5" in keys
    assert "activitysim_config_settings_yaml_configs" not in keys
    assert "activitysim_config_settings_yaml_configs_mp" not in keys
    assert "zarr_skims" not in keys
    assert any(key.startswith("atlas_static::") for key in keys)


def test_restart_preflight_requires_atlas_static_inputs_in_vehicle_stage(tmp_path):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    state = SimpleNamespace(current_major_stage=WorkflowState.Stage.vehicle_ownership_model)

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )

    paths = {item["path"] for item in missing}
    assert any(path.endswith("atlas/atlas_input/psid_names.Rdat") for path in paths)
    assert any(path.endswith("atlas/atlas_input/accessbility_2015.RData") for path in paths)


def test_restart_preflight_requires_zarr_skims_when_resuming_compiled_supply_demand(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    state = SimpleNamespace(
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
        asim_compiled=True,
    )

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )

    keys = {item["key"] for item in missing}
    paths = {item["path"] for item in missing}
    assert "zarr_skims" in keys
    assert any(path.endswith("activitysim/output/cache/skims.zarr") for path in paths)


def test_restart_preflight_requires_beam_region_dir_when_resuming_supply_demand(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    state = SimpleNamespace(
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
        asim_compiled=False,
    )

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )

    keys = {item["key"] for item in missing}
    paths = {item["path"] for item in missing}
    assert "beam_mutable_data_dir" in keys
    assert "beam_region_input_dir" in keys
    assert "beam_primary_config_file" in keys
    assert any(path.endswith("beam/input") for path in paths)
    assert any(path.endswith("beam/input/test") for path in paths)
    assert any(path.endswith("beam/input/test/beam.conf") for path in paths)


def test_repair_restart_beam_inputs_from_source_repopulates_missing_primary_config(
    tmp_path, monkeypatch
):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    settings = _restart_settings()
    state = SimpleNamespace()

    calls = []

    class _BeamPreprocessor:
        def copy_data_to_mutable_location(self, settings_arg, output_dir):
            calls.append((settings_arg, output_dir))
            target = (
                Path(output_dir)
                / settings_arg.run.region
                / settings_arg.beam.config
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("beam", encoding="utf-8")

    class _Factory:
        def get_preprocessor(self, model_name, state_arg):
            assert model_name == "beam"
            assert state_arg is state
            return _BeamPreprocessor()

    monkeypatch.setattr(run_module, "ModelFactory", lambda: _Factory())

    repaired = run_module._repair_restart_beam_inputs_from_source(
        settings=settings,
        state=state,
        workspace=workspace,
    )

    assert repaired is True
    assert calls == [(settings, workspace.get_beam_mutable_data_dir())]
    assert (
        Path(workspace.get_beam_mutable_data_dir())
        / settings.run.region
        / settings.beam.config
    ).exists()


def test_restart_preflight_requires_activitysim_iteration_outputs_for_traffic_assignment(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    state = SimpleNamespace(
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
        current_sub_stage=WorkflowState.Stage.traffic_assignment,
        current_year=2017,
        current_inner_iter=0,
        asim_compiled=True,
    )

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )

    keys = {item["key"] for item in missing}
    paths = {item["path"] for item in missing}
    assert "activitysim_iteration_beam_plans_parquet" in keys
    assert "activitysim_iteration_households_parquet" in keys
    assert "activitysim_iteration_persons_parquet" in keys
    assert any(
        path.endswith("activitysim/output/year-2017-iteration-0/beam_plans.parquet")
        for path in paths
    )
    assert any(
        path.endswith("activitysim/output/year-2017-iteration-0/households.parquet")
        for path in paths
    )
    assert any(
        path.endswith("activitysim/output/year-2017-iteration-0/persons.parquet")
        for path in paths
    )


def test_restart_repair_rewinds_stale_supply_demand_state_when_atlas_outputs_incomplete(
    tmp_path,
):
    archive_run_dir = tmp_path / "archive-run"
    atlas_output_dir = archive_run_dir / "atlas" / "atlas_output"
    atlas_output_dir.mkdir(parents=True, exist_ok=True)
    (atlas_output_dir / "householdv_2017.csv").write_text("hh\n", encoding="utf-8")
    (atlas_output_dir / "vehicles_2017.csv").write_text("veh\n", encoding="utf-8")

    writes = []
    state = SimpleNamespace(
        Stage=WorkflowState.Stage,
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
        current_sub_stage=WorkflowState.Stage.activity_demand,
        current_inner_iter=3,
        sub_stage_progress="postprocessor",
        current_year=2017,
        forecast_year=2023,
        write_state=lambda: writes.append("written"),
    )

    repaired = run_module._repair_restart_state_for_incomplete_atlas_outputs(
        settings=_restart_settings(),
        state=state,
        archive_run_dir=str(archive_run_dir),
    )

    assert repaired is True
    assert state.current_major_stage == WorkflowState.Stage.vehicle_ownership_model
    assert state.current_sub_stage is None
    assert state.current_inner_iter == 0
    assert state.sub_stage_progress is None
    assert writes == ["written"]


def test_restart_repair_keeps_supply_demand_when_all_atlas_outputs_complete(tmp_path):
    archive_run_dir = tmp_path / "archive-run"
    atlas_output_dir = archive_run_dir / "atlas" / "atlas_output"
    atlas_output_dir.mkdir(parents=True, exist_ok=True)
    for atlas_year in (2017, 2019, 2021, 2023):
        (atlas_output_dir / f"householdv_{atlas_year}.csv").write_text(
            "hh\n", encoding="utf-8"
        )
        (atlas_output_dir / f"vehicles_{atlas_year}.csv").write_text(
            "veh\n", encoding="utf-8"
        )
        (atlas_output_dir / f"vehicles2_{atlas_year}.csv").write_text(
            "veh2\n", encoding="utf-8"
        )

    writes = []
    state = SimpleNamespace(
        Stage=WorkflowState.Stage,
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
        current_sub_stage=WorkflowState.Stage.activity_demand,
        current_inner_iter=0,
        sub_stage_progress=None,
        current_year=2017,
        forecast_year=2023,
        write_state=lambda: writes.append("written"),
    )

    repaired = run_module._repair_restart_state_for_incomplete_atlas_outputs(
        settings=_restart_settings(),
        state=state,
        archive_run_dir=str(archive_run_dir),
    )

    assert repaired is False
    assert state.current_major_stage == WorkflowState.Stage.supply_demand_loop
    assert state.current_sub_stage == WorkflowState.Stage.activity_demand
    assert writes == []


def test_build_atlas_static_inputs_fallback_uses_atlas_static_key_scheme(tmp_path):
    settings = _restart_settings()
    workspace = DummyWorkspace(str(tmp_path / "local-run"), settings=settings)
    atlas_input_dir = Path(workspace.get_atlas_mutable_input_dir())
    (atlas_input_dir / "psid_names.Rdat").parent.mkdir(parents=True, exist_ok=True)
    (atlas_input_dir / "psid_names.Rdat").write_text("psid", encoding="utf-8")
    (
        atlas_input_dir / "adopt" / "baseline" / "used_vehicles_2017.csv"
    ).parent.mkdir(parents=True, exist_ok=True)
    (
        atlas_input_dir / "adopt" / "baseline" / "used_vehicles_2017.csv"
    ).write_text("used", encoding="utf-8")

    fallback = run_module.build_atlas_static_inputs_fallback(workspace)

    assert fallback["psid_names"] == str(atlas_input_dir / "psid_names.Rdat")
    assert (
        fallback["adopt/baseline/used_vehicles_2017"]
        == str(atlas_input_dir / "adopt" / "baseline" / "used_vehicles_2017.csv")
    )
    assert "atlas_static_psid_names.Rdat" not in fallback


def test_rehydrate_missing_local_artifacts_from_archive_is_idempotent_and_preserves_existing(
    tmp_path,
):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    workspace = DummyWorkspace(str(local_run_dir))
    state = SimpleNamespace()

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )
    assert missing

    for artifact in missing:
        archive_path = run_module._map_local_path_to_archive(
            local_path=artifact["path"],
            local_run_dir=str(local_run_dir),
            archive_run_dir=str(archive_run_dir),
        )
        assert archive_path is not None
        archive_file = Path(archive_path)
        archive_file.parent.mkdir(parents=True, exist_ok=True)
        archive_file.write_text(f"archive-{artifact['key']}", encoding="utf-8")

    first_summary = run_module._rehydrate_missing_local_artifacts_from_archive(
        missing_artifacts=missing,
        local_run_dir=str(local_run_dir),
        archive_run_dir=str(archive_run_dir),
    )
    assert first_summary["copied"] == len(missing)
    assert first_summary["copy_errors"] == 0

    config_settings = next(
        item["path"]
        for item in missing
        if item["key"] == "activitysim_config_settings_yaml_configs"
    )
    Path(config_settings).write_text("local-only", encoding="utf-8")

    second_summary = run_module._rehydrate_missing_local_artifacts_from_archive(
        missing_artifacts=missing,
        local_run_dir=str(local_run_dir),
        archive_run_dir=str(archive_run_dir),
    )
    assert second_summary["copied"] == 0
    assert second_summary["skipped_existing"] >= len(missing)
    assert Path(config_settings).read_text(encoding="utf-8") == "local-only"


def test_rehydrate_missing_local_artifacts_from_archive_partial_archive_missing_counts(
    tmp_path,
):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    workspace = DummyWorkspace(str(local_run_dir))
    state = SimpleNamespace()

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )
    assert missing

    missing_archive_key = "activitysim_config_settings_yaml_configs_mp"
    for artifact in missing:
        if artifact["key"] == missing_archive_key:
            continue
        archive_path = run_module._map_local_path_to_archive(
            local_path=artifact["path"],
            local_run_dir=str(local_run_dir),
            archive_run_dir=str(archive_run_dir),
        )
        assert archive_path is not None
        archive_file = Path(archive_path)
        archive_file.parent.mkdir(parents=True, exist_ok=True)
        archive_file.write_text(f"archive-{artifact['key']}", encoding="utf-8")

    summary = run_module._rehydrate_missing_local_artifacts_from_archive(
        missing_artifacts=missing,
        local_run_dir=str(local_run_dir),
        archive_run_dir=str(archive_run_dir),
    )

    assert summary["copied"] == len(missing) - 1
    assert summary["skipped_missing_archive"] == 1
    assert summary["skipped_unmapped"] == 0
    assert summary["copy_errors"] == 0

    remaining = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )
    assert {item["key"] for item in remaining} == {missing_archive_key}
    assert bool(remaining) is True


def test_bundle_rehydrate_mode_copies_manifest_listed_artifact(tmp_path):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    rel_path = os.path.join("activitysim", "data", "households.csv")

    archive_file = archive_run_dir / rel_path
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    archive_file.write_text("archive-households", encoding="utf-8")

    summary = run_module._rehydrate_bundle_local_artifacts_from_archive(
        bundle_manifest={
            "schema_version": 1,
            "artifacts": [
                {
                    "key": "activitysim_input_households.csv",
                    "rel_path": rel_path,
                    "reason": "test bundle artifact",
                }
            ],
        },
        local_run_dir=str(local_run_dir),
        archive_run_dir=str(archive_run_dir),
    )

    assert summary["copied"] == 1
    assert summary["copy_errors"] == 0
    assert (local_run_dir / rel_path).read_text(encoding="utf-8") == "archive-households"


def test_bundle_rehydrate_mode_repairs_existing_directory_from_manifest(tmp_path):
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    rel_dir = os.path.join("activitysim", "configs", "configs_mp")

    local_dir = local_run_dir / rel_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "settings.yaml").write_text("local-settings", encoding="utf-8")

    archive_dir = archive_run_dir / rel_dir
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "settings.yaml").write_text("archive-settings", encoding="utf-8")
    (archive_dir / "constants.yaml").write_text("archive-constants", encoding="utf-8")

    summary = run_module._rehydrate_bundle_local_artifacts_from_archive(
        bundle_manifest={
            "schema_version": 1,
            "artifacts": [
                {
                    "key": "activitysim_config_dir_configs_mp",
                    "rel_path": rel_dir,
                    "reason": "test bundle config dir",
                    "kind": "dir",
                }
            ],
        },
        local_run_dir=str(local_run_dir),
        archive_run_dir=str(archive_run_dir),
    )

    assert summary["copied"] == 1
    assert summary["skipped_existing"] >= 1
    assert (local_dir / "settings.yaml").read_text(encoding="utf-8") == "local-settings"
    assert (local_dir / "constants.yaml").read_text(encoding="utf-8") == "archive-constants"


def test_resume_rewind_guardrail_blocks_by_default_and_allows_override(tmp_path):
    archive_state_path = tmp_path / "archive-run" / "run_state.yaml"
    archive_state_path.parent.mkdir(parents=True, exist_ok=True)
    archive_state_path.write_text(
        "year: 2022\nstage: null\niteration: 0\nasim_compiled: false\n",
        encoding="utf-8",
    )
    state = SimpleNamespace(current_year=2019)

    with pytest.raises(RuntimeError, match="Refusing rewind resume"):
        run_module._enforce_resume_rewind_guardrail(
            state=state,
            archive_state_path=str(archive_state_path),
            allow_rewind_resume=False,
        )

    run_module._enforce_resume_rewind_guardrail(
        state=state,
        archive_state_path=str(archive_state_path),
        allow_rewind_resume=True,
    )


def test_main_strict_restart_preflight_fails_on_missing_artifacts(tmp_path, monkeypatch):
    class StateStub:
        def __init__(self, run_info_path: str):
            self.run_info_path = run_info_path
            self.data_initialized = True
            self.current_year = 2018
            self.current_inner_iter = 0
            self.file_loc = None
            self.mirror_file_loc = None

        def set_run_info_path(self, path: str) -> None:
            self.run_info_path = path

        def set_data_initialized(self, initialized: bool) -> None:
            self.data_initialized = initialized

    class WorkspaceStub:
        def __init__(self, _settings, local_root: str, folder_name: str):
            self.full_path = os.path.join(local_root, folder_name)
            os.makedirs(self.full_path, exist_ok=True)

    archive_root = tmp_path / "archive-root"
    local_root = tmp_path / "local-root"
    run_name = "restart-run-002"
    state = StateStub(str(archive_root / run_name / "run_state.yaml"))
    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(archive_root),
            local_workspace_root=str(local_root),
            enable_archive_copy=False,
            output_run_name="unused",
            restart_rehydrate_mode="off",
            restart_strict=True,
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
        allow_rewind_resume=False,
    )

    missing_artifact = {
        "key": "usim_datastore_base_h5",
        "path": str(local_root / run_name / "urbansim" / "data" / "usim_000.h5"),
        "reason": "required for restart",
    }
    missing_responses = [[missing_artifact], [missing_artifact]]
    bootstrap_calls = []

    monkeypatch.setattr(run_module, "parse_args_and_settings", lambda: settings)
    monkeypatch.setattr(run_module.WorkflowState, "from_settings", lambda _s: state)
    monkeypatch.setattr(run_module, "_log_local_storage_info", lambda: None)
    monkeypatch.setattr(
        run_module,
        "resolve_consist_db_paths",
        lambda **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        run_module,
        "restore_local_consist_db_from_snapshot",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module,
        "seed_local_consist_db_from_shared",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(run_module, "_resolve_cache_epoch", lambda _settings: "test-epoch")
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(run_module.cr, "create_tracker", lambda **_kwargs: object())
    monkeypatch.setattr(run_module, "ConsistDbSnapshotManager", lambda **_kwargs: object())
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
    monkeypatch.setattr(
        run_module,
        "_find_missing_restart_local_artifacts",
        lambda **_kwargs: missing_responses.pop(0),
    )
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)
    monkeypatch.setattr(
        run_module,
        "run_bootstrap_phase",
        lambda **kwargs: bootstrap_calls.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="Strict restart preflight failed"):
        run_module.main()

    assert bootstrap_calls == []


def test_main_forces_bootstrap_when_restart_artifacts_remain_missing(
    tmp_path, monkeypatch
):
    class StopAfterBootstrap(RuntimeError):
        pass

    class StateStub:
        def __init__(self, run_info_path: str):
            self.run_info_path = run_info_path
            self.data_initialized = True
            self.current_year = 2018
            self.current_inner_iter = 0
            self.file_loc = None
            self.mirror_file_loc = None

        def set_run_info_path(self, path: str) -> None:
            self.run_info_path = path

        def set_data_initialized(self, initialized: bool) -> None:
            self.data_initialized = initialized

    class WorkspaceStub:
        def __init__(self, _settings, local_root: str, folder_name: str):
            self.full_path = os.path.join(local_root, folder_name)
            os.makedirs(self.full_path, exist_ok=True)

    archive_root = tmp_path / "archive-root"
    local_root = tmp_path / "local-root"
    run_name = "restart-run-001"
    state = StateStub(str(archive_root / run_name / "run_state.yaml"))
    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(archive_root),
            local_workspace_root=str(local_root),
            enable_archive_copy=False,
            output_run_name="unused",
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
    )

    missing_artifact = {
        "key": "usim_datastore_base_h5",
        "path": str(local_root / run_name / "urbansim" / "data" / "usim_000.h5"),
        "reason": "required for restart",
    }
    missing_responses = [[missing_artifact], [missing_artifact]]
    bootstrap_calls = []

    monkeypatch.setattr(run_module, "parse_args_and_settings", lambda: settings)
    monkeypatch.setattr(run_module.WorkflowState, "from_settings", lambda _s: state)
    monkeypatch.setattr(run_module, "_log_local_storage_info", lambda: None)
    monkeypatch.setattr(
        run_module,
        "resolve_consist_db_paths",
        lambda **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        run_module,
        "restore_local_consist_db_from_snapshot",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module,
        "seed_local_consist_db_from_shared",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(run_module, "_resolve_cache_epoch", lambda _settings: "test-epoch")
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(run_module.cr, "create_tracker", lambda **_kwargs: object())
    monkeypatch.setattr(run_module, "ConsistDbSnapshotManager", lambda **_kwargs: object())
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
    monkeypatch.setattr(
        run_module,
        "_find_missing_restart_local_artifacts",
        lambda **_kwargs: missing_responses.pop(0),
    )
    monkeypatch.setattr(
        run_module,
        "_rehydrate_missing_local_artifacts_from_archive",
        lambda **_kwargs: {
            "copied": 0,
            "skipped_existing": 0,
            "skipped_missing_archive": 1,
            "skipped_unmapped": 0,
            "copy_errors": 0,
        },
    )
    monkeypatch.setattr(run_module, "_run_resume_doctor_diagnostics", lambda **_kwargs: None)
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)

    def _fake_run_bootstrap_phase(**kwargs):
        bootstrap_calls.append(kwargs)
        raise StopAfterBootstrap("stop after bootstrap decision")

    monkeypatch.setattr(run_module, "run_bootstrap_phase", _fake_run_bootstrap_phase)

    with pytest.raises(StopAfterBootstrap, match="stop after bootstrap decision"):
        run_module.main()

    assert len(bootstrap_calls) == 1


def test_main_enables_external_paths_for_archive_to_local_tracker_topology(
    tmp_path, monkeypatch
):
    class StopAfterBootstrap(RuntimeError):
        pass

    class StateStub:
        def __init__(self):
            self.run_info_path = None
            self.data_initialized = False
            self.file_loc = None
            self.mirror_file_loc = None

        def set_run_info_path(self, path: str) -> None:
            self.run_info_path = path

        def set_data_initialized(self, initialized: bool) -> None:
            self.data_initialized = initialized

    class WorkspaceStub:
        def __init__(self, _settings, local_root: str, folder_name: str):
            self.full_path = os.path.join(local_root, folder_name)
            os.makedirs(self.full_path, exist_ok=True)

    archive_root = tmp_path / "archive-root"
    local_root = tmp_path / "local-root"
    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(archive_root),
            local_workspace_root=str(local_root),
            enable_archive_copy=False,
            output_run_name="tracker-topology-test",
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
    )
    state = StateStub()
    tracker_kwargs = {}

    def _capture_create_tracker(**kwargs):
        tracker_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(run_module, "parse_args_and_settings", lambda: settings)
    monkeypatch.setattr(run_module.WorkflowState, "from_settings", lambda _s: state)
    monkeypatch.setattr(run_module, "_log_local_storage_info", lambda: None)
    monkeypatch.setattr(
        run_module,
        "resolve_consist_db_paths",
        lambda **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        run_module,
        "restore_local_consist_db_from_snapshot",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        run_module,
        "seed_local_consist_db_from_shared",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(run_module, "_resolve_cache_epoch", lambda _settings: "test-epoch")
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(run_module.cr, "create_tracker", _capture_create_tracker)
    monkeypatch.setattr(run_module, "ConsistDbSnapshotManager", lambda **_kwargs: object())
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
    monkeypatch.setattr(
        run_module,
        "build_restart_bundle_manifest",
        lambda **_kwargs: {"artifacts": []},
    )
    monkeypatch.setattr(
        run_module,
        "write_restart_bundle_manifest",
        lambda **_kwargs: str(tmp_path / "archive-root" / "manifest.yaml"),
    )
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)

    def _stop_after_bootstrap(**_kwargs):
        raise StopAfterBootstrap("stop after tracker init")

    monkeypatch.setattr(run_module, "run_bootstrap_phase", _stop_after_bootstrap)

    with pytest.raises(StopAfterBootstrap, match="stop after tracker init"):
        run_module.main()

    archive_run_dir = Path(tracker_kwargs["run_dir"])
    workspace_mount = Path(tracker_kwargs["mounts"]["workspace"])
    assert archive_run_dir.parent == archive_root
    assert archive_run_dir.name.startswith(settings.run.output_run_name)
    assert workspace_mount.parent == local_root
    assert workspace_mount.name == archive_run_dir.name
    assert tracker_kwargs["allow_external_paths"] is True


def test_resolve_consist_db_paths_uses_local_run_dir_by_default():
    settings = SimpleNamespace(
        run=SimpleNamespace(),
        shared=SimpleNamespace(
            database=SimpleNamespace(enabled=True, path="/global/scratch/provenance.duckdb")
        ),
    )

    local_path, archive_path = snapshot_module.resolve_consist_db_paths(
        settings=settings,
        local_run_dir="/local/job123/run-abc",
        archive_run_dir="/global/scratch/run-abc",
    )

    assert local_path == "/local/job123/run-abc/.consist/provenance.duckdb"
    assert archive_path == "/global/scratch/run-abc/.consist/provenance.duckdb"


def test_resolve_consist_db_paths_uses_configured_path_when_local_disabled():
    settings = SimpleNamespace(
        run=SimpleNamespace(consist_db_local_run=False),
        shared=SimpleNamespace(
            database=SimpleNamespace(enabled=True, path="/global/scratch/provenance.duckdb")
        ),
    )

    local_path, archive_path = snapshot_module.resolve_consist_db_paths(
        settings=settings,
        local_run_dir="/local/job123/run-abc",
        archive_run_dir="/global/scratch/run-abc",
    )

    assert local_path == "/global/scratch/provenance.duckdb"
    assert archive_path == "/global/scratch/provenance.duckdb"


def test_resolve_consist_db_paths_returns_none_when_db_disabled():
    settings = SimpleNamespace(
        run=SimpleNamespace(),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path="ignored")),
    )

    local_path, archive_path = snapshot_module.resolve_consist_db_paths(
        settings=settings,
        local_run_dir="/local/job123/run-abc",
        archive_run_dir="/global/scratch/run-abc",
    )

    assert local_path is None
    assert archive_path is None


def test_mirror_consist_db_to_archive_copies_db_and_wal(tmp_path):
    local_db = tmp_path / "local" / "provenance.duckdb"
    local_db.parent.mkdir(parents=True, exist_ok=True)
    local_db.write_text("db", encoding="utf-8")
    local_wal = tmp_path / "local" / "provenance.duckdb.wal"
    local_wal.write_text("wal", encoding="utf-8")

    archive_db = tmp_path / "archive" / "provenance.duckdb"
    snapshot_module.mirror_consist_db_to_archive(str(local_db), str(archive_db))

    assert archive_db.read_text(encoding="utf-8") == "db"
    assert (tmp_path / "archive" / "provenance.duckdb.wal").read_text(
        encoding="utf-8"
    ) == "wal"


def test_restore_local_consist_db_from_snapshot_hydrates_missing_local_db(tmp_path):
    archive_run_dir = tmp_path / "archive-run"
    latest_dir = snapshot_module.snapshot_latest_dir(str(archive_run_dir))
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "provenance.duckdb").write_text("db", encoding="utf-8")
    (latest_dir / "provenance.duckdb.wal").write_text("wal", encoding="utf-8")
    (latest_dir / snapshot_module.snapshot_meta_filename("provenance.duckdb")).write_text(
        json.dumps({"snapshot_ts_utc": "2026-02-24T01:02:03Z"}),
        encoding="utf-8",
    )

    local_db = tmp_path / "local-run" / ".consist" / "provenance.duckdb"
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_restore_on_start=True,
            consist_db_restore_strict=False,
        )
    )

    restored = snapshot_module.restore_local_consist_db_from_snapshot(
        settings=settings,
        local_db_path=str(local_db),
        archive_run_dir=str(archive_run_dir),
    )

    assert restored is True
    assert local_db.read_text(encoding="utf-8") == "db"
    assert local_db.with_suffix(".duckdb.wal").read_text(encoding="utf-8") == "wal"
    assert (
        local_db.parent / snapshot_module.snapshot_meta_filename(local_db.name)
    ).exists()


def test_seed_local_consist_db_from_shared_hydrates_missing_local_db(tmp_path):
    shared_db = tmp_path / "shared" / "provenance.duckdb"
    shared_db.parent.mkdir(parents=True, exist_ok=True)
    shared_db.write_text("db", encoding="utf-8")
    shared_wal = shared_db.with_suffix(".duckdb.wal")
    shared_wal.write_text("wal", encoding="utf-8")

    local_db = tmp_path / "local-run" / ".consist" / "provenance.duckdb"
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_seed_from_shared_on_start=True,
            consist_db_seed_strict=False,
        )
    )

    seeded = snapshot_module.seed_local_consist_db_from_shared(
        settings=settings,
        local_db_path=str(local_db),
        shared_db_path=str(shared_db),
    )

    assert seeded is True
    assert local_db.read_text(encoding="utf-8") == "db"
    assert local_db.with_suffix(".duckdb.wal").read_text(encoding="utf-8") == "wal"


def test_seed_local_consist_db_from_shared_disabled_returns_false(tmp_path):
    shared_db = tmp_path / "shared" / "provenance.duckdb"
    shared_db.parent.mkdir(parents=True, exist_ok=True)
    shared_db.write_text("db", encoding="utf-8")
    local_db = tmp_path / "local-run" / ".consist" / "provenance.duckdb"
    settings = SimpleNamespace(
        run=SimpleNamespace(consist_db_seed_from_shared_on_start=False)
    )

    seeded = snapshot_module.seed_local_consist_db_from_shared(
        settings=settings,
        local_db_path=str(local_db),
        shared_db_path=str(shared_db),
    )

    assert seeded is False
    assert not local_db.exists()


def test_seed_local_consist_db_from_shared_missing_source_returns_false(tmp_path):
    local_db = tmp_path / "local-run" / ".consist" / "provenance.duckdb"
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_seed_from_shared_on_start=True,
            consist_db_seed_strict=False,
        )
    )

    seeded = snapshot_module.seed_local_consist_db_from_shared(
        settings=settings,
        local_db_path=str(local_db),
        shared_db_path=str(tmp_path / "shared" / "missing.duckdb"),
    )

    assert seeded is False
    assert not local_db.exists()


def test_snapshot_manager_triggers_outer_iteration_snapshots(tmp_path):
    tracker = DummySnapshotTracker()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=600,
            consist_db_snapshot_on_outer_iteration=True,
            consist_db_snapshot_keep_last=3,
            consist_db_local_run=True,
        )
    )
    local_db_path = tmp_path / "local" / ".consist" / "provenance.duckdb"
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(local_db_path),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    did_snapshot = manager.on_outer_iteration_boundary(year=2018, iteration=0)

    assert did_snapshot is True
    assert len(tracker.calls) == 1
    latest_db = (
        snapshot_module.snapshot_latest_dir(str(tmp_path / "archive-run"))
        / "provenance.duckdb"
    )
    assert latest_db.exists()


def test_snapshot_manager_interval_snapshot_safe_point_behavior(tmp_path):
    tracker = DummySnapshotTracker()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=3600,
            consist_db_snapshot_on_outer_iteration=False,
            consist_db_snapshot_keep_last=3,
            consist_db_local_run=True,
        )
    )
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(tmp_path / "local" / ".consist" / "provenance.duckdb"),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    first = manager.maybe_snapshot_interval(reason="safe_point_1")
    second = manager.maybe_snapshot_interval(reason="safe_point_2")

    assert first is True
    assert second is False
    assert len(tracker.calls) == 1


def test_snapshot_manager_final_snapshot(tmp_path):
    tracker = DummySnapshotTracker()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=600,
            consist_db_snapshot_on_outer_iteration=True,
            consist_db_snapshot_keep_last=3,
            consist_db_local_run=True,
        )
    )
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(tmp_path / "local" / ".consist" / "provenance.duckdb"),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    did_snapshot = manager.final_snapshot()

    assert did_snapshot is True
    assert len(tracker.calls) == 1
    assert "finalize" in Path(tracker.calls[0]["dest_path"]).parent.name


def test_snapshot_failures_are_non_fatal_and_logged(tmp_path, caplog):
    tracker = DummySnapshotTracker(fail=True)
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=600,
            consist_db_snapshot_on_outer_iteration=True,
            consist_db_snapshot_keep_last=3,
            consist_db_local_run=True,
        )
    )
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(tmp_path / "local" / ".consist" / "provenance.duckdb"),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    with caplog.at_level("WARNING"):
        did_snapshot = manager.on_outer_iteration_boundary(year=2018, iteration=0)

    assert did_snapshot is False
    assert "snapshot failed" in caplog.text.lower()


def test_snapshot_retention_keeps_expected_number_of_snapshots(tmp_path):
    tracker = DummySnapshotTracker()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            consist_db_snapshot_enabled=True,
            consist_db_snapshot_interval_seconds=0,
            consist_db_snapshot_on_outer_iteration=True,
            consist_db_snapshot_keep_last=2,
            consist_db_local_run=True,
        )
    )
    manager = snapshot_module.ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=str(tmp_path / "local" / ".consist" / "provenance.duckdb"),
        archive_run_dir=str(tmp_path / "archive-run"),
    )

    manager.snapshot(reason="snap_a", checkpoint=True)
    manager.snapshot(reason="snap_b", checkpoint=True)
    manager.snapshot(reason="snap_c", checkpoint=True)

    history_dir = snapshot_module.snapshot_history_dir(str(tmp_path / "archive-run"))
    history_entries = [path for path in history_dir.iterdir() if path.is_dir()]
    assert len(history_entries) == 2
