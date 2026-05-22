from types import SimpleNamespace
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path

import pytest
from consist.types import CacheOptions

from pilates.runtime.consist_audit import (
    emit_consist_audit_event,
    reset_consist_audit_state,
)
from pilates.runtime import bootstrap as bootstrap_runtime
from pilates.runtime import cache_recovery as cache_recovery_module
from pilates.runtime import launcher as run_module
from pilates.atlas.inputs import build_atlas_static_inputs_fallback
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

    @staticmethod
    def _realize_output_paths(output_paths):
        for path in (output_paths or {}).values():
            target = Path(path)
            if target.suffix:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("hydrated", encoding="utf-8")
            else:
                target.mkdir(parents=True, exist_ok=True)
                if target.parts[-2:] == ("urbansim", "data"):
                    (target / "usim_000.h5").write_text(
                        "usim",
                        encoding="utf-8",
                    )

    def run(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses[len(self.calls) - 1]
        if response["execute_fn"] and kwargs.get("fn") is not None:
            kwargs["fn"]()
        if response.get("hydrate_output_paths") or response["execute_fn"]:
            self._realize_output_paths(kwargs.get("output_paths"))
        return SimpleNamespace(
            cache_hit=response["cache_hit"],
            run=SimpleNamespace(
                id=response["run_id"],
                meta=response.get("meta"),
            ),
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
        (
            output_path.parent
            / snapshot_module.snapshot_meta_filename(output_path.name)
        ).write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )


class TraceCapableTrackerStub:
    def trace(self, **_kwargs):
        return nullcontext()

    def find_matching_run(self, **_kwargs):
        return None

    def register_run_output_recovery_copies(self, *_args, **_kwargs):
        return None

    def log_h5_container(
        self,
        path,
        key=None,
        direction="output",
        *,
        container_recovery_unit=None,
        child_recovery_policy=None,
        representation_policy=None,
        **_meta,
    ):
        return (
            path,
            key,
            direction,
            container_recovery_unit,
            child_recovery_policy,
            representation_policy,
        )


def _settings(cache_enabled=True, code_identity=None):
    return SimpleNamespace(
        run=SimpleNamespace(
            bootstrap_cache_enabled=cache_enabled,
            consist_code_identity=code_identity,
            region="test",
            models=SimpleNamespace(
                land_use="urbansim",
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
    assert result["run_reference"] == {"probe_run_id": "bootstrap_probe"}
    cache_options = first_call["cache_options"]
    assert isinstance(cache_options, CacheOptions)
    assert cache_options.cache_hydration == "outputs-requested"


def test_log_bootstrap_result_summary_includes_cache_miss_fields(caplog):
    bootstrap_result = {
        "bootstrap_cache_hit": True,
        "cache_probe_hit": True,
        "replay_hydration_complete": True,
        "fallback_rerun_triggered": False,
        "run_reference": {"probe_run_id": "bootstrap_probe"},
        "staged_artifact_summary": {"copied_records_total": 2},
        "cache_miss_explanation": {
            "reason": "config_changed",
            "candidate_run_id": "bootstrap_prior",
        },
    }

    with caplog.at_level(logging.INFO, logger=bootstrap_runtime.__name__):
        bootstrap_runtime.log_bootstrap_result_summary(bootstrap_result)

    assert "Bootstrap phase complete: cache_hit=True probe_hit=True" in caplog.text
    assert "cache_miss_reason=config_changed" in caplog.text
    assert "cache_miss_candidate_run_id=bootstrap_prior" in caplog.text


def test_log_bootstrap_result_summary_preserves_injected_logger(caplog):
    bootstrap_result = {
        "bootstrap_cache_hit": False,
        "cache_probe_hit": False,
        "replay_hydration_complete": False,
        "fallback_rerun_triggered": False,
        "run_reference": {"probe_run_id": "bootstrap_probe"},
        "staged_artifact_summary": {"copied_records_total": 2},
        "cache_miss_explanation": None,
    }
    injected_logger = logging.getLogger("pilates.tests.bootstrap_summary")

    with caplog.at_level(logging.INFO, logger=injected_logger.name):
        bootstrap_runtime.log_bootstrap_result_summary(
            bootstrap_result,
            log=injected_logger,
        )

    assert "Bootstrap phase complete: cache_hit=False probe_hit=False" in caplog.text
    assert caplog.records[0].name == injected_logger.name


def test_find_missing_bootstrap_workspace_artifacts_respects_surface_bootstrap_owned_keys(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "bootstrap"))
    surface = SimpleNamespace(
        is_bootstrap_owned_artifact_key=lambda key: key.startswith("beam_")
    )

    missing = bootstrap_runtime._find_missing_bootstrap_workspace_artifacts(
        settings=_settings(cache_enabled=True),
        workspace=workspace,
        surface=surface,
    )

    assert {item["key"] for item in missing} == {
        "beam_mutable_data_dir",
        "beam_region_input_dir",
        "beam_primary_config_file",
    }


def test_run_bootstrap_phase_propagates_code_identity_to_probe(monkeypatch):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[
            {"cache_hit": False, "execute_fn": True, "run_id": "bootstrap_probe"}
        ]
    )

    run_module.run_bootstrap_phase(
        tracker=tracker,
        settings=_settings(cache_enabled=True, code_identity="callable_module"),
        state=_state(),
        workspace=DummyWorkspace(),
        scenario_id="seattle-baseline",
        seed=None,
    )

    assert len(tracker.calls) == 1
    cache_options = tracker.calls[0]["cache_options"]
    assert isinstance(cache_options, CacheOptions)
    assert cache_options.code_identity == "callable_module"


def test_run_bootstrap_phase_warns_when_fast_hashing_with_bootstrap_cache(
    monkeypatch, caplog
):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[
            {"cache_hit": False, "execute_fn": True, "run_id": "bootstrap_probe"}
        ]
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(
            bootstrap_cache_enabled=True,
            consist_code_identity=None,
            consist_hashing_strategy="fast",
        )
    )

    with caplog.at_level(logging.WARNING):
        run_module.run_bootstrap_phase(
            tracker=tracker,
            settings=settings,
            state=_state(),
            workspace=DummyWorkspace(),
            scenario_id="seattle-baseline",
            seed=None,
        )

    assert "Bootstrap cache is enabled with fast hashing" in caplog.text


def test_run_bootstrap_phase_cache_miss_logs_explanation_and_writes_audit_fields(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    explanation = {
        "reason": "config_changed",
        "candidate_run_id": "bootstrap_prior",
        "confidence": "high",
        "mismatched_components": ["config_hash"],
        "details": {
            "config_keys_changed": ["run.start_year", "models.beam.enabled"],
            "identity_inputs_changed": ["start_year"],
            "fallbacks_used": ["identity_snapshot_json"],
        },
    }
    tracker = DummyTracker(
        responses=[
            {
                "cache_hit": False,
                "execute_fn": True,
                "run_id": "bootstrap_probe",
                "meta": {"cache_miss_explplanation": explanation},
            }
        ]
    )
    workspace = DummyWorkspace(full_path=str(tmp_path / "bootstrap-run"))

    with caplog.at_level(logging.DEBUG):
        result = run_module.run_bootstrap_phase(
            tracker=tracker,
            settings=_settings(cache_enabled=True),
            state=_state(),
            workspace=workspace,
            scenario_id="seattle-baseline",
            seed=12345,
        )

    assert result["cache_miss_explanation"] == explanation
    assert (
        "BOOTSTRAP CACHE MISS. Initialization executed for this workspace. "
        "reason=config_changed candidate_run_id=bootstrap_prior"
    ) in caplog.text
    assert "BOOTSTRAP cache miss details:" in caplog.text
    assert "config_keys_changed" in caplog.text
    assert "fallbacks_used" in caplog.text

    events_path = (
        Path(workspace.full_path)
        / ".workflow"
        / "diagnostics"
        / "consist_restart_audit.jsonl"
    )
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bootstrap_event = next(
        event for event in events if event["event_type"] == "bootstrap_resolution"
    )
    assert bootstrap_event["cache_miss_reason"] == "config_changed"
    assert bootstrap_event["cache_miss_candidate_run_id"] == "bootstrap_prior"
    assert bootstrap_event["cache_miss_explanation"] == explanation


def test_run_bootstrap_phase_cache_hit_replays_without_fallback_rerun(monkeypatch):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[
            {
                "cache_hit": True,
                "execute_fn": False,
                "run_id": "bootstrap_probe",
                "hydrate_output_paths": True,
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

    assert len(tracker.calls) == 1
    cache_options = tracker.calls[0]["cache_options"]
    assert isinstance(cache_options, CacheOptions)
    assert cache_options.cache_hydration == "outputs-requested"
    assert "output_paths" in tracker.calls[0]
    assert result["bootstrap_cache_hit"] is True
    assert result["cache_probe_hit"] is True
    assert result["replay_hydration_complete"] is True
    assert result["fallback_rerun"] is False
    assert result["fallback_rerun_triggered"] is False
    assert result["staged_artifact_summary"]["copied_records_total"] == 0
    assert result["run_reference"] == {"probe_run_id": "bootstrap_probe"}


def test_run_bootstrap_phase_requests_exact_workspace_invariants_for_replay(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[
            {
                "cache_hit": True,
                "execute_fn": False,
                "run_id": "bootstrap_probe",
                "hydrate_output_paths": True,
            },
        ]
    )
    workspace = DummyWorkspace(str(tmp_path / "bootstrap-run"))
    state = SimpleNamespace(
        start_year=2017,
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
    )
    surface = SimpleNamespace(
        is_bootstrap_owned_artifact_key=lambda key: (
            key.startswith("activitysim_config_settings_yaml_")
            or key
            in {
                "beam_mutable_data_dir",
                "beam_region_input_dir",
                "beam_primary_config_file",
                "usim_datastore_base_h5",
            }
        )
    )

    result = run_module.run_bootstrap_phase(
        tracker=tracker,
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
        scenario_id="seattle-baseline",
        seed=12345,
        surface=surface,
    )

    assert result["bootstrap_cache_hit"] is True
    assert result["replay_hydration_complete"] is True
    output_paths = tracker.calls[0]["output_paths"]
    assert output_paths["activitysim_config_settings_yaml_configs"].endswith(
        "activitysim/configs/configs/settings.yaml"
    )
    assert output_paths["activitysim_config_settings_yaml_configs_extended"].endswith(
        "activitysim/configs/configs_extended/settings.yaml"
    )
    assert output_paths["activitysim_config_settings_yaml_configs_mp"].endswith(
        "activitysim/configs/configs_mp/settings.yaml"
    )
    assert output_paths["activitysim_config_settings_yaml_configs_sh_compile"].endswith(
        "activitysim/configs/configs_sh_compile/settings.yaml"
    )
    assert output_paths["beam_primary_config_file"].endswith(
        "beam/input/test/beam.conf"
    )


def test_run_bootstrap_phase_cache_hit_missing_workspace_invariants_triggers_fallback_rerun(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[
            {"cache_hit": True, "execute_fn": False, "run_id": "bootstrap_probe"},
            {"cache_hit": False, "execute_fn": True, "run_id": "bootstrap_fallback"},
        ],
    )
    workspace = DummyWorkspace(str(tmp_path / "bootstrap-run"))
    state = SimpleNamespace(
        start_year=2017,
        current_major_stage=WorkflowState.Stage.supply_demand_loop,
    )

    result = run_module.run_bootstrap_phase(
        tracker=tracker,
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
        scenario_id="seattle-baseline",
        seed=12345,
    )

    assert len(tracker.calls) == 2
    fallback_options = tracker.calls[1]["cache_options"]
    assert isinstance(fallback_options, CacheOptions)
    assert fallback_options.cache_mode == "off"
    assert result["bootstrap_cache_hit"] is True
    assert result["cache_probe_hit"] is True
    assert result["replay_hydration_complete"] is False
    assert result["fallback_rerun"] is True
    assert result["fallback_rerun_triggered"] is True
    assert result["staged_artifact_summary"]["copied_records_total"] == 2
    assert result["run_reference"] == {
        "probe_run_id": "bootstrap_probe",
        "materialization_run_id": "bootstrap_fallback",
    }


def test_run_with_cache_recovery_logs_cache_miss_explanation(caplog):
    explanation = {
        "reason": "inputs_changed",
        "candidate_run_id": "step_prior",
        "confidence": "medium",
        "matched_components": ["config_hash"],
        "mismatched_components": ["input_hash"],
        "details": {
            "input_keys_added": ["beam_skims_input"],
            "input_artifact_changes": {
                "beam_skims_input": {"change": "upstream_run_drift"}
            },
        },
    }
    outputs = object()

    def _run_step(_cache_options):
        return SimpleNamespace(
            cache_hit=False,
            run=SimpleNamespace(
                id="step_run", meta={"cache_miss_explanation": explanation}
            ),
        )

    with caplog.at_level(logging.DEBUG):
        result, recovered_outputs, metadata = (
            cache_recovery_module.run_with_cache_recovery(
                stage_name="atlas",
                step_name="atlas_run",
                run_step=_run_step,
                read_outputs=lambda: outputs,
                recover_outputs=lambda _result: None,
            )
        )

    assert result.run.id == "step_run"
    assert recovered_outputs is outputs
    assert metadata["initial_cache_hit"] is False
    assert metadata["cache_miss_explanation"] == explanation
    assert (
        "[atlas] Cache miss for atlas_run. reason=inputs_changed "
        "candidate_run_id=step_prior"
    ) in caplog.text
    assert "[atlas] Cache miss details for atlas_run:" in caplog.text
    assert "input_keys_added" in caplog.text


def test_run_bootstrap_phase_writes_bootstrap_audit_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[
            {
                "cache_hit": True,
                "execute_fn": False,
                "run_id": "bootstrap_probe",
                "hydrate_output_paths": True,
            },
        ]
    )
    workspace = DummyWorkspace(full_path=str(tmp_path / "bootstrap-run"))

    result = run_module.run_bootstrap_phase(
        tracker=tracker,
        settings=_settings(cache_enabled=True),
        state=_state(),
        workspace=workspace,
        scenario_id="seattle-baseline",
        seed=12345,
    )

    diagnostics_dir = Path(workspace.full_path) / ".workflow" / "diagnostics"
    events_path = diagnostics_dir / "consist_restart_audit.jsonl"
    summary_path = diagnostics_dir / "consist_restart_audit_summary.json"

    assert result["bootstrap_cache_hit"] is True
    assert events_path.exists()
    assert summary_path.exists()

    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bootstrap_event = next(
        event for event in events if event["event_type"] == "bootstrap_resolution"
    )
    assert bootstrap_event["resolution_mode"] == "cache_hit_replay_hydrated"
    assert bootstrap_event["bootstrap_cache_enabled"] is True
    assert bootstrap_event["bootstrap_cache_hit"] is True
    assert bootstrap_event["cache_probe_hit"] is True
    assert bootstrap_event["replay_hydration_complete"] is True
    assert bootstrap_event["fallback_rerun"] is False
    assert bootstrap_event["fallback_rerun_triggered"] is False
    assert bootstrap_event["scenario_id"] == "seattle-baseline"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["event_counts"]["bootstrap_resolution"] == 1


def test_consist_audit_summary_tracks_restart_hydration_snapshot(tmp_path):
    reset_consist_audit_state()
    workspace = DummyWorkspace(full_path=str(tmp_path / "restart-audit"))

    emit_consist_audit_event(
        workspace=workspace,
        event_type="run_context",
        run_name="restart-audit-run",
    )
    emit_consist_audit_event(
        workspace=workspace,
        event_type="restart_hydration",
        frontier_stage="traffic_assignment",
        frontier_step="beam_preprocess",
        success=True,
        hydrated_keys=["beam_plans_asim_out", "households_asim_out"],
        missing_keys=[],
        producer_steps_by_key={"beam_plans_asim_out": "activitysim_postprocess"},
        fallback_reason=None,
    )

    summary_path = (
        Path(workspace.full_path)
        / ".workflow"
        / "diagnostics"
        / "consist_restart_audit_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["event_counts"]["restart_hydration"] == 1
    assert summary["restart_hydration"]["event_count"] == 1
    assert summary["restart_hydration"]["latest_frontier_stage"] == "traffic_assignment"
    assert summary["restart_hydration"]["latest_frontier_step"] == "beam_preprocess"
    assert summary["restart_hydration"]["latest_success"] is True
    assert summary["restart_hydration"]["latest_hydrated_key_count"] == 2
    assert summary["restart_hydration"]["latest_missing_key_count"] == 0


def test_run_bootstrap_phase_cache_hit_missing_replay_outputs_triggers_fallback_rerun(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[
            {"cache_hit": True, "execute_fn": False, "run_id": "bootstrap_probe"},
            {"cache_hit": False, "execute_fn": True, "run_id": "bootstrap_fallback"},
        ],
    )
    workspace = DummyWorkspace(str(tmp_path / "bootstrap-run"))

    result = run_module.run_bootstrap_phase(
        tracker=tracker,
        settings=_settings(cache_enabled=True),
        state=_state(),
        workspace=workspace,
        scenario_id="seattle-baseline",
        seed=12345,
    )

    assert len(tracker.calls) == 2
    fallback_options = tracker.calls[1]["cache_options"]
    assert isinstance(fallback_options, CacheOptions)
    assert result["bootstrap_cache_hit"] is True
    assert result["cache_probe_hit"] is True
    assert result["replay_hydration_complete"] is False
    assert result["fallback_rerun"] is True
    assert result["fallback_rerun_triggered"] is True
    assert result["staged_artifact_summary"]["copied_records_total"] == 2
    assert result["run_reference"] == {
        "probe_run_id": "bootstrap_probe",
        "materialization_run_id": "bootstrap_fallback",
    }


def test_run_bootstrap_phase_cache_hit_hydrates_declared_output_paths_without_fallback(
    monkeypatch,
):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[
            {
                "cache_hit": True,
                "execute_fn": False,
                "run_id": "bootstrap_probe",
                "hydrate_output_paths": True,
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

    assert len(tracker.calls) == 1
    assert result["bootstrap_cache_hit"] is True
    assert result["cache_probe_hit"] is True
    assert result["replay_hydration_complete"] is True
    assert result["fallback_rerun"] is False
    assert result["fallback_rerun_triggered"] is False


def test_run_bootstrap_phase_cache_disabled_uses_cache_off(monkeypatch):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[{"cache_hit": False, "execute_fn": True, "run_id": "bootstrap_off"}]
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
    assert cache_options.code_identity is None
    assert result["bootstrap_cache_hit"] is False
    assert result["run_reference"] == {"probe_run_id": "bootstrap_off"}


def test_run_bootstrap_phase_cache_disabled_preserves_code_identity(monkeypatch):
    monkeypatch.setattr(run_module, "Initialization", DummyInitialization)
    monkeypatch.setattr(run_module, "build_step_consist_kwargs", lambda *_a, **_k: {})

    tracker = DummyTracker(
        responses=[{"cache_hit": False, "execute_fn": True, "run_id": "bootstrap_off"}]
    )

    run_module.run_bootstrap_phase(
        tracker=tracker,
        settings=_settings(cache_enabled=False, code_identity="callable_module"),
        state=_state(),
        workspace=DummyWorkspace(),
        scenario_id="seattle-baseline",
        seed=None,
    )

    cache_options = tracker.calls[0]["cache_options"]
    assert isinstance(cache_options, CacheOptions)
    assert cache_options.cache_mode == "off"
    assert cache_options.code_identity == "callable_module"


def test_seed_bootstrap_artifacts_to_coupler_publishes_beam_defaults(tmp_path):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    scenario_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    plans = scenario_dir / "plans.parquet"
    households = scenario_dir / "households.parquet"
    persons = scenario_dir / "persons.parquet"
    plans.write_text("plans", encoding="utf-8")
    households.write_text("households", encoding="utf-8")
    persons.write_text("persons", encoding="utf-8")
    beam_conf = tmp_path / "beam" / "input" / "sfbay" / "beam.conf"
    beam_conf.write_text(
        'beam.inputDirectory="production/sfbay"\nfolder = ${beam.inputDirectory}"/urbansim"',
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(config="beam.conf", scenario_folder="urbansim"),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("plans_beam_in") is not None
    assert coupler.get("households_beam_in") is not None
    assert coupler.get("persons_beam_in") is not None
    assert isinstance(coupler.get("plans_beam_in"), str)
    assert isinstance(coupler.get("households_beam_in"), str)
    assert isinstance(coupler.get("persons_beam_in"), str)


def test_seed_bootstrap_artifacts_to_coupler_falls_back_to_config_exchange_folder(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    default_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    default_dir.mkdir(parents=True, exist_ok=True)
    config_dir = default_dir / "2018"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in ("plans.parquet", "households.parquet", "persons.parquet"):
        (config_dir / name).write_text(name, encoding="utf-8")

    beam_conf = tmp_path / "beam" / "input" / "sfbay" / "beam.conf"
    beam_conf.write_text(
        'beam.inputDirectory="production/sfbay"\nfolder = ${beam.inputDirectory}"/urbansim/2018"',
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(config="beam.conf", scenario_folder="urbansim"),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("plans_beam_in") is not None
    assert coupler.get("households_beam_in") is not None
    assert coupler.get("persons_beam_in") is not None


def test_seed_bootstrap_artifacts_to_coupler_prefers_yaml_scenario_folder_when_present(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    default_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    default_dir.mkdir(parents=True, exist_ok=True)
    for name in ("plans.parquet", "households.parquet", "persons.parquet"):
        (default_dir / name).write_text(f"default-{name}", encoding="utf-8")

    config_dir = default_dir / "2018"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in ("plans.parquet", "households.parquet", "persons.parquet"):
        (config_dir / name).write_text(f"config-{name}", encoding="utf-8")

    beam_conf = tmp_path / "beam" / "input" / "sfbay" / "beam.conf"
    beam_conf.write_text(
        'beam.inputDirectory="production/sfbay"\nfolder = ${beam.inputDirectory}"/urbansim/2018"',
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(config="beam.conf", scenario_folder="urbansim"),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("plans_beam_in") == str(default_dir / "plans.parquet")
    assert coupler.get("households_beam_in") == str(default_dir / "households.parquet")
    assert coupler.get("persons_beam_in") == str(default_dir / "persons.parquet")


@pytest.mark.parametrize("extension", ["csv", "csv.gz"])
def test_seed_bootstrap_artifacts_to_coupler_falls_back_to_csv_formats_when_parquet_missing(
    tmp_path, extension
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    scenario_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim" / "2018"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("plans", "households", "persons"):
        (scenario_dir / f"{stem}.{extension}").write_text(stem, encoding="utf-8")

    beam_conf = tmp_path / "beam" / "input" / "sfbay" / "beam.conf"
    beam_conf.write_text(
        'beam.inputDirectory="production/sfbay"\nfolder = ${beam.inputDirectory}"/urbansim/2018"',
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(config="beam.conf", scenario_folder="urbansim"),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("plans_beam_in") is not None
    assert coupler.get("households_beam_in") is not None
    assert coupler.get("persons_beam_in") is not None
    assert isinstance(coupler.get("plans_beam_in"), str)
    assert isinstance(coupler.get("households_beam_in"), str)
    assert isinstance(coupler.get("persons_beam_in"), str)


def test_seed_bootstrap_artifacts_to_coupler_publishes_initial_warmstart(tmp_path):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    router_dir = tmp_path / "beam" / "input" / "sfbay" / "r5" / "network"
    router_dir.mkdir(parents=True, exist_ok=True)
    warmstart = router_dir / "init.linkstats.csv.gz"
    warmstart.write_text("linkstats", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
            warmstart_linkstats_path="r5/network/init.linkstats.csv.gz",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("linkstats_warmstart") is not None
    assert isinstance(coupler.get("linkstats_warmstart"), str)


def test_seed_bootstrap_artifacts_to_coupler_uses_configured_warmstart_path(tmp_path):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    warmstart = tmp_path / "beam" / "input" / "sfbay" / "custom" / "warmstart.parquet"
    warmstart.parent.mkdir(parents=True, exist_ok=True)
    warmstart.write_text("linkstats", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
            warmstart_linkstats_path="custom/warmstart.parquet",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("linkstats_warmstart") == str(warmstart)


def test_seed_bootstrap_artifacts_to_coupler_expands_router_directory_in_warmstart_path(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    warmstart = (
        tmp_path
        / "beam"
        / "input"
        / "sfbay"
        / "r5"
        / "network"
        / "init.linkstats.parquet"
    )
    warmstart.parent.mkdir(parents=True, exist_ok=True)
    warmstart.write_text("linkstats", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
            warmstart_linkstats_path="{router_directory}/init.linkstats.parquet",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("linkstats_warmstart") == str(warmstart)


def test_seed_bootstrap_artifacts_to_coupler_does_not_probe_router_dir_when_warmstart_unset(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    router_dir = tmp_path / "beam" / "input" / "sfbay" / "r5" / "network"
    router_dir.mkdir(parents=True, exist_ok=True)
    (router_dir / "init.linkstats.csv.gz").write_text("linkstats", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("linkstats_warmstart") is None


def test_seed_bootstrap_artifacts_to_coupler_consumes_stage_boundary_policy(
    monkeypatch,
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    artifact_path = tmp_path / "policy" / "seed.txt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("policy", encoding="utf-8")

    policy_rule = SimpleNamespace(
        name="custom_bootstrap_artifact",
        semantic_keys=("custom_bootstrap_artifact",),
        resolve=lambda **_kwargs: {"custom_bootstrap_artifact": str(artifact_path)},
    )
    monkeypatch.setattr(
        run_module.bootstrap_runtime,
        "bootstrap_stage_boundary_durability_policy",
        lambda: (policy_rule,),
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand=None, travel=None),
        )
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("custom_bootstrap_artifact") == str(artifact_path)


def test_seed_bootstrap_artifacts_to_coupler_publishes_activitysim_compile_artifacts(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    zarr_dir = tmp_path / "activitysim" / "output" / "cache" / "skims.zarr"
    zarr_dir.mkdir(parents=True, exist_ok=True)
    (zarr_dir / "zarr.json").write_text("{}", encoding="utf-8")

    sharrow_cache = tmp_path / "shared_cache" / "numba"
    sharrow_cache.mkdir(parents=True, exist_ok=True)
    (sharrow_cache / "cache.bin").write_text("cache", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand="activitysim", travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("zarr_skims") is not None
    assert coupler.get("asim_sharrow_cache_dir") is not None
    assert isinstance(coupler.get("zarr_skims"), str)
    assert isinstance(coupler.get("asim_sharrow_cache_dir"), str)


def test_seed_bootstrap_artifacts_to_coupler_publishes_sharrow_cache_without_zarr(
    tmp_path,
):
    class DummyCoupler:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def view(self, _namespace):
            return self

    workspace = DummyWorkspace(full_path=str(tmp_path))
    sharrow_cache = tmp_path / "shared_cache" / "numba"
    sharrow_cache.mkdir(parents=True, exist_ok=True)
    (sharrow_cache / "cache.bin").write_text("cache", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(activity_demand="activitysim", travel="beam"),
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            router_directory="r5/network",
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
    )
    state = SimpleNamespace(full_settings=settings)
    coupler = DummyCoupler()

    run_module.bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
    )

    assert coupler.get("zarr_skims") is None
    assert coupler.get("asim_sharrow_cache_dir") == str(sharrow_cache)


def test_bootstrap_output_invariant_accepts_valid_result():
    run_module._assert_bootstrap_output_invariant(
        {
            "bootstrap_cache_hit": False,
            "run_reference": {"probe_run_id": "bootstrap_probe"},
            "staged_artifact_summary": {"copied_records_total": 2},
        }
    )


def test_bootstrap_output_invariant_accepts_zero_copied_records_for_valid_summary():
    run_module._assert_bootstrap_output_invariant(
        {
            "bootstrap_cache_hit": False,
            "run_reference": {"probe_run_id": "bootstrap_probe"},
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


def test_format_hpc_restart_command_requires_account_placeholder():
    settings = SimpleNamespace(settings_file="scenarios/settings-seattle.yaml")

    command = run_module._format_hpc_restart_command(
        settings=settings,
        archive_state_path="/tmp/pilates run/run_state.yaml",
    )

    assert (
        command
        == "./hpc/job_runner.sh -c scenarios/settings-seattle.yaml -a '<slurm_account>' -s '/tmp/pilates run/run_state.yaml'"
    )


def test_resolve_run_storage_roots_expands_env_vars(monkeypatch, tmp_path):
    archive_root = tmp_path / "archive-root"
    local_root = tmp_path / "local-root"
    monkeypatch.setenv("PILATES_TEST_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("PILATES_TEST_LOCAL_ROOT", str(local_root))

    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory="$PILATES_TEST_ARCHIVE_ROOT",
            local_workspace_root="$PILATES_TEST_LOCAL_ROOT",
        )
    )

    resolved_archive_root, resolved_local_root = run_module._resolve_run_storage_roots(
        settings
    )

    assert resolved_archive_root == str(archive_root.resolve())
    assert resolved_local_root == str(local_root.resolve())


def test_resolve_run_storage_roots_defaults_local_to_archive_root(tmp_path):
    archive_root = tmp_path / "archive-root"
    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(archive_root),
            local_workspace_root=None,
        )
    )

    resolved_archive_root, resolved_local_root = run_module._resolve_run_storage_roots(
        settings
    )

    assert resolved_archive_root == str(archive_root.resolve())
    assert resolved_local_root == str(archive_root.resolve())


def test_prepare_run_context_resolves_storage_tracker_and_state_paths(
    tmp_path, monkeypatch
):
    class WorkspaceStub:
        def __init__(self, _settings, local_root: str, folder_name: str):
            self.full_path = os.path.join(local_root, folder_name)
            os.makedirs(self.full_path, exist_ok=True)

    class SnapshotStub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class TrackerStub:
        def __init__(self):
            self.trace_calls = []

        def trace(self, **kwargs):
            self.trace_calls.append(kwargs)
            return nullcontext()

        def find_matching_run(self, **_kwargs):
            return None

        def register_run_output_recovery_copies(self, *_args, **_kwargs):
            return None

        def log_h5_container(
            self,
            path,
            key=None,
            direction="output",
            *,
            container_recovery_unit=None,
            child_recovery_policy=None,
            representation_policy=None,
            **_meta,
        ):
            return (
                path,
                key,
                direction,
                container_recovery_unit,
                child_recovery_policy,
                representation_policy,
            )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="seattle",
            output_directory=str(tmp_path / "archive-root"),
            local_workspace_root=str(tmp_path / "local-root"),
            enable_archive_copy=False,
            output_run_name="prepare-context-test",
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
        settings_file="scenarios/settings-seattle.yaml",
    )
    state = SimpleNamespace(
        run_info_path=None,
        data_initialized=False,
    )
    state.set_run_info_path = lambda path: setattr(state, "run_info_path", path)

    tracker = TrackerStub()
    create_tracker_kwargs = {}

    def _create_tracker(**kwargs):
        create_tracker_kwargs.update(kwargs)
        return tracker

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
    monkeypatch.setattr(run_module, "_resolve_cache_epoch", lambda _settings: 7)
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(run_module.cr, "create_tracker", _create_tracker)
    monkeypatch.setattr(
        run_module,
        "ConsistDbSnapshotManager",
        lambda **kwargs: SnapshotStub(**kwargs),
    )
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)

    prepared = run_module._prepare_run_context(settings=settings, state=state)

    assert prepared.settings is settings
    assert prepared.state is state
    assert prepared.tracker is tracker
    assert prepared.cache_epoch == 7
    assert prepared.is_restart_run is False
    assert prepared.run_name.startswith("pilates-run--seattle--prepare-context-test--")
    assert prepared.archive_run_dir.startswith(str(tmp_path / "archive-root"))
    assert prepared.local_run_dir.startswith(str(tmp_path / "local-root"))
    assert state.file_loc == prepared.archive_state_path
    assert state.mirror_file_loc == prepared.local_state_path
    assert state.run_info_path == prepared.archive_state_path
    assert create_tracker_kwargs["run_dir"] == prepared.archive_run_dir
    assert create_tracker_kwargs["mounts"]["workspace"] == prepared.local_run_dir
    assert tracker.trace_calls == [
        {
            "name": "workspace_setup",
            "model": "pilates_orchestrator",
            "tags": [f"scenario_id:{prepared.scenario_id}"],
        }
    ]


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
            region="seattle",
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
    monkeypatch.setattr(
        run_module.cr, "create_tracker", lambda **_kwargs: TraceCapableTrackerStub()
    )
    monkeypatch.setattr(
        run_module, "ConsistDbSnapshotManager", lambda **_kwargs: SnapshotStub()
    )
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
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
    assert (
        "./hpc/job_runner.sh -c scenarios/settings-seattle.yaml -a '<slurm_account>' -s "
        in caplog.text
    )
    assert "run_state.yaml" in caplog.text


def test_pre_scenario_failure_emits_registered_failure_hooks():
    class FailureSink:
        def __init__(self):
            self.calls = []

        def on_run_failed(self, run, error):
            self.calls.append((run, error))

    notifier = FailureSink()
    publisher = FailureSink()
    prepared = SimpleNamespace(
        run_name="pilates-run--seattle--base",
        run_notifier=notifier,
        run_publisher=publisher,
    )
    error = RuntimeError("contract failed")

    run_module._emit_pre_scenario_failure(prepared, error)

    assert notifier.calls[0][1] is error
    assert publisher.calls[0][1] is error
    failure_run = notifier.calls[0][0]
    assert failure_run.id == "pilates-run--seattle--base"
    assert failure_run.model_name == "pilates_orchestrator"
    assert failure_run.status == "failed"
    assert "scenario_header" in failure_run.tags
    assert failure_run.meta["launcher_failure"] is True


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


def test_restart_preflight_skips_activitysim_locals_outside_supply_demand_stage(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    state = SimpleNamespace(
        current_major_stage=WorkflowState.Stage.vehicle_ownership_model
    )

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
    state = SimpleNamespace(
        current_major_stage=WorkflowState.Stage.vehicle_ownership_model
    )

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=state,
        workspace=workspace,
    )

    paths = {item["path"] for item in missing}
    assert any(path.endswith("atlas/atlas_input/psid_names.Rdat") for path in paths)
    assert any(
        path.endswith("atlas/atlas_input/accessbility_2015.RData") for path in paths
    )


def test_restart_preflight_does_not_require_zarr_skims_when_resuming_compiled_supply_demand(
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
    assert "zarr_skims" not in keys


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


def test_restart_preflight_consumes_shared_policy_hook(tmp_path, monkeypatch):
    workspace = DummyWorkspace(str(tmp_path / "local-run"))
    policy_path = tmp_path / "policy" / "seed.txt"

    monkeypatch.setattr(
        run_module.restart_runtime,
        "restart_required_local_artifact_policy",
        lambda: (
            SimpleNamespace(
                name="custom_restart_artifact",
                semantic_keys=("custom_restart_artifact",),
                resolve=lambda **_kwargs: {"custom_restart_artifact": str(policy_path)},
                notes="test policy",
            ),
        ),
    )

    missing = run_module._find_missing_restart_local_artifacts(
        settings=_restart_settings(),
        state=SimpleNamespace(),
        workspace=workspace,
    )

    assert missing == [
        {
            "key": "custom_restart_artifact",
            "path": str(policy_path.resolve()),
            "reason": "Restart policy 'custom_restart_artifact' requires custom_restart_artifact (test policy)",
        }
    ]


def test_build_atlas_static_inputs_fallback_uses_atlas_static_key_scheme(tmp_path):
    settings = _restart_settings()
    workspace = DummyWorkspace(str(tmp_path / "local-run"), settings=settings)
    atlas_input_dir = Path(workspace.get_atlas_mutable_input_dir())
    (atlas_input_dir / "psid_names.Rdat").parent.mkdir(parents=True, exist_ok=True)
    (atlas_input_dir / "psid_names.Rdat").write_text("psid", encoding="utf-8")
    (atlas_input_dir / "adopt" / "baseline" / "used_vehicles_2017.csv").parent.mkdir(
        parents=True, exist_ok=True
    )
    (atlas_input_dir / "adopt" / "baseline" / "used_vehicles_2017.csv").write_text(
        "used", encoding="utf-8"
    )

    fallback = build_atlas_static_inputs_fallback(workspace)

    assert fallback["psid_names"] == str(atlas_input_dir / "psid_names.Rdat")
    assert fallback["adopt/baseline/used_vehicles_2017"] == str(
        atlas_input_dir / "adopt" / "baseline" / "used_vehicles_2017.csv"
    )
    assert "atlas_static_psid_names.Rdat" not in fallback


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
    monkeypatch.delenv("PILATES_LOCAL_RUN_DIR", raising=False)
    monkeypatch.delenv("PILATES_ARCHIVE_RUN_DIR", raising=False)
    monkeypatch.delenv("PILATES_ENABLE_ARCHIVE_COPY", raising=False)
    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="seattle",
            output_directory=str(archive_root),
            local_workspace_root=str(local_root),
            enable_archive_copy=False,
            output_run_name="tracker-topology-test",
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
        settings_file=None,
    )
    state = StateStub()
    tracker_kwargs = {}

    def _capture_create_tracker(**kwargs):
        tracker_kwargs.update(kwargs)
        return TraceCapableTrackerStub()

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
    monkeypatch.setattr(
        run_module, "_resolve_cache_epoch", lambda _settings: "test-epoch"
    )
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(run_module.cr, "create_tracker", _capture_create_tracker)
    monkeypatch.setattr(
        run_module, "ConsistDbSnapshotManager", lambda **_kwargs: object()
    )
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)

    def _stop_after_bootstrap(**_kwargs):
        raise StopAfterBootstrap("stop after tracker init")

    monkeypatch.setattr(run_module, "run_bootstrap_phase", _stop_after_bootstrap)

    with pytest.raises(StopAfterBootstrap, match="stop after tracker init"):
        run_module.main()

    archive_run_dir = Path(tracker_kwargs["run_dir"])
    repo_root = Path(__file__).resolve().parents[1]
    inputs_mount = Path(tracker_kwargs["mounts"]["inputs"])
    workspace_mount = Path(tracker_kwargs["mounts"]["workspace"])
    project_root = Path(tracker_kwargs["project_root"])
    assert archive_run_dir.parent == archive_root
    assert archive_run_dir.name.startswith(
        f"pilates-run--{settings.run.region}--{settings.run.output_run_name}--"
    )
    assert inputs_mount == repo_root.resolve()
    assert workspace_mount.parent == local_root
    assert workspace_mount.name == archive_run_dir.name
    assert project_root == repo_root.resolve()
    assert tracker_kwargs["allow_external_paths"] is True
    assert os.environ["PILATES_LOCAL_RUN_DIR"] == str(workspace_mount)
    assert os.environ["PILATES_ARCHIVE_RUN_DIR"] == str(archive_run_dir)
    assert os.environ["PILATES_ENABLE_ARCHIVE_COPY"] == "0"


def test_main_restart_strict_defers_missing_artifact_failure_until_after_bootstrap(
    tmp_path, monkeypatch
):
    class StopAfterBootstrap(RuntimeError):
        pass

    class SnapshotStub:
        def final_snapshot(self):
            return True

    class StateStub:
        def __init__(self, run_info_path: str):
            self.run_info_path = run_info_path
            self.data_initialized = True
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
    run_name = "restart-strict-test"
    archive_run_dir = archive_root / run_name
    archive_run_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = archive_run_dir / "run_state.yaml"
    run_state_path.write_text("state", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(archive_root),
            local_workspace_root=str(local_root),
            enable_archive_copy=False,
            output_run_name="unused-on-restart",
            restart_strict=True,
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
        settings_file=None,
        allow_rewind_resume=False,
    )
    state = StateStub(str(run_state_path))

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
    monkeypatch.setattr(
        run_module, "_resolve_cache_epoch", lambda _settings: "test-epoch"
    )
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(
        run_module.cr, "create_tracker", lambda **_kwargs: TraceCapableTrackerStub()
    )
    monkeypatch.setattr(
        run_module, "ConsistDbSnapshotManager", lambda **_kwargs: SnapshotStub()
    )
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)
    missing_before_bootstrap = [
        {
            "key": "activitysim_config_settings_yaml_configs",
            "path": "/missing/settings.yaml",
            "reason": "test",
        }
    ]
    missing_sequences = [
        list(missing_before_bootstrap),
        [],
    ]

    def _missing_artifacts(**_kwargs):
        if not missing_sequences:
            return []
        return missing_sequences.pop(0)

    monkeypatch.setattr(
        run_module,
        "_find_missing_restart_local_artifacts",
        _missing_artifacts,
    )
    monkeypatch.setattr(
        run_module,
        "run_bootstrap_phase",
        lambda **_kwargs: {
            "bootstrap_cache_hit": False,
            "run_reference": {"probe_run_id": "bootstrap-run"},
            "staged_artifact_summary": {"copied_records_total": 0},
        },
    )
    monkeypatch.setattr(
        run_module,
        "_build_scenario_runtime_contract",
        lambda **_kwargs: {
            "scenario_kwargs": {},
            "schema_steps_all": (),
            "schema_steps_enabled": (),
            "coupler_schema": {},
            "required_output_keys": (),
        },
    )
    monkeypatch.setattr(
        run_module.cr,
        "scenario",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            StopAfterBootstrap("reached scenario")
        ),
    )

    with pytest.raises(StopAfterBootstrap, match="reached scenario"):
        run_module.main()

    assert missing_sequences == []


def test_main_restart_preflight_uses_surface_deferred_artifact_classification(
    tmp_path, monkeypatch, caplog
):
    class StopAfterBootstrap(RuntimeError):
        pass

    class StateStub:
        def __init__(self, run_info_path: str):
            self.run_info_path = run_info_path
            self.data_initialized = True
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
    run_name = "restart-surface-deferred"
    archive_run_dir = archive_root / run_name
    archive_run_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = archive_run_dir / "run_state.yaml"
    run_state_path.write_text("state", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(archive_root),
            local_workspace_root=str(local_root),
            enable_archive_copy=False,
            output_run_name="unused-on-restart",
            restart_strict=False,
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
        settings_file=None,
        allow_rewind_resume=False,
    )
    state = StateStub(str(run_state_path))
    surface = SimpleNamespace(
        profile=SimpleNamespace(),
        is_restart_prebootstrap_deferred_artifact_key=lambda key: key == "deferred_key",
    )

    monkeypatch.setattr(
        run_module, "build_enabled_workflow_surface", lambda *_args, **_kwargs: surface
    )
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
    monkeypatch.setattr(
        run_module, "_resolve_cache_epoch", lambda _settings: "test-epoch"
    )
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(
        run_module.cr, "create_tracker", lambda **_kwargs: TraceCapableTrackerStub()
    )
    monkeypatch.setattr(
        run_module, "ConsistDbSnapshotManager", lambda **_kwargs: object()
    )
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)
    monkeypatch.setattr(
        run_module,
        "_find_missing_restart_local_artifacts",
        lambda **_kwargs: [
            {"key": "blocking_key", "path": "/missing/blocking", "reason": "blocking"},
            {"key": "deferred_key", "path": "/missing/deferred", "reason": "deferred"},
        ],
    )
    monkeypatch.setattr(
        run_module,
        "run_bootstrap_phase",
        lambda **_kwargs: (_ for _ in ()).throw(
            StopAfterBootstrap("stop after classification")
        ),
    )

    with caplog.at_level("INFO"):
        with pytest.raises(StopAfterBootstrap, match="stop after classification"):
            run_module.main(settings=settings, state=state)

    assert (
        "missing local workspace inputs while data_initialized=True: blocking_key:/missing/blocking"
        in caplog.text
    )
    assert (
        "deferring bootstrap-owned workspace inputs until bootstrap hydration: deferred_key:/missing/deferred"
        in caplog.text
    )


def test_main_restart_strict_still_fails_when_required_artifacts_remain_missing(
    tmp_path, monkeypatch
):
    class WorkspaceStub:
        def __init__(self, _settings, local_root: str, folder_name: str):
            self.full_path = os.path.join(local_root, folder_name)
            os.makedirs(self.full_path, exist_ok=True)

    archive_root = tmp_path / "archive-root"
    local_root = tmp_path / "local-root"
    run_name = "restart-strict-missing"
    archive_run_dir = archive_root / run_name
    archive_run_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = archive_run_dir / "run_state.yaml"
    run_state_path.write_text("state", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(archive_root),
            local_workspace_root=str(local_root),
            enable_archive_copy=False,
            output_run_name="unused-on-restart",
            restart_strict=True,
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
        settings_file=None,
        allow_rewind_resume=False,
    )
    state = SimpleNamespace(
        run_info_path=str(run_state_path),
        data_initialized=True,
        file_loc=None,
        mirror_file_loc=None,
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
    monkeypatch.setattr(
        run_module, "_resolve_cache_epoch", lambda _settings: "test-epoch"
    )
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(
        run_module.cr, "create_tracker", lambda **_kwargs: TraceCapableTrackerStub()
    )
    monkeypatch.setattr(
        run_module, "ConsistDbSnapshotManager", lambda **_kwargs: object()
    )
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)

    missing = [
        {"key": "usim_datastore_base_h5", "path": "/missing/base.h5", "reason": "test"}
    ]

    monkeypatch.setattr(
        run_module,
        "_find_missing_restart_local_artifacts",
        lambda **_kwargs: list(missing),
    )
    monkeypatch.setattr(
        run_module,
        "run_bootstrap_phase",
        lambda **_kwargs: {
            "bootstrap_cache_hit": False,
            "run_reference": {"probe_run_id": "bootstrap-run"},
            "staged_artifact_summary": {"copied_records_total": 0},
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Strict restart preflight failed; required restart artifacts are still missing after restart bootstrap",
    ):
        run_module.main()


def test_main_restart_strict_fails_without_atlas_repair_paths(tmp_path, monkeypatch):

    class SnapshotStub:
        def final_snapshot(self):
            return True

    class WorkspaceStub:
        def __init__(self, _settings, local_root: str, folder_name: str):
            self.full_path = os.path.join(local_root, folder_name)
            os.makedirs(self.full_path, exist_ok=True)

        def get_atlas_mutable_input_dir(self):
            return os.path.join(self.full_path, "atlas", "atlas_input")

    archive_root = tmp_path / "archive-root"
    local_root = tmp_path / "local-root"
    run_name = "restart-strict-atlas-repair"
    archive_run_dir = archive_root / run_name
    archive_run_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = archive_run_dir / "run_state.yaml"
    run_state_path.write_text("state", encoding="utf-8")
    year2017 = archive_run_dir / "atlas" / "atlas_input" / "year2017"
    year2017.mkdir(parents=True, exist_ok=True)
    for filename in (
        "households.csv",
        "blocks.csv",
        "persons.csv",
        "residential.csv",
        "jobs.csv",
    ):
        (year2017 / filename).write_text("seed\n", encoding="utf-8")

    year2021 = archive_run_dir / "atlas" / "atlas_input" / "year2021"
    year2021.mkdir(parents=True, exist_ok=True)
    for filename in (
        "households.csv",
        "blocks.csv",
        "persons.csv",
        "grave.csv",
        "residential.csv",
        "jobs.csv",
    ):
        (year2021 / filename).write_text("prior\n", encoding="utf-8")
    (year2021 / "vehicles_output.RData").write_text("vehicles\n", encoding="utf-8")
    (year2021 / "households_output.RData").write_text("households\n", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            output_directory=str(archive_root),
            local_workspace_root=str(local_root),
            enable_archive_copy=False,
            output_run_name="unused-on-restart",
            restart_strict=True,
            models=SimpleNamespace(vehicle_ownership="atlas"),
        ),
        shared=SimpleNamespace(database=SimpleNamespace(enabled=False, path=None)),
        settings_file=None,
        allow_rewind_resume=False,
    )
    state = SimpleNamespace(
        Stage=WorkflowState.Stage,
        run_info_path=str(run_state_path),
        data_initialized=True,
        file_loc=None,
        mirror_file_loc=None,
        current_major_stage=WorkflowState.Stage.vehicle_ownership_model,
        current_year=2023,
        start_year=2017,
        forecast_year=2023,
    )
    state.set_run_info_path = lambda path: setattr(state, "run_info_path", path)
    state.set_data_initialized = lambda initialized: setattr(
        state, "data_initialized", initialized
    )

    def _atlas_only_missing(*, workspace, **_kwargs):
        missing = []
        required_paths = {
            "atlas_restart_seed::2017::households": os.path.join(
                workspace.get_atlas_mutable_input_dir(),
                "year2017",
                "households.csv",
            ),
            "atlas_restart_seed::2017::blocks": os.path.join(
                workspace.get_atlas_mutable_input_dir(),
                "year2017",
                "blocks.csv",
            ),
            "atlas_restart_prior::2021::vehicles_output_RData": os.path.join(
                workspace.get_atlas_mutable_input_dir(),
                "year2021",
                "vehicles_output.RData",
            ),
            "atlas_restart_prior::2021::households": os.path.join(
                workspace.get_atlas_mutable_input_dir(),
                "year2021",
                "households.csv",
            ),
            "atlas_restart_prior::2021::grave": os.path.join(
                workspace.get_atlas_mutable_input_dir(),
                "year2021",
                "grave.csv",
            ),
            "atlas_restart_prior::2021::households_output_RData": os.path.join(
                workspace.get_atlas_mutable_input_dir(),
                "year2021",
                "households_output.RData",
            ),
        }
        for key, path in required_paths.items():
            if not os.path.exists(path):
                missing.append(
                    {
                        "key": key,
                        "path": path,
                        "reason": "test",
                    }
                )
        return missing

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
    monkeypatch.setattr(
        run_module, "_resolve_cache_epoch", lambda _settings: "test-epoch"
    )
    monkeypatch.setattr(run_module, "_get_consist_schemas", lambda: None)
    monkeypatch.setattr(
        run_module.cr, "create_tracker", lambda **_kwargs: TraceCapableTrackerStub()
    )
    monkeypatch.setattr(
        run_module, "ConsistDbSnapshotManager", lambda **_kwargs: SnapshotStub()
    )
    monkeypatch.setattr(run_module, "Workspace", WorkspaceStub)
    monkeypatch.setattr(run_module.cr, "set_tracker", lambda _tracker: None)
    monkeypatch.setattr(
        run_module, "_find_missing_restart_local_artifacts", _atlas_only_missing
    )
    monkeypatch.setattr(
        run_module,
        "run_bootstrap_phase",
        lambda **_kwargs: {
            "bootstrap_cache_hit": False,
            "run_reference": {"probe_run_id": "bootstrap-run"},
            "staged_artifact_summary": {"copied_records_total": 0},
        },
    )
    monkeypatch.setattr(
        run_module,
        "_build_scenario_runtime_contract",
        lambda **_kwargs: {
            "scenario_kwargs": {},
            "schema_steps_all": (),
            "schema_steps_enabled": (),
            "coupler_schema": {},
            "required_output_keys": (),
        },
    )
    with pytest.raises(
        RuntimeError,
        match="Strict restart preflight failed; required restart artifacts are still missing after restart bootstrap",
    ):
        run_module.main()

    local_run_dir = local_root / run_name
    assert not (local_run_dir / "atlas" / "atlas_input" / "year2017").exists()
    assert not (local_run_dir / "atlas" / "atlas_input" / "year2021").exists()


def test_resolve_consist_db_paths_uses_local_run_dir_by_default():
    settings = SimpleNamespace(
        run=SimpleNamespace(),
        shared=SimpleNamespace(
            database=SimpleNamespace(
                enabled=True, path="/global/scratch/provenance.duckdb"
            )
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
            database=SimpleNamespace(
                enabled=True, path="/global/scratch/provenance.duckdb"
            )
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
    (
        latest_dir / snapshot_module.snapshot_meta_filename("provenance.duckdb")
    ).write_text(
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
