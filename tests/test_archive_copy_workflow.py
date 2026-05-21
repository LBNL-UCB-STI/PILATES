from pathlib import Path
from types import SimpleNamespace
import json
import logging
import queue
import hashlib

import pytest

from pilates.runtime import consist_audit
from pilates.utils import coupler_helpers as ch
from pilates.workflows.artifact_keys import ASIM_SHARROW_CACHE_DIR
from pilates.workflows.orchestration import StepRef, run_workflow
from pilates.workflows.steps import StepOutputsHolder


class DummyCoupler:
    def __init__(self) -> None:
        self.values = {}

    def set(self, key, value):
        self.values[key] = value

    def get(self, key, default=None):
        return self.values.get(key, default)

    def update(self, mapping):
        self.values.update(mapping)


class DummyWorkspace:
    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def full_path(self) -> str:
        return str(self._root)


class ExecutingScenario:
    def run(self, **kwargs):
        fn = kwargs["fn"]
        execution_options = kwargs.get("execution_options")
        runtime_kwargs = kwargs.get("runtime_kwargs") or getattr(
            execution_options, "runtime_kwargs", None
        )
        runtime_kwargs = dict(runtime_kwargs or {})
        fn(**runtime_kwargs)
        return SimpleNamespace(cache_hit=False)


def _write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _lifecycle_summary(root: Path) -> dict:
    return json.loads(
        (
            root / ".workflow" / "diagnostics" / "artifact_lifecycle_audit_summary.json"
        ).read_text(encoding="utf-8")
    )


@pytest.fixture(autouse=True)
def _reset_archive_state(monkeypatch):
    ch.stop_archive_worker(timeout=1)
    ch._archive_queue = None
    ch._archive_thread = None
    ch._archive_pending_tasks.clear()
    ch._archive_queued_destinations.clear()
    ch._archive_inflight_signature.clear()
    ch._archive_last_copied_signature.clear()
    ch._archive_last_copied_details.clear()
    ch._archive_last_recovery_root_registration_signature.clear()
    consist_audit.reset_consist_audit_state()
    monkeypatch.delenv("PILATES_ENABLE_ARCHIVE_COPY", raising=False)
    monkeypatch.delenv("PILATES_LOCAL_RUN_DIR", raising=False)
    monkeypatch.delenv("PILATES_ARCHIVE_RUN_DIR", raising=False)
    yield
    ch.stop_archive_worker(timeout=1)
    ch._archive_queue = None
    ch._archive_thread = None
    ch._archive_pending_tasks.clear()
    ch._archive_queued_destinations.clear()
    ch._archive_inflight_signature.clear()
    ch._archive_last_copied_signature.clear()
    ch._archive_last_copied_details.clear()
    ch._archive_last_recovery_root_registration_signature.clear()
    consist_audit.reset_consist_audit_state()


def test_archive_copy_copies_file_and_preserves_relative_path(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "beam" / "output" / "linkstats.csv.gz"
    _write_file(source, "linkstats")

    ch._enqueue_archive_copy("linkstats", str(source))
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = archive_root / "beam" / "output" / "linkstats.csv.gz"
    assert archived.exists()
    assert archived.read_text() == "linkstats"


def test_local_archive_copy_does_not_write_recovery_roots(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    recovery_root_calls = []
    tracker = SimpleNamespace(
        current_consist=None,
        set_artifact_recovery_roots=lambda *args, **kwargs: recovery_root_calls.append(
            (args, kwargs)
        ),
    )
    monkeypatch.setattr(ch.cr, "current_tracker", lambda: tracker)
    monkeypatch.setattr(ch.cr, "log_output", lambda *args, **kwargs: "artifact")

    source = local_root / "beam" / "output" / "inputs" / "households.csv.gz"
    _write_file(source, "households")
    ch.log_output_only(
        key="beam_input_households_archived",
        path=str(source),
        description="mock BEAM input snapshot",
        facet={
            "artifact_family": "beam_input_archived",
            "source_role": "households_beam_in",
            "snapshot_role": "beam_input_households",
            "snapshot_reason": "exact_rewind",
            "storage_event": "snapshot_copy",
            "year": 2030,
            "iteration": 0,
        },
    )
    ch.flush_archive_queue(timeout=5)

    assert recovery_root_calls == []
    assert (
        _lifecycle_summary(local_root)["local_to_scratch_recovery_roots_written"] == 0
    )


def test_consist_audit_files_are_archived_with_separate_local_and_archive_roots(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    workspace = DummyWorkspace(local_root)

    consist_audit.emit_consist_audit_event(
        workspace=workspace,
        event_type="run_context",
        scenario_id="seattle-baseline",
        restart_run=False,
        workspace_root=str(local_root),
    )
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    local_events = (
        local_root / ".workflow" / "diagnostics" / "consist_restart_audit.jsonl"
    )
    local_summary = (
        local_root / ".workflow" / "diagnostics" / "consist_restart_audit_summary.json"
    )
    archive_events = (
        archive_root / ".workflow" / "diagnostics" / "consist_restart_audit.jsonl"
    )
    archive_summary = (
        archive_root
        / ".workflow"
        / "diagnostics"
        / "consist_restart_audit_summary.json"
    )

    assert local_events.exists()
    assert local_summary.exists()
    assert archive_events.exists()
    assert archive_summary.exists()
    assert archive_events.read_text() == local_events.read_text()
    assert archive_summary.read_text() == local_summary.read_text()


def test_consist_audit_rotates_attempt_scoped_files_without_overwriting_history(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    workspace = DummyWorkspace(local_root)
    diagnostics_dir = local_root / ".workflow" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    legacy_events = diagnostics_dir / "consist_restart_audit.jsonl"
    legacy_summary = diagnostics_dir / "consist_restart_audit_summary.json"
    legacy_events.write_text("legacy-flat\n", encoding="utf-8")
    legacy_summary.write_text('{"legacy": true}', encoding="utf-8")

    consist_audit.emit_consist_audit_event(
        workspace=workspace,
        event_type="run_context",
        scenario_id="scenario-a",
        run_name="run-a",
        restart_run=False,
        workspace_root=str(local_root),
    )
    consist_audit.emit_consist_audit_event(
        workspace=workspace,
        event_type="step_resolution",
        stage_name="land_use",
        step_name="urbansim_preprocess",
        resolution_mode="executed",
        workspace_root=str(local_root),
    )
    ch.flush_archive_queue(timeout=5)

    first_attempt_dirs = sorted((diagnostics_dir / "attempts").glob("*"))
    assert len(first_attempt_dirs) == 1
    first_attempt_dir = first_attempt_dirs[0]
    first_attempt_events = first_attempt_dir / "consist_restart_audit.jsonl"
    first_attempt_summary = first_attempt_dir / "consist_restart_audit_summary.json"
    assert first_attempt_events.exists()
    assert first_attempt_summary.exists()
    assert "legacy-flat" not in first_attempt_events.read_text(encoding="utf-8")

    consist_audit.emit_consist_audit_event(
        workspace=workspace,
        event_type="run_context",
        scenario_id="scenario-a",
        run_name="run-b",
        restart_run=True,
        workspace_root=str(local_root),
    )
    consist_audit.emit_consist_audit_event(
        workspace=workspace,
        event_type="step_resolution",
        stage_name="land_use",
        step_name="urbansim_run",
        resolution_mode="executed",
        workspace_root=str(local_root),
    )
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    attempt_dirs = sorted((diagnostics_dir / "attempts").glob("*"))
    assert len(attempt_dirs) == 2

    latest_events = legacy_events.read_text(encoding="utf-8").splitlines()
    assert len([line for line in latest_events if line.strip()]) == 2
    assert "urbansim_run" in legacy_events.read_text(encoding="utf-8")
    assert "urbansim_preprocess" not in legacy_events.read_text(encoding="utf-8")

    archived_attempt_dirs = sorted(
        (archive_root / ".workflow" / "diagnostics" / "attempts").glob("*")
    )
    assert len(archived_attempt_dirs) == 2
    assert (
        archive_root / ".workflow" / "diagnostics" / "consist_restart_audit.jsonl"
    ).exists()
    assert (
        archive_root
        / ".workflow"
        / "diagnostics"
        / "consist_restart_audit_summary.json"
    ).exists()


def test_artifact_lifecycle_summary_updates_on_log_and_copy(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "beam" / "output" / "inputs" / "plans.csv.gz"
    _write_file(source, "plans")

    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="beam_input_plans_archived",
        path=str(source),
        artifact_family="beam_input_archived",
        source_role="plans_beam_in",
        snapshot_role="beam_input_plans",
        snapshot_reason="exact_rewind",
        storage_event="snapshot_copy",
        year=2030,
        iteration=2,
        artifact_id="artifact-1",
        producing_run_id="run-1",
    )

    summary_after_log = _lifecycle_summary(local_root)
    assert summary_after_log["snapshot_artifacts_logged"] == 1
    assert summary_after_log["snapshot_artifacts_missing_required_facets"] == 0
    assert summary_after_log["event_counts"]["artifact_logged"] == 1

    assert (
        ch.archive_copy_now(
            key="beam_input_plans_archived",
            path=str(source),
        )
        is True
    )
    ch.flush_archive_queue(timeout=5)

    summary_after_copy = _lifecycle_summary(local_root)
    assert (
        summary_after_copy["copied_artifacts_eligible_for_recovery_root_registration"]
        == 1
    )
    assert (
        summary_after_copy[
            "phase2_candidate_copied_artifacts_eligible_for_recovery_root_registration"
        ]
        == 0
    )
    assert summary_after_copy["local_to_scratch_recovery_roots_written"] == 0
    assert summary_after_copy["contract_status_by_family"]["beam_input_archived"] == (
        "transitional"
    )
    assert summary_after_copy["phase2_candidate_families"] == [
        "usim_input_archive",
        "usim_population_source_h5",
    ]
    assert summary_after_copy["safe_families_for_phase2"] == []
    assert "beam_input_archived" not in summary_after_copy["phase2_candidate_families"]


def test_artifact_lifecycle_summary_classifies_blockers(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    copied_before_log = local_root / "beam" / "output" / "persons.csv.gz"
    _write_file(copied_before_log, "persons")
    assert (
        ch.archive_copy_now(
            key="beam_input_persons_archived",
            path=str(copied_before_log),
        )
        is True
    )
    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="beam_input_persons_archived",
        path=str(copied_before_log),
        artifact_family="beam_input_archived",
        source_role="persons_beam_in",
        snapshot_role="beam_input_persons",
        snapshot_reason="exact_rewind",
        storage_event="snapshot_copy",
        year=2030,
        iteration=0,
    )

    zarr_dir = local_root / "activitysim" / "cache" / "skims.zarr"
    _write_file(zarr_dir / "0" / "values", "zarr")
    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="asim_input_skims_zarr_archived",
        path=str(zarr_dir),
        artifact_family="asim_input_archived",
        source_role="zarr_skims",
        snapshot_role="asim_input_skims_zarr",
        snapshot_reason="exact_rewind",
        storage_event="snapshot_copy",
        year=2030,
        iteration=0,
    )
    assert (
        ch.archive_copy_now(
            key="asim_input_skims_zarr_archived",
            path=str(zarr_dir),
        )
        is True
    )

    h5_path = local_root / "urbansim" / "data" / "model_data_2030.h5"
    _write_file(h5_path, "h5")
    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="usim_datastore_h5",
        path=str(h5_path),
        artifact_family="usim_datastore_h5",
        source_role="usim_input_merged_2030",
        snapshot_role="usim_input_merged",
        snapshot_reason="post_merge_handoff",
        storage_event="merged_h5_output",
        year=2030,
        h5_container=True,
    )
    assert ch.archive_copy_now(key="usim_datastore_h5", path=str(h5_path)) is True

    summary = _lifecycle_summary(local_root)
    assert summary["copied_artifacts_blocked_artifact_logging_after_copying"] == 1
    assert summary["directory_artifacts_blocked_shallow_directory_signatures"] == 1
    assert summary["h5_parent_child_artifacts_requiring_policy"] == 1
    assert summary["phase2_recommendation"] == "defer"
    assert summary["blocker_counts_by_reason"]["artifact_logging_after_copying"] == 1
    assert summary["blocker_counts_by_reason"]["shallow_directory_signature"] == 1
    assert summary["blocker_counts_by_reason"]["h5_parent_child_policy"] == 1
    assert "h5_parent_child_policy" not in summary["phase2_blocker_counts_by_reason"]
    assert summary["unknown_event_keys"] == []


def test_artifact_lifecycle_summary_treats_zarr_skims_as_zarr_artifact(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    zarr_dir = local_root / "activitysim" / "cache" / "skims.zarr"
    _write_file(zarr_dir / "0" / "values", "zarr")
    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="zarr_skims",
        path=str(zarr_dir),
        artifact_family="zarr_skims",
        artifact_driver="zarr",
        source_role="zarr_skims",
        snapshot_role="zarr_skims",
        snapshot_reason="exact_rewind",
        storage_event="snapshot_copy",
        year=2030,
        iteration=0,
    )
    assert ch.archive_copy_now(key="zarr_skims", path=str(zarr_dir)) is True

    summary = _lifecycle_summary(local_root)
    assert summary["contract_status_by_family"]["zarr_skims"] == "stable"
    assert "zarr_skims" not in summary["phase2_candidate_families"]
    assert "zarr_skims" not in summary["safe_families_for_phase2"]
    assert "zarr_skims" not in summary["blocked_families_for_phase2"]


def test_artifact_lifecycle_summary_preserves_attempts_on_run_context(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))

    workspace = DummyWorkspace(local_root)
    consist_audit.emit_artifact_lifecycle_audit_event(
        workspace=workspace,
        event_type="run_context",
        run_name="first",
    )
    consist_audit.emit_artifact_lifecycle_audit_event(
        workspace=workspace,
        event_type="artifact_logged",
        key="beam_input_plans_archived",
        path=str(local_root / "old.csv"),
        artifact_family="beam_input_archived",
    )
    consist_audit.emit_artifact_lifecycle_audit_event(
        workspace=workspace,
        event_type="run_context",
        run_name="second",
    )

    events_path = (
        local_root / ".workflow" / "diagnostics" / "artifact_lifecycle_audit.jsonl"
    )
    events = events_path.read_text(encoding="utf-8").splitlines()
    assert len(events) == 3
    assert "first" in events[0]
    assert "beam_input_plans_archived" in events[1]
    assert "second" in events[2]

    attempt_dirs = sorted(
        (local_root / ".workflow" / "diagnostics" / "attempts").glob("attempt_*")
    )
    assert len(attempt_dirs) == 2
    assert (
        len(
            (attempt_dirs[0] / "artifact_lifecycle_audit.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 2
    )
    assert (
        len(
            (attempt_dirs[1] / "artifact_lifecycle_audit.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 1
    )

    summary = _lifecycle_summary(local_root)
    assert summary["phase2_recommendation_basis"] == "aggregate_attempts"
    assert len(summary["attempt_summaries"]) == 2
    assert summary["latest_attempt_summary"]["attempt_number"] == 2
    assert "beam_input_archived" in summary["families_seen"]


def test_artifact_lifecycle_summary_loads_existing_aggregate_on_new_process(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))

    workspace = DummyWorkspace(local_root)
    source = local_root / "beam" / "input" / "plans.csv.gz"
    _write_file(source, "plans")
    consist_audit.emit_artifact_lifecycle_audit_event(
        workspace=workspace,
        event_type="artifact_logged",
        key="beam_input_plans_archived",
        path=str(source),
        artifact_family="beam_input_archived",
        source_role="plans_beam_in",
        snapshot_role="beam_input_plans",
        snapshot_reason="exact_rewind",
        storage_event="snapshot_copy",
        year=2030,
        iteration=0,
    )
    consist_audit.reset_consist_audit_state()
    consist_audit.emit_artifact_lifecycle_audit_event(
        workspace=workspace,
        event_type="run_context",
        run_name="restart",
    )

    summary = _lifecycle_summary(local_root)
    assert "beam_input_archived" in summary["families_seen"]
    assert len(summary["attempt_summaries"]) == 2
    assert summary["attempt_summaries"][0]["event_counts"]["artifact_logged"] == 1
    assert summary["attempt_summaries"][1]["event_counts"]["run_context"] == 1


def test_artifact_lifecycle_summary_classifies_unknown_keys(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))

    source = local_root / "mystery" / "artifact.txt"
    _write_file(source, "mystery")
    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="beam_input_mystery",
        path=str(source),
    )

    summary = _lifecycle_summary(local_root)
    assert "unknown" not in summary["blocked_families_for_phase2"]
    assert summary["unknown_event_keys"] == ["beam_input_mystery"]
    assert summary["blocker_counts_by_reason"]["unclassified_family"] == 1
    assert summary["blocking_reasons_by_family"]["unknown"] == ["unclassified_family"]


def test_artifact_lifecycle_summary_keeps_atlas_observe_only_diagnostic(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "atlas" / "atlas_output" / "vehicles2_2030.csv"
    _write_file(source, "vehicles")
    assert ch.archive_copy_now(key="atlas_vehicles2_output", path=str(source)) is True

    summary = _lifecycle_summary(local_root)
    assert summary["contract_status_by_family"]["atlas_observe_only"] == "deferred"
    assert "atlas_observe_only" not in summary["phase2_candidate_families"]
    assert "atlas_observe_only" not in summary["blocked_families_for_phase2"]
    assert "artifact_not_logged" not in summary["phase2_blocker_counts_by_reason"]
    assert summary["diagnostic_blocking_reasons_by_family"]["atlas_observe_only"] == [
        "artifact_not_logged"
    ]


def test_artifact_lifecycle_summary_defers_usim_h5_snapshots_explicitly(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "urbansim" / "data" / "usim_input_merged_2030.h5"
    _write_file(source, "h5")
    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="usim_input_merged_2030",
        path=str(source),
        artifact_family="usim_input_merged",
        source_role="usim_input_archive",
        snapshot_role="usim_input_merged",
        snapshot_reason="post_merge_handoff",
        storage_event="merged_h5_output",
        year=2030,
        h5_container=True,
    )
    assert ch.archive_copy_now(key="usim_input_merged_2030", path=str(source)) is True

    summary = _lifecycle_summary(local_root)
    assert summary["blocking_reasons_by_family"]["usim_input_merged"] == [
        "h5_parent_child_policy"
    ]
    assert (
        "artifact_not_logged"
        not in summary["blocking_reasons_by_family"]["usim_input_merged"]
    )


def test_artifact_lifecycle_summary_allows_policy_eligible_usim_h5_parent(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "urbansim" / "data" / "model_data_2030.h5"
    _write_file(source, "h5")
    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="usim_population_source_h5",
        path=str(source),
        artifact_family="usim_population_source_h5",
        source_role="usim_population_source_h5",
        snapshot_role="usim_population_source_h5",
        snapshot_reason="post_merge_handoff",
        storage_event="merged_h5_output",
        year=2030,
        h5_container=True,
        container_recovery_unit="parent_file",
        child_recovery_policy="descriptive_only",
    )
    assert (
        ch.archive_copy_now(key="usim_population_source_h5", path=str(source)) is True
    )

    summary = _lifecycle_summary(local_root)
    assert summary["copied_artifacts_joined_to_logged_artifacts"] == 1
    assert summary["copied_artifacts_eligible_for_recovery_root_registration"] == 1
    assert summary["contract_status_by_family"]["usim_population_source_h5"] == (
        "stable"
    )
    assert summary["phase2_candidate_families"] == [
        "usim_input_archive",
        "usim_population_source_h5",
    ]
    assert "usim_population_source_h5" in summary["safe_families_for_phase2"]
    assert "usim_population_source_h5" not in summary["blocked_families_for_phase2"]
    assert "h5_parent_child_policy" not in summary["blocker_counts_by_reason"]


@pytest.mark.parametrize(
    "key, family",
    [
        ("usim_input_archive_2030", "usim_input_archive"),
        ("usim_population_source_h5", "usim_population_source_h5"),
    ],
)
def test_artifact_lifecycle_summary_starts_h5_phase2_eligibility(
    monkeypatch, tmp_path, key, family
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "urbansim" / "data" / f"{key}.h5"
    _write_file(source, "h5")
    fields = {
        "event_type": "artifact_logged",
        "key": key,
        "path": str(source),
        "artifact_family": family,
        "year": 2030,
        "h5_container": True,
        "container_recovery_unit": "parent_file",
        "child_recovery_policy": "descriptive_only",
    }
    if family in {"usim_input_archive", "usim_input_merged"}:
        fields.update(
            {
                "source_role": "usim_datastore_h5",
                "snapshot_role": family,
                "snapshot_reason": "exact_rewind",
                "storage_event": "snapshot_copy",
            }
        )
    consist_audit.emit_artifact_lifecycle_audit_event(**fields)
    assert ch.archive_copy_now(key=key, path=str(source)) is True

    summary = _lifecycle_summary(local_root)
    assert summary["contract_status_by_family"][family] == "stable"
    assert family in summary["phase2_candidate_families"]
    assert family in summary["safe_families_for_phase2"]
    assert family not in summary["blocked_families_for_phase2"]


@pytest.mark.parametrize(
    "key, family, artifact_meta, expected_safe_families",
    [
        (
            "usim_input_archive_2030",
            "usim_input_archive",
            {
                "source_role": "usim_datastore_h5",
                "snapshot_role": "usim_input_archive",
                "snapshot_reason": "pre_merge_input",
                "storage_event": "snapshot_move",
            },
            ["usim_input_archive"],
        ),
        (
            "usim_population_source_h5",
            "usim_population_source_h5",
            {
                "source_role": "usim_population_source_h5",
                "snapshot_role": "usim_population_source_h5",
                "snapshot_reason": "post_merge_handoff",
                "storage_event": "merged_h5_output",
            },
            ["usim_population_source_h5"],
        ),
    ],
)
def test_phase2_recovery_root_registration_adopts_only_narrow_h5_families(
    monkeypatch, tmp_path, key, family, artifact_meta, expected_safe_families
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    registration_calls = []

    tracker = SimpleNamespace(
        register_run_output_recovery_copies=lambda run_id, recovery_root, **kwargs: (
            registration_calls.append((run_id, recovery_root, kwargs))
            or SimpleNamespace(
                registered={"usim_population_source_h5": object()},
                blocked={},
                summary="registered=1",
            )
        ),
    )
    monkeypatch.setattr(ch.cr, "current_tracker", lambda: tracker)
    monkeypatch.setattr(
        ch.cr, "current_run", lambda: SimpleNamespace(id="run-123")
    )
    monkeypatch.setattr(ch.cr, "current_run_id", lambda: "run-123")
    monkeypatch.setattr(
        ch,
        "_find_current_run_output_artifact",
        lambda *, key, path: SimpleNamespace(key=key, container_uri=str(path)),
    )

    source = local_root / "urbansim" / "data" / "model_data_2030.h5"
    _write_file(source, "h5")
    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key=key,
        path=str(source),
        artifact_family=family,
        **artifact_meta,
        year=2030,
        h5_container=True,
        container_recovery_unit="parent_file",
        child_recovery_policy="descriptive_only",
    )
    assert ch.archive_copy_now(key=key, path=str(source))

    expected_hash = hashlib.sha256(b"h5").hexdigest()
    assert registration_calls == [
        (
            "run-123",
            str(archive_root),
            {
                "append": True,
                "content_hashes": {key: expected_hash},
                "verify": True,
            },
        )
    ]

    summary = _lifecycle_summary(local_root)
    assert summary["local_to_scratch_recovery_roots_written"] == 1
    assert summary["phase2_candidate_families"] == [
        "usim_input_archive",
        "usim_population_source_h5",
    ]
    assert summary["safe_families_for_phase2"] == expected_safe_families


def test_phase2_recovery_root_registration_skips_blocked_h5_family(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    registration_calls = []
    tracker = SimpleNamespace(
        register_run_output_recovery_copies=lambda *args, **kwargs: registration_calls.append(
            (args, kwargs)
        )
    )
    monkeypatch.setattr(ch.cr, "current_tracker", lambda: tracker)
    monkeypatch.setattr(
        ch.cr, "current_run", lambda: SimpleNamespace(id="run-123")
    )
    monkeypatch.setattr(ch.cr, "current_run_id", lambda: "run-123")
    monkeypatch.setattr(ch, "_find_current_run_output_artifact", lambda **_kwargs: None)

    source = local_root / "urbansim" / "data" / "model_data_2030.h5"
    _write_file(source, "h5")
    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="usim_datastore_h5",
        path=str(source),
        artifact_family="usim_datastore_h5",
        source_role="usim_datastore_h5",
        snapshot_role="usim_datastore_h5",
        snapshot_reason="post_merge_handoff",
        storage_event="merged_h5_output",
        year=2030,
        h5_container=True,
        container_recovery_unit="parent_file",
        child_recovery_policy="descriptive_only",
    )
    assert ch.archive_copy_now(key="usim_datastore_h5", path=str(source))

    assert registration_calls == []
    summary = _lifecycle_summary(local_root)
    assert summary["local_to_scratch_recovery_roots_written"] == 0
    assert "usim_datastore_h5" not in summary["safe_families_for_phase2"]


def test_phase2_recovery_root_registration_prefers_artifact_hash_when_full_hashing(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    registration_calls = []
    tracker = SimpleNamespace(
        identity=SimpleNamespace(hashing_strategy="full"),
        register_run_output_recovery_copies=lambda run_id, recovery_root, **kwargs: (
            registration_calls.append((run_id, recovery_root, kwargs))
            or SimpleNamespace(
                registered={"usim_population_source_h5": object()},
                blocked={},
                summary="registered=1",
            )
        ),
    )
    monkeypatch.setattr(ch.cr, "current_tracker", lambda: tracker)
    monkeypatch.setattr(
        ch.cr, "current_run", lambda: SimpleNamespace(id="run-123")
    )
    monkeypatch.setattr(ch.cr, "current_run_id", lambda: "run-123")

    source = local_root / "urbansim" / "data" / "model_data_2030.h5"
    _write_file(source, "h5")
    artifact = SimpleNamespace(
        key="usim_population_source_h5",
        hash="artifact-hash-123",
        container_uri=str(source),
    )
    monkeypatch.setattr(
        ch,
        "_find_current_run_output_artifact",
        lambda *, key, path: artifact,
    )
    monkeypatch.setattr(
        ch,
        "_sha256_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sha called")),
    )
    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="usim_population_source_h5",
        path=str(source),
        artifact_family="usim_population_source_h5",
        source_role="usim_population_source_h5",
        snapshot_role="usim_population_source_h5",
        snapshot_reason="post_merge_handoff",
        storage_event="merged_h5_output",
        year=2030,
        h5_container=True,
        container_recovery_unit="parent_file",
        child_recovery_policy="descriptive_only",
    )
    assert ch.archive_copy_now(key="usim_population_source_h5", path=str(source))

    assert registration_calls == [
        (
            "run-123",
            str(archive_root),
            {
                "append": True,
                "content_hashes": {"usim_population_source_h5": "artifact-hash-123"},
                "verify": True,
            },
        )
    ]


def test_artifact_lifecycle_summary_keeps_h5_child_tables_ineligible(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))

    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="urbansim_postprocess_usim_households_table_updated",
        path=str(local_root / "urbansim" / "data" / "model_data.h5"),
        artifact_family="usim_datastore_h5",
        artifact_driver="h5_table",
        h5_parent_key="usim_datastore_h5",
        h5_table_name="households",
        child_recovery_policy="descriptive_only",
    )

    summary = _lifecycle_summary(local_root)
    assert summary["h5_child_table_artifacts_ineligible"] == 1
    assert summary["contract_status_by_family"]["usim_datastore_h5"] == "stable"
    assert summary["blocking_reasons_by_family"]["usim_datastore_h5"] == [
        "h5_child_table_ineligible"
    ]
    assert "usim_datastore_h5" not in summary["phase2_candidate_families"]


def test_artifact_lifecycle_summary_accepts_sanitized_artifact_year(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "urbansim" / "data" / "input_data_for_2030_outputs.h5"
    _write_file(source, "h5")
    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="usim_input_archive_2030",
        path=str(source),
        artifact_family="usim_input_archive",
        artifact_year=2030,
        source_role="usim_datastore_h5",
        snapshot_role="usim_input_archive",
        snapshot_reason="pre_merge_input",
        storage_event="snapshot_move",
        sanitized_lifecycle_fields={"year": "artifact_year"},
        h5_container=True,
    )
    assert ch.archive_copy_now(key="usim_input_archive_2030", path=str(source)) is True

    summary = _lifecycle_summary(local_root)
    assert summary["snapshot_artifacts_logged"] == 1
    assert summary["snapshot_artifacts_missing_required_facets"] == 0
    assert summary["contract_status_by_family"]["usim_input_archive"] == "stable"
    assert (
        "missing_required_snapshot_facets"
        not in summary["blocking_reasons_by_family"]["usim_input_archive"]
    )


def test_artifact_lifecycle_summary_accepts_sanitized_artifact_iteration(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = (
        local_root
        / "activitysim"
        / "output"
        / "inputs-year-2030-iteration-0"
        / "households.csv"
    )
    _write_file(source, "household_id\n1\n")
    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="asim_input_households_csv_archived",
        path=str(source),
        artifact_family="asim_input_archived",
        artifact_year=2030,
        artifact_iteration=0,
        source_role="households_asim_in",
        snapshot_role="asim_input_households_csv",
        snapshot_reason="exact_rewind",
        storage_event="snapshot_copy",
        sanitized_lifecycle_fields={
            "iteration": "artifact_iteration",
            "year": "artifact_year",
        },
    )
    assert (
        ch.archive_copy_now(key="asim_input_households_csv_archived", path=str(source))
        is True
    )

    summary = _lifecycle_summary(local_root)
    assert summary["snapshot_artifacts_logged"] == 1
    assert summary["snapshot_artifacts_missing_required_facets"] == 0
    assert summary["contract_status_by_family"]["asim_input_archived"] == (
        "transitional"
    )
    assert "missing_required_snapshot_facets" not in summary[
        "blocking_reasons_by_family"
    ].get("asim_input_archived", [])


def test_artifact_lifecycle_summary_uses_first_log_for_copy_order(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "urbansim" / "data" / "custom_mpo_model_data.h5"
    _write_file(source, "h5")
    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="usim_population_source_h5",
        path=str(source),
        artifact_family="usim_population_source_h5",
        artifact_year=2030,
        h5_container=True,
        container_recovery_unit="parent_file",
        child_recovery_policy="descriptive_only",
    )
    assert (
        ch.archive_copy_now(key="usim_population_source_h5", path=str(source)) is True
    )
    consist_audit.emit_artifact_lifecycle_audit_event(
        event_type="artifact_logged",
        key="usim_population_source_h5",
        path=str(source),
        artifact_family="usim_population_source_h5",
        artifact_year=2030,
        h5_container=True,
        snapshot_role="usim_population_source_h5",
        snapshot_reason="post_merge_handoff",
        storage_event="merged_h5_output",
        container_recovery_unit="parent_file",
        child_recovery_policy="descriptive_only",
    )

    summary = _lifecycle_summary(local_root)
    assert summary["copied_artifacts_blocked_artifact_logging_after_copying"] == 0
    assert summary["contract_status_by_family"]["usim_population_source_h5"] == (
        "stable"
    )
    assert "usim_population_source_h5" in summary["safe_families_for_phase2"]


def test_artifact_lifecycle_summary_blocks_copy_only_promotions(monkeypatch, tmp_path):
    run_dir = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(run_dir))

    consist_audit.emit_artifact_lifecycle_audit_event(
        run_dir=run_dir,
        event_type="promotion_status",
        recovery_root=str(tmp_path / "nfs"),
        destination_run_dir=str(tmp_path / "nfs" / "run"),
        status="promoted",
        copy_performed=True,
        verified=True,
        artifact_metadata_updated=False,
    )

    summary = _lifecycle_summary(run_dir)
    assert summary["copy_only_promotions_db_tracker_metadata_unavailable"] == 1
    assert (
        summary["blocker_counts_by_reason"]["copy_only_promotion_metadata_unavailable"]
        == 1
    )
    assert summary["phase2_recommendation"] == "defer"
    assert "post_run_promotion" not in summary["phase2_candidate_families"]


def test_resolve_existing_path_materializes_local_from_archive(
    monkeypatch, tmp_path, caplog
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    local_path = local_root / "activitysim" / "data" / "households.csv"
    archive_path = archive_root / "activitysim" / "data" / "households.csv"
    _write_file(archive_path, "archived-households")

    with caplog.at_level(logging.INFO):
        resolved = ch.resolve_existing_path(
            str(local_path), materialize_from_archive=True
        )
    assert resolved == str(local_path)
    assert local_path.exists()
    assert local_path.read_text() == "archived-households"
    assert "materializing from archive" in caplog.text
    assert "Materialized local path from archive" in caplog.text


def test_archive_copy_rejects_non_allowlisted_directory(monkeypatch, tmp_path, caplog):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    directory = local_root / "beam" / "output" / "raw_dir"
    _write_file(directory / "file.txt", "data")

    ch._enqueue_archive_copy("beam_output_dir", str(directory))

    assert "not allowlisted" in caplog.text
    assert not (archive_root / "beam" / "output" / "raw_dir").exists()


def test_archive_copy_allows_zarr_directories(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    directory = local_root / "activitysim" / "cache" / "skims.zarr"
    _write_file(directory / "0" / "values", "zarr")

    ch._enqueue_archive_copy("asim_input_skims_zarr_archived", str(directory))
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = archive_root / "activitysim" / "cache" / "skims.zarr" / "0" / "values"
    assert archived.exists()
    assert archived.read_text() == "zarr"


def test_archive_copy_allows_beam_raw_od_zarr_directories(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    directory = (
        local_root
        / "beam"
        / "beam_output"
        / "sfbay"
        / "year-2017-iteration-0"
        / "ITERS"
        / "it.1"
        / "1.activitySimODSkims_current.zarr"
    )
    _write_file(directory / "0" / "values", "zarr")

    ch._enqueue_archive_copy("raw_od_skims_zarr_2019_0", str(directory))
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = (
        archive_root
        / "beam"
        / "beam_output"
        / "sfbay"
        / "year-2017-iteration-0"
        / "ITERS"
        / "it.1"
        / "1.activitySimODSkims_current.zarr"
        / "0"
        / "values"
    )
    assert archived.exists()
    assert archived.read_text() == "zarr"


def test_archive_copy_allows_beam_config_reference_snapshot_directory(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    directory = (
        local_root
        / "beam"
        / "beam_output"
        / "inputs-year-2019-iteration-0"
        / "beam_input_config_references_archived"
    )
    _write_file(directory / "scenario" / "network.csv", "network")

    ch._enqueue_archive_copy("beam_input_config_references_archived", str(directory))
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = (
        archive_root
        / "beam"
        / "beam_output"
        / "inputs-year-2019-iteration-0"
        / "beam_input_config_references_archived"
        / "scenario"
        / "network.csv"
    )
    assert archived.exists()
    assert archived.read_text() == "network"


def test_archive_copy_allows_activitysim_sharrow_cache_directory(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    directory = local_root / "shared_cache" / "numba"
    _write_file(directory / "nested" / "entry.bin", "cache")

    ch._enqueue_archive_copy(ASIM_SHARROW_CACHE_DIR, str(directory))
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = archive_root / "shared_cache" / "numba" / "nested" / "entry.bin"
    assert archived.exists()
    assert archived.read_text() == "cache"


@pytest.mark.parametrize(
    "key, relpath",
    [
        ("urbansim_bootstrap_data_root", "urbansim/data/hsize_ct_000.csv"),
        ("beam_mutable_data_dir", "beam/input/test/beam.conf"),
        ("activitysim_bootstrap_data_root", "activitysim/data/households.csv"),
        (
            "activitysim_bootstrap_configs_root",
            "activitysim/configs/configs/settings.yaml",
        ),
    ],
)
def test_archive_copy_allows_bootstrap_runtime_directories(
    monkeypatch, tmp_path, key, relpath
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    path = local_root / relpath
    _write_file(path, "bootstrap")

    directory = path.parent
    if key == "urbansim_bootstrap_data_root":
        directory = local_root / "urbansim" / "data"
    if path.name == "households.csv":
        directory = local_root / "activitysim" / "data"
    if key == "activitysim_bootstrap_configs_root":
        directory = local_root / "activitysim" / "configs"

    ch._enqueue_archive_copy(key, str(directory))
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = archive_root / relpath
    assert archived.exists()
    assert archived.read_text() == "bootstrap"


def test_archive_copy_allows_atlas_year_input_directory(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    directory = local_root / "atlas" / "atlas_input" / "year2030"
    _write_file(directory / "vehicles_output.RData", "atlas-rdata")

    ch.enqueue_archive_copy(
        key="atlas_input_year_dir_2030",
        path=str(directory),
    )
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = (
        archive_root / "atlas" / "atlas_input" / "year2030" / "vehicles_output.RData"
    )
    assert archived.exists()
    assert archived.read_text() == "atlas-rdata"


def test_archive_copy_dedupes_same_signature(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "beam" / "output" / "linkstats.csv.gz"
    _write_file(source, "linkstats")

    copy_calls = []
    original_copy2 = ch.shutil.copy2

    def _counting_copy2(src, dst, *args, **kwargs):
        copy_calls.append((src, dst))
        return original_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(ch.shutil, "copy2", _counting_copy2)

    ch._enqueue_archive_copy("linkstats", str(source))
    ch._enqueue_archive_copy("linkstats", str(source))
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    archived = archive_root / "beam" / "output" / "linkstats.csv.gz"
    assert archived.exists()
    assert archived.read_text() == "linkstats"
    assert len(copy_calls) == 1


def test_archive_copy_coalesces_pending_updates_for_same_destination(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    def _ensure_queue_only():
        if ch._archive_queue is None:
            ch._archive_queue = queue.Queue()

    monkeypatch.setattr(ch, "_ensure_archive_worker", _ensure_queue_only)

    source = (
        local_root / ".workflow" / "diagnostics" / "consist_restart_audit_summary.json"
    )
    _write_file(source, '{"event_count": 1}')
    ch._enqueue_archive_copy(
        "workflow_diagnostics_consist_restart_audit_summary",
        str(source),
    )

    _write_file(source, '{"event_count": 2, "latest": true}')
    ch._enqueue_archive_copy(
        "workflow_diagnostics_consist_restart_audit_summary",
        str(source),
    )

    dest = str(
        archive_root
        / ".workflow"
        / "diagnostics"
        / "consist_restart_audit_summary.json"
    )
    assert ch._archive_queue is not None
    assert ch._archive_queue.qsize() == 1
    assert dest in ch._archive_pending_tasks

    key, pending_src, pending_dest, _is_dir, _signature = ch._archive_pending_tasks[
        dest
    ]
    assert key == "workflow_diagnostics_consist_restart_audit_summary"
    assert pending_src == str(source)
    assert pending_dest == dest


def test_archive_copy_now_copies_file_and_preserves_relative_path(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / ".workflow" / "year_2018_iteration_0.yaml"
    _write_file(source, "manifest")

    assert ch.archive_copy_now(key="workflow_manifest", path=str(source)) is True

    archived = archive_root / ".workflow" / "year_2018_iteration_0.yaml"
    assert archived.exists()
    assert archived.read_text() == "manifest"


def test_workflow_manifest_is_tracked_as_restart_support_only(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    workspace = DummyWorkspace(local_root)
    consist_audit.emit_artifact_lifecycle_audit_event(
        workspace=workspace,
        event_type="run_context",
        run_name="restart",
    )

    source = local_root / ".workflow" / "year_2018_iteration_0.yaml"
    _write_file(source, "manifest")
    assert ch.archive_copy_now(key="workflow_manifest", path=str(source)) is True
    assert ch.archive_copy_now(key="workflow_manifest", path=str(source)) is True

    summary = _lifecycle_summary(local_root)
    assert summary["restart_support_keys"] == ["workflow_manifest"]
    assert summary["unknown_event_keys"] == []
    assert "workflow_manifest" not in summary["blocked_families_for_phase2"]
    assert "workflow_manifest" not in summary["safe_families_for_phase2"]


def test_archive_copy_destination_returns_preserved_relative_path(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "urbansim" / "data" / "model_data_2021.h5"
    _write_file(source, "h5")

    assert ch.archive_copy_destination(
        key="usim_population_source_h5",
        path=str(source),
    ) == str(archive_root / "urbansim" / "data" / "model_data_2021.h5")


def test_archive_copy_now_force_recopies_matching_signature(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "urbansim" / "data" / "model_data_2021.h5"
    archived = archive_root / "urbansim" / "data" / "model_data_2021.h5"
    _write_file(source, "fresh")
    _write_file(archived, "stale")

    signature = ch._archive_path_signature(str(source), is_dir=False)
    ch._archive_last_copied_signature[str(archived)] = signature

    assert (
        ch.archive_copy_now(
            key="usim_population_source_h5",
            path=str(source),
            force=True,
        )
        is True
    )
    assert archived.read_text() == "fresh"


def test_archive_copy_now_emits_already_copied_checkpoint_outside_archive_lock(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    source = local_root / "urbansim" / "data" / "custom_mpo_model_data.h5"
    _write_file(source, "h5")
    assert ch.archive_copy_now(key="usim_datastore_h5", path=str(source)) is True

    events = []

    def _record_checkpoint(event_type, **fields):
        assert not ch._archive_lock.locked()
        events.append((event_type, fields))

    monkeypatch.setattr(ch, "_emit_artifact_lifecycle_event", _record_checkpoint)

    assert ch.archive_copy_now(key="usim_datastore_h5", path=str(source)) is True
    assert events
    event_type, fields = events[-1]
    assert event_type == "archive_copy_checkpoint"
    assert fields["storage_event"] == "local_to_scratch_copy_already_present"


def test_flush_archive_queue_can_fail_on_timeout():
    ch._archive_queue = queue.Queue()
    ch._archive_queue.put(("pending",))
    with pytest.raises(TimeoutError, match="Flush timed out"):
        ch.flush_archive_queue(timeout=0.01, fail_on_timeout=True)


def test_log_output_only_enqueues_archive_copy(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ch.cr, "log_output", lambda *args, **kwargs: "artifact")
    monkeypatch.setattr(
        ch, "_enqueue_archive_copy", lambda key, path: calls.append((key, path))
    )

    out_path = tmp_path / "out.txt"
    _write_file(out_path, "output")

    ch.log_output_only(
        key="beam_output_plans_xml",
        path=str(out_path),
        description="mock output",
    )

    assert calls == [("beam_output_plans_xml", str(out_path))]


def test_log_and_set_output_enqueues_archive_copy_and_sets_coupler(
    monkeypatch, tmp_path
):
    calls = []
    coupler = DummyCoupler()
    monkeypatch.setattr(ch.cr, "log_output", lambda *args, **kwargs: "artifact")
    monkeypatch.setattr(ch.cr, "current_run", lambda: object())
    monkeypatch.setattr(
        ch, "_enqueue_archive_copy", lambda key, path: calls.append((key, path))
    )

    out_path = tmp_path / "out.txt"
    _write_file(out_path, "output")

    ch.log_and_set_output(
        key="linkstats",
        path=str(out_path),
        description="mock output",
        coupler=coupler,
    )

    assert calls == [("linkstats", str(out_path))]
    assert coupler.get("linkstats") is not None


def test_mocked_workflow_archives_logged_outputs(monkeypatch, tmp_path):
    local_root = tmp_path / "local" / "run"
    archive_root = tmp_path / "archive" / "run"
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    workspace = DummyWorkspace(local_root)
    coupler = DummyCoupler()
    outputs_holder = StepOutputsHolder()
    scenario = ExecutingScenario()
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2030, iteration=0)

    def _mock_step(*, workspace, coupler, **_kwargs):
        file_out = Path(workspace.full_path) / "mock" / "linkstats.csv.gz"
        dir_out = Path(workspace.full_path) / "mock" / "skims.zarr"
        _write_file(file_out, "stats")
        _write_file(dir_out / "0" / "values", "zarr")
        ch.log_and_set_output(
            key="linkstats",
            path=str(file_out),
            description="mock linkstats",
            coupler=coupler,
        )
        ch.log_output_only(
            key="asim_input_skims_zarr_archived",
            path=str(dir_out),
            description="mock zarr archive",
        )

    _mock_step.__consist_step__ = SimpleNamespace(
        model="mock_archive_step", outputs=["linkstats"]
    )

    run_workflow(
        stage_name="mock_archive_stage",
        steps=[StepRef(name="mock_archive_step", step_func=_mock_step)],
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        name_suffix="2030_iter0",
        runtime_kwargs_extra={"coupler": coupler, "outputs_holder": outputs_holder},
    )
    ch.flush_archive_queue(timeout=5)
    ch.stop_archive_worker(timeout=5)

    assert (archive_root / "mock" / "linkstats.csv.gz").exists()
    assert (archive_root / "mock" / "skims.zarr" / "0" / "values").exists()
    assert coupler.get("linkstats") is not None
