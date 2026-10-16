from __future__ import annotations

from types import SimpleNamespace

import consist
import pytest

from pilates.runtime import restart as restart_runtime


def test_prebootstrap_missing_artifacts_are_split_by_surface():
    surface = SimpleNamespace(
        is_restart_prebootstrap_deferred_artifact_key=lambda key: (
            key == "bootstrap_owned"
        )
    )
    artifacts = [
        {"key": "runtime_owned", "path": "/tmp/runtime", "reason": "runtime"},
        {"key": "bootstrap_owned", "path": "/tmp/bootstrap", "reason": "bootstrap"},
    ]

    blocking, deferred = restart_runtime.split_prebootstrap_missing_artifacts(
        artifacts,
        surface=surface,
    )

    assert blocking == [artifacts[0]]
    assert deferred == [artifacts[1]]


def test_enforce_postbootstrap_missing_artifacts_raises_when_strict():
    artifacts = [
        {"key": "runtime_owned", "path": "/tmp/runtime", "reason": "runtime"},
    ]
    settings = SimpleNamespace(run=SimpleNamespace(restart_strict=True))

    with pytest.raises(RuntimeError, match="Strict restart preflight failed"):
        restart_runtime.enforce_postbootstrap_missing_artifacts(
            artifacts,
            settings=settings,
        )


def test_restore_restart_coupler_snapshot_from_captured_snapshot(tmp_path):
    db_path = str(tmp_path / "provenance.duckdb")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tracker = consist.Tracker(run_dir=run_dir, db_path=db_path)

    sample_file = run_dir / "skims.omx"
    sample_file.write_text("skim-content\n")

    with tracker.scenario("fresh_scenario") as sc_fresh:

        def step_fn():
            art = tracker.log_artifact(sample_file, key="zarr_skims")
            sc_fresh.coupler.set("zarr_skims", art)

        sc_fresh.run(step_fn, name="produce_skims")
        sc_fresh.capture_coupler_snapshot(phase="post")

    settings = SimpleNamespace(run=SimpleNamespace(archive_run_dir=None))
    state = SimpleNamespace(scenario_id="fresh_scenario", run_info_path=None)
    workspace = SimpleNamespace(full_path=run_dir, archive_run_dir=None)

    with tracker.scenario("restart_scenario") as sc_restart:
        restored = restart_runtime.restore_restart_coupler_snapshot(
            scenario=sc_restart,
            tracker=tracker,
            settings=settings,
            state=state,
            workspace=workspace,
        )

        assert "zarr_skims" in restored
        assert "zarr_skims" in sc_restart.coupler
        assert sc_restart.coupler["zarr_skims"].key == "zarr_skims"


def test_restore_restart_coupler_snapshot_fallback_to_history(tmp_path):
    db_path = str(tmp_path / "provenance.duckdb")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tracker = consist.Tracker(run_dir=run_dir, db_path=db_path)

    legacy_file = run_dir / "legacy_output.csv"
    legacy_file.write_text("legacy-data\n")

    with tracker.scenario("legacy_scenario") as sc_legacy:

        def step_fn():
            tracker.log_artifact(legacy_file, key="legacy_plans", direction="output")

        sc_legacy.run(step_fn, name="produce_plans")
        # No sc_legacy.capture_coupler_snapshot(...) called to simulate legacy runs

    settings = SimpleNamespace(run=SimpleNamespace(archive_run_dir=None))
    state = SimpleNamespace(scenario_id="legacy_scenario", run_info_path=None)
    workspace = SimpleNamespace(full_path=run_dir, archive_run_dir=None)

    with tracker.scenario("restart_scenario") as sc_restart:
        restored = restart_runtime.restore_restart_coupler_snapshot(
            scenario=sc_restart,
            tracker=tracker,
            settings=settings,
            state=state,
            workspace=workspace,
        )

        assert "legacy_plans" in restored
        assert "legacy_plans" in sc_restart.coupler
        assert sc_restart.coupler["legacy_plans"].key == "legacy_plans"


def test_restore_restart_coupler_snapshot_stages_missing_files_from_archive(tmp_path):
    db_path = str(tmp_path / "provenance.duckdb")
    archive_dir = tmp_path / "archive_run"
    archive_dir.mkdir()

    tracker_arch = consist.Tracker(run_dir=archive_dir, db_path=db_path)
    arch_file = archive_dir / "outputs" / "plans.csv"
    arch_file.parent.mkdir(parents=True, exist_ok=True)
    arch_file.write_text("archived-plan-data\n")

    with tracker_arch.scenario("archive_scenario") as sc_arch:

        def step_fn():
            art = tracker_arch.log_artifact(arch_file, key="beam_plans")
            sc_arch.coupler.set("beam_plans", art)

        sc_arch.run(step_fn, name="produce_archive_plans")
        sc_arch.capture_coupler_snapshot(phase="post")

    fresh_ws = tmp_path / "fresh_workspace"
    fresh_ws.mkdir()
    tracker_fresh = consist.Tracker(run_dir=fresh_ws, db_path=db_path)

    target_missing_file = fresh_ws / "outputs" / "plans.csv"
    assert not target_missing_file.exists()

    settings = SimpleNamespace(run=SimpleNamespace(archive_run_dir=archive_dir))
    state = SimpleNamespace(
        scenario_id="archive_scenario", run_info_path=str(archive_dir / "run_info.yaml")
    )
    workspace = SimpleNamespace(full_path=fresh_ws, archive_run_dir=archive_dir)

    with tracker_fresh.scenario("restart_scenario") as sc_restart:
        restored = restart_runtime.restore_restart_coupler_snapshot(
            scenario=sc_restart,
            tracker=tracker_fresh,
            settings=settings,
            state=state,
            workspace=workspace,
        )

        assert "beam_plans" in restored
        assert "beam_plans" in sc_restart.coupler
        assert target_missing_file.exists()
        assert target_missing_file.read_text() == "archived-plan-data\n"
