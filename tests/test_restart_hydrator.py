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


def test_restore_restart_coupler_snapshot_handles_beam_input_uri_and_skips_external_schemes(
    tmp_path,
):

    db_path = str(tmp_path / "provenance.duckdb")
    archive_dir = tmp_path / "archive_run"
    archive_dir.mkdir()

    external_dir = tmp_path / "external"
    external_file = external_dir / "ref.csv"
    external_file.parent.mkdir(parents=True, exist_ok=True)
    external_file.write_text("external-ref-data\n")

    tracker_arch = consist.Tracker(
        run_dir=archive_dir,
        db_path=db_path,
        mounts={
            "beam_input": str(archive_dir / "beam" / "input"),
            "inputs": str(external_dir),
        },
    )
    arch_beam_file = archive_dir / "beam" / "input" / "transit" / "schedule.csv"
    arch_beam_file.parent.mkdir(parents=True, exist_ok=True)
    arch_beam_file.write_text("beam-schedule-data\n")

    with tracker_arch.scenario("archive_scenario") as sc_arch:

        def step_fn():
            art_beam = tracker_arch.log_artifact(
                arch_beam_file,
                key="beam_schedule",
            )
            sc_arch.coupler.set("beam_schedule", art_beam)
            art_ext = tracker_arch.log_artifact(
                external_file,
                key="external_ref",
            )
            sc_arch.coupler.set("external_ref", art_ext)

        sc_arch.run(step_fn, name="produce_archive_artifacts")
        sc_arch.capture_coupler_snapshot(phase="post")

    fresh_ws = tmp_path / "fresh_workspace"
    fresh_ws.mkdir()
    tracker_fresh = consist.Tracker(
        run_dir=fresh_ws,
        db_path=db_path,
        mounts={
            "beam_input": str(fresh_ws / "beam" / "input"),
            "inputs": str(external_dir),
        },
    )

    target_beam_file = fresh_ws / "beam" / "input" / "transit" / "schedule.csv"
    assert not target_beam_file.exists()

    settings = SimpleNamespace(run=SimpleNamespace(archive_run_dir=archive_dir))
    state = SimpleNamespace(
        scenario_id="archive_scenario",
        run_info_path=str(archive_dir / "run_info.yaml"),
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

        assert "beam_schedule" in restored
        assert "external_ref" in restored
        assert target_beam_file.exists()
        assert target_beam_file.read_text() == "beam-schedule-data\n"
        assert not (fresh_ws / "inputs").exists()


def test_rewind_uncommitted_activitysim_restart_to_provider_boundary_preserves_completed_beam():
    from pilates.runtime.launcher import (
        _rewind_uncommitted_activitysim_restart_to_provider_boundary,
    )

    class _FakeStage:
        activity_demand = "activity_demand"
        supply_demand_loop = "supply_demand_loop"
        traffic_assignment = "traffic_assignment"

    state = SimpleNamespace(
        is_restart_run=True,
        Stage=_FakeStage,
        current_major_stage="supply_demand_loop",
        current_sub_stage="traffic_assignment",
        sub_stage_progress="beam_run_completed",
        is_enabled=lambda stage: True,
        major_stage_order=("supply_demand_loop",),
    )

    rewound = _rewind_uncommitted_activitysim_restart_to_provider_boundary(state=state)
    assert rewound is False
    assert state.current_major_stage == "supply_demand_loop"


def test_restore_restart_coupler_snapshot_stages_from_recovery_roots(tmp_path):
    import shutil

    db_path = str(tmp_path / "provenance.duckdb")
    archive_dir = tmp_path / "archive_run"
    archive_dir.mkdir()
    recovery_root_dir = tmp_path / "recovery_root"
    recovery_root_dir.mkdir()

    node1_ws = tmp_path / "node1_workspace"
    node1_ws.mkdir()
    node1_file = node1_ws / "outputs" / "plans.csv"
    node1_file.parent.mkdir(parents=True, exist_ok=True)
    node1_file.write_text("node1-data\n")

    recovery_file = recovery_root_dir / "outputs" / "plans.csv"
    recovery_file.parent.mkdir(parents=True, exist_ok=True)
    recovery_file.write_text("recovery-plan-data\n")

    tracker_node1 = consist.Tracker(run_dir=node1_ws, db_path=db_path)
    with tracker_node1.scenario("archive_scenario") as sc_arch:

        def step_fn():
            art = tracker_node1.log_artifact(
                node1_file,
                key="beam_plans",
            )
            tracker_node1.set_artifact_recovery_roots(art, [recovery_root_dir])
            sc_arch.coupler.set("beam_plans", art)

        sc_arch.run(step_fn, name="produce_archive_plans")
        sc_arch.capture_coupler_snapshot(phase="post")

    shutil.rmtree(node1_ws)

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
        assert target_missing_file.exists()
        assert target_missing_file.read_text() == "recovery-plan-data\n"


def test_restore_restart_coupler_snapshot_prunes_wiped_ephemeral_scratch_artifact(
    tmp_path,
):
    import shutil

    db_path = str(tmp_path / "provenance.duckdb")
    archive_dir = tmp_path / "archive_run"
    archive_dir.mkdir()

    node1_ws = tmp_path / "node1_workspace"
    node1_ws.mkdir()
    persistent_src = node1_ws / "outputs" / "linkstats.csv.gz"
    persistent_src.parent.mkdir(parents=True, exist_ok=True)
    persistent_src.write_text("linkstats-data\n")

    archived_copy = archive_dir / "outputs" / "linkstats.csv.gz"
    archived_copy.parent.mkdir(parents=True, exist_ok=True)
    archived_copy.write_text("linkstats-data\n")

    ephemeral_src = node1_ws / "outputs" / "beam_plans_out.csv.gz"
    ephemeral_src.write_text("ephemeral-plans-out\n")

    tracker_node1 = consist.Tracker(run_dir=node1_ws, db_path=db_path)
    with tracker_node1.scenario("archive_scenario") as sc_arch:

        def step_fn():
            art_linkstats = tracker_node1.log_artifact(
                persistent_src,
                key="linkstats",
            )
            art_plans_out = tracker_node1.log_artifact(
                ephemeral_src,
                key="beam_plans_out",
            )
            sc_arch.coupler.set("linkstats", art_linkstats)
            sc_arch.coupler.set("beam_plans_out", art_plans_out)

        sc_arch.run(step_fn, name="produce_beam_outputs")
        sc_arch.capture_coupler_snapshot(phase="post")

    # Simulate SLURM wiping node1 local scratch
    shutil.rmtree(node1_ws)

    fresh_ws = tmp_path / "fresh_workspace"
    fresh_ws.mkdir()
    tracker_fresh = consist.Tracker(run_dir=fresh_ws, db_path=db_path)

    # Iteration 1 with restart_strict=True, but replan_PercentOfIter=0.0:
    # pruning beam_plans_out is expected and must NOT raise!
    settings = SimpleNamespace(
        run=SimpleNamespace(archive_run_dir=archive_dir, restart_strict=True),
        beam=SimpleNamespace(replan_PercentOfIter=0.0),
    )
    state = SimpleNamespace(
        scenario_id="archive_scenario",
        iteration=1,
        run_info_path=str(archive_dir / "run_info.yaml"),
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

        assert "linkstats" in restored
        assert "linkstats" in sc_restart.coupler
        assert "beam_plans_out" not in restored
        assert sc_restart.coupler.get("beam_plans_out") is None
        assert "beam_plans_out" in restored.pruned_keys
        assert (fresh_ws / "outputs" / "linkstats.csv.gz").exists()


def test_restore_restart_coupler_snapshot_stages_directory_artifacts_and_enforces_warmstart_strictness(
    tmp_path,
):
    import shutil

    db_path = str(tmp_path / "provenance.duckdb")
    archive_dir = tmp_path / "archive_run"
    archive_dir.mkdir()

    node1_ws = tmp_path / "node1_workspace"
    node1_ws.mkdir()
    zarr_src = node1_ws / "skims" / "skims.zarr"
    zarr_src.mkdir(parents=True, exist_ok=True)
    (zarr_src / ".zgroup").write_text('{"zarr_format": 2}\n')

    zarr_arch = archive_dir / "skims" / "skims.zarr"
    shutil.copytree(zarr_src, zarr_arch)

    ephemeral_plans = node1_ws / "outputs" / "beam_plans_out.csv.gz"
    ephemeral_plans.parent.mkdir(parents=True, exist_ok=True)
    ephemeral_plans.write_text("ephemeral-plans\n")

    tracker_node1 = consist.Tracker(run_dir=node1_ws, db_path=db_path)
    with tracker_node1.scenario("archive_scenario") as sc_arch:

        def step_fn():
            art_zarr = tracker_node1.log_artifact(
                zarr_src,
                key="zarr_skims",
                driver="zarr",
                meta={"directory_artifact": True},
            )
            art_plans = tracker_node1.log_artifact(
                ephemeral_plans,
                key="beam_plans_out",
            )
            sc_arch.coupler.set("zarr_skims", art_zarr)
            sc_arch.coupler.set("beam_plans_out", art_plans)

        sc_arch.run(step_fn, name="produce_zarr_and_plans")
        sc_arch.capture_coupler_snapshot(phase="post")

    shutil.rmtree(node1_ws)

    fresh_ws = tmp_path / "fresh_workspace"
    fresh_ws.mkdir()
    tracker_fresh = consist.Tracker(run_dir=fresh_ws, db_path=db_path)

    # When replan_PercentOfIter > 0 and iteration > 0, missing beam_plans_out raises under restart_strict=True
    strict_settings = SimpleNamespace(
        run=SimpleNamespace(archive_run_dir=archive_dir, restart_strict=True),
        beam=SimpleNamespace(replan_PercentOfIter=0.2),
    )
    state = SimpleNamespace(
        scenario_id="archive_scenario",
        iteration=1,
        run_info_path=str(archive_dir / "run_info.yaml"),
    )
    workspace = SimpleNamespace(full_path=fresh_ws, archive_run_dir=archive_dir)

    with tracker_fresh.scenario("restart_scenario") as sc_restart:
        with pytest.raises(
            RuntimeError,
            match="Strict restart failed: convergence artifact 'beam_plans_out'",
        ):
            restart_runtime.restore_restart_coupler_snapshot(
                scenario=sc_restart,
                tracker=tracker_fresh,
                settings=strict_settings,
                state=state,
                workspace=workspace,
            )

        # Even though beam_plans_out triggered strict policy, zarr_skims directory artifact was materialized
        assert (fresh_ws / "skims" / "skims.zarr" / ".zgroup").exists()
        assert "zarr_skims" in sc_restart.coupler
        assert sc_restart.coupler.get("beam_plans_out") is None
