from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from consist import BindingResult, CacheOptions, ExecutionOptions, define_step
from consist.api import archive_run_output_files
from consist.core.tracker import Tracker
from consist.models.artifact import Artifact

from pilates.workflows.beam_checkpoint import (
    HistoricalOutputRequest,
    PinnedClosureMember,
    beam_postprocess_is_in_progress,
    mark_beam_postprocess_in_progress,
    publish_beam_run_checkpoint,
    read_beam_run_checkpoint,
)
from pilates.workflows.stages import supply_demand_beam as beam_stage
from pilates.workflows.steps.beam import (
    _materialize_native_outputs,
)


def _init_tracker(tmp_path: Path) -> Tracker:
    db_path = tmp_path / "consist" / "provenance.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(db_path),
        hashing_strategy="fast",
    )


# ---------------------------------------------------------------------------
# Scenario 1: Crash during beam_run
# Run is uncommitted in Consist. On restart, cache miss triggers clean rerun.
# ---------------------------------------------------------------------------


def test_scenario_1_crash_during_beam_run_uncommitted_and_clean_rerun_on_restart(
    tmp_path: Path,
) -> None:
    tracker = _init_tracker(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    beam_config = workspace / "beam.conf"
    beam_config.write_text("beam.sampling = 0.1\n", encoding="utf-8")

    execution_count = 0

    @define_step(
        model="beam_run",
        inputs={"config": None},
        output_paths={"linkstats": "0.linkstats.csv"},
        input_binding="paths",
        cache_hydration="outputs-all",
    )
    def failing_beam_run(config: Path, ctx: Any) -> None:
        nonlocal execution_count
        execution_count += 1
        if execution_count == 1:
            # Simulate crash midway through beam_run (e.g., JVM OOM or node preemption)
            raise RuntimeError("BEAM simulation crashed during execution")
        # Second run succeeds and writes output
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        (ctx.run_dir / "0.linkstats.csv").write_text("link,volume,capacity\n1,100,500\n")

    # Run 1: Crash during beam_run
    with pytest.raises(RuntimeError, match="BEAM simulation crashed"):
        with tracker.scenario("beam-workflow") as scenario:
            scenario.run(
                fn=failing_beam_run,
                binding=BindingResult(inputs={"config": beam_config}),
                execution_options=ExecutionOptions(
                    input_binding="paths", inject_context="ctx"
                ),
                cache_options=CacheOptions(
                    cache_hydration="outputs-all",
                ),
            )

    # Verify no committed completed run in tracker
    records = tracker.find_runs(model="beam_run", status="completed")
    assert len(records) == 0

    # Run 2: Restart after crash -> triggers clean rerun (cache miss)
    with tracker.scenario("beam-workflow") as scenario:
        result = scenario.run(
            fn=failing_beam_run,
            binding=BindingResult(inputs={"config": beam_config}),
            execution_options=ExecutionOptions(
                input_binding="paths", inject_context="ctx"
            ),
            cache_options=CacheOptions(
                cache_hydration="outputs-all",
            ),
        )

    assert result.cache_hit is False
    assert execution_count == 2
    assert "linkstats" in result.outputs
    assert result.outputs["linkstats"].path.read_text().startswith("link,volume")
    # Verify now committed
    records = tracker.find_runs(model="beam_run", status="completed")
    assert len(records) == 1


# ---------------------------------------------------------------------------
# Scenario 2: Crash after beam_run commits before beam_postprocess starts
# beam_run is committed. On restart, cache hit hydrates outputs and beam_postprocess runs.
# ---------------------------------------------------------------------------


def test_scenario_2_crash_after_beam_run_commits_hydrates_and_completes_postprocess(
    tmp_path: Path,
) -> None:
    tracker = _init_tracker(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    beam_config = workspace / "beam.conf"
    beam_config.write_text("beam.sampling = 0.2\n", encoding="utf-8")

    beam_run_calls = 0
    postprocess_calls = 0

    @define_step(
        model="beam_run",
        inputs={"config": None},
        output_paths={"linkstats": "0.linkstats.csv"},
        input_binding="paths",
        cache_hydration="outputs-all",
    )
    def step_beam_run(config: Path, ctx: Any) -> None:
        nonlocal beam_run_calls
        beam_run_calls += 1
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        (ctx.run_dir / "0.linkstats.csv").write_text("link,volume\n42,500\n")

    @define_step(
        model="beam_postprocess",
        inputs={"linkstats": None},
        output_paths={"skims": "final_skims.omx"},
        input_binding="paths",
        cache_hydration="outputs-all",
    )
    def step_beam_postprocess(linkstats: Path, ctx: Any) -> None:
        nonlocal postprocess_calls
        postprocess_calls += 1
        assert linkstats.read_text().startswith("link,volume")
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        (ctx.run_dir / "final_skims.omx").write_text("omx_data_matrix\n")

    # Invocation 1: beam_run runs and commits. Process stops/crashes before postprocess starts.
    with tracker.scenario("beam-workflow") as scenario:
        run_res = scenario.run(
            fn=step_beam_run,
            binding=BindingResult(inputs={"config": beam_config}),
            execution_options=ExecutionOptions(
                input_binding="paths", inject_context="ctx"
            ),
            cache_options=CacheOptions(
                cache_hydration="outputs-all",
            ),
        )
        assert run_res.cache_hit is False
        assert beam_run_calls == 1

    # Invocation 2: Restart workflow.
    # beam_run hits cache, hydrates outputs, and postprocess executes to completion.
    with tracker.scenario("beam-workflow") as scenario:
        re_run_res = scenario.run(
            fn=step_beam_run,
            binding=BindingResult(inputs={"config": beam_config}),
            execution_options=ExecutionOptions(
                input_binding="paths", inject_context="ctx"
            ),
            cache_options=CacheOptions(
                cache_hydration="outputs-all",
            ),
        )
        assert re_run_res.cache_hit is True
        assert beam_run_calls == 1  # Not re-executed!

        post_res = scenario.run(
            fn=step_beam_postprocess,
            binding=BindingResult(inputs={"linkstats": re_run_res.outputs["linkstats"].path}),
            execution_options=ExecutionOptions(
                input_binding="paths", inject_context="ctx"
            ),
            cache_options=CacheOptions(
                cache_hydration="outputs-all",
            ),
        )
        assert post_res.cache_hit is False
        assert postprocess_calls == 1
        assert post_res.outputs["skims"].path.read_text() == "omx_data_matrix\n"


# ---------------------------------------------------------------------------
# Scenario 3: Crash partway through beam_postprocess
# Mutation gate / staging cleanup and allow_in_progress resume.
# ---------------------------------------------------------------------------


def test_scenario_3_crash_partway_through_postprocess_mutation_gate_and_atomic_staging(
    tmp_path: Path,
) -> None:
    archive_run_dir = tmp_path / "archive_run"
    archive_run_dir.mkdir(parents=True, exist_ok=True)
    beam_output_dir = tmp_path / "beam_output"
    beam_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup a valid checkpoint published after beam_run
    closure_member = PinnedClosureMember(
        member_id="beam-run-1:linkstats",
        role="linkstats",
        producer_run_id="beam-run-1",
        output_key="linkstats",
        artifact_identity="hash-linkstats",
        artifact_kind="file",
        driver="other",
        destination=beam_output_dir / "0.linkstats.csv",
        required=True,
    )
    (archive_run_dir / "snapshot.json").write_text("{}", encoding="utf-8")

    checkpoint = publish_beam_run_checkpoint(
        archive_run_dir=archive_run_dir,
        producer_run_id="beam-run-1",
        scope={"year": 2021, "iteration": 0},
        snapshot_ref="snapshot.json",
        skim_variant="standard",
        output_requests=[
            HistoricalOutputRequest(
                key="linkstats",
                destination=beam_output_dir / "0.linkstats.csv",
                required=True,
            )
        ],
        closure_members=[closure_member],
    )
    assert checkpoint.producer_run_id == "beam-run-1"
    assert read_beam_run_checkpoint(archive_run_dir) == checkpoint

    # 2. Simulate postprocess started: marked in-progress
    mark_beam_postprocess_in_progress(archive_run_dir, checkpoint)
    assert beam_postprocess_is_in_progress(archive_run_dir) is True
    # Normal read_beam_run_checkpoint returns None because next_boundary == _IN_PROGRESS
    assert read_beam_run_checkpoint(archive_run_dir) is None
    # But with allow_in_progress=True, it reads the checkpoint for recovery!
    resumed_ckpt = read_beam_run_checkpoint(archive_run_dir, allow_in_progress=True)
    assert resumed_ckpt is not None
    assert resumed_ckpt.producer_run_id == "beam-run-1"

    # 3. Simulate crash leaving behind dirty partial staging files in workspace
    # 3. Simulate crash leaving behind dirty partial staging files across workspace roots
    staging_file = beam_output_dir / ".staging_skims.omx.tmp"
    staging_file.write_text("corrupted partial omx write")
    dirty_output_dir = beam_output_dir / ".pilates-consist-outputs" / "beam_postprocess"
    dirty_output_dir.mkdir(parents=True, exist_ok=True)
    (dirty_output_dir / "dirty_chunk.tmp").write_text("dirty chunk")

    asim_output_dir = tmp_path / "asim_output"
    asim_cache_dir = asim_output_dir / "cache"
    asim_cache_dir.mkdir(parents=True, exist_ok=True)
    dirty_zarr_staging = asim_cache_dir / ".staging_skims.zarr.tmp"
    dirty_zarr_staging.mkdir(parents=True, exist_ok=True)
    (dirty_zarr_staging / "dirty_chunk.zarr").write_text("dirty zarr")

    beam_mutable_dir = tmp_path / "beam_mutable"
    beam_mutable_dir.mkdir(parents=True, exist_ok=True)
    dirty_omx_staging = beam_mutable_dir / ".staging_final_skims.omx.tmp"
    dirty_omx_staging.write_text("dirty omx")

    workspace = SimpleNamespace(
        full_path=str(tmp_path),
        get_beam_output_dir=lambda: str(beam_output_dir),
        get_asim_output_dir=lambda: str(asim_output_dir),
        get_beam_mutable_data_dir=lambda: str(beam_mutable_dir),
    )

    # 4. Clean interrupted workspace across all candidate roots
    beam_stage._clean_interrupted_beam_postprocess_workspace(workspace=workspace)
    assert not staging_file.exists()
    assert not dirty_output_dir.exists()
    assert not dirty_zarr_staging.exists()
    assert not dirty_omx_staging.exists()

    # 5. Atomic staging test in _materialize_native_outputs (files and directories)
    source_file = tmp_path / "source_skims.omx"
    source_file.write_text("complete final omx data")
    destination_file = beam_output_dir / "skims.omx"

    _materialize_native_outputs(
        source_paths={"skims": source_file},
        declared_outputs={"skims": destination_file},
    )
    assert destination_file.exists()
    assert destination_file.read_text() == "complete final omx data"
    assert not (beam_output_dir / ".staging_skims.omx.tmp").exists()

    # 6. Type transitions: leftover staging file when source is a directory
    source_dir = tmp_path / "source_dir_skims"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "data.bin").write_text("dir data")
    dest_dir = beam_output_dir / "dir_skims"
    leftover_staging_file = beam_output_dir / f".staging_{dest_dir.name}.tmp"
    leftover_staging_file.write_text("leftover file instead of dir")

    _materialize_native_outputs(
        source_paths={"dir_skims": source_dir},
        declared_outputs={"dir_skims": dest_dir},
    )
    assert dest_dir.is_dir()
    assert (dest_dir / "data.bin").read_text() == "dir data"
    assert not leftover_staging_file.exists()

    # 7. Type transitions: leftover staging dir when source is a file
    source_file2 = tmp_path / "source_file2.txt"
    source_file2.write_text("file2 data")
    dest_file2 = beam_output_dir / "file2.txt"
    leftover_staging_dir = beam_output_dir / f".staging_{dest_file2.name}.tmp"
    leftover_staging_dir.mkdir(parents=True, exist_ok=True)
    (leftover_staging_dir / "junk.txt").write_text("junk")

    _materialize_native_outputs(
        source_paths={"file2": source_file2},
        declared_outputs={"file2": dest_file2},
    )
    assert dest_file2.is_file()
    assert dest_file2.read_text() == "file2 data"
    assert not leftover_staging_dir.exists()


# ---------------------------------------------------------------------------
# Scenario 4: Scratch wiped, beam_run bytes exist only under archive/recovery root
# Restart hydrates outputs from recovery roots back into scratch workspace.
# ---------------------------------------------------------------------------


def test_scenario_4_scratch_wiped_hydrates_from_recovery_root(tmp_path: Path) -> None:
    tracker = _init_tracker(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    beam_config = workspace / "beam.conf"
    beam_config.write_text("beam.mode = fast\n", encoding="utf-8")
    recovery_root = tmp_path / "archive_recovery_root"
    recovery_root.mkdir(parents=True, exist_ok=True)

    @define_step(
        model="beam_run",
        inputs={"config": None},
        output_paths={"linkstats": "output/linkstats.csv"},
        input_binding="paths",
        cache_hydration="outputs-all",
    )
    def produce_linkstats(config: Path, ctx: Any) -> None:
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        out_file = ctx.run_dir / "output" / "linkstats.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("link,speed,volume\n1,30,120\n")

    # 1. Initial run produces output in run_dir
    with tracker.scenario("beam-workflow") as scenario:
        result = scenario.run(
            fn=produce_linkstats,
            binding=BindingResult(inputs={"config": beam_config}),
            execution_options=ExecutionOptions(
                input_binding="paths", inject_context="ctx"
            ),
            cache_options=CacheOptions(
                cache_hydration="outputs-all",
            ),
        )
        assert result.cache_hit is False
        produced_path = result.outputs["linkstats"].path
        assert produced_path.exists()
        assert produced_path.read_text().startswith("link,speed")

    # 2. Archive output files to recovery root
    archive_report = archive_run_output_files(
        run_id=result.run.id,
        recovery_root=recovery_root,
        tracker=tracker,
        verify=False,
        append=True,
    )
    assert archive_report.complete is True
    target_path = archive_report["linkstats"].target_path
    assert target_path is not None
    assert target_path.exists()
    assert target_path.read_text().startswith("link,speed")

    # 3. Simulate ephemeral node scratch wipe: delete the scratch run directory!
    shutil.rmtree(tracker.run_dir)
    assert not produced_path.exists()

    # 4. Re-execute step on fresh scratch node with identical inputs:
    # Tracker should detect cache hit and hydrate the missing outputs from recovery_root!
    with tracker.scenario("beam-workflow") as scenario:
        rerun_result = scenario.run(
            fn=produce_linkstats,
            binding=BindingResult(inputs={"config": beam_config}),
            execution_options=ExecutionOptions(
                input_binding="paths", inject_context="ctx"
            ),
            cache_options=CacheOptions(
                cache_hydration="outputs-all",
            ),
        )
        assert rerun_result.cache_hit is True
        materialized = rerun_result.run.meta.get("materialized_outputs", {})
        assert "linkstats" in materialized
        hydrated_path = Path(materialized["linkstats"])
        assert hydrated_path.exists()
        assert hydrated_path.read_text().startswith("link,speed")


# ---------------------------------------------------------------------------
# Scenario 5: Config change triggers cache miss
# Altering launch config alters Merkle identity, triggering cache miss.
# ---------------------------------------------------------------------------


def test_scenario_5_config_change_triggers_cache_miss(tmp_path: Path) -> None:
    tracker = _init_tracker(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config_v1 = workspace / "beam_v1.conf"
    config_v1.write_text("beam.agents = 1000\n", encoding="utf-8")
    config_v2 = workspace / "beam_v2.conf"
    config_v2.write_text("beam.agents = 2000\n", encoding="utf-8")

    execution_calls: list[str] = []

    @define_step(
        model="beam_run",
        inputs={"config": None},
        output_paths={"linkstats": "linkstats.csv"},
        input_binding="paths",
        cache_hydration="outputs-all",
    )
    def configurable_beam_run(config: Path, ctx: Any) -> None:
        content = config.read_text()
        execution_calls.append(content.strip())
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        (ctx.run_dir / "linkstats.csv").write_text(f"result for {content}")

    # 1. Run with Config V1 -> executed (miss)
    with tracker.scenario("beam-workflow") as scenario:
        res1 = scenario.run(
            fn=configurable_beam_run,
            binding=BindingResult(inputs={"config": config_v1}),
            execution_options=ExecutionOptions(
                input_binding="paths", inject_context="ctx"
            ),
            cache_options=CacheOptions(
                cache_hydration="outputs-all",
            ),
        )
        assert res1.cache_hit is False
        assert execution_calls == ["beam.agents = 1000"]

    # 2. Run again with same Config V1 -> cached (hit)
    with tracker.scenario("beam-workflow") as scenario:
        res2 = scenario.run(
            fn=configurable_beam_run,
            binding=BindingResult(inputs={"config": config_v1}),
            execution_options=ExecutionOptions(
                input_binding="paths", inject_context="ctx"
            ),
            cache_options=CacheOptions(
                cache_hydration="outputs-all",
            ),
        )
        assert res2.cache_hit is True
        assert len(execution_calls) == 1

    # 3. Run with Config V2 (different config) -> cache miss, re-executed!
    with tracker.scenario("beam-workflow") as scenario:
        res3 = scenario.run(
            fn=configurable_beam_run,
            binding=BindingResult(inputs={"config": config_v2}),
            execution_options=ExecutionOptions(
                input_binding="paths", inject_context="ctx"
            ),
            cache_options=CacheOptions(
                cache_hydration="outputs-all",
            ),
        )
        assert res3.cache_hit is False
        assert len(execution_calls) == 2
        assert execution_calls == ["beam.agents = 1000", "beam.agents = 2000"]


# ---------------------------------------------------------------------------
# Integration: _run_beam_steps archives outputs and recovers on restart
# ---------------------------------------------------------------------------


def test_run_beam_steps_archives_outputs_and_recovers_interrupted_postprocess(
    tmp_path: Path, monkeypatch
) -> None:
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    state_file = archive_dir / "run_info.yaml"
    state_file.write_text("{}", encoding="utf-8")

    class Coupler:
        def __init__(self) -> None:
            self.store: dict[str, Any] = {}

        def get(self, key: str, default: Any = None) -> Any:
            return self.store.get(key, default)

        def set_from_artifact(self, key: str, art: Any) -> None:
            self.store[key] = art

    tracker = _init_tracker(tmp_path)
    scenario = SimpleNamespace(
        coupler=Coupler(),
        tracker=tracker,
    )
    workspace = SimpleNamespace(
        full_path=str(tmp_path / "workspace"),
        get_beam_output_dir=lambda: str(tmp_path / "workspace" / "beam_output"),
    )
    state = SimpleNamespace(
        is_restart_run=True,
        run_info_path=str(state_file),
        year=2021,
        iteration=0,
    )
    context = SimpleNamespace(
        settings=SimpleNamespace(beam=SimpleNamespace(full_skim=None)),
        state=state,
        workspace=workspace,
    )

    # Setup committed checkpoint with postprocess in progress
    closure_member = PinnedClosureMember(
        member_id="beam-run-1:events",
        role="events",
        producer_run_id="beam-run-1",
        output_key="events_parquet_2021_0",
        artifact_identity="hash-events",
        artifact_kind="file",
        driver="other",
        destination=tmp_path / "workspace" / "beam_output" / "events.parquet",
        required=True,
    )
    (archive_dir / "snapshot.json").write_text("{}", encoding="utf-8")

    checkpoint = publish_beam_run_checkpoint(
        archive_run_dir=archive_dir,
        producer_run_id="beam-run-1",
        scope={"year": 2021, "iteration": 0},
        snapshot_ref="snapshot.json",
        skim_variant="standard",
        output_requests=[
            HistoricalOutputRequest(
                key="events_parquet_2021_0",
                destination=tmp_path / "workspace" / "beam_output" / "events.parquet",
                required=True,
            )
        ],
        closure_members=[closure_member],
    )
    mark_beam_postprocess_in_progress(archive_dir, checkpoint)

    # Patch postprocess resumption helpers
    monkeypatch.setattr(
        beam_stage, "beam_checkpoint_resume_requested", lambda **_: True
    )
    monkeypatch.setattr(
        beam_stage, "checkpoint_fingerprint_matches", lambda *_, **__: True
    )
    monkeypatch.setattr(
        beam_stage, "verify_archive_visible_pinned_closure_bytes", lambda **_: None
    )
    restored_item = SimpleNamespace(
        path=tmp_path / "workspace" / "beam_output" / "events.parquet",
        artifact=Artifact(key="events", container_uri="workspace://events.parquet", driver="other"),
    )
    monkeypatch.setattr(
        beam_stage, "_open_beam_checkpoint_tracker", lambda *_, **__: SimpleNamespace()
    )
    monkeypatch.setattr(
        beam_stage, "validate_pinned_closure_snapshot", lambda *_, **__: None
    )
    monkeypatch.setattr(
        beam_stage, "hydrate_pinned_closure", lambda *_, **__: {closure_member.member_id: restored_item}
    )
    monkeypatch.setattr(
        beam_stage,
        "beam_postprocess",
        SimpleNamespace(
            resolve_inputs=lambda *_, **__: SimpleNamespace(
                require_complete=lambda: None,
                binding=BindingResult(inputs={"events": restored_item.artifact}),
            )
        ),
    )
    monkeypatch.setattr(
        beam_stage,
        "_rebind_beam_postprocess_closure_destinations",
        lambda members, **_: members,
    )
    monkeypatch.setattr(
        beam_stage, "_validate_rebound_postprocess_inputs", lambda *_, **__: None
    )
    monkeypatch.setattr(
        beam_stage,
        "execute_step",
        lambda **kwargs: (
            SimpleNamespace(run=SimpleNamespace(id="postprocess-resumed")),
            SimpleNamespace(_iter_record_items=lambda: [("skims", "/path/to/skims", "")]),
        ),
    )
    monkeypatch.setattr(
        beam_stage, "step_output_handoff_mapping", lambda *_args, **_kwargs: {"skims": "/path/to/skims"}
    )

    handoff = beam_stage._run_beam_steps(
        scenario=scenario,
        year=2021,
        iteration=0,
        context=context,
    )
    assert handoff == {"skims": "/path/to/skims"}
