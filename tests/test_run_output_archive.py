"""Regression coverage for native-output archival on split workspace storage."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from consist import OutputSet, RunResult
from consist.core.output_sets import register_output_sets
from consist.core.tracker import Tracker

from pilates.runtime.run_output_archive import archive_completed_run
from pilates.workflows.artifact_keys import ZARR_SKIMS


@pytest.mark.parametrize("zarr_key", (ZARR_SKIMS, "raw_od_skims_zarr_2018_0"))
def test_archive_completed_run_snapshots_direct_zarr_and_keeps_other_outputs(
    monkeypatch, tmp_path: Path, zarr_key: str
) -> None:
    """A direct Zarr output gets one durable snapshot without losing scalars."""

    local_root = tmp_path / "local"
    archive_root = tmp_path / "archive"
    tracker = Tracker(
        run_dir=archive_root,
        db_path=str(archive_root / "provenance.duckdb"),
        mounts={"workspace": str(local_root)},
        archive_mounts={"workspace": "."},
        hashing_strategy="full",
    )
    summary = local_root / "activitysim" / "output" / "final.parquet"
    summary.parent.mkdir(parents=True)
    summary.write_bytes(b"parquet bytes\n")
    zarr = local_root / "activitysim" / "output" / "cache" / "skims.zarr"
    (zarr / "nested").mkdir(parents=True)
    (zarr / ".zgroup").write_text("{}\n", encoding="utf-8")
    (zarr / "nested" / "0.0").write_bytes(b"skim bytes\n")

    run_id = "activitysim_snapshot"
    with tracker.start_run(run_id, model="activitysim"):
        logged_summary = tracker.log_output(summary, key="summary")
        logged_zarr = tracker.log_output(zarr, key=zarr_key, artifact_kind="directory")
    run = tracker.get_run(run_id)
    assert run is not None
    result = RunResult(
        run=run,
        outputs={"summary": logged_summary, zarr_key: logged_zarr},
    )
    original_archive = tracker.archive_run_outputs
    archive_calls: list[tuple[str, tuple[str, ...] | None]] = []

    def capture_archive(
        run_id: str,
        archive_path: str | Path,
        *,
        keys: tuple[str, ...] | None = None,
        mode: str,
    ):
        del archive_path
        archive_calls.append((run_id, keys))
        return original_archive(
            run_id, archive_root / "consist-recovery" / run_id, keys=keys, mode=mode
        )

    monkeypatch.setattr(tracker, "archive_run_outputs", capture_archive)
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")

    archived = archive_completed_run(tracker=tracker, result=result)

    recovery_root = archive_root / "consist-recovery" / str(run.id)
    assert archive_calls == [(str(run.id), ("summary",))]
    assert set(archived.outputs) == {"summary", zarr_key}
    assert archived.outputs["summary"].recovery_roots == [str(recovery_root.resolve())]
    assert archived.outputs[zarr_key].recovery_roots == [str(recovery_root.resolve())]
    snapshot = recovery_root / "activitysim" / "output" / "cache" / "skims.zarr"
    assert (snapshot / "nested" / "0.0").read_bytes() == b"skim bytes\n"

    shutil.rmtree(zarr)
    retried = archive_completed_run(tracker=tracker, result=result)
    assert set(retried.outputs) == {"summary", zarr_key}
    assert archive_calls == [(str(run.id), ("summary",)), (str(run.id), ("summary",))]

    destination = local_root / "hydrated" / "skims.zarr"
    hydrated = tracker.hydrate_run_outputs_to_destinations(
        str(run.id),
        destinations_by_key={zarr_key: destination},
        on_missing="raise",
    )

    assert hydrated[zarr_key].path == destination.resolve()
    assert (destination / "nested" / "0.0").read_bytes() == b"skim bytes\n"


def test_archive_completed_run_snapshots_direct_zarr_excluded_from_output_set(
    monkeypatch, tmp_path: Path
) -> None:
    """A direct Zarr beneath an OutputSet root remains separately recoverable."""

    local_root = tmp_path / "local"
    archive_root = tmp_path / "archive"
    tracker = Tracker(
        run_dir=archive_root,
        db_path=str(archive_root / "provenance.duckdb"),
        mounts={"workspace": str(local_root)},
        archive_mounts={"workspace": "."},
        hashing_strategy="full",
    )
    beam_root = local_root / "beam" / "output" / "seattle" / "year-2018-iteration-0"
    report = beam_root / "ITERS" / "it.0" / "events.parquet"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"event bytes\n")
    zarr = beam_root / "ITERS" / "it.0" / "od_skims.zarr"
    (zarr / "nested").mkdir(parents=True)
    (zarr / ".zgroup").write_text("{}\n", encoding="utf-8")
    (zarr / "nested" / "0.0").write_bytes(b"skim bytes\n")

    run_id = "beam_run"
    with tracker.start_run(run_id, model="beam"):
        logged_zarr = tracker.log_output(
            zarr,
            key="raw_od_skims_zarr_2018_0",
            artifact_kind="directory",
        )
        output_set = register_output_sets(
            tracker=tracker,
            output_sets={
                "beam_run_outputs": OutputSet(
                    root=beam_root,
                    include="**/*",
                    exclude="**/*.zarr/**",
                    recursive=True,
                )
            },
            config=None,
            output_base_dir=tracker.run_artifact_dir(),
        )["beam_run_outputs"]
    run = tracker.get_run(run_id)
    assert run is not None

    archive_calls: list[tuple[str, tuple[str, ...] | None]] = []
    original_archive = tracker.archive_run_outputs

    def capture_archive(
        run_id: str,
        archive_path: str | Path,
        *,
        keys: tuple[str, ...] | None = None,
        mode: str,
    ):
        archive_calls.append((run_id, keys))
        return original_archive(run_id, archive_path, keys=keys, mode=mode)

    monkeypatch.setattr(tracker, "archive_run_outputs", capture_archive)
    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))
    monkeypatch.setenv("PILATES_ENABLE_ARCHIVE_COPY", "1")

    archived = archive_completed_run(
        tracker=tracker,
        result=RunResult(
            run=run,
            outputs={
                "raw_od_skims_zarr_2018_0": logged_zarr,
                "beam_run_outputs": output_set,
            },
        ),
    )

    recovery_root = archive_root / "consist-recovery" / run_id
    assert archive_calls == [(run_id, ("beam_run_outputs",))]
    assert (recovery_root / "beam" / "output" / "seattle").is_dir()
    assert (
        recovery_root
        / "beam"
        / "output"
        / "seattle"
        / "year-2018-iteration-0"
        / "ITERS"
        / "it.0"
        / "od_skims.zarr"
        / "nested"
        / "0.0"
    ).read_bytes() == b"skim bytes\n"
    assert archived.outputs["raw_od_skims_zarr_2018_0"].recovery_roots == [
        str(recovery_root.resolve())
    ]
