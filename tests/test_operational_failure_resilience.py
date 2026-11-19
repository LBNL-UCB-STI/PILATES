import json
import os
import pytest
from unittest.mock import MagicMock, patch

from pilates.utils.consist_runtime import dispose_tracker
from pilates.utils.consist_db_snapshot import (
    mirror_consist_db_to_archive,
    _fallback_copy_db_snapshot,
)
from pilates.activitysim.runner import validate_activitysim_zarr_skims
from pilates.activitysim.preprocessor import _resolve_preprocessor_workers
from pilates.beam.postprocessor import find_latest_beam_iteration
from pilates.beam.runner import rename_beam_output_directory


def test_dispose_tracker_releases_engine():
    # 1. None tracker
    dispose_tracker(None)

    # 2. Tracker without db or engine
    mock_tracker_no_db = MagicMock(spec=[])
    dispose_tracker(mock_tracker_no_db)

    # 3. Tracker with db and engine
    mock_tracker = MagicMock()
    mock_engine = MagicMock()
    mock_tracker.db.engine = mock_engine
    dispose_tracker(mock_tracker)
    mock_engine.dispose.assert_called_once()

    # 4. Tracker where dispose raises
    mock_failing_engine = MagicMock()
    mock_failing_engine.dispose.side_effect = RuntimeError("Lock release error")
    mock_tracker.db.engine = mock_failing_engine
    # Should not propagate exception
    dispose_tracker(mock_tracker)


def test_mirror_consist_db_to_archive_atomic_and_checkpoint(tmp_path):
    local_dir = tmp_path / "local"
    archive_dir = tmp_path / "archive"
    local_dir.mkdir()
    archive_dir.mkdir()

    db_file = local_dir / "provenance.duckdb"
    wal_file = local_dir / "provenance.duckdb.wal"
    db_file.write_text("dummy db content")
    wal_file.write_text("dummy wal content")

    archive_db_file = archive_dir / "provenance.duckdb"
    archive_wal_file = archive_dir / "provenance.duckdb.wal"

    mock_tracker = MagicMock()
    mock_conn = MagicMock()
    mock_tracker.db.engine.begin.return_value.__enter__.return_value = mock_conn

    mirror_consist_db_to_archive(
        str(db_file),
        str(archive_db_file),
        tracker=mock_tracker,
    )

    mock_conn.exec_driver_sql.assert_called_once_with("CHECKPOINT")
    assert archive_db_file.read_text() == "dummy db content"
    assert archive_wal_file.read_text() == "dummy wal content"

    # Ensure no leftover temporary files exist in archive_dir
    tmp_files = list(archive_dir.glob("*.tmp.*"))
    assert len(tmp_files) == 0


def test_mirror_consist_db_to_archive_removes_stale_wal_when_none_in_source(tmp_path):
    local_dir = tmp_path / "local"
    archive_dir = tmp_path / "archive"
    local_dir.mkdir()
    archive_dir.mkdir()

    db_file = local_dir / "provenance.duckdb"
    db_file.write_text("dummy db content")

    archive_db_file = archive_dir / "provenance.duckdb"
    archive_wal_file = archive_dir / "provenance.duckdb.wal"
    archive_wal_file.write_text("old stale wal content")

    mirror_consist_db_to_archive(
        str(db_file),
        str(archive_db_file),
    )

    assert archive_db_file.read_text() == "dummy db content"
    assert not archive_wal_file.exists()


def test_fallback_copy_db_snapshot_atomic(tmp_path):
    local_dir = tmp_path / "local"
    archive_dir = tmp_path / "archive"
    local_dir.mkdir()
    archive_dir.mkdir()

    db_file = local_dir / "provenance.duckdb"
    db_file.write_text("fallback db content")

    dest_db = archive_dir / "snapshot_01.duckdb"
    success = _fallback_copy_db_snapshot(
        local_db_path=str(db_file),
        dest_db_path=dest_db,
        checkpoint=True,
    )
    assert success is True
    assert dest_db.read_text() == "fallback db content"
    tmp_files = list(archive_dir.glob("*.tmp.*"))
    assert len(tmp_files) == 0


def test_validate_activitysim_zarr_skims_checks_array_metadata(tmp_path):
    zarr_dir = tmp_path / "skims.zarr"
    zarr_dir.mkdir()
    (zarr_dir / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")

    # Valid array child
    array_dir = zarr_dir / "sov"
    array_dir.mkdir()
    (array_dir / ".zarray").write_text(
        json.dumps({"zarr_format": 2, "shape": [100, 100], "chunks": [10, 10]}),
        encoding="utf-8",
    )
    assert validate_activitysim_zarr_skims(zarr_dir) is None

    # Corrupt array metadata
    (array_dir / ".zarray").write_text(
        json.dumps({"zarr_format": 2}),  # missing shape
        encoding="utf-8",
    )
    rejection = validate_activitysim_zarr_skims(zarr_dir)
    assert rejection is not None
    assert "missing shape" in rejection

    # Non-json array metadata
    (array_dir / ".zarray").write_text("not json", encoding="utf-8")
    rejection2 = validate_activitysim_zarr_skims(zarr_dir)
    assert rejection2 is not None
    assert "invalid Zarr array metadata" in rejection2


def test_resolve_preprocessor_workers_bounds():
    # Test capping by num_tasks
    assert _resolve_preprocessor_workers(0) == 1
    assert _resolve_preprocessor_workers(2) <= 2

    # Test default ceiling (max 16)
    with patch.dict(os.environ, {}, clear=True):
        workers = _resolve_preprocessor_workers(100)
        assert workers <= 16

    # Test environment variable override
    with patch("os.cpu_count", return_value=32):
        with patch.dict(os.environ, {"PILATES_PREPROCESSOR_WORKERS": "4"}):
            workers = _resolve_preprocessor_workers(100)
            assert workers == 4


def test_find_latest_beam_iteration_excludes_archived_dirs(tmp_path):
    beam_out = tmp_path / "beam_output"
    beam_out.mkdir()

    # Archived previous run
    archived_iter = beam_out / "sfbay" / "year-2018-iteration-0" / "ITERS" / "it.10"
    archived_iter.mkdir(parents=True)

    # Newer active run
    active_iter = beam_out / "sfbay" / "beam_run_2026" / "ITERS" / "it.5"
    active_iter.mkdir(parents=True)

    # Ensure active_iter has older mtime to prove exclusion logic works regardless of timestamp
    os.utime(str(beam_out / "sfbay" / "year-2018-iteration-0" / "ITERS"), (2000, 2000))
    os.utime(str(beam_out / "sfbay" / "beam_run_2026" / "ITERS"), (1000, 1000))

    iter_path, it_num = find_latest_beam_iteration(str(beam_out))
    assert iter_path == str(active_iter)
    assert it_num == 5


def test_rename_beam_output_directory_raises_when_no_iteration(tmp_path):
    empty_dir = tmp_path / "empty_beam"
    empty_dir.mkdir()
    settings = MagicMock()

    with pytest.raises(RuntimeError, match="No BEAM iteration output directory found"):
        rename_beam_output_directory(str(empty_dir), settings, year=2020)
