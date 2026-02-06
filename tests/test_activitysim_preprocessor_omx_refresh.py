import os
import shutil
import time
from pathlib import Path

from pilates.activitysim.preprocessor import _should_refresh_skims_copy


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_should_refresh_skims_copy_when_dest_missing(tmp_path: Path) -> None:
    source = tmp_path / "beam" / "skims.omx"
    dest = tmp_path / "asim" / "skims.omx"
    _write(source, b"source-data")

    assert _should_refresh_skims_copy(str(source), str(dest)) is True


def test_should_refresh_skims_copy_when_source_is_newer(tmp_path: Path) -> None:
    source = tmp_path / "beam" / "skims.omx"
    dest = tmp_path / "asim" / "skims.omx"
    _write(source, b"same-size")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)

    time.sleep(0.01)
    _write(source, b"same-size")

    assert _should_refresh_skims_copy(str(source), str(dest)) is True


def test_should_not_refresh_skims_copy_when_dest_is_up_to_date(tmp_path: Path) -> None:
    source = tmp_path / "beam" / "skims.omx"
    dest = tmp_path / "asim" / "skims.omx"
    _write(source, b"stable")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)

    future = time.time() + 60
    os.utime(dest, (future, future))

    assert _should_refresh_skims_copy(str(source), str(dest)) is False


def test_should_refresh_skims_copy_when_size_differs(tmp_path: Path) -> None:
    source = tmp_path / "beam" / "skims.omx"
    dest = tmp_path / "asim" / "skims.omx"
    _write(source, b"large-content")
    _write(dest, b"small")

    assert _should_refresh_skims_copy(str(source), str(dest)) is True
