from pathlib import Path
from types import SimpleNamespace

from pilates.urbansim.inputs import build_urbansim_inputs
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
)


class _WorkspaceStub:
    def __init__(self, full_root: Path, usim_dir: Path = None) -> None:
        self._root = full_root
        self.full_path = str(full_root)
        self._usim_dir = usim_dir or full_root

    def get_usim_mutable_data_dir(self) -> str:
        return str(self._usim_dir)


def _settings_stub():
    return SimpleNamespace(
        run=SimpleNamespace(region="test"),
        urbansim=SimpleNamespace(
            region_mappings={"region_to_region_id": {"test": "000"}},
            input_file_template="usim_{region_id}.h5",
            output_file_template="usim_{year}.h5",
        ),
    )


def test_build_urbansim_inputs_prefers_output_for_non_start_year(tmp_path: Path):
    settings = _settings_stub()
    workspace = _WorkspaceStub(tmp_path, tmp_path)
    state = SimpleNamespace(is_start_year=lambda: False)

    base_h5 = tmp_path / "usim_000.h5"
    output_h5 = tmp_path / "usim_2019.h5"
    base_h5.write_text("base")
    output_h5.write_text("current")

    inputs, _ = build_urbansim_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        year=2019,
    )

    assert inputs[USIM_DATASTORE_BASE_H5] == str(base_h5)
    assert inputs[USIM_DATASTORE_CURRENT_H5] == str(output_h5)


def test_build_urbansim_inputs_falls_back_to_base_when_output_missing(
    tmp_path: Path,
):
    settings = _settings_stub()
    workspace = _WorkspaceStub(tmp_path, tmp_path)
    state = SimpleNamespace(is_start_year=lambda: False)

    base_h5 = tmp_path / "usim_000.h5"
    base_h5.write_text("base")

    inputs, _ = build_urbansim_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        year=2019,
    )

    assert inputs[USIM_DATASTORE_BASE_H5] == str(base_h5)
    assert inputs[USIM_DATASTORE_CURRENT_H5] == str(base_h5)


def test_build_urbansim_inputs_falls_back_to_archive_base_when_local_missing(
    tmp_path: Path,
):
    settings = _settings_stub()
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    workspace = _WorkspaceStub(local_run_dir, local_run_dir / "urbansim" / "data")
    state = SimpleNamespace(
        is_start_year=lambda: False,
        run_info_path=str(archive_run_dir / "run_state.yaml"),
    )

    archive_base = archive_run_dir / "urbansim" / "data" / "usim_000.h5"
    archive_base.parent.mkdir(parents=True, exist_ok=True)
    archive_base.write_text("base")

    inputs, _ = build_urbansim_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        year=2019,
    )

    assert inputs[USIM_DATASTORE_BASE_H5] == str(archive_base)
    assert inputs[USIM_DATASTORE_CURRENT_H5] == str(archive_base)


def test_build_urbansim_inputs_prefers_archive_output_for_current_when_local_missing(
    tmp_path: Path,
):
    settings = _settings_stub()
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    workspace = _WorkspaceStub(local_run_dir, local_run_dir / "urbansim" / "data")
    state = SimpleNamespace(
        is_start_year=lambda: False,
        run_info_path=str(archive_run_dir / "run_state.yaml"),
    )

    archive_base = archive_run_dir / "urbansim" / "data" / "usim_000.h5"
    archive_output = archive_run_dir / "urbansim" / "data" / "usim_2019.h5"
    archive_base.parent.mkdir(parents=True, exist_ok=True)
    archive_base.write_text("base")
    archive_output.write_text("current")

    inputs, _ = build_urbansim_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        year=2019,
    )

    assert inputs[USIM_DATASTORE_BASE_H5] == str(archive_base)
    assert inputs[USIM_DATASTORE_CURRENT_H5] == str(archive_output)
