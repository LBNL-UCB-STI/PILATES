from pathlib import Path
from types import SimpleNamespace

from pilates.urbansim import inputs as urbansim_inputs_module
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


class _StateStub:
    class Stage:
        land_use = "land_use"

    def __init__(
        self,
        *,
        start_year: bool,
        land_use_enabled: bool,
        run_info_path: str | None = None,
    ):
        self._start_year = start_year
        self._land_use_enabled = land_use_enabled
        self.run_info_path = run_info_path

    def is_start_year(self) -> bool:
        return self._start_year

    def is_enabled(self, stage) -> bool:
        return stage == self.Stage.land_use and self._land_use_enabled


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
    state = _StateStub(start_year=False, land_use_enabled=True)

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


def test_build_urbansim_inputs_uses_base_for_current_on_start_year(tmp_path: Path):
    settings = _settings_stub()
    workspace = _WorkspaceStub(tmp_path, tmp_path)
    state = _StateStub(start_year=True, land_use_enabled=True)

    base_h5 = tmp_path / "usim_000.h5"
    base_h5.write_text("base")

    inputs, _ = build_urbansim_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        year=2018,
    )

    assert inputs[USIM_DATASTORE_BASE_H5] == str(base_h5)
    assert inputs[USIM_DATASTORE_CURRENT_H5] == str(base_h5)


def test_build_urbansim_inputs_falls_back_to_base_when_output_missing(
    tmp_path: Path,
):
    settings = _settings_stub()
    workspace = _WorkspaceStub(tmp_path, tmp_path)
    state = _StateStub(start_year=False, land_use_enabled=True)

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
    state = _StateStub(
        start_year=False,
        land_use_enabled=True,
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
    state = _StateStub(
        start_year=False,
        land_use_enabled=True,
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


def test_build_urbansim_inputs_forwards_surface(monkeypatch, tmp_path: Path):
    captured = {}
    original = urbansim_inputs_module.build_binding_plan
    surface = SimpleNamespace(profile=None, step_surface=lambda _name: None)

    def _record_surface(*args, **kwargs):
        captured["surface"] = kwargs.get("surface")
        return original(*args, **kwargs)

    monkeypatch.setattr(urbansim_inputs_module, "build_binding_plan", _record_surface)

    settings = _settings_stub()
    workspace = _WorkspaceStub(tmp_path, tmp_path)
    state = _StateStub(start_year=True, land_use_enabled=True)

    base_h5 = tmp_path / "usim_000.h5"
    base_h5.write_text("base")

    build_urbansim_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        year=2018,
        surface=surface,
    )

    assert captured["surface"] is surface
