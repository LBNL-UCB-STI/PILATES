from pathlib import Path
from types import SimpleNamespace

from pilates.urbansim.preprocessor import (
    _restore_missing_mutable_urbansim_supporting_inputs,
)


class _WorkspaceStub:
    def __init__(self, full_path: Path, usim_dir: Path) -> None:
        self.full_path = str(full_path)
        self._usim_dir = usim_dir

    def get_usim_mutable_data_dir(self) -> str:
        return str(self._usim_dir)


def _settings(tmp_path: Path):
    return SimpleNamespace(
        run=SimpleNamespace(region="test"),
        beam=SimpleNamespace(local_input_folder=str(tmp_path / "beam-source")),
        shared=SimpleNamespace(skims=SimpleNamespace(fname="skims.omx")),
        urbansim=SimpleNamespace(
            local_data_input_folder=str(tmp_path / "usim-source"),
            region_mappings={"region_to_region_id": {"test": "000"}},
        ),
    )


def test_restore_missing_mutable_urbansim_supporting_inputs_prefers_archive_then_source(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    local_run_dir = tmp_path / "local-run"
    archive_run_dir = tmp_path / "archive-run"
    usim_dir = local_run_dir / "urbansim" / "data"
    workspace = _WorkspaceStub(local_run_dir, usim_dir)
    state = SimpleNamespace(run_info_path=str(archive_run_dir / "run_state.yaml"))

    usim_dir.mkdir(parents=True, exist_ok=True)
    Path(state.run_info_path).parent.mkdir(parents=True, exist_ok=True)
    Path(state.run_info_path).write_text("year: 2023\n", encoding="utf-8")

    source_dir = tmp_path / "usim-source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "beam-source" / "test").mkdir(parents=True, exist_ok=True)

    archive_hh = archive_run_dir / "urbansim" / "data" / "hsize_ct_000.csv"
    archive_hh.parent.mkdir(parents=True, exist_ok=True)
    archive_hh.write_text("archive-hh", encoding="utf-8")

    (source_dir / "income_rates_000.csv").write_text("source-income", encoding="utf-8")
    (source_dir / "relmap_000.csv").write_text("source-relmap", encoding="utf-8")
    (source_dir / "schools_2010.csv").write_text("source-schools", encoding="utf-8")
    (source_dir / "blocks_school_districts_2010.csv").write_text(
        "source-districts",
        encoding="utf-8",
    )
    (tmp_path / "beam-source" / "test" / "skims.omx").write_text(
        "source-skims",
        encoding="utf-8",
    )

    existing_local = usim_dir / "relmap_000.csv"
    existing_local.write_text("existing-local", encoding="utf-8")

    restored = _restore_missing_mutable_urbansim_supporting_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
    )

    assert (usim_dir / "hsize_ct_000.csv").read_text(encoding="utf-8") == "archive-hh"
    assert (usim_dir / "income_rates_000.csv").read_text(
        encoding="utf-8"
    ) == "source-income"
    assert existing_local.read_text(encoding="utf-8") == "existing-local"
    assert (usim_dir / "schools_2010.csv").read_text(
        encoding="utf-8"
    ) == "source-schools"
    assert (usim_dir / "blocks_school_districts_2010.csv").read_text(
        encoding="utf-8"
    ) == "source-districts"
    assert (usim_dir / "skims_mpo_000.omx").read_text(
        encoding="utf-8"
    ) == "source-skims"
    assert set(restored) == {
        "omx_skims",
        "hh_size",
        "income_rates",
        "schools",
        "school_districts",
    }


def test_restore_missing_mutable_urbansim_supporting_inputs_uses_omx_export_name_for_zarr_mode(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.shared.skims.fname = "skims.zarr"
    local_run_dir = tmp_path / "local-run"
    usim_dir = local_run_dir / "urbansim" / "data"
    workspace = _WorkspaceStub(local_run_dir, usim_dir)
    state = SimpleNamespace(
        run_info_path=str(tmp_path / "archive-run" / "run_state.yaml")
    )

    usim_dir.mkdir(parents=True, exist_ok=True)
    Path(state.run_info_path).parent.mkdir(parents=True, exist_ok=True)
    Path(state.run_info_path).write_text("year: 2023\n", encoding="utf-8")

    source_dir = tmp_path / "usim-source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "hsize_ct_000.csv").write_text("hh", encoding="utf-8")
    (source_dir / "income_rates_000.csv").write_text("income", encoding="utf-8")
    (source_dir / "relmap_000.csv").write_text("relmap", encoding="utf-8")
    (source_dir / "schools_2010.csv").write_text("schools", encoding="utf-8")
    (source_dir / "blocks_school_districts_2010.csv").write_text(
        "districts",
        encoding="utf-8",
    )

    beam_region_dir = tmp_path / "beam-source" / "test"
    beam_region_dir.mkdir(parents=True, exist_ok=True)
    (beam_region_dir / "skims.omx").write_text("omx-export", encoding="utf-8")
    (beam_region_dir / "skims.zarr").mkdir(parents=True, exist_ok=True)

    restored = _restore_missing_mutable_urbansim_supporting_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
    )

    assert (usim_dir / "skims_mpo_000.omx").read_text(encoding="utf-8") == "omx-export"
    assert restored["omx_skims"] == usim_dir / "skims_mpo_000.omx"
