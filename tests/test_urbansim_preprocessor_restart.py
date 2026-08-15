from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.urbansim.preprocessor import (
    _restore_missing_mutable_urbansim_supporting_inputs,
    stage_urbansim_run_workspace,
)
from pilates.urbansim.outputs import UrbanSimPreprocessOutputs, UrbanSimRunOutputs
from pilates.urbansim.runner import UrbanSimLaunchContext, UrbansimRunner
from pilates.workflows.input_authority import requires_prior_beam_skim_handoff


class _WorkspaceStub:
    def __init__(
        self,
        full_path: Path,
        usim_dir: Path,
        beam_dir: Path | None = None,
    ) -> None:
        self.full_path = str(full_path)
        self._usim_dir = usim_dir
        self._beam_dir = beam_dir or full_path / "beam" / "input"

    def get_usim_mutable_data_dir(self) -> str:
        return str(self._usim_dir)

    def get_beam_mutable_data_dir(self) -> str:
        return str(self._beam_dir)


def _settings(tmp_path: Path):
    return SimpleNamespace(
        run=SimpleNamespace(
            region="test",
            start_year=2020,
            models=SimpleNamespace(travel="beam"),
        ),
        beam=SimpleNamespace(local_input_folder=str(tmp_path / "beam-source")),
        shared=SimpleNamespace(
            skims=SimpleNamespace(fname="skims.omx"),
            geography=SimpleNamespace(
                zones=SimpleNamespace(
                    zone_type="taz",
                    source_file=str(tmp_path / "geography" / "zones.geojson"),
                    canonical_id_col="zone_id",
                    activitysim_index_col="TAZ",
                    source_crs="EPSG:4326",
                ),
                alternative_zones=None,
            ),
        ),
        activitysim=None,
        urbansim=SimpleNamespace(
            local_data_input_folder=str(tmp_path / "usim-source"),
            region_mappings={"region_to_region_id": {"test": "000"}},
            input_file_template="model_data_{region_id}.h5",
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


def _stage_sources(tmp_path: Path) -> tuple[Path, Path]:
    source_dir = tmp_path / "usim-source"
    source_dir.mkdir(parents=True)
    for filename, contents in {
        "hsize_ct_000.csv": "households",
        "income_rates_000.csv": "income",
        "relmap_000.csv": "relmap",
        "schools_2010.csv": "schools",
        "blocks_school_districts_2010.csv": "districts",
    }.items():
        (source_dir / filename).write_text(contents, encoding="utf-8")

    beam_skims = tmp_path / "beam-source" / "test" / "skims.omx"
    beam_skims.parent.mkdir(parents=True)
    beam_skims.write_text("base-skims", encoding="utf-8")

    datastore = tmp_path / "bound" / "datastore.h5"
    datastore.parent.mkdir()
    datastore.write_bytes(b"bound datastore")
    canonical_zones = tmp_path / "geography" / "zones.geojson"
    canonical_zones.parent.mkdir()
    canonical_zones.write_text("canonical zones", encoding="utf-8")
    return datastore, beam_skims


def test_stage_urbansim_run_workspace_writes_conventional_mutable_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    datastore, _ = _stage_sources(tmp_path)
    usim_dir = tmp_path / "run" / "urbansim" / "data"
    workspace = _WorkspaceStub(tmp_path / "run", usim_dir)
    state = SimpleNamespace(
        current_year=2020,
        start_year=2020,
        iteration=0,
        run_info_path=None,
    )
    monkeypatch.setattr(
        "pilates.utils.zone_utils.get_block_to_zone_mapping",
        lambda *_args, **_kwargs: {"000000000000001": "1"},
    )

    outputs = stage_urbansim_run_workspace(
        settings=settings,
        state=state,
        workspace=workspace,
        usim_datastore_h5=datastore,
        final_skims_omx=None,
    )

    assert (usim_dir / "model_data_000.h5").read_bytes() == b"bound datastore"
    assert (usim_dir / "skims_mpo_000.omx").read_text(encoding="utf-8") == "base-skims"
    for filename in (
        "hsize_ct_000.csv",
        "income_rates_000.csv",
        "relmap_000.csv",
        "schools_2010.csv",
        "blocks_school_districts_2010.csv",
    ):
        assert (usim_dir / filename).exists()
    assert (usim_dir / "geoid_to_zone.csv").read_text(encoding="utf-8") == (
        "GEOID,zone_id\n000000000000001,1\n"
    )
    assert (
        outputs.prepared_inputs["usim_datastore_h5"] == usim_dir / "model_data_000.h5"
    )


def test_stage_urbansim_run_workspace_ignores_archived_static_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    datastore, _ = _stage_sources(tmp_path)
    run_dir = tmp_path / "run"
    archive_dir = tmp_path / "archive"
    usim_dir = run_dir / "urbansim" / "data"
    workspace = _WorkspaceStub(run_dir, usim_dir)
    run_info_path = archive_dir / "run_state.yaml"
    run_info_path.parent.mkdir(parents=True)
    run_info_path.write_text("year: 2020\n", encoding="utf-8")
    archived_hh_size = archive_dir / "urbansim" / "data" / "hsize_ct_000.csv"
    archived_hh_size.parent.mkdir(parents=True)
    archived_hh_size.write_text("archived households", encoding="utf-8")
    state = SimpleNamespace(
        current_year=2020,
        start_year=2020,
        iteration=0,
        run_info_path=str(run_info_path),
    )
    monkeypatch.setattr(
        "pilates.utils.zone_utils.get_block_to_zone_mapping",
        lambda *_args, **_kwargs: {"000000000000001": "1"},
    )

    stage_urbansim_run_workspace(
        settings=settings,
        state=state,
        workspace=workspace,
        usim_datastore_h5=datastore,
        final_skims_omx=None,
    )

    assert (usim_dir / "hsize_ct_000.csv").read_text(encoding="utf-8") == ("households")


def test_stage_urbansim_run_workspace_replaces_base_skims_after_beam_iteration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    datastore, _ = _stage_sources(tmp_path)
    usim_dir = tmp_path / "run" / "urbansim" / "data"
    workspace = _WorkspaceStub(tmp_path / "run", usim_dir)
    final_skims = tmp_path / "beam-final" / "final_skims.omx"
    final_skims.parent.mkdir()
    final_skims.write_text("final-skims", encoding="utf-8")
    state = SimpleNamespace(
        current_year=2021,
        start_year=2020,
        iteration=0,
        run_info_path=None,
    )
    monkeypatch.setattr(
        "pilates.utils.zone_utils.get_block_to_zone_mapping",
        lambda *_args, **_kwargs: {"000000000000001": "1"},
    )

    outputs = stage_urbansim_run_workspace(
        settings=settings,
        state=state,
        workspace=workspace,
        usim_datastore_h5=datastore,
        final_skims_omx=final_skims,
    )

    assert (usim_dir / "skims_mpo_000.omx").read_text(encoding="utf-8") == "final-skims"
    assert outputs.prepared_inputs["usim_skims_input_updated"] == (
        usim_dir / "skims_mpo_000.omx"
    )


def test_stage_urbansim_run_workspace_accepts_optional_explicit_skims_at_bootstrap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    datastore, _ = _stage_sources(tmp_path)
    usim_dir = tmp_path / "run" / "urbansim" / "data"
    workspace = _WorkspaceStub(tmp_path / "run", usim_dir)
    optional_skims = tmp_path / "resolved" / "optional_skims.omx"
    optional_skims.parent.mkdir()
    optional_skims.write_text("resolved-optional-skims", encoding="utf-8")
    state = SimpleNamespace(
        current_year=2020,
        start_year=2020,
        iteration=0,
        run_info_path=None,
    )
    monkeypatch.setattr(
        "pilates.utils.zone_utils.get_block_to_zone_mapping",
        lambda *_args, **_kwargs: {"000000000000001": "1"},
    )

    assert not requires_prior_beam_skim_handoff(settings=settings, state=state)
    stage_urbansim_run_workspace(
        settings=settings,
        state=state,
        workspace=workspace,
        usim_datastore_h5=datastore,
        final_skims_omx=optional_skims,
    )

    assert (usim_dir / "skims_mpo_000.omx").read_text(encoding="utf-8") == (
        "resolved-optional-skims"
    )


def test_stage_urbansim_run_workspace_rejects_missing_bound_datastore(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    usim_dir = tmp_path / "run" / "urbansim" / "data"
    workspace = _WorkspaceStub(tmp_path / "run", usim_dir)
    state = SimpleNamespace(
        current_year=2020,
        start_year=2020,
        iteration=0,
        run_info_path=None,
    )

    with pytest.raises(RuntimeError, match="bound datastore H5 to exist"):
        stage_urbansim_run_workspace(
            settings=settings,
            state=state,
            workspace=workspace,
            usim_datastore_h5=tmp_path / "missing.h5",
            final_skims_omx=None,
        )


def test_stage_urbansim_run_workspace_ignores_unbound_current_workspace_skims(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    datastore, _ = _stage_sources(tmp_path)
    usim_dir = tmp_path / "run" / "urbansim" / "data"
    beam_dir = tmp_path / "run" / "beam" / "input"
    workspace = _WorkspaceStub(tmp_path / "run", usim_dir, beam_dir)
    poison_skims = beam_dir / "test" / "skims.omx"
    poison_skims.parent.mkdir(parents=True)
    poison_skims.write_text("unbound-current-workspace-skims", encoding="utf-8")
    state = SimpleNamespace(
        current_year=2022,
        atlas_interval_start_year=2020,
        start_year=2020,
        iteration=0,
        run_info_path=None,
    )
    monkeypatch.setattr(
        "pilates.utils.zone_utils.get_block_to_zone_mapping",
        lambda *_args, **_kwargs: {"000000000000001": "1"},
    )

    assert not requires_prior_beam_skim_handoff(settings=settings, state=state)
    outputs = stage_urbansim_run_workspace(
        settings=settings,
        state=state,
        workspace=workspace,
        usim_datastore_h5=datastore,
        final_skims_omx=None,
    )

    assert (usim_dir / "skims_mpo_000.omx").read_text(encoding="utf-8") == (
        "base-skims"
    )
    assert "usim_skims_input_updated" not in outputs.prepared_inputs


def test_stage_urbansim_run_workspace_ignores_unbound_restart_workspace_skims(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    datastore, _ = _stage_sources(tmp_path)
    usim_dir = tmp_path / "run" / "urbansim" / "data"
    workspace = _WorkspaceStub(tmp_path / "run", usim_dir)
    restart_dir = tmp_path / "restart"
    run_info_path = restart_dir / "run_state.yaml"
    run_info_path.parent.mkdir()
    run_info_path.write_text("year: 2020\n", encoding="utf-8")
    poison_skims = restart_dir / "beam" / "input" / "test" / "skims.omx"
    poison_skims.parent.mkdir(parents=True)
    poison_skims.write_text("unbound-restart-workspace-skims", encoding="utf-8")
    state = SimpleNamespace(
        current_year=2022,
        atlas_interval_start_year=2020,
        start_year=2020,
        iteration=0,
        run_info_path=str(run_info_path),
    )
    monkeypatch.setattr(
        "pilates.utils.zone_utils.get_block_to_zone_mapping",
        lambda *_args, **_kwargs: {"000000000000001": "1"},
    )

    assert not requires_prior_beam_skim_handoff(settings=settings, state=state)
    stage_urbansim_run_workspace(
        settings=settings,
        state=state,
        workspace=workspace,
        usim_datastore_h5=datastore,
        final_skims_omx=None,
    )

    assert (usim_dir / "skims_mpo_000.omx").read_text(encoding="utf-8") == (
        "base-skims"
    )


@pytest.mark.parametrize("final_skims_name", [None, "missing-final-skims.omx"])
def test_stage_urbansim_run_workspace_requires_final_skims_after_beam_iteration(
    tmp_path: Path,
    monkeypatch,
    final_skims_name: str | None,
) -> None:
    settings = _settings(tmp_path)
    datastore, _ = _stage_sources(tmp_path)
    usim_dir = tmp_path / "run" / "urbansim" / "data"
    workspace = _WorkspaceStub(tmp_path / "run", usim_dir)
    state = SimpleNamespace(
        current_year=2021,
        start_year=2020,
        iteration=0,
        run_info_path=None,
    )
    monkeypatch.setattr(
        "pilates.utils.zone_utils.get_block_to_zone_mapping",
        lambda *_args, **_kwargs: {"000000000000001": "1"},
    )

    with pytest.raises(
        FileNotFoundError, match="requires the resolved final_skims_omx"
    ):
        stage_urbansim_run_workspace(
            settings=settings,
            state=state,
            workspace=workspace,
            usim_datastore_h5=datastore,
            final_skims_omx=(
                None if final_skims_name is None else tmp_path / final_skims_name
            ),
        )


def test_urbansim_runner_stages_scalar_inputs_before_running(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = _WorkspaceStub(tmp_path / "run", tmp_path / "run" / "urbansim" / "data")
    datastore = tmp_path / "bound" / "datastore.h5"
    progress: list[str] = []
    runner = object.__new__(UrbansimRunner)
    runner.state = SimpleNamespace(
        set_sub_stage_progress=progress.append,
    )
    staged = UrbanSimPreprocessOutputs(
        usim_mutable_data_dir=Path(workspace.get_usim_mutable_data_dir()),
        prepared_inputs={"usim_datastore_h5": datastore},
    )
    launch_context = UrbanSimLaunchContext(
        mutable_data_dir=Path(workspace.get_usim_mutable_data_dir()),
        output_datastore=tmp_path / "run" / "urbansim" / "data" / "output.h5",
    )
    captured: dict[str, object] = {}

    def _run(
        _self: UrbansimRunner,
        inputs: UrbanSimPreprocessOutputs,
        run_launch_context: UrbanSimLaunchContext,
        model_run_hash: str | None = None,
    ) -> UrbanSimRunOutputs:
        captured["run"] = (inputs, run_launch_context, model_run_hash)
        return UrbanSimRunOutputs(usim_datastore_h5=tmp_path / "output.h5")

    monkeypatch.setattr(UrbansimRunner, "_run", _run)

    result = runner.run(staged, launch_context, model_run_hash="run-hash")

    assert progress == ["runner"]
    assert captured["run"] == (staged, launch_context, "run-hash")
    assert result.usim_datastore_h5 == tmp_path / "output.h5"


def test_urbansim_runner_uses_resolved_launch_context_for_container_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    settings.urbansim.client_data_folder = "/data"
    settings.urbansim.client_base_folder = "/"
    runner = object.__new__(UrbansimRunner)
    runner.model_name = "urbansim"
    runner.state = SimpleNamespace(current_year=2020, full_settings=settings)
    staged_root = tmp_path / "staged" / "urbansim" / "data"
    launch_context = UrbanSimLaunchContext(
        mutable_data_dir=tmp_path / "resolved" / "urbansim" / "data",
        output_datastore=tmp_path / "resolved" / "urbansim" / "data" / "output.h5",
    )
    staged = UrbanSimPreprocessOutputs(
        usim_mutable_data_dir=staged_root,
        prepared_inputs={"usim_datastore_h5": tmp_path / "bound" / "datastore.h5"},
    )
    calls: dict[str, object] = {}

    def run_container(*_args: object, **kwargs: object) -> bool:
        calls.update(kwargs)
        launch_context.output_datastore.parent.mkdir(parents=True)
        launch_context.output_datastore.write_bytes(b"output")
        return True

    monkeypatch.setattr(
        UrbansimRunner,
        "get_model_and_image",
        staticmethod(lambda *_args: ("urbansim", "image")),
    )
    monkeypatch.setattr(UrbansimRunner, "get_usim_cmd", lambda _self: "run")
    monkeypatch.setattr(UrbansimRunner, "run_container", run_container)

    result = runner._run(staged, launch_context)

    assert calls["volumes"] == {
        str(launch_context.mutable_data_dir): {"bind": "/data", "mode": "rw"}
    }
    assert result.usim_datastore_h5 == launch_context.output_datastore
