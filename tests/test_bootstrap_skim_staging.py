"""Unit tests for static input resolution and workspace preparation across UrbanSim and BEAM preprocessors."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.beam.preprocessor import BeamPreprocessor, ensure_beam_mutable_workspace
from pilates.urbansim.preprocessor import (
    resolve_urbansim_initial_datastore_source_path,
    selected_urbansim_static_input_sources,
)


def test_resolve_urbansim_initial_datastore_source_path_succeeds_without_skims(
    tmp_path: Path,
):
    """resolve_urbansim_initial_datastore_source_path resolves the HDF5 datastore without requiring skims.omx."""
    usim_data_dir = tmp_path / "urbansim_data"
    usim_data_dir.mkdir(parents=True)
    h5_file = usim_data_dir / "custom_mpo_06197001_model_data_2017.h5"
    h5_file.touch()

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            start_year=2017,
            end_year=2017,
            models=SimpleNamespace(
                land_use=None,
                travel="beam",
                activity_demand="activitysim",
                vehicle_ownership=None,
            ),
        ),
        shared=SimpleNamespace(
            skims=SimpleNamespace(fname="nonexistent_skims.omx"),
        ),
        urbansim=SimpleNamespace(
            region_id="06197001",
            local_data_input_folder=str(usim_data_dir),
            input_file_template="custom_mpo_{region_id}_model_data.h5",
            input_file_template_year="custom_mpo_{region_id}_model_data_{start_year}.h5",
            region_mappings={"region_to_region_id": {"sfbay": "06197001"}},
        ),
    )

    resolved = resolve_urbansim_initial_datastore_source_path(settings)
    assert resolved == h5_file


def test_urbansim_static_input_sources_raises_when_skims_missing(
    tmp_path: Path,
):
    """When skims are missing in beam_prod, selected_urbansim_static_input_sources raises RuntimeError."""
    prod_dir = tmp_path / "beam_prod" / "sfbay"
    prod_dir.mkdir(parents=True)
    usim_data_dir = tmp_path / "urbansim_data"
    usim_data_dir.mkdir(parents=True)
    for name in (
        "hsize_ct_06197001.csv",
        "income_rates_06197001.csv",
        "relmap_06197001.csv",
        "schools_2010.csv",
        "blocks_school_districts_2010.csv",
    ):
        (usim_data_dir / name).touch()
    zones_file = tmp_path / "zones.geojson"
    zones_file.write_text("{}", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            start_year=2017,
            end_year=2017,
            models=SimpleNamespace(
                land_use="urbansim",
                travel="beam",
                activity_demand="activitysim",
                vehicle_ownership=None,
            ),
        ),
        shared=SimpleNamespace(
            skims=SimpleNamespace(fname="nonexistent_skims.omx"),
            geography=SimpleNamespace(
                zones=SimpleNamespace(source_file=str(zones_file))
            ),
        ),
        urbansim=SimpleNamespace(
            region_id="06197001",
            local_data_input_folder=str(usim_data_dir),
            input_file_template="custom_mpo_{region_id}_model_data.h5",
            input_file_template_year="custom_mpo_{region_id}_model_data_{start_year}.h5",
            region_mappings={"region_to_region_id": {"sfbay": "06197001"}},
        ),
        beam=SimpleNamespace(
            local_input_folder=str(tmp_path / "beam_prod"),
            full_skim=SimpleNamespace(skims_kind="activitySimOmx"),
        ),
    )
    workspace = SimpleNamespace(
        full_path=str(tmp_path / "workspace"),
        get_asim_mutable_data_dir=lambda: str(
            tmp_path / "workspace" / "activitysim" / "data"
        ),
    )

    with pytest.raises(
        RuntimeError, match="Missing configured UrbanSim static input files"
    ):
        selected_urbansim_static_input_sources(settings, workspace)


def test_urbansim_static_input_sources_does_not_use_activitysim_skims_fallback(
    tmp_path: Path, monkeypatch
):
    """When skims are missing in beam_prod, selected_urbansim_static_input_sources fails fast without crawling activitysim/data."""
    prod_dir = tmp_path / "beam_prod" / "sfbay"
    prod_dir.mkdir(parents=True)
    usim_data_dir = tmp_path / "urbansim_data"
    usim_data_dir.mkdir(parents=True)
    for name in (
        "hsize_ct_06197001.csv",
        "income_rates_06197001.csv",
        "relmap_06197001.csv",
        "schools_2010.csv",
        "blocks_school_districts_2010.csv",
    ):
        (usim_data_dir / name).touch()

    mock_root = tmp_path / "mock_repo"
    asim_data_dir = mock_root / "pilates" / "activitysim" / "data" / "sfbay"
    asim_data_dir.mkdir(parents=True)
    fallback_skims = asim_data_dir / "skims.omx"
    fallback_skims.write_text("fallback_bytes")
    zones_file = tmp_path / "zones.geojson"
    zones_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "pilates.urbansim.preprocessor.find_project_root",
        lambda *args, **kwargs: str(mock_root),
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            start_year=2017,
            end_year=2017,
            models=SimpleNamespace(
                land_use="urbansim",
                travel="beam",
                activity_demand="activitysim",
                vehicle_ownership=None,
            ),
        ),
        shared=SimpleNamespace(
            skims=SimpleNamespace(fname="skims.omx"),
            geography=SimpleNamespace(
                zones=SimpleNamespace(source_file=str(zones_file))
            ),
        ),
        urbansim=SimpleNamespace(
            region_id="06197001",
            local_data_input_folder=str(usim_data_dir),
            input_file_template="custom_mpo_{region_id}_model_data.h5",
            input_file_template_year="custom_mpo_{region_id}_model_data_{start_year}.h5",
            region_mappings={"region_to_region_id": {"sfbay": "06197001"}},
        ),
        beam=SimpleNamespace(
            local_input_folder=str(tmp_path / "beam_prod"),
            full_skim=SimpleNamespace(skims_kind="activitySimOmx"),
        ),
    )
    workspace = SimpleNamespace(
        full_path=str(tmp_path / "workspace"),
        get_asim_mutable_data_dir=lambda: str(
            tmp_path / "workspace" / "activitysim" / "data"
        ),
    )

    with pytest.raises(
        RuntimeError, match="Missing configured UrbanSim static input files"
    ):
        selected_urbansim_static_input_sources(settings, workspace)


def test_resolve_urbansim_initial_datastore_source_path_falls_back_to_base_template_when_year_h5_missing(
    tmp_path: Path,
):
    """When year-specific HDF5 is absent, falls back to base template path."""
    usim_data_dir = tmp_path / "urbansim_data"
    usim_data_dir.mkdir(parents=True)

    settings = SimpleNamespace(
        run=SimpleNamespace(region="sfbay", start_year=2017),
        urbansim=SimpleNamespace(
            region_id="06197001",
            local_data_input_folder=str(usim_data_dir),
            input_file_template="custom_mpo_{region_id}_model_data.h5",
            input_file_template_year="custom_mpo_{region_id}_model_data_{start_year}.h5",
            region_mappings={"region_to_region_id": {"sfbay": "06197001"}},
        ),
    )

    resolved = resolve_urbansim_initial_datastore_source_path(settings)
    assert resolved == usim_data_dir / "custom_mpo_06197001_model_data.h5"
    assert not resolved.is_file()


def test_beam_find_production_path_raises_without_sources_fallback(
    tmp_path: Path, monkeypatch
):
    """BeamPreprocessor._find_beam_production_path fails fast and ignores sources/PILATES fallback."""
    mock_root = tmp_path / "mock_repo"
    alt_dir = (
        mock_root / "sources" / "PILATES" / "pilates" / "beam" / "production" / "sfbay"
    )
    alt_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "pilates.beam.preprocessor.find_project_root",
        lambda *args, **kwargs: str(mock_root),
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(region="sfbay"),
        beam=SimpleNamespace(local_input_folder="pilates/beam/production"),
    )

    with pytest.raises(
        FileNotFoundError,
        match="BEAM production input directory does not exist at configured path",
    ):
        BeamPreprocessor._find_beam_production_path(settings, "sfbay")


def test_beam_copy_data_ignores_activitysim_fallback_and_raises_on_missing_source_uri(
    tmp_path: Path, monkeypatch
):
    """ensure_beam_mutable_workspace does not crawl activitysim/data for skims and raises if explicit source_uri is missing."""
    prod_dir = tmp_path / "beam_prod" / "sfbay"
    prod_dir.mkdir(parents=True)
    (prod_dir / "sfbay.conf").write_text("beam.test = true")

    mock_root = tmp_path / "mock_repo"
    asim_dir = mock_root / "pilates" / "activitysim" / "data" / "sfbay"
    asim_dir.mkdir(parents=True)
    (asim_dir / "skims.omx").write_text("should_not_be_copied")

    monkeypatch.setattr(
        "pilates.beam.preprocessor.find_project_root",
        lambda *args, **kwargs: str(mock_root),
    )

    output_dir = tmp_path / "workspace" / "beam" / "input"
    settings_no_uri = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(travel="beam"),
        ),
        shared=SimpleNamespace(
            skims=SimpleNamespace(fname="skims.omx"),
        ),
        beam=SimpleNamespace(
            local_input_folder=str(tmp_path / "beam_prod"),
        ),
        activitysim=None,
    )

    ensure_beam_mutable_workspace(settings_no_uri, output_dir=str(output_dir))
    assert not (output_dir / "sfbay" / "skims.omx").exists()

    settings_missing_uri = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(travel="beam"),
        ),
        shared=SimpleNamespace(
            skims=SimpleNamespace(fname="skims.omx"),
        ),
        beam=SimpleNamespace(
            local_input_folder=str(tmp_path / "beam_prod"),
        ),
        activitysim=SimpleNamespace(
            admission=SimpleNamespace(
                initial_skims=SimpleNamespace(
                    mode="strict",
                    expectation=SimpleNamespace(
                        kind="declared_digest",
                        identity="sha256:file:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                        source_uri=str(tmp_path / "does_not_exist.omx"),
                    ),
                )
            ),
        ),
    )

    with pytest.raises(
        FileNotFoundError, match="Configured initial_skims source_uri does not exist"
    ):
        ensure_beam_mutable_workspace(settings_missing_uri, output_dir=str(output_dir))

    settings_opaque_uri = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(travel="beam"),
        ),
        shared=SimpleNamespace(
            skims=SimpleNamespace(fname="skims.omx"),
        ),
        beam=SimpleNamespace(
            local_input_folder=str(tmp_path / "beam_prod"),
        ),
        activitysim=SimpleNamespace(
            admission=SimpleNamespace(
                initial_skims=SimpleNamespace(
                    mode="strict",
                    expectation=SimpleNamespace(
                        kind="declared_digest",
                        identity="sha256:file:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                        source_uri="inputs://activitysim/skims.omx",
                    ),
                )
            ),
        ),
    )
    ensure_beam_mutable_workspace(settings_opaque_uri, output_dir=str(output_dir))
    assert not (output_dir / "sfbay" / "skims.omx").exists()


def test_beam_copy_data_stages_skims_from_activitysim_candidate(tmp_path: Path):
    """When beam production directory lacks skims.omx, fallback candidate is staged."""
    prod_dir = tmp_path / "beam_prod" / "sfbay"
    prod_dir.mkdir(parents=True)
    (prod_dir / "sfbay.conf").write_text("beam.test = true")

    asim_dir = tmp_path / "pilates" / "activitysim" / "data" / "sfbay"
    asim_dir.mkdir(parents=True)
    src_skims = asim_dir / "skims.omx"
    src_skims.write_text("mock_skims_bytes")
    src_sidecar = asim_dir / "skims.omx.provenance.json"
    src_sidecar.write_text('{"mock": "provenance"}')

    output_dir = tmp_path / "workspace" / "beam" / "input"

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(travel="beam"),
        ),
        shared=SimpleNamespace(
            skims=SimpleNamespace(fname="skims.omx"),
        ),
        beam=SimpleNamespace(
            local_input_folder=str(tmp_path / "beam_prod"),
        ),
        activitysim=SimpleNamespace(
            admission=SimpleNamespace(
                initial_skims=SimpleNamespace(
                    mode="strict",
                    expectation=SimpleNamespace(
                        kind="declared_digest",
                        identity="sha256:file:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                        source_uri=str(src_skims),
                    ),
                )
            ),
        ),
    )

    ensure_beam_mutable_workspace(settings, output_dir=str(output_dir))

    staged_skims = output_dir / "sfbay" / "skims.omx"
    staged_sidecar = output_dir / "sfbay" / "skims.omx.provenance.json"

    assert staged_skims.exists()
    assert staged_skims.read_text() == "mock_skims_bytes"
    assert staged_sidecar.exists()
    assert staged_sidecar.read_text() == '{"mock": "provenance"}'


def test_beam_copy_data_cleans_up_broken_symlink(tmp_path: Path):
    """If destination has a dangling symlink for skims.omx, it is replaced cleanly."""
    prod_dir = tmp_path / "beam_prod" / "sfbay"
    prod_dir.mkdir(parents=True)

    asim_skims = tmp_path / "skims_real.omx"
    asim_skims.write_text("fresh_skims")

    output_dir = tmp_path / "workspace" / "beam" / "input"
    dest_dir = output_dir / "sfbay"
    dest_dir.mkdir(parents=True)

    broken_link = dest_dir / "skims.omx"
    broken_link.symlink_to(tmp_path / "non_existent_file.omx")
    assert broken_link.is_symlink()
    assert not broken_link.exists()  # Broken

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            models=SimpleNamespace(travel="beam"),
        ),
        shared=SimpleNamespace(
            skims=SimpleNamespace(fname="skims.omx"),
        ),
        beam=SimpleNamespace(
            local_input_folder=str(tmp_path / "beam_prod"),
        ),
        activitysim=SimpleNamespace(
            admission=SimpleNamespace(
                initial_skims=SimpleNamespace(
                    mode="strict",
                    expectation=SimpleNamespace(
                        kind="declared_digest",
                        identity="sha256:file:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                        source_uri=str(asim_skims),
                    ),
                )
            ),
        ),
    )

    ensure_beam_mutable_workspace(settings, output_dir=str(output_dir))

    assert broken_link.exists()
    assert not broken_link.is_symlink()
    assert broken_link.read_text() == "fresh_skims"
