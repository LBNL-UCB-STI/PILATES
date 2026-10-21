"""Unit tests for bootstrap skim staging and resilience across UrbanSim and BEAM preprocessors."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pilates.beam.preprocessor import BeamPreprocessor
from pilates.urbansim.preprocessor import UrbansimPreprocessor


def _make_dummy_state():
    state = MagicMock()
    state.full_settings = None
    return state


def test_urbansim_copy_data_skips_missing_skims_when_land_use_disabled(tmp_path: Path):
    """When land_use is disabled, missing BEAM production skims should not cause an error."""
    prod_dir = tmp_path / "beam_prod" / "sfbay"
    prod_dir.mkdir(parents=True)
    usim_data_dir = tmp_path / "urbansim_data"
    usim_data_dir.mkdir(parents=True)
    h5_file = usim_data_dir / "custom_mpo_06197001_model_data_2017.h5"
    h5_file.touch()

    output_dir = tmp_path / "workspace" / "urbansim" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        beam=SimpleNamespace(
            local_input_folder=str(tmp_path / "beam_prod"),
        ),
    )

    preprocessor = UrbansimPreprocessor(
        model_name="urbansim", state=_make_dummy_state()
    )
    in_recs, out_recs = preprocessor.copy_data_to_mutable_location(
        settings, str(output_dir)
    )

    # Should succeed, copy the h5 file, but skip skims.omx
    assert (output_dir / "custom_mpo_06197001_model_data.h5").exists()
    assert not (output_dir / "skims_mpo_06197001.omx").exists()


def test_urbansim_copy_data_raises_when_land_use_enabled_and_skims_missing(
    tmp_path: Path,
):
    """When land_use is enabled, missing skims everywhere must raise FileNotFoundError."""
    prod_dir = tmp_path / "beam_prod" / "sfbay"
    prod_dir.mkdir(parents=True)
    usim_data_dir = tmp_path / "urbansim_data"
    usim_data_dir.mkdir(parents=True)
    h5_file = usim_data_dir / "custom_mpo_06197001_model_data_2017.h5"
    h5_file.touch()

    output_dir = tmp_path / "workspace" / "urbansim" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        ),
    )

    preprocessor = UrbansimPreprocessor(
        model_name="urbansim", state=_make_dummy_state()
    )
    with pytest.raises(FileNotFoundError, match="UrbanSim skims file not found"):
        preprocessor.copy_data_to_mutable_location(settings, str(output_dir))


def test_urbansim_copy_data_uses_activitysim_skims_fallback(
    tmp_path: Path, monkeypatch
):
    """When skims are missing in beam_prod, UrbansimPreprocessor falls back to activitysim/data."""
    prod_dir = tmp_path / "beam_prod" / "sfbay"
    prod_dir.mkdir(parents=True)
    usim_data_dir = tmp_path / "urbansim_data"
    usim_data_dir.mkdir(parents=True)
    h5_file = usim_data_dir / "custom_mpo_06197001_model_data_2017.h5"
    h5_file.touch()

    # Create fallback skims in mock project root
    mock_root = tmp_path / "mock_repo"
    asim_data_dir = mock_root / "pilates" / "activitysim" / "data" / "sfbay"
    asim_data_dir.mkdir(parents=True)
    fallback_skims = asim_data_dir / "skims.omx"
    fallback_skims.write_text("fallback_bytes")

    monkeypatch.setattr(
        "pilates.urbansim.preprocessor.find_project_root",
        lambda *args, **kwargs: str(mock_root),
    )

    output_dir = tmp_path / "workspace" / "urbansim" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        ),
    )

    preprocessor = UrbansimPreprocessor(
        model_name="urbansim", state=_make_dummy_state()
    )
    in_recs, out_recs = preprocessor.copy_data_to_mutable_location(
        settings, str(output_dir)
    )

    target_skims = output_dir / "skims_mpo_06197001.omx"
    assert target_skims.exists()
    assert target_skims.read_text() == "fallback_bytes"


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

    preprocessor = BeamPreprocessor(model_name="beam", state=_make_dummy_state())
    preprocessor.copy_data_to_mutable_location(settings, str(output_dir))

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

    preprocessor = BeamPreprocessor(model_name="beam", state=_make_dummy_state())
    preprocessor.copy_data_to_mutable_location(settings, str(output_dir))

    assert broken_link.exists()
    assert not broken_link.is_symlink()
    assert broken_link.read_text() == "fresh_skims"
