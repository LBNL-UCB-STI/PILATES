from pathlib import Path

import pytest

from pilates.atlas.preprocessor import AtlasPreprocessor
import pilates.atlas.preprocessor as atlas_preprocessor_module


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_copy_data_to_mutable_location_uses_fallback_source_for_missing_required_files(
    tmp_path, monkeypatch
):
    primary_source = tmp_path / "primary_atlas_input"
    project_root = tmp_path / "project_root"
    fallback_source = project_root / "pilates" / "atlas" / "atlas_input"
    output_dir = tmp_path / "output"

    required_relpaths = (
        "psid_names.Rdat",
        "adopt/zev_mandate/new_vehicles_biannual_values_2021.csv",
    )
    _touch(fallback_source / required_relpaths[0])
    _touch(fallback_source / required_relpaths[1], content="Year,value\n2021,1\n")

    # Primary source intentionally does not contain required files.
    primary_source.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        atlas_preprocessor_module,
        "atlas_static_input_relpaths",
        lambda _settings: required_relpaths,
    )
    monkeypatch.setattr(
        atlas_preprocessor_module,
        "find_project_root",
        lambda start_path=None: str(project_root),
    )

    preprocessor = AtlasPreprocessor.__new__(AtlasPreprocessor)
    settings = {
        "atlas": {
            "host_input_folder": str(primary_source),
            "scenario": "zev_mandate",
            "adscen": "zev_mandate",
        }
    }

    _inputs, outputs = preprocessor.copy_data_to_mutable_location(
        settings=settings,
        output_dir=str(output_dir),
    )

    output_paths = {
        Path(record.file_path).relative_to(output_dir).as_posix()
        for record in outputs.all_records()
    }
    assert set(required_relpaths) <= output_paths

    inputs_by_key = {record.short_name: record for record in _inputs.all_records()}
    psid_record = inputs_by_key["psid_names"]
    assert psid_record.metadata["atlas_static_input"] is True
    assert psid_record.metadata["atlas_relpath"] == "psid_names.Rdat"
    assert psid_record.metadata["atlas_source_origin"] == "fallback"
    assert psid_record.metadata["atlas_input_group"] == "global"

    adopt_record = inputs_by_key["adopt/zev_mandate/new_vehicles_biannual_values_2021"]
    assert adopt_record.metadata["atlas_static_input"] is True
    assert adopt_record.metadata["atlas_source_origin"] == "fallback"
    assert adopt_record.metadata["atlas_input_group"] == "adopt"
    assert adopt_record.metadata["atlas_scenario"] == "zev_mandate"
    assert adopt_record.metadata["atlas_input_year"] == 2021
    assert adopt_record.metadata["profile_file_schema"] is True


def test_copy_data_to_mutable_location_raises_when_required_static_file_missing(
    tmp_path, monkeypatch
):
    primary_source = tmp_path / "primary_atlas_input"
    project_root = tmp_path / "project_root"
    fallback_source = project_root / "pilates" / "atlas" / "atlas_input"
    output_dir = tmp_path / "output"

    required_relpaths = ("adopt/zev_mandate/new_vehicles_biannual_values_2021.csv",)
    primary_source.mkdir(parents=True, exist_ok=True)
    fallback_source.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        atlas_preprocessor_module,
        "atlas_static_input_relpaths",
        lambda _settings: required_relpaths,
    )
    monkeypatch.setattr(
        atlas_preprocessor_module,
        "find_project_root",
        lambda start_path=None: str(project_root),
    )

    preprocessor = AtlasPreprocessor.__new__(AtlasPreprocessor)
    settings = {
        "atlas": {
            "host_input_folder": str(primary_source),
            "scenario": "zev_mandate",
            "adscen": "zev_mandate",
        }
    }

    with pytest.raises(RuntimeError, match="Missing required ATLAS static input files"):
        preprocessor.copy_data_to_mutable_location(
            settings=settings,
            output_dir=str(output_dir),
        )
