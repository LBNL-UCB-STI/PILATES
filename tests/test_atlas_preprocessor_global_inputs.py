from pathlib import Path

from pilates.atlas.preprocessor import _discover_global_atlas_input_files


def test_discover_global_atlas_input_files_includes_rdat_and_rdata(tmp_path):
    (tmp_path / "cpi.csv").write_text("Year,CPI\n2017,1.0\n", encoding="utf-8")
    (tmp_path / "accessbility_2015.RData").write_text("x", encoding="utf-8")
    (tmp_path / "psid_names.Rdat").write_text("x", encoding="utf-8")
    (tmp_path / "psid_names_lower.rdat").write_text("x", encoding="utf-8")
    (tmp_path / "README.txt").write_text("ignore", encoding="utf-8")

    found = _discover_global_atlas_input_files(str(tmp_path))
    found_names = {Path(path).name for path, _label in found}

    assert "cpi.csv" in found_names
    assert "accessbility_2015.RData" in found_names
    assert "psid_names.Rdat" in found_names
    assert "psid_names_lower.rdat" in found_names
    assert "README.txt" not in found_names

