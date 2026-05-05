from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_generate_schema_erd_script_writes_expected_relationships(tmp_path: Path):
    output_path = tmp_path / "pilates_schema_erd.mmd"
    cmd = [
        sys.executable,
        "pilates/database/scripts/generate_schema_erd.py",
        "--output",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")
    assert content.startswith("erDiagram")
    assert "PersonsAsimIn }o--|| HouseholdsAsimIn" in content
    assert "HouseholdsAsimIn }o--|| LandUseAsimIn" in content
    assert "BeamPlansAsimOut }o--|| tripsAsimOut" in content
    assert (
        "UrbansimPostprocessUsimPersonsTable }o--|| "
        "UrbansimPostprocessUsimHouseholdsTable" in content
    )
    assert (
        "UrbansimPostprocessUsimWorkLocationsTable }o--|| "
        "UrbansimPostprocessUsimPersonsTable" in content
    )
    assert (
        "UrbansimPostprocessUsimBlocksTable }o--|| "
        "UrbansimPostprocessUsimTazZoneGeomsTable" in content
    )


def test_generate_schema_erd_script_dot_format(tmp_path: Path):
    output_path = tmp_path / "pilates_schema_erd.dot"
    cmd = [
        sys.executable,
        "pilates/database/scripts/generate_schema_erd.py",
        "--format",
        "dot",
        "--output",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")
    assert content.startswith("digraph erd")
    assert '"PersonsAsimIn" -> "HouseholdsAsimIn"' in content


def test_generate_schema_erd_script_html_format(tmp_path: Path):
    output_path = tmp_path / "pilates_schema_erd.html"
    cmd = [
        sys.executable,
        "pilates/database/scripts/generate_schema_erd.py",
        "--format",
        "html",
        "--output",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in content
    assert "cytoscape.min.js" in content
    assert 'id="cy"' in content
    assert "PersonsAsimIn" in content
    assert "tripsAsimOut" in content
