from __future__ import annotations

from pathlib import Path

from pilates.utils.zone_utils import copy_canonical_zone_source_to_dir


def test_copy_canonical_zone_source_to_dir_is_noop_for_same_geojson_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "taz_sfbay.geojson"
    source.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    copied = copy_canonical_zone_source_to_dir(str(source), str(tmp_path))

    assert copied == str(source)
    assert source.read_text(encoding="utf-8") == '{"type":"FeatureCollection","features":[]}'
