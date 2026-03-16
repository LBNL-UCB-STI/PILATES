<<<<<<< HEAD
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
=======
from pathlib import Path
from types import SimpleNamespace

from pilates.utils.zone_utils import (
    copy_canonical_zone_source_to_dir,
    resolve_canonical_zone_source,
    resolve_canonical_zone_source_path,
)


class _WorkspaceStub:
    def __init__(self, asim_dir: Path):
        self._asim_dir = str(asim_dir)

    def get_asim_mutable_data_dir(self):
        return self._asim_dir


def test_resolve_canonical_zone_source_uses_fallback_when_primary_missing(tmp_path):
    fallback_path = tmp_path / "fallback.geojson"
    fallback_path.write_text("{}", encoding="utf-8")
    settings = SimpleNamespace(
        shared=SimpleNamespace(
            geography=SimpleNamespace(
                zones=SimpleNamespace(
                    source_file=str(tmp_path / "missing.geojson"),
                    canonical_id_col="TAZ",
                    activitysim_index_col="TAZ",
                ),
                alternative_zones=SimpleNamespace(
                    zone_type="taz",
                    source_file=str(fallback_path),
                    canonical_id_col="objectid",
                    activitysim_index_col="TAZ",
                    source_crs="EPSG:26910",
                ),
            )
        ),
        activitysim=None,
    )

    resolved = resolve_canonical_zone_source_path(settings)

    assert resolved == str(fallback_path)


def test_resolve_canonical_zone_source_prefers_mutable_activitysim_copy(tmp_path):
    source_path = tmp_path / "zones.geojson"
    source_path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    asim_dir = tmp_path / "asim"
    asim_dir.mkdir()
    mutable_copy = asim_dir / "zones.geojson"
    mutable_copy.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    settings = SimpleNamespace(
        shared=SimpleNamespace(
            geography=SimpleNamespace(
                zones=SimpleNamespace(
                    source_file=str(source_path),
                    canonical_id_col="TAZ",
                    activitysim_index_col="TAZ",
                )
            )
        ),
        activitysim=SimpleNamespace(),
    )

    resolved = resolve_canonical_zone_source_path(
        settings, _WorkspaceStub(asim_dir)
    )

    assert resolved == str(mutable_copy)


def test_resolve_canonical_zone_source_returns_alternative_metadata(tmp_path):
    fallback_path = tmp_path / "fallback.geojson"
    fallback_path.write_text("{}", encoding="utf-8")
    settings = SimpleNamespace(
        shared=SimpleNamespace(
            geography=SimpleNamespace(
                zones=SimpleNamespace(
                    zone_type="taz",
                    source_file=str(tmp_path / "missing.geojson"),
                    canonical_id_col="primary_id",
                    activitysim_index_col="PRIMARY",
                    source_crs=None,
                ),
                alternative_zones=SimpleNamespace(
                    zone_type="taz",
                    source_file=str(fallback_path),
                    canonical_id_col="alt_id",
                    activitysim_index_col="ALT",
                    source_crs="EPSG:26910",
                ),
            )
        ),
        activitysim=None,
    )

    resolved_path, source_config = resolve_canonical_zone_source(settings)

    assert resolved_path == str(fallback_path)
    assert source_config["canonical_id_col"] == "alt_id"
    assert source_config["activitysim_index_col"] == "ALT"
    assert source_config["source_crs"] == "EPSG:26910"


def test_copy_canonical_zone_source_to_dir_copies_shapefile_sidecars(tmp_path):
    shape_dir = tmp_path / "shape"
    shape_dir.mkdir()
    for suffix in (".shp", ".dbf", ".shx", ".prj", ".cpg"):
        (shape_dir / f"zones{suffix}").write_text(suffix, encoding="utf-8")

    dest_dir = tmp_path / "dest"
    output = copy_canonical_zone_source_to_dir(str(shape_dir / "zones.shp"), str(dest_dir))

    assert output == str(dest_dir / "zones.shp")
    for suffix in (".shp", ".dbf", ".shx", ".prj", ".cpg"):
        assert (dest_dir / f"zones{suffix}").exists()
>>>>>>> 118eca6 (keep temporarily old settings and changes to few tests)
