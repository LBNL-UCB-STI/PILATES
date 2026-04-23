from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.activitysim import preprocessor as activitysim_preprocessor
from pilates.utils.zone_utils import (
    copy_canonical_zone_source_to_dir,
    resolve_canonical_zone_source,
)


def test_copy_canonical_zone_source_to_dir_is_noop_for_same_geojson_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "taz_sfbay.geojson"
    source.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    copied = copy_canonical_zone_source_to_dir(str(source), str(tmp_path))

    assert copied == str(source)
    assert source.read_text(encoding="utf-8") == '{"type":"FeatureCollection","features":[]}'


def test_resolve_canonical_zone_source_prefers_staged_copy_without_warning_when_primary_exists(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    primary = tmp_path / "sources" / "taz_sfbay.geojson"
    staged_dir = tmp_path / "workspace" / "activitysim" / "data"
    staged = staged_dir / primary.name
    primary.parent.mkdir(parents=True, exist_ok=True)
    staged_dir.mkdir(parents=True, exist_ok=True)
    primary.write_text("primary", encoding="utf-8")
    staged.write_text("staged", encoding="utf-8")

    settings = SimpleNamespace(
        shared=SimpleNamespace(
            geography=SimpleNamespace(
                zones=SimpleNamespace(
                    source_file=str(primary),
                    canonical_id_col="zone_id",
                    activitysim_index_col="TAZ",
                    zone_type="taz",
                )
            )
        ),
        activitysim=SimpleNamespace(),
    )
    workspace = SimpleNamespace(get_asim_mutable_data_dir=lambda: str(staged_dir))

    with caplog.at_level("INFO"):
        resolved, _source_config = resolve_canonical_zone_source(settings, workspace)

    assert resolved == str(staged)
    assert not any(
        "Primary canonical zone source unavailable" in record.message
        for record in caplog.records
    )


def test_activitysim_copy_data_to_mutable_location_skips_duplicate_zone_records_when_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    asim_dir = tmp_path / "activitysim" / "data"
    asim_dir.mkdir(parents=True, exist_ok=True)
    staged_zone = asim_dir / "taz_sfbay.geojson"
    staged_zone.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    monkeypatch.setattr(
        activitysim_preprocessor,
        "find_project_root",
        lambda start_path: str(tmp_path),
    )
    monkeypatch.setattr(
        activitysim_preprocessor,
        "_copytree_if_needed",
        lambda source_dir, dest_dir: str(dest_dir),
    )
    monkeypatch.setattr(
        activitysim_preprocessor,
        "_ensure_required_asim_config_dirs",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        activitysim_preprocessor,
        "get_setting",
        lambda settings, key, default=None: (
            getattr(getattr(settings, key.split(".", 1)[0]), key.split(".", 1)[1], default)
            if "." in key and hasattr(settings, key.split(".", 1)[0])
            else getattr(settings, key, default)
        ),
    )

    def _unexpected_copy(source_path: str, dest_dir: str) -> str:
        raise AssertionError(
            f"copy_canonical_zone_source_to_dir should not run for staged source {source_path} -> {dest_dir}"
        )

    monkeypatch.setattr(
        activitysim_preprocessor,
        "copy_canonical_zone_source_to_dir",
        _unexpected_copy,
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(region="test"),
        shared=SimpleNamespace(
            geography=SimpleNamespace(
                zones=SimpleNamespace(
                    source_file=str(staged_zone),
                    canonical_id_col="zone_id",
                    activitysim_index_col="TAZ",
                    zone_type="taz",
                )
            )
        ),
        activitysim=SimpleNamespace(
            local_configs_folder="asim-configs",
            local_mutable_configs_folder="activitysim/configs",
            main_configs_dir="configs",
            clipped_geoms_path=None,
        ),
        beam=SimpleNamespace(local_input_folder="beam-input", router_directory="router"),
    )
    workspace = SimpleNamespace(get_asim_mutable_data_dir=lambda: str(asim_dir))

    with caplog.at_level("INFO"):
        input_records, output_records = activitysim_preprocessor._copy_data_to_mutable_location(
            settings,
            str(asim_dir),
            workspace,
        )

    assert input_records.all_records() == []
    assert output_records.all_records() == []
    assert any(
        "Canonical zones already at destination" in record.message
        for record in caplog.records
    )
