from __future__ import annotations

from pathlib import Path

from pilates.activitysim.preprocessor import _copy_path_if_needed, _copytree_if_needed


def test_copy_path_if_needed_is_noop_for_same_file(tmp_path: Path) -> None:
    source = tmp_path / "clipped.geojson"
    source.write_text("{}", encoding="utf-8")

    copied = _copy_path_if_needed(str(source), str(source))

    assert copied == str(source)
    assert source.read_text(encoding="utf-8") == "{}"


def test_copytree_if_needed_is_noop_for_same_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "configs"
    source_dir.mkdir()
    config = source_dir / "settings.yaml"
    config.write_text("x: 1\n", encoding="utf-8")

    copied = _copytree_if_needed(str(source_dir), str(source_dir))

    assert copied == str(source_dir)
    assert config.read_text(encoding="utf-8") == "x: 1\n"
