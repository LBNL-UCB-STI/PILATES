from pathlib import Path

import pytest

from pilates.activitysim.preprocessor import _ensure_required_asim_config_dirs


def test_ensure_required_asim_config_dirs_synthesizes_missing_overlays(
    tmp_path: Path,
) -> None:
    configs_dest_dir = tmp_path / "configs_mutable"
    configs_dir = configs_dest_dir / "configs"
    configs_extended_dir = configs_dest_dir / "configs_extended"
    configs_dir.mkdir(parents=True)
    configs_extended_dir.mkdir(parents=True)

    (configs_dir / "settings.yaml").write_text("sample: 1\n", encoding="utf-8")
    (configs_extended_dir / "logging.yaml").write_text("sample: 1\n", encoding="utf-8")

    _ensure_required_asim_config_dirs(
        configs_dest_dir=str(configs_dest_dir),
        main_configs_dir="configs",
    )

    assert (configs_dest_dir / "configs_mp").is_dir()
    assert (configs_dest_dir / "configs_sh_compile").is_dir()
    assert (configs_dest_dir / "configs_mp" / "settings.yaml").is_file()
    assert (configs_dest_dir / "configs_sh_compile" / "settings.yaml").is_file()


def test_ensure_required_asim_config_dirs_raises_without_any_base_dir(
    tmp_path: Path,
) -> None:
    configs_dest_dir = tmp_path / "configs_mutable"
    configs_dest_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="missing all base directories"):
        _ensure_required_asim_config_dirs(
            configs_dest_dir=str(configs_dest_dir),
            main_configs_dir="configs",
        )
