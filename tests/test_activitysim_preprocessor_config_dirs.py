from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pilates.activitysim.preprocessor import (
    _disable_activitysim_vehicle_ownership_models_for_atlas,
    _ensure_required_asim_config_dirs,
)
from pilates.config.models import ActivitySimConfig


def _activitysim_config(*, main_configs_dir: str) -> ActivitySimConfig:
    return ActivitySimConfig(
        local_input_folder="activitysim/input",
        local_mutable_data_folder="activitysim/data",
        local_output_folder="activitysim/output",
        local_configs_folder="pilates/activitysim/configs",
        local_mutable_configs_folder="activitysim/configs",
        validation_folder="activitysim/validation",
        command_template="activitysim run -c {config}",
        final_plans_folder="activitysim/final-plans",
        main_configs_dir=main_configs_dir,
    )


@pytest.mark.parametrize(
    "main_configs_dir",
    ("/ambient/configs", "../configs", "scenarios/../../ambient/configs"),
)
def test_activitysim_main_configs_dir_rejects_non_logical_relative_paths(
    main_configs_dir: str,
) -> None:
    """An unsafe config root must fail before adapter, staging, or launch."""

    with pytest.raises(ValueError, match="logical relative path"):
        _activitysim_config(main_configs_dir=main_configs_dir)


def test_activitysim_main_configs_dir_preserves_nested_logical_path() -> None:
    config = _activitysim_config(main_configs_dir="scenarios/sfbay/configs")

    assert config.main_configs_dir == "scenarios/sfbay/configs"


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


def test_disable_activitysim_vehicle_ownership_models_for_atlas(tmp_path: Path) -> None:
    configs_dest_dir = tmp_path / "configs_mutable"
    config_dir = configs_dest_dir / "configs"
    mp_dir = configs_dest_dir / "configs_mp"
    config_dir.mkdir(parents=True)
    mp_dir.mkdir(parents=True)
    for directory in (config_dir, mp_dir):
        (directory / "settings.yaml").write_text(
            yaml.safe_dump(
                {
                    "models": [
                        "initialize_landuse",
                        "auto_ownership_simulate",
                        "mandatory_tour_frequency",
                        "vehicle_type_choice",
                    ],
                    "chunk_size": 0,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    settings = SimpleNamespace(
        run=SimpleNamespace(models=SimpleNamespace(vehicle_ownership="atlas"))
    )

    removed = _disable_activitysim_vehicle_ownership_models_for_atlas(
        settings=settings,
        configs_dest_dir=str(configs_dest_dir),
        main_configs_dir="configs",
    )

    assert set(removed) == {
        str(config_dir / "settings.yaml"),
        str(mp_dir / "settings.yaml"),
    }
    updated = yaml.safe_load((config_dir / "settings.yaml").read_text(encoding="utf-8"))
    assert updated["models"] == ["initialize_landuse", "mandatory_tour_frequency"]
    assert updated["chunk_size"] == 0


def test_disable_activitysim_vehicle_ownership_models_skips_non_atlas(
    tmp_path: Path,
) -> None:
    configs_dest_dir = tmp_path / "configs_mutable"
    config_dir = configs_dest_dir / "configs"
    config_dir.mkdir(parents=True)
    settings_path = config_dir / "settings.yaml"
    settings_path.write_text(
        yaml.safe_dump({"models": ["auto_ownership_simulate"]}),
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(models=SimpleNamespace(vehicle_ownership=None))
    )

    removed = _disable_activitysim_vehicle_ownership_models_for_atlas(
        settings=settings,
        configs_dest_dir=str(configs_dest_dir),
        main_configs_dir="configs",
    )

    assert removed == {}
    updated = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    assert updated["models"] == ["auto_ownership_simulate"]
