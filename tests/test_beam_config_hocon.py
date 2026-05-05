from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pilates.beam.config_hocon import (
    beam_config_env_overrides,
    beam_primary_config_path,
    load_resolved_beam_config_tree,
)


def _make_settings():
    return SimpleNamespace(
        run=SimpleNamespace(region="sfbay"),
        beam=SimpleNamespace(
            local_mutable_data_folder="beam/input",
            config="beam.conf",
        ),
    )


def test_beam_config_env_overrides_match_adapter_contract(tmp_path):
    settings = _make_settings()
    config_root = tmp_path / "beam" / "input" / "sfbay"
    config_root.mkdir(parents=True, exist_ok=True)

    overrides = beam_config_env_overrides(
        settings,
        workspace_path=str(tmp_path),
    )

    assert overrides == {
        "PWD": str(tmp_path / "beam"),
        "inputDirectory": str(config_root),
    }


def test_load_resolved_beam_config_tree_resolves_pwd_relative_values(tmp_path):
    settings = _make_settings()
    config_path = beam_primary_config_path(settings, workspace_path=str(tmp_path))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        'beam.outputs.base = ${PWD}"/output"\n',
        encoding="utf-8",
    )

    resolved = load_resolved_beam_config_tree(
        config_path,
        env_overrides=beam_config_env_overrides(
            settings,
            workspace_path=str(tmp_path),
        ),
    )

    assert resolved["beam"]["outputs"]["base"] == str(tmp_path / "beam" / "output")
