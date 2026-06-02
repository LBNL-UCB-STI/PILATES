from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.impacts.preprocessor import ImpactsPreprocessor
from pilates.workspace import Workspace


def _make_settings(*, seed_dir: Path, workspace_input_dir: str) -> SimpleNamespace:
    return SimpleNamespace(
        impacts=SimpleNamespace(
            local_input_folder=str(seed_dir),
            local_mutable_data_folder=workspace_input_dir,
            local_output_folder="impacts/impacts_output",
            input_manifest_filename="inputs_manifest.yaml",
        )
    )


def test_copy_data_to_mutable_location_copies_seed_tree(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    nested_dir = seed_dir / "nested"
    nested_dir.mkdir(parents=True)
    (seed_dir / "settings.yaml").write_text("seed: true\n", encoding="utf-8")
    (nested_dir / "reference.csv").write_text("value\n", encoding="utf-8")

    settings = _make_settings(
        seed_dir=seed_dir,
        workspace_input_dir="impacts/impacts_input",
    )
    workspace = Workspace(settings, str(tmp_path), "run")
    state = SimpleNamespace(
        full_settings=settings,
        set_sub_stage_progress=lambda *_args, **_kwargs: None,
    )
    preprocessor = ImpactsPreprocessor("impacts", state)

    input_records, output_records = preprocessor.copy_data_to_mutable_location(
        settings,
        workspace.get_impacts_input_dir(),
        workspace,
    )

    dest_dir = Path(workspace.get_impacts_input_dir())
    assert (dest_dir / "settings.yaml").read_text(encoding="utf-8") == "seed: true\n"
    assert (dest_dir / "nested" / "reference.csv").read_text(encoding="utf-8") == "value\n"
    assert input_records.all_records()
    assert output_records.all_records()
    assert any(
        record.short_name == "settings.yaml" for record in output_records.all_records()
    )
    assert any(
        record.short_name == "impacts_bootstrap_data_root"
        for record in output_records.all_records()
    )


def test_copy_data_to_mutable_location_raises_when_seed_missing(tmp_path: Path) -> None:
    seed_dir = tmp_path / "missing-seed"
    settings = _make_settings(
        seed_dir=seed_dir,
        workspace_input_dir="impacts/impacts_input",
    )
    workspace = Workspace(settings, str(tmp_path), "run")
    state = SimpleNamespace(
        full_settings=settings,
        set_sub_stage_progress=lambda *_args, **_kwargs: None,
    )
    preprocessor = ImpactsPreprocessor("impacts", state)

    with pytest.raises(FileNotFoundError, match="Impacts seed data directory not found"):
        preprocessor.copy_data_to_mutable_location(
            settings,
            workspace.get_impacts_input_dir(),
            workspace,
        )
