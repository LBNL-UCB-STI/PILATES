from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.atlas.outputs import AtlasPreprocessOutputs
from pilates.atlas import runner as atlas_runner
from pilates.atlas.runner import AtlasLaunchContext, AtlasRunner


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(vehicle_ownership_freq=1),
        atlas=SimpleNamespace(
            num_processes=1,
            sample_size=1,
            beamac=0,
            mod=0,
            adscen="baseline",
            rebfactor=1,
            taxfactor=1,
            discIncent=0,
            max_retries=1,
            container_input_folder="/input",
            container_output_folder="/output",
            basedir="/atlas",
            codedir="/atlas/code",
            command_template="atlas {0}",
        ),
    )


def test_atlas_runner_stages_selected_static_inputs_before_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    atlas_input = tmp_path / "atlas" / "atlas_input"
    atlas_output = tmp_path / "atlas" / "atlas_output"
    source = tmp_path / "selected-static" / "modeaccessibility.csv"
    source.parent.mkdir(parents=True)
    source.write_text("static input\n", encoding="utf-8")
    settings = _settings()
    state = SimpleNamespace(
        full_settings=settings,
        current_year=2030,
        forecast_year=2030,
        set_sub_stage_progress=lambda _stage: None,
    )
    workspace = SimpleNamespace(
        get_atlas_mutable_input_dir=lambda: str(atlas_input),
        get_atlas_output_dir=lambda: str(atlas_output),
    )
    runner = AtlasRunner("atlas", state)
    monkeypatch.setattr(runner, "get_model_and_image", lambda *_args: (None, "atlas"))
    monkeypatch.setattr(
        atlas_runner,
        "selected_atlas_static_input_sources",
        lambda _settings: (("modeaccessibility.csv", source),),
    )

    def successful_container(**_kwargs: object) -> bool:
        assert (atlas_input / "modeaccessibility.csv").read_text(
            encoding="utf-8"
        ) == "static input\n"
        atlas_output.mkdir(parents=True, exist_ok=True)
        (atlas_output / "householdv_2030.csv").write_text("household_id\n1\n")
        (atlas_output / "vehicles_2030.csv").write_text("vehicle_id\n1\n")
        continuation_dir = atlas_input / "year2030"
        continuation_dir.mkdir(parents=True)
        (continuation_dir / "vehicles_output.RData").write_text("vehicles")
        (continuation_dir / "households_output.RData").write_text("households")
        return True

    monkeypatch.setattr(runner, "run_container", successful_container)

    runner.run(
        AtlasPreprocessOutputs(atlas_mutable_input_dir=atlas_input),
        AtlasLaunchContext(input_root=atlas_input, output_root=atlas_output),
    )

    assert (atlas_input / "modeaccessibility.csv").exists()


def test_atlas_run_rejects_missing_canonical_continuation_rdata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    atlas_input = tmp_path / "atlas" / "atlas_input"
    atlas_output = tmp_path / "atlas" / "atlas_output"
    atlas_input.mkdir(parents=True)
    atlas_output.mkdir(parents=True)
    settings = _settings()
    state = SimpleNamespace(
        full_settings=settings,
        current_year=2030,
        forecast_year=2030,
        set_sub_stage_progress=lambda _stage: None,
    )
    workspace = SimpleNamespace(
        get_atlas_mutable_input_dir=lambda: str(atlas_input),
        get_atlas_output_dir=lambda: str(atlas_output),
    )
    runner = AtlasRunner("atlas", state)
    monkeypatch.setattr(runner, "get_model_and_image", lambda *_args: (None, "atlas"))

    def successful_container(**_kwargs: object) -> bool:
        (atlas_output / "householdv_2030.csv").write_text("household_id\n1\n")
        (atlas_output / "vehicles_2030.csv").write_text("vehicle_id\n1\n")
        (atlas_input / "year2030").mkdir()
        (atlas_input / "year2030" / "vehicles_output.RData").write_text("rdata")
        return True

    monkeypatch.setattr(runner, "run_container", successful_container)

    with pytest.raises(RuntimeError, match="continuation.*households_output.RData"):
        runner.run(
            AtlasPreprocessOutputs(atlas_mutable_input_dir=atlas_input),
            AtlasLaunchContext(input_root=atlas_input, output_root=atlas_output),
        )
