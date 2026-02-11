from pathlib import Path
from types import SimpleNamespace

from pilates.workflows.stages.vehicle_ownership import select_atlas_usim_input_path


class _WorkspaceStub:
    def __init__(self, usim_dir: Path):
        self._usim_dir = str(usim_dir)

    def get_usim_mutable_data_dir(self) -> str:
        return self._usim_dir


def _settings(output_template: str = "model_data_{year}.h5") -> SimpleNamespace:
    return SimpleNamespace(
        urbansim=SimpleNamespace(output_file_template=output_template),
    )


def _state(*, forecast_year: int = 2023, run_info_path: str = None) -> SimpleNamespace:
    return SimpleNamespace(
        forecast_year=forecast_year,
        run_info_path=run_info_path,
    )


def test_select_atlas_usim_input_path_prefers_forecast_output_from_previous_run(tmp_path):
    previous_run_dir = tmp_path / "previous_run"
    previous_usim_dir = previous_run_dir / "urbansim" / "data"
    previous_usim_dir.mkdir(parents=True)
    run_info_path = previous_run_dir / "run_info.json"
    run_info_path.write_text("{}")

    expected = previous_usim_dir / "model_data_2023.h5"
    expected.write_text("")

    workspace_usim_dir = tmp_path / "workspace" / "urbansim" / "data"
    workspace_usim_dir.mkdir(parents=True)
    current = workspace_usim_dir / "custom_current.h5"
    current.write_text("")
    default = workspace_usim_dir / "custom_default.h5"
    default.write_text("")

    selected = select_atlas_usim_input_path(
        settings=_settings(),
        state=_state(run_info_path=str(run_info_path)),
        workspace=_WorkspaceStub(workspace_usim_dir),
        fallback_current_path=current,
        fallback_default_path=default,
    )

    assert selected == str(expected)


def test_select_atlas_usim_input_path_falls_back_to_current(tmp_path):
    workspace_usim_dir = tmp_path / "workspace" / "urbansim" / "data"
    workspace_usim_dir.mkdir(parents=True)
    current = workspace_usim_dir / "custom_current.h5"
    current.write_text("")
    default = workspace_usim_dir / "custom_default.h5"
    default.write_text("")

    selected = select_atlas_usim_input_path(
        settings=_settings(),
        state=_state(),
        workspace=_WorkspaceStub(workspace_usim_dir),
        fallback_current_path=current,
        fallback_default_path=default,
    )

    assert selected == str(current)


def test_select_atlas_usim_input_path_falls_back_to_default(tmp_path):
    workspace_usim_dir = tmp_path / "workspace" / "urbansim" / "data"
    workspace_usim_dir.mkdir(parents=True)
    default = workspace_usim_dir / "custom_default.h5"
    default.write_text("")

    selected = select_atlas_usim_input_path(
        settings=_settings(),
        state=_state(),
        workspace=_WorkspaceStub(workspace_usim_dir),
        fallback_current_path=None,
        fallback_default_path=default,
    )

    assert selected == str(default)


def test_select_atlas_usim_input_path_returns_forecast_path_when_missing(tmp_path):
    workspace_usim_dir = tmp_path / "workspace" / "urbansim" / "data"
    workspace_usim_dir.mkdir(parents=True)

    selected = select_atlas_usim_input_path(
        settings=_settings(),
        state=_state(),
        workspace=_WorkspaceStub(workspace_usim_dir),
        fallback_current_path=None,
        fallback_default_path=None,
    )

    assert selected == str(workspace_usim_dir / "model_data_2023.h5")


def test_select_atlas_usim_input_path_can_prefer_current_over_forecast(tmp_path):
    workspace_usim_dir = tmp_path / "workspace" / "urbansim" / "data"
    workspace_usim_dir.mkdir(parents=True)
    current = workspace_usim_dir / "custom_current.h5"
    current.write_text("")
    forecast = workspace_usim_dir / "model_data_2023.h5"
    forecast.write_text("")

    selected = select_atlas_usim_input_path(
        settings=_settings(),
        state=_state(),
        workspace=_WorkspaceStub(workspace_usim_dir),
        fallback_current_path=current,
        fallback_default_path=None,
        prefer_forecast_output=False,
    )

    assert selected == str(current)
