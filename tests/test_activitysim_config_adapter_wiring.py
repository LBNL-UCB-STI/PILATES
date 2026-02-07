from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.artifact_constants import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_PERSONS_IN,
)
from pilates.workflows.steps import StepOutputsHolder, make_activitysim_preprocess_step, make_activitysim_run_step


class DummyCoupler:
    def __init__(self) -> None:
        self._data = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value) -> None:
        self._data[key] = value


class DummyPreprocessor:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def preprocess(self, workspace) -> RecordStore:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        land_use = self.output_dir / "land_use.csv"
        households = self.output_dir / "households.csv"
        persons = self.output_dir / "persons.csv"
        for path in (land_use, households, persons):
            path.write_text("dummy")
        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=str(land_use),
                    short_name=ASIM_LAND_USE_IN,
                    description="ActivitySim land use input table",
                ),
                FileRecord(
                    file_path=str(households),
                    short_name=ASIM_HOUSEHOLDS_IN,
                    description="ActivitySim households input table",
                ),
                FileRecord(
                    file_path=str(persons),
                    short_name=ASIM_PERSONS_IN,
                    description="ActivitySim persons input table",
                ),
            ]
        )


class DummyWorkspace:
    def __init__(self, configs_dir: Path, data_dir: Path) -> None:
        self._configs_dir = Path(configs_dir)
        self._data_dir = Path(data_dir)

    def get_asim_mutable_configs_dir(self) -> str:
        return str(self._configs_dir)

    def get_asim_mutable_data_dir(self) -> str:
        return str(self._data_dir)

    def get_asim_output_dir(self) -> str:
        return str(self._data_dir / "output")


def _make_settings() -> SimpleNamespace:
    activitysim = SimpleNamespace(main_configs_dir="base")
    return SimpleNamespace(activitysim=activitysim)


def _setup_config(tmp_path: Path) -> tuple:
    """Create ActivitySim config directory structure for testing."""
    asim_root = tmp_path / "asim_configs"
    config_root = asim_root / "base"
    config_root.mkdir(parents=True, exist_ok=True)

    # Create a dummy config file
    (config_root / "settings.yaml").write_text("models: []\\n")

    return asim_root


def _make_state() -> SimpleNamespace:
    return SimpleNamespace(year=2020, iteration=0)


def _wire_step(monkeypatch, tracker, run_id) -> None:
    from pilates.utils import consist_runtime as cr
    from pilates.workflows import steps as steps_module

    monkeypatch.setattr(cr, "current_tracker", lambda: tracker)
    monkeypatch.setattr(
        cr,
        "current_run",
        lambda: SimpleNamespace(id=run_id) if run_id is not None else None,
    )
    monkeypatch.setattr(
        cr,
        "log_output",
        lambda path, **kwargs: SimpleNamespace(path=path),
    )
    monkeypatch.setattr(
        cr,
        "log_input",
        lambda path, **kwargs: SimpleNamespace(path=path),
    )

    dummy_preprocessor = DummyPreprocessor(Path(tracker._asim_output_dir))

    def _get_preprocessor(self, *args, **kwargs):
        return dummy_preprocessor

    monkeypatch.setattr(steps_module.ModelFactory, "get_preprocessor", _get_preprocessor)


def test_activitysim_canonicalize_config_with_run_id(monkeypatch, tmp_path):
    pytest.importorskip("consist.integrations.activitysim")

    asim_configs_root = _setup_config(tmp_path)
    workspace = DummyWorkspace(asim_configs_root, tmp_path / "asim_data")
    settings = _make_settings()
    state = _make_state()
    outputs_holder = StepOutputsHolder()
    coupler = DummyCoupler()

    tracker = MagicMock()
    tracker._asim_output_dir = workspace.get_asim_mutable_data_dir()

    _wire_step(monkeypatch, tracker, run_id="run-123")

    # Set up activitysim_preprocess prerequisite output
    from pilates.activitysim.outputs import ActivitySimPreprocessOutputs
    from pathlib import Path
    data_dir = Path(workspace.get_asim_mutable_data_dir())
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(workspace.get_asim_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_holder.activitysim_preprocess = ActivitySimPreprocessOutputs(
        mutable_data_dir=data_dir,
        land_use_table=data_dir / "land_use.csv",
        households_table=data_dir / "households.csv",
        persons_table=data_dir / "persons.csv"
    )

    step_fn = make_activitysim_run_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
    )

    # Mock the runner to avoid actual ActivitySim execution
    from pilates.activitysim.runner import ActivitysimRunner
    with monkeypatch.context() as m:
        m.setattr(ActivitysimRunner, "run", lambda self, *args, **kwargs: None)
        step_fn(settings=settings, state=state, workspace=workspace)

    assert tracker.canonicalize_config.call_count == 1
    args, kwargs = tracker.canonicalize_config.call_args
    assert kwargs.get("run_id") == "run-123"
    config_dirs = args[1]
    assert Path(config_dirs[0]) == asim_configs_root / "base"


def test_activitysim_canonicalize_config_without_run_id(monkeypatch, tmp_path):
    pytest.importorskip("consist.integrations.activitysim")

    asim_configs_root = _setup_config(tmp_path)
    workspace = DummyWorkspace(asim_configs_root, tmp_path / "asim_data")
    settings = _make_settings()
    state = _make_state()
    outputs_holder = StepOutputsHolder()
    coupler = DummyCoupler()

    tracker = MagicMock()
    tracker._asim_output_dir = workspace.get_asim_mutable_data_dir()

    _wire_step(monkeypatch, tracker, run_id=None)

    # Set up activitysim_preprocess prerequisite output
    from pilates.activitysim.outputs import ActivitySimPreprocessOutputs
    from pathlib import Path
    data_dir = Path(workspace.get_asim_mutable_data_dir())
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(workspace.get_asim_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_holder.activitysim_preprocess = ActivitySimPreprocessOutputs(
        mutable_data_dir=data_dir,
        land_use_table=data_dir / "land_use.csv",
        households_table=data_dir / "households.csv",
        persons_table=data_dir / "persons.csv"
    )

    step_fn = make_activitysim_run_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
    )

    # Mock the runner to avoid actual ActivitySim execution
    from pilates.activitysim.runner import ActivitysimRunner
    with monkeypatch.context() as m:
        m.setattr(ActivitysimRunner, "run", lambda self, *args, **kwargs: None)
        step_fn(settings=settings, state=state, workspace=workspace)

    assert tracker.canonicalize_config.call_count == 1
    _, kwargs = tracker.canonicalize_config.call_args
    assert "run_id" not in kwargs
