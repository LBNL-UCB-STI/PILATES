from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from consist.core.step_context import StepContext

from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.artifact_constants import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_PERSONS_IN,
)
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
)


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
        self.full_path = str(configs_dir.parent)

    def get_asim_mutable_configs_dir(self) -> str:
        return str(self._configs_dir)

    def get_asim_mutable_data_dir(self) -> str:
        return str(self._data_dir)


def _fixture_root() -> Path:
    return (
        Path(__file__).resolve().parent
        / "fixtures"
        / "consist"
        / "activitysim_small"
    )


def _make_settings() -> SimpleNamespace:
    activitysim = SimpleNamespace(main_configs_dir="base")
    return SimpleNamespace(activitysim=activitysim)


def _make_state() -> SimpleNamespace:
    return SimpleNamespace(year=2020, iteration=0)


def _wire_common(monkeypatch, tracker, run_id) -> None:
    from pilates.utils import consist_runtime as cr

    monkeypatch.setattr(cr, "current_tracker", lambda: tracker)
    monkeypatch.setattr(
        cr,
        "current_run",
        lambda: SimpleNamespace(id=run_id) if run_id is not None else None,
    )
    monkeypatch.setattr(
        "pilates.workflows.step_consist_meta.build_step_consist_kwargs",
        lambda model, settings, workspace_path=None: {"config": {"model": model}},
    )


def test_activitysim_run_metadata_canonicalize_config_with_run_id(monkeypatch, tmp_path):
    pytest.importorskip("consist")
    from consist.integrations.activitysim import ActivitySimConfigAdapter

    fixture_root = _fixture_root()
    workspace = DummyWorkspace(fixture_root, tmp_path / "asim_data")
    settings = _make_settings()
    tracker = MagicMock()

    _wire_common(monkeypatch, tracker, run_id="run-123")

    step_fn = make_activitysim_run_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    meta = step_fn.__consist_step__
    ctx = StepContext(
        func_name=step_fn.__name__,
        model=meta.model,
        settings=settings,
        runtime_kwargs={"workspace": workspace},
    )

    resolved = meta.config(ctx)
    assert tracker.canonicalize_config.call_count == 1
    args, kwargs = tracker.canonicalize_config.call_args
    assert isinstance(args[0], ActivitySimConfigAdapter)
    assert kwargs.get("run_id") == "run-123"
    config_dirs = args[1]
    assert Path(config_dirs[0]) == fixture_root / "base"
    assert "canonical_config_identity_hash" in resolved


def test_activitysim_run_metadata_canonicalize_config_without_run_id(
    monkeypatch, tmp_path
):
    pytest.importorskip("consist")

    fixture_root = _fixture_root()
    workspace = DummyWorkspace(fixture_root, tmp_path / "asim_data")
    settings = _make_settings()
    tracker = MagicMock()

    _wire_common(monkeypatch, tracker, run_id=None)

    step_fn = make_activitysim_run_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    meta = step_fn.__consist_step__
    ctx = StepContext(
        func_name=step_fn.__name__,
        model=meta.model,
        settings=settings,
        runtime_kwargs={"workspace": workspace},
    )

    meta.config(ctx)
    assert tracker.canonicalize_config.call_count == 1
    _, kwargs = tracker.canonicalize_config.call_args
    assert kwargs.get("run_id") is None


def test_activitysim_preprocess_does_not_canonicalize_in_step_body(monkeypatch, tmp_path):
    pytest.importorskip("consist")
    from pilates.utils import consist_runtime as cr
    from pilates.workflows import steps as steps_module

    fixture_root = _fixture_root()
    workspace = DummyWorkspace(fixture_root, tmp_path / "asim_data")
    settings = _make_settings()
    state = _make_state()
    tracker = MagicMock()

    monkeypatch.setattr(cr, "current_tracker", lambda: tracker)
    monkeypatch.setattr(cr, "current_run", lambda: None)
    monkeypatch.setattr(cr, "log_output", lambda path, **kwargs: SimpleNamespace(path=path))
    monkeypatch.setattr(cr, "log_input", lambda path, **kwargs: SimpleNamespace(path=path))

    dummy_preprocessor = DummyPreprocessor(tmp_path / "asim_preprocess")
    Path(workspace.get_asim_mutable_data_dir()).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        steps_module.ModelFactory,
        "get_preprocessor",
        lambda self, *args, **kwargs: dummy_preprocessor,
    )

    step_fn = make_activitysim_preprocess_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    step_fn(settings=settings, state=state, workspace=workspace)

    assert tracker.canonicalize_config.call_count == 0
