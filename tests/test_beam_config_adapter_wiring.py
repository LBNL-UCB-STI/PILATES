from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import MagicMock

import pytest

from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.steps import StepOutputsHolder, make_beam_preprocess_step


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

    def preprocess(self, workspace, inputs) -> RecordStore:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prepared = self.output_dir / "beam_input.txt"
        prepared.write_text("dummy")
        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=str(prepared),
                    short_name="beam_input_dummy",
                    description="Dummy BEAM input file",
                )
            ]
        )


class DummyWorkspace:
    def __init__(self, beam_dir: Path) -> None:
        self._beam_dir = Path(beam_dir)

    def get_beam_mutable_data_dir(self) -> str:
        return str(self._beam_dir)


def _make_settings(region: str, primary_conf: str) -> SimpleNamespace:
    run = SimpleNamespace(region=region)
    beam = SimpleNamespace(config=primary_conf)
    return SimpleNamespace(run=run, beam=beam)


def _make_state() -> SimpleNamespace:
    return SimpleNamespace(year=2020, iteration=0)


def _wire_step(monkeypatch, tracker, run_id, preprocessor: DummyPreprocessor) -> None:
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

    def _get_preprocessor(self, *args, **kwargs):
        return preprocessor

    monkeypatch.setattr(steps_module.ModelFactory, "get_preprocessor", _get_preprocessor)


def _setup_config(tmp_path: Path) -> Tuple[DummyWorkspace, SimpleNamespace]:
    beam_root = tmp_path / "beam"
    region = "test_region"
    config_root = beam_root / region
    config_root.mkdir(parents=True, exist_ok=True)
    primary_conf = "beam.conf"
    (config_root / primary_conf).write_text("beam.test = 1\n")

    workspace = DummyWorkspace(beam_root)
    settings = _make_settings(region=region, primary_conf=primary_conf)
    return workspace, settings


def test_beam_canonicalize_config_with_run_id(monkeypatch, tmp_path):
    pytest.importorskip("consist")
    from consist.integrations.beam import BeamConfigAdapter

    workspace, settings = _setup_config(tmp_path)
    state = _make_state()
    outputs_holder = StepOutputsHolder()
    coupler = DummyCoupler()

    tracker = MagicMock()
    preprocessor = DummyPreprocessor(tmp_path / "beam_inputs")

    _wire_step(monkeypatch, tracker, run_id="run-456", preprocessor=preprocessor)

    step_fn = make_beam_preprocess_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
    )
    step_fn(settings=settings, state=state, workspace=workspace)

    assert tracker.canonicalize_config.call_count == 1
    args, kwargs = tracker.canonicalize_config.call_args
    assert isinstance(args[0], BeamConfigAdapter)
    assert kwargs.get("run_id") == "run-456"
    config_dirs = args[1]
    assert Path(config_dirs[0]) == (
        Path(workspace.get_beam_mutable_data_dir()) / settings.run.region
    )


def test_beam_canonicalize_config_without_run_id(monkeypatch, tmp_path):
    pytest.importorskip("consist")
    from consist.integrations.beam import BeamConfigAdapter

    workspace, settings = _setup_config(tmp_path)
    state = _make_state()
    outputs_holder = StepOutputsHolder()
    coupler = DummyCoupler()

    tracker = MagicMock()
    preprocessor = DummyPreprocessor(tmp_path / "beam_inputs")

    _wire_step(monkeypatch, tracker, run_id=None, preprocessor=preprocessor)

    step_fn = make_beam_preprocess_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
    )
    step_fn(settings=settings, state=state, workspace=workspace)

    assert tracker.canonicalize_config.call_count == 1
    args, kwargs = tracker.canonicalize_config.call_args
    assert isinstance(args[0], BeamConfigAdapter)
    assert "run_id" not in kwargs
