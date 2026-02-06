from pathlib import Path
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from consist.core.step_context import StepContext

from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_beam_preprocess_step,
    make_beam_run_step,
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
        self.full_path = str(beam_dir.parent)

    def get_beam_mutable_data_dir(self) -> str:
        return str(self._beam_dir)


def _make_settings(region: str, primary_conf: str) -> SimpleNamespace:
    run = SimpleNamespace(region=region)
    beam = SimpleNamespace(config=primary_conf)
    return SimpleNamespace(run=run, beam=beam)


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
    tracker.prepare_config.return_value = SimpleNamespace(
        identity_hash="beam-plan-hash",
        adapter_version="beam-adapter-v1",
    )


def _make_step_context(*, step_fn, model, settings, workspace):
    sig = inspect.signature(StepContext)
    kwargs = {
        "func_name": step_fn.__name__,
        "model": model,
        "runtime_kwargs": {"workspace": workspace},
    }
    if "settings" in sig.parameters:
        kwargs["settings"] = settings
    if "runtime_settings" in sig.parameters:
        kwargs["runtime_settings"] = settings
    if "runtime_workspace" in sig.parameters:
        kwargs["runtime_workspace"] = workspace
    if "consist_settings" in sig.parameters:
        kwargs["consist_settings"] = SimpleNamespace()
    if "consist_workspace" in sig.parameters:
        kwargs["consist_workspace"] = Path(workspace.full_path)
    return StepContext(**kwargs)


def _setup_config(tmp_path: Path):
    beam_root = tmp_path / "beam"
    region = "test_region"
    config_root = beam_root / region
    config_root.mkdir(parents=True, exist_ok=True)
    primary_conf = "beam.conf"
    (config_root / primary_conf).write_text("beam.test = 1\n")
    workspace = DummyWorkspace(beam_root)
    settings = _make_settings(region=region, primary_conf=primary_conf)
    return workspace, settings


def test_beam_run_metadata_prepare_config_with_run_id(monkeypatch, tmp_path):
    pytest.importorskip("consist")
    from consist.integrations.beam import BeamConfigAdapter

    workspace, settings = _setup_config(tmp_path)
    tracker = MagicMock()
    _wire_common(monkeypatch, tracker, run_id="run-456")

    step_fn = make_beam_run_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    meta = step_fn.__consist_step__
    ctx = _make_step_context(
        step_fn=step_fn,
        model=meta.model,
        settings=settings,
        workspace=workspace,
    )

    resolved = meta.config(ctx)
    assert tracker.prepare_config.call_count == 1
    args, kwargs = tracker.prepare_config.call_args
    assert isinstance(args[0], BeamConfigAdapter)
    config_dirs = args[1]
    assert Path(config_dirs[0]) == (
        Path(workspace.get_beam_mutable_data_dir()) / settings.run.region
    )
    assert resolved["canonical_config_identity_hash"] == "beam-plan-hash"
    assert resolved["canonical_config_adapter_version"] == "beam-adapter-v1"
    assert tracker.canonicalize_config.call_count == 0


def test_beam_run_metadata_prepare_config_without_run_id(monkeypatch, tmp_path):
    pytest.importorskip("consist")

    workspace, settings = _setup_config(tmp_path)
    tracker = MagicMock()
    _wire_common(monkeypatch, tracker, run_id=None)

    step_fn = make_beam_run_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    meta = step_fn.__consist_step__
    ctx = _make_step_context(
        step_fn=step_fn,
        model=meta.model,
        settings=settings,
        workspace=workspace,
    )

    resolved = meta.config(ctx)
    assert tracker.prepare_config.call_count == 1
    assert resolved["canonical_config_identity_hash"] == "beam-plan-hash"
    assert tracker.canonicalize_config.call_count == 0


def test_beam_preprocess_does_not_canonicalize_in_step_body(monkeypatch, tmp_path):
    pytest.importorskip("consist")
    from pilates.utils import consist_runtime as cr
    from pilates.workflows import steps as steps_module

    workspace, settings = _setup_config(tmp_path)
    state = _make_state()
    tracker = MagicMock()
    preprocessor = DummyPreprocessor(tmp_path / "beam_inputs")

    monkeypatch.setattr(cr, "current_tracker", lambda: tracker)
    monkeypatch.setattr(cr, "current_run", lambda: None)
    monkeypatch.setattr(cr, "log_output", lambda path, **kwargs: SimpleNamespace(path=path))
    monkeypatch.setattr(cr, "log_input", lambda path, **kwargs: SimpleNamespace(path=path))
    monkeypatch.setattr(
        steps_module.ModelFactory,
        "get_preprocessor",
        lambda self, *args, **kwargs: preprocessor,
    )

    step_fn = make_beam_preprocess_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    step_fn(settings=settings, state=state, workspace=workspace)

    assert tracker.canonicalize_config.call_count == 0
    assert tracker.prepare_config.call_count == 0
