from pathlib import Path
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from consist.core.step_context import StepContext

from pilates.activitysim.outputs import ActivitySimPreprocessOutputs
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_PERSONS_IN,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
)


class DummyCoupler:
    def __init__(self, data=None) -> None:
        self._data = dict(data or {})

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value) -> None:
        self._data[key] = value


class DummyPreprocessor:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def preprocess(self, workspace) -> ActivitySimPreprocessOutputs:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        land_use = self.output_dir / "land_use.csv"
        households = self.output_dir / "households.csv"
        persons = self.output_dir / "persons.csv"
        for path in (land_use, households, persons):
            path.write_text("dummy")
        return ActivitySimPreprocessOutputs(
            mutable_data_dir=self.output_dir,
            land_use_table=land_use,
            households_table=households,
            persons_table=persons,
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
        Path(__file__).resolve().parent / "fixtures" / "consist" / "activitysim_small"
    )


def _make_settings() -> SimpleNamespace:
    activitysim = SimpleNamespace(main_configs_dir="base")
    return SimpleNamespace(activitysim=activitysim)


def _make_state() -> SimpleNamespace:
    return SimpleNamespace(year=2020, iteration=0)


def _wire_common(monkeypatch) -> None:
    monkeypatch.setattr(
        "pilates.workflows.step_consist_meta.build_step_consist_kwargs",
        lambda model, settings, workspace_path=None: {
            "config": {"model": model},
            "identity_inputs": [("shim", Path("/tmp/identity"))],
        },
    )


def _make_step_context(
    *,
    step_fn,
    model,
    context_settings,
    runtime_workspace,
    runtime_settings_override=None,
):
    sig = inspect.signature(StepContext)
    kwargs = {
        "func_name": step_fn.__name__,
        "model": model,
        "runtime_kwargs": {"workspace": runtime_workspace},
    }
    if runtime_settings_override is not None:
        kwargs["runtime_kwargs"]["settings"] = runtime_settings_override

    if "settings" in sig.parameters:
        kwargs["settings"] = context_settings
    if "runtime_settings" in sig.parameters:
        kwargs["runtime_settings"] = (
            runtime_settings_override
            if runtime_settings_override is not None
            else context_settings
        )
    if "runtime_workspace" in sig.parameters:
        kwargs["runtime_workspace"] = runtime_workspace
    if "consist_settings" in sig.parameters:
        kwargs["consist_settings"] = SimpleNamespace()
    if "consist_workspace" in sig.parameters:
        kwargs["consist_workspace"] = Path(runtime_workspace.full_path)
    return StepContext(**kwargs)


def test_activitysim_run_metadata_emits_adapter_and_identity_inputs(
    monkeypatch, tmp_path
):
    pytest.importorskip("consist")
    from consist.integrations.activitysim import ActivitySimConfigAdapter

    fixture_root = _fixture_root()
    workspace = DummyWorkspace(fixture_root, tmp_path / "asim_data")
    settings = _make_settings()
    _wire_common(monkeypatch)

    step_fn = make_activitysim_run_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    meta = step_fn.__consist_step__
    ctx = _make_step_context(
        step_fn=step_fn,
        model=meta.model,
        context_settings=settings,
        runtime_workspace=workspace,
    )

    resolved_config = meta.config(ctx)
    resolved_adapter = meta.adapter(ctx)
    resolved_identity_inputs = meta.identity_inputs(ctx)
    assert getattr(meta, "config_plan", None) is None
    assert getattr(meta, "hash_inputs", None) is None
    adapter = resolved_adapter
    assert isinstance(adapter, ActivitySimConfigAdapter)
    assert adapter.root_dirs == [fixture_root / "base"]
    assert resolved_config["model"] == "activitysim_run"
    assert resolved_identity_inputs == [("shim", Path("/tmp/identity"))]


def test_activitysim_run_metadata_adapter_is_none_when_config_root_missing(
    monkeypatch, tmp_path
):
    pytest.importorskip("consist")

    workspace = DummyWorkspace(
        tmp_path / "missing_configs_root", tmp_path / "asim_data"
    )
    settings = _make_settings()
    _wire_common(monkeypatch)

    step_fn = make_activitysim_run_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    meta = step_fn.__consist_step__
    ctx = _make_step_context(
        step_fn=step_fn,
        model=meta.model,
        context_settings=settings,
        runtime_workspace=workspace,
    )

    assert meta.adapter(ctx) is None


def test_activitysim_run_metadata_adapter_includes_overlay_config_roots_when_present(
    monkeypatch, tmp_path
):
    pytest.importorskip("consist")
    from consist.integrations.activitysim import ActivitySimConfigAdapter

    configs_root = tmp_path / "asim_configs"
    (configs_root / "base").mkdir(parents=True)
    (configs_root / "configs_mp").mkdir(parents=True)
    (configs_root / "configs_sh_compile").mkdir(parents=True)

    (configs_root / "base" / "settings.yaml").write_text("models: []\n")
    (configs_root / "configs_mp" / "settings.yaml").write_text("models: []\n")
    (configs_root / "configs_sh_compile" / "settings.yaml").write_text("models: []\n")

    workspace = DummyWorkspace(configs_root, tmp_path / "asim_data")
    settings = _make_settings()
    _wire_common(monkeypatch)

    step_fn = make_activitysim_run_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    meta = step_fn.__consist_step__
    ctx = _make_step_context(
        step_fn=step_fn,
        model=meta.model,
        context_settings=settings,
        runtime_workspace=workspace,
    )

    adapter = meta.adapter(ctx)
    assert isinstance(adapter, ActivitySimConfigAdapter)
    assert adapter.root_dirs == [
        configs_root / "base",
        configs_root / "configs_mp",
        configs_root / "configs_sh_compile",
    ]


def test_activitysim_run_metadata_filters_adapter_covered_identity_inputs(
    monkeypatch, tmp_path
):
    pytest.importorskip("consist")

    fixture_root = _fixture_root()
    workspace = DummyWorkspace(fixture_root, tmp_path / "asim_data")
    settings = _make_settings()
    monkeypatch.setattr(
        "pilates.workflows.step_consist_meta.build_step_consist_kwargs",
        lambda model, settings, workspace_path=None: {
            "config": {"model": model},
            "identity_inputs": [
                ("asim_mutable_configs", Path("/tmp/asim/configs")),
                (
                    "asim_mutable_configs/settings.yaml",
                    Path("/tmp/asim/configs/settings.yaml"),
                ),
                ("external_marker", Path("/tmp/external")),
            ],
        },
    )

    step_fn = make_activitysim_run_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    meta = step_fn.__consist_step__
    ctx = _make_step_context(
        step_fn=step_fn,
        model=meta.model,
        context_settings=settings,
        runtime_workspace=workspace,
    )

    assert meta.adapter(ctx) is not None
    assert meta.identity_inputs(ctx) == [("external_marker", Path("/tmp/external"))]


def test_activitysim_run_metadata_keeps_identity_inputs_when_adapter_missing(
    monkeypatch, tmp_path
):
    pytest.importorskip("consist")

    workspace = DummyWorkspace(
        tmp_path / "missing_configs_root", tmp_path / "asim_data"
    )
    settings = _make_settings()
    identity_inputs = [("asim_mutable_configs", Path("/tmp/asim/configs"))]
    monkeypatch.setattr(
        "pilates.workflows.step_consist_meta.build_step_consist_kwargs",
        lambda model, settings, workspace_path=None: {
            "config": {"model": model},
            "identity_inputs": list(identity_inputs),
        },
    )

    step_fn = make_activitysim_run_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    meta = step_fn.__consist_step__
    ctx = _make_step_context(
        step_fn=step_fn,
        model=meta.model,
        context_settings=settings,
        runtime_workspace=workspace,
    )

    assert meta.adapter(ctx) is None
    assert meta.identity_inputs(ctx) == identity_inputs


def test_activitysim_preprocess_does_not_canonicalize_in_step_body(
    monkeypatch, tmp_path
):
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
    monkeypatch.setattr(
        cr, "log_output", lambda path, **kwargs: SimpleNamespace(path=path)
    )
    monkeypatch.setattr(
        cr, "log_input", lambda path, **kwargs: SimpleNamespace(path=path)
    )

    dummy_preprocessor = DummyPreprocessor(tmp_path / "asim_preprocess")
    Path(workspace.get_asim_mutable_data_dir()).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        steps_module.ModelFactory,
        "get_preprocessor",
        lambda self, *args, **kwargs: dummy_preprocessor,
    )
    population_source_h5 = tmp_path / "population_source.h5"
    population_source_h5.write_text("stub", encoding="utf-8")

    step_fn = make_activitysim_preprocess_step(
        coupler=DummyCoupler({USIM_POPULATION_SOURCE_H5: str(population_source_h5)}),
        outputs_holder=StepOutputsHolder(),
    )
    step_fn(settings=settings, state=state, workspace=workspace)

    assert tracker.canonicalize_config.call_count == 0
    assert tracker.prepare_config.call_count == 0
    assert tracker.prepare_config_resolver.call_count == 0


def test_activitysim_metadata_uses_runtime_settings_over_ctx_settings(
    monkeypatch, tmp_path
):
    pytest.importorskip("consist")

    fixture_root = _fixture_root()
    workspace = DummyWorkspace(fixture_root, tmp_path / "asim_data")
    _wire_common(monkeypatch)

    step_fn = make_activitysim_run_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    meta = step_fn.__consist_step__
    ctx = _make_step_context(
        step_fn=step_fn,
        model=meta.model,
        context_settings=SimpleNamespace(),
        runtime_workspace=workspace,
        runtime_settings_override=_make_settings(),
    )

    resolved_adapter = meta.adapter(ctx)
    resolved_identity_inputs = meta.identity_inputs(ctx)
    assert resolved_adapter is not None
    assert resolved_identity_inputs == [("shim", Path("/tmp/identity"))]
