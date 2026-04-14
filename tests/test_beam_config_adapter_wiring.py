import json
from pathlib import Path
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import os
import pytest
from consist.core.step_context import StepContext

pytest.importorskip("openmatrix")

from pilates.beam.outputs import BeamPreprocessOutputs
from pilates.beam.preprocessor import BeamPreprocessor
from pilates.utils.beam import get_beam_omx_skims_name
from pilates.workflows.artifact_keys import (
    BEAM_CONFIG_FILE,
    BEAM_FULL_SKIMS,
    BEAM_MUTABLE_DATA_DIR,
    BEAM_HOUSEHOLDS_IN,
    BEAM_OUTPUT_DIR,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
)
from pilates.workflows.steps.beam import _archive_beam_config_references
from pilates.workflows.steps.beam import (
    make_beam_full_skim_step,
    make_beam_postprocess_step,
    make_beam_preprocess_step,
    make_beam_run_step,
)
from pilates.workflows.steps.shared import StepOutputsHolder


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

    def preprocess(
        self,
        workspace,
        *,
        activity_demand_outputs=None,
        previous_beam_outputs=None,
        beam_preprocess_inputs=None,
    ) -> BeamPreprocessOutputs:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prepared = self.output_dir / "beam_input.txt"
        prepared.write_text("dummy")
        return BeamPreprocessOutputs(
            beam_mutable_data_dir=self.output_dir,
            prepared_inputs={"beam_input_dummy": prepared},
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
    return SimpleNamespace(
        year=2020,
        current_year=2020,
        iteration=0,
        current_inner_iter=0,
    )


def _wire_common(monkeypatch) -> None:
    monkeypatch.setattr(
        "pilates.workflows.step_consist_meta.build_step_consist_kwargs",
        lambda model, settings, workspace_path=None: {
            "config": {"model": model},
            "identity_inputs": [("shim", Path("/tmp/identity"))],
        },
    )


def _make_step_context(*, step_fn, model, settings, workspace, state=None):
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
    if "state" in sig.parameters and state is not None:
        kwargs["state"] = state
    if "runtime_state" in sig.parameters and state is not None:
        kwargs["runtime_state"] = state
    if "consist_settings" in sig.parameters:
        kwargs["consist_settings"] = SimpleNamespace()
    if "consist_workspace" in sig.parameters:
        kwargs["consist_workspace"] = Path(workspace.full_path)
    if "consist_state" in sig.parameters and state is not None:
        kwargs["consist_state"] = state
    return StepContext(**kwargs)


def _setup_config(tmp_path: Path):
    beam_root = tmp_path / "beam"
    region = "test_region"
    config_root = beam_root / region
    config_root.mkdir(parents=True, exist_ok=True)
    (beam_root / "output").mkdir(parents=True, exist_ok=True)
    primary_conf = "beam.conf"
    (config_root / primary_conf).write_text("beam.test = 1\n")
    workspace = DummyWorkspace(beam_root)
    workspace.get_beam_output_dir = lambda: str(beam_root / "output")
    workspace.get_asim_output_dir = lambda: str(tmp_path / "asim" / "output")
    workspace.get_atlas_output_dir = lambda: str(tmp_path / "atlas" / "output")
    settings = _make_settings(region=region, primary_conf=primary_conf)
    settings.activitysim = SimpleNamespace(file_format="parquet")
    settings.runtime = SimpleNamespace(
        flags=SimpleNamespace(
            activity_demand_enabled=True,
            vehicle_ownership_model_enabled=True,
        )
    )
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.write_skims_to_omx = True
    settings.shared = SimpleNamespace(geography=SimpleNamespace(zones=None))
    settings.beam.scenario_folder = "scenario"
    return workspace, settings


def _beam_postprocess_expected_inputs(settings, state, workspace):
    beam_output_dir = workspace.get_beam_output_dir()
    zarr_path = None
    if getattr(settings, "activitysim", None) is not None:
        candidate = os.path.join(workspace.get_asim_output_dir(), "cache", "skims.zarr")
        if os.path.exists(candidate):
            zarr_path = candidate
    return {
        "beam_output_dir": beam_output_dir if os.path.exists(beam_output_dir) else None,
        "zarr_skims": zarr_path,
    }


def _beam_postprocess_expected_outputs(settings, state, workspace):
    if getattr(settings, "activitysim", None) is not None:
        candidate = os.path.join(workspace.get_asim_output_dir(), "cache", "skims.zarr")
        if os.path.exists(candidate):
            return {"zarr_skims": candidate}
    return {}


def _beam_run_expected_inputs(settings, state, workspace):
    zarr_path = None
    if getattr(settings, "activitysim", None) is not None:
        candidate = os.path.join(workspace.get_asim_output_dir(), "cache", "skims.zarr")
        if os.path.exists(candidate):
            zarr_path = candidate
    return {
        "beam_mutable_data_dir": workspace.get_beam_mutable_data_dir(),
        "zarr_skims": zarr_path,
    }


def _beam_run_expected_outputs(settings, state, workspace):
    return {"beam_output_dir": workspace.get_beam_output_dir()}


def test_beam_run_metadata_emits_adapter_and_identity_inputs(monkeypatch, tmp_path):
    pytest.importorskip("consist")
    from consist.integrations.beam import BeamConfigAdapter

    workspace, settings = _setup_config(tmp_path)
    state = _make_state()
    _wire_common(monkeypatch)

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
        state=state,
    )

    resolved_config = meta.config(ctx)
    resolved_adapter = meta.adapter(ctx)
    resolved_identity_inputs = meta.identity_inputs(ctx)
    assert getattr(meta, "config_plan", None) is None
    assert getattr(meta, "hash_inputs", None) is None
    adapter = resolved_adapter
    assert isinstance(adapter, BeamConfigAdapter)
    assert adapter.root_dirs == [
        Path(workspace.get_beam_mutable_data_dir()) / settings.run.region
    ]
    assert adapter.primary_config == (
        Path(workspace.get_beam_mutable_data_dir()) / settings.run.region / settings.beam.config
    )
    assert adapter.env_overrides == {
        "PWD": str(Path(workspace.full_path) / "beam"),
        "inputDirectory": str(
            Path(workspace.get_beam_mutable_data_dir()) / settings.run.region
        ),
    }
    assert resolved_config["model"] == "beam_run"
    assert resolved_identity_inputs == [("shim", Path("/tmp/identity"))]


@pytest.mark.parametrize(
    "factory,expected_inputs,expected_outputs,expected_cache_hydration",
    [
        (
            make_beam_preprocess_step,
            BeamPreprocessor.expected_inputs,
            BeamPreprocessor.expected_outputs,
            "metadata",
        ),
        (
            make_beam_run_step,
            _beam_run_expected_inputs,
            _beam_run_expected_outputs,
            "inputs-missing",
        ),
        (
            make_beam_postprocess_step,
            _beam_postprocess_expected_inputs,
            _beam_postprocess_expected_outputs,
            "inputs-missing",
        ),
        (
            make_beam_full_skim_step,
            lambda settings, state, workspace: {
                BEAM_CONFIG_FILE: Path(
                    workspace.get_beam_mutable_data_dir()
                )
                / settings.run.region
                / settings.beam.config,
                BEAM_MUTABLE_DATA_DIR: workspace.get_beam_mutable_data_dir(),
                BEAM_OUTPUT_DIR: workspace.get_beam_output_dir(),
            },
            lambda settings, state, workspace: {
                BEAM_FULL_SKIMS: Path(workspace.get_beam_output_dir())
                / settings.run.region
                / f"year-{state.current_year}-iteration-{state.current_inner_iter}"
                / "skimsODFull.csv.gz",
            },
            "inputs-missing",
        ),
    ],
)
def test_beam_steps_declare_path_bound_replay_contract(
    factory,
    expected_inputs,
    expected_outputs,
    expected_cache_hydration,
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("consist")
    workspace, settings = _setup_config(tmp_path)
    state = _make_state()
    _wire_common(monkeypatch)

    step_fn = factory(coupler=DummyCoupler(), outputs_holder=StepOutputsHolder())
    meta = step_fn.__consist_step__
    ctx = _make_step_context(
        step_fn=step_fn,
        model=meta.model,
        settings=settings,
        workspace=workspace,
        state=state,
    )

    assert meta.input_binding == "paths"
    assert meta.cache_hydration == expected_cache_hydration
    assert callable(meta.inputs)
    assert callable(meta.output_paths)
    assert meta.inputs(ctx) == expected_inputs(settings, state, workspace)
    assert meta.output_paths(ctx) == expected_outputs(settings, state, workspace)


def test_beam_run_metadata_adapter_is_none_when_primary_config_missing(
    monkeypatch, tmp_path
):
    pytest.importorskip("consist")

    workspace, settings = _setup_config(tmp_path)
    _wire_common(monkeypatch)
    (Path(workspace.get_beam_mutable_data_dir()) / settings.run.region / settings.beam.config).unlink()

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

    assert meta.adapter(ctx) is None


def test_beam_preprocess_does_not_canonicalize_in_step_body(monkeypatch, tmp_path):
    pytest.importorskip("consist")
    from pilates.utils import consist_runtime as cr
    import pilates.workflows.steps.beam as steps_module

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
    assert tracker.prepare_config_resolver.call_count == 0


def test_beam_config_reference_archival_resolves_input_directory(
    tmp_path,
):
    workspace, settings = _setup_config(tmp_path)
    config_root = (
        Path(workspace.get_beam_mutable_data_dir()) / settings.run.region
    )
    config_path = config_root / settings.beam.config
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "sample.csv").write_text("value\n1\n", encoding="utf-8")
    config_path.write_text(
        'beam.agentsim.agents.vehicles.vehicleTypesFilePath = ${inputDirectory}"/sample.csv"\n',
        encoding="utf-8",
    )

    snapshot_dir = tmp_path / "snapshot"
    archive_root = _archive_beam_config_references(
        settings=settings,
        workspace=workspace,
        snapshot_dir=snapshot_dir,
    )

    assert archive_root == snapshot_dir / "beam_input_config_references_archived"
    assert (archive_root / "sample.csv").exists()
    manifest = json.loads(
        (archive_root / "__archive_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["sample.csv"] == str(config_root / "sample.csv")


def test_beam_preprocess_consumes_fallback_input_mapping(monkeypatch, tmp_path):
    captured = {}

    class CapturingPreprocessor:
        def preprocess(
            self,
            workspace,
            *,
            activity_demand_outputs=None,
            previous_beam_outputs=None,
            beam_preprocess_inputs=None,
        ) -> BeamPreprocessOutputs:
            captured["activity_demand_outputs"] = activity_demand_outputs
            captured["previous_beam_outputs"] = previous_beam_outputs
            captured["beam_preprocess_inputs"] = beam_preprocess_inputs
            return BeamPreprocessOutputs(
                beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
                prepared_inputs={},
            )

    workspace, settings = _setup_config(tmp_path)
    state = _make_state()
    plans = tmp_path / "plans.parquet"
    households = tmp_path / "households.parquet"
    persons = tmp_path / "persons.parquet"
    for path in (plans, households, persons):
        path.write_text("x")

    import pilates.workflows.steps.beam as steps_module

    monkeypatch.setattr(
        steps_module.ModelFactory,
        "get_preprocessor",
        lambda self, *args, **kwargs: CapturingPreprocessor(),
    )

    step_fn = make_beam_preprocess_step(
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    step_fn(
        settings=settings,
        state=state,
        workspace=workspace,
        beam_preprocess_inputs={
            BEAM_PLANS_IN: str(plans),
            BEAM_HOUSEHOLDS_IN: str(households),
            BEAM_PERSONS_IN: str(persons),
        },
    )

    assert captured["activity_demand_outputs"] is None
    assert captured["previous_beam_outputs"] is None
    assert captured["beam_preprocess_inputs"] == {
        BEAM_PLANS_IN: str(plans),
        BEAM_HOUSEHOLDS_IN: str(households),
        BEAM_PERSONS_IN: str(persons),
    }
