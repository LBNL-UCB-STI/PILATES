from pathlib import Path
from types import SimpleNamespace
import yaml

from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    BEAM_PLANS_OUT,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS,
    LINKSTATS_WARMSTART,
)
from pilates.workflows.orchestration import ManifestConfig, _recover_cached_outputs
from pilates.workflows.orchestration import run_manifested_steps
from pilates.workflows.orchestration import StepRef
from pilates.workflows.steps import StepOutputsHolder


class DummyScenario:
    def __init__(self, cache_hit: bool = True) -> None:
        self.cache_hit = cache_hit
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(cache_hit=self.cache_hit)


class DummyWorkspace:
    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def full_path(self) -> str:
        return str(self._root)

    def get_asim_mutable_data_dir(self) -> str:
        return str(self._root / "activitysim" / "data")

    def get_beam_mutable_data_dir(self) -> str:
        return str(self._root / "beam" / "input")

    def get_beam_output_dir(self) -> str:
        return str(self._root / "beam" / "output")


class DummyCoupler:
    def __init__(self) -> None:
        self.values = {}

    def get(self, _key, default=None):
        return self.values.get(_key, default)

    def set(self, _key, _value):
        self.values[_key] = _value

    def update(self, _mapping):
        self.values.update(_mapping)


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test")


def test_recover_activitysim_preprocess_outputs(tmp_path):
    workspace = DummyWorkspace(tmp_path)
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    _write_file(asim_dir / "households.csv")
    _write_file(asim_dir / "persons.csv")
    _write_file(asim_dir / "land_use.csv")
    _write_file(asim_dir / "skims.omx")

    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="activitysim_preprocess",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        step_inputs=None,
    )

    assert outputs is not None
    assert holder.activitysim_preprocess is not None
    assert holder.activitysim_preprocess.mutable_data_dir == asim_dir
    assert coupler.get(ASIM_HOUSEHOLDS_IN) is not None
    assert coupler.get(ASIM_PERSONS_IN) is not None
    assert coupler.get(ASIM_LAND_USE_IN) is not None
    holder.activitysim_preprocess.validate()


def test_recover_activitysim_preprocess_outputs_from_archive_only(tmp_path, monkeypatch):
    local_root = tmp_path / "local-run"
    archive_root = tmp_path / "archive-run"
    workspace = DummyWorkspace(local_root)
    local_asim_dir = Path(workspace.get_asim_mutable_data_dir())
    archive_asim_dir = archive_root / local_asim_dir.relative_to(local_root)

    _write_file(archive_asim_dir / "households.csv")
    _write_file(archive_asim_dir / "persons.csv")
    _write_file(archive_asim_dir / "land_use.csv")
    _write_file(archive_asim_dir / "skims.omx")

    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="activitysim_preprocess",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        step_inputs=None,
    )

    assert outputs is not None
    assert holder.activitysim_preprocess is not None
    assert (local_asim_dir / "households.csv").exists()
    assert (local_asim_dir / "persons.csv").exists()
    assert (local_asim_dir / "land_use.csv").exists()
    assert (local_asim_dir / "skims.omx").exists()
    assert coupler.get(ASIM_HOUSEHOLDS_IN) is not None
    assert coupler.get(ASIM_PERSONS_IN) is not None
    assert coupler.get(ASIM_LAND_USE_IN) is not None
    holder.activitysim_preprocess.validate()


def test_recover_beam_preprocess_outputs(tmp_path):
    workspace = DummyWorkspace(tmp_path)
    beam_dir = Path(workspace.get_beam_mutable_data_dir())
    plans_path = beam_dir / "seattle" / "urbansim" / "plans.parquet"
    households_path = beam_dir / "seattle" / "urbansim" / "households.parquet"
    persons_path = beam_dir / "seattle" / "urbansim" / "persons.parquet"
    linkstats_path = beam_dir / "seattle" / "r5" / "init.linkstats.csv.gz"
    for path in (plans_path, households_path, persons_path, linkstats_path):
        _write_file(path)

    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="beam_preprocess",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        step_inputs={
            BEAM_PLANS_IN: str(plans_path),
            BEAM_HOUSEHOLDS_IN: str(households_path),
            BEAM_PERSONS_IN: str(persons_path),
            LINKSTATS_WARMSTART: str(linkstats_path),
        },
    )

    assert outputs is not None
    assert holder.beam_preprocess is not None
    assert holder.beam_preprocess.prepared_inputs[BEAM_PLANS_IN] == plans_path
    assert coupler.get(BEAM_PLANS_IN) is not None
    assert coupler.get(BEAM_HOUSEHOLDS_IN) is not None
    assert coupler.get(BEAM_PERSONS_IN) is not None
    assert coupler.get(LINKSTATS_WARMSTART) is not None
    holder.beam_preprocess.validate()


def test_run_manifested_steps_recovers_cache_hit(tmp_path):
    workspace = DummyWorkspace(tmp_path)
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    _write_file(asim_dir / "households.csv")
    _write_file(asim_dir / "persons.csv")
    _write_file(asim_dir / "land_use.csv")

    def _noop_step(**_kwargs):
        raise AssertionError("step function should not execute on cache hit")
    _noop_step.__consist_step__ = object()

    holder = StepOutputsHolder()
    scenario = DummyScenario(cache_hit=True)
    coupler = DummyCoupler()
    manifest_path = tmp_path / "manifest.json"
    manifest_config = ManifestConfig(path=manifest_path)
    state = SimpleNamespace(year=2018, iteration=0)

    run_manifested_steps(
        stage_name="activity_demand_preprocess",
        steps=[
            StepRef(
                name="activitysim_preprocess",
                step_func=_noop_step,
                input_keys=None,
                inputs=None,
            )
        ],
        outputs_holder=holder,
        manifest_config=manifest_config,
        scenario=scenario,
        state=state,
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        name_suffix="2018_iter0",
        iteration=0,
    )

    assert holder.activitysim_preprocess is not None
    assert coupler.get(ASIM_HOUSEHOLDS_IN) is not None
    holder.activitysim_preprocess.validate()
    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["activitysim_preprocess"]["cache_hit"]


def test_recover_beam_run_outputs_from_cached_run_artifacts(tmp_path, monkeypatch):
    workspace = DummyWorkspace(tmp_path)
    beam_iter_dir = (
        Path(workspace.get_beam_output_dir())
        / "seattle"
        / "year-2018-iteration-0"
        / "ITERS"
        / "it.0"
    )
    linkstats_path = beam_iter_dir / "0.linkstats.parquet"
    plans_path = beam_iter_dir / "0.plans.csv.gz"
    events_path = beam_iter_dir / "0.events.parquet"
    for path in (linkstats_path, plans_path, events_path):
        _write_file(path)

    class DummyTracker:
        def get_run_outputs(self, run_id):
            assert run_id == "cached-run-id"
            return {
                "linkstats_parquet_2018_0": str(linkstats_path),
                "beam_plans_out_2018_0": str(plans_path),
                "events_parquet_2018_0": str(events_path),
            }

    monkeypatch.setattr(
        "pilates.workflows.orchestration.cr.current_tracker",
        lambda: DummyTracker(),
    )

    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="beam_run",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        step_inputs=None,
        cached_outputs={
            LINKSTATS: str(linkstats_path),
            BEAM_PLANS_OUT: str(plans_path),
        },
        run_id="cached-run-id",
    )

    assert outputs is not None
    assert holder.beam_run is not None
    assert holder.beam_run.raw_outputs["events_parquet_2018_0"] == events_path
    assert coupler.get("events_parquet_2018_0") is not None
    assert coupler.get("linkstats_parquet_2018_0") is not None
