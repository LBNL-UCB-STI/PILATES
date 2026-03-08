from pathlib import Path
from types import SimpleNamespace

import yaml

from pilates.activitysim.outputs import ActivitySimPreprocessOutputs
from pilates.beam.outputs import BeamPreprocessOutputs
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_PERSONS_IN,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
)
from pilates.workflows.orchestration import (
    ManifestConfig,
    StepRef,
    _recover_cached_outputs,
    _update_coupler_from_outputs,
    run_manifested_steps,
)
from pilates.workflows.outputs_base import serialize_step_outputs
from pilates.workflows.steps import StepOutputsHolder


class DummyCoupler:
    def __init__(self) -> None:
        self.values = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value) -> None:
        self.values[key] = value

    def update(self, mapping) -> None:
        self.values.update(mapping)


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


class DummyScenario:
    def __init__(self, *, cache_hit: bool = False) -> None:
        self.cache_hit = cache_hit
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.cache_hit:
            return SimpleNamespace(cache_hit=True)

        fn = kwargs["fn"]
        runtime_kwargs = kwargs["execution_options"].runtime_kwargs
        fn(**runtime_kwargs)
        return SimpleNamespace(cache_hit=False)


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def _seed_manifest(manifest_path: Path, step_name: str, outputs) -> None:
    manifest = {
        step_name: {
            "completed_at": "2026-01-01T00:00:00",
            "cache_hit": True,
            "outputs": serialize_step_outputs(outputs),
        }
    }
    manifest_path.write_text(yaml.safe_dump(manifest))


def _snapshot_state(
    *,
    holder: StepOutputsHolder,
    step_name: str,
    coupler: DummyCoupler,
    coupler_keys,
):
    outputs = holder.get_attribute(step_name)
    assert outputs is not None
    mapping = {
        key: str(value)
        for key, value in outputs.to_record_store().to_mapping().items()
    }
    return {
        "type": type(outputs).__name__,
        "serialized": serialize_step_outputs(outputs),
        "mapping": mapping,
        "coupler": {key: str(coupler.get(key)) for key in coupler_keys},
    }


def _run_step_mode(
    *,
    step_name: str,
    mode_label: str,
    step_func,
    workspace: DummyWorkspace,
    settings,
    state,
    coupler_keys,
    holder=None,
    coupler=None,
    holder_seed=None,
    manifest_outputs=None,
    step_inputs=None,
    cache_hit: bool = False,
):
    holder = holder or StepOutputsHolder()
    coupler = coupler or DummyCoupler()
    scenario = DummyScenario(cache_hit=cache_hit)
    if holder_seed is not None:
        holder_seed(holder)
    manifest_path = Path(workspace.full_path) / f"{step_name}_{mode_label}.yaml"

    if manifest_outputs is not None:
        _seed_manifest(manifest_path, step_name, manifest_outputs)

    run_manifested_steps(
        stage_name=f"{step_name}_parity",
        steps=[
            StepRef(
                name=step_name,
                step_func=step_func,
                inputs=step_inputs,
            )
        ],
        outputs_holder=holder,
        manifest_config=ManifestConfig(path=manifest_path),
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        name_suffix="parity",
        iteration=getattr(state, "iteration", 0),
    )

    return {
        "scenario": scenario,
        "snapshot": _snapshot_state(
            holder=holder,
            step_name=step_name,
            coupler=coupler,
            coupler_keys=coupler_keys,
        ),
    }


def test_activitysim_preprocess_downstream_state_matches_across_fresh_cache_and_manifest(
    tmp_path,
):
    workspace = DummyWorkspace(tmp_path)
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    households = asim_dir / "households.csv"
    persons = asim_dir / "persons.csv"
    land_use = asim_dir / "land_use.csv"
    for path in (households, persons, land_use):
        _write_file(path)

    settings = SimpleNamespace()
    state = SimpleNamespace(year=2018, iteration=0)
    coupler_keys = [ASIM_HOUSEHOLDS_IN, ASIM_PERSONS_IN, ASIM_LAND_USE_IN]

    fresh_holder = StepOutputsHolder()
    fresh_coupler = DummyCoupler()
    fresh_outputs = ActivitySimPreprocessOutputs(
        mutable_data_dir=asim_dir,
        land_use_table=land_use,
        households_table=households,
        persons_table=persons,
    )

    def _fresh_step(**_runtime_kwargs):
        fresh_holder.set_attribute("activitysim_preprocess", fresh_outputs)
        _update_coupler_from_outputs(
            fresh_outputs,
            coupler=fresh_coupler,
            workspace=workspace,
        )

    _fresh_step.__consist_step__ = object()

    fresh_result = _run_step_mode(
        step_name="activitysim_preprocess",
        mode_label="fresh",
        step_func=_fresh_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        holder=fresh_holder,
        coupler=fresh_coupler,
    )
    fresh_snapshot = fresh_result["snapshot"]

    def _should_not_run(**_runtime_kwargs):
        raise AssertionError("cache-hit and manifest paths should not execute the step")

    _should_not_run.__consist_step__ = object()

    cache_result = _run_step_mode(
        step_name="activitysim_preprocess",
        mode_label="cache",
        step_func=_should_not_run,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        cache_hit=True,
    )

    manifest_outputs = _recover_cached_outputs(
        step_name="activitysim_preprocess",
        outputs_holder=StepOutputsHolder(),
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=DummyCoupler(),
        step_inputs=None,
    )
    assert manifest_outputs is not None
    manifest_result = _run_step_mode(
        step_name="activitysim_preprocess",
        mode_label="manifest",
        step_func=_should_not_run,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        manifest_outputs=manifest_outputs,
    )

    assert len(fresh_result["scenario"].calls) == 1
    assert len(cache_result["scenario"].calls) == 1
    assert manifest_result["scenario"].calls == []
    assert cache_result["snapshot"] == fresh_snapshot
    assert manifest_result["snapshot"] == fresh_snapshot


def test_beam_preprocess_downstream_state_matches_across_fresh_cache_and_manifest(
    tmp_path,
):
    workspace = DummyWorkspace(tmp_path)
    beam_dir = Path(workspace.get_beam_mutable_data_dir())
    plans = beam_dir / "seattle" / "urbansim" / "plans.parquet"
    households = beam_dir / "seattle" / "urbansim" / "households.parquet"
    persons = beam_dir / "seattle" / "urbansim" / "persons.parquet"
    warmstart = beam_dir / "seattle" / "r5" / "init.linkstats.csv.gz"
    for path in (plans, households, persons, warmstart):
        _write_file(path)

    settings = SimpleNamespace()
    state = SimpleNamespace(year=2018, iteration=0)
    step_inputs = {
        BEAM_PLANS_IN: str(plans),
        BEAM_HOUSEHOLDS_IN: str(households),
        BEAM_PERSONS_IN: str(persons),
        LINKSTATS_WARMSTART: str(warmstart),
    }
    coupler_keys = [
        BEAM_PLANS_IN,
        BEAM_HOUSEHOLDS_IN,
        BEAM_PERSONS_IN,
        LINKSTATS_WARMSTART,
    ]

    fresh_holder = StepOutputsHolder()
    fresh_coupler = DummyCoupler()
    fresh_outputs = BeamPreprocessOutputs(
        beam_mutable_data_dir=beam_dir,
        prepared_inputs={
            BEAM_PLANS_IN: plans,
            BEAM_HOUSEHOLDS_IN: households,
            BEAM_PERSONS_IN: persons,
            LINKSTATS_WARMSTART: warmstart,
        },
    )

    def _fresh_step(**_runtime_kwargs):
        fresh_holder.set_attribute("beam_preprocess", fresh_outputs)
        _update_coupler_from_outputs(
            fresh_outputs,
            coupler=fresh_coupler,
            workspace=workspace,
        )

    _fresh_step.__consist_step__ = object()

    fresh_result = _run_step_mode(
        step_name="beam_preprocess",
        mode_label="fresh",
        step_func=_fresh_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        holder=fresh_holder,
        coupler=fresh_coupler,
        holder_seed=lambda holder: holder.set_attribute(
            "activitysim_postprocess",
            object(),
        ),
        step_inputs=step_inputs,
    )
    fresh_snapshot = fresh_result["snapshot"]

    def _should_not_run(**_runtime_kwargs):
        raise AssertionError("cache-hit and manifest paths should not execute the step")

    _should_not_run.__consist_step__ = object()

    cache_result = _run_step_mode(
        step_name="beam_preprocess",
        mode_label="cache",
        step_func=_should_not_run,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        holder_seed=lambda holder: holder.set_attribute(
            "activitysim_postprocess",
            object(),
        ),
        step_inputs=step_inputs,
        cache_hit=True,
    )

    manifest_outputs = _recover_cached_outputs(
        step_name="beam_preprocess",
        outputs_holder=StepOutputsHolder(),
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=DummyCoupler(),
        step_inputs=step_inputs,
    )
    assert manifest_outputs is not None
    manifest_result = _run_step_mode(
        step_name="beam_preprocess",
        mode_label="manifest",
        step_func=_should_not_run,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        holder_seed=lambda holder: holder.set_attribute(
            "activitysim_postprocess",
            object(),
        ),
        step_inputs=step_inputs,
        manifest_outputs=manifest_outputs,
    )

    assert len(fresh_result["scenario"].calls) == 1
    assert len(cache_result["scenario"].calls) == 1
    assert manifest_result["scenario"].calls == []
    assert cache_result["snapshot"] == fresh_snapshot
    assert manifest_result["snapshot"] == fresh_snapshot
