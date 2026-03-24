from pathlib import Path
from types import SimpleNamespace

import yaml
import pytest
from consist import define_step
from consist.types import BindingResult

from pilates.activitysim.outputs import (
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
    write_asim_run_marker,
)
from pilates.beam.outputs import BeamPreprocessOutputs, BeamRunOutputs
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_SHARROW_CACHE_DIR,
    ASIM_PERSONS_IN,
    BEAM_CONFIG_FILE,
    USIM_DATASTORE_BASE_H5,
    USIM_INPUT_NEXT,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PLANS_OUT,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
)
from pilates.workflows.binding import build_binding_plan
from pilates.workflows.orchestration import (
    ManifestConfig,
    StepRef,
    _recover_cached_outputs,
    _recover_step_outputs,
    _update_coupler_from_outputs,
    run_manifested_steps,
    run_workflow,
)
from pilates.workflows.outputs_base import serialize_step_outputs
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_postprocess_step,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
    make_beam_run_step,
)


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

    def get_asim_output_dir(self) -> str:
        return str(self._root / "activitysim" / "output")

    def get_usim_mutable_data_dir(self) -> str:
        return str(self._root / "urbansim" / "data")

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


class _BindingEnvelope:
    def __init__(self, inputs):
        self.inputs = inputs

    def to_binding_result(self) -> BindingResult:
        return BindingResult(inputs=self.inputs)


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def _recover_manifest_outputs(
    *,
    step_name: str,
    step_func,
    workspace: DummyWorkspace,
    settings,
    state,
    holder=None,
    coupler=None,
    step_inputs=None,
):
    holder = holder or StepOutputsHolder()
    coupler = coupler or DummyCoupler()
    return _recover_step_outputs(
        step_name=step_name,
        step_func=step_func,
        outputs_holder=holder,
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
        step_inputs=step_inputs,
        cached_outputs=None,
        run_id=None,
        publish_outputs=True,
    )


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


def _build_step_binding(
    *,
    step_name: str,
    coupler: DummyCoupler,
    settings,
    state,
    workspace: DummyWorkspace,
):
    return build_binding_plan(
        step_name=step_name,
        coupler=coupler,
        settings=settings,
        state=state,
        workspace=workspace,
    )


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
    binding=None,
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
                binding=binding,
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


def test_run_workflow_cache_recovery_uses_binding_inputs_for_non_manifested_steps(
    monkeypatch, tmp_path
):
    captured = {}

    @define_step(model="dummy_binding_step")
    def _dummy_step(settings, state, workspace):
        return None

    def _fake_recover_step_outputs(*, step_inputs=None, **_kwargs):
        captured["step_inputs"] = step_inputs
        return object()

    monkeypatch.setattr(
        "pilates.workflows.orchestration._recover_step_outputs",
        _fake_recover_step_outputs,
    )

    run_workflow(
        stage_name="dummy_stage",
        steps=[
            StepRef(
                name="dummy_binding_step",
                step_func=_dummy_step,
                binding=_BindingEnvelope({"artifact_a": "value-a"}),
            )
        ],
        scenario=DummyScenario(cache_hit=True),
        state=SimpleNamespace(year=2018, iteration=0),
        settings=SimpleNamespace(),
        workspace=DummyWorkspace(tmp_path),
        coupler=DummyCoupler(),
        outputs_holder=StepOutputsHolder(),
        name_suffix="dummy",
    )

    assert captured["step_inputs"] == {"artifact_a": "value-a"}


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
    state = SimpleNamespace(year=2018, forecast_year=2018, iteration=0)
    coupler_keys = [ASIM_HOUSEHOLDS_IN, ASIM_PERSONS_IN, ASIM_LAND_USE_IN]

    fresh_holder = StepOutputsHolder()
    fresh_coupler = DummyCoupler()
    fresh_runtime_step = make_activitysim_preprocess_step(
        coupler=fresh_coupler,
        outputs_holder=fresh_holder,
    )
    fresh_outputs = ActivitySimPreprocessOutputs(
        mutable_data_dir=asim_dir,
        land_use_table=land_use,
        households_table=households,
        persons_table=persons,
    )

    def _fresh_step(**_runtime_kwargs):
        fresh_holder.set_attribute("activitysim_preprocess", fresh_outputs)
        fresh_runtime_step.pilates_output_replayer(
            fresh_outputs,
            settings,
            state,
            workspace,
            fresh_holder,
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

    cache_holder = StepOutputsHolder()
    cache_coupler = DummyCoupler()
    cache_step = make_activitysim_preprocess_step(
        coupler=cache_coupler,
        outputs_holder=cache_holder,
    )
    cache_result = _run_step_mode(
        step_name="activitysim_preprocess",
        mode_label="cache",
        step_func=cache_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        holder=cache_holder,
        coupler=cache_coupler,
        cache_hit=True,
    )

    manifest_holder = StepOutputsHolder()
    manifest_coupler = DummyCoupler()
    manifest_step = make_activitysim_preprocess_step(
        coupler=manifest_coupler,
        outputs_holder=manifest_holder,
    )
    manifest_outputs = _recover_manifest_outputs(
        step_name="activitysim_preprocess",
        step_func=manifest_step,
        holder=manifest_holder,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler=manifest_coupler,
    )
    assert manifest_outputs is not None
    manifest_result = _run_step_mode(
        step_name="activitysim_preprocess",
        mode_label="manifest",
        step_func=manifest_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        holder=StepOutputsHolder(),
        coupler=manifest_coupler,
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
    state = SimpleNamespace(year=2018, forecast_year=2018, iteration=0)
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


def test_activitysim_run_downstream_state_matches_across_fresh_cache_and_manifest(
    tmp_path,
):
    workspace = DummyWorkspace(tmp_path)
    asim_output_dir = Path(workspace.get_asim_output_dir())
    final_pipeline = asim_output_dir / "final_pipeline"
    households = final_pipeline / "households" / "final.parquet"
    persons = final_pipeline / "persons" / "final.parquet"
    beam_plans = final_pipeline / "beam_plans" / "final.parquet"
    for path in (households, persons, beam_plans):
        _write_file(path)
    write_asim_run_marker(asim_output_dir, year=2018, iteration=0)

    settings = SimpleNamespace()
    state = SimpleNamespace(year=2018, forecast_year=2018, iteration=0)
    coupler_keys = [
        "households_asim_out_temp",
        "persons_asim_out_temp",
        "beam_plans_asim_out_temp",
    ]

    upstream_preprocess = ActivitySimPreprocessOutputs(
        mutable_data_dir=Path(workspace.get_asim_mutable_data_dir()),
        land_use_table=Path(workspace.get_asim_mutable_data_dir()) / "land_use.csv",
        households_table=Path(workspace.get_asim_mutable_data_dir()) / "households.csv",
        persons_table=Path(workspace.get_asim_mutable_data_dir()) / "persons.csv",
    )
    for path in (
        upstream_preprocess.land_use_table,
        upstream_preprocess.households_table,
        upstream_preprocess.persons_table,
    ):
        _write_file(path)

    fresh_holder = StepOutputsHolder()
    fresh_coupler = DummyCoupler()
    fresh_runtime_step = make_activitysim_run_step(
        coupler=fresh_coupler,
        outputs_holder=fresh_holder,
    )
    fresh_outputs = ActivitySimRunOutputs(
        output_dir=asim_output_dir,
        raw_outputs={
            "households_asim_out_temp": households,
            "persons_asim_out_temp": persons,
            "beam_plans_asim_out_temp": beam_plans,
        },
    )

    def _fresh_step(**_runtime_kwargs):
        fresh_holder.set_attribute("activitysim_run", fresh_outputs)
        fresh_runtime_step.pilates_output_replayer(
            fresh_outputs,
            settings,
            state,
            workspace,
            fresh_holder,
        )

    _fresh_step.__consist_step__ = object()

    fresh_result = _run_step_mode(
        step_name="activitysim_run",
        mode_label="fresh",
        step_func=_fresh_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        holder=fresh_holder,
        coupler=fresh_coupler,
        holder_seed=lambda holder: holder.set_attribute(
            "activitysim_preprocess",
            upstream_preprocess,
        ),
    )
    fresh_snapshot = fresh_result["snapshot"]

    cache_holder = StepOutputsHolder()
    cache_coupler = DummyCoupler()
    cache_step = make_activitysim_run_step(
        coupler=cache_coupler,
        outputs_holder=cache_holder,
    )
    cache_result = _run_step_mode(
        step_name="activitysim_run",
        mode_label="cache",
        step_func=cache_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        holder=cache_holder,
        coupler=cache_coupler,
        holder_seed=lambda holder: holder.set_attribute(
            "activitysim_preprocess",
            upstream_preprocess,
        ),
        cache_hit=True,
    )

    manifest_holder = StepOutputsHolder()
    manifest_holder.set_attribute("activitysim_preprocess", upstream_preprocess)
    manifest_coupler = DummyCoupler()
    manifest_step = make_activitysim_run_step(
        coupler=manifest_coupler,
        outputs_holder=manifest_holder,
    )
    manifest_outputs = _recover_manifest_outputs(
        step_name="activitysim_run",
        step_func=manifest_step,
        holder=manifest_holder,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler=manifest_coupler,
    )
    assert manifest_outputs is not None
    manifest_result = _run_step_mode(
        step_name="activitysim_run",
        mode_label="manifest",
        step_func=manifest_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        coupler=manifest_coupler,
        holder_seed=lambda holder: holder.set_attribute(
            "activitysim_preprocess",
            upstream_preprocess,
        ),
        manifest_outputs=manifest_outputs,
    )

    assert len(fresh_result["scenario"].calls) == 1
    assert len(cache_result["scenario"].calls) == 1
    assert manifest_result["scenario"].calls == []
    assert cache_result["snapshot"] == fresh_snapshot
    assert manifest_result["snapshot"] == fresh_snapshot


@pytest.mark.parametrize(
    ("cache_present", "expects_optional_key"),
    [(False, False), (True, True)],
)
def test_activitysim_run_binding_tracks_optional_sharrow_cache_dir(
    tmp_path,
    cache_present,
    expects_optional_key,
):
    workspace = DummyWorkspace(tmp_path)
    asim_output_dir = Path(workspace.get_asim_output_dir())
    final_pipeline = asim_output_dir / "final_pipeline"
    households = final_pipeline / "households" / "final.parquet"
    persons = final_pipeline / "persons" / "final.parquet"
    beam_plans = final_pipeline / "beam_plans" / "final.parquet"
    for path in (households, persons, beam_plans):
        _write_file(path)
    write_asim_run_marker(asim_output_dir, year=2018, iteration=0)

    settings = SimpleNamespace()
    state = SimpleNamespace(year=2018, forecast_year=2018, iteration=0)
    coupler_keys = [
        "households_asim_out_temp",
        "persons_asim_out_temp",
        "beam_plans_asim_out_temp",
    ]

    upstream_preprocess = ActivitySimPreprocessOutputs(
        mutable_data_dir=Path(workspace.get_asim_mutable_data_dir()),
        land_use_table=Path(workspace.get_asim_mutable_data_dir()) / "land_use.csv",
        households_table=Path(workspace.get_asim_mutable_data_dir()) / "households.csv",
        persons_table=Path(workspace.get_asim_mutable_data_dir()) / "persons.csv",
    )
    for path in (
        upstream_preprocess.land_use_table,
        upstream_preprocess.households_table,
        upstream_preprocess.persons_table,
    ):
        _write_file(path)

    binding_coupler = DummyCoupler()
    binding_coupler.set(ASIM_LAND_USE_IN, str(upstream_preprocess.land_use_table))
    binding_coupler.set(ASIM_HOUSEHOLDS_IN, str(upstream_preprocess.households_table))
    binding_coupler.set(ASIM_PERSONS_IN, str(upstream_preprocess.persons_table))
    zarr_path = Path(workspace.get_asim_output_dir()) / "cache" / "skims.zarr"
    _write_file(zarr_path)
    binding_coupler.set(ZARR_SKIMS, str(zarr_path))
    cache_dir = Path(workspace.full_path) / "shared_cache" / "numba"
    if cache_present:
        _write_file(cache_dir / "entry.bin")
        binding_coupler.set(ASIM_SHARROW_CACHE_DIR, str(cache_dir))

    binding = _build_step_binding(
        step_name="activitysim_run",
        coupler=binding_coupler,
        settings=settings,
        state=state,
        workspace=workspace,
    )
    assert binding.missing_required == []

    fresh_holder = StepOutputsHolder()
    fresh_coupler = DummyCoupler()
    fresh_runtime_step = make_activitysim_run_step(
        coupler=fresh_coupler,
        outputs_holder=fresh_holder,
    )
    fresh_outputs = ActivitySimRunOutputs(
        output_dir=asim_output_dir,
        raw_outputs={
            "households_asim_out_temp": households,
            "persons_asim_out_temp": persons,
            "beam_plans_asim_out_temp": beam_plans,
        },
    )

    def _fresh_step(**_runtime_kwargs):
        fresh_holder.set_attribute("activitysim_run", fresh_outputs)
        fresh_runtime_step.pilates_output_replayer(
            fresh_outputs,
            settings,
            state,
            workspace,
            fresh_holder,
        )

    _fresh_step.__consist_step__ = object()

    cache_holder = StepOutputsHolder()
    cache_coupler = DummyCoupler()
    cache_step = make_activitysim_run_step(
        coupler=cache_coupler,
        outputs_holder=cache_holder,
    )

    manifest_holder = StepOutputsHolder()
    manifest_holder.set_attribute("activitysim_preprocess", upstream_preprocess)
    manifest_coupler = DummyCoupler()
    manifest_step = make_activitysim_run_step(
        coupler=manifest_coupler,
        outputs_holder=manifest_holder,
    )
    manifest_outputs = _recover_manifest_outputs(
        step_name="activitysim_run",
        step_func=manifest_step,
        holder=manifest_holder,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler=manifest_coupler,
    )
    assert manifest_outputs is not None

    fresh_result = _run_step_mode(
        step_name="activitysim_run",
        mode_label="fresh",
        step_func=_fresh_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        holder=fresh_holder,
        coupler=fresh_coupler,
        holder_seed=lambda holder: holder.set_attribute(
            "activitysim_preprocess",
            upstream_preprocess,
        ),
        binding=binding,
    )
    cache_result = _run_step_mode(
        step_name="activitysim_run",
        mode_label="cache",
        step_func=cache_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        holder=cache_holder,
        coupler=cache_coupler,
        holder_seed=lambda holder: holder.set_attribute(
            "activitysim_preprocess",
            upstream_preprocess,
        ),
        binding=binding,
        cache_hit=True,
    )
    manifest_result = _run_step_mode(
        step_name="activitysim_run",
        mode_label="manifest",
        step_func=manifest_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        coupler=manifest_coupler,
        holder_seed=lambda holder: holder.set_attribute(
            "activitysim_preprocess",
            upstream_preprocess,
        ),
        binding=binding,
        manifest_outputs=manifest_outputs,
    )

    assert len(fresh_result["scenario"].calls) == 1
    assert len(cache_result["scenario"].calls) == 1
    assert manifest_result["scenario"].calls == []
    fresh_binding = fresh_result["scenario"].calls[0]["binding"]
    cache_binding = cache_result["scenario"].calls[0]["binding"]
    assert ASIM_SHARROW_CACHE_DIR not in (fresh_binding.input_keys or [])
    assert ASIM_SHARROW_CACHE_DIR not in (cache_binding.input_keys or [])
    if expects_optional_key:
        assert ASIM_SHARROW_CACHE_DIR in (fresh_binding.optional_input_keys or [])
        assert ASIM_SHARROW_CACHE_DIR in (cache_binding.optional_input_keys or [])
    else:
        assert ASIM_SHARROW_CACHE_DIR not in (fresh_binding.optional_input_keys or [])
        assert ASIM_SHARROW_CACHE_DIR not in (cache_binding.optional_input_keys or [])


@pytest.mark.parametrize(
    ("warmstart_present", "expects_optional_key"),
    [(False, False), (True, True)],
)
def test_beam_run_binding_tracks_optional_warmstart(
    tmp_path,
    warmstart_present,
    expects_optional_key,
):
    workspace = DummyWorkspace(tmp_path)
    beam_dir = Path(workspace.get_beam_mutable_data_dir())
    config_path = beam_dir / "seattle" / "beam.conf"
    plans = beam_dir / "seattle" / "urbansim" / "plans.parquet"
    households = beam_dir / "seattle" / "urbansim" / "households.parquet"
    persons = beam_dir / "seattle" / "urbansim" / "persons.parquet"
    for path in (config_path, plans, households, persons):
        _write_file(path)

    warmstart = beam_dir / "seattle" / "r5" / "init.linkstats.csv.gz"
    if warmstart_present:
        _write_file(warmstart)

    beam_preprocess_outputs = BeamPreprocessOutputs(
        beam_mutable_data_dir=beam_dir,
        prepared_inputs={
            BEAM_PLANS_IN: plans,
            BEAM_HOUSEHOLDS_IN: households,
            BEAM_PERSONS_IN: persons,
            **({LINKSTATS_WARMSTART: warmstart} if warmstart_present else {}),
        },
    )

    settings = SimpleNamespace()
    state = SimpleNamespace(year=2018, forecast_year=2018, iteration=0)
    coupler_keys = [
        "linkstats",
        "beam_plans_out",
    ]

    binding_coupler = DummyCoupler()
    binding_coupler.set(BEAM_CONFIG_FILE, str(config_path))
    binding_coupler.set(BEAM_PLANS_IN, str(plans))
    binding_coupler.set(BEAM_HOUSEHOLDS_IN, str(households))
    binding_coupler.set(BEAM_PERSONS_IN, str(persons))
    if warmstart_present:
        binding_coupler.set(LINKSTATS_WARMSTART, str(warmstart))

    binding = _build_step_binding(
        step_name="beam_run",
        coupler=binding_coupler,
        settings=settings,
        state=state,
        workspace=workspace,
    )
    assert binding.missing_required == []

    beam_output_dir = Path(workspace.full_path) / "beam" / "output"
    linkstats = beam_output_dir / "linkstats_2018_0.csv.gz"
    beam_plans = beam_output_dir / "beam_plans_2018_0.parquet"
    _write_file(linkstats)
    _write_file(beam_plans)
    fresh_outputs = BeamRunOutputs(
        beam_output_dir=beam_output_dir,
        raw_outputs={
            LINKSTATS: linkstats,
            BEAM_PLANS_OUT: beam_plans,
        },
    )

    fresh_holder = StepOutputsHolder()
    fresh_coupler = DummyCoupler()
    fresh_runtime_step = make_beam_run_step(
        coupler=fresh_coupler,
        outputs_holder=fresh_holder,
    )

    def _fresh_step(**_runtime_kwargs):
        fresh_holder.set_attribute("beam_run", fresh_outputs)
        fresh_runtime_step.pilates_output_replayer(
            fresh_outputs,
            settings,
            state,
            workspace,
            fresh_holder,
        )

    _fresh_step.__consist_step__ = object()

    manifest_holder = StepOutputsHolder()
    manifest_coupler = DummyCoupler()
    manifest_step = make_beam_run_step(
        coupler=manifest_coupler,
        outputs_holder=manifest_holder,
    )

    fresh_result = _run_step_mode(
        step_name="beam_run",
        mode_label="fresh",
        step_func=_fresh_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        holder=fresh_holder,
        coupler=fresh_coupler,
        holder_seed=lambda holder: holder.set_attribute(
            "beam_preprocess",
            beam_preprocess_outputs,
        ),
        binding=binding,
    )
    manifest_result = _run_step_mode(
        step_name="beam_run",
        mode_label="manifest",
        step_func=manifest_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        coupler=manifest_coupler,
        holder_seed=lambda holder: holder.set_attribute(
            "beam_preprocess",
            beam_preprocess_outputs,
        ),
        binding=binding,
        manifest_outputs=fresh_outputs,
    )

    assert len(fresh_result["scenario"].calls) == 1
    assert manifest_result["scenario"].calls == []
    fresh_binding = fresh_result["scenario"].calls[0]["binding"]
    assert LINKSTATS_WARMSTART not in (fresh_binding.input_keys or [])
    if expects_optional_key:
        assert LINKSTATS_WARMSTART in (fresh_binding.optional_input_keys or [])
    else:
        assert LINKSTATS_WARMSTART not in (fresh_binding.optional_input_keys or [])


def test_activitysim_postprocess_downstream_state_matches_across_fresh_cache_and_manifest(
    tmp_path,
):
    workspace = DummyWorkspace(tmp_path)
    asim_output_dir = Path(workspace.get_asim_output_dir())
    iter_dir = asim_output_dir / "year-2018-iteration-0"
    households = iter_dir / "households.parquet"
    persons = iter_dir / "persons.parquet"
    beam_plans = iter_dir / "beam_plans.parquet"
    for path in (households, persons, beam_plans):
        _write_file(path)

    inputs_dir = asim_output_dir / "inputs-year-2018-iteration-0"
    archived_households = inputs_dir / "households.csv"
    archived_persons = inputs_dir / "persons.csv"
    archived_land_use = inputs_dir / "land_use.csv"
    archived_skims = inputs_dir / "skims.omx"
    archived_zarr = inputs_dir / "skims.zarr"
    for path in (
        archived_households,
        archived_persons,
        archived_land_use,
        archived_skims,
        archived_zarr,
    ):
        _write_file(path)

    usim_h5 = Path(workspace.get_usim_mutable_data_dir()) / "usim_2018.h5"
    _write_file(usim_h5)

    settings = SimpleNamespace(
        urbansim=SimpleNamespace(region_id="000", region_mappings={"region_to_region_id": {}}),
        run=SimpleNamespace(region="test"),
    )
    state = SimpleNamespace(year=2018, forecast_year=2018, iteration=0)
    coupler_keys = [
        "households_asim_out",
        "persons_asim_out",
        "beam_plans_asim_out",
        "asim_input_households_csv_archived",
        "asim_input_persons_csv_archived",
        "asim_input_land_use_csv_archived",
        "asim_input_skims_omx_archived",
        "asim_input_skims_zarr_archived",
        USIM_INPUT_NEXT,
    ]

    fresh_holder = StepOutputsHolder()
    fresh_coupler = DummyCoupler()
    fresh_runtime_step = make_activitysim_postprocess_step(
        coupler=fresh_coupler,
        outputs_holder=fresh_holder,
    )
    fresh_outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=usim_h5,
        asim_output_dir=asim_output_dir,
        processed_outputs={
            "households_asim_out": households,
            "persons_asim_out": persons,
            "beam_plans_asim_out": beam_plans,
            "asim_input_households_csv_archived": archived_households,
            "asim_input_persons_csv_archived": archived_persons,
            "asim_input_land_use_csv_archived": archived_land_use,
            "asim_input_skims_omx_archived": archived_skims,
            "asim_input_skims_zarr_archived": archived_zarr,
        },
        usim_datastore_key=USIM_INPUT_NEXT,
    )

    def _fresh_step(**_runtime_kwargs):
        fresh_holder.set_attribute("activitysim_postprocess", fresh_outputs)
        fresh_runtime_step.pilates_output_replayer(
            fresh_outputs,
            settings,
            state,
            workspace,
            fresh_holder,
        )

    _fresh_step.__consist_step__ = object()

    fresh_result = _run_step_mode(
        step_name="activitysim_postprocess",
        mode_label="fresh",
        step_func=_fresh_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        holder=fresh_holder,
        coupler=fresh_coupler,
        holder_seed=lambda holder: holder.set_attribute(
            "activitysim_run",
            object(),
        ),
    )
    fresh_snapshot = fresh_result["snapshot"]

    cache_holder = StepOutputsHolder()
    cache_coupler = DummyCoupler()
    cache_step = make_activitysim_postprocess_step(
        coupler=cache_coupler,
        outputs_holder=cache_holder,
    )
    cache_result = _run_step_mode(
        step_name="activitysim_postprocess",
        mode_label="cache",
        step_func=cache_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        holder=cache_holder,
        coupler=cache_coupler,
        holder_seed=lambda holder: holder.set_attribute(
            "activitysim_run",
            object(),
        ),
        step_inputs={USIM_DATASTORE_BASE_H5: str(usim_h5)},
        cache_hit=True,
    )

    manifest_holder = StepOutputsHolder()
    manifest_coupler = DummyCoupler()
    manifest_step = make_activitysim_postprocess_step(
        coupler=manifest_coupler,
        outputs_holder=manifest_holder,
    )
    manifest_outputs = _recover_manifest_outputs(
        step_name="activitysim_postprocess",
        step_func=manifest_step,
        holder=manifest_holder,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler=manifest_coupler,
        step_inputs={USIM_DATASTORE_BASE_H5: str(usim_h5)},
    )
    assert manifest_outputs is not None
    manifest_result = _run_step_mode(
        step_name="activitysim_postprocess",
        mode_label="manifest",
        step_func=manifest_step,
        workspace=workspace,
        settings=settings,
        state=state,
        coupler_keys=coupler_keys,
        coupler=manifest_coupler,
        holder_seed=lambda holder: holder.set_attribute(
            "activitysim_run",
            object(),
        ),
        step_inputs={USIM_DATASTORE_BASE_H5: str(usim_h5)},
        manifest_outputs=manifest_outputs,
    )

    assert len(fresh_result["scenario"].calls) == 1
    assert len(cache_result["scenario"].calls) == 1
    assert manifest_result["scenario"].calls == []
    assert cache_result["snapshot"] == fresh_snapshot
    assert manifest_result["snapshot"] == fresh_snapshot
