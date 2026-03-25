"""
Cross-cutting workflow invariant tests.

This file is intended as executable architecture documentation for three core
workflow guarantees:

1. Declared step outputs are represented in the coupler schema.
2. Query-critical BEAM outputs carry structured facet metadata.
3. Manifest restore paths preserve both outputs-holder and coupler state.

The tests avoid running heavy model components and instead focus on the
orchestration contracts that must stay stable across refactors.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from consist import define_step
import yaml

from pilates.activitysim.outputs import (
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.atlas.outputs import AtlasPreprocessOutputs, AtlasRunOutputs
from pilates.beam.outputs import BeamPostprocessOutputs, BeamRunOutputs
from pilates.workflows import steps
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_PERSONS_IN,
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_PLANS_XML,
    BEAM_PLANS_OUT,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_INPUT_NEXT,
)
from pilates.workflows.coupler_schema import build_coupler_schema
from pilates.workflows.orchestration import (
    ManifestConfig,
    StepRef,
    _recover_step_outputs,
    run_manifested_steps,
)
from pilates.workflows.outputs_base import serialize_step_outputs
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_compile_step,
    make_activitysim_postprocess_step,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
    make_atlas_postprocess_step,
    make_atlas_preprocess_step,
    make_atlas_run_step,
    make_beam_full_skim_step,
    make_beam_postprocess_step,
    make_beam_preprocess_step,
    make_beam_run_step,
    make_urbansim_postprocess_step,
    make_urbansim_preprocess_step,
    make_urbansim_run_step,
)
from pilates.workflows.steps import beam as steps_beam
from pilates.runtime import launcher as run_module


class _SchemaCoupler:
    def get(self, _key, default=None):
        return default

    def set(self, _key, _value):
        return None

    def update(self, _mapping):
        return None

    def set_from_artifact(self, _key, _value):
        return None

    def declare_outputs(self, *args, **kwargs):
        return None


def _declared_schema_steps():
    coupler = _SchemaCoupler()
    holder = StepOutputsHolder()
    return [
        make_urbansim_preprocess_step(coupler=coupler, outputs_holder=holder),
        make_urbansim_run_step(coupler=coupler, outputs_holder=holder),
        make_urbansim_postprocess_step(coupler=coupler, outputs_holder=holder),
        make_atlas_preprocess_step(coupler=coupler, outputs_holder=holder),
        make_atlas_run_step(coupler=coupler, outputs_holder=holder),
        make_atlas_postprocess_step(coupler=coupler, outputs_holder=holder),
        make_activitysim_preprocess_step(coupler=coupler, outputs_holder=holder),
        make_activitysim_compile_step(coupler=coupler, outputs_holder=holder),
        make_activitysim_run_step(coupler=coupler, outputs_holder=holder),
        make_activitysim_postprocess_step(coupler=coupler, outputs_holder=holder),
        make_beam_preprocess_step(coupler=coupler, outputs_holder=holder),
        make_beam_run_step(coupler=coupler, outputs_holder=holder),
        make_beam_postprocess_step(coupler=coupler, outputs_holder=holder),
        make_beam_full_skim_step(coupler=coupler, outputs_holder=holder),
    ]


def test_declared_step_outputs_are_present_in_coupler_schema():
    """
    Document coupler schema completeness.

    Every output a step declares through Consist metadata should have a schema
    entry and description, otherwise startup ``require_outputs`` enforcement and
    downstream coupler expectations can drift silently.
    """
    schema_steps = _declared_schema_steps()
    schema = build_coupler_schema(schema_steps, settings=SimpleNamespace())

    missing = []
    runtime_settings = SimpleNamespace()
    for step_fn in schema_steps:
        meta = step_fn.__consist_step__
        declared = set(
            build_coupler_schema(
                [step_fn],
                settings=runtime_settings,
                include_extras=False,
            ).keys()
        )
        for key in sorted(declared):
            if key not in schema:
                missing.append((meta.model, key))
            else:
                assert schema[key], f"Schema entry for {key!r} should have a description"

    assert not missing, f"Declared step outputs missing from coupler schema: {missing}"


def test_beam_run_output_logger_includes_phys_sim_facets(monkeypatch, tmp_path):
    """
    Document BEAM phys-sim linkstats facet behavior.

    This test captures the BEAM run output logger directly and verifies that a
    phys-sim linkstats key emits queryable facet metadata (family/year/iteration
    and phys-sim/sub-iteration dimensions).
    """
    captured = {}

    def _fake_make_beam_step_function(**kwargs):
        captured["output_logger"] = kwargs["output_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_beam,
        "_make_beam_step_function",
        _fake_make_beam_step_function,
    )

    steps.make_beam_run_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    output_logger = captured["output_logger"]
    calls = []

    def _log_output_only(*, key, path, description, **meta):
        calls.append((key, meta))

    monkeypatch.setattr(steps_beam, "log_output_only", _log_output_only)

    key = "linkstats_unmodified_parquet__y2018__i0__phys_sim_iter3__beam_sub_iter0"
    outputs = BeamRunOutputs(
        beam_output_dir=tmp_path,
        raw_outputs={key: tmp_path / "linkstats.parquet"},
    )

    output_logger(
        outputs,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(),
    )

    meta = next(meta for logged_key, meta in calls if logged_key == key)
    assert meta["facet_schema_version"] == "v1"
    assert meta["facet_index"] is True
    assert meta["facet"]["artifact_family"] == "linkstats_unmodified_phys_sim_iter_parquet"
    assert meta["facet"]["phys_sim_iteration"] == 3
    assert meta["facet"]["beam_sub_iteration"] == 0


def test_beam_postprocess_output_logger_publishes_promoted_run_keys_without_recordstore(
    monkeypatch, tmp_path
):
    captured = {}

    def _fake_make_beam_step_function(**kwargs):
        captured["output_logger"] = kwargs["output_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_beam,
        "_make_beam_step_function",
        _fake_make_beam_step_function,
    )

    steps.make_beam_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    output_logger = captured["output_logger"]
    log_and_set_calls = []
    log_only_calls = []

    def _fake_log_and_set_output(*, key, path, description, coupler, **meta):
        log_and_set_calls.append((key, path, description, meta))

    def _fake_log_output_only(*, key, path, description, **meta):
        log_only_calls.append((key, path, description, meta))

    monkeypatch.setattr(steps_beam, "log_and_set_output", _fake_log_and_set_output)
    monkeypatch.setattr(steps_beam, "log_output_only", _fake_log_output_only)

    zarr_skims = tmp_path / "skims.zarr"
    zarr_skims.write_text("zarr", encoding="utf-8")
    linkstats_iter = tmp_path / "linkstats.csv.gz"
    linkstats_iter.write_text("linkstats", encoding="utf-8")
    linkstats_sub = tmp_path / "linkstats_sub.parquet"
    linkstats_sub.write_text("linkstats-sub", encoding="utf-8")
    phys_sim = tmp_path / "phys_sim.parquet"
    phys_sim.write_text("phys", encoding="utf-8")
    plans_iter = tmp_path / "plans.xml.gz"
    plans_iter.write_text("plans", encoding="utf-8")
    output_plans = tmp_path / "output_plans.xml.gz"
    output_plans.write_text("output-plans", encoding="utf-8")
    experienced_plans = tmp_path / "experienced_plans.xml.gz"
    experienced_plans.write_text("experienced", encoding="utf-8")
    output_experienced_plans = tmp_path / "output_experienced_plans.xml.gz"
    output_experienced_plans.write_text("output-experienced", encoding="utf-8")
    split_event = tmp_path / "event.parquet"
    split_event.write_text("event", encoding="utf-8")
    split_links = tmp_path / "links.parquet"
    split_links.write_text("links", encoding="utf-8")

    upstream = BeamRunOutputs(
        beam_output_dir=tmp_path,
        raw_outputs={
            "linkstats_2018_0": linkstats_iter,
            "linkstats_parquet_2018_0_sub1": linkstats_sub,
            "linkstats_unmodified_parquet__y2018__i0__phys_sim_iter3": phys_sim,
            "beam_plans_out_2018_0": plans_iter,
            "beam_output_plans_xml_2018_0": output_plans,
            "beam_experienced_plans_xml_2018_0": experienced_plans,
            "beam_output_experienced_plans_xml_2018_0": output_experienced_plans,
        },
    )
    outputs = BeamPostprocessOutputs(
        zarr_skims=zarr_skims,
        final_skims_omx=None,
        split_events={"events_parquet_2018_0_type_PathTraversal": split_event},
        split_event_links={"path_traversal_links_2018_0": split_links},
    )

    output_logger(
        outputs,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(beam_run=upstream),
    )

    by_key = {
        key: (path, description, meta)
        for key, path, description, meta in log_and_set_calls
    }
    assert by_key["zarr_skims"][0] == str(zarr_skims)
    assert by_key[LINKSTATS][0] == str(linkstats_iter)
    assert by_key[LINKSTATS_WARMSTART][0] == str(linkstats_iter)
    assert by_key[BEAM_PLANS_OUT][0] == str(plans_iter)
    assert by_key[BEAM_OUTPUT_PLANS_XML][0] == str(output_plans)
    assert by_key[BEAM_EXPERIENCED_PLANS_XML][0] == str(experienced_plans)
    assert (
        by_key[BEAM_OUTPUT_EXPERIENCED_PLANS_XML][0]
        == str(output_experienced_plans)
    )
    assert by_key["linkstats_unmodified_parquet__y2018__i0__phys_sim_iter3"][0] == str(
        phys_sim
    )
    assert "linkstats_parquet_2018_0_sub1" not in by_key
    assert by_key[LINKSTATS][2]["facet"]["artifact_family"] == "linkstats"
    assert by_key[LINKSTATS][2]["facet"]["year"] == 2018
    assert by_key["linkstats_unmodified_parquet__y2018__i0__phys_sim_iter3"][2][
        "facet"
    ]["phys_sim_iteration"] == 3

    log_only_keys = {key for key, _path, _description, _meta in log_only_calls}
    assert "linkstats_parquet_2018_0_sub1" in log_only_keys
    assert "events_parquet_2018_0_type_PathTraversal" in log_only_keys
    assert "path_traversal_links_2018_0" in log_only_keys


class _ManifestWorkspace:
    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def full_path(self) -> str:
        return str(self._root)

    def get_asim_mutable_data_dir(self) -> str:
        return str(self._root / "activitysim" / "data")

    def get_atlas_mutable_input_dir(self) -> str:
        return str(self._root / "atlas" / "atlas_input")

    def get_atlas_output_dir(self) -> str:
        return str(self._root / "atlas" / "atlas_output")


class _ManifestCoupler:
    def __init__(self) -> None:
        self.values = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value

    def update(self, mapping):
        self.values.update(mapping)


class _ManifestScenario:
    def __init__(self, cache_hit: bool) -> None:
        self.cache_hit = cache_hit
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(cache_hit=self.cache_hit)


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def _prepare_activitysim_preprocess_manifest(
    *,
    workspace: _ManifestWorkspace,
    manifest_path: Path,
    mutate_outputs=None,
) -> dict:
    seed_holder = StepOutputsHolder()
    seed_coupler = _ManifestCoupler()
    step_func = make_activitysim_preprocess_step(
        coupler=seed_coupler,
        outputs_holder=seed_holder,
    )
    outputs = _recover_step_outputs(
        step=StepRef(name="activitysim_preprocess", step_func=step_func),
        step_name="activitysim_preprocess",
        outputs_holder=seed_holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2018, iteration=0),
        workspace=workspace,
        coupler=seed_coupler,
        step_inputs=None,
        cached_outputs=None,
        run_id=None,
        publish_outputs=True,
    )
    assert outputs is not None

    serialized = serialize_step_outputs(seed_holder.activitysim_preprocess)
    if mutate_outputs is not None:
        mutate_outputs(serialized)

    manifest = {
        "activitysim_preprocess": {
            "completed_at": "2026-01-01T00:00:00",
            "cache_hit": True,
            "outputs": serialized,
        }
    }
    manifest_path.write_text(yaml.safe_dump(manifest))
    return manifest


def test_manifest_restore_skips_run_and_rehydrates_coupler(tmp_path):
    """
    Document manifest fast-path behavior.

    When a manifest entry is valid, the step should be skipped, outputs should
    be restored, and coupler keys should still be rehydrated so downstream
    consumers observe the same state as a non-skipped execution.
    """
    workspace = _ManifestWorkspace(tmp_path)
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    _write_file(asim_dir / "households.csv")
    _write_file(asim_dir / "persons.csv")
    _write_file(asim_dir / "land_use.csv")

    manifest_path = tmp_path / "manifest.yaml"
    _prepare_activitysim_preprocess_manifest(
        workspace=workspace,
        manifest_path=manifest_path,
    )

    holder = StepOutputsHolder()
    coupler = _ManifestCoupler()
    scenario = _ManifestScenario(cache_hit=False)
    state = SimpleNamespace(year=2018, iteration=0)
    step_func = make_activitysim_preprocess_step(coupler=coupler, outputs_holder=holder)

    run_manifested_steps(
        stage_name="activity_demand_preprocess",
        steps=[
            StepRef(
                name="activitysim_preprocess",
                step_func=step_func,
            )
        ],
        outputs_holder=holder,
        manifest_config=ManifestConfig(path=manifest_path),
        scenario=scenario,
        state=state,
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        name_suffix="2018_i0",
        iteration=0,
    )

    assert scenario.calls == []
    assert holder.activitysim_preprocess is not None
    assert coupler.get(ASIM_HOUSEHOLDS_IN) is not None
    assert coupler.get(ASIM_PERSONS_IN) is not None
    assert coupler.get(ASIM_LAND_USE_IN) is not None


def test_manifest_restore_reseeds_epoch_parent_linkage(tmp_path):
    workspace = _ManifestWorkspace(tmp_path)
    run_path = tmp_path / "run" / "households.parquet"
    _write_file(run_path)
    manifest_path = tmp_path / "manifest_lineage.yaml"
    manifest = {
        "activitysim_run": {
            "completed_at": "2026-01-01T00:00:00",
            "cache_hit": False,
            "run_id": "asim-restored-run",
            "outputs": serialize_step_outputs(
                ActivitySimRunOutputs(
                    output_dir=tmp_path / "run",
                    raw_outputs={"households_asim_out_temp": run_path},
                )
            ),
        }
    }
    manifest_path.write_text(yaml.safe_dump(manifest))

    @define_step(model="activitysim_run")
    def _manifest_step(settings, state, workspace):
        raise AssertionError("manifest restore should skip execution")

    class _UnderlyingScenario:
        def __init__(self) -> None:
            self.calls = []

        def run(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                cache_hit=False, run=SimpleNamespace(id="beam-live-run")
            )

    underlying = _UnderlyingScenario()
    proxy = run_module._EpochTaggingScenarioProxy(
        underlying,
        scenario_id="scenario-alpha",
        seed=777,
    )
    holder = StepOutputsHolder()
    coupler = _ManifestCoupler()
    state = SimpleNamespace(year=2030, forecast_year=2030, iteration=1)

    run_manifested_steps(
        stage_name="activity_demand_run",
        steps=[StepRef(name="activitysim_run", step_func=_manifest_step)],
        outputs_holder=holder,
        manifest_config=ManifestConfig(path=manifest_path),
        scenario=proxy,
        state=state,
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        name_suffix="2030_i1",
        iteration=1,
    )

    proxy.run(model="beam_run", year=2030, iteration=1)

    assert underlying.calls[0]["parent_run_id"] == "asim-restored-run"


def test_stale_manifest_entry_forces_rerun_and_rewrites_outputs(tmp_path):
    """
    Document stale-manifest recovery behavior.

    If serialized outputs reference missing files, orchestration should treat
    the entry as stale, execute the step path (or cache-hit recovery path), and
    rewrite manifest outputs with valid paths.
    """
    workspace = _ManifestWorkspace(tmp_path)
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    _write_file(asim_dir / "households.csv")
    _write_file(asim_dir / "persons.csv")
    _write_file(asim_dir / "land_use.csv")

    manifest_path = tmp_path / "manifest_stale.yaml"

    bad_path = str(tmp_path / "missing_households.csv")

    _prepare_activitysim_preprocess_manifest(
        workspace=workspace,
        manifest_path=manifest_path,
        mutate_outputs=lambda data: data.update({"households_table": bad_path}),
    )

    holder = StepOutputsHolder()
    coupler = _ManifestCoupler()
    scenario = _ManifestScenario(cache_hit=True)
    state = SimpleNamespace(year=2018, iteration=0)
    step_func = make_activitysim_preprocess_step(coupler=coupler, outputs_holder=holder)

    run_manifested_steps(
        stage_name="activity_demand_preprocess",
        steps=[
            StepRef(
                name="activitysim_preprocess",
                step_func=step_func,
            )
        ],
        outputs_holder=holder,
        manifest_config=ManifestConfig(path=manifest_path),
        scenario=scenario,
        state=state,
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        name_suffix="2018_i0",
        iteration=0,
    )

    assert len(scenario.calls) == 1
    assert holder.activitysim_preprocess is not None
    manifest = yaml.safe_load(manifest_path.read_text())
    refreshed_outputs = manifest["activitysim_preprocess"]["outputs"]
    assert refreshed_outputs["households_table"] != bad_path
    assert Path(refreshed_outputs["households_table"]).exists()


def test_stale_manifest_entry_invalidates_downstream_steps(tmp_path):
    """
    A stale upstream manifest entry must invalidate later manifest entries too.

    Otherwise manifest replay can mix freshly rerun upstream outputs with stale
    downstream outputs from an earlier iteration.
    """
    workspace = _ManifestWorkspace(tmp_path)
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    _write_file(asim_dir / "households.csv")
    _write_file(asim_dir / "persons.csv")
    _write_file(asim_dir / "land_use.csv")
    old_run_path = tmp_path / "run-old" / "households.parquet"
    new_run_path = tmp_path / "run-new" / "households.parquet"
    _write_file(old_run_path)
    _write_file(new_run_path)

    stale_preprocess_outputs = ActivitySimPreprocessOutputs(
        mutable_data_dir=asim_dir,
        households_table=tmp_path / "missing_households.csv",
        persons_table=asim_dir / "persons.csv",
        land_use_table=asim_dir / "land_use.csv",
    )
    old_run_outputs = ActivitySimRunOutputs(
        output_dir=tmp_path / "run-old",
        raw_outputs={"households_asim_out_temp": old_run_path},
    )

    manifest_path = tmp_path / "manifest_downstream_stale.yaml"
    manifest = {
        "activitysim_preprocess": {
            "completed_at": "2026-01-01T00:00:00",
            "cache_hit": False,
            "outputs": serialize_step_outputs(stale_preprocess_outputs),
        },
        "activitysim_run": {
            "completed_at": "2026-01-01T00:05:00",
            "cache_hit": False,
            "outputs": serialize_step_outputs(old_run_outputs),
        },
    }
    manifest_path.write_text(yaml.safe_dump(manifest))

    holder = StepOutputsHolder()
    coupler = _ManifestCoupler()
    state = SimpleNamespace(year=2018, iteration=0)

    def _rerun_preprocess(**_kwargs):
        holder.activitysim_preprocess = ActivitySimPreprocessOutputs(
            mutable_data_dir=asim_dir,
            households_table=asim_dir / "households.csv",
            persons_table=asim_dir / "persons.csv",
            land_use_table=asim_dir / "land_use.csv",
        )

    def _rerun_run(**_kwargs):
        holder.activitysim_run = ActivitySimRunOutputs(
            output_dir=tmp_path / "run-new",
            raw_outputs={"households_asim_out_temp": new_run_path},
        )

    for fn, model in (
        (_rerun_preprocess, "activitysim_preprocess"),
        (_rerun_run, "activitysim_run"),
    ):
        fn.__consist_step__ = SimpleNamespace(model=model)

    class _ExecutingManifestScenario:
        def __init__(self) -> None:
            self.calls = []

        def run(self, **kwargs):
            self.calls.append(kwargs)
            kwargs["fn"]()
            return SimpleNamespace(cache_hit=False)

    scenario = _ExecutingManifestScenario()
    run_manifested_steps(
        stage_name="activity_demand",
        steps=[
            StepRef(name="activitysim_preprocess", step_func=_rerun_preprocess),
            StepRef(name="activitysim_run", step_func=_rerun_run),
        ],
        outputs_holder=holder,
        manifest_config=ManifestConfig(path=manifest_path),
        scenario=scenario,
        state=state,
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        name_suffix="2018_i0",
        iteration=0,
    )

    assert [call["fn"].__name__ for call in scenario.calls] == [
        "_rerun_preprocess",
        "_rerun_run",
    ]
    manifest = yaml.safe_load(manifest_path.read_text())
    refreshed_run_outputs = manifest["activitysim_run"]["outputs"]
    assert refreshed_run_outputs["raw_outputs"]["households_asim_out_temp"] == str(
        new_run_path
    )


def test_stale_manifest_entry_invalidates_downstream_steps_across_invocations(tmp_path):
    workspace = _ManifestWorkspace(tmp_path)
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    _write_file(asim_dir / "households.csv")
    _write_file(asim_dir / "persons.csv")
    _write_file(asim_dir / "land_use.csv")
    old_run_path = tmp_path / "run-old" / "households.parquet"
    new_run_path = tmp_path / "run-new" / "households.parquet"
    old_post_path = tmp_path / "post-old" / "households.parquet"
    new_post_path = tmp_path / "post-new" / "households.parquet"
    usim_h5 = tmp_path / "post-new" / "usim_input_next.h5"
    for path in (old_run_path, new_run_path, old_post_path, new_post_path, usim_h5):
        _write_file(path)

    stale_preprocess_outputs = ActivitySimPreprocessOutputs(
        mutable_data_dir=asim_dir,
        households_table=tmp_path / "missing_households.csv",
        persons_table=asim_dir / "persons.csv",
        land_use_table=asim_dir / "land_use.csv",
    )
    old_run_outputs = ActivitySimRunOutputs(
        output_dir=tmp_path / "run-old",
        raw_outputs={"households_asim_out_temp": old_run_path},
    )
    old_postprocess_outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=usim_h5,
        asim_output_dir=tmp_path / "post-old",
        processed_outputs={"households_asim_out": old_post_path},
        usim_datastore_key="usim_datastore_h5",
    )

    manifest_path = tmp_path / "manifest_split_downstream_stale.yaml"
    manifest = {
        "activitysim_preprocess": {
            "completed_at": "2026-01-01T00:00:00",
            "cache_hit": False,
            "outputs": serialize_step_outputs(stale_preprocess_outputs),
        },
        "activitysim_run": {
            "completed_at": "2026-01-01T00:05:00",
            "cache_hit": False,
            "outputs": serialize_step_outputs(old_run_outputs),
        },
        "activitysim_postprocess": {
            "completed_at": "2026-01-01T00:10:00",
            "cache_hit": False,
            "outputs": serialize_step_outputs(old_postprocess_outputs),
        },
    }
    manifest_path.write_text(yaml.safe_dump(manifest))

    holder = StepOutputsHolder()
    coupler = _ManifestCoupler()
    state = SimpleNamespace(year=2018, forecast_year=2018, iteration=0)

    def _rerun_preprocess(**_kwargs):
        holder.activitysim_preprocess = ActivitySimPreprocessOutputs(
            mutable_data_dir=asim_dir,
            households_table=asim_dir / "households.csv",
            persons_table=asim_dir / "persons.csv",
            land_use_table=asim_dir / "land_use.csv",
        )

    def _rerun_run(**_kwargs):
        holder.activitysim_run = ActivitySimRunOutputs(
            output_dir=tmp_path / "run-new",
            raw_outputs={"households_asim_out_temp": new_run_path},
        )

    def _rerun_postprocess(**_kwargs):
        holder.activitysim_postprocess = ActivitySimPostprocessOutputs(
            usim_datastore_h5=usim_h5,
            asim_output_dir=tmp_path / "post-new",
            processed_outputs={"households_asim_out": new_post_path},
            usim_datastore_key="usim_datastore_h5",
        )

    for fn, model in (
        (_rerun_preprocess, "activitysim_preprocess"),
        (_rerun_run, "activitysim_run"),
        (_rerun_postprocess, "activitysim_postprocess"),
    ):
        fn.__consist_step__ = SimpleNamespace(model=model)

    class _ExecutingManifestScenario:
        def __init__(self) -> None:
            self.calls = []

        def run(self, **kwargs):
            self.calls.append(kwargs)
            kwargs["fn"]()
            return SimpleNamespace(cache_hit=False)

    preprocess_scenario = _ExecutingManifestScenario()
    run_manifested_steps(
        stage_name="activity_demand_preprocess",
        steps=[StepRef(name="activitysim_preprocess", step_func=_rerun_preprocess)],
        outputs_holder=holder,
        manifest_config=ManifestConfig(path=manifest_path),
        scenario=preprocess_scenario,
        state=state,
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        name_suffix="2018_i0",
        iteration=0,
    )

    assert [call["fn"].__name__ for call in preprocess_scenario.calls] == [
        "_rerun_preprocess"
    ]
    pruned_manifest = yaml.safe_load(manifest_path.read_text())
    assert "activitysim_run" not in pruned_manifest
    assert "activitysim_postprocess" not in pruned_manifest

    downstream_scenario = _ExecutingManifestScenario()
    run_manifested_steps(
        stage_name="activity_demand_run",
        steps=[
            StepRef(name="activitysim_run", step_func=_rerun_run),
            StepRef(name="activitysim_postprocess", step_func=_rerun_postprocess),
        ],
        outputs_holder=holder,
        manifest_config=ManifestConfig(path=manifest_path),
        scenario=downstream_scenario,
        state=state,
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        name_suffix="2018_i0",
        iteration=0,
    )

    assert [call["fn"].__name__ for call in downstream_scenario.calls] == [
        "_rerun_run",
        "_rerun_postprocess",
    ]
    refreshed_manifest = yaml.safe_load(manifest_path.read_text())
    assert (
        refreshed_manifest["activitysim_run"]["outputs"]["raw_outputs"][
            "households_asim_out_temp"
        ]
        == str(new_run_path)
    )
    assert (
        refreshed_manifest["activitysim_postprocess"]["outputs"]["processed_outputs"][
            "households_asim_out"
        ]
        == str(new_post_path)
    )


def test_manifest_restore_remaps_workspace_rooted_atlas_paths(tmp_path):
    old_root = tmp_path / "old-job" / "pilates-workspace" / "consist-run"
    new_root = tmp_path / "new-job" / "pilates-workspace" / "consist-run"
    workspace = _ManifestWorkspace(new_root)

    current_atlas_input_dir = Path(workspace.get_atlas_mutable_input_dir())
    current_atlas_output_dir = Path(workspace.get_atlas_output_dir())
    current_households_csv = current_atlas_input_dir / "year2023" / "households.csv"
    current_householdv_csv = current_atlas_output_dir / "householdv_2023.csv"
    current_vehicles_csv = current_atlas_output_dir / "vehicles_2023.csv"
    for path in (
        current_households_csv,
        current_householdv_csv,
        current_vehicles_csv,
    ):
        _write_file(path)

    old_atlas_input_dir = old_root / "atlas" / "atlas_input"
    old_atlas_output_dir = old_root / "atlas" / "atlas_output"
    manifest = {
        "atlas_preprocess": {
            "completed_at": "2026-01-01T00:00:00",
            "cache_hit": False,
            "outputs": serialize_step_outputs(
                AtlasPreprocessOutputs(
                    atlas_mutable_input_dir=old_atlas_input_dir,
                    prepared_inputs={
                        "atlas_households_csv": old_atlas_input_dir
                        / "year2023"
                        / "households.csv"
                    },
                )
            ),
        },
        "atlas_run": {
            "completed_at": "2026-01-01T00:05:00",
            "cache_hit": False,
            "outputs": serialize_step_outputs(
                AtlasRunOutputs(
                    atlas_output_dir=old_atlas_output_dir,
                    raw_outputs={
                        "householdv_2023": old_atlas_output_dir / "householdv_2023.csv",
                        "vehicles_2023": old_atlas_output_dir / "vehicles_2023.csv",
                    },
                )
            ),
        },
    }
    manifest_path = tmp_path / "atlas_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))

    holder = StepOutputsHolder()
    coupler = _ManifestCoupler()
    scenario = _ManifestScenario(cache_hit=False)
    state = SimpleNamespace(year=2023, forecast_year=2029, atlas_year=2023, iteration=0)

    run_manifested_steps(
        stage_name="atlas",
        steps=[
            StepRef(
                name="atlas_preprocess",
                step_func=make_atlas_preprocess_step(
                    coupler=coupler,
                    outputs_holder=holder,
                ),
            ),
            StepRef(
                name="atlas_run",
                step_func=make_atlas_run_step(
                    coupler=coupler,
                    outputs_holder=holder,
                ),
            ),
        ],
        outputs_holder=holder,
        manifest_config=ManifestConfig(path=manifest_path),
        scenario=scenario,
        state=state,
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        name_suffix="2023_i0",
        iteration=0,
    )

    assert scenario.calls == []
    assert holder.atlas_preprocess is not None
    assert holder.atlas_run is not None
    assert holder.atlas_preprocess.atlas_mutable_input_dir == current_atlas_input_dir
    assert holder.atlas_preprocess.prepared_inputs["atlas_households_csv"] == current_households_csv
    assert holder.atlas_run.atlas_output_dir == current_atlas_output_dir
    assert holder.atlas_run.raw_outputs["householdv_2023"] == current_householdv_csv
    assert holder.atlas_run.raw_outputs["vehicles_2023"] == current_vehicles_csv
