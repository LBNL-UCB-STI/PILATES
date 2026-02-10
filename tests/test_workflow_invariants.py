from __future__ import annotations

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

from pathlib import Path
from types import SimpleNamespace

import yaml

from pilates.beam.outputs import BeamRunOutputs
from pilates.workflows import steps
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_PERSONS_IN,
)
from pilates.workflows.coupler_schema import build_coupler_schema
from pilates.workflows.orchestration import (
    ManifestConfig,
    StepRef,
    _recover_cached_outputs,
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
    for step_fn in schema_steps:
        meta = step_fn.__consist_step__
        declared = set(meta.outputs or []) | set(meta.schema_outputs or [])
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

    def _fake_make_generic_step_function(**kwargs):
        captured["output_logger"] = kwargs["output_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_beam,
        "_make_generic_step_function",
        _fake_make_generic_step_function,
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


class _ManifestWorkspace:
    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def full_path(self) -> str:
        return str(self._root)

    def get_asim_mutable_data_dir(self) -> str:
        return str(self._root / "activitysim" / "data")


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
    outputs = _recover_cached_outputs(
        step_name="activitysim_preprocess",
        outputs_holder=seed_holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=seed_coupler,
        step_inputs=None,
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

    def _should_not_run(**_kwargs):
        raise AssertionError("step function should not execute on manifest restore")

    _should_not_run.__consist_step__ = object()

    holder = StepOutputsHolder()
    coupler = _ManifestCoupler()
    scenario = _ManifestScenario(cache_hit=False)
    state = SimpleNamespace(year=2018, iteration=0)

    run_manifested_steps(
        stage_name="activity_demand_preprocess",
        steps=[
            StepRef(
                name="activitysim_preprocess",
                step_func=_should_not_run,
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

    def _noop_step(**_kwargs):
        return None

    _noop_step.__consist_step__ = object()

    holder = StepOutputsHolder()
    coupler = _ManifestCoupler()
    scenario = _ManifestScenario(cache_hit=True)
    state = SimpleNamespace(year=2018, iteration=0)

    run_manifested_steps(
        stage_name="activity_demand_preprocess",
        steps=[
            StepRef(
                name="activitysim_preprocess",
                step_func=_noop_step,
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
