"""
Narrative tests for startup workflow step contract validation.

These tests document the invariant that step wiring must be internally
consistent before long runs begin.

The validator checks four layers together:

1. ``StepOutputsHolder`` fields (runtime in-memory handoff contract)
2. ``STEP_OUTPUTS_CLASSES`` (typed output reconstruction contract)
3. ``STEP_DEPENDENCIES`` (execution ordering contract)
4. Declared step models (Consist metadata contract)

This file intentionally mutates one layer at a time to show what class of
integration drift is caught and what the expected startup failure looks like.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
import pytest
from consist import define_step

from pilates.activitysim.runner import ActivitysimRunner
from pilates.atlas.postprocessor import AtlasPostprocessor
from pilates.runtime import scenario_runtime
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_PERSONS_IN,
    BEAM_HOUSEHOLDS_IN,
    BEAM_FULL_SKIMS,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    LINKSTATS,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.orchestration import StepRef
from pilates.workflows.orchestration import _build_step_run_kwargs
from pilates.workflows.outputs_base import declared_outputs_for_step_outputs_class
from pilates.workflows.outputs_base import required_outputs_for_step_outputs_class
from pilates.workflows.surface import build_enabled_workflow_surface
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
from pilates.workflows.steps import shared as step_shared


class _DummyCoupler:
    def get(self, _key, default=None):
        return default

    def set(self, _key, _value):
        return None

    def set_from_artifact(self, _key, _value):
        return None


def _declared_schema_steps():
    coupler = _DummyCoupler()
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


def _validation_runtime_context(tmp_path: Path):
    workspace = SimpleNamespace(
        full_path=str(tmp_path),
        get_asim_mutable_data_dir=lambda: str(tmp_path / "activitysim" / "data"),
        get_asim_mutable_configs_dir=lambda: str(tmp_path / "activitysim" / "configs"),
        get_asim_output_dir=lambda: str(tmp_path / "activitysim" / "output"),
        get_usim_mutable_data_dir=lambda: str(tmp_path / "urbansim" / "data"),
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam" / "data"),
        get_beam_output_dir=lambda: str(tmp_path / "beam" / "output"),
        get_atlas_mutable_input_dir=lambda: str(tmp_path / "atlas" / "input"),
        get_atlas_output_dir=lambda: str(tmp_path / "atlas" / "output"),
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        shared=SimpleNamespace(skims=SimpleNamespace(fname="skims.omx")),
        urbansim=SimpleNamespace(
            input_file_template="custom_{region_id}.h5",
            output_file_template="usim_{year}.h5",
            region_mappings={"region_to_region_id": {"seattle": "123"}},
        ),
        runtime=SimpleNamespace(
            flags=SimpleNamespace(
                activity_demand_enabled=True,
                vehicle_ownership_model_enabled=True,
            ),
        ),
        activitysim=SimpleNamespace(persist_sharrow_cache=True),
        beam=SimpleNamespace(config="beam.conf", scenario_folder="scenario"),
        atlas=SimpleNamespace(model_dump=lambda: {"max_retries": 1}),
    )
    state = SimpleNamespace(
        year=2025,
        current_year=2025,
        forecast_year=2025,
        iteration=1,
        current_inner_iter=1,
        start_year=2017,
        is_start_year=lambda: False,
    )
    return settings, state, workspace


def test_validate_workflow_step_contracts_passes_for_current_setup():
    """Happy-path: current declared steps satisfy all contract invariants."""
    step_shared.validate_workflow_step_contracts(
        declared_steps=_declared_schema_steps()
    )


def test_filter_schema_steps_for_enabled_models_supports_surface_subset():
    settings = SimpleNamespace(
        runtime=SimpleNamespace(
            flags_initialized=True,
            flags=SimpleNamespace(
                land_use_enabled=False,
                vehicle_ownership_model_enabled=False,
                activity_demand_enabled=True,
                traffic_assignment_enabled=True,
                replanning_enabled=False,
            )
        ),
        run=SimpleNamespace(models=SimpleNamespace()),
    )
    surface = build_enabled_workflow_surface(settings)

    filtered = scenario_runtime.filter_schema_steps_for_enabled_models(
        _declared_schema_steps(),
        include_optional=True,
        surface=surface,
    )

    filtered_names = {
        getattr(getattr(step, "__consist_step__", None), "model", None) for step in filtered
    }
    assert "activitysim_preprocess" in filtered_names
    assert "beam_preprocess" in filtered_names
    assert "urbansim_preprocess" not in filtered_names
    assert "atlas_preprocess" not in filtered_names


def test_validate_workflow_step_contracts_flags_output_provider_catalog_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    settings, state, workspace = _validation_runtime_context(tmp_path)

    monkeypatch.setattr(
        AtlasPostprocessor,
        "expected_outputs",
        staticmethod(
            lambda *_args, **_kwargs: {
                "atlas_output_dir": str(tmp_path / "atlas" / "output"),
                "atlas_vehicles2_output": str(tmp_path / "atlas" / "output" / "vehicles2_2025.csv"),
            }
        ),
    )

    with pytest.raises(RuntimeError, match="atlas_postprocess.*missing required catalog output keys"):
        step_shared.validate_workflow_step_contracts(
            declared_steps=_declared_schema_steps(),
            settings=settings,
            state=state,
            workspace=workspace,
        )


def test_validate_workflow_step_contracts_flags_missing_required_input_provider_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    settings, state, workspace = _validation_runtime_context(tmp_path)

    monkeypatch.setattr(
        ActivitysimRunner,
        "declared_expected_inputs",
        staticmethod(
            lambda *_args, **_kwargs: {
                ASIM_LAND_USE_IN: str(tmp_path / "activitysim" / "data" / "land_use.csv"),
                ASIM_HOUSEHOLDS_IN: str(tmp_path / "activitysim" / "data" / "households.csv"),
                ASIM_PERSONS_IN: str(tmp_path / "activitysim" / "data" / "persons.csv"),
            }
        ),
    )

    with pytest.raises(RuntimeError, match="activitysim_run.*missing required catalog input keys.*zarr_skims"):
        step_shared.validate_workflow_step_contracts(
            declared_steps=_declared_schema_steps(),
            settings=settings,
            state=state,
            workspace=workspace,
        )


def test_invoke_contract_provider_does_not_mask_keyword_provider_typeerror():
    def _provider(settings, state, workspace):
        raise TypeError("intentional provider failure")

    with pytest.raises(TypeError, match="intentional provider failure"):
        step_shared._invoke_contract_provider(
            _provider,
            settings=object(),
            state=object(),
            workspace=object(),
        )


def test_validate_workflow_step_contracts_detects_holder_output_drift(monkeypatch):
    """Removing a tracked output class is detected as holder/output drift."""
    patched_classes = dict(step_shared.STEP_OUTPUTS_CLASSES)
    patched_classes.pop("beam_run", None)
    monkeypatch.setattr(step_shared, "STEP_OUTPUTS_CLASSES", patched_classes)

    with pytest.raises(RuntimeError, match="Missing output classes"):
        step_shared.validate_workflow_step_contracts(
            declared_steps=_declared_schema_steps()
        )


def test_validate_workflow_step_contracts_detects_bad_dependency_reference(monkeypatch):
    """Dependencies referencing unknown steps fail validation."""
    patched_deps = {key: dict(value) for key, value in step_shared.STEP_DEPENDENCIES.items()}
    beam_run_spec = dict(patched_deps["beam_run"])
    beam_run_spec["depends_on"] = ["not_a_real_step"]
    patched_deps["beam_run"] = beam_run_spec
    patched_runtime_deps = {
        key: dict(value) for key, value in step_shared.STEP_RUNTIME_DEPENDENCIES.items()
    }
    patched_runtime_beam_run = dict(patched_runtime_deps["beam_run"])
    patched_runtime_beam_run["depends_on"] = ["not_a_real_step"]
    patched_runtime_deps["beam_run"] = patched_runtime_beam_run
    monkeypatch.setattr(step_shared, "STEP_DEPENDENCIES", patched_deps)
    monkeypatch.setattr(step_shared, "STEP_RUNTIME_DEPENDENCIES", patched_runtime_deps)

    with pytest.raises(RuntimeError, match="depends_on unknown steps"):
        step_shared.validate_workflow_step_contracts(
            declared_steps=_declared_schema_steps()
        )


def test_validate_step_ready_enforces_untracked_activitysim_compile_inputs():
    holder = StepOutputsHolder()

    with pytest.raises(
        RuntimeError,
        match="activitysim_compile requires activitysim_preprocess to complete first",
    ):
        step_shared.validate_step_ready("activitysim_compile", holder)


def test_validate_workflow_step_contracts_flags_untracked_declared_steps():
    """New declared steps must be tracked or explicitly allowlisted."""
    @define_step(model="unexpected_new_step")
    def _extra_step(*args, **kwargs):
        return None

    with pytest.raises(RuntimeError, match="Declared step names are not tracked"):
        step_shared.validate_workflow_step_contracts(
            declared_steps=[*_declared_schema_steps(), _extra_step]
        )


def test_validate_workflow_step_contracts_allows_explicit_untracked_allowlist():
    """Allowlist supports intentional transitional or non-tracked step models."""
    @define_step(model="intentional_untracked_step")
    def _extra_step(*args, **kwargs):
        return None

    step_shared.validate_workflow_step_contracts(
        declared_steps=[*_declared_schema_steps(), _extra_step],
        allow_untracked_declared={"intentional_untracked_step"},
    )


def test_validate_workflow_step_contracts_allows_filtered_declared_subset_when_requested():
    subset = [
        step
        for step in _declared_schema_steps()
        if step.__consist_step__.model.startswith("activitysim_")
        or step.__consist_step__.model.startswith("beam_")
    ]

    step_shared.validate_workflow_step_contracts(
        declared_steps=subset,
        require_all_tracked_declared=False,
    )


def test_validate_workflow_step_contracts_reports_output_contract_conflicts():
    """Tracked-step metadata outputs must match canonical StepOutputs declarations."""
    steps = _declared_schema_steps()

    @define_step(model="urbansim_run", outputs=["metadata_override"])
    def _conflicting_urbansim_run(*args, **kwargs):
        return None

    steps = [
        step for step in steps if step.__consist_step__.model != "urbansim_run"
    ] + [_conflicting_urbansim_run]

    expected = (
        "Step 'urbansim_run': canonical required outputs ['usim_datastore_h5'] "
        "and declared outputs ['usim_datastore_h5'] conflict with metadata outputs "
        "['metadata_override']. "
        "Fix: remove metadata override or update declared_outputs in UrbanSimRunOutputs."
    )

    with pytest.raises(RuntimeError, match=re.escape(expected)):
        step_shared.validate_workflow_step_contracts(declared_steps=steps)


def test_validate_workflow_step_contracts_flags_missing_canonical_outputs_when_metadata_declares_outputs():
    """Tracked metadata outputs without canonical declared_outputs are rejected."""
    steps = _declared_schema_steps()

    @define_step(model="beam_run", outputs=["beam_linkstats"])
    def _beam_run_with_metadata_override(*args, **kwargs):
        return None

    steps = [step for step in steps if step.__consist_step__.model != "beam_run"] + [
        _beam_run_with_metadata_override
    ]

    expected = (
        "Step 'beam_run': canonical required outputs ['linkstats', 'beam_plans_out'] "
        "and declared outputs ['linkstats', 'beam_plans_out'] conflict with metadata outputs "
        "['beam_linkstats']. Fix: remove metadata override or update declared_outputs in "
        "BeamRunOutputs."
    )

    with pytest.raises(RuntimeError, match=re.escape(expected)):
        step_shared.validate_workflow_step_contracts(declared_steps=steps)


def test_tracked_beam_step_output_classes_define_explicit_canonical_outputs():
    """
    Tracked BEAM steps must keep explicit canonical output contracts.

    Compatibility exceptions should remain rare; beam_postprocess is one of
    them because whether zarr skims are required depends on which downstream
    models are active.
    """
    allowed_empty = frozenset({"beam_postprocess"})
    expected = {
        "beam_preprocess": (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN),
        "beam_run": (LINKSTATS, BEAM_PLANS_OUT),
        "beam_full_skim": (BEAM_FULL_SKIMS,),
    }

    for step_name, expected_outputs in expected.items():
        outputs_class = step_shared.STEP_OUTPUTS_CLASSES[step_name]
        canonical = required_outputs_for_step_outputs_class(outputs_class)
        if not canonical and step_name in allowed_empty:
            continue
        assert canonical == expected_outputs


def test_atlas_preprocess_output_class_defines_strict_core_contract():
    outputs_class = step_shared.STEP_OUTPUTS_CLASSES["atlas_preprocess"]

    declared = declared_outputs_for_step_outputs_class(outputs_class)
    required = required_outputs_for_step_outputs_class(outputs_class)

    assert declared == (
        "atlas_households_csv",
        "atlas_blocks_csv",
        "atlas_persons_csv",
        "atlas_residential_csv",
        "atlas_jobs_csv",
    )
    assert required == declared


def test_atlas_run_output_class_expands_stateful_required_outputs():
    outputs_class = step_shared.STEP_OUTPUTS_CLASSES["atlas_run"]

    required = required_outputs_for_step_outputs_class(
        outputs_class,
        state=SimpleNamespace(year=2019, forecast_year=2021, iteration=0),
    )

    assert required == ("householdv_2021", "vehicles_2021")


def test_runtime_step_kwargs_use_required_outputs_not_declared_outputs():
    """Runtime step launches must enforce required outputs, not the full declared schema."""

    step_funcs = {
        step.__consist_step__.model: step for step in _declared_schema_steps()
    }

    for step_name, outputs_class in step_shared.STEP_OUTPUTS_CLASSES.items():
        declared = tuple(declared_outputs_for_step_outputs_class(outputs_class))
        required = tuple(required_outputs_for_step_outputs_class(outputs_class))
        if declared == required:
            continue

        step_func = step_funcs[step_name]
        settings = SimpleNamespace(
            run=SimpleNamespace(region="test"),
            urbansim=SimpleNamespace(
                region_mappings={"region_to_region_id": {"test": "000"}},
                input_file_template="usim_{region_id}.h5",
            ),
        )
        workspace = SimpleNamespace(
            full_path="/tmp/workspace",
            get_asim_output_dir=lambda: "/tmp/activitysim/output",
            get_asim_mutable_data_dir=lambda: "/tmp/activitysim/data",
            get_beam_output_dir=lambda: "/tmp/beam/output",
            get_beam_mutable_data_dir=lambda: "/tmp/beam/data",
            get_usim_mutable_data_dir=lambda: "/tmp/usim/data",
        )
        run_kwargs = _build_step_run_kwargs(
            step=StepRef(
                name=step_name,
                step_func=step_func,
                year=2023,
                iteration=0,
            ),
            settings=settings,
            state=SimpleNamespace(),
            workspace=workspace,
            runtime_kwargs={},
            stage_name="test_stage",
            default_iteration=0,
        )

        assert tuple(run_kwargs["outputs"]) == required
        assert set(declared) - set(run_kwargs["outputs"]) == set(declared) - set(required)
