from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from consist.types import OutputArtifactSpec

from pilates.activitysim.outputs import ActivitySimRunOutputs
from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
from pilates.activitysim.outputs import ASIM_OUTPUT_KEY_MAP
from pilates.activitysim.outputs import ASIM_OPTIONAL_RUN_OUTPUT_KEYS
from pilates.activitysim.outputs import ASIM_REQUIRED_RUN_OUTPUT_KEYS
from pilates.runtime.scenario_runtime import SchemaCoupler
from pilates.workflows.artifact_keys import ZARR_SKIMS
from pilates.workflows.orchestration import StepRef
from pilates.workflows.orchestration import _build_step_run_kwargs
from pilates.workflows.outputs_base import ValidationContext
from pilates.workflows.outputs_base import declared_outputs_for_step_outputs_class
from pilates.workflows.steps import StepOutputsHolder
from pilates.workflows.steps import activitysim as activitysim_steps
from pilates.workflows.steps.activitysim import activitysim_run_output_paths
from pilates.workflows.steps import make_activitysim_run_step


def test_activitysim_run_outputs_expose_canonical_declared_outputs():
    declared = declared_outputs_for_step_outputs_class(ActivitySimRunOutputs)
    expected = tuple(dict.fromkeys(ASIM_OUTPUT_KEY_MAP.values()))
    assert declared == expected
    assert ActivitySimRunOutputs.declared_output_keys() == expected


def test_activitysim_run_outputs_warn_on_unrecognized_output_keys(tmp_path, caplog):
    canonical_path = tmp_path / "households.parquet"
    canonical_path.write_text("ok", encoding="utf-8")
    extra_path = tmp_path / "mystery.parquet"
    extra_path.write_text("ok", encoding="utf-8")

    outputs = ActivitySimRunOutputs(
        output_dir=tmp_path,
        raw_outputs={
            "households_asim_out_temp": canonical_path,
            "mystery_output": extra_path,
        },
    )

    with caplog.at_level(logging.WARNING):
        outputs.validate(context=ValidationContext(step_name="activitysim_run"))

    assert "Unrecognized ActivitySim run output key 'mystery_output'" in caplog.text
    assert "households_asim_out_temp" not in caplog.text


def test_activitysim_run_outputs_expose_required_output_subset():
    required = ActivitySimRunOutputs.required_output_keys()

    assert required == ASIM_REQUIRED_RUN_OUTPUT_KEYS
    assert set(ASIM_OPTIONAL_RUN_OUTPUT_KEYS).isdisjoint(required)
    assert set(required) | set(ASIM_OPTIONAL_RUN_OUTPUT_KEYS) == set(
        ActivitySimRunOutputs.declared_output_keys()
    )


def test_activitysim_run_stepref_uses_required_outputs_for_runtime_contract():
    step_func = make_activitysim_run_step(
        coupler=SchemaCoupler(),
        outputs_holder=StepOutputsHolder(),
    )
    step = StepRef(
        name="activitysim_run",
        step_func=step_func,
        year=2023,
        iteration=0,
    )

    run_kwargs = _build_step_run_kwargs(
        step=step,
        settings=SimpleNamespace(run=None),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(
            full_path="/tmp/workspace",
            get_asim_output_dir=lambda: "/tmp/activitysim/output",
            get_asim_mutable_data_dir=lambda: "/tmp/activitysim/data",
        ),
        runtime_kwargs={},
        stage_name="activity_demand_run",
        default_iteration=0,
    )

    assert tuple(run_kwargs["outputs"]) == ASIM_REQUIRED_RUN_OUTPUT_KEYS
    assert "school_shadow_prices_asim_out" not in run_kwargs["outputs"]
    assert "workplace_shadow_prices_asim_out" not in run_kwargs["outputs"]


def test_activitysim_omx_stepref_requires_zarr_output_for_cache_recovery():
    step_func = make_activitysim_run_step(
        coupler=SchemaCoupler(),
        outputs_holder=StepOutputsHolder(),
        skim_mode="omx",
    )
    step = StepRef(
        name="activitysim_run",
        step_func=step_func,
        declared_outputs=[*ASIM_REQUIRED_RUN_OUTPUT_KEYS, ZARR_SKIMS],
        cache_hydration="outputs-requested",
        validate_cached_outputs="eager",
        year=2023,
        iteration=0,
    )

    run_kwargs = _build_step_run_kwargs(
        step=step,
        settings=SimpleNamespace(run=None),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(
            full_path="/tmp/workspace",
            get_asim_output_dir=lambda: "/tmp/activitysim/output",
            get_asim_mutable_data_dir=lambda: "/tmp/activitysim/data",
        ),
        runtime_kwargs={},
        stage_name="activity_demand_run",
        default_iteration=0,
    )

    assert tuple(run_kwargs["outputs"]) == (*ASIM_REQUIRED_RUN_OUTPUT_KEYS, ZARR_SKIMS)
    assert run_kwargs["cache_options"].cache_hydration == "outputs-requested"
    assert run_kwargs["cache_options"].validate_cached_outputs == "eager"


def test_activitysim_run_output_paths_are_single_source_of_truth_for_logging():
    """Declared paths publish only cache-recoverable ActivitySim artifacts."""
    workspace = SimpleNamespace(
        get_asim_output_dir=lambda: "/tmp/activitysim/output",
    )

    output_paths = activitysim_run_output_paths(
        settings=SimpleNamespace(run=None),
        state=SimpleNamespace(year=2023, forecast_year=2023, iteration=0),
        workspace=workspace,
        produces_zarr=True,
    )

    assert tuple(output_paths) == (*ASIM_REQUIRED_RUN_OUTPUT_KEYS, ZARR_SKIMS)
    assert "asim_output_dir" not in output_paths
    assert all(
        isinstance(output_spec, OutputArtifactSpec)
        for output_spec in output_paths.values()
    )
    assert output_paths["households_asim_out"].facet == {
        "artifact_family": "households",
        "year": 2023,
        "iteration": 0,
    }
    assert output_paths["households_asim_out"].facet_schema_version == "v1"
    assert output_paths["households_asim_out"].facet_index is True


def test_activitysim_run_output_contract_logs_once_and_updates_coupler(tmp_path):
    """The declarative contract must replace manual output publication."""
    consist = pytest.importorskip("consist")
    from consist.models.run import RunArtifactLink
    from sqlmodel import Session, select

    workspace = SimpleNamespace(
        get_asim_output_dir=lambda: str(tmp_path / "activitysim" / "output"),
    )
    output_paths = activitysim_run_output_paths(
        settings=SimpleNamespace(run=None),
        state=SimpleNamespace(year=2023, forecast_year=2023, iteration=0),
        workspace=workspace,
        produces_zarr=False,
    )
    for output_spec in output_paths.values():
        path = Path(output_spec.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test output", encoding="utf-8")

    tracker = consist.Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        mounts={"workspace": str(tmp_path)},
    )
    with tracker.scenario("activitysim-output-contract", model="pilates") as scenario:
        result = scenario.run(
            fn=lambda: None,
            name="activitysim_run",
            model="activitysim",
            year=2023,
            iteration=0,
            output_paths=output_paths,
        )

        assert tuple(result.outputs) == tuple(output_paths)
        assert scenario.coupler.get("households_asim_out") == result.outputs[
            "households_asim_out"
        ]
        run_id = result.run.id

    with Session(tracker.engine) as session:
        links = session.exec(
            select(RunArtifactLink).where(
                RunArtifactLink.run_id == run_id,
                RunArtifactLink.direction == "output",
            )
        ).all()

    assert len(links) == len(output_paths)


def test_activitysim_preprocess_and_postprocess_output_paths_leave_h5_to_manual_logging(
    monkeypatch,
):
    workspace = SimpleNamespace()
    state = SimpleNamespace(year=2023, forecast_year=2023, iteration=0)
    preprocess_outputs = {
        "asim_mutable_data_dir": "/tmp/asim/data",
        activitysim_steps.ASIM_HOUSEHOLDS_IN: "/tmp/asim/data/households.csv",
    }
    postprocess_outputs = {
        "asim_output_dir": "/tmp/asim/output",
        "households_asim_out": "/tmp/asim/output/households.parquet",
        "usim_datastore_h5": "/tmp/asim/output/usim.h5",
    }
    monkeypatch.setattr(
        activitysim_steps.ActivitysimPreprocessor,
        "expected_outputs",
        staticmethod(lambda *_args: preprocess_outputs),
    )
    monkeypatch.setattr(
        activitysim_steps.ActivitysimPostprocessor,
        "expected_outputs",
        staticmethod(lambda *_args: postprocess_outputs),
    )

    preprocess_paths = activitysim_steps.activitysim_preprocess_output_paths(
        settings=SimpleNamespace(), state=state, workspace=workspace
    )
    postprocess_paths = activitysim_steps.activitysim_postprocess_output_paths(
        settings=SimpleNamespace(), state=state, workspace=workspace
    )

    assert isinstance(
        preprocess_paths[activitysim_steps.ASIM_HOUSEHOLDS_IN], OutputArtifactSpec
    )
    assert isinstance(postprocess_paths["households_asim_out"], OutputArtifactSpec)
    assert "usim_datastore_h5" not in postprocess_paths


def test_activitysim_postprocess_outputs_require_processed_asim_tables_but_not_usim_next():
    required = ActivitySimPostprocessOutputs.required_output_keys()

    assert required == ASIM_REQUIRED_RUN_OUTPUT_KEYS
    assert "usim_input_next" not in required


def test_activitysim_postprocess_validation_requires_usim_output_when_land_use_enabled(
    tmp_path,
):
    asim_output_dir = tmp_path / "asim"
    asim_output_dir.mkdir()
    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=asim_output_dir,
        processed_outputs={},
    )

    with pytest.raises(
        AssertionError,
        match="usim_input_next/usim_datastore_h5 is required",
    ):
        outputs.validate(
            context=ValidationContext(
                step_name="activitysim_postprocess",
                settings=SimpleNamespace(land_use_enabled=True),
            )
        )


def test_activitysim_postprocess_validation_allows_missing_usim_output_when_land_use_disabled(
    tmp_path,
):
    asim_output_dir = tmp_path / "asim"
    asim_output_dir.mkdir()
    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=asim_output_dir,
        processed_outputs={},
    )

    outputs.validate(
        context=ValidationContext(
            step_name="activitysim_postprocess",
            settings=SimpleNamespace(land_use_enabled=False),
        )
    )
