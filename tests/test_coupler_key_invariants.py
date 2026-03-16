"""
Coupler key-set invariants for stage boundaries.

These tests pin a stable subset of coupler keys that each major workflow
boundary must publish. The goal is to catch silent regressions in downstream
artifact publication during the simplification refactor, especially where
typed outputs currently bridge through RecordStore updates.
"""

from pathlib import Path
from types import SimpleNamespace

from pilates.activitysim.outputs import (
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.beam.outputs import BeamPostprocessOutputs, BeamPreprocessOutputs, BeamRunOutputs
from pilates.generic.model_factory import ModelFactory
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    BEAM_PERSONS_IN,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_INPUT_MERGED_PREFIX,
    ZARR_SKIMS,
)
from pilates.workflows.orchestration import ManifestConfig
from pilates.workflows.stages.land_use import run_land_use_stage
from pilates.workflows.stages.supply_demand import (
    ActivityDemandPhaseInputs,
    TrafficAssignmentPhaseInputs,
    _run_activity_demand_phase,
    _run_traffic_assignment_phase,
)
from pilates.workflows.stages.vehicle_ownership import run_vehicle_ownership_stage
from pilates.workflows.steps import StepOutputsHolder
from tests.test_stage_contracts import _write_file

pytest_plugins = ("tests.test_stage_contracts",)


def _coupler_keys(coupler) -> set[str]:
    return set(coupler.keys())


def _assert_coupler_contains(coupler, expected_keys: set[str]) -> None:
    observed = _coupler_keys(coupler)
    missing = expected_keys - observed
    assert not missing, f"Missing coupler keys: {sorted(missing)}; observed={sorted(observed)}"


def test_land_use_boundary_publishes_urbansim_datastore_family(stage_env):
    outputs_holder = StepOutputsHolder()

    run_land_use_stage(
        scenario=stage_env["scenario"],
        state=stage_env["state"],
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=stage_env["state"].forecast_year,
        outputs_holder_year=outputs_holder,
    )

    _assert_coupler_contains(
        stage_env["coupler"],
        {USIM_DATASTORE_H5, USIM_DATASTORE_BASE_H5},
    )


def test_vehicle_ownership_boundary_preserves_urbansim_datastore_family(stage_env):
    stage_env["coupler"].set(USIM_DATASTORE_CURRENT_H5, stage_env["usim_input_path"])
    stage_env["coupler"].set(USIM_DATASTORE_BASE_H5, stage_env["usim_input_path"])

    run_vehicle_ownership_stage(
        scenario=stage_env["scenario"],
        state=stage_env["state"],
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=stage_env["state"].forecast_year,
        build_atlas_static_inputs_fallback=lambda _workspace: {},
    )

    _assert_coupler_contains(
        stage_env["coupler"],
        {USIM_DATASTORE_H5, USIM_DATASTORE_BASE_H5},
    )


def test_activity_demand_boundary_publishes_activitysim_key_family(
    stage_env, monkeypatch, tmp_path
):
    stage_env["coupler"].set(USIM_DATASTORE_CURRENT_H5, stage_env["usim_input_path"])
    stage_env["coupler"].set(USIM_DATASTORE_BASE_H5, stage_env["usim_input_path"])

    state = stage_env["state"]
    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    original_get_preprocessor = ModelFactory.get_preprocessor
    original_get_runner = ModelFactory.get_runner
    original_get_postprocessor = ModelFactory.get_postprocessor

    class _ActivitySimPreprocessor:
        def preprocess(self, workspace):
            input_dir = Path(workspace.get_asim_mutable_data_dir())
            outputs = {
                ASIM_LAND_USE_IN: input_dir / "land_use.csv",
                ASIM_HOUSEHOLDS_IN: input_dir / "households.csv",
                ASIM_PERSONS_IN: input_dir / "persons.csv",
                ASIM_OMX_SKIMS: input_dir / "skims.omx",
            }
            for path in outputs.values():
                _write_file(path)
            return ActivitySimPreprocessOutputs(
                mutable_data_dir=input_dir,
                land_use_table=outputs[ASIM_LAND_USE_IN],
                households_table=outputs[ASIM_HOUSEHOLDS_IN],
                persons_table=outputs[ASIM_PERSONS_IN],
                omx_skims=outputs[ASIM_OMX_SKIMS],
            )

    class _ActivitySimRunner:
        def run(self, input_store, workspace, *, extra_inputs=None):
            output_dir = Path(workspace.get_asim_output_dir())
            raw_outputs = {
                "beam_plans_asim_out": output_dir / "beam_plans.csv",
                "households_asim_out": output_dir / "households.csv",
                "persons_asim_out": output_dir / "persons.csv",
            }
            for path in raw_outputs.values():
                _write_file(path)
            return ActivitySimRunOutputs(
                output_dir=output_dir,
                raw_outputs=raw_outputs,
            )

    class _ActivitySimPostprocessor:
        def postprocess(self, raw_outputs, workspace):
            output_dir = Path(workspace.get_asim_output_dir()) / "postprocess"
            processed_outputs = {
                "beam_plans_asim_out": output_dir / "beam_plans.parquet",
                "households_asim_out": output_dir / "households.parquet",
                "persons_asim_out": output_dir / "persons.parquet",
            }
            updated_usim = (
                Path(workspace.get_usim_mutable_data_dir())
                / f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}.h5"
            )
            for path in (*processed_outputs.values(), updated_usim):
                _write_file(path)
            return ActivitySimPostprocessOutputs(
                usim_datastore_h5=updated_usim,
                asim_output_dir=output_dir,
                processed_outputs=processed_outputs,
                usim_datastore_key=f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}",
            )

    def _patched_get_preprocessor(self, model_name, state=None, *_args, **_kwargs):
        if model_name == "activitysim":
            return _ActivitySimPreprocessor()
        return original_get_preprocessor(self, model_name, state)

    def _patched_get_runner(self, model_name, state=None, *_args, **_kwargs):
        if model_name == "activitysim":
            return _ActivitySimRunner()
        return original_get_runner(self, model_name, state)

    def _patched_get_postprocessor(self, model_name, state=None, *_args, **_kwargs):
        if model_name == "activitysim":
            return _ActivitySimPostprocessor()
        return original_get_postprocessor(self, model_name, state)

    monkeypatch.setattr(ModelFactory, "get_preprocessor", _patched_get_preprocessor)
    monkeypatch.setattr(ModelFactory, "get_runner", _patched_get_runner)
    monkeypatch.setattr(ModelFactory, "get_postprocessor", _patched_get_postprocessor)

    outputs_holder = StepOutputsHolder()
    original_scenario_run = stage_env["scenario"].run

    def _scenario_run_with_activitysim_upstream(**kwargs):
        input_keys = kwargs.get("input_keys") or []
        if "beam_plans_asim_out" in input_keys and outputs_holder.activitysim_run is not None:
            for key, value in outputs_holder.activitysim_run.to_record_store().to_mapping().items():
                stage_env["coupler"].set(key, value)
        return original_scenario_run(**kwargs)

    monkeypatch.setattr(stage_env["scenario"], "run", _scenario_run_with_activitysim_upstream)

    _run_activity_demand_phase(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        inputs=ActivityDemandPhaseInputs(
            year=state.forecast_year,
            iteration=0,
            usim_inputs={
                USIM_DATASTORE_CURRENT_H5: stage_env["usim_input_path"],
                USIM_DATASTORE_BASE_H5: stage_env["usim_input_path"],
            },
        ),
        outputs_holder=outputs_holder,
        manifest_config=ManifestConfig(path=tmp_path / "activity_demand_manifest.yaml"),
    )

    _assert_coupler_contains(
        stage_env["coupler"],
        {
            USIM_DATASTORE_H5,
            USIM_DATASTORE_BASE_H5,
            ASIM_LAND_USE_IN,
            ASIM_HOUSEHOLDS_IN,
            ASIM_PERSONS_IN,
            ASIM_OMX_SKIMS,
            ZARR_SKIMS,
        },
    )


def test_traffic_assignment_boundary_publishes_beam_key_family(
    stage_env, monkeypatch, tmp_path
):
    state = stage_env["state"]
    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.traffic_assignment
    state.current_inner_iter = 0

    zarr_path = Path(stage_env["workspace"].get_asim_output_dir()) / "cache" / "skims.zarr"
    _write_file(zarr_path)
    stage_env["coupler"].set(ZARR_SKIMS, str(zarr_path))

    activity_outputs = {
        "beam_plans_asim_out": tmp_path / "beam_plans.parquet",
        "households_asim_out": tmp_path / "households.parquet",
        "persons_asim_out": tmp_path / "persons.parquet",
    }
    for path in activity_outputs.values():
        _write_file(path)
    activity_demand_outputs = {
        short_name: str(path) for short_name, path in activity_outputs.items()
    }

    original_get_preprocessor = ModelFactory.get_preprocessor
    original_get_runner = ModelFactory.get_runner
    original_get_postprocessor = ModelFactory.get_postprocessor

    class _BeamPreprocessor:
        def preprocess(
            self,
            workspace,
            *,
            activity_demand_outputs=None,
            previous_beam_outputs=None,
            beam_preprocess_inputs=None,
        ):
            prepared_inputs = {
                BEAM_PLANS_IN: Path(activity_demand_outputs["beam_plans_asim_out"]),
                BEAM_HOUSEHOLDS_IN: Path(activity_demand_outputs["households_asim_out"]),
                BEAM_PERSONS_IN: Path(activity_demand_outputs["persons_asim_out"]),
            }
            return BeamPreprocessOutputs(
                beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
                prepared_inputs=prepared_inputs,
            )

    class _BeamRunner:
        def run(self, input_store, workspace, *, extra_inputs=None):
            output_dir = Path(workspace.get_beam_output_dir())
            run_outputs = {
                f"linkstats_parquet_{state.forecast_year}_0": output_dir / "linkstats.parquet",
                f"beam_plans_out_{state.forecast_year}_0": output_dir / "plans.xml",
                f"events_parquet_{state.forecast_year}_0": output_dir / "events.parquet",
                f"raw_od_skims_zarr_{state.forecast_year}_0": output_dir / "od_skims.zarr",
            }
            for path in run_outputs.values():
                _write_file(path)
            return BeamRunOutputs(
                beam_output_dir=output_dir,
                raw_outputs=run_outputs,
            )

    class _BeamPostprocessor:
        def postprocess(self, raw_outputs, workspace):
            _write_file(zarr_path)
            final_skims_omx = Path(workspace.get_beam_output_dir()) / "final_skims.omx"
            _write_file(final_skims_omx)
            return BeamPostprocessOutputs(
                zarr_skims=zarr_path,
                final_skims_omx=final_skims_omx,
            )

    def _patched_get_preprocessor(self, model_name, state=None, *_args, **_kwargs):
        if model_name == "beam":
            return _BeamPreprocessor()
        return original_get_preprocessor(self, model_name, state)

    def _patched_get_runner(self, model_name, state=None, *_args, **_kwargs):
        if model_name == "beam":
            return _BeamRunner()
        return original_get_runner(self, model_name, state)

    def _patched_get_postprocessor(self, model_name, state=None, *_args, **_kwargs):
        if model_name == "beam":
            return _BeamPostprocessor()
        return original_get_postprocessor(self, model_name, state)

    monkeypatch.setattr(ModelFactory, "get_preprocessor", _patched_get_preprocessor)
    monkeypatch.setattr(ModelFactory, "get_runner", _patched_get_runner)
    monkeypatch.setattr(ModelFactory, "get_postprocessor", _patched_get_postprocessor)

    outputs_holder = StepOutputsHolder()
    outputs_holder.activitysim_postprocess = SimpleNamespace()
    _run_traffic_assignment_phase(
        scenario=stage_env["scenario"],
        state=state,
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        inputs=TrafficAssignmentPhaseInputs(
            year=state.forecast_year,
            iteration=0,
            activity_demand_outputs=activity_demand_outputs,
            previous_beam_outputs=None,
        ),
        outputs_holder=outputs_holder,
    )

    _assert_coupler_contains(
        stage_env["coupler"],
        {
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
            BEAM_PLANS_OUT,
            LINKSTATS,
            LINKSTATS_WARMSTART,
            ZARR_SKIMS,
        },
    )
