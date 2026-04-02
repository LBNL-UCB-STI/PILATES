from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import pprint
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import pandas as pd
import pytest

from consist import Tracker

from pilates.runtime import bootstrap as bootstrap_runtime
from pilates.runtime import launcher as launcher_runtime
from pilates.runtime import restart as restart_runtime
from pilates.runtime.consist_audit import emit_consist_audit_event, reset_consist_audit_state
from pilates.runtime.scenario_runtime import (
    ScenarioParentLinkProxy,
    resolve_cache_epoch,
    resolve_scenario_id,
    resolve_seed,
)
from pilates.utils import consist_runtime as cr
from pilates.workspace import Workspace
from pilates.workflows.artifact_keys import (
    BEAM_PLANS_OUT,
    LINKSTATS,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    ZARR_SKIMS,
)
from pilates.workflows.stages.land_use import run_land_use_stage
from pilates.workflows.stages.supply_demand import run_supply_demand_stage
from pilates.workflows.stages.vehicle_ownership import run_vehicle_ownership_stage
from pilates.workflows.steps import StepOutputsHolder
from workflow_state import WorkflowState

from tests.test_golden_stub_workflow import (
    DummyPostprocessor,
    DummyPreprocessor,
    DummyRunner,
    _build_settings,
    _write_csv,
    _write_file,
    _write_parquet,
    _write_usim_toy_h5,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOW_RESTART_EQUIVALENCE") != "1",
    reason="Set RUN_SLOW_RESTART_EQUIVALENCE=1 to run the full restart-equivalence harness.",
)


class _StopWorkflow(BaseException):
    pass


@dataclass
class _StubRuntime:
    settings: Any
    workspace: Workspace
    state: WorkflowState
    tracker: Tracker
    scenario_id: str
    seed: Optional[int]
    local_root: Path
    db_path: Path


def _build_test_settings(root: Path, run_name: str) -> Any:
    root.mkdir(parents=True, exist_ok=True)
    settings = _build_settings(root)
    settings.run.output_run_name = run_name
    settings.run.start_year = 2017
    settings.run.end_year = 2019
    settings.run.supply_demand_iters = 2
    settings.run.travel_model_freq = 1
    settings.run.output_directory = str(root / "outputs")
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = True
    settings.replanning_enabled = False
    settings.state_file_loc = str(root / "state.yaml")
    settings.atlas.beamac = 0
    return settings


def _build_workspace_inputs(settings: Any, workspace: Workspace, state: WorkflowState) -> None:
    Path(settings.urbansim.local_data_input_folder).mkdir(parents=True, exist_ok=True)

    usim_dir = Path(workspace.get_usim_mutable_data_dir())
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    asim_out_dir = Path(workspace.get_asim_output_dir())
    asim_configs_dir = Path(workspace.get_asim_mutable_configs_dir())
    beam_dir = Path(workspace.get_beam_mutable_data_dir())
    atlas_input_dir = Path(workspace.get_atlas_mutable_input_dir())
    atlas_output_dir = Path(workspace.get_atlas_output_dir())
    for path in (
        usim_dir,
        asim_dir,
        asim_out_dir,
        asim_configs_dir,
        beam_dir,
        atlas_input_dir,
        atlas_output_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    region_id = settings.urbansim.region_mappings["region_to_region_id"][settings.run.region]
    _write_usim_toy_h5(
        usim_dir / settings.urbansim.input_file_template.format(region_id=region_id)
    )
    _write_usim_toy_h5(usim_dir / "usim_2017.h5")
    _write_usim_toy_h5(usim_dir / "usim_2018.h5", with_year_prefix=2018)
    _write_usim_toy_h5(usim_dir / "usim_2019.h5", with_year_prefix=2019)
    _write_usim_toy_h5(usim_dir / "usim_input_merged2018.h5", with_year_prefix=2018)
    _write_usim_toy_h5(usim_dir / "usim_input_merged2019.h5", with_year_prefix=2019)

    _write_csv(
        asim_dir / "land_use.csv",
        pd.DataFrame(
            {
                "TAZ": [1, 2],
                "TOTPOP": [120, 80],
                "TOTHH": [1, 1],
                "TOTEMP": [40, 35],
                "COUNTY": [1, 1],
                "area_type": [2, 3],
            }
        ),
    )
    _write_csv(
        asim_dir / "households.csv",
        pd.DataFrame(
            {
                "household_id": [1, 2],
                "block_id": [1001, 1002],
                "income": [100000.0, 70000.0],
                "persons": [2, 1],
                "cars": [1, 2],
                "TAZ": [1, 2],
                "HHT": [4, 1],
            }
        ),
    )
    _write_csv(
        asim_dir / "persons.csv",
        pd.DataFrame(
            {
                "person_id": [11, 12, 21],
                "household_id": [1, 1, 2],
                "age": [40, 35, 17],
                "sex": [1, 2, 1],
                "worker": [1, 1, 0],
                "student": [0, 0, 1],
                "member_id": [1, 2, 1],
                "TAZ": [1, 1, 2],
            }
        ),
    )
    _write_file(asim_dir / "skims.omx")
    _write_file(beam_dir / "plans.csv")
    _write_file(beam_dir / "households.csv")
    _write_file(beam_dir / "persons.csv")
    _write_file(beam_dir / "linkstats.csv.gz")
    beam_region_dir = beam_dir / settings.run.region
    beam_scenario_dir = beam_region_dir / settings.beam.scenario_folder
    _write_file(beam_region_dir / settings.beam.config, "beam-config")
    _write_file(beam_scenario_dir / "plans.parquet")
    _write_file(beam_scenario_dir / "households.parquet")
    _write_file(beam_scenario_dir / "persons.parquet")
    _write_file(Path(workspace.full_path) / "shared_cache" / "numba" / "compile-cache.bin")
    _write_file(asim_out_dir / "cache" / "skims.zarr")
    state.data_initialized = True
    state.write_state()


def _install_model_factory_stubs(monkeypatch, settings: Any) -> None:
    from pilates.activitysim.outputs import (
        ActivitySimCompileOutputs,
        ActivitySimPostprocessOutputs,
        ActivitySimPreprocessOutputs,
        ActivitySimRunOutputs,
        normalize_asim_output_key,
    )
    from pilates.atlas.outputs import (
        AtlasPostprocessOutputs,
        AtlasPreprocessOutputs,
        AtlasRunOutputs,
    )
    from pilates.beam.outputs import BeamPostprocessOutputs, BeamPreprocessOutputs, BeamRunOutputs
    from pilates.generic.model_factory import ModelFactory
    from pilates.generic.records import RecordStore
    from pilates.urbansim.outputs import (
        UrbanSimPostprocessOutputs,
        UrbanSimPreprocessOutputs,
        UrbanSimRunOutputs,
    )

    def _write_asim_outputs(workspace: Workspace, year: int, iteration: int) -> dict[str, Path]:
        final_dir = Path(workspace.get_asim_output_dir()) / "final_pipeline"
        temp_outputs = {
            "accessibility_asim_out_temp": final_dir / "accessibility" / "final.parquet",
            "disaggregate_accessibility_asim_out_temp": final_dir / "disaggregate_accessibility" / "final.parquet",
            "joint_tour_participants_asim_out_temp": final_dir / "joint_tour_participants" / "final.parquet",
            "land_use_asim_out_temp": final_dir / "land_use" / "final.parquet",
            "non_mandatory_tour_destination_accessibility_asim_out_temp": final_dir / "non_mandatory_tour_destination_accessibility" / "final.parquet",
            "households_asim_out_temp": final_dir / "households" / "final.parquet",
            "persons_asim_out_temp": final_dir / "persons" / "final.parquet",
            "tours_asim_out_temp": final_dir / "tours" / "final.parquet",
            "trips_asim_out_temp": final_dir / "trips" / "final.parquet",
            "beam_plans_asim_out_temp": final_dir / "beam_plans" / "final.parquet",
        }
        _write_parquet(
            temp_outputs["accessibility_asim_out_temp"],
            pd.DataFrame({"year": [year], "iteration": [iteration], "accessibility": [1.0]}),
        )
        _write_parquet(
            temp_outputs["land_use_asim_out_temp"],
            pd.DataFrame({"zone_id": [1, 2], "TOTPOP": [120, 80], "year": [year, year]}),
        )
        _write_parquet(
            temp_outputs["disaggregate_accessibility_asim_out_temp"],
            pd.DataFrame({"person_id": [11, 21], "zone_id": [1, 2], "year": [year, year]}),
        )
        _write_parquet(
            temp_outputs["joint_tour_participants_asim_out_temp"],
            pd.DataFrame({"tour_id": [100], "person_id": [12], "iteration": [iteration]}),
        )
        _write_parquet(
            temp_outputs["households_asim_out_temp"],
            pd.DataFrame({"household_id": [1, 2], "auto_ownership": [1, 2], "iteration": [iteration, iteration]}),
        )
        _write_parquet(
            temp_outputs["persons_asim_out_temp"],
            pd.DataFrame({"person_id": [11, 12, 21], "year": [year, year, year]}),
        )
        _write_parquet(
            temp_outputs["tours_asim_out_temp"],
            pd.DataFrame({"tour_id": [100, 200], "iteration": [iteration, iteration]}),
        )
        _write_parquet(
            temp_outputs["trips_asim_out_temp"],
            pd.DataFrame({"trip_id": [1001, 1002, 2001], "year": [year, year, year]}),
        )
        _write_parquet(
            temp_outputs["non_mandatory_tour_destination_accessibility_asim_out_temp"],
            pd.DataFrame({"person_id": [11, 21], "destination": [2, 1], "year": [year, year]}),
        )
        _write_parquet(
            temp_outputs["beam_plans_asim_out_temp"],
            pd.DataFrame({"trip_id": [1001, 1002], "iteration": [iteration, iteration]}),
        )
        return temp_outputs

    def record_builder(
        model_name: str,
        phase: str,
        *,
        state: Any = None,
        workspace: Any = None,
        raw_outputs: Any = None,
        input_store: Any = None,
    ) -> Any:
        del input_store
        if phase == "preprocess":
            if model_name == "urbansim":
                usim_dir = Path(workspace.get_usim_mutable_data_dir())
                region_id = settings.urbansim.region_mappings["region_to_region_id"][settings.run.region]
                input_path = usim_dir / settings.urbansim.input_file_template.format(region_id=region_id)
                return UrbanSimPreprocessOutputs(
                    usim_mutable_data_dir=usim_dir,
                    prepared_inputs={
                        USIM_DATASTORE_BASE_H5: input_path,
                        USIM_DATASTORE_CURRENT_H5: input_path,
                    },
                )
            if model_name == "activitysim":
                asim_dir = Path(workspace.get_asim_mutable_data_dir())
                return ActivitySimPreprocessOutputs(
                    mutable_data_dir=asim_dir,
                    land_use_table=asim_dir / "land_use.csv",
                    households_table=asim_dir / "households.csv",
                    persons_table=asim_dir / "persons.csv",
                    omx_skims=asim_dir / "skims.omx",
                )
            if model_name == "beam":
                beam_dir = Path(workspace.get_beam_mutable_data_dir())
                return BeamPreprocessOutputs(
                    beam_mutable_data_dir=beam_dir,
                    prepared_inputs={
                        "plans_beam_in": beam_dir / "plans.csv",
                        "households_beam_in": beam_dir / "households.csv",
                        "persons_beam_in": beam_dir / "persons.csv",
                        "linkstats_warmstart": beam_dir / "linkstats.csv.gz",
                    },
                )
            if model_name == "atlas":
                atlas_input_dir = Path(workspace.get_atlas_mutable_input_dir())
                year_dir = atlas_input_dir / f"year{state.year}"
                outputs = {
                    "atlas_households_csv": year_dir / "households.csv",
                    "atlas_blocks_csv": year_dir / "blocks.csv",
                    "atlas_persons_csv": year_dir / "persons.csv",
                    "atlas_residential_csv": year_dir / "residential_units.csv",
                    "atlas_jobs_csv": year_dir / "jobs.csv",
                }
                for path in outputs.values():
                    _write_csv(path, pd.DataFrame({"id": [1, 2]}))
                return AtlasPreprocessOutputs(
                    atlas_mutable_input_dir=atlas_input_dir,
                    prepared_inputs=outputs,
                )
            return RecordStore()
        if phase == "run":
            if model_name == "urbansim":
                output_path = Path(workspace.get_usim_mutable_data_dir()) / f"usim_{state.forecast_year}.h5"
                _write_usim_toy_h5(output_path, with_year_prefix=state.forecast_year)
                return UrbanSimRunOutputs(
                    usim_datastore_h5=output_path,
                    raw_outputs={"usim_forecast_output": output_path},
                )
            if model_name == "activitysim_compile":
                zarr_path = Path(workspace.get_asim_output_dir()) / "cache" / "skims.zarr"
                _write_file(zarr_path)
                return ActivitySimCompileOutputs(
                    zarr_skims=zarr_path,
                    sharrow_cache_dir=Path(workspace.full_path) / "shared_cache" / "numba",
                )
            if model_name == "activitysim":
                return ActivitySimRunOutputs(
                    output_dir=Path(workspace.get_asim_output_dir()),
                    raw_outputs=_write_asim_outputs(workspace, state.current_year, state.current_inner_iter),
                )
            if model_name == "beam":
                beam_output_dir = Path(workspace.get_beam_output_dir())
                linkstats = beam_output_dir / "promoted_linkstats.csv.gz"
                plans = beam_output_dir / "promoted_plans.parquet"
                beam_output_dir.mkdir(parents=True, exist_ok=True)
                with gzip.open(linkstats, "wt", encoding="utf-8") as handle:
                    pd.DataFrame(
                        {
                            "link": [1, 2],
                            "year": [state.current_year, state.current_year],
                            "iteration": [state.current_inner_iter, state.current_inner_iter],
                        }
                    ).to_csv(handle, index=False)
                _write_parquet(
                    plans,
                    pd.DataFrame({"trip_id": [1001, 1002], "year": [state.current_year, state.current_year]}),
                )
                return BeamRunOutputs(
                    beam_output_dir=beam_output_dir,
                    raw_outputs={LINKSTATS: linkstats, BEAM_PLANS_OUT: plans},
                )
            if model_name == "atlas":
                atlas_output_dir = Path(workspace.get_atlas_output_dir())
                atlas_output_dir.mkdir(parents=True, exist_ok=True)
                householdv = atlas_output_dir / f"householdv_{state.year}.csv"
                vehicles = atlas_output_dir / f"vehicles_{state.year}.csv"
                _write_csv(householdv, pd.DataFrame({"household_id": [1, 2], "nvehicles": [1, 2]}))
                _write_csv(
                    vehicles,
                    pd.DataFrame(
                        {
                            "household_id": [1, 2],
                            "vehicle_id": [1, 1],
                            "bodytype": ["sedan", "suv"],
                            "pred_power": ["gasoline", "electricity"],
                            "modelyear": [2018, 2020],
                        }
                    ),
                )
                return AtlasRunOutputs(
                    atlas_output_dir=atlas_output_dir,
                    raw_outputs={f"householdv_{state.year}": householdv, f"vehicles_{state.year}": vehicles},
                )
            return RecordStore()
        if phase == "postprocess":
            if model_name == "urbansim":
                merged = Path(workspace.get_usim_mutable_data_dir()) / f"usim_input_merged{state.forecast_year}.h5"
                _write_usim_toy_h5(merged, with_year_prefix=state.forecast_year)
                return UrbanSimPostprocessOutputs(
                    usim_datastore_h5=merged,
                    processed_outputs={f"usim_input_merged{state.forecast_year}": merged},
                )
            if model_name == "activitysim":
                processed_outputs = {}
                iter_dir = Path(workspace.get_asim_output_dir()) / f"year-{state.current_year}-iteration-{state.current_inner_iter}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                for short_name, source_path in raw_outputs.raw_outputs.items():
                    source_path = Path(source_path)
                    clean_name = short_name.replace("_asim_out_temp", "")
                    target_path = iter_dir / f"{clean_name}.parquet"
                    shutil.copy2(source_path, target_path)
                    processed_outputs[normalize_asim_output_key(clean_name)] = target_path
                merged = Path(workspace.get_usim_mutable_data_dir()) / f"usim_input_merged{state.forecast_year}.h5"
                return ActivitySimPostprocessOutputs(
                    usim_datastore_h5=merged,
                    asim_output_dir=Path(workspace.get_asim_output_dir()),
                    processed_outputs=processed_outputs,
                    usim_datastore_key="usim_datastore_h5",
                )
            if model_name == "beam":
                zarr = Path(workspace.get_asim_output_dir()) / "cache" / "skims.zarr"
                final_skims = Path(workspace.get_beam_output_dir()) / "final_skims.omx"
                _write_file(zarr)
                _write_file(final_skims)
                return BeamPostprocessOutputs(zarr_skims=zarr, final_skims_omx=final_skims)
            if model_name == "atlas":
                vehicles2 = Path(workspace.get_atlas_output_dir()) / f"vehicles2_{state.year}.csv"
                _write_csv(
                    vehicles2,
                    pd.DataFrame(
                        {
                            "household_id": [1, 2],
                            "vehicle_id": [1, 1],
                            "bodytype": ["sedan", "suv"],
                            "pred_power": ["gasoline", "electricity"],
                            "modelyear": [2018, 2020],
                            "vehicleTypeId": ["sedan_gasoline_2018", "suv_electricity_2020"],
                        }
                    ),
                )
                usim_h5 = Path(workspace.get_usim_mutable_data_dir()) / f"usim_input_merged{state.main_forecast_year}.h5"
                if not usim_h5.exists():
                    _write_usim_toy_h5(usim_h5, with_year_prefix=state.main_forecast_year)
                return AtlasPostprocessOutputs(
                    atlas_output_dir=Path(workspace.get_atlas_output_dir()),
                    usim_datastore_h5=usim_h5,
                    processed_outputs={"atlas_vehicles2_output": vehicles2},
                )
            return RecordStore()
        return RecordStore()

    from pilates.generic.model_factory import ModelFactory
    from pilates.workflows.steps import shared as shared_steps

    original_consist_step_meta = shared_steps.consist_step_meta

    def _test_consist_step_meta(step_model: str):
        meta = dict(original_consist_step_meta(step_model))
        meta["identity_inputs"] = lambda _ctx: None
        if str(step_model).startswith(("beam_", "activitysim_")):
            meta["adapter"] = lambda _ctx: None
        return meta

    monkeypatch.setattr(shared_steps, "consist_step_meta", _test_consist_step_meta)

    def _make_preprocessor(self, model_name, state=None, *_args, **_kwargs):
        return DummyPreprocessor(model_name, record_builder, state=state)

    def _make_runner(self, model_name, state=None, *_args, **_kwargs):
        return DummyRunner(model_name, record_builder, state=state)

    def _make_postprocessor(self, model_name, state=None, *_args, **_kwargs):
        return DummyPostprocessor(model_name, record_builder, state=state)

    monkeypatch.setattr(ModelFactory, "get_preprocessor", _make_preprocessor)
    monkeypatch.setattr(ModelFactory, "get_runner", _make_runner)
    monkeypatch.setattr(ModelFactory, "get_postprocessor", _make_postprocessor)


def _make_runtime(tmp_path: Path, monkeypatch, *, name: str) -> _StubRuntime:
    root = tmp_path / name
    settings = _build_test_settings(root, name)
    workspace = Workspace(settings, output_path=str(root), folder_name="run")
    state = WorkflowState.from_settings(settings)
    _build_workspace_inputs(settings, workspace, state)
    _install_model_factory_stubs(monkeypatch, settings)
    monkeypatch.setattr(cr, "_schema_for_key", lambda _key: None)
    db_path = root / "provenance.duckdb"
    tracker = Tracker(
        run_dir=Path(workspace.full_path) / "consist_runs",
        db_path=str(db_path),
        mounts={"inputs": str(Path.cwd()), "workspace": str(workspace.full_path)},
        project_root=str(Path.cwd()),
        hashing_strategy="fast",
    )
    return _StubRuntime(
        settings=settings,
        workspace=workspace,
        state=state,
        tracker=tracker,
        scenario_id=resolve_scenario_id(settings),
        seed=resolve_seed(settings),
        local_root=root,
        db_path=db_path,
    )


def _query_facet(runtime: _StubRuntime) -> dict[str, Any]:
    facet = {"scenario_id": runtime.scenario_id}
    if runtime.seed is not None:
        facet["seed"] = runtime.seed
    return facet


def _emit_run_context(runtime: _StubRuntime, *, restart_run: bool, archive_run_dir: Optional[Path] = None) -> None:
    emit_consist_audit_event(
        workspace=runtime.workspace,
        event_type="run_context",
        scenario_id=runtime.scenario_id,
        seed=runtime.seed,
        run_name=runtime.settings.run.output_run_name,
        workspace_root=runtime.workspace.full_path,
        local_run_dir=runtime.workspace.full_path,
        archive_run_dir=str(archive_run_dir) if archive_run_dir is not None else None,
        archive_state_path=str(archive_run_dir / "state.yaml") if archive_run_dir is not None else None,
        restart_run=restart_run,
        data_initialized=bool(runtime.state.data_initialized),
    )


def _debug(message: str) -> None:
    if os.environ.get("RUN_SLOW_RESTART_EQUIVALENCE") == "1":
        print(f"[restart-equivalence] {message}", flush=True)


def _stage_runner(
    runtime: _StubRuntime,
    *,
    interruption: Optional[Callable[[str, WorkflowState], None]] = None,
) -> None:
    contract = launcher_runtime._build_scenario_runtime_contract(
        settings=runtime.settings,
        scenario_id=runtime.scenario_id,
        seed=runtime.seed,
        cache_epoch=resolve_cache_epoch(runtime.settings),
    )
    scenario_kwargs = dict(contract["scenario_kwargs"])
    coupler_schema = contract["coupler_schema"]

    from pilates.workflows.stages import supply_demand as supply_demand_stage
    from pilates.workflows.stages import vehicle_ownership as vehicle_ownership_stage
    from pilates.workflows.stages import land_use as land_use_stage

    original_activity_phase = supply_demand_stage._run_activity_demand_phase
    original_traffic_phase = supply_demand_stage._run_traffic_assignment_phase
    original_flush_archive_queue = vehicle_ownership_stage.flush_archive_queue
    original_supply_flush_archive_queue = supply_demand_stage.flush_archive_queue
    original_supply_archive_copy_now = supply_demand_stage.archive_copy_now
    original_vehicle_archive_copy_now = vehicle_ownership_stage.archive_copy_now
    original_land_use_flush_archive_queue = land_use_stage.flush_archive_queue
    original_land_use_archive_copy_now = land_use_stage.archive_copy_now

    def _maybe_interrupt(name: str, state: WorkflowState) -> None:
        if interruption is not None:
            interruption(name, state)

    def _wrapped_activity_phase(*args, **kwargs):
        result = original_activity_phase(*args, **kwargs)
        _maybe_interrupt("after_activitysim_postprocess", kwargs["state"])
        return result

    def _wrapped_traffic_phase(*args, **kwargs):
        result = original_traffic_phase(*args, **kwargs)
        _maybe_interrupt("after_beam_postprocess", kwargs["state"])
        return result

    atlas_flush_count = {"count": 0}

    def _wrapped_flush_archive_queue(*args, **kwargs):
        del args, kwargs
        result = None
        atlas_flush_count["count"] += 1
        if atlas_flush_count["count"] == 1:
            _maybe_interrupt("after_first_atlas_subyear", runtime.state)
        return result

    supply_demand_stage._run_activity_demand_phase = _wrapped_activity_phase
    supply_demand_stage._run_traffic_assignment_phase = _wrapped_traffic_phase
    vehicle_ownership_stage.flush_archive_queue = _wrapped_flush_archive_queue
    supply_demand_stage.flush_archive_queue = lambda **_kwargs: None
    supply_demand_stage.archive_copy_now = lambda **_kwargs: None
    vehicle_ownership_stage.archive_copy_now = lambda **_kwargs: None
    land_use_stage.flush_archive_queue = lambda **_kwargs: None
    land_use_stage.archive_copy_now = lambda **_kwargs: None
    try:
        _debug(f"stage_runner:start name={runtime.settings.run.output_run_name}")
        cr.set_enabled(True)
        with cr.use_tracker(runtime.tracker):
            with cr.scenario(
                name=runtime.settings.run.output_run_name,
                tracker=runtime.tracker,
                tags=[runtime.settings.run.output_run_name],
                model="pilates_orchestrator",
                **scenario_kwargs,
            ) as scenario:
                tagged = ScenarioParentLinkProxy(scenario)
                coupler = tagged.coupler
                coupler.declare_outputs(*coupler_schema.keys(), warn_undefined=True, description=coupler_schema)
                bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
                    settings=runtime.settings,
                    state=runtime.state,
                    workspace=runtime.workspace,
                    coupler=coupler,
                )
                for year in runtime.state:
                    _debug(
                        f"year_loop name={runtime.settings.run.output_run_name} year={year} "
                        f"major={getattr(runtime.state.current_major_stage, 'name', None)} "
                        f"sub={getattr(runtime.state.current_sub_stage, 'name', None)} "
                        f"iter={runtime.state.current_inner_iter}"
                    )
                    usim_inputs: Dict[str, Any] = {}
                    outputs_holder_year = StepOutputsHolder()
                    if runtime.state.should_run(WorkflowState.Stage.land_use):
                        _debug(f"run_land_use year={year}")
                        usim_inputs = run_land_use_stage(
                            scenario=tagged,
                            state=runtime.state,
                            settings=runtime.settings,
                            workspace=runtime.workspace,
                            coupler=coupler,
                            year=year,
                            outputs_holder_year=outputs_holder_year,
                        )
                        runtime.state.complete_step(WorkflowState.Stage.land_use)
                    if runtime.state.should_run(WorkflowState.Stage.vehicle_ownership_model):
                        _debug(f"run_vehicle_ownership year={year} forecast={runtime.state.forecast_year}")
                        run_vehicle_ownership_stage(
                            scenario=tagged,
                            state=runtime.state,
                            settings=runtime.settings,
                            workspace=runtime.workspace,
                            coupler=coupler,
                            year=year,
                            build_atlas_static_inputs_fallback=launcher_runtime.build_atlas_static_inputs_fallback,
                        )
                        runtime.state.complete_step(WorkflowState.Stage.vehicle_ownership_model)
                    if runtime.state.should_run(WorkflowState.Stage.supply_demand_loop):
                        _debug(
                            f"run_supply_demand year={year} forecast={runtime.state.forecast_year} "
                            f"sub={getattr(runtime.state.current_sub_stage, 'name', None)} "
                            f"iter={runtime.state.current_inner_iter}"
                        )
                        run_supply_demand_stage(
                            scenario=tagged,
                            state=runtime.state,
                            settings=runtime.settings,
                            workspace=runtime.workspace,
                            coupler=coupler,
                            year=year,
                            usim_inputs=usim_inputs,
                            build_manifest_path=launcher_runtime.build_manifest_path,
                        )
                    _maybe_interrupt("after_year_complete", runtime.state)
                _debug(f"stage_runner:complete name={runtime.settings.run.output_run_name}")
    finally:
        supply_demand_stage._run_activity_demand_phase = original_activity_phase
        supply_demand_stage._run_traffic_assignment_phase = original_traffic_phase
        vehicle_ownership_stage.flush_archive_queue = original_flush_archive_queue
        supply_demand_stage.flush_archive_queue = original_supply_flush_archive_queue
        supply_demand_stage.archive_copy_now = original_supply_archive_copy_now
        vehicle_ownership_stage.archive_copy_now = original_vehicle_archive_copy_now
        land_use_stage.flush_archive_queue = original_land_use_flush_archive_queue
        land_use_stage.archive_copy_now = original_land_use_archive_copy_now
        cr.set_enabled(None)


def _copy_tracker_db(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _resume_runtime(
    interrupted: _StubRuntime,
    local_runtime: _StubRuntime,
) -> dict[str, Any]:
    _debug(
        f"resume_runtime:start archive={interrupted.settings.run.output_run_name} "
        f"local={local_runtime.settings.run.output_run_name}"
    )
    _copy_tracker_db(interrupted.db_path, local_runtime.db_path)
    local_runtime.state = WorkflowState.from_settings(local_runtime.settings)
    reconstruction = restart_runtime.reconstruct_restart_completed_run_outputs(
        tracker=local_runtime.tracker,
        state=local_runtime.state,
        local_run_dir=local_runtime.workspace.full_path,
        archive_run_dir=interrupted.workspace.full_path,
        workflow_stage=WorkflowState.Stage,
        query_facet=_query_facet(local_runtime),
    )
    _debug(
        f"resume_runtime:reconstruction mode={reconstruction.get('discovery_mode')} "
        f"run_ids={len(reconstruction.get('run_ids', []))}"
    )
    emit_consist_audit_event(
        workspace=local_runtime.workspace,
        event_type="restart_discovery",
        discovery_mode=reconstruction.get("discovery_mode"),
        fallback_reason=reconstruction.get("fallback_reason"),
        discovered_run_count=len(reconstruction.get("run_ids", [])),
        query_target_count=len(reconstruction.get("query_targets", [])),
        matched_query_target_count=len(reconstruction.get("matched_query_targets", [])),
        unmatched_query_target_count=len(reconstruction.get("unmatched_query_targets", [])),
        query_targets=reconstruction.get("query_targets", []),
        matched_run_ids_by_target=reconstruction.get("matched_query_targets", []),
        unmatched_query_targets=reconstruction.get("unmatched_query_targets", []),
        atlas_gap_detected=bool(reconstruction.get("atlas_gap_detected", False)),
        restored_run_diagnostics=reconstruction.get("restored_run_diagnostics", []),
    )
    return {"reconstruction": reconstruction}


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_dataframe(frame: pd.DataFrame) -> str:
    normalized = frame.sort_index(axis=0).sort_index(axis=1)
    payload = normalized.to_json(orient="split", index=True, date_format="iso")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_csv(path: Path, *, compression: Optional[str] = None) -> str:
    frame = pd.read_csv(path, compression=compression)
    return _hash_dataframe(frame)


def _hash_parquet(path: Path) -> str:
    frame = pd.read_parquet(path)
    return _hash_dataframe(frame)


def _hash_h5_tables(path: Path) -> str:
    digests: dict[str, str] = {}
    with pd.HDFStore(path, mode="r") as store:
        for key in sorted(store.keys()):
            digests[key] = _hash_dataframe(store.get(key))
    payload = json.dumps(digests, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _digest_bundle(workspace: Workspace) -> dict[str, str]:
    asim_dir = Path(workspace.get_asim_output_dir()) / "final_pipeline"
    atlas_output_dir = Path(workspace.get_atlas_output_dir())
    usim_dir = Path(workspace.get_usim_mutable_data_dir())
    bundle = {
        "usim_merged_2019": _hash_h5_tables(usim_dir / "usim_input_merged2019.h5"),
        "atlas_vehicles2_2019": _hash_csv(atlas_output_dir / "vehicles2_2019.csv"),
        "asim_households": _hash_parquet(asim_dir / "households" / "final.parquet"),
        "asim_persons": _hash_parquet(asim_dir / "persons" / "final.parquet"),
        "asim_trips": _hash_parquet(asim_dir / "trips" / "final.parquet"),
        "beam_linkstats": _hash_csv(
            Path(workspace.get_beam_output_dir()) / "promoted_linkstats.csv.gz",
            compression="gzip",
        ),
        "beam_plans": _hash_parquet(Path(workspace.get_beam_output_dir()) / "promoted_plans.parquet"),
    }
    return bundle


def _manifest_snapshot(workspace: Workspace) -> dict[str, dict[str, list[str]]]:
    manifests: dict[str, dict[str, list[str]]] = {}
    workflow_dir = Path(workspace.full_path) / ".workflow"
    for manifest_path in sorted(workflow_dir.rglob("*.yaml")):
        payload = json.loads(json.dumps(__import__("yaml").safe_load(manifest_path.read_text(encoding="utf-8")) or {}))
        manifests[str(manifest_path.relative_to(workflow_dir))] = {
            "steps": sorted(payload.keys()),
            "steps_with_run_id": sorted(
                step_name for step_name, step_meta in payload.items() if (step_meta or {}).get("run_id")
            ),
        }
    return manifests


def _latest_completed_runs(tracker: Tracker) -> pd.DataFrame:
    def _run_facet(run: Any) -> dict[str, Any]:
        metadata = getattr(run, "metadata", None)
        if metadata is None:
            return {}
        if isinstance(metadata, dict):
            facet = metadata.get("facet", {})
            return facet if isinstance(facet, dict) else {}
        facet = getattr(metadata, "facet", None)
        return facet if isinstance(facet, dict) else {}

    rows = []
    for run in tracker.run_set(label="equivalence", limit=200000):
        if str(getattr(run, "status", "")).lower() != "completed":
            continue
        facet = _run_facet(run)
        model = getattr(run, "model_name", None) or facet.get("model")
        rows.append(
            {
                "run_id": str(getattr(run, "id", "")),
                "parent_run_id": getattr(run, "parent_run_id", None),
                "model": model,
                "year": getattr(run, "year", None) or facet.get("year"),
                "iteration": getattr(run, "iteration", None) or facet.get("iteration"),
                "created_at": str(getattr(run, "created_at", "")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["year", "iteration", "model", "created_at"])
    return frame.groupby(["year", "iteration", "model"], as_index=False).tail(1).reset_index(drop=True)


def _normalized_parent_edges(tracker: Tracker) -> set[tuple[Any, ...]]:
    latest = _latest_completed_runs(tracker)
    if latest.empty:
        return set()
    run_lookup = latest.set_index("run_id").to_dict(orient="index")
    edges = set()
    for row in latest.to_dict(orient="records"):
        parent = run_lookup.get(row["parent_run_id"])
        edges.add(
            (
                row["model"],
                int(row["year"]) if row["year"] is not None else None,
                int(row["iteration"]) if row["iteration"] is not None else None,
                parent["model"] if parent else None,
                int(parent["year"]) if parent and parent["year"] is not None else None,
                int(parent["iteration"]) if parent and parent["iteration"] is not None else None,
            )
        )
    return edges


def _run_index_rows(tracker: Tracker, archive_run_dir: str) -> set[tuple[Any, ...]]:
    analysis_src = Path(__file__).resolve().parents[1] / "analysis" / "src"
    import sys

    if str(analysis_src) not in sys.path:
        sys.path.insert(0, str(analysis_src))
    from pilates_consist_analysis.run_index import build_run_index

    frame = build_run_index(tracker, archive_run_dir=archive_run_dir).frame
    completed = frame.loc[frame["is_completed_status"] == True, ["scenario_id", "year", "iteration", "model"]]
    def _normalize_value(value: Any) -> Any:
        return None if pd.isna(value) else value

    return {
        tuple(_normalize_value(value) for value in row)
        for row in completed.itertuples(index=False, name=None)
    }


def _audit_snapshot(workspace: Workspace) -> dict[str, Any]:
    summary_path = Path(workspace.full_path) / ".workflow" / "diagnostics" / "consist_restart_audit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "steps_with_incomplete_hydration": summary.get("steps_with_incomplete_hydration", {}),
        "steps_using_custom_recovery": summary.get("steps_using_custom_recovery", {}),
        "restart_discovery": summary.get("restart_discovery", {}),
    }


def _snapshot(runtime: _StubRuntime) -> dict[str, Any]:
    _debug(f"snapshot:start name={runtime.settings.run.output_run_name}")
    coupler = getattr(getattr(runtime.tracker, "scenario", None), "coupler", None)
    out = {
        "artifact_digests": _digest_bundle(runtime.workspace),
        "manifest_snapshot": _manifest_snapshot(runtime.workspace),
        "parent_edges": _normalized_parent_edges(runtime.tracker),
        "run_index_rows": _run_index_rows(runtime.tracker, runtime.workspace.full_path),
        "audit": _audit_snapshot(runtime.workspace),
    }
    _debug(f"snapshot:complete name={runtime.settings.run.output_run_name}")
    return out


def _assert_equivalent(baseline: dict[str, Any], resumed: dict[str, Any]) -> None:
    def _describe_difference(name: str, expected: Any, actual: Any) -> str:
        if isinstance(expected, dict) and isinstance(actual, dict):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            changed = sorted(key for key in set(expected) & set(actual) if expected[key] != actual[key])
            parts = [f"{name} mismatch"]
            if missing:
                parts.append(f"missing keys={missing[:5]}")
            if extra:
                parts.append(f"extra keys={extra[:5]}")
            if changed:
                sample = changed[:3]
                rendered = {
                    key: {"baseline": expected[key], "resumed": actual[key]}
                    for key in sample
                }
                parts.append(f"changed sample={pprint.pformat(rendered, compact=True)}")
            return "; ".join(parts)
        if isinstance(expected, set) and isinstance(actual, set):
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            return (
                f"{name} mismatch; "
                f"missing sample={missing[:5]}; extra sample={extra[:5]}"
            )
        return (
            f"{name} mismatch; "
            f"baseline={pprint.pformat(expected, compact=True)}; "
            f"resumed={pprint.pformat(actual, compact=True)}"
        )

    assert resumed["artifact_digests"] == baseline["artifact_digests"], _describe_difference(
        "artifact_digests", baseline["artifact_digests"], resumed["artifact_digests"]
    )
    assert resumed["manifest_snapshot"] == baseline["manifest_snapshot"], _describe_difference(
        "manifest_snapshot", baseline["manifest_snapshot"], resumed["manifest_snapshot"]
    )
    assert resumed["parent_edges"] == baseline["parent_edges"], _describe_difference(
        "parent_edges", baseline["parent_edges"], resumed["parent_edges"]
    )
    assert resumed["run_index_rows"] == baseline["run_index_rows"], _describe_difference(
        "run_index_rows", baseline["run_index_rows"], resumed["run_index_rows"]
    )
    assert resumed["audit"]["steps_with_incomplete_hydration"] == {}, _describe_difference(
        "steps_with_incomplete_hydration", {}, resumed["audit"]["steps_with_incomplete_hydration"]
    )
    compatibility_fallbacks = {
        step_name: mode_counts
        for step_name, mode_counts in resumed["audit"]["steps_using_custom_recovery"].items()
        if "used_compatibility_fallback" in mode_counts
    }
    assert compatibility_fallbacks == {}, _describe_difference(
        "steps_using_custom_recovery.compatibility_fallbacks", {}, compatibility_fallbacks
    )


def _interrupt_after(boundary: str) -> Callable[[str, WorkflowState], None]:
    def _fn(event: str, state: WorkflowState) -> None:
        del state
        if event == boundary:
            raise _StopWorkflow(boundary)

    return _fn


@pytest.fixture(autouse=True)
def _reset_audit_state():
    reset_consist_audit_state()
    yield
    reset_consist_audit_state()


@pytest.fixture
def baseline_snapshot(tmp_path, monkeypatch):
    runtime = _make_runtime(tmp_path, monkeypatch, name="baseline")
    _emit_run_context(runtime, restart_run=False)
    _stage_runner(runtime)
    return _snapshot(runtime)


def _run_resumed_case(tmp_path, monkeypatch, *, stop_boundary: str) -> dict[str, Any]:
    archive_runtime = _make_runtime(tmp_path, monkeypatch, name=f"archive_{stop_boundary}")
    _emit_run_context(archive_runtime, restart_run=False)
    _debug(f"resumed_case:interrupting boundary={stop_boundary}")
    with pytest.raises(_StopWorkflow):
        _stage_runner(archive_runtime, interruption=_interrupt_after(stop_boundary))

    resumed_runtime = _make_runtime(tmp_path, monkeypatch, name=f"resume_{stop_boundary}")
    _emit_run_context(
        resumed_runtime,
        restart_run=True,
        archive_run_dir=Path(archive_runtime.workspace.full_path),
    )
    resume_state = _resume_runtime(archive_runtime, resumed_runtime)

    contract = launcher_runtime._build_scenario_runtime_contract(
        settings=resumed_runtime.settings,
        scenario_id=resumed_runtime.scenario_id,
        seed=resumed_runtime.seed,
        cache_epoch=resolve_cache_epoch(resumed_runtime.settings),
    )
    scenario_kwargs = dict(contract["scenario_kwargs"])
    coupler_schema = contract["coupler_schema"]
    cr.set_enabled(True)
    try:
        _debug(f"resumed_case:resume_execute boundary={stop_boundary}")
        with cr.use_tracker(resumed_runtime.tracker):
            with cr.scenario(
                name=resumed_runtime.settings.run.output_run_name,
                tracker=resumed_runtime.tracker,
                tags=[resumed_runtime.settings.run.output_run_name],
                model="pilates_orchestrator",
                **scenario_kwargs,
            ) as scenario:
                tagged = ScenarioParentLinkProxy(scenario)
                for restored in resume_state["reconstruction"].get("restored_run_diagnostics", []):
                    tagged.remember_restored_run_id(
                        model_name=restored.get("model"),
                        year=restored.get("year"),
                        iteration=restored.get("iteration"),
                        run_id=restored.get("run_id"),
                    )
                coupler = tagged.coupler
                coupler.declare_outputs(*coupler_schema.keys(), warn_undefined=True, description=coupler_schema)
                bootstrap_runtime.seed_bootstrap_artifacts_to_coupler(
                    settings=resumed_runtime.settings,
                    state=resumed_runtime.state,
                    workspace=resumed_runtime.workspace,
                    coupler=coupler,
                )
                for year in resumed_runtime.state:
                    usim_inputs: Dict[str, Any] = {}
                    outputs_holder_year = StepOutputsHolder()
                    if resumed_runtime.state.should_run(WorkflowState.Stage.land_use):
                        usim_inputs = run_land_use_stage(
                            scenario=tagged,
                            state=resumed_runtime.state,
                            settings=resumed_runtime.settings,
                            workspace=resumed_runtime.workspace,
                            coupler=coupler,
                            year=year,
                            outputs_holder_year=outputs_holder_year,
                        )
                        resumed_runtime.state.complete_step(WorkflowState.Stage.land_use)
                    if resumed_runtime.state.should_run(WorkflowState.Stage.vehicle_ownership_model):
                        run_vehicle_ownership_stage(
                            scenario=tagged,
                            state=resumed_runtime.state,
                            settings=resumed_runtime.settings,
                            workspace=resumed_runtime.workspace,
                            coupler=coupler,
                            year=year,
                            build_atlas_static_inputs_fallback=launcher_runtime.build_atlas_static_inputs_fallback,
                        )
                        resumed_runtime.state.complete_step(WorkflowState.Stage.vehicle_ownership_model)
                    if resumed_runtime.state.should_run(WorkflowState.Stage.supply_demand_loop):
                        run_supply_demand_stage(
                            scenario=tagged,
                            state=resumed_runtime.state,
                            settings=resumed_runtime.settings,
                            workspace=resumed_runtime.workspace,
                            coupler=coupler,
                            year=year,
                            usim_inputs=usim_inputs,
                            build_manifest_path=launcher_runtime.build_manifest_path,
                        )
    finally:
        cr.set_enabled(None)
    _debug(f"resumed_case:resume_complete boundary={stop_boundary}")
    return _snapshot(resumed_runtime)


@pytest.mark.parametrize(
    "stop_boundary",
    [
        "after_activitysim_postprocess",
        "after_beam_postprocess",
        "after_first_atlas_subyear",
        "after_year_complete",
    ],
)
def test_stubbed_restart_resume_matches_uninterrupted_baseline(
    baseline_snapshot,
    tmp_path,
    monkeypatch,
    stop_boundary,
):
    resumed = _run_resumed_case(tmp_path, monkeypatch, stop_boundary=stop_boundary)
    _assert_equivalent(baseline_snapshot, resumed)
