from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

import pytest
import pandas as pd
import yaml

from pilates.config import load_config
from pilates.generic.records import FileRecord, RecordStore
from pilates.utils import consist_runtime as cr
from pilates.utils.provenance_report import write_provenance_report
from pilates.workspace import Workspace
from pilates.activitysim.outputs import normalize_asim_output_key
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_INPUT_MERGED_PREFIX,
    ZARR_SKIMS,
)
from pilates.workflows.stages.land_use import run_land_use_stage
from pilates.workflows.stages.supply_demand import run_supply_demand_stage
from pilates.workflows.stages.vehicle_ownership import run_vehicle_ownership_stage
from pilates.workflows.steps import StepOutputsHolder
from workflow_state import WorkflowState


class DummyPreprocessor:
    def __init__(self, model_name, record_builder, state=None):
        self.model_name = model_name
        self._record_builder = record_builder
        self._state = state

    def preprocess(self, workspace, previous_records=RecordStore()):
        return self._record_builder(
            self.model_name, "preprocess", state=self._state, workspace=workspace
        )


class DummyRunner:
    def __init__(self, model_name, record_builder, state=None):
        self.model_name = model_name
        self._record_builder = record_builder
        self._state = state

    def run(self, input_store, workspace):
        return self._record_builder(
            self.model_name, "run", state=self._state, workspace=workspace
        )


class DummyPostprocessor:
    def __init__(self, model_name, record_builder, state=None):
        self.model_name = model_name
        self._record_builder = record_builder
        self._state = state

    def postprocess(self, raw_outputs, workspace):
        return self._record_builder(
            self.model_name,
            "postprocess",
            state=self._state,
            workspace=workspace,
            raw_outputs=raw_outputs,
        )


def _write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _write_usim_toy_h5(path: Path, *, with_year_prefix: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    table_prefix = f"/{with_year_prefix}" if with_year_prefix is not None else ""
    households_key = f"{table_prefix}/households" if table_prefix else "households"
    blocks_key = f"{table_prefix}/blocks" if table_prefix else "blocks"
    persons_key = f"{table_prefix}/persons" if table_prefix else "persons"
    residential_key = (
        f"{table_prefix}/residential_units" if table_prefix else "residential_units"
    )
    jobs_key = f"{table_prefix}/jobs" if table_prefix else "jobs"
    graveyard_key = f"{table_prefix}/graveyard" if table_prefix else "graveyard"

    households = pd.DataFrame({"income": [100000.0, 70000.0]}, index=[1, 2])
    households.index.name = "household_id"

    blocks = pd.DataFrame({"zone_id": [10, 11]}, index=["0001", "0002"])
    blocks.index.name = "block_id"

    persons = pd.DataFrame(
        {"household_id": [1, 2], "age": [40, 35]},
        index=[101, 102],
    )
    persons.index.name = "person_id"

    residential_units = pd.DataFrame(
        {"block_id": ["0001", "0002"], "year_built": [1990, 2005]},
        index=[1001, 1002],
    )
    residential_units.index.name = "unit_id"

    jobs = pd.DataFrame({"block_id": ["0001", "0002"]}, index=[5001, 5002])
    jobs.index.name = "job_id"

    graveyard = pd.DataFrame({"household_id": [1]}, index=[201])
    graveyard.index.name = "person_id"

    households.to_hdf(path, key=households_key, mode="w")
    blocks.to_hdf(path, key=blocks_key, mode="a")
    persons.to_hdf(path, key=persons_key, mode="a")
    residential_units.to_hdf(path, key=residential_key, mode="a")
    jobs.to_hdf(path, key=jobs_key, mode="a")
    graveyard.to_hdf(path, key=graveyard_key, mode="a")


def _build_settings(tmp_path: Path):
    config = {
        "run": {
            "region": "test",
            "scenario": "test",
            "start_year": 2017,
            "end_year": 2018,
            "output_directory": str(tmp_path / "outputs"),
            "output_run_name": "golden_stub",
            "supply_demand_iters": 1,
            "models": {
                "land_use": "urbansim",
                "travel": "beam",
                "activity_demand": "activitysim",
                "vehicle_ownership": "atlas",
            },
        },
        "shared": {
            "geography": {
                "FIPS": {"county": ["00001"]},
                "local_crs": "EPSG:4326",
            },
            "skims": {"fname": "skims.omx"},
            "database": {
                "enabled": False,
                "type": "duckdb",
                "path": str(tmp_path / "db.duckdb"),
            },
        },
        "infrastructure": {
            "container_manager": "docker",
            "singularity_images": {},
            "docker_images": {},
            "docker_config": {"stdout": False, "pull_latest": False},
        },
        "urbansim": {
            "local_data_input_folder": str(tmp_path / "usim_input"),
            "local_mutable_data_folder": "urbansim/data",
            "client_base_folder": "/usim",
            "client_data_folder": "/usim/data",
            "input_file_template": "usim_{region_id}.h5",
            "input_file_template_year": "usim_{region_id}_{year}.h5",
            "output_file_template": "usim_{year}.h5",
            "command_template": "run_usim",
            "region_mappings": {"region_to_region_id": {"test": "000"}},
        },
        "atlas": {
            "host_input_folder": "atlas/input",
            "warmstart_input_folder": "atlas/warmstart",
            "host_mutable_input_folder": "atlas/atlas_input",
            "host_output_folder": "atlas/atlas_output",
            "container_input_folder": "/atlas/input",
            "container_output_folder": "/atlas/output",
            "basedir": "/atlas",
            "codedir": "/atlas/code",
            "command_template": "atlas {0}",
        },
        "activitysim": {
            "local_input_folder": "activitysim/input",
            "local_mutable_data_folder": "activitysim/data",
            "local_output_folder": "activitysim/output",
            "local_configs_folder": "activitysim/configs",
            "local_mutable_configs_folder": "activitysim/configs_mutable",
            "validation_folder": "activitysim/validation",
            "command_template": "asim run",
            "final_plans_folder": "activitysim/final_plans",
            "region_mappings": {"region_to_subdir": {"test": "test"}},
        },
        "beam": {
            "config": "beam.conf",
            "local_input_folder": "beam/input",
            "local_mutable_data_folder": "beam/input",
            "local_output_folder": "beam/output",
            "scenario_folder": "beam/scenario",
            "router_directory": "router",
            "skims_shapefile": "beam/skims.shp",
            "skim_zone_source_id_col": "id",
            "skim_zone_geoid_col": "geoid",
        },
    }
    config_path = tmp_path / "settings.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return load_config(str(config_path))


@pytest.fixture
def golden_stub_env(tmp_path, monkeypatch):
    consist = pytest.importorskip("consist")

    settings = _build_settings(tmp_path)
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = True
    settings.replanning_enabled = False
    settings.state_file_loc = str(tmp_path / "state.yaml")
    settings.atlas.beamac = 0

    workspace = Workspace(settings, output_path=str(tmp_path), folder_name="run")
    state = WorkflowState.from_settings(settings)
    Path(settings.urbansim.local_data_input_folder).mkdir(parents=True, exist_ok=True)

    usim_dir = Path(workspace.get_usim_mutable_data_dir())
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    asim_configs_dir = Path(workspace.get_asim_mutable_configs_dir())
    asim_out_dir = Path(workspace.get_asim_output_dir())
    beam_dir = Path(workspace.get_beam_mutable_data_dir())
    atlas_input_dir = Path(workspace.get_atlas_mutable_input_dir())
    atlas_output_dir = Path(workspace.get_atlas_output_dir())

    for path in (
        usim_dir,
        asim_dir,
        asim_configs_dir,
        asim_out_dir,
        beam_dir,
        atlas_input_dir,
        atlas_output_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    region_id = settings.urbansim.region_mappings["region_to_region_id"][
        settings.run.region
    ]
    usim_input_path = usim_dir / settings.urbansim.input_file_template.format(
        region_id=region_id
    )
    usim_output_path = usim_dir / settings.urbansim.output_file_template.format(
        year=state.forecast_year
    )
    usim_merged_path = usim_dir / f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}.h5"
    _write_usim_toy_h5(usim_input_path)
    _write_usim_toy_h5(usim_output_path)
    _write_usim_toy_h5(usim_merged_path, with_year_prefix=2019)
    _write_usim_toy_h5(usim_dir / "usim_2017.h5")
    _write_usim_toy_h5(usim_dir / "usim_2019.h5", with_year_prefix=2019)

    land_use_path = asim_dir / "land_use.csv"
    households_path = asim_dir / "households.csv"
    persons_path = asim_dir / "persons.csv"
    omx_path = asim_dir / "skims.omx"
    _write_csv(
        land_use_path,
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
        households_path,
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
        persons_path,
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
                "workplace_taz": [2, 2, -1],
                "school_taz": [-1, -1, 2],
                "ptype": [1, 1, 6],
                "pemploy": [1, 1, 4],
                "pstudent": [3, 3, 1],
                "home_x": [0.0, 0.1, 1.0],
                "home_y": [0.0, 0.2, 1.1],
            }
        ),
    )
    _write_file(omx_path)

    zarr_path = asim_out_dir / "cache" / "skims.zarr"
    _write_file(zarr_path)
    asim_households_out_path = (
        asim_out_dir / "final_pipeline" / "households" / "final.parquet"
    )
    asim_persons_out_path = asim_out_dir / "final_pipeline" / "persons" / "final.parquet"
    asim_tours_out_path = asim_out_dir / "final_pipeline" / "tours" / "final.parquet"
    asim_trips_out_path = asim_out_dir / "final_pipeline" / "trips" / "final.parquet"
    asim_beam_plans_out_path = (
        asim_out_dir / "final_pipeline" / "beam_plans" / "final.parquet"
    )
    _write_parquet(
        asim_households_out_path,
        pd.DataFrame(
            {
                "household_id": [1, 2],
                "block_id": [1001, 1002],
                "home_zone_id": [1, 2],
                "income": [100000.0, 70000.0],
                "hhsize": [2, 1],
                "HHT": [4, 1],
                "auto_ownership": [1, 2],
            }
        ),
    )
    _write_parquet(
        asim_persons_out_path,
        pd.DataFrame(
            {
                "person_id": [11, 12, 21],
                "household_id": [1, 1, 2],
                "age": [40, 35, 17],
                "PNUM": [1, 2, 1],
                "sex": [1, 2, 1],
                "pemploy": [1, 1, 4],
                "pstudent": [3, 3, 1],
                "ptype": [1, 1, 6],
                "home_x": [0.0, 0.1, 1.0],
                "home_y": [0.0, 0.2, 1.1],
                "home_zone_id": [1, 1, 2],
                "is_worker": [True, True, False],
            }
        ),
    )
    _write_parquet(
        asim_tours_out_path,
        pd.DataFrame(
            {
                "tour_id": [100, 200],
                "person_id": [11, 21],
                "household_id": [1, 2],
                "tour_type": ["work", "school"],
                "tour_num": [1, 1],
                "tour_count": [1, 1],
                "primary_purpose": ["work", "school"],
                "destination": [2.0, 2.0],
                "origin": [1.0, 2.0],
                "start": [8.0, 7.5],
                "end": [17.0, 15.0],
                "duration": [9.0, 7.5],
                "tour_mode": ["DRIVEALONE", "WALK"],
            }
        ),
    )
    _write_parquet(
        asim_trips_out_path,
        pd.DataFrame(
            {
                "trip_id": [1001, 1002, 2001],
                "person_id": [11, 11, 21],
                "household_id": [1, 1, 2],
                "tour_id": [100, 100, 200],
                "primary_purpose": ["work", "work", "school"],
                "trip_num": [1, 2, 1],
                "outbound": [True, False, True],
                "trip_count": [2, 2, 1],
                "destination": [2, 1, 2],
                "origin": [1, 2, 2],
                "purpose": ["work", "home", "school"],
                "depart": [8.0, 17.0, 7.5],
                "trip_mode": ["DRIVEALONE", "DRIVEALONE", "WALK"],
            }
        ),
    )
    _write_parquet(
        asim_beam_plans_out_path,
        pd.DataFrame(
            {
                "tour_id": [100, 100, 200],
                "trip_id": [1001, 1002, 2001],
                "person_id": [11, 11, 21],
                "number_of_participants": [1.0, 1.0, 1.0],
                "tour_mode": ["DRIVEALONE", "DRIVEALONE", "WALK"],
                "trip_mode": ["DRIVEALONE", "DRIVEALONE", "WALK"],
                "PlanElementIndex": [0, 1, 0],
                "ActivityElement": ["activity", "activity", "activity"],
                "ActivityType": ["home", "work", "school"],
                "x": [0.0, 1.0, 1.0],
                "y": [0.0, 1.0, 1.1],
                "departure_time": [8.0, 17.0, 7.5],
                "trip_dur_min": [30.0, 32.0, 20.0],
                "trip_cost_dollars": [3.5, 3.5, 0.0],
            }
        ),
    )

    beam_plans_path = beam_dir / "plans.csv"
    beam_households_path = beam_dir / "households.csv"
    beam_persons_path = beam_dir / "persons.csv"
    beam_linkstats_path = beam_dir / "linkstats.csv.gz"
    _write_file(beam_plans_path)
    _write_file(beam_households_path)
    _write_file(beam_persons_path)
    _write_file(beam_linkstats_path)

    promoted_linkstats = Path(workspace.get_beam_output_dir()) / "promoted_linkstats.csv.gz"
    promoted_plans = Path(workspace.get_beam_output_dir()) / "promoted_plans.parquet"
    _write_file(promoted_linkstats)
    _write_file(promoted_plans)

    def record_builder(model_name, phase, state=None, workspace=None, raw_outputs=None):
        if phase == "preprocess":
            if model_name == "activitysim":
                return RecordStore(
                    recordList=[
                        FileRecord(file_path=str(land_use_path), short_name=ASIM_LAND_USE_IN),
                        FileRecord(file_path=str(households_path), short_name=ASIM_HOUSEHOLDS_IN),
                        FileRecord(file_path=str(persons_path), short_name=ASIM_PERSONS_IN),
                        FileRecord(file_path=str(omx_path), short_name=ASIM_OMX_SKIMS),
                    ]
                )
            if model_name == "beam":
                return RecordStore(
                    recordList=[
                        FileRecord(file_path=str(beam_plans_path), short_name=BEAM_PLANS_IN),
                        FileRecord(file_path=str(beam_households_path), short_name=BEAM_HOUSEHOLDS_IN),
                        FileRecord(file_path=str(beam_persons_path), short_name=BEAM_PERSONS_IN),
                        FileRecord(file_path=str(beam_linkstats_path), short_name=LINKSTATS_WARMSTART),
                    ]
                )
            return RecordStore()
        if phase == "run":
            if model_name == "urbansim":
                return RecordStore(
                    recordList=[
                        FileRecord(file_path=str(usim_output_path), short_name=USIM_FORECAST_OUTPUT)
                    ]
                )
            if model_name == "activitysim_compile":
                return RecordStore(
                    recordList=[FileRecord(file_path=str(zarr_path), short_name=ZARR_SKIMS)]
                )
            if model_name == "activitysim":
                return RecordStore(
                    recordList=[
                        FileRecord(
                            file_path=str(asim_households_out_path),
                            short_name="households_asim_out_temp",
                        ),
                        FileRecord(
                            file_path=str(asim_persons_out_path),
                            short_name="persons_asim_out_temp",
                        ),
                        FileRecord(
                            file_path=str(asim_tours_out_path),
                            short_name="tours_asim_out_temp",
                        ),
                        FileRecord(
                            file_path=str(asim_trips_out_path),
                            short_name="trips_asim_out_temp",
                        ),
                        FileRecord(
                            file_path=str(asim_beam_plans_out_path),
                            short_name="beam_plans_asim_out_temp",
                        ),
                    ]
                )
            if model_name == "beam":
                return RecordStore(
                    recordList=[
                        FileRecord(file_path=str(promoted_linkstats), short_name=LINKSTATS),
                        FileRecord(file_path=str(promoted_plans), short_name=BEAM_PLANS_OUT),
                    ]
                )
            if model_name == "atlas":
                assert state is not None
                assert workspace is not None
                output_year = state.forecast_year
                atlas_output_dir = Path(workspace.get_atlas_output_dir())
                hhv_path = atlas_output_dir / f"householdv_{output_year}.csv"
                veh_path = atlas_output_dir / f"vehicles_{output_year}.csv"
                hhv_path.parent.mkdir(parents=True, exist_ok=True)

                pd.DataFrame(
                    {
                        "household_id": [1, 2],
                        "nvehicles": [1, 2],
                    }
                ).to_csv(hhv_path, index=False)

                pd.DataFrame(
                    {
                        "household_id": [1, 2],
                        "vehicle_id": [1, 1],
                        "bodytype": ["sedan", "suv"],
                        "pred_power": ["gasoline", "electricity"],
                        "modelyear": [2018, 2020],
                    }
                ).to_csv(veh_path, index=False)

                return RecordStore(
                    recordList=[
                        FileRecord(
                            file_path=str(hhv_path),
                            short_name=f"householdv_{output_year}",
                        ),
                        FileRecord(
                            file_path=str(veh_path),
                            short_name=f"vehicles_{output_year}",
                        ),
                    ]
                )
            return RecordStore()
        if phase == "postprocess":
            if model_name == "urbansim":
                return RecordStore(
                    recordList=[
                        FileRecord(
                            file_path=str(usim_merged_path),
                            short_name=f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}",
                        )
                    ]
                )
            if model_name == "activitysim":
                assert state is not None
                assert workspace is not None
                assert raw_outputs is not None

                iter_dir = Path(workspace.get_asim_output_dir()) / (
                    f"year-{state.current_year}-iteration-{state.current_inner_iter}"
                )
                iter_dir.mkdir(parents=True, exist_ok=True)
                output_records = []
                for rec in raw_outputs.all_records():
                    source_path = Path(rec.file_path)
                    if not source_path.is_absolute():
                        source_path = Path(workspace.full_path) / source_path
                    clean_name = re.sub(r"_asim_out_temp$", "", rec.short_name or "")
                    target_path = iter_dir / f"{clean_name}.parquet"
                    if source_path.exists():
                        shutil.copy2(source_path, target_path)
                    output_records.append(
                        FileRecord(
                            file_path=str(target_path),
                            short_name=normalize_asim_output_key(clean_name),
                            description=f"ActivitySim output file: {clean_name}",
                            year=state.forecast_year,
                            iteration=state.current_inner_iter,
                        )
                    )
                return RecordStore(recordList=output_records)
            return RecordStore()
        return RecordStore()

    from pilates.generic.model_factory import ModelFactory
    from pilates.atlas.preprocessor import AtlasPreprocessor
    from pilates.atlas.postprocessor import AtlasPostprocessor

    def _make_preprocessor(self, model_name, state=None, major_stage=None):
        if model_name == "atlas":
            return AtlasPreprocessor(model_name, state, major_stage)
        return DummyPreprocessor(model_name, record_builder, state=state)

    def _make_runner(self, model_name, state=None, major_stage=None):
        return DummyRunner(model_name, record_builder, state=state)

    def _make_postprocessor(self, model_name, state=None, major_stage=None):
        if model_name == "atlas":
            return AtlasPostprocessor(model_name, state, major_stage)
        return DummyPostprocessor(model_name, record_builder, state=state)

    monkeypatch.setattr(ModelFactory, "get_preprocessor", _make_preprocessor)
    monkeypatch.setattr(ModelFactory, "get_runner", _make_runner)
    monkeypatch.setattr(ModelFactory, "get_postprocessor", _make_postprocessor)

    tracker = consist.Tracker(
        run_dir=Path(workspace.full_path) / "consist_runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        mounts={
            "inputs": str(Path.cwd()),
            "workspace": str(workspace.full_path),
        },
        project_root=str(Path.cwd()),
    )

    cr.set_enabled(True)
    try:
        with cr.use_tracker(tracker):
            with cr.scenario(
                name="golden_stub_workflow",
                tracker=tracker,
                tags=["golden_stub_workflow"],
                model="test_orchestrator",
            ) as scenario:
                yield {
                    "settings": settings,
                    "workspace": workspace,
                    "state": state,
                    "scenario": scenario,
                    "coupler": scenario.coupler,
                    "tracker": tracker,
                    "usim_input_path": str(usim_input_path),
                }
    finally:
        cr.set_enabled(None)


def test_golden_stub_workflow_stage_contract_with_real_consist(golden_stub_env, tmp_path):
    settings = golden_stub_env["settings"]
    workspace = golden_stub_env["workspace"]
    state = golden_stub_env["state"]
    scenario = golden_stub_env["scenario"]
    coupler = golden_stub_env["coupler"]
    tracker = golden_stub_env["tracker"]

    outputs_holder_year = StepOutputsHolder()
    usim_inputs = run_land_use_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        outputs_holder_year=outputs_holder_year,
    )
    assert USIM_DATASTORE_BASE_H5 in usim_inputs
    assert USIM_DATASTORE_CURRENT_H5 in usim_inputs
    assert usim_inputs[USIM_DATASTORE_BASE_H5].endswith("usim_000.h5")
    assert usim_inputs[USIM_DATASTORE_CURRENT_H5].endswith(
        f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}.h5"
    )
    assert coupler.get(USIM_DATASTORE_BASE_H5) is not None
    assert coupler.get(USIM_DATASTORE_H5) is not None

    run_vehicle_ownership_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        build_atlas_static_inputs_fallback=lambda _workspace: {},
    )
    assert (Path(workspace.get_atlas_mutable_input_dir()) / "year2017" / "households.csv").exists()
    assert (Path(workspace.get_atlas_output_dir()) / "vehicles2_2017.csv").exists()
    assert not (Path(workspace.get_atlas_output_dir()) / "vehicles2_2019.csv").exists()
    asim_mutable_dir = Path(workspace.get_asim_mutable_data_dir())
    land_use_cols = set(pd.read_csv(asim_mutable_dir / "land_use.csv").columns)
    households_cols = set(pd.read_csv(asim_mutable_dir / "households.csv").columns)
    persons_cols = set(pd.read_csv(asim_mutable_dir / "persons.csv").columns)
    assert {"TAZ", "TOTPOP", "TOTHH", "TOTEMP"} <= land_use_cols
    assert {"household_id", "block_id", "income", "persons", "TAZ"} <= households_cols
    assert {"person_id", "household_id", "age", "TAZ", "ptype", "pemploy", "pstudent"} <= persons_cols

    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    manifest_dir = tmp_path / "workflow_manifests"

    def _build_manifest_path(_workspace, year, iteration):
        manifest_dir.mkdir(parents=True, exist_ok=True)
        return manifest_dir / f"manifest_{year}_{iteration}.yaml"

    asim_archive_year = state.current_year
    asim_archive_iteration = state.current_inner_iter
    run_supply_demand_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=_build_manifest_path,
    )

    assert coupler.get(ZARR_SKIMS) is not None
    assert coupler.get(LINKSTATS) is not None
    assert coupler.get(BEAM_PLANS_OUT) is not None
    assert (manifest_dir / f"manifest_{state.forecast_year}_0.yaml").exists()
    asim_output_dir = Path(workspace.get_asim_output_dir()) / "final_pipeline"
    households_out_cols = set(
        pd.read_parquet(asim_output_dir / "households" / "final.parquet").columns
    )
    persons_out_cols = set(
        pd.read_parquet(asim_output_dir / "persons" / "final.parquet").columns
    )
    tours_out_cols = set(
        pd.read_parquet(asim_output_dir / "tours" / "final.parquet").columns
    )
    trips_out_cols = set(
        pd.read_parquet(asim_output_dir / "trips" / "final.parquet").columns
    )
    beam_plans_out_cols = set(
        pd.read_parquet(asim_output_dir / "beam_plans" / "final.parquet").columns
    )
    assert {"household_id", "home_zone_id", "hhsize", "auto_ownership"} <= households_out_cols
    assert {"person_id", "household_id", "PNUM", "home_zone_id", "is_worker"} <= persons_out_cols
    assert {"tour_id", "person_id", "household_id", "tour_type", "tour_mode"} <= tours_out_cols
    assert {"trip_id", "tour_id", "person_id", "household_id", "trip_mode"} <= trips_out_cols
    assert {"tour_id", "trip_id", "person_id", "PlanElementIndex", "ActivityElement", "ActivityType"} <= beam_plans_out_cols
    asim_archive_dir = (
        Path(workspace.get_asim_output_dir())
        / f"year-{asim_archive_year}-iteration-{asim_archive_iteration}"
    )
    assert (asim_archive_dir / "households.parquet").exists()
    assert (asim_archive_dir / "persons.parquet").exists()
    assert (asim_archive_dir / "beam_plans.parquet").exists()

    runs = tracker.find_runs(tags=["golden_stub_workflow"])
    assert runs, "Expected at least one scenario run for golden stub workflow"
    scenario_run = runs[0]
    assert scenario_run.status in {"running", "completed"}
    assert "steps" in scenario_run.meta
    assert scenario_run.meta["steps"]

    scenario_artifacts = tracker.get_artifacts_for_run(scenario_run.id)
    assert len(scenario_artifacts.inputs) > 0
    assert len(scenario_artifacts.outputs) > 0

    report = write_provenance_report(
        tracker=tracker,
        run_id=scenario_run.id,
        output_path=Path(workspace.full_path) / "golden_stub_provenance_report.md",
    )
    report_path = Path(workspace.full_path) / "golden_stub_provenance_report.md"
    assert report_path.exists()
    assert "```mermaid" in report
