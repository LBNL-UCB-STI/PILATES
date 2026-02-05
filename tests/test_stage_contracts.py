from pathlib import Path

import pytest
import yaml

from pilates.config import load_config
from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_INPUT_MERGED_PREFIX,
    ZARR_SKIMS,
)
from pilates.workspace import Workspace
from pilates.workflows.stages.land_use import run_land_use_stage
from pilates.workflows.stages.supply_demand import run_supply_demand_stage
from pilates.workflows.stages.vehicle_ownership import run_vehicle_ownership_stage
from workflow_state import WorkflowState


class CouplerStub:
    def __init__(self) -> None:
        self._values = {}

    def set(self, key, value) -> None:
        self._values[key] = value

    def set_from_artifact(self, key, value) -> None:
        self._values[key] = value

    def get(self, key, default=None):
        return self._values.get(key, default)

    def pop(self, key, default=None):
        return self._values.pop(key, default)

    def keys(self):
        return list(self._values.keys())

    def require(self, key):
        if key not in self._values:
            raise KeyError(f"Coupler missing key={key!r}")
        return self._values[key]


class FakeScenario:
    def __init__(self, coupler: CouplerStub) -> None:
        self.coupler = coupler

    def run(self, **kwargs):
        import inspect

        inputs = kwargs.get("inputs") or {}
        input_keys = kwargs.get("input_keys") or []
        for key, value in inputs.items():
            self.coupler.set(key, value)
        for key in input_keys:
            self.coupler.require(key)

        runtime_kwargs = kwargs.get("runtime_kwargs") or {}
        fn_kwargs = dict(runtime_kwargs)
        fn = kwargs["fn"]
        sig = inspect.signature(fn)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )
        if accepts_kwargs:
            fn_kwargs.update(inputs)
            for key in input_keys:
                fn_kwargs.setdefault(key, self.coupler.get(key))
        else:
            allowed = set(sig.parameters.keys())
            for key, value in inputs.items():
                if key in allowed:
                    fn_kwargs[key] = value
            for key in input_keys:
                if key in allowed:
                    fn_kwargs.setdefault(key, self.coupler.get(key))
        fn(**fn_kwargs)
        return {"status": "ok"}


class DummyPreprocessor:
    def __init__(self, model_name, record_builder):
        self.model_name = model_name
        self._record_builder = record_builder

    def preprocess(self, workspace, previous_records=RecordStore()):
        return self._record_builder(self.model_name, "preprocess")


class DummyRunner:
    def __init__(self, model_name, record_builder):
        self.model_name = model_name
        self._record_builder = record_builder

    def run(self, input_store, workspace):
        return self._record_builder(self.model_name, "run")


class DummyPostprocessor:
    def __init__(self, model_name, record_builder):
        self.model_name = model_name
        self._record_builder = record_builder

    def postprocess(self, raw_outputs, workspace):
        return self._record_builder(self.model_name, "postprocess")


def _write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _build_settings(tmp_path: Path):
    config = {
        "run": {
            "region": "test",
            "scenario": "test",
            "start_year": 2017,
            "end_year": 2018,
            "output_directory": str(tmp_path / "outputs"),
            "output_run_name": "test_run",
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
            "database": {"enabled": False, "type": "duckdb", "path": str(tmp_path / "db.duckdb")},
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
    with config_path.open("w") as handle:
        yaml.safe_dump(config, handle)
    return load_config(str(config_path))


@pytest.fixture
def stage_env(tmp_path, monkeypatch):
    from pilates.utils import consist_runtime as cr

    cr.set_enabled(False)
    settings = _build_settings(tmp_path)
    settings.land_use_enabled = True
    settings.vehicle_ownership_model_enabled = True
    settings.activity_demand_enabled = True
    settings.traffic_assignment_enabled = True
    settings.replanning_enabled = False
    settings.state_file_loc = str(tmp_path / "state.yaml")

    workspace = Workspace(settings, output_path=str(tmp_path), folder_name="run")
    state = WorkflowState.from_settings(settings)

    usim_dir = Path(workspace.get_usim_mutable_data_dir())
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    asim_configs_dir = Path(workspace.get_asim_mutable_configs_dir())
    asim_out_dir = Path(workspace.get_asim_output_dir())
    beam_dir = Path(workspace.get_beam_mutable_data_dir())
    beam_out_dir = Path(workspace.get_beam_output_dir())
    atlas_input_dir = Path(workspace.get_atlas_mutable_input_dir())
    atlas_output_dir = Path(workspace.get_atlas_output_dir())

    for path in (
        usim_dir,
        asim_dir,
        asim_configs_dir,
        asim_out_dir,
        beam_dir,
        beam_out_dir,
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
    _write_file(usim_input_path)
    _write_file(usim_output_path)
    _write_file(usim_merged_path)

    land_use_path = asim_dir / "land_use.csv"
    households_path = asim_dir / "households.csv"
    persons_path = asim_dir / "persons.csv"
    omx_path = asim_dir / "skims.omx"
    _write_file(land_use_path)
    _write_file(households_path)
    _write_file(persons_path)
    _write_file(omx_path)

    zarr_path = asim_out_dir / "cache" / "skims.zarr"
    _write_file(zarr_path)

    beam_plans_path = beam_dir / "plans.csv"
    beam_households_path = beam_dir / "households.csv"
    beam_persons_path = beam_dir / "persons.csv"
    beam_linkstats_path = beam_dir / "linkstats.csv.gz"
    _write_file(beam_plans_path)
    _write_file(beam_households_path)
    _write_file(beam_persons_path)
    _write_file(beam_linkstats_path)

    def record_builder(model_name, phase):
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
            return RecordStore()
        return RecordStore()

    from pilates.generic.model_factory import ModelFactory

    def _make_preprocessor(self, model_name, state=None, major_stage=None):
        return DummyPreprocessor(model_name, record_builder)

    def _make_runner(self, model_name, state=None, major_stage=None):
        return DummyRunner(model_name, record_builder)

    def _make_postprocessor(self, model_name, state=None, major_stage=None):
        return DummyPostprocessor(model_name, record_builder)

    monkeypatch.setattr(ModelFactory, "get_preprocessor", _make_preprocessor)
    monkeypatch.setattr(ModelFactory, "get_runner", _make_runner)
    monkeypatch.setattr(ModelFactory, "get_postprocessor", _make_postprocessor)

    coupler = CouplerStub()
    scenario = FakeScenario(coupler)

    env = {
        "settings": settings,
        "workspace": workspace,
        "state": state,
        "coupler": coupler,
        "scenario": scenario,
        "usim_input_path": str(usim_input_path),
    }
    try:
        yield env
    finally:
        cr.set_enabled(None)


def test_land_use_stage_contract(stage_env):
    from pilates.workflows.steps import StepOutputsHolder

    outputs_holder = StepOutputsHolder()

    usim_inputs = run_land_use_stage(
        scenario=stage_env["scenario"],
        state=stage_env["state"],
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=stage_env["state"].forecast_year,
        outputs_holder_year=outputs_holder,
    )
    assert USIM_DATASTORE_H5 in usim_inputs
    assert stage_env["coupler"].get(USIM_DATASTORE_H5) is not None


def test_vehicle_ownership_stage_contract(stage_env):
    stage_env["coupler"].set(USIM_DATASTORE_H5, stage_env["usim_input_path"])
    run_vehicle_ownership_stage(
        scenario=stage_env["scenario"],
        state=stage_env["state"],
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=stage_env["state"].forecast_year,
        build_atlas_static_inputs_fallback=lambda workspace: {},
    )
    assert stage_env["coupler"].get(USIM_DATASTORE_H5) is not None


def test_supply_demand_stage_contract(stage_env, tmp_path):
    stage_env["coupler"].set(USIM_DATASTORE_H5, stage_env["usim_input_path"])
    usim_inputs = {USIM_DATASTORE_H5: stage_env["usim_input_path"]}
    state = stage_env["state"]
    state.current_major_stage = state.Stage.supply_demand_loop
    state.current_sub_stage = state.Stage.activity_demand
    state.current_inner_iter = 0

    def _build_manifest_path(workspace, year, iteration):
        return tmp_path / f"manifest_{year}_{iteration}.json"

    run_supply_demand_stage(
        scenario=stage_env["scenario"],
        state=stage_env["state"],
        settings=stage_env["settings"],
        workspace=stage_env["workspace"],
        coupler=stage_env["coupler"],
        year=stage_env["state"].forecast_year,
        usim_inputs=usim_inputs,
        build_manifest_path=_build_manifest_path,
    )
    assert stage_env["coupler"].get(ZARR_SKIMS) is not None
