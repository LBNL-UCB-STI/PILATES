"""
Golden stub workflow contract test (with real Consist runtime).

This module is intentionally broader than the focused invariant tests. It acts
as executable documentation for how a full PILATES year/iteration passes
through the staged workflow when:

1. The orchestrator uses real Consist scenario/step contexts.
2. Model execution is replaced with deterministic stubs that emit realistic
   records and file shapes.
3. Stage boundaries still honor the same coupler/input/output contracts used in
   production.

What this test demonstrates:
- Land use stage publishes UrbanSim datastore handles into coupler state.
- Vehicle ownership stage consumes datastore inputs and materializes expected
  Atlas side effects.
- Supply-demand stage executes ActivitySim + BEAM wiring, archives ActivitySim
  outputs, and writes a manifest.
- Provenance plumbing remains healthy: scenario runs/steps exist, artifacts are
  attached, and a markdown provenance report can be produced.

Why keep this large:
- It is a guardrail against regressions caused by refactors that are individually
  type-safe but break cross-stage integration behavior.
- It doubles as a reference implementation for developers adding new models or
  changing stage orchestration.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

import pytest
import pandas as pd
import yaml

from pilates.config import load_config
from pilates.atlas.outputs import AtlasRunOutputs
from pilates.activitysim.outputs import (
    ActivitySimCompileOutputs,
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
    normalize_asim_output_key,
)
from pilates.beam.outputs import (
    BeamPostprocessOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
)
from pilates.generic.records import RecordStore
from pilates.urbansim.outputs import (
    UrbanSimPostprocessOutputs,
    UrbanSimPreprocessOutputs,
    UrbanSimRunOutputs,
)
from pilates.utils import consist_runtime as cr
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.utils.provenance_report import write_provenance_report
from pilates.workspace import Workspace
from pilates.workflows.artifact_keys import (
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


EXPECTED_STAGE_MODELS = (
    "initialization",
    "urbansim_preprocess",
    "urbansim_run",
    "urbansim_postprocess",
    "atlas_preprocess",
    "atlas_run",
    "atlas_postprocess",
    "activitysim_preprocess",
    "activitysim_compile",
    "activitysim_run",
    "activitysim_postprocess",
    "beam_preprocess",
    "beam_run",
    "beam_postprocess",
)

EXPECTED_MANIFEST_STEPS = {
    "activitysim_preprocess",
    "activitysim_run",
    "activitysim_postprocess",
}

EXPECTED_ASIM_TEMP_OUTPUT_KEYS = {
    "accessibility_asim_out_temp",
    "disaggregate_accessibility_asim_out_temp",
    "joint_tour_participants_asim_out_temp",
    "land_use_asim_out_temp",
    "non_mandatory_tour_destination_accessibility_asim_out_temp",
    "households_asim_out_temp",
    "persons_asim_out_temp",
    "tours_asim_out_temp",
    "trips_asim_out_temp",
    "beam_plans_asim_out_temp",
}

EXPECTED_ASIM_ARCHIVE_OUTPUT_KEYS = {
    normalize_asim_output_key("accessibility"),
    normalize_asim_output_key("disaggregate_accessibility"),
    normalize_asim_output_key("joint_tour_participants"),
    normalize_asim_output_key("land_use"),
    normalize_asim_output_key("non_mandatory_tour_destination_accessibility"),
    normalize_asim_output_key("households"),
    normalize_asim_output_key("persons"),
    normalize_asim_output_key("tours"),
    normalize_asim_output_key("trips"),
    normalize_asim_output_key("beam_plans"),
}


def _artifact_map(raw_artifacts):
    """Normalize Consist artifact collections to a ``{key: artifact}`` mapping."""
    if isinstance(raw_artifacts, dict):
        return raw_artifacts
    artifact_map = {}
    for idx, artifact in enumerate(raw_artifacts or []):
        key = getattr(artifact, "key", None) or f"artifact_{idx}"
        artifact_map[key] = artifact
    return artifact_map


def _artifact_keys(raw_artifacts):
    return set(_artifact_map(raw_artifacts).keys())


class DummyPreprocessor:
    """Deterministic preprocessor stub delegated to ``record_builder``."""

    def __init__(self, model_name, record_builder, state=None):
        self.model_name = model_name
        self._record_builder = record_builder
        self._state = state

    def preprocess(self, workspace, previous_records=None, **kwargs):
        return self._record_builder(
            self.model_name,
            "preprocess",
            state=self._state,
            workspace=workspace,
        )


class DummyRunner:
    """Deterministic runner stub delegated to ``record_builder``."""

    def __init__(self, model_name, record_builder, state=None):
        self.model_name = model_name
        self._record_builder = record_builder
        self._state = state

    def run(self, input_store, workspace, **kwargs):
        return self._record_builder(
            self.model_name,
            "run",
            state=self._state,
            workspace=workspace,
            input_store=input_store,
        )


class DummyPostprocessor:
    """Deterministic postprocessor stub delegated to ``record_builder``."""

    def __init__(self, model_name, record_builder, state=None):
        self.model_name = model_name
        self._record_builder = record_builder
        self._state = state

    def postprocess(self, raw_outputs, workspace, **kwargs):
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
    """Create a minimal UrbanSim-style HDF5 with core tables used in tests."""
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
    """Build a compact, fully-enabled workflow config for golden-path testing."""
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
    """
    Assemble a full workflow harness with real Consist and stub model behavior.

    The fixture creates realistic input/output files, patches ``ModelFactory``
    to return deterministic stubs, and yields a live Consist ``scenario`` with
    a working coupler so the stage runners execute their normal orchestration
    logic.
    """

    consist = pytest.importorskip("consist")
    from pilates.workflows import step_consist_meta as step_consist_meta_module
    from pilates.workflows.steps import shared as shared_steps_module

    original_consist_step_meta = step_consist_meta_module.consist_step_meta

    def _patched_consist_step_meta(model):
        meta = dict(original_consist_step_meta(model))
        if str(model).startswith("beam_"):
            meta["adapter"] = lambda _ctx: None
        return meta

    monkeypatch.setattr(
        step_consist_meta_module,
        "consist_step_meta",
        _patched_consist_step_meta,
    )
    monkeypatch.setattr(
        shared_steps_module,
        "consist_step_meta",
        _patched_consist_step_meta,
    )

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

    beam_config_path = beam_dir / settings.run.region / settings.beam.config
    _write_file(beam_config_path)

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
    sharrow_cache_dir = Path(workspace.full_path) / "shared_cache" / "numba"
    _write_file(zarr_path)
    _write_file(sharrow_cache_dir / "compile-cache.bin")
    asim_households_out_path = (
        asim_out_dir / "final_pipeline" / "households" / "final.parquet"
    )
    asim_accessibility_out_path = (
        asim_out_dir / "final_pipeline" / "accessibility" / "final.parquet"
    )
    asim_disagg_accessibility_out_path = (
        asim_out_dir
        / "final_pipeline"
        / "disaggregate_accessibility"
        / "final.parquet"
    )
    asim_joint_tour_participants_out_path = (
        asim_out_dir
        / "final_pipeline"
        / "joint_tour_participants"
        / "final.parquet"
    )
    asim_land_use_out_path = (
        asim_out_dir / "final_pipeline" / "land_use" / "final.parquet"
    )
    asim_non_mandatory_accessibility_out_path = (
        asim_out_dir
        / "final_pipeline"
        / "non_mandatory_tour_destination_accessibility"
        / "final.parquet"
    )
    asim_persons_out_path = asim_out_dir / "final_pipeline" / "persons" / "final.parquet"
    asim_tours_out_path = asim_out_dir / "final_pipeline" / "tours" / "final.parquet"
    asim_trips_out_path = asim_out_dir / "final_pipeline" / "trips" / "final.parquet"
    asim_beam_plans_out_path = (
        asim_out_dir / "final_pipeline" / "beam_plans" / "final.parquet"
    )
    _write_parquet(
        asim_accessibility_out_path,
        pd.DataFrame({"person_id": [11, 21], "accessibility": [1.2, 0.8]}),
    )
    _write_parquet(
        asim_disagg_accessibility_out_path,
        pd.DataFrame({"person_id": [11, 21], "zone_id": [1, 2], "utility": [0.5, 0.3]}),
    )
    _write_parquet(
        asim_joint_tour_participants_out_path,
        pd.DataFrame({"tour_id": [100], "person_id": [12], "participant_num": [1]}),
    )
    _write_parquet(
        asim_land_use_out_path,
        pd.DataFrame({"zone_id": [1, 2], "TOTPOP": [120, 80], "TOTEMP": [15, 20]}),
    )
    _write_parquet(
        asim_non_mandatory_accessibility_out_path,
        pd.DataFrame({"person_id": [11, 21], "destination": [2, 1], "utility": [0.7, 0.4]}),
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
    final_skims_omx = Path(workspace.get_beam_output_dir()) / "final_skims.omx"
    _write_file(promoted_linkstats)
    _write_file(promoted_plans)
    _write_file(final_skims_omx)

    def record_builder(
        model_name,
        phase,
        state=None,
        workspace=None,
        raw_outputs=None,
        input_store=None,
    ):
        """
        Build deterministic RecordStore responses for each model/phase pair.

        Contract intent:
        - ``preprocess`` emits explicit upstream dependencies for the next step.
        - ``run`` emits realistic model outputs with production short-names.
        - ``postprocess`` emits finalized artifacts consumed by downstream stages.
        """
        if phase == "preprocess":
            if model_name == "urbansim":
                assert workspace is not None
                return UrbanSimPreprocessOutputs(
                    usim_mutable_data_dir=Path(workspace.get_usim_mutable_data_dir()),
                    prepared_inputs={
                        USIM_DATASTORE_BASE_H5: usim_input_path,
                        USIM_DATASTORE_CURRENT_H5: usim_input_path,
                    },
                )
            if model_name == "activitysim":
                assert workspace is not None
                return ActivitySimPreprocessOutputs(
                    mutable_data_dir=Path(workspace.get_asim_mutable_data_dir()),
                    land_use_table=land_use_path,
                    households_table=households_path,
                    persons_table=persons_path,
                    omx_skims=omx_path,
                )
            if model_name == "beam":
                assert workspace is not None
                return BeamPreprocessOutputs(
                    beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
                    prepared_inputs={
                        BEAM_PLANS_IN: beam_plans_path,
                        BEAM_HOUSEHOLDS_IN: beam_households_path,
                        BEAM_PERSONS_IN: beam_persons_path,
                        LINKSTATS_WARMSTART: beam_linkstats_path,
                    },
                )
            return RecordStore()
        if phase == "run":
            if model_name == "urbansim":
                return UrbanSimRunOutputs(
                    usim_datastore_h5=usim_output_path,
                    raw_outputs={
                        USIM_FORECAST_OUTPUT: usim_output_path,
                    },
                )
            if model_name == "activitysim_compile":
                return ActivitySimCompileOutputs(
                    zarr_skims=zarr_path,
                    sharrow_cache_dir=sharrow_cache_dir,
                )
            if model_name == "activitysim":
                assert workspace is not None
                return ActivitySimRunOutputs(
                    output_dir=Path(workspace.get_asim_output_dir()),
                    raw_outputs={
                        "accessibility_asim_out_temp": asim_accessibility_out_path,
                        "disaggregate_accessibility_asim_out_temp": asim_disagg_accessibility_out_path,
                        "joint_tour_participants_asim_out_temp": asim_joint_tour_participants_out_path,
                        "land_use_asim_out_temp": asim_land_use_out_path,
                        "non_mandatory_tour_destination_accessibility_asim_out_temp": asim_non_mandatory_accessibility_out_path,
                        "households_asim_out_temp": asim_households_out_path,
                        "persons_asim_out_temp": asim_persons_out_path,
                        "tours_asim_out_temp": asim_tours_out_path,
                        "trips_asim_out_temp": asim_trips_out_path,
                        "beam_plans_asim_out_temp": asim_beam_plans_out_path,
                    },
                )
            if model_name == "beam":
                assert workspace is not None
                return BeamRunOutputs(
                    beam_output_dir=Path(workspace.get_beam_output_dir()),
                    raw_outputs={
                        LINKSTATS: promoted_linkstats,
                        BEAM_PLANS_OUT: promoted_plans,
                    },
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

                return AtlasRunOutputs(
                    atlas_output_dir=atlas_output_dir,
                    raw_outputs={
                        f"householdv_{output_year}": hhv_path,
                        f"vehicles_{output_year}": veh_path,
                    },
                )
            return RecordStore()
        if phase == "postprocess":
            if model_name == "urbansim":
                return UrbanSimPostprocessOutputs(
                    usim_datastore_h5=usim_merged_path,
                    processed_outputs={
                        f"{USIM_INPUT_MERGED_PREFIX}{state.forecast_year}": usim_merged_path,
                    },
                )
            if model_name == "activitysim":
                assert state is not None
                assert workspace is not None
                assert raw_outputs is not None
                assert isinstance(raw_outputs, ActivitySimRunOutputs)

                iter_dir = Path(workspace.get_asim_output_dir()) / (
                    f"year-{state.current_year}-iteration-{state.current_inner_iter}"
                )
                iter_dir.mkdir(parents=True, exist_ok=True)
                processed_outputs = {}
                for short_name, source_path in raw_outputs.raw_outputs.items():
                    source_path = Path(source_path)
                    if not source_path.is_absolute():
                        source_path = Path(workspace.full_path) / source_path
                    clean_name = re.sub(r"_asim_out_temp$", "", short_name or "")
                    target_path = iter_dir / f"{clean_name}.parquet"
                    if source_path.exists():
                        shutil.copy2(source_path, target_path)
                    processed_outputs[normalize_asim_output_key(clean_name)] = target_path
                return ActivitySimPostprocessOutputs(
                    usim_datastore_h5=usim_merged_path,
                    asim_output_dir=Path(workspace.get_asim_output_dir()),
                    processed_outputs=processed_outputs,
                    usim_datastore_key=USIM_DATASTORE_H5,
                )
            if model_name == "beam":
                return BeamPostprocessOutputs(
                    zarr_skims=zarr_path,
                    final_skims_omx=final_skims_omx,
                )
            return RecordStore()
        return RecordStore()

    from pilates.generic.model_factory import ModelFactory
    from pilates.atlas.preprocessor import AtlasPreprocessor
    from pilates.atlas.postprocessor import AtlasPostprocessor

    def _make_preprocessor(self, model_name, state=None, *_args, **_kwargs):
        if model_name == "atlas":
            return AtlasPreprocessor(model_name, state)
        return DummyPreprocessor(model_name, record_builder, state=state)

    def _make_runner(self, model_name, state=None, *_args, **_kwargs):
        return DummyRunner(model_name, record_builder, state=state)

    def _make_postprocessor(self, model_name, state=None, *_args, **_kwargs):
        if model_name == "atlas":
            return AtlasPostprocessor(model_name, state)
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
                    "usim_merged_path": str(usim_merged_path),
                    "zarr_path": str(zarr_path),
                    "sharrow_cache_dir": str(sharrow_cache_dir),
                    "promoted_linkstats": str(promoted_linkstats),
                    "promoted_plans": str(promoted_plans),
                }
    finally:
        cr.set_enabled(None)


def test_golden_stub_workflow_stage_contract_with_real_consist(golden_stub_env, tmp_path):
    """
    End-to-end narrative test for stage contracts and provenance continuity.

    The assertions are grouped by workflow phase:
    - Phase 1: land use contract and coupler publication.
    - Phase 2: vehicle ownership contract and Atlas/ActivitySim side effects.
    - Phase 3: supply-demand contract, manifest creation, and archived outputs.
    - Phase 4: Consist run/artifact/report integrity.
    """
    settings = golden_stub_env["settings"]
    workspace = golden_stub_env["workspace"]
    state = golden_stub_env["state"]
    scenario = golden_stub_env["scenario"]
    coupler = golden_stub_env["coupler"]
    tracker = golden_stub_env["tracker"]
    usim_input_path = Path(golden_stub_env["usim_input_path"])
    usim_merged_path = Path(golden_stub_env["usim_merged_path"])
    zarr_path = Path(golden_stub_env["zarr_path"])
    promoted_linkstats = Path(golden_stub_env["promoted_linkstats"])
    promoted_plans = Path(golden_stub_env["promoted_plans"])

    # Phase 0: initialization contract represented explicitly in this golden test.
    assert state.data_initialized is False
    init_marker = Path(workspace.full_path) / ".golden_init_marker.txt"
    with scenario.trace(
        "initialization",
        model="initialization",
        year=state.current_year,
        iteration=0,
        tags=["init"],
    ):
        cr.log_input(usim_input_path, key="golden_init_source")
        _write_file(init_marker, "initialized")
        cr.log_output(init_marker, key="golden_init_marker")
        state.set_data_initialized(True)
    assert state.data_initialized is True
    assert init_marker.exists()

    # Phase 1: land-use stage publishes UrbanSim datastore handles.
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
    coupler_base_h5 = artifact_to_path(coupler.get(USIM_DATASTORE_BASE_H5), workspace)
    coupler_current_h5 = artifact_to_path(coupler.get(USIM_DATASTORE_H5), workspace)
    assert coupler_base_h5 is not None
    assert coupler_current_h5 is not None
    assert Path(coupler_base_h5).resolve() == Path(usim_inputs[USIM_DATASTORE_BASE_H5]).resolve()
    assert Path(coupler_current_h5).resolve() == usim_merged_path.resolve()

    # Phase 2: vehicle ownership stage consumes datastore handles and
    # produces Atlas outputs plus ActivitySim-ready tables.
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
    vehicles2 = pd.read_csv(Path(workspace.get_atlas_output_dir()) / "vehicles2_2017.csv")
    assert {
        "household_id",
        "vehicle_id",
        "bodytype",
        "pred_power",
        "modelyear",
        "vehicleTypeId",
    } <= set(vehicles2.columns)
    assert vehicles2["vehicleTypeId"].tolist() == [
        "2018_Ford_Fusion_AWD",
        "2020_Tesla_Model_Y",
    ]
    asim_mutable_dir = Path(workspace.get_asim_mutable_data_dir())
    land_use = pd.read_csv(asim_mutable_dir / "land_use.csv")
    households = pd.read_csv(asim_mutable_dir / "households.csv")
    persons = pd.read_csv(asim_mutable_dir / "persons.csv")
    land_use_cols = set(land_use.columns)
    households_cols = set(households.columns)
    persons_cols = set(persons.columns)
    assert {"TAZ", "TOTPOP", "TOTHH", "TOTEMP"} <= land_use_cols
    assert {"household_id", "block_id", "income", "persons", "TAZ"} <= households_cols
    assert {"person_id", "household_id", "age", "TAZ", "ptype", "pemploy", "pstudent"} <= persons_cols
    assert int(land_use["TOTPOP"].sum()) == 200
    assert int(households["persons"].sum()) == 3
    assert int((persons["worker"] == 1).sum()) == 2

    # Phase 3: supply-demand loop (ActivitySim + BEAM) writes coupler outputs,
    # manifest state, and year/iteration archives.
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

    zarr_from_coupler = artifact_to_path(coupler.get(ZARR_SKIMS), workspace)
    linkstats_from_coupler = artifact_to_path(coupler.get(LINKSTATS), workspace)
    plans_from_coupler = artifact_to_path(coupler.get(BEAM_PLANS_OUT), workspace)
    assert zarr_from_coupler is not None
    assert linkstats_from_coupler is not None
    assert plans_from_coupler is not None
    assert Path(zarr_from_coupler).resolve() == zarr_path.resolve()
    assert Path(linkstats_from_coupler).resolve() == promoted_linkstats.resolve()
    assert Path(plans_from_coupler).resolve() == promoted_plans.resolve()

    manifest_path = manifest_dir / f"manifest_{state.forecast_year}_0.yaml"
    assert manifest_path.exists()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == EXPECTED_MANIFEST_STEPS
    for step_name in EXPECTED_MANIFEST_STEPS:
        step_manifest = manifest[step_name]
        assert isinstance(step_manifest.get("cache_hit"), bool)
        assert step_manifest.get("outputs")
    assert (
        set(manifest["activitysim_run"]["outputs"]["raw_outputs"])
        == EXPECTED_ASIM_TEMP_OUTPUT_KEYS
    )
    assert (
        set(manifest["activitysim_postprocess"]["outputs"]["processed_outputs"])
        == EXPECTED_ASIM_ARCHIVE_OUTPUT_KEYS
    )

    asim_output_dir = Path(workspace.get_asim_output_dir()) / "final_pipeline"
    households_out = pd.read_parquet(asim_output_dir / "households" / "final.parquet")
    persons_out = pd.read_parquet(asim_output_dir / "persons" / "final.parquet")
    tours_out = pd.read_parquet(asim_output_dir / "tours" / "final.parquet")
    trips_out = pd.read_parquet(asim_output_dir / "trips" / "final.parquet")
    beam_plans_out = pd.read_parquet(asim_output_dir / "beam_plans" / "final.parquet")
    households_out_cols = set(households_out.columns)
    persons_out_cols = set(persons_out.columns)
    tours_out_cols = set(tours_out.columns)
    trips_out_cols = set(trips_out.columns)
    beam_plans_out_cols = set(beam_plans_out.columns)
    assert {"household_id", "home_zone_id", "hhsize", "auto_ownership"} <= households_out_cols
    assert {"person_id", "household_id", "PNUM", "home_zone_id", "is_worker"} <= persons_out_cols
    assert {"tour_id", "person_id", "household_id", "tour_type", "tour_mode"} <= tours_out_cols
    assert {"trip_id", "tour_id", "person_id", "household_id", "trip_mode"} <= trips_out_cols
    assert {"tour_id", "trip_id", "person_id", "PlanElementIndex", "ActivityElement", "ActivityType"} <= beam_plans_out_cols
    assert households_out["auto_ownership"].tolist() == [1, 2]
    assert int(persons_out["is_worker"].sum()) == 2
    assert set(tours_out["tour_mode"]) == {"DRIVEALONE", "WALK"}
    assert len(trips_out) == 3
    assert set(beam_plans_out["ActivityType"]) == {"home", "work", "school"}

    asim_archive_dir = (
        Path(workspace.get_asim_output_dir())
        / f"year-{asim_archive_year}-iteration-{asim_archive_iteration}"
    )
    for key in EXPECTED_ASIM_ARCHIVE_OUTPUT_KEYS:
        archive_name = key.replace("_asim_out", "")
        assert (asim_archive_dir / f"{archive_name}.parquet").exists()

    # Phase 4: provenance sanity checks against real Consist tracker output.
    runs = tracker.find_runs(tags=["golden_stub_workflow"])
    assert runs, "Expected at least one scenario run for golden stub workflow"
    scenario_run = runs[0]
    assert scenario_run.status in {"running", "completed"}
    assert "steps" in scenario_run.meta
    assert "mounts" in scenario_run.meta
    steps = scenario_run.meta["steps"]
    assert steps
    assert [step.get("model") for step in steps] == list(EXPECTED_STAGE_MODELS)

    for step in steps:
        step_artifacts = tracker.get_artifacts_for_run(step["id"])
        step_input_keys = set((step.get("inputs") or {}).values())
        step_output_keys = set((step.get("outputs") or {}).values())
        assert step_output_keys, f"Expected outputs recorded for step {step['id']}"
        assert step_output_keys <= _artifact_keys(step_artifacts.outputs)
        assert step_input_keys <= _artifact_keys(step_artifacts.inputs)
        step_run = tracker.get_run(step["id"])
        assert step_run is not None
        assert "cache_epoch" in (step_run.meta or {})

    scenario_artifacts = tracker.get_artifacts_for_run(scenario_run.id)
    scenario_inputs = _artifact_map(scenario_artifacts.inputs)
    scenario_outputs = _artifact_map(scenario_artifacts.outputs)
    assert scenario_inputs
    assert scenario_outputs

    expected_scenario_output_keys = {
        USIM_DATASTORE_BASE_H5,
        USIM_DATASTORE_H5,
        ZARR_SKIMS,
        LINKSTATS,
        BEAM_PLANS_OUT,
        "atlas_vehicles2_output",
    } | EXPECTED_ASIM_ARCHIVE_OUTPUT_KEYS
    assert expected_scenario_output_keys <= set(scenario_outputs)

    step_output_keys = {
        key
        for step in steps
        for key in (step.get("outputs") or {}).values()
    }
    assert step_output_keys <= set(scenario_outputs)

    for key in expected_scenario_output_keys:
        artifact = scenario_outputs[key]
        artifact_path = getattr(artifact, "path", None)
        assert artifact_path is not None, f"Artifact {key} did not expose a concrete path"
        assert Path(artifact_path).exists(), f"Artifact path for {key} does not exist"

    zarr_meta = scenario_outputs[ZARR_SKIMS].meta or {}
    assert zarr_meta.get("year") == state.forecast_year
    assert zarr_meta.get("iteration") == 0

    linkstats_meta = scenario_outputs[LINKSTATS].meta or {}
    assert linkstats_meta.get("year") == state.forecast_year
    assert linkstats_meta.get("iteration") == 0

    for key in EXPECTED_ASIM_ARCHIVE_OUTPUT_KEYS:
        meta = scenario_outputs[key].meta or {}
        assert meta.get("year") == state.forecast_year
        assert meta.get("iteration") == 0
        description = str(meta.get("description", ""))
        assert description.startswith("ActivitySim output file:")

    report = write_provenance_report(
        tracker=tracker,
        run_id=scenario_run.id,
        output_path=Path(workspace.full_path) / "golden_stub_provenance_report.md",
    )
    report_path = Path(workspace.full_path) / "golden_stub_provenance_report.md"
    assert report_path.exists()
    assert "```mermaid" in report
