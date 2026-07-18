"""
Stubbed workflow matrix tests built on the golden Consist harness.

These tests complement the broad golden workflow by exercising smaller run
shapes that have broken in production:

1. BEAM-only supply-demand with no ActivitySim zarr handoff.
2. ActivitySim+BEAM supply-demand where OMX archive output is legitimately absent.
3. Land use + ATLAS stage wiring with stage-local output ownership checks.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from pilates.activitysim.outputs import (
    normalize_asim_output_key,
)
from pilates.beam.runner import BeamRunner
from pilates.atlas.outputs import AtlasRunOutputs
from pilates.atlas.runner import AtlasRunner
from pilates.config.models import ZonesConfig
from pilates.generic.records import FileRecord, RecordStore
from pilates.runtime.context import WorkflowRuntimeContext
from pilates.urbansim.outputs import UrbanSimRunOutputs
from pilates.urbansim.runner import UrbansimRunner
from pilates.utils import consist_runtime as cr
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    LINKSTATS,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_H5,
)
from pilates.workflows.stages.land_use import run_land_use_stage as _run_land_use_stage
from pilates.workflows.stages.supply_demand import (
    run_supply_demand_stage as _run_supply_demand_stage,
)
from pilates.workflows.stages.vehicle_ownership import (
    run_vehicle_ownership_stage as _run_vehicle_ownership_stage,
)
from pilates.workflows.steps import StepOutputsHolder
from tests.workflow_contract_harness import (
    write_file as _write_file,
    write_parquet as _write_parquet,
)
from workflow_state import WorkflowState


def run_land_use_stage(
    *, context=None, settings=None, state=None, workspace=None, surface=None, **kwargs
):
    kwargs.pop("outputs_holder_year", None)
    context = context or WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
    return _run_land_use_stage(context=context, **kwargs)


def run_vehicle_ownership_stage(
    *, context=None, settings=None, state=None, workspace=None, surface=None, **kwargs
):
    context = context or WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
    return _run_vehicle_ownership_stage(context=context, **kwargs)


def run_supply_demand_stage(
    *, context=None, settings=None, state=None, workspace=None, surface=None, **kwargs
):
    context = context or WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
        surface=surface,
    )
    return _run_supply_demand_stage(context=context, **kwargs)


def _reconfigure_models(
    settings,
    *,
    tmp_path: Path,
    state_file_name: str,
    land_use: bool,
    vehicle_ownership: bool,
    activity_demand: bool,
    traffic_assignment: bool,
) -> WorkflowState:
    settings.run.models.land_use = "urbansim" if land_use else None
    settings.run.models.vehicle_ownership = "atlas" if vehicle_ownership else None
    settings.run.models.activity_demand = "activitysim" if activity_demand else None
    settings.run.models.travel = "beam" if traffic_assignment else None

    settings.land_use_enabled = land_use
    settings.vehicle_ownership_model_enabled = vehicle_ownership
    settings.activity_demand_enabled = activity_demand
    settings.traffic_assignment_enabled = traffic_assignment

    runtime = getattr(settings, "runtime", None)
    flags = getattr(runtime, "flags", None)
    if flags is not None:
        flags.land_use_enabled = land_use
        flags.vehicle_ownership_model_enabled = vehicle_ownership
        flags.activity_demand_enabled = activity_demand
        flags.traffic_assignment_enabled = traffic_assignment

    state_file = tmp_path / state_file_name
    settings.state_file_loc = str(state_file)
    runtime_options = getattr(runtime, "options", None)
    if runtime_options is not None:
        runtime_options.state_file_loc = str(state_file)

    return WorkflowState.from_settings(settings)


def _mark_initialized(
    env: dict,
    state: WorkflowState,
    *,
    marker_name: str,
    publish_initial_usim_datastore: bool = False,
) -> None:
    workspace = env["workspace"]
    scenario = env["scenario"]
    source_path = Path(env["usim_input_path"])
    marker = Path(workspace.full_path) / marker_name
    marker_key = marker_name.lstrip(".").replace(".", "_")
    with scenario.trace(
        "initialization",
        model="initialization",
        year=state.current_year,
        iteration=0,
        tags=["init", "stub-matrix"],
    ):
        cr.log_input(source_path, key=f"{marker_key}_source")
        _write_file(marker, "initialized")
        cr.log_output(marker, key=f"{marker_key}_marker")
        if publish_initial_usim_datastore:
            base_datastore_artifact = cr.log_input(
                source_path, key=USIM_DATASTORE_BASE_H5
            )
            scenario.coupler.set_from_artifact(
                USIM_DATASTORE_BASE_H5, base_datastore_artifact
            )
            current_datastore_artifact = cr.log_input(
                source_path, key=USIM_DATASTORE_H5
            )
            scenario.coupler.set_from_artifact(
                USIM_DATASTORE_H5, current_datastore_artifact
            )
        state.set_data_initialized(True)


def _manifest_builder(root: Path, prefix: str):
    def _build(_workspace, year: int, iteration: int) -> Path:
        manifest_dir = root / prefix
        manifest_dir.mkdir(parents=True, exist_ok=True)
        return manifest_dir / f"{year}_{iteration}.yaml"

    return _build


def _publish_input_roles(
    scenario, state: WorkflowState, roles: dict[str, Path]
) -> None:
    with scenario.trace(
        "stub_input_publication",
        model="initialization",
        year=state.current_year,
        iteration=0,
        tags=["init", "stub-matrix"],
    ):
        for key, path in roles.items():
            scenario.coupler.set_from_artifact(key, cr.log_input(path, key=key))


def _scenario_run(env: dict):
    tracker = env["tracker"]
    runs = tracker.find_runs(tags=["golden_stub_workflow"])
    assert runs
    return runs[0]


def _steps_by_model(env: dict) -> dict[str, dict]:
    scenario_run = _scenario_run(env)
    return {step["model"]: step for step in scenario_run.meta["steps"]}


def _scenario_output_keys(env: dict) -> set[str]:
    tracker = env["tracker"]
    scenario_run = _scenario_run(env)
    outputs = tracker.get_artifacts_for_run(scenario_run.id).outputs
    return {
        getattr(artifact, "key", None)
        for artifact in outputs or []
        if getattr(artifact, "key", None)
    }


def test_stubbed_beam_only_supply_demand_runs_without_activitysim_zarr_inputs(
    golden_stub_env,
    monkeypatch,
    tmp_path: Path,
) -> None:
    env = golden_stub_env
    settings = env["settings"]
    workspace = env["workspace"]
    scenario = env["scenario"]
    coupler = scenario.coupler

    state = _reconfigure_models(
        settings,
        tmp_path=tmp_path,
        state_file_name="beam_only_state.yaml",
        land_use=False,
        vehicle_ownership=False,
        activity_demand=False,
        traffic_assignment=True,
    )
    _mark_initialized(env, state, marker_name=".beam_only_initialized.txt")

    zarr_path = Path(env["zarr_path"])
    if zarr_path.exists():
        if zarr_path.is_dir():
            shutil.rmtree(zarr_path)
        else:
            zarr_path.unlink()

    beam_input_dir = Path(workspace.get_beam_mutable_data_dir())
    beam_scenario_dir = (
        beam_input_dir / settings.run.region / settings.beam.scenario_folder
    )
    beam_config_path = beam_input_dir / settings.run.region / settings.beam.config
    beam_output_dir = Path(workspace.get_beam_output_dir())
    beam_plans = beam_input_dir / "plans.parquet"
    beam_households = beam_input_dir / "households.parquet"
    beam_persons = beam_input_dir / "persons.parquet"
    beam_input_tables = {
        beam_plans: pd.DataFrame({"trip_id": [1], "person_id": [1]}),
        beam_households: pd.DataFrame({"household_id": [1], "cars": [0]}),
        beam_persons: pd.DataFrame({"person_id": [1], "household_id": [1]}),
    }
    for path, table in beam_input_tables.items():
        _write_parquet(path, table)
        _write_parquet(beam_scenario_dir / path.name, table)
    r5_dir = beam_input_dir / settings.run.region / "r5"
    _write_file(r5_dir / "network.osm.pbf", "osm source")
    _write_file(
        beam_config_path,
        "\n".join(
            (
                f'beam.inputDirectory = "{beam_input_dir / settings.run.region}"',
                'beam.routing.r5.directory = ${beam.inputDirectory}"/r5"',
            )
        )
        + "\n",
    )
    _publish_input_roles(
        scenario,
        state,
        {
            BEAM_PLANS_IN: beam_plans,
            BEAM_HOUSEHOLDS_IN: beam_households,
            BEAM_PERSONS_IN: beam_persons,
        },
    )

    def _fake_beam_run(self, input_store, _workspace):
        assert "zarr_skims" not in input_store.to_mapping()
        events = beam_output_dir / "events.parquet"
        linkstats = beam_output_dir / "linkstats.csv.gz"
        plans_out = beam_output_dir / "plans.parquet"
        _write_parquet(
            events,
            pd.DataFrame(
                {
                    "type": ["PathTraversal"],
                    "links": ["1"],
                    "linkTravelTime": ["60.0"],
                }
            ),
        )
        for path in (linkstats, plans_out):
            _write_file(path, "stub")
        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=str(events),
                    short_name=f"events_parquet_{state.forecast_year}_{state.iteration}",
                ),
                FileRecord(file_path=str(linkstats), short_name=LINKSTATS),
                FileRecord(file_path=str(plans_out), short_name=BEAM_PLANS_OUT),
            ]
        )

    monkeypatch.setattr(BeamRunner, "_run", _fake_beam_run)

    run_supply_demand_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        usim_inputs={},
        build_manifest_path=_manifest_builder(tmp_path, "beam_only_manifests"),
    )

    steps = _steps_by_model(env)
    assert "beam_preprocess" in steps
    assert "beam_run" in steps
    assert "beam_postprocess" in steps
    assert "activitysim_postprocess" not in steps
    beam_postprocess_outputs = set(
        (steps["beam_postprocess"].get("outputs") or {}).values()
    )
    assert "zarr_skims" not in beam_postprocess_outputs
    assert (
        f"events_parquet_{state.forecast_year}_{state.iteration}_type_PathTraversal"
        in beam_postprocess_outputs
    )


def test_stubbed_activitysim_beam_supply_demand_allows_missing_optional_omx_archive(
    golden_stub_env,
    monkeypatch,
    tmp_path: Path,
) -> None:
    env = golden_stub_env
    settings = env["settings"]
    workspace = env["workspace"]
    scenario = env["scenario"]
    coupler = scenario.coupler

    state = _reconfigure_models(
        settings,
        tmp_path=tmp_path,
        state_file_name="asim_beam_state.yaml",
        land_use=False,
        vehicle_ownership=False,
        activity_demand=True,
        traffic_assignment=True,
    )
    _mark_initialized(env, state, marker_name=".asim_beam_initialized.txt")

    asim_input_dir = Path(workspace.get_asim_mutable_data_dir())
    asim_output_dir = Path(workspace.get_asim_output_dir())
    beam_input_dir = Path(workspace.get_beam_mutable_data_dir())
    beam_output_dir = Path(workspace.get_beam_output_dir())
    sharrow_cache_dir = Path(env["sharrow_cache_dir"])
    zarr_path = Path(env["zarr_path"])
    raw_beam_zarr_path = beam_output_dir / "raw-od-skims.zarr"

    temp_names = (
        "accessibility",
        "disaggregate_accessibility",
        "joint_tour_participants",
        "land_use",
        "non_mandatory_tour_destination_accessibility",
        "households",
        "persons",
        "tours",
        "trips",
        "beam_plans",
    )
    raw_outputs = {}
    processed_outputs = {}
    iteration_dir = (
        asim_output_dir / f"year-{state.current_year}-iteration-{state.iteration}"
    )
    for name in temp_names:
        temp_path = asim_output_dir / "final_pipeline" / name / "final.parquet"
        _write_parquet(temp_path, pd.DataFrame({"id": [1]}))
        raw_outputs[f"{name}_asim_out_temp"] = temp_path
        processed_path = iteration_dir / f"{name}.parquet"
        _write_parquet(processed_path, pd.DataFrame({"id": [1]}))
        processed_outputs[normalize_asim_output_key(name)] = processed_path

    processed_outputs["asim_input_households_csv_archived"] = (
        asim_input_dir / "households.csv"
    )
    processed_outputs["asim_input_persons_csv_archived"] = (
        asim_input_dir / "persons.csv"
    )
    processed_outputs["asim_input_land_use_csv_archived"] = (
        asim_input_dir / "land_use.csv"
    )
    processed_outputs["asim_input_skims_zarr_archived"] = zarr_path

    beam_plans = beam_input_dir / "plans.parquet"
    beam_households = beam_input_dir / "households.parquet"
    beam_persons = beam_input_dir / "persons.parquet"
    _write_parquet(beam_plans, pd.DataFrame({"trip_id": [1], "person_id": [1]}))
    _write_parquet(beam_households, pd.DataFrame({"household_id": [1], "cars": [0]}))
    _write_parquet(beam_persons, pd.DataFrame({"person_id": [1], "household_id": [1]}))
    fixture_zarr = Path(__file__).parent / "fixtures" / "skims" / "mini_skims.zarr"
    shutil.copytree(fixture_zarr, zarr_path, dirs_exist_ok=True)
    shutil.copytree(fixture_zarr, raw_beam_zarr_path)
    import zarr

    for skim_path in (zarr_path, raw_beam_zarr_path):
        skim_store = zarr.open_group(skim_path, mode="a")
        skim_store["otaz"][:] = range(5)
        skim_store["dtaz"][:] = range(5)
        skim_store.attrs["original_zone_ids"] = [
            "1",
            "2",
            "3",
            "4",
            "5",
        ]
        zarr.consolidate_metadata(skim_path)
    zones_path = tmp_path / "canonical_zones.geojson"
    zones = gpd.GeoDataFrame(
        {"zone_id": ["1", "2", "3", "4", "5"]},
        geometry=[
            Polygon([(index, 0), (index + 1, 0), (index + 1, 1), (index, 1)])
            for index in range(5)
        ],
        crs="EPSG:4326",
    )
    zones.to_file(zones_path, driver="GeoJSON")
    settings.shared.geography.zones = ZonesConfig(
        zone_type="taz",
        source_file=str(zones_path),
        canonical_id_col="zone_id",
        activitysim_index_col="TAZ",
        source_crs="EPSG:4326",
    )
    _publish_input_roles(
        scenario,
        state,
        {
            BEAM_PLANS_IN: beam_plans,
            BEAM_HOUSEHOLDS_IN: beam_households,
            BEAM_PERSONS_IN: beam_persons,
        },
    )
    r5_dir = beam_input_dir / settings.run.region / "r5"
    _write_file(r5_dir / "network.osm.pbf", "osm source")
    _write_file(
        beam_input_dir / settings.run.region / settings.beam.config,
        "\n".join(
            (
                f'beam.inputDirectory = "{beam_input_dir / settings.run.region}"',
                'beam.routing.r5.directory = ${beam.inputDirectory}"/r5"',
            )
        )
        + "\n",
    )

    def _fake_beam_run(self, _input_store, _workspace):
        linkstats = beam_output_dir / "linkstats.csv.gz"
        plans_out = beam_output_dir / "plans.parquet"
        _write_file(linkstats, "stub")
        _write_file(plans_out, "stub")
        raw_zarr_key = f"raw_od_skims_zarr_{state.forecast_year}_{state.iteration}"
        return RecordStore(
            recordList=[
                FileRecord(file_path=str(linkstats), short_name=LINKSTATS),
                FileRecord(file_path=str(plans_out), short_name=BEAM_PLANS_OUT),
                FileRecord(file_path=str(raw_beam_zarr_path), short_name=raw_zarr_key),
            ]
        )

    monkeypatch.setattr(BeamRunner, "_run", _fake_beam_run)

    run_supply_demand_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        usim_inputs={},
        build_manifest_path=_manifest_builder(tmp_path, "asim_beam_manifests"),
    )

    steps = _steps_by_model(env)
    assert "activitysim" in steps
    assert "beam_run" in steps


def test_stubbed_land_use_atlas_stage_keeps_usim_datastore_out_of_atlas_run_outputs(
    golden_stub_env,
    monkeypatch,
    tmp_path: Path,
) -> None:
    env = golden_stub_env
    settings = env["settings"]
    workspace = env["workspace"]
    scenario = env["scenario"]
    coupler = scenario.coupler

    state = _reconfigure_models(
        settings,
        tmp_path=tmp_path,
        state_file_name="atlas_stage_state.yaml",
        land_use=True,
        vehicle_ownership=True,
        activity_demand=False,
        traffic_assignment=False,
    )
    _mark_initialized(
        env,
        state,
        marker_name=".atlas_stage_initialized.txt",
        publish_initial_usim_datastore=True,
    )

    zones_path = tmp_path / "canonical_zones.geojson"
    zones = gpd.GeoDataFrame(
        {"zone_id": ["10", "11"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        ],
        crs="EPSG:4326",
    )
    zones.to_file(zones_path, driver="GeoJSON")
    settings.shared.geography.zones = ZonesConfig(
        zone_type="taz",
        source_file=str(zones_path),
        canonical_id_col="zone_id",
        activitysim_index_col="TAZ",
        source_crs="EPSG:4326",
    )
    blocks = gpd.GeoDataFrame(
        {"GEOID": ["0001", "0002"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        ],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(
        "pilates.utils.geog.get_block_geoms",
        lambda *_args, **_kwargs: blocks.copy(),
    )

    usim_output_path = Path(
        workspace.get_usim_mutable_data_dir()
    ) / settings.urbansim.output_file_template.format(year=state.forecast_year)

    def _fake_urbansim_run(self, _inputs, _workspace, _model_run_hash=None):
        return UrbanSimRunOutputs(usim_datastore_h5=usim_output_path)

    def _fake_atlas_run(self, _inputs, _workspace):
        atlas_output_dir = Path(_workspace.get_atlas_output_dir())
        output_year = self.state.forecast_year
        households_path = atlas_output_dir / f"householdv_{output_year}.csv"
        vehicles_path = atlas_output_dir / f"vehicles_{output_year}.csv"
        pd.DataFrame({"household_id": [1, 2], "nvehicles": [1, 2]}).to_csv(
            households_path, index=False
        )
        pd.DataFrame(
            {
                "household_id": [1, 2],
                "vehicle_id": [1, 1],
                "bodytype": ["sedan", "suv"],
                "pred_power": ["gasoline", "electricity"],
                "modelyear": [2018, 2020],
            }
        ).to_csv(vehicles_path, index=False)
        return AtlasRunOutputs(
            atlas_output_dir=atlas_output_dir,
            raw_outputs={
                f"householdv_{output_year}": households_path,
                f"vehicles_{output_year}": vehicles_path,
            },
        )

    monkeypatch.setattr(UrbansimRunner, "_run", _fake_urbansim_run)
    monkeypatch.setattr(AtlasRunner, "_run", _fake_atlas_run)

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

    run_vehicle_ownership_stage(
        scenario=scenario,
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=coupler,
        year=state.forecast_year,
        build_atlas_static_inputs_fallback=lambda _workspace: {},
    )

    assert USIM_DATASTORE_H5 in usim_inputs
    assert coupler.get(USIM_DATASTORE_H5) is not None

    steps = _steps_by_model(env)
    assert "atlas_run" in steps
    atlas_run_outputs = set((steps["atlas_run"].get("outputs") or {}).values())
    assert "atlas_output_dir" in atlas_run_outputs
    assert USIM_DATASTORE_H5 not in atlas_run_outputs
