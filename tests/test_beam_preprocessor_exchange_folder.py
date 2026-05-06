from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pandas as pd

from pilates.beam.beam_input_staging import (
    _normalize_beam_vehicle_columns,
    summarize_population_consistency,
    summarize_vehicle_category_consistency,
    validate_population_consistency,
)
from pilates.beam.preprocessor import BeamPreprocessor
from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.artifact_keys import (
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
)


def _make_preprocessor(
    region: str = "sfbay",
    scenario_folder: str = "urbansim",
    config: str = "sfbay-pilates-base-omx.conf",
    *,
    activity_demand_enabled: bool = False,
):
    settings = SimpleNamespace(
        run=SimpleNamespace(region=region),
        beam=SimpleNamespace(
            scenario_folder=scenario_folder,
            config=config,
            discard_plans_every_year=False,
        ),
        activitysim=SimpleNamespace(file_format="parquet"),
        shared=SimpleNamespace(geography=SimpleNamespace(zones=None)),
        vehicle_ownership_model_enabled=False,
        activity_demand_enabled=activity_demand_enabled,
    )
    state = SimpleNamespace(
        full_settings=settings,
        current_year=2018,
        forecast_year=2018,
        current_inner_iter=0,
        run_info_path=None,
    )
    state.set_sub_stage_progress = lambda _value: None
    return BeamPreprocessor("beam", state)


def _make_workspace(tmp_path: Path):
    workspace = MagicMock()
    workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam" / "input")
    workspace.full_path = str(tmp_path)
    return workspace


def test_resolve_beam_exchange_scenario_folder_reads_config_folder(tmp_path):
    preprocessor = _make_preprocessor()
    workspace = _make_workspace(tmp_path)

    base_input_dir = tmp_path / "beam" / "input" / "sfbay"
    base_input_dir.mkdir(parents=True, exist_ok=True)
    config_path = base_input_dir / "sfbay-pilates-base-omx.conf"
    config_path.write_text(
        "beam.exchange.scenario {\n"
        '  source = "urbansim_v2"\n'
        '  folder = ${beam.inputDirectory}"/urbansim/2018"\n'
        "}\n",
        encoding="utf-8",
    )

    resolved = preprocessor._resolve_beam_exchange_scenario_folder(workspace)

    assert resolved == str(base_input_dir / "urbansim" / "2018")


def test_resolve_beam_exchange_scenario_folder_supports_bare_inputdirectory(tmp_path):
    preprocessor = _make_preprocessor()
    workspace = _make_workspace(tmp_path)

    base_input_dir = tmp_path / "beam" / "input" / "sfbay"
    base_input_dir.mkdir(parents=True, exist_ok=True)
    config_path = base_input_dir / "sfbay-pilates-base-omx.conf"
    config_path.write_text(
        'beam.exchange.scenario {\n  folder = ${inputDirectory}"/urbansim/2018"\n}\n',
        encoding="utf-8",
    )

    resolved = preprocessor._resolve_beam_exchange_scenario_folder(workspace)

    assert resolved == str(base_input_dir / "urbansim" / "2018")


def test_resolve_beam_exchange_scenario_folder_reads_included_config_folder(tmp_path):
    preprocessor = _make_preprocessor()
    workspace = _make_workspace(tmp_path)

    base_input_dir = tmp_path / "beam" / "input" / "sfbay"
    base_input_dir.mkdir(parents=True, exist_ok=True)
    config_path = base_input_dir / "sfbay-pilates-base-omx.conf"
    include_path = base_input_dir / "scenario-override.conf"
    config_path.write_text(
        'include "scenario-override.conf"\n',
        encoding="utf-8",
    )
    include_path.write_text(
        "beam.exchange.scenario {\n"
        '  folder = ${beam.inputDirectory}"/urbansim/2018"\n'
        "}\n",
        encoding="utf-8",
    )

    resolved = preprocessor._resolve_beam_exchange_scenario_folder(workspace)

    assert resolved == str(base_input_dir / "urbansim" / "2018")


def test_resolve_beam_exchange_scenario_folder_falls_back_on_unparseable_folder(
    tmp_path,
):
    preprocessor = _make_preprocessor(scenario_folder="urbansim")
    workspace = _make_workspace(tmp_path)

    base_input_dir = tmp_path / "beam" / "input" / "sfbay"
    base_input_dir.mkdir(parents=True, exist_ok=True)
    config_path = base_input_dir / "sfbay-pilates-base-omx.conf"
    config_path.write_text(
        "beam.exchange.scenario {\n"
        "  # missing ${beam.inputDirectory} placeholder on purpose\n"
        '  folder = "/app/input/sfbay/urbansim/2018"\n'
        "}\n",
        encoding="utf-8",
    )

    resolved = preprocessor._resolve_beam_exchange_scenario_folder(workspace)

    assert resolved == str(base_input_dir / "urbansim")


def test_resolve_beam_exchange_scenario_folder_falls_back_when_config_missing(tmp_path):
    preprocessor = _make_preprocessor(scenario_folder="urbansim")
    workspace = _make_workspace(tmp_path)

    base_input_dir = tmp_path / "beam" / "input" / "sfbay"
    base_input_dir.mkdir(parents=True, exist_ok=True)

    resolved = preprocessor._resolve_beam_exchange_scenario_folder(workspace)

    assert resolved == str(base_input_dir / "urbansim")


def test_beam_preprocess_registers_existing_default_scenario_inputs(
    monkeypatch, tmp_path
):
    preprocessor = _make_preprocessor(scenario_folder="urbansim")
    workspace = _make_workspace(tmp_path)

    scenario_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("plans", "households", "persons"):
        (scenario_dir / f"{stem}.parquet").write_text(stem, encoding="utf-8")

    monkeypatch.setattr(
        preprocessor, "_update_beam_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(preprocessor, "_handle_linkstats", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        preprocessor,
        "_copy_plans_from_asim",
        lambda *_args, **_kwargs: RecordStore(),
    )

    outputs = preprocessor.preprocess(workspace)

    assert outputs.prepared_inputs == {
        BEAM_PLANS_IN: scenario_dir / "plans.parquet",
        BEAM_HOUSEHOLDS_IN: scenario_dir / "households.parquet",
        BEAM_PERSONS_IN: scenario_dir / "persons.parquet",
    }


def test_beam_preprocess_fails_early_when_default_scenario_inputs_missing(
    monkeypatch, tmp_path
):
    preprocessor = _make_preprocessor(scenario_folder="urbansim")
    workspace = _make_workspace(tmp_path)

    scenario_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "plans.parquet").write_text("plans", encoding="utf-8")
    (scenario_dir / "households.parquet").write_text("households", encoding="utf-8")

    monkeypatch.setattr(
        preprocessor, "_update_beam_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(preprocessor, "_handle_linkstats", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        preprocessor,
        "_copy_plans_from_asim",
        lambda *_args, **_kwargs: RecordStore(),
    )

    with pytest.raises(FileNotFoundError, match=r"persons\.\[parquet\|csv\|csv\.gz\]"):
        preprocessor.preprocess(workspace)


def test_beam_preprocess_does_not_fallback_to_defaults_when_activitysim_enabled(
    monkeypatch, tmp_path
):
    preprocessor = _make_preprocessor(
        scenario_folder="urbansim",
        activity_demand_enabled=True,
    )
    workspace = _make_workspace(tmp_path)

    scenario_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("plans", "households", "persons"):
        (scenario_dir / f"{stem}.parquet").write_text(stem, encoding="utf-8")

    monkeypatch.setattr(
        preprocessor, "_update_beam_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(preprocessor, "_handle_linkstats", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        preprocessor,
        "_copy_plans_from_asim",
        lambda *_args, **_kwargs: RecordStore(),
    )

    with pytest.raises(
        RuntimeError, match="expected ActivitySim to stage the canonical"
    ):
        preprocessor.preprocess(workspace)


def test_beam_preprocess_fails_when_vehicle_households_are_missing_from_staged_households(
    monkeypatch, tmp_path
):
    preprocessor = _make_preprocessor(
        scenario_folder="urbansim",
        activity_demand_enabled=True,
    )
    preprocessor.settings.vehicle_ownership_model_enabled = True
    workspace = _make_workspace(tmp_path)

    scenario_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        preprocessor, "_update_beam_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(preprocessor, "_handle_linkstats", lambda *args, **kwargs: None)

    def _fake_copy_plans(_input_records, _workspace):
        pd.DataFrame({"household_id": [1, 2], "cars": [1, 1]}).to_parquet(
            scenario_dir / "households.parquet",
            index=False,
        )
        pd.DataFrame({"person_id": [11, 21], "household_id": [1, 2]}).to_parquet(
            scenario_dir / "persons.parquet",
            index=False,
        )
        pd.DataFrame({"trip_id": [1], "person_id": [11]}).to_parquet(
            scenario_dir / "plans.parquet",
            index=False,
        )
        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=str(scenario_dir / "households.parquet"),
                    short_name=BEAM_HOUSEHOLDS_IN,
                ),
                FileRecord(
                    file_path=str(scenario_dir / "persons.parquet"),
                    short_name=BEAM_PERSONS_IN,
                ),
                FileRecord(
                    file_path=str(scenario_dir / "plans.parquet"),
                    short_name=BEAM_PLANS_IN,
                ),
            ]
        )

    def _fake_copy_vehicles(_workspace, source_path=None):
        _ = source_path
        pd.DataFrame(
            {
                "vehicleId": [100],
                "householdId": [999],
                "vehicleTypeId": ["sedan_gas_2015"],
            }
        ).to_parquet(scenario_dir / "vehicles.parquet", index=False)
        return FileRecord(
            file_path=str(scenario_dir / "vehicles.parquet"),
            short_name="vehicles_beam_in",
        )

    monkeypatch.setattr(preprocessor, "_copy_plans_from_asim", _fake_copy_plans)
    monkeypatch.setattr(preprocessor, "_copy_vehicles_from_atlas", _fake_copy_vehicles)

    with pytest.raises(ValueError, match="reference households that are absent"):
        preprocessor.preprocess(workspace)


def test_beam_preprocess_prefers_explicit_atlas_vehicle_input_over_workspace_fallback(
    monkeypatch, tmp_path
):
    preprocessor = _make_preprocessor(
        scenario_folder="urbansim",
        activity_demand_enabled=True,
    )
    preprocessor.settings.vehicle_ownership_model_enabled = True
    workspace = _make_workspace(tmp_path)

    scenario_dir = tmp_path / "beam" / "input" / "sfbay" / "urbansim"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    explicit_vehicles = tmp_path / "restored" / "vehicles.csv.gz"
    explicit_vehicles.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"vehicle_id": [100], "household_id": [1], "vehicleTypeId": ["sedan_gas_2015"]}
    ).to_csv(explicit_vehicles, index=False, compression="gzip")

    monkeypatch.setattr(
        preprocessor, "_update_beam_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(preprocessor, "_handle_linkstats", lambda *args, **kwargs: None)

    def _fake_copy_plans(_input_records, _workspace):
        pd.DataFrame({"household_id": [1], "cars": [1]}).to_parquet(
            scenario_dir / "households.parquet",
            index=False,
        )
        pd.DataFrame({"person_id": [11], "household_id": [1]}).to_parquet(
            scenario_dir / "persons.parquet",
            index=False,
        )
        pd.DataFrame({"trip_id": [1], "person_id": [11]}).to_parquet(
            scenario_dir / "plans.parquet",
            index=False,
        )
        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=str(scenario_dir / "households.parquet"),
                    short_name=BEAM_HOUSEHOLDS_IN,
                ),
                FileRecord(
                    file_path=str(scenario_dir / "persons.parquet"),
                    short_name=BEAM_PERSONS_IN,
                ),
                FileRecord(
                    file_path=str(scenario_dir / "plans.parquet"),
                    short_name=BEAM_PLANS_IN,
                ),
            ]
        )

    captured = {}

    def _fake_copy_vehicles(_workspace, source_path=None):
        captured["source_path"] = source_path
        pd.DataFrame(
            {
                "vehicleId": [100],
                "householdId": [1],
                "vehicleTypeId": ["sedan_gas_2015"],
            }
        ).to_parquet(scenario_dir / "vehicles.parquet", index=False)
        return FileRecord(
            file_path=str(scenario_dir / "vehicles.parquet"),
            short_name="vehicles_beam_in",
        )

    monkeypatch.setattr(preprocessor, "_copy_plans_from_asim", _fake_copy_plans)
    monkeypatch.setattr(preprocessor, "_copy_vehicles_from_atlas", _fake_copy_vehicles)

    outputs = preprocessor.preprocess(
        workspace,
        beam_preprocess_inputs={ATLAS_VEHICLES2_OUTPUT: str(explicit_vehicles)},
    )

    assert captured["source_path"] == str(explicit_vehicles)
    assert (
        outputs.prepared_inputs["vehicles_beam_in"] == scenario_dir / "vehicles.parquet"
    )


def test_normalize_beam_vehicle_columns_synthesizes_global_ids_for_household_local_ids():
    normalized = _normalize_beam_vehicle_columns(
        pd.DataFrame(
            {
                "vehicle_id": [1, 2, 1],
                "household_id": [10, 10, 20],
                "vehicleTypeId": ["sedan_gas_2015", "suv_gas_2015", "sedan_gas_2015"],
            }
        )
    )

    assert list(normalized.columns[:3]) == ["vehicleId", "householdId", "vehicleTypeId"]
    assert normalized["vehicleId"].tolist() == [
        "hh-10-veh-1",
        "hh-10-veh-2",
        "hh-20-veh-1",
    ]
    assert normalized["sourceVehicleId"].tolist() == [1, 2, 1]
    assert not normalized["vehicleId"].duplicated().any()
    assert str(normalized["sourceVehicleId"].dtype) == "int64"
    assert str(normalized["householdId"].dtype) == "int64"


def test_summarize_population_consistency_reports_shortfalls_and_duplicates():
    households = pd.DataFrame({"household_id": [1, 2, 3], "cars": [2, 1, 0]}).set_index(
        "household_id"
    )
    vehicles = pd.DataFrame(
        {
            "vehicleId": [1001, 1001, 2001],
            "householdId": [1, 2, 99],
            "vehicleTypeId": ["sedan_gas_2015", "sedan_gas_2015", "sedan_gas_2015"],
        }
    )

    report = summarize_population_consistency(households=households, vehicles=vehicles)

    assert report["total_households"] == 3
    assert report["total_vehicle_rows"] == 3
    assert report["duplicate_vehicle_ids"] == 2
    assert report["households_with_vehicle_count_mismatch"] == 1
    assert report["households_with_car_shortfall"] == 1
    assert report["missing_vehicle_households"] == 1
    assert report["sample_household_shortfalls"] == [
        {"household_id": 1, "cars": 2, "vehicle_row_count": 1}
    ]


def test_summarize_vehicle_category_consistency_warns_on_non_car_rows(tmp_path):
    workspace = _make_workspace(tmp_path)
    settings = SimpleNamespace(
        run=SimpleNamespace(region="sfbay"),
        beam=SimpleNamespace(config="sfbay-pilates-base-omx.conf"),
    )

    base_input_dir = tmp_path / "beam" / "input" / "sfbay"
    scenario_dir = base_input_dir / "urbansim"
    atlas_vehicle_types = (
        base_input_dir / "atlas-vehicles" / "vehicleTypes_baseline.csv"
    )
    atlas_vehicle_types.parent.mkdir(parents=True, exist_ok=True)
    atlas_vehicle_types.write_text(
        "vehicleTypeId,vehicleCategory\nsedan_gas_2015,Car\nebike_1,Bike\n",
        encoding="utf-8",
    )
    (base_input_dir / "sfbay-pilates-base-omx.conf").write_text(
        'beam.agentsim.agents.vehicles.vehicleTypesFilePath = ${beam.inputDirectory}"/atlas-vehicles/vehicleTypes_baseline.csv"\n',
        encoding="utf-8",
    )

    households = pd.DataFrame({"cars": [2]}, index=pd.Index([1], name="household_id"))
    vehicles = pd.DataFrame(
        {
            "vehicleId": ["hh-1-veh-1", "hh-1-veh-2"],
            "householdId": [1, 1],
            "vehicleTypeId": ["sedan_gas_2015", "ebike_1"],
        }
    )

    report = summarize_vehicle_category_consistency(
        households=households,
        vehicles=vehicles,
        settings=settings,
        workspace=workspace,
    )

    assert report["status"] == "ok"
    assert report["households_with_car_category_shortfall"] == 1
    assert report["unmatched_vehicle_types"] == 0
    assert report["non_car_vehicle_rows"] == 1
    assert report["sample_household_car_shortfalls"] == [
        {"household_id": 1, "cars": 2, "car_vehicle_row_count": 1}
    ]
    assert report["vehicle_types_path"] == str(atlas_vehicle_types)


def test_validate_population_consistency_logs_advisory_car_category_shortfall(
    tmp_path, caplog
):
    caplog.set_level("INFO")
    workspace = _make_workspace(tmp_path)
    settings = SimpleNamespace(
        run=SimpleNamespace(region="sfbay"),
        beam=SimpleNamespace(config="sfbay-pilates-base-omx.conf"),
        activitysim=SimpleNamespace(file_format="parquet"),
    )

    base_input_dir = tmp_path / "beam" / "input" / "sfbay"
    scenario_dir = base_input_dir / "urbansim"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"household_id": [1], "cars": [2]}).to_parquet(
        scenario_dir / "households.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "vehicleId": ["hh-1-veh-1", "hh-1-veh-2"],
            "householdId": [1, 1],
            "vehicleTypeId": ["sedan_gas_2015", "ebike_1"],
        }
    ).to_parquet(scenario_dir / "vehicles.parquet", index=False)
    atlas_vehicle_types = (
        base_input_dir / "atlas-vehicles" / "vehicleTypes_baseline.csv"
    )
    atlas_vehicle_types.parent.mkdir(parents=True, exist_ok=True)
    atlas_vehicle_types.write_text(
        "vehicleTypeId,vehicleCategory\nsedan_gas_2015,Car\nebike_1,Bike\n",
        encoding="utf-8",
    )
    (base_input_dir / "sfbay-pilates-base-omx.conf").write_text(
        'beam.agentsim.agents.vehicles.vehicleTypesFilePath = ${beam.inputDirectory}"/atlas-vehicles/vehicleTypes_baseline.csv"\n',
        encoding="utf-8",
    )

    validate_population_consistency(
        workspace=workspace,
        settings=settings,
        resolve_beam_exchange_scenario_folder_fn=lambda _workspace: str(scenario_dir),
    )

    assert "Validated BEAM staged household vehicles" in caplog.text
    assert (
        "BEAM staged households have fewer staged Car-category vehicles than required by households.cars."
        in caplog.text
    )
    assert "households_with_car_category_shortfall=1" in caplog.text
