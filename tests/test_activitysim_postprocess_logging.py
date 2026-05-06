from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd

from pilates.activitysim.outputs import (
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.activitysim.postprocessor import (
    ActivitysimPostprocessor,
    create_usim_input_data,
)
from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.binding import BindingPlan
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_POPULATION_BLOCKS_TABLE,
    USIM_POPULATION_HOUSEHOLDS_TABLE,
    USIM_POPULATION_JOBS_TABLE,
    USIM_POPULATION_PERSONS_TABLE,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows import steps
from pilates.workflows.steps import StepOutputsHolder
from pilates.workflows.steps import activitysim as steps_activitysim


def _dummy_coupler():
    return SimpleNamespace(
        get=lambda *args, **kwargs: None,
        set=lambda *args, **kwargs: None,
        update=lambda *args, **kwargs: None,
        set_from_artifact=lambda *args, **kwargs: None,
    )


@pytest.mark.parametrize(
    ("current_year", "forecast_year"),
    [
        (2021, 2023),
        (2023, 2023),
    ],
)
def test_activitysim_postprocess_archived_inputs_use_forecast_year_directory(
    current_year,
    forecast_year,
    tmp_path,
) -> None:
    settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        urbansim=SimpleNamespace(
            input_file_template="model_data_{region_id}.h5",
            output_file_template="model_data_{year}.h5",
            region_mappings={"region_to_region_id": {"seattle": "06197001"}},
        ),
    )
    state = SimpleNamespace(
        year=current_year,
        current_year=current_year,
        forecast_year=forecast_year,
        iteration=0,
        current_inner_iter=0,
        full_settings=settings,
        set_sub_stage_progress=lambda _stage: None,
        is_enabled=lambda _stage: False,
    )
    asim_data_dir = tmp_path / "asim" / "data"
    asim_output_dir = tmp_path / "asim" / "output"
    usim_data_dir = tmp_path / "urbansim" / "data"
    asim_data_dir.mkdir(parents=True)
    asim_output_dir.mkdir(parents=True)
    usim_data_dir.mkdir(parents=True)
    for filename in ("households.csv", "persons.csv", "land_use.csv", "skims.omx"):
        (asim_data_dir / filename).write_text(filename)

    raw_households = tmp_path / "raw_households.parquet"
    raw_households.write_text("raw")
    workspace = SimpleNamespace(
        get_asim_mutable_data_dir=lambda: str(asim_data_dir),
        get_asim_output_dir=lambda: str(asim_output_dir),
        get_usim_mutable_data_dir=lambda: str(usim_data_dir),
    )

    expected_outputs = ActivitysimPostprocessor.expected_outputs(
        settings,
        state,
        workspace,
    )
    assert (
        Path(expected_outputs["asim_input_households_csv_archived"]).parent.name
        == f"inputs-year-{forecast_year}-iteration-0"
    )
    assert (
        Path(expected_outputs["households_asim_out"]).parent.name
        == f"year-{current_year}-iteration-0"
    )

    outputs = ActivitysimPostprocessor("activitysim", state).postprocess(
        ActivitySimRunOutputs(
            output_dir=asim_output_dir,
            raw_outputs={"households": raw_households},
        ),
        workspace,
    )

    archived_households = outputs.processed_outputs[
        "asim_input_households_csv_archived"
    ]
    assert archived_households == (
        asim_output_dir / f"inputs-year-{forecast_year}-iteration-0" / "households.csv"
    )
    assert archived_households.exists()
    assert outputs.processed_outputs["households_asim_out"] == (
        asim_output_dir / f"year-{current_year}-iteration-0" / "households.parquet"
    )


def test_activitysim_postprocess_recovery_uses_forecast_year_archived_inputs(
    tmp_path,
) -> None:
    current_year = 2019
    forecast_year = 2021
    asim_output_dir = tmp_path / "activitysim" / "output"
    iter_dir = asim_output_dir / f"year-{current_year}-iteration-0"
    forecast_inputs_dir = asim_output_dir / f"inputs-year-{forecast_year}-iteration-0"
    legacy_inputs_dir = asim_output_dir / f"inputs-year-{current_year}-iteration-0"
    for name in ("persons.parquet", "households.parquet", "beam_plans.parquet"):
        path = iter_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    for name in ("households.csv", "persons.csv", "land_use.csv"):
        path = forecast_inputs_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"forecast-{name}", encoding="utf-8")
    legacy_inputs_dir.mkdir(parents=True, exist_ok=True)

    recovered = steps_activitysim._recover_activitysim_postprocess_outputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(
            year=current_year,
            forecast_year=forecast_year,
            iteration=0,
            is_enabled=lambda _stage: False,
        ),
        workspace=SimpleNamespace(
            get_asim_output_dir=lambda: str(asim_output_dir),
            full_path=str(tmp_path),
        ),
        coupler=_dummy_coupler(),
        outputs_holder=StepOutputsHolder(),
        step_inputs={},
        cached_outputs=None,
        run_id=None,
    )

    assert recovered is not None
    assert recovered.processed_outputs["asim_input_households_csv_archived"] == (
        forecast_inputs_dir / "households.csv"
    )
    assert "asim_input_households_csv_archived" not in {
        key
        for key, path in recovered.processed_outputs.items()
        if str(path).startswith(str(legacy_inputs_dir))
    }


def test_create_usim_input_data_writes_forecast_year_target(tmp_path) -> None:
    current_year = 2019
    forecast_year = 2021
    usim_dir = tmp_path / "urbansim" / "data"
    usim_dir.mkdir(parents=True)
    current_h5 = usim_dir / f"model_data_{current_year}.h5"
    forecast_h5 = usim_dir / f"model_data_{forecast_year}.h5"

    with pd.HDFStore(current_h5, mode="w") as store:
        store.put("/2019/blocks", pd.DataFrame({"block_id": [1]}))
    with pd.HDFStore(forecast_h5, mode="w") as store:
        store.put("/2021/households", pd.DataFrame({"household_id": [1]}))
        store.put("/2021/persons", pd.DataFrame({"person_id": [10]}))
        store.put("/2021/jobs", pd.DataFrame({"job_id": [100]}))

    settings = SimpleNamespace(
        urbansim=SimpleNamespace(output_file_template="model_data_{year}.h5")
    )
    state = SimpleNamespace(forecast_year=forecast_year)

    next_path, record = create_usim_input_data(
        settings,
        state,
        asim_output_dict={
            "households": pd.DataFrame({"household_id": [2]}),
            "persons": pd.DataFrame({"person_id": [20]}),
        },
        tables_updated_by_asim=["households", "persons"],
        asim_source_paths=[],
        current_input_store_path=str(current_h5),
        population_source_store_path=str(forecast_h5),
        target_store_path=str(forecast_h5),
    )

    archive_h5 = usim_dir / "input_data_for_2021_outputs.h5"
    assert Path(next_path) == forecast_h5
    assert record.short_name == USIM_DATASTORE_H5
    assert forecast_h5.exists()
    assert archive_h5.exists()
    assert current_h5.exists()
    with pd.HDFStore(forecast_h5, mode="r") as store:
        assert "/households" in store
        assert "/persons" in store
        assert "/jobs" in store


def test_activitysim_postprocess_logs_content_hash(monkeypatch, tmp_path) -> None:
    step_fn = steps.make_activitysim_postprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=SimpleNamespace(),
    )
    output_logger = step_fn.pilates_output_replayer
    calls = []

    def _log_output_only(*, key, path, description, **meta):
        calls.append((key, meta))

    monkeypatch.setattr(steps_activitysim, "log_output_only", _log_output_only)

    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=tmp_path,
        processed_outputs={"asim_input_skims_zarr_archived": tmp_path / "skims.zarr"},
        processed_output_hashes={"asim_input_skims_zarr_archived": "abc123"},
    )

    output_logger(
        outputs,
        settings=SimpleNamespace(),
        state=SimpleNamespace(forecast_year=0, iteration=0),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(),
    )

    assert len(calls) == 1
    assert calls[0][0] == "asim_input_skims_zarr_archived"
    assert calls[0][1]["content_hash"] == "abc123"
    assert calls[0][1]["facet"]["artifact_family"] == "asim_input_archived"
    assert calls[0][1]["facet"]["source_role"] == "zarr_skims"
    assert calls[0][1]["facet"]["snapshot_role"] == "asim_input_skims_zarr"
    assert calls[0][1]["facet"]["snapshot_reason"] == "exact_rewind"
    assert calls[0][1]["facet"]["storage_event"] == "snapshot_copy"


def test_activitysim_preprocess_step_forwards_surface_to_runtime_resolution(
    monkeypatch,
) -> None:
    captured = {}

    def _fake_runtime_inputs(**kwargs):
        captured["surface"] = kwargs.get("surface")
        return {"population_source_h5_path": None}

    monkeypatch.setattr(
        steps_activitysim,
        "_resolve_activitysim_preprocess_runtime_inputs",
        _fake_runtime_inputs,
    )
    monkeypatch.setattr(
        steps_activitysim,
        "build_standard_step",
        lambda **kwargs: SimpleNamespace(
            pilates_input_logger=kwargs["spec"].input_logger
        ),
    )

    step_fn = steps.make_activitysim_preprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=SimpleNamespace(),
        surface="sentinel-surface",
    )

    step_fn.pilates_input_logger(
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2023),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(),
    )

    assert captured["surface"] == "sentinel-surface"


def test_activitysim_postprocess_logs_source_input_files(monkeypatch, tmp_path) -> None:
    usim_next = tmp_path / "urbansim" / "data" / "model_data_next.h5"
    usim_next.parent.mkdir(parents=True, exist_ok=True)
    usim_next.write_text("x")
    fake_postprocessor = SimpleNamespace(
        postprocess=lambda _raw_outputs, _workspace: ActivitySimPostprocessOutputs(
            usim_datastore_h5=usim_next,
            asim_output_dir=tmp_path,
            processed_outputs={},
        )
    )
    monkeypatch.setattr(
        steps_activitysim.ModelFactory,
        "get_postprocessor",
        lambda self, *args, **kwargs: fake_postprocessor,
    )
    monkeypatch.setattr(steps_activitysim, "log_output_only", lambda **_kwargs: None)
    monkeypatch.setattr(steps_activitysim, "log_and_set_output", lambda **_kwargs: None)
    calls = []

    def _log_input_only(*, key, path, description, **meta):
        calls.append((key, path))

    monkeypatch.setattr(steps_activitysim, "log_input_only", _log_input_only)

    asim_input_dir = tmp_path / "asim" / "data"
    asim_output_dir = tmp_path / "asim" / "output"
    usim_data_dir = tmp_path / "urbansim" / "data"
    (asim_input_dir).mkdir(parents=True)
    (asim_output_dir / "cache").mkdir(parents=True)
    usim_data_dir.mkdir(parents=True, exist_ok=True)

    for rel in ("households.csv", "persons.csv", "land_use.csv", "skims.omx"):
        (asim_input_dir / rel).write_text("x")
    (asim_output_dir / "cache" / "skims.zarr").mkdir(parents=True, exist_ok=True)
    (usim_data_dir / "model_data_06197001.h5").write_text("x")
    (usim_data_dir / "model_data_2023.h5").write_text("x")

    workspace = SimpleNamespace(
        get_asim_mutable_data_dir=lambda: str(asim_input_dir),
        get_asim_output_dir=lambda: str(asim_output_dir),
        get_usim_mutable_data_dir=lambda: str(usim_data_dir),
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        urbansim=SimpleNamespace(
            input_file_template="model_data_{region_id}.h5",
            output_file_template="model_data_{year}.h5",
            region_mappings={"region_to_region_id": {"seattle": "06197001"}},
        ),
    )
    state = SimpleNamespace(
        forecast_year=2023,
        iteration=0,
        is_enabled=lambda stage: True,
        Stage=SimpleNamespace(land_use="land_use"),
    )

    outputs_holder = StepOutputsHolder()
    outputs_holder.activitysim_run = ActivitySimRunOutputs(
        output_dir=asim_output_dir,
        raw_outputs={},
    )
    coupler = SimpleNamespace(
        get=lambda key, default=None: {
            USIM_POPULATION_SOURCE_H5: str(usim_data_dir / "model_data_2023.h5"),
            USIM_DATASTORE_CURRENT_H5: str(usim_data_dir / "model_data_06197001.h5"),
            USIM_FORECAST_OUTPUT: str(usim_data_dir / "model_data_2023.h5"),
        }.get(key, default),
        set=lambda *args, **kwargs: None,
        update=lambda *args, **kwargs: None,
        set_from_artifact=lambda *args, **kwargs: None,
    )
    step_fn = steps.make_activitysim_postprocess_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
    )
    step_fn(settings=settings, state=state, workspace=workspace)

    keys = {key for key, _path in calls}
    assert "households_asim_in" in keys
    assert "persons_asim_in" in keys
    assert "land_use_asim_in" in keys
    assert "omx_skims" in keys
    assert "zarr_skims" in keys
    assert "usim_population_source_h5" in keys
    assert "usim_datastore_h5" in keys


def test_activitysim_postprocess_step_forwards_surface_to_runtime_resolution(
    monkeypatch,
) -> None:
    captured = {}

    def _fake_runtime_inputs(**kwargs):
        captured["surface"] = kwargs.get("surface")
        return {
            "population_source_h5_path": None,
            "current_input_h5_path": None,
        }

    monkeypatch.setattr(
        steps_activitysim,
        "_resolve_activitysim_postprocess_runtime_inputs",
        _fake_runtime_inputs,
    )
    monkeypatch.setattr(steps_activitysim, "log_input_only", lambda **_kwargs: None)
    monkeypatch.setattr(
        steps_activitysim,
        "build_standard_step",
        lambda **kwargs: SimpleNamespace(
            pilates_input_logger=kwargs["spec"].input_logger
        ),
    )

    step_fn = steps.make_activitysim_postprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=SimpleNamespace(),
        surface="sentinel-surface",
    )

    step_fn.pilates_input_logger(
        settings=SimpleNamespace(),
        state=SimpleNamespace(
            forecast_year=2023,
            iteration=0,
            is_enabled=lambda _stage: True,
            Stage=SimpleNamespace(land_use="land_use"),
        ),
        workspace=SimpleNamespace(
            get_asim_mutable_data_dir=lambda: "/tmp/missing",
            get_asim_output_dir=lambda: "/tmp/missing",
        ),
        holder=SimpleNamespace(),
    )

    assert captured["surface"] == "sentinel-surface"


def test_activitysim_postprocess_rejects_legacy_only_run_outputs(
    monkeypatch, tmp_path
) -> None:
    class _LegacyOnlyRunOutputs:
        def to_record_store(self) -> RecordStore:
            return RecordStore(
                recordList=[
                    FileRecord(
                        file_path=str(tmp_path / "raw.parquet"),
                        short_name="households_asim_out_temp",
                    )
                ]
            )

    fake_postprocessor = SimpleNamespace(
        postprocess=lambda raw_outputs, _workspace: raw_outputs
    )
    monkeypatch.setattr(
        steps_activitysim.ModelFactory,
        "get_postprocessor",
        lambda self, *args, **kwargs: fake_postprocessor,
    )

    asim_input_dir = tmp_path / "asim" / "data"
    asim_output_dir = tmp_path / "asim" / "output"
    usim_data_dir = tmp_path / "urbansim" / "data"
    asim_input_dir.mkdir(parents=True)
    (asim_output_dir / "cache").mkdir(parents=True)
    usim_data_dir.mkdir(parents=True)

    workspace = SimpleNamespace(
        get_asim_mutable_data_dir=lambda: str(asim_input_dir),
        get_asim_output_dir=lambda: str(asim_output_dir),
        get_usim_mutable_data_dir=lambda: str(usim_data_dir),
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        urbansim=SimpleNamespace(
            input_file_template="model_data_{region_id}.h5",
            output_file_template="model_data_{year}.h5",
            region_mappings={"region_to_region_id": {"seattle": "06197001"}},
        ),
    )
    state = SimpleNamespace(
        forecast_year=2023,
        iteration=0,
        is_enabled=lambda stage: False,
        Stage=SimpleNamespace(land_use="land_use"),
    )

    step_fn = steps.make_activitysim_postprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=SimpleNamespace(activitysim_run=_LegacyOnlyRunOutputs()),
    )

    with pytest.raises(TypeError, match="ActivitySimRunOutputs"):
        step_fn(settings=settings, state=state, workspace=workspace)


def test_activitysim_postprocess_normalizes_legacy_usim_input_key(
    tmp_path,
) -> None:
    record_store = RecordStore(
        recordList=[
            FileRecord(
                file_path=str(tmp_path / "usim_2018.h5"),
                short_name="usim_input_2018",
            )
        ]
    )

    outputs = ActivitySimPostprocessOutputs.from_record_store(
        record_store,
        workspace=SimpleNamespace(
            full_path=str(tmp_path),
            get_asim_output_dir=lambda: str(tmp_path / "asim" / "output"),
        ),
    )

    assert outputs.usim_datastore_h5 == tmp_path / "usim_2018.h5"
    assert outputs.usim_datastore_key == USIM_DATASTORE_H5
    assert list(outputs.to_record_store().to_mapping()) == [USIM_DATASTORE_H5]


def test_activitysim_preprocess_logs_selected_usim_h5_tables(
    monkeypatch, tmp_path
) -> None:
    fake_preprocessor = SimpleNamespace(
        preprocess=lambda _workspace, **_kwargs: ActivitySimPreprocessOutputs(
            mutable_data_dir=asim_data_dir,
            land_use_table=asim_data_dir / "land_use.csv",
            households_table=asim_data_dir / "households.csv",
            persons_table=asim_data_dir / "persons.csv",
            omx_skims=None,
        )
    )
    monkeypatch.setattr(
        steps_activitysim.ModelFactory,
        "get_preprocessor",
        lambda self, *args, **kwargs: fake_preprocessor,
    )
    input_calls = []

    monkeypatch.setattr(
        steps_activitysim,
        "build_binding_plan",
        lambda **_kwargs: BindingPlan(
            source_by_key={USIM_POPULATION_SOURCE_H5: "explicit"},
            inputs={
                USIM_POPULATION_SOURCE_H5: str(tmp_path / "model_data.h5"),
                USIM_POPULATION_HOUSEHOLDS_TABLE: "/2025/households",
                USIM_POPULATION_PERSONS_TABLE: "/2025/persons",
                USIM_POPULATION_JOBS_TABLE: "/2025/jobs",
                USIM_POPULATION_BLOCKS_TABLE: "/2025/blocks",
            },
        ),
    )
    monkeypatch.setattr(
        steps_activitysim,
        "log_and_set_input",
        lambda **kwargs: input_calls.append(kwargs),
    )
    monkeypatch.setattr(steps_activitysim, "_log_step_records", lambda **_kwargs: None)

    h5_path = tmp_path / "model_data.h5"
    with pd.HDFStore(h5_path, mode="w") as store:
        for table_name in ("households", "persons", "jobs", "blocks"):
            store.put(f"/2025/{table_name}", pd.DataFrame({"value": [1]}))
    asim_data_dir = tmp_path / "asim_data"
    asim_data_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("land_use.csv", "households.csv", "persons.csv"):
        (asim_data_dir / filename).write_text("x")

    step_fn = steps.make_activitysim_preprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=StepOutputsHolder(),
    )

    step_fn(
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2023, start_year=2017, forecast_year=2025),
        workspace=SimpleNamespace(get_asim_mutable_data_dir=lambda: str(asim_data_dir)),
    )

    assert len(input_calls) == 1
    assert input_calls[0]["path"] == str(h5_path)
    assert input_calls[0]["child_selection"] == "include_only"
    assert input_calls[0]["child_specs"]["/2025/households"].key == (
        "activitysim_preprocess_usim_households_table_input"
    )


def test_execute_activitysim_preprocess_forwards_resolved_population_table_paths() -> (
    None
):
    captured = {}

    class _Preprocessor:
        def preprocess(self, _workspace, **kwargs):
            captured.update(kwargs)
            return ActivitySimPreprocessOutputs(
                mutable_data_dir=tmp_path,
                land_use_table=tmp_path / "land_use.csv",
                households_table=tmp_path / "households.csv",
                persons_table=tmp_path / "persons.csv",
                omx_skims=None,
            )

    tmp_path = Path("/tmp")
    steps_activitysim._execute_activitysim_preprocess(
        _Preprocessor(),
        workspace=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
        population_source_h5_path="/tmp/model_data_2025.h5",
        usim_population_households_table="/2025/households",
        usim_population_persons_table="/2025/persons",
        usim_population_jobs_table="/2025/jobs",
        usim_population_blocks_table="/2025/blocks",
    )

    assert captured["population_source_h5_path"] == "/tmp/model_data_2025.h5"
    assert captured["usim_population_households_table"] == "/2025/households"
    assert captured["usim_population_persons_table"] == "/2025/persons"
    assert captured["usim_population_jobs_table"] == "/2025/jobs"
    assert captured["usim_population_blocks_table"] == "/2025/blocks"


def test_activitysim_preprocess_runtime_inputs_prefers_forecast_year(
    monkeypatch,
    tmp_path,
) -> None:
    population_h5 = tmp_path / "model_data_2021.h5"
    population_h5.write_text("population")
    captured_years = []

    def _fake_build_binding_plan(**kwargs):
        captured_years.append(kwargs["year"])
        return BindingPlan(
            inputs={USIM_POPULATION_SOURCE_H5: str(population_h5)},
            source_by_key={USIM_POPULATION_SOURCE_H5: "fallback"},
        )

    monkeypatch.setattr(
        steps_activitysim,
        "build_binding_plan",
        _fake_build_binding_plan,
    )
    monkeypatch.setattr(
        steps_activitysim,
        "resolved_value_for_key",
        lambda *, resolved, key, coupler: (resolved.inputs or {}).get(key),
    )

    runtime_inputs = steps_activitysim._resolve_activitysim_preprocess_runtime_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2019, forecast_year=2021),
        workspace=SimpleNamespace(full_path=str(tmp_path)),
        coupler=_dummy_coupler(),
    )

    assert captured_years == [2021]
    assert runtime_inputs["population_source_h5_path"] == str(population_h5)


def test_activitysim_postprocess_runtime_inputs_alias_population_source_to_current(
    monkeypatch,
    tmp_path,
) -> None:
    h5_path = tmp_path / "model_data_2025.h5"
    h5_path.write_text("x")
    monkeypatch.setattr(
        steps_activitysim,
        "build_binding_plan",
        lambda **_kwargs: BindingPlan(inputs={}, source_by_key={}),
    )

    runtime_inputs = steps_activitysim._resolve_activitysim_postprocess_runtime_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(
            year=2023,
            forecast_year=2025,
            Stage=SimpleNamespace(land_use="land_use"),
            is_enabled=lambda _stage: True,
        ),
        workspace=SimpleNamespace(full_path=str(tmp_path)),
        coupler=_dummy_coupler(),
        step_inputs={USIM_POPULATION_SOURCE_H5: str(h5_path)},
    )

    assert runtime_inputs["population_source_h5_path"] == str(h5_path)
    assert runtime_inputs["current_input_h5_path"] == str(h5_path)


@pytest.mark.parametrize(
    ("current_year", "forecast_year"),
    [
        (2019, 2021),
        (2021, 2023),
    ],
)
def test_activitysim_postprocess_runtime_inputs_split_population_and_current_years(
    monkeypatch,
    tmp_path,
    current_year,
    forecast_year,
) -> None:
    population_h5 = tmp_path / f"model_data_{forecast_year}.h5"
    current_h5 = tmp_path / f"model_data_{current_year}.h5"
    population_h5.write_text("population")
    current_h5.write_text("current")

    captured_years = []

    def _fake_build_binding_plan(**kwargs):
        year = kwargs["year"]
        captured_years.append(year)
        if year == forecast_year:
            return BindingPlan(
                inputs={USIM_POPULATION_SOURCE_H5: str(population_h5)},
                source_by_key={USIM_POPULATION_SOURCE_H5: "fake"},
            )
        if year == current_year:
            return BindingPlan(
                inputs={USIM_DATASTORE_CURRENT_H5: str(current_h5)},
                source_by_key={USIM_DATASTORE_CURRENT_H5: "fake"},
            )
        raise AssertionError(f"Unexpected binding year: {year}")

    monkeypatch.setattr(
        steps_activitysim,
        "build_binding_plan",
        _fake_build_binding_plan,
    )
    monkeypatch.setattr(
        steps_activitysim,
        "resolved_value_for_key",
        lambda *, resolved, key, coupler: (resolved.inputs or {}).get(key),
    )

    runtime_inputs = steps_activitysim._resolve_activitysim_postprocess_runtime_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(
            year=current_year,
            forecast_year=forecast_year,
            Stage=SimpleNamespace(land_use="land_use"),
            is_enabled=lambda _stage: True,
        ),
        workspace=SimpleNamespace(full_path=str(tmp_path)),
        coupler=_dummy_coupler(),
    )

    assert captured_years == [forecast_year, current_year]
    assert runtime_inputs["population_source_h5_path"] == str(population_h5)
    assert runtime_inputs["current_input_h5_path"] == str(current_h5)


def test_activitysim_postprocess_logs_updated_usim_h5_tables(
    monkeypatch, tmp_path
) -> None:
    step_fn = steps.make_activitysim_postprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=SimpleNamespace(),
    )
    output_logger = step_fn.pilates_output_replayer
    publish_calls = []

    monkeypatch.setattr(steps_activitysim, "log_output_only", lambda **_kwargs: None)
    monkeypatch.setattr(
        steps_activitysim,
        "log_and_set_output",
        lambda **kwargs: publish_calls.append(kwargs),
    )

    h5_path = tmp_path / "next_iteration.h5"
    h5_path.write_text("x")
    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=h5_path,
        asim_output_dir=tmp_path,
        processed_outputs={},
    )

    output_logger(
        outputs,
        settings=SimpleNamespace(),
        state=SimpleNamespace(forecast_year=2030, iteration=2),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(),
    )

    assert len(publish_calls) == 1
    assert publish_calls[0]["key"] == "usim_datastore_h5"
    assert publish_calls[0]["child_selection"] == "include_only"
    assert {
        path: spec.key for path, spec in publish_calls[0]["child_specs"].items()
    } == {
        "/households": "activitysim_postprocess_usim_households_table_updated",
        "/persons": "activitysim_postprocess_usim_persons_table_updated",
    }


def test_activitysim_postprocess_publishes_beam_handoff_outputs_to_coupler(
    monkeypatch, tmp_path
) -> None:
    step_fn = steps.make_activitysim_postprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=SimpleNamespace(),
    )
    output_logger = step_fn.pilates_output_replayer
    output_only_calls = []
    publish_calls = []

    monkeypatch.setattr(
        steps_activitysim,
        "log_output_only",
        lambda **kwargs: output_only_calls.append(kwargs["key"]),
    )
    monkeypatch.setattr(
        steps_activitysim,
        "log_and_set_output",
        lambda **kwargs: publish_calls.append(kwargs["key"]),
    )

    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=tmp_path,
        processed_outputs={
            "beam_plans_asim_out": tmp_path / "beam_plans.parquet",
            "households_asim_out": tmp_path / "households.parquet",
            "persons_asim_out": tmp_path / "persons.parquet",
            "trips_asim_out": tmp_path / "trips.parquet",
        },
    )

    output_logger(
        outputs,
        settings=SimpleNamespace(),
        state=SimpleNamespace(forecast_year=2030, iteration=0),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(),
    )

    assert set(publish_calls) == {
        "beam_plans_asim_out",
        "households_asim_out",
        "persons_asim_out",
    }
    assert output_only_calls == ["trips_asim_out"]
