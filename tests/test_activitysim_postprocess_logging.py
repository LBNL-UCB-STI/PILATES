from types import SimpleNamespace

import pytest

from pilates.activitysim.outputs import (
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.binding import BindingPlan
from pilates.workflows.artifact_keys import USIM_INPUT_NEXT
from pilates.workflows import steps
from pilates.workflows.steps import activitysim as steps_activitysim


def _dummy_coupler():
    return SimpleNamespace(
        get=lambda *args, **kwargs: None,
        set=lambda *args, **kwargs: None,
        update=lambda *args, **kwargs: None,
        set_from_artifact=lambda *args, **kwargs: None,
    )


def test_activitysim_postprocess_logs_content_hash(monkeypatch, tmp_path) -> None:
    step_fn = steps.make_activitysim_postprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=SimpleNamespace(),
    )
    output_logger = step_fn.__pilates_output_replayer__
    calls = []
    h5_table_calls = []

    def _log_output_only(*, key, path, description, **meta):
        calls.append((key, meta))

    monkeypatch.setattr(steps_activitysim, "log_output_only", _log_output_only)
    monkeypatch.setattr(
        steps_activitysim,
        "_log_named_h5_tables",
        lambda **kwargs: h5_table_calls.append(kwargs),
    )

    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=tmp_path,
        processed_outputs={
            "asim_input_skims_zarr_archived": tmp_path / "skims.zarr"
        },
        processed_output_hashes={
            "asim_input_skims_zarr_archived": "abc123"
        },
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
    assert h5_table_calls == []


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
    monkeypatch.setattr(steps_activitysim, "_log_named_h5_tables", lambda **_kwargs: None)
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

    step_fn = steps.make_activitysim_postprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=SimpleNamespace(
            activitysim_run=ActivitySimRunOutputs(output_dir=asim_output_dir, raw_outputs={})
        ),
    )
    step_fn(settings=settings, state=state, workspace=workspace)

    keys = {key for key, _path in calls}
    assert "households_asim_in" in keys
    assert "persons_asim_in" in keys
    assert "land_use_asim_in" in keys
    assert "omx_skims" in keys
    assert "zarr_skims" in keys
    assert "usim_datastore_h5" in keys
    assert "usim_forecast_output" in keys


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
    assert outputs.usim_datastore_key == USIM_INPUT_NEXT
    assert list(outputs.to_record_store().to_mapping()) == [USIM_INPUT_NEXT]


def test_activitysim_preprocess_logs_selected_usim_h5_tables(monkeypatch, tmp_path) -> None:
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
    table_calls = []

    monkeypatch.setattr(
        steps_activitysim,
        "build_binding_plan",
        lambda **_kwargs: BindingPlan(
            source_by_key={"usim_h5_updated": "explicit"},
            inputs={"usim_h5_updated": str(tmp_path / "model_data.h5")},
            metadata={
                "selected_key_by_semantic_key": {
                    "usim_h5_updated": "usim_datastore_base_h5"
                }
            },
        ),
    )
    monkeypatch.setattr(
        steps_activitysim,
        "selected_candidate_key",
        lambda _resolved, _key: "usim_datastore_base_h5",
    )
    monkeypatch.setattr(steps_activitysim, "log_and_set_input", lambda **_kwargs: None)
    monkeypatch.setattr(
        steps_activitysim,
        "_log_named_h5_tables",
        lambda **kwargs: table_calls.append(kwargs),
    )
    monkeypatch.setattr(steps_activitysim, "_log_step_records", lambda **_kwargs: None)

    h5_path = tmp_path / "model_data.h5"
    h5_path.write_text("x")
    asim_data_dir = tmp_path / "asim_data"
    asim_data_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("land_use.csv", "households.csv", "persons.csv"):
        (asim_data_dir / filename).write_text("x")

    step_fn = steps.make_activitysim_preprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=SimpleNamespace(),
    )

    step_fn(
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2023, start_year=2017),
        workspace=SimpleNamespace(get_asim_mutable_data_dir=lambda: str(asim_data_dir)),
    )

    assert len(table_calls) == 1
    assert table_calls[0]["direction"] == "input"
    assert table_calls[0]["path"] == str(h5_path)
    assert table_calls[0]["table_keys"]["/households"] == (
        "activitysim_preprocess_usim_households_table_input"
    )
    assert table_calls[0]["table_keys"]["/2017/households"] == (
        "activitysim_preprocess_usim_households_table_start_year_input"
    )


def test_activitysim_postprocess_logs_updated_usim_h5_tables(monkeypatch, tmp_path) -> None:
    step_fn = steps.make_activitysim_postprocess_step(
        coupler=_dummy_coupler(),
        outputs_holder=SimpleNamespace(),
    )
    output_logger = step_fn.__pilates_output_replayer__
    table_calls = []

    monkeypatch.setattr(steps_activitysim, "log_output_only", lambda **_kwargs: None)
    monkeypatch.setattr(steps_activitysim, "log_and_set_output", lambda **_kwargs: None)
    monkeypatch.setattr(
        steps_activitysim,
        "_log_named_h5_tables",
        lambda **kwargs: table_calls.append(kwargs),
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

    assert len(table_calls) == 1
    assert table_calls[0]["direction"] == "output"
    assert table_calls[0]["table_keys"] == {
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
    output_logger = step_fn.__pilates_output_replayer__
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
    monkeypatch.setattr(steps_activitysim, "_log_named_h5_tables", lambda **_kwargs: None)

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
