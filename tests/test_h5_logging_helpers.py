import types
from types import SimpleNamespace

import h5py

from pilates.utils import consist_runtime as cr
from pilates.utils import coupler_helpers as ch
from pilates.workflows.steps import shared as steps_shared


def _install_consist_stub(monkeypatch, calls):
    def _log_input(path, key=None, enabled=None, **meta):
        calls.append(("input", key, meta))
        return {"path": path, "key": key}

    def _log_output(path, key=None, enabled=None, **meta):
        calls.append(("output", key, meta))
        return {"path": path, "key": key}

    def _log_h5_container(path, key=None, direction="input", **meta):
        calls.append(("h5_container", direction, key, meta))
        return {"path": path, "key": key}

    stub = types.SimpleNamespace(
        log_input=_log_input,
        log_output=_log_output,
    )
    monkeypatch.setattr(cr, "consist", stub)
    monkeypatch.setattr(cr, "log_h5_container", _log_h5_container)


def test_log_output_only_uses_h5_container_when_flagged(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    ch.log_output_only(
        key="usim_datastore_h5",
        path="/tmp/data.h5",
        description="test",
        h5_container=True,
    )

    assert calls
    assert calls[0][0] == "h5_container"
    assert calls[0][1] == "output"


def test_log_input_only_falls_back_without_h5_flag(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    ch.log_input_only(
        key="usim_datastore_h5",
        path="/tmp/data.h5",
        description="test",
    )

    assert calls
    assert calls[0][0] == "input"


def test_h5_container_filter_excludes_flattened_pandas_internal_names(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    ch.log_output_only(
        key="usim_datastore_h5",
        path="/tmp/data.h5",
        description="test",
        h5_container=True,
    )

    assert calls
    assert calls[0][0] == "h5_container"
    table_filter = calls[0][3]["table_filter"]
    assert callable(table_filter)
    assert table_filter("/2023/households") is True
    assert table_filter("/axis1") is False
    assert table_filter("/2023/travel_data_axis1_level0") is False


def test_log_output_only_preserves_h5_table_paths_metadata(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    ch.log_output_only(
        key="usim_datastore_h5",
        path="/tmp/data.h5",
        description="test",
        h5_tables_used=["households", "/2023/persons"],
    )

    assert calls
    assert calls[0][0] == "h5_container"
    meta = calls[0][3]
    assert meta["h5_table_paths"] == ["/2023/persons", "/households"]
    assert meta["h5_table_count"] == 2


def test_log_named_h5_tables_skips_missing_datasets(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        steps_shared.cr,
        "log_h5_table",
        lambda path, key=None, table_path=None, direction="input", **meta: calls.append(
            (path, key, table_path, direction, meta)
        ),
    )

    h5_path = tmp_path / "data.h5"
    with h5py.File(h5_path, "w") as h5_file:
        grp = h5_file.create_group("2023")
        grp.create_dataset("households", data=[1, 2, 3])

    steps_shared._log_named_h5_tables(
        path=str(h5_path),
        direction="input",
        table_keys={
            "/households": "usim_households_root",
            "/2023/households": "usim_households_2023",
        },
    )

    assert len(calls) == 1
    assert calls[0][1] == "usim_households_2023"
    assert calls[0][2] == "/2023/households"


def test_log_named_h5_tables_logs_root_datasets_when_present(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        steps_shared.cr,
        "log_h5_table",
        lambda path, key=None, table_path=None, direction="input", **meta: calls.append(
            (path, key, table_path, direction, meta)
        ),
    )

    h5_path = tmp_path / "data.h5"
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("households", data=[1, 2, 3])

    steps_shared._log_named_h5_tables(
        path=str(h5_path),
        direction="input",
        table_keys={
            "/households": "usim_households_root",
            "/2023/households": "usim_households_2023",
        },
    )

    assert len(calls) == 1
    assert calls[0][1] == "usim_households_root"
    assert calls[0][2] == "/households"


def test_log_beam_r5_osm_input_skips_when_config_cache_tables_missing(monkeypatch):
    session_opened = {"value": False}

    class _UnexpectedSession:
        def __init__(self, *_args, **_kwargs):
            session_opened["value"] = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Inspector:
        def has_table(self, table_name, schema=None):
            return False

    monkeypatch.setattr(cr, "current_run", lambda: SimpleNamespace(id="beam-run-1"))
    monkeypatch.setattr(steps_shared, "inspect", lambda _engine: _Inspector())

    import sqlmodel

    monkeypatch.setattr(sqlmodel, "Session", _UnexpectedSession)

    steps_shared._log_beam_r5_osm_input(
        tracker=SimpleNamespace(db=SimpleNamespace(engine=object())),
        settings=SimpleNamespace(
            beam=SimpleNamespace(config="seattle-pilates.conf"),
            run=SimpleNamespace(region="seattle"),
        ),
        workspace=SimpleNamespace(get_beam_mutable_data_dir=lambda: "/tmp/beam"),
    )

    assert session_opened["value"] is False
