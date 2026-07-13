import types

import pytest
from consist.types import H5ChildSpec

from pilates.utils import consist_runtime as cr
from pilates.utils import coupler_helpers as ch

_H5_PARENT_POLICY = {
    "container_recovery_unit": "parent_file",
    "child_recovery_policy": "descriptive_only",
}


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
        **_H5_PARENT_POLICY,
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
        **_H5_PARENT_POLICY,
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
        **_H5_PARENT_POLICY,
    )

    assert calls
    assert calls[0][0] == "h5_container"
    meta = calls[0][3]
    assert meta["h5_table_paths"] == ["/2023/persons", "/households"]
    assert meta["h5_table_count"] == 2


def test_log_output_only_passes_child_specs_to_h5_container(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    child_specs = {
        "/2023/households": H5ChildSpec(
            key="usim_households_2023",
            description="Households table",
            metadata={"h5_table_name": "households"},
        )
    }

    ch.log_output_only(
        key="usim_datastore_h5",
        path="/tmp/data.h5",
        description="test",
        h5_container=True,
        h5_tables_used=["/2023/households"],
        child_specs=child_specs,
        child_selection="include_only",
        **_H5_PARENT_POLICY,
    )

    assert calls
    assert calls[0][0] == "h5_container"
    meta = calls[0][3]
    assert meta["child_selection"] == "include_only"
    assert meta["child_specs"]["/2023/households"].key == "usim_households_2023"
    assert (
        meta["child_specs"]["/2023/households"].metadata["h5_table_name"]
        == "households"
    )


def test_h5_container_requires_explicit_recovery_policy(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    with pytest.raises(ValueError, match="explicit recovery policy"):
        ch.log_output_only(
            key="usim_datastore_h5",
            path="/tmp/data.h5",
            description="test",
            h5_container=True,
        )

    assert calls == []


def test_h5_recovery_policy_fields_are_preserved_in_container_metadata(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    ch.log_output_only(
        key="usim_datastore_h5",
        path="/tmp/data.h5",
        description="test",
        h5_container=True,
        representation_policy="native_h5",
        **_H5_PARENT_POLICY,
    )

    assert calls
    assert calls[0][0] == "h5_container"
    meta = calls[0][3]
    assert meta["container_recovery_unit"] == "parent_file"
    assert meta["child_recovery_policy"] == "descriptive_only"
    assert meta["representation_policy"] == "native_h5"
