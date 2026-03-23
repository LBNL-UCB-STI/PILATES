from types import SimpleNamespace

from pilates.atlas.inputs import build_atlas_inputs
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_H5_UPDATED,
)


def test_build_atlas_inputs_prefers_current_over_updated_and_fallback():
    coupler = {
        USIM_DATASTORE_CURRENT_H5: "/tmp/current.h5",
        USIM_H5_UPDATED: "/tmp/updated.h5",
        USIM_DATASTORE_BASE_H5: "/tmp/base.h5",
    }

    inputs, _ = build_atlas_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(),
        year=2018,
        coupler=coupler,
        usim_datastore_h5_path="/tmp/fallback.h5",
    )

    assert inputs[USIM_DATASTORE_CURRENT_H5] == "/tmp/current.h5"
    assert inputs[USIM_DATASTORE_BASE_H5] == "/tmp/base.h5"


def test_build_atlas_inputs_uses_updated_when_current_missing():
    coupler = {
        USIM_H5_UPDATED: "/tmp/updated.h5",
    }

    inputs, _ = build_atlas_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(),
        year=2018,
        coupler=coupler,
        usim_datastore_h5_path="/tmp/fallback.h5",
    )

    assert inputs[USIM_DATASTORE_CURRENT_H5] == "/tmp/updated.h5"
    assert inputs[USIM_DATASTORE_BASE_H5] == "/tmp/updated.h5"


def test_build_atlas_inputs_prefers_base_over_current_when_available():
    coupler = {
        USIM_DATASTORE_CURRENT_H5: "/tmp/current.h5",
        USIM_DATASTORE_BASE_H5: "/tmp/base.h5",
    }

    inputs, _ = build_atlas_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(),
        year=2018,
        coupler=coupler,
        usim_datastore_h5_path="/tmp/fallback.h5",
    )

    assert inputs[USIM_DATASTORE_CURRENT_H5] == "/tmp/current.h5"
    assert inputs[USIM_DATASTORE_BASE_H5] == "/tmp/base.h5"


def test_build_atlas_inputs_uses_fallback_when_coupler_missing():
    inputs, _ = build_atlas_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(),
        year=2018,
        coupler={},
        usim_datastore_h5_path="/tmp/fallback.h5",
    )

    assert inputs[USIM_DATASTORE_CURRENT_H5] == "/tmp/fallback.h5"
    assert inputs[USIM_DATASTORE_BASE_H5] == "/tmp/fallback.h5"
