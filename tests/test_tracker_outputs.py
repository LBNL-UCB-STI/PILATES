from __future__ import annotations

import pytest

from pilates.workflows import tracker_outputs
from pilates.workflows.steps import activitysim as activitysim_steps


class _FakeTracker:
    def __init__(self, run_outputs):
        self.run_outputs = run_outputs

    def get_run_outputs(self, run_id):
        assert run_id == "run-1"
        return self.run_outputs


def test_load_tracker_run_outputs_canonicalizes_and_drops_none(monkeypatch):
    tracker = _FakeTracker(
        {
            "urbansim/usim_datastore_h5": "/tmp/model_data.h5",
            "ignored_none": None,
        }
    )
    monkeypatch.setattr(tracker_outputs.cr, "current_tracker", lambda: tracker)

    assert tracker_outputs.load_tracker_run_outputs("run-1") == {
        "usim_datastore_h5": "/tmp/model_data.h5",
    }


def test_merge_canonical_output_mappings_prefers_later_mappings():
    assert tracker_outputs.merge_canonical_output_mappings(
        {"urbansim/usim_datastore_h5": "/tmp/model_data_a.h5"},
        {"usim_datastore_h5": "/tmp/model_data_b.h5"},
    ) == {"usim_datastore_h5": "/tmp/model_data_b.h5"}


def test_activitysim_cached_run_outputs_use_shared_tracker_helper(monkeypatch):
    tracker = _FakeTracker({"urbansim/usim_datastore_h5": "/tmp/model_data.h5"})
    monkeypatch.setattr(activitysim_steps.cr, "current_tracker", lambda: tracker)

    assert activitysim_steps._resolve_cached_run_outputs("run-1") == {
        "usim_datastore_h5": "/tmp/model_data.h5",
    }


def test_load_tracker_run_outputs_raises_without_active_tracker(monkeypatch):
    monkeypatch.setattr(tracker_outputs.cr, "current_tracker", lambda: None)

    with pytest.raises(RuntimeError, match="no active Consist tracker"):
        tracker_outputs.load_tracker_run_outputs("run-1")


def test_load_tracker_run_outputs_raises_without_get_run_outputs(monkeypatch):
    monkeypatch.setattr(tracker_outputs.cr, "current_tracker", lambda: object())

    with pytest.raises(RuntimeError, match="get_run_outputs"):
        tracker_outputs.load_tracker_run_outputs("run-1")
