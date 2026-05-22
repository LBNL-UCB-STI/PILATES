from __future__ import annotations

import pytest

from pilates.utils.consist_capabilities import (
    check_consist_runtime_capabilities,
    require_consist_runtime_capabilities,
)


class _OldTracker:
    def find_runs(self, **_kwargs):
        return []

    def log_h5_container(self, path, key=None, direction="output"):
        return path, key, direction


class _NewTracker:
    def find_matching_run(self, **_kwargs):
        return None

    def register_run_output_recovery_copies(self, *_args, **_kwargs):
        return None

    def log_h5_container(
        self,
        path,
        key=None,
        direction="output",
        *,
        container_recovery_unit=None,
        child_recovery_policy=None,
        representation_policy=None,
    ):
        return (
            path,
            key,
            direction,
            container_recovery_unit,
            child_recovery_policy,
            representation_policy,
        )


def test_consist_capability_report_rejects_old_tracker() -> None:
    report = check_consist_runtime_capabilities(_OldTracker())

    assert report.ok is False
    assert "tracker.find_matching_run" in report.missing
    assert "tracker.register_run_output_recovery_copies" in report.missing
    assert any("log_h5_container" in item for item in report.missing)


def test_consist_capability_report_accepts_new_tracker() -> None:
    report = check_consist_runtime_capabilities(_NewTracker())

    assert report.ok is True
    assert report.missing == ()


def test_require_consist_capabilities_raises_readable_error() -> None:
    with pytest.raises(RuntimeError, match="semantic run matching"):
        require_consist_runtime_capabilities(_OldTracker())
