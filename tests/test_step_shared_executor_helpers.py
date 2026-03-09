from __future__ import annotations

from pathlib import Path

import pytest

from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.steps.shared import (
    StepOutputsHolder,
    _build_required_input_store,
)


class _FakeStepOutputs:
    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def _iter_record_items(self):
        for record in self._store.all_records():
            yield record.short_name, Path(record.file_path), record.description


class _LegacyOnlyStepOutputs:
    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def to_record_store(self) -> RecordStore:
        return self._store


class _KeysOnlyCoupler:
    def __init__(self, keys: list[str]) -> None:
        self._keys = keys

    def keys(self):
        return list(self._keys)


def _store_with_record(path: Path, key: str) -> RecordStore:
    return RecordStore(
        recordList=[
            FileRecord(
                file_path=str(path),
                short_name=key,
                description=f"record for {key}",
            )
        ]
    )


def test_build_required_input_store_raises_when_upstream_missing() -> None:
    holder = StepOutputsHolder()

    with pytest.raises(RuntimeError, match="ActivitySim preprocess must complete first"):
        _build_required_input_store(
            outputs_holder=holder,
            upstream_attr="activitysim_preprocess",
            missing_message="ActivitySim preprocess must complete first",
            context="activitysim_run",
        )


def test_build_required_input_store_merges_optional_extra_inputs(tmp_path: Path) -> None:
    holder = StepOutputsHolder()
    holder.activitysim_preprocess = _FakeStepOutputs(
        _store_with_record(tmp_path / "input-a.txt", "input_a")
    )
    extra_inputs = _store_with_record(tmp_path / "input-b.txt", "input_b")

    merged = _build_required_input_store(
        outputs_holder=holder,
        upstream_attr="activitysim_preprocess",
        missing_message="ActivitySim preprocess must complete first",
        context="activitysim_run",
        extra_inputs=extra_inputs,
    )

    assert set(merged.to_mapping().keys()) == {"input_a", "input_b"}


def test_build_required_input_store_warns_for_keys_missing_from_coupler(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    holder = StepOutputsHolder()
    holder.beam_preprocess = _FakeStepOutputs(
        _store_with_record(tmp_path / "missing.txt", "missing_key")
    )
    coupler = _KeysOnlyCoupler(keys=["some_other_key"])

    with caplog.at_level("WARNING"):
        _build_required_input_store(
            outputs_holder=holder,
            upstream_attr="beam_preprocess",
            missing_message="BEAM preprocess must complete first",
            context="beam_run",
            coupler=coupler,
        )

    assert "[beam_run] Input RecordStore keys missing from coupler" in caplog.text
    assert "missing_key" in caplog.text


def test_build_required_input_store_rejects_legacy_only_outputs(tmp_path: Path) -> None:
    holder = StepOutputsHolder()
    holder.activitysim_preprocess = _LegacyOnlyStepOutputs(
        _store_with_record(tmp_path / "legacy.txt", "legacy_key")
    )

    with pytest.raises(TypeError, match="_iter_record_items"):
        _build_required_input_store(
            outputs_holder=holder,
            upstream_attr="activitysim_preprocess",
            missing_message="ActivitySim preprocess must complete first",
            context="activitysim_run",
        )
