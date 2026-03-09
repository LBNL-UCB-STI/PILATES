from __future__ import annotations

from pathlib import Path

import pytest

from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.steps.shared import (
    StepOutputsHolder,
    _build_required_input_store,
    _execute_postprocess,
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


class _HashTrackingStepOutputs(_FakeStepOutputs):
    def __init__(self, store: RecordStore, hashes: dict[str, str]) -> None:
        super().__init__(store)
        self.input_hashes = hashes


class _RunHashTrackingStepOutputs(_FakeStepOutputs):
    def __init__(self, store: RecordStore, hashes: dict[str, str]) -> None:
        super().__init__(store)
        self.raw_output_hashes = hashes

    def to_postprocess_record_store(self) -> RecordStore:
        record_store = self._store
        setattr(
            record_store,
            "activitysim_source_input_paths",
            {
                "households_asim_in": str(Path(record_store.all_records()[0].file_path).with_name("households.csv")),
                "zarr_skims": str(Path(record_store.all_records()[0].file_path).parent / "cache" / "skims.zarr"),
            },
        )
        setattr(
            record_store,
            "activitysim_source_input_hashes",
            {
                "households_asim_in": "preprocess-hash",
                "zarr_skims": "zarr-hash",
            },
        )
        return record_store


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


def _record_by_short_name(store: RecordStore, short_name: str) -> FileRecord | None:
    for record in store.all_records():
        if record.short_name == short_name:
            return record
    return None


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


def test_build_required_input_store_preserves_content_hashes(tmp_path: Path) -> None:
    holder = StepOutputsHolder()
    source_path = tmp_path / "input.txt"
    store = _store_with_record(source_path, "input_a")
    holder.activitysim_preprocess = _HashTrackingStepOutputs(
        store,
        hashes={"input_a": "hash-input-a"},
    )

    materialized = _build_required_input_store(
        outputs_holder=holder,
        upstream_attr="activitysim_preprocess",
        missing_message="ActivitySim preprocess must complete first",
        context="activitysim_run",
        warn_missing_coupler_inputs=False,
    )

    record = _record_by_short_name(materialized, "input_a")
    assert record is not None
    assert record.content_hash == "hash-input-a"


def test_execute_postprocess_uses_activitysim_run_postprocess_record_store(
    monkeypatch, tmp_path: Path
) -> None:
    holder = StepOutputsHolder()
    raw_store = RecordStore(
        recordList=[
            FileRecord(
                file_path=str(tmp_path / "raw.parquet"),
                short_name="households_asim_out_temp",
                description="raw output",
                content_hash="raw-hash",
            )
        ]
    )
    holder.activitysim_run = _RunHashTrackingStepOutputs(
        raw_store,
        hashes={"households_asim_out_temp": "raw-hash"},
    )

    zarr_path = tmp_path / "asim-output" / "cache" / "skims.zarr"
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    zarr_path.write_text("zarr", encoding="utf-8")

    captured = {}

    def _fake_run_postprocessor(postprocessor, raw_outputs, workspace, model_run_hash=None):
        captured["postprocessor"] = postprocessor
        captured["raw_outputs"] = raw_outputs
        captured["workspace"] = workspace
        return raw_outputs

    monkeypatch.setattr(
        "pilates.workflows.steps.shared.run_postprocessor",
        _fake_run_postprocessor,
    )

    workspace = type(
        "Workspace",
        (),
        {"get_asim_output_dir": lambda self: str(tmp_path / "asim-output")},
    )()

    result = _execute_postprocess(
        postprocessor=object(),
        workspace=workspace,
        outputs_holder=holder,
    )

    assert result is captured["raw_outputs"]
    raw_record = _record_by_short_name(captured["raw_outputs"], "households_asim_out_temp")
    assert raw_record is not None and raw_record.content_hash == "raw-hash"
    assert getattr(captured["raw_outputs"], "activitysim_source_input_hashes") == {
        "households_asim_in": "preprocess-hash",
        "zarr_skims": "zarr-hash",
    }
