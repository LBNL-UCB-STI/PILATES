from __future__ import annotations

from pathlib import Path

import pytest

from pilates.beam.outputs import BeamPreprocessOutputs
from pilates.workflows.steps.beam import (
    _execute_beam_full_skim,
    _execute_beam_preprocess,
)
from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.artifact_keys import BEAM_HOUSEHOLDS_IN, BEAM_PLANS_IN
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


class _ArtifactValue:
    def __init__(self, path: Path, *, content_hash: str | None = None) -> None:
        self.path = str(path)
        self.content_hash = content_hash


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


def _beam_preprocess_outputs(
    tmp_path: Path,
    prepared_inputs: dict[str, Path],
) -> BeamPreprocessOutputs:
    beam_dir = tmp_path / "beam-input"
    beam_dir.mkdir(parents=True, exist_ok=True)
    return BeamPreprocessOutputs(
        beam_mutable_data_dir=beam_dir,
        prepared_inputs=prepared_inputs,
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
    extra_path = tmp_path / "input-b.txt"
    extra_path.write_text("input_b", encoding="utf-8")
    extra_inputs = {"input_b": extra_path}

    merged = _build_required_input_store(
        outputs_holder=holder,
        upstream_attr="activitysim_preprocess",
        missing_message="ActivitySim preprocess must complete first",
        context="activitysim_run",
        workspace=type("Workspace", (), {"full_path": str(tmp_path)})(),
        extra_inputs=extra_inputs,
    )

    assert set(merged.to_mapping().keys()) == {"input_a", "input_b"}


def test_build_required_input_store_preserves_extra_input_content_hashes(
    tmp_path: Path,
) -> None:
    holder = StepOutputsHolder()
    holder.activitysim_preprocess = _FakeStepOutputs(
        _store_with_record(tmp_path / "input-a.txt", "input_a")
    )
    extra_path = tmp_path / "input-b.txt"
    extra_path.write_text("input_b", encoding="utf-8")

    merged = _build_required_input_store(
        outputs_holder=holder,
        upstream_attr="activitysim_preprocess",
        missing_message="ActivitySim preprocess must complete first",
        context="activitysim_run",
        workspace=type("Workspace", (), {"full_path": str(tmp_path)})(),
        extra_inputs={"input_b": _ArtifactValue(extra_path, content_hash="hash-b")},
        warn_missing_coupler_inputs=False,
    )

    record = _record_by_short_name(merged, "input_b")
    assert record is not None
    assert record.content_hash == "hash-b"


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

    class _Postprocessor:
        def postprocess(self, raw_outputs, workspace, model_run_hash=None):
            captured["postprocessor"] = self
            captured["raw_outputs"] = raw_outputs
            captured["workspace"] = workspace
            return raw_outputs

    workspace = type(
        "Workspace",
        (),
        {"get_asim_output_dir": lambda self: str(tmp_path / "asim-output")},
    )()

    result = _execute_postprocess(
        postprocessor=_Postprocessor(),
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


def test_execute_beam_preprocess_aliases_fallback_inputs(tmp_path: Path) -> None:
    activity_path = tmp_path / "plans.parquet"
    activity_path.write_text("plans", encoding="utf-8")
    warmstart_path = tmp_path / "history.parquet"
    warmstart_path.write_text("history", encoding="utf-8")
    fallback_path = tmp_path / "fallback.parquet"
    fallback_path.write_text("fallback", encoding="utf-8")

    captured = {}

    class _Preprocessor:
        def preprocess(
            self,
            workspace,
            *,
            activity_demand_outputs=None,
            previous_beam_outputs=None,
            beam_preprocess_inputs=None,
        ):
            captured["workspace"] = workspace
            captured["activity_demand_outputs"] = activity_demand_outputs
            captured["previous_beam_outputs"] = previous_beam_outputs
            captured["beam_preprocess_inputs"] = beam_preprocess_inputs
            return _beam_preprocess_outputs(
                tmp_path,
                {
                    "beam_plans_out": activity_path,
                    "linkstats_parquet_2018_0": warmstart_path,
                    "beam_plans": fallback_path,
                },
            )

    workspace = type("Workspace", (), {"full_path": str(tmp_path)})()
    holder = StepOutputsHolder()

    result = _execute_beam_preprocess(
        preprocessor=_Preprocessor(),
        workspace=workspace,
        outputs_holder=holder,
        activity_demand_outputs={"beam_plans_out": activity_path},
        previous_beam_outputs={"linkstats_parquet_2018_0": warmstart_path},
        beam_preprocess_inputs={BEAM_PLANS_IN: fallback_path},
    )

    assert result.prepared_inputs == {
        "beam_plans_out": activity_path,
        "linkstats_parquet_2018_0": warmstart_path,
        "beam_plans": fallback_path,
    }
    assert captured["activity_demand_outputs"] == {"beam_plans_out": activity_path}
    assert captured["previous_beam_outputs"] == {
        "linkstats_parquet_2018_0": warmstart_path
    }
    assert captured["beam_preprocess_inputs"] == {BEAM_PLANS_IN: fallback_path}


def test_execute_beam_preprocess_keeps_earlier_duplicate_key_canonical(
    tmp_path: Path,
) -> None:
    canonical_path = tmp_path / "canonical.parquet"
    canonical_path.write_text("canonical", encoding="utf-8")
    fallback_path = tmp_path / "fallback.parquet"
    fallback_path.write_text("fallback", encoding="utf-8")

    class _Preprocessor:
        def preprocess(self, workspace, **kwargs):
            return _beam_preprocess_outputs(
                tmp_path,
                {
                    "beam_plans": canonical_path,
                    "fallback_duplicate": fallback_path,
                },
            )

    result = _execute_beam_preprocess(
        preprocessor=_Preprocessor(),
        workspace=type("Workspace", (), {"full_path": str(tmp_path)})(),
        outputs_holder=StepOutputsHolder(),
        activity_demand_outputs={"beam_plans": canonical_path},
        previous_beam_outputs=None,
        beam_preprocess_inputs={BEAM_PLANS_IN: fallback_path},
    )

    mapping = result.prepared_inputs
    assert mapping["beam_plans"] == canonical_path
    assert str(fallback_path) in {str(path) for path in mapping.values()}


def test_execute_beam_preprocess_keeps_previous_outputs_from_overwriting_canonical(
    tmp_path: Path,
) -> None:
    canonical_path = tmp_path / "canonical.parquet"
    canonical_path.write_text("canonical", encoding="utf-8")
    previous_path = tmp_path / "previous.parquet"
    previous_path.write_text("previous", encoding="utf-8")

    class _Preprocessor:
        def preprocess(self, workspace, **kwargs):
            return _beam_preprocess_outputs(
                tmp_path,
                {
                    "beam_plans": canonical_path,
                    "previous_duplicate": previous_path,
                },
            )

    result = _execute_beam_preprocess(
        preprocessor=_Preprocessor(),
        workspace=type("Workspace", (), {"full_path": str(tmp_path)})(),
        outputs_holder=StepOutputsHolder(),
        activity_demand_outputs={"beam_plans": canonical_path},
        previous_beam_outputs={"beam_plans": previous_path},
        beam_preprocess_inputs=None,
    )

    mapping = result.prepared_inputs
    assert mapping["beam_plans"] == canonical_path
    assert str(previous_path) in {str(path) for path in mapping.values()}


def test_execute_beam_preprocess_preserves_three_source_precedence_and_aliases(
    tmp_path: Path,
) -> None:
    canonical_path = tmp_path / "canonical.parquet"
    canonical_path.write_text("canonical", encoding="utf-8")
    previous_path = tmp_path / "previous.parquet"
    previous_path.write_text("previous", encoding="utf-8")
    fallback_path = tmp_path / "fallback.parquet"
    fallback_path.write_text("fallback", encoding="utf-8")
    households_path = tmp_path / "households.csv"
    households_path.write_text("households", encoding="utf-8")
    warmstart_path = tmp_path / "warmstart.parquet"
    warmstart_path.write_text("warmstart", encoding="utf-8")

    class _Preprocessor:
        def preprocess(self, workspace, **kwargs):
            return _beam_preprocess_outputs(
                tmp_path,
                {
                    "beam_plans": canonical_path,
                    "households": households_path,
                    "linkstats_parquet_2018_0": warmstart_path,
                    "previous_duplicate": previous_path,
                    "fallback_duplicate": fallback_path,
                },
            )

    result = _execute_beam_preprocess(
        preprocessor=_Preprocessor(),
        workspace=type("Workspace", (), {"full_path": str(tmp_path)})(),
        outputs_holder=StepOutputsHolder(),
        activity_demand_outputs={"beam_plans": canonical_path},
        previous_beam_outputs={
            "beam_plans": previous_path,
            "linkstats_parquet_2018_0": warmstart_path,
        },
        beam_preprocess_inputs={
            BEAM_PLANS_IN: fallback_path,
            BEAM_HOUSEHOLDS_IN: households_path,
        },
    )

    mapping = result.prepared_inputs
    assert mapping["beam_plans"] == canonical_path
    assert mapping["households"] == households_path
    assert mapping["linkstats_parquet_2018_0"] == warmstart_path
    mapped_values = {str(path) for path in mapping.values()}
    assert str(previous_path) in mapped_values
    assert str(fallback_path) in mapped_values


def test_execute_beam_preprocess_omits_missing_optional_inputs(tmp_path: Path) -> None:
    captured = {}

    class _Preprocessor:
        def preprocess(
            self,
            workspace,
            *,
            activity_demand_outputs=None,
            previous_beam_outputs=None,
            beam_preprocess_inputs=None,
        ):
            captured["activity_demand_outputs"] = activity_demand_outputs
            captured["previous_beam_outputs"] = previous_beam_outputs
            captured["beam_preprocess_inputs"] = beam_preprocess_inputs
            return _beam_preprocess_outputs(tmp_path, {})

    workspace = type("Workspace", (), {"full_path": str(tmp_path)})()

    result = _execute_beam_preprocess(
        preprocessor=_Preprocessor(),
        workspace=workspace,
        outputs_holder=StepOutputsHolder(),
        activity_demand_outputs={"beam_plans_out": None},
        previous_beam_outputs=None,
        beam_preprocess_inputs={BEAM_HOUSEHOLDS_IN: None},
    )

    assert result.prepared_inputs == {}
    assert captured["activity_demand_outputs"] == {"beam_plans_out": None}
    assert captured["previous_beam_outputs"] is None
    assert captured["beam_preprocess_inputs"] == {BEAM_HOUSEHOLDS_IN: None}


def test_execute_beam_full_skim_materializes_warm_start_inputs(tmp_path: Path) -> None:
    holder = StepOutputsHolder()
    holder.beam_preprocess = _beam_preprocess_outputs(
        tmp_path,
        {"beam_plans": tmp_path / "plans.txt"},
    )
    (tmp_path / "plans.txt").write_text("plans", encoding="utf-8")
    warmstart_path = tmp_path / "history.parquet"
    warmstart_path.write_text("history", encoding="utf-8")
    captured = {}

    class _Runner:
        def run(self, input_outputs, workspace, *, previous_beam_outputs=None):
            captured["input_outputs"] = input_outputs
            captured["workspace"] = workspace
            captured["previous_beam_outputs"] = previous_beam_outputs
            return object()

    workspace = type("Workspace", (), {"full_path": str(tmp_path)})()
    result = _execute_beam_full_skim(
        runner=_Runner(),
        workspace=workspace,
        outputs_holder=holder,
        previous_beam_outputs={"linkstats_parquet_2018_0": warmstart_path},
        context="beam_full_skim_run",
    )

    assert result is not None
    assert captured["input_outputs"] is holder.beam_preprocess
    assert captured["input_outputs"].prepared_inputs["beam_plans"] == tmp_path / "plans.txt"
    assert captured["previous_beam_outputs"] == {
        "linkstats_parquet_2018_0": warmstart_path
    }
