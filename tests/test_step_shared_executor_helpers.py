from __future__ import annotations

from pathlib import Path

from pilates.activitysim.outputs import ActivitySimPreprocessOutputs, ActivitySimRunOutputs
from pilates.atlas.outputs import AtlasPreprocessOutputs
from pilates.beam.outputs import BeamPreprocessOutputs, BeamRunOutputs
from pilates.urbansim.outputs import UrbanSimPreprocessOutputs
from pilates.workflows.steps.activitysim import (
    _execute_activitysim_postprocess,
    _execute_activitysim_preprocess,
)
from pilates.workflows.steps.beam import (
    _execute_beam_postprocess,
    _execute_beam_full_skim,
    _execute_beam_preprocess,
)
from pilates.workflows.steps.urbansim_atlas import (
    _execute_atlas_preprocess_typed,
    _execute_urbansim_preprocess_typed,
)
from pilates.workflows.artifact_keys import BEAM_HOUSEHOLDS_IN, BEAM_PLANS_IN
from pilates.workflows.steps.shared import StepOutputsHolder


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


def test_execute_activitysim_postprocess_forwards_typed_run_outputs_with_metadata_hook(
    monkeypatch, tmp_path: Path
) -> None:
    holder = StepOutputsHolder()
    raw_path = tmp_path / "raw.parquet"
    raw_path.write_text("raw", encoding="utf-8")
    holder.activitysim_run = ActivitySimRunOutputs(
        output_dir=tmp_path / "asim-output",
        raw_outputs={"households_asim_out_temp": raw_path},
        raw_output_hashes={"households_asim_out_temp": "raw-hash"},
        source_input_paths={
            "households_asim_in": tmp_path / "households.csv",
            "zarr_skims": tmp_path / "asim-output" / "cache" / "skims.zarr",
        },
        source_input_hashes={
            "households_asim_in": "preprocess-hash",
            "zarr_skims": "zarr-hash",
        },
    )

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

    result = _execute_activitysim_postprocess(
        postprocessor=_Postprocessor(),
        workspace=workspace,
        outputs_holder=holder,
    )

    assert result is holder.activitysim_run
    assert captured["raw_outputs"] is holder.activitysim_run
    assert captured["raw_outputs"].raw_output_hashes == {
        "households_asim_out_temp": "raw-hash",
    }
    assert captured["raw_outputs"].source_input_hashes == {
        "households_asim_in": "preprocess-hash",
        "zarr_skims": "zarr-hash",
    }
    assert captured["raw_outputs"].source_input_paths == {
        "households_asim_in": tmp_path / "households.csv",
        "zarr_skims": tmp_path / "asim-output" / "cache" / "skims.zarr",
    }


def test_execute_activitysim_preprocess_strips_runtime_only_kwargs(
    tmp_path: Path,
) -> None:
    captured = {}

    class _Preprocessor:
        def preprocess(self, workspace):
            captured["workspace"] = workspace
            return ActivitySimPreprocessOutputs(
                mutable_data_dir=tmp_path,
                land_use_table=tmp_path / "land_use.csv",
                households_table=tmp_path / "households.csv",
                persons_table=tmp_path / "persons.csv",
            )

    workspace = type("Workspace", (), {"full_path": str(tmp_path)})()
    result = _execute_activitysim_preprocess(
        preprocessor=_Preprocessor(),
        workspace=workspace,
        outputs_holder=StepOutputsHolder(),
        coupler=object(),
        context="activitysim_preprocess",
    )

    assert result.mutable_data_dir == tmp_path


def test_execute_beam_postprocess_forwards_explicit_zarr_skims_when_supported(
    tmp_path: Path,
) -> None:
    holder = StepOutputsHolder()
    beam_output_dir = tmp_path / "beam-output"
    beam_output_dir.mkdir(parents=True, exist_ok=True)
    run_output = beam_output_dir / "events.parquet"
    run_output.write_text("events", encoding="utf-8")
    holder.beam_run = BeamRunOutputs(
        beam_output_dir=beam_output_dir,
        raw_outputs={"events_parquet_2018_0": run_output},
    )

    captured = {}

    class _Postprocessor:
        def postprocess(
            self,
            raw_outputs,
            workspace,
            model_run_hash=None,
            zarr_skims=None,
        ):
            captured["raw_outputs"] = raw_outputs
            captured["workspace"] = workspace
            captured["zarr_skims"] = zarr_skims
            return raw_outputs

    workspace = type("Workspace", (), {"full_path": str(tmp_path)})()
    resolved_zarr = str(tmp_path / "restored" / "skims.zarr")

    result = _execute_beam_postprocess(
        postprocessor=_Postprocessor(),
        workspace=workspace,
        outputs_holder=holder,
        zarr_skims=resolved_zarr,
    )

    assert result is holder.beam_run
    assert captured["raw_outputs"] is holder.beam_run
    assert captured["workspace"] is workspace
    assert captured["zarr_skims"] == resolved_zarr
    assert captured["workspace"] is workspace


def test_execute_urbansim_preprocess_strips_runtime_only_kwargs(
    tmp_path: Path,
) -> None:
    captured = {}

    class _Preprocessor:
        def preprocess(self, workspace):
            captured["workspace"] = workspace
            return UrbanSimPreprocessOutputs(usim_mutable_data_dir=tmp_path)

    workspace = type("Workspace", (), {"full_path": str(tmp_path)})()
    result = _execute_urbansim_preprocess_typed(
        preprocessor=_Preprocessor(),
        workspace=workspace,
        outputs_holder=StepOutputsHolder(),
        coupler=object(),
        context="urbansim_preprocess",
    )

    assert result.usim_mutable_data_dir == tmp_path
    assert captured["workspace"] is workspace


def test_execute_atlas_preprocess_strips_runtime_only_kwargs(
    tmp_path: Path,
) -> None:
    captured = {}

    class _Preprocessor:
        def preprocess(self, workspace):
            captured["workspace"] = workspace
            return AtlasPreprocessOutputs(
                atlas_mutable_input_dir=tmp_path,
                prepared_inputs={},
            )

    workspace = type("Workspace", (), {"full_path": str(tmp_path)})()
    result = _execute_atlas_preprocess_typed(
        preprocessor=_Preprocessor(),
        workspace=workspace,
        outputs_holder=StepOutputsHolder(),
        coupler=object(),
        context="atlas_preprocess",
    )

    assert result.atlas_mutable_input_dir == tmp_path
    assert captured["workspace"] is workspace


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
