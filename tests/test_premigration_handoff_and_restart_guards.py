from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

if "openmatrix" not in sys.modules:
    openmatrix_stub = types.ModuleType("openmatrix")
    openmatrix_stub.File = object
    sys.modules["openmatrix"] = openmatrix_stub

if "geopandas" not in sys.modules:
    geopandas_stub = types.ModuleType("geopandas")
    geopandas_stub.GeoDataFrame = object
    sys.modules["geopandas"] = geopandas_stub

from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
from pilates.atlas.outputs import AtlasRunOutputs
from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
)
from pilates.workflows.orchestration import (
    ManifestConfig,
    StepRef,
    _recover_cached_outputs,
    run_manifested_steps,
)
from pilates.workflows.outputs_base import serialize_step_outputs
from pilates.workflows.stages.supply_demand import _derive_beam_run_input_keys
from pilates.workflows.stages import vehicle_ownership as vehicle_ownership_stage
from pilates.workflows.steps import StepOutputsHolder


class _DictCoupler:
    def __init__(self) -> None:
        self.values = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value

    def update(self, mapping):
        self.values.update(mapping)


class _ActivitySimWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.full_path = str(root)

    def get_asim_output_dir(self) -> str:
        return str(self.root / "activitysim" / "output")

    def get_usim_mutable_data_dir(self) -> str:
        return str(self.root / "urbansim" / "data")


class _VehicleOwnershipWorkspace:
    def __init__(
        self,
        root: Path,
        *,
        atlas_input_data: RecordStore | None = None,
    ) -> None:
        self.root = root
        self.full_path = str(root)
        self.input_data = {"atlas": atlas_input_data} if atlas_input_data else {}

    def get_usim_mutable_data_dir(self) -> str:
        return str(self.root / "urbansim" / "data")

    def get_atlas_mutable_input_dir(self) -> str:
        return str(self.root / "atlas" / "atlas_input")

    def get_atlas_output_dir(self) -> str:
        return str(self.root / "atlas" / "atlas_output")


class _WorkflowStateStub:
    def __init__(
        self,
        *,
        year: int,
        forecast_year: int,
        start_year: int,
        settings: object,
        run_info_path: str | None = None,
    ) -> None:
        self.year = year
        self.current_year = year
        self.forecast_year = forecast_year
        self.start_year = start_year
        self.full_settings = settings
        self.run_info_path = run_info_path
        self.sub_stage_progress = None

    def is_start_year(self) -> bool:
        return self.year == self.start_year

    def set_sub_stage_progress(self, value: str) -> None:
        self.sub_stage_progress = value


class _ScenarioExecutesStep:
    def __init__(self) -> None:
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["fn"](
            settings=kwargs["execution_options"].runtime_kwargs["settings"],
            state=kwargs["execution_options"].runtime_kwargs["state"],
            workspace=kwargs["execution_options"].runtime_kwargs["workspace"],
        )
        return SimpleNamespace(cache_hit=False)


def _write_file(path: Path, contents: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def _activitysim_postprocess_outputs(
    workspace: _ActivitySimWorkspace,
    *,
    year: int,
    iteration: int,
) -> ActivitySimPostprocessOutputs:
    asim_output_dir = Path(workspace.get_asim_output_dir())
    iter_dir = asim_output_dir / f"year-{year}-iteration-{iteration}"
    households = _write_file(iter_dir / "households.parquet")
    persons = _write_file(iter_dir / "persons.parquet")
    beam_plans = _write_file(iter_dir / "beam_plans.parquet")
    usim_h5 = _write_file(
        Path(workspace.get_usim_mutable_data_dir()) / f"usim_{year}.h5",
        contents="h5",
    )
    return ActivitySimPostprocessOutputs(
        usim_datastore_h5=usim_h5,
        asim_output_dir=asim_output_dir,
        processed_outputs={
            "households_asim_out": households,
            "persons_asim_out": persons,
            "beam_plans_asim_out": beam_plans,
        },
    )


def _vehicle_ownership_settings() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(region="test"),
        urbansim=SimpleNamespace(
            region_mappings={"region_to_region_id": {"test": "000"}},
            input_file_template="usim_{region_id}.h5",
            output_file_template="usim_{year}.h5",
        ),
        atlas=SimpleNamespace(scenario="baseline"),
    )


def test_recover_activitysim_postprocess_outputs_from_cache_hit_artifacts(tmp_path):
    workspace = _ActivitySimWorkspace(tmp_path)
    asim_output_dir = Path(workspace.get_asim_output_dir())
    iter_dir = asim_output_dir / "year-2018-iteration-0"
    _write_file(iter_dir / "households.parquet")
    _write_file(iter_dir / "persons.parquet")
    _write_file(iter_dir / "beam_plans.parquet")

    inputs_dir = asim_output_dir / "inputs-year-2018-iteration-0"
    _write_file(inputs_dir / "households.csv")
    _write_file(inputs_dir / "persons.csv")
    _write_file(inputs_dir / "land_use.csv")
    _write_file(inputs_dir / "skims.omx")

    usim_input = _write_file(
        Path(workspace.get_usim_mutable_data_dir()) / "usim_2018.h5",
        contents="h5",
    )

    coupler = _DictCoupler()
    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="activitysim_postprocess",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2018, forecast_year=2018, iteration=0),
        workspace=workspace,
        coupler=coupler,
        step_inputs={USIM_DATASTORE_BASE_H5: str(usim_input)},
    )

    assert outputs is not None
    assert holder.activitysim_postprocess is not None
    assert holder.activitysim_postprocess.processed_outputs[
        "households_asim_out"
    ] == iter_dir / "households.parquet"
    assert holder.activitysim_postprocess.processed_outputs[
        "persons_asim_out"
    ] == iter_dir / "persons.parquet"
    assert holder.activitysim_postprocess.processed_outputs[
        "beam_plans_asim_out"
    ] == iter_dir / "beam_plans.parquet"
    assert holder.activitysim_postprocess.usim_datastore_h5 == usim_input
    assert coupler.get("households_asim_out") == str(iter_dir / "households.parquet")
    assert coupler.get("beam_plans_asim_out") == str(iter_dir / "beam_plans.parquet")
    assert coupler.get("usim_input_2018") == str(usim_input)
    holder.activitysim_postprocess.validate()


def test_stale_manifest_entry_for_activitysim_postprocess_forces_rerun(tmp_path):
    workspace = _ActivitySimWorkspace(tmp_path)
    manifest_path = tmp_path / "activitysim_postprocess_manifest.yaml"
    stale_outputs = _activitysim_postprocess_outputs(workspace, year=2018, iteration=0)
    serialized = serialize_step_outputs(stale_outputs)
    serialized["processed_outputs"]["beam_plans_asim_out"] = str(
        tmp_path / "missing-beam-plans.parquet"
    )
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "activitysim_postprocess": {
                    "completed_at": "2026-01-01T00:00:00",
                    "cache_hit": True,
                    "outputs": serialized,
                }
            }
        ),
        encoding="utf-8",
    )

    holder = StepOutputsHolder()
    holder.activitysim_run = SimpleNamespace()
    scenario = _ScenarioExecutesStep()
    coupler = _DictCoupler()

    def _rerun_postprocess(**kwargs):
        holder.activitysim_postprocess = _activitysim_postprocess_outputs(
            kwargs["workspace"],
            year=kwargs["state"].year,
            iteration=kwargs["state"].iteration,
        )

    _rerun_postprocess.__consist_step__ = object()

    run_manifested_steps(
        stage_name="activity_demand_postprocess",
        steps=[
            StepRef(
                name="activitysim_postprocess",
                step_func=_rerun_postprocess,
                year=2018,
            )
        ],
        outputs_holder=holder,
        manifest_config=ManifestConfig(path=manifest_path),
        scenario=scenario,
        state=SimpleNamespace(year=2018, forecast_year=2018, iteration=0),
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        name_suffix="2018_iter0",
        iteration=0,
    )

    assert len(scenario.calls) == 1
    assert holder.activitysim_postprocess is not None
    rewritten_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    rewritten_outputs = rewritten_manifest["activitysim_postprocess"]["outputs"]
    assert rewritten_outputs["processed_outputs"]["beam_plans_asim_out"].endswith(
        "beam_plans.parquet"
    )
    assert "missing-beam-plans.parquet" not in yaml.safe_dump(rewritten_manifest)


def test_vehicle_ownership_stage_uses_current_for_start_year_and_forecast_for_subyears(
    tmp_path, monkeypatch
):
    settings = _vehicle_ownership_settings()
    workspace = _VehicleOwnershipWorkspace(tmp_path)
    current_h5 = _write_file(
        Path(workspace.get_usim_mutable_data_dir()) / "custom_current.h5",
        contents="current",
    )
    forecast_h5 = _write_file(
        Path(workspace.get_usim_mutable_data_dir()) / "usim_2024.h5",
        contents="forecast",
    )
    state = _WorkflowStateStub(
        year=2020,
        forecast_year=2024,
        start_year=2020,
        settings=settings,
    )

    monkeypatch.setattr(
        vehicle_ownership_stage,
        "build_urbansim_inputs",
        lambda *_args, **_kwargs: (
            {
                USIM_DATASTORE_CURRENT_H5: str(current_h5),
                USIM_DATASTORE_BASE_H5: str(current_h5),
            },
            {},
        ),
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "atlas_static_input_keys_for_interval",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "merge_model_expected_inputs",
        lambda _model_name, inputs, *_args, **_kwargs: inputs,
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "enqueue_archive_copy",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "flush_archive_queue",
        lambda **_kwargs: None,
    )

    captured_calls = []

    def _fake_run_workflow(*, steps, state, workspace, outputs_holder, **_kwargs):
        captured_calls.append((state.year, steps))
        if any(step.name == "atlas_run" for step in steps):
            raw_output = _write_file(
                Path(workspace.get_atlas_output_dir()) / f"households_{state.year}.csv"
            )
            outputs_holder.atlas_run = AtlasRunOutputs(
                atlas_output_dir=Path(workspace.get_atlas_output_dir()),
                raw_outputs={"atlas_households_csv": raw_output},
            )

    monkeypatch.setattr(vehicle_ownership_stage, "run_workflow", _fake_run_workflow)

    vehicle_ownership_stage.run_vehicle_ownership_stage(
        scenario=SimpleNamespace(),
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=_DictCoupler(),
        year=state.forecast_year,
        build_atlas_static_inputs_fallback=lambda _workspace: {},
    )

    preprocess_calls = [
        (year, next(step for step in steps if step.name == "atlas_preprocess"))
        for year, steps in captured_calls
        if any(step.name == "atlas_preprocess" for step in steps)
    ]
    assert [year for year, _ in preprocess_calls] == [2020, 2022, 2024]
    assert preprocess_calls[0][1].inputs[USIM_DATASTORE_CURRENT_H5] == str(current_h5)
    assert preprocess_calls[0][1].inputs[USIM_DATASTORE_BASE_H5] == str(current_h5)
    for _, step in preprocess_calls[1:]:
        assert step.inputs[USIM_DATASTORE_CURRENT_H5] == str(forecast_h5)
        assert step.inputs[USIM_DATASTORE_BASE_H5] == str(forecast_h5)


def test_vehicle_ownership_stage_uses_workspace_static_registry_without_gap_fill(
    tmp_path, monkeypatch
):
    settings = _vehicle_ownership_settings()
    workspace_static = _write_file(tmp_path / "atlas-static" / "psid_names.Rdat")
    fallback_static = _write_file(tmp_path / "atlas-fallback" / "modeaccessibility.csv")
    atlas_input_data = RecordStore(
        recordList=[
            FileRecord(
                file_path=str(workspace_static),
                short_name="psid_names",
                description="workspace atlas static",
            )
        ]
    )
    workspace = _VehicleOwnershipWorkspace(tmp_path, atlas_input_data=atlas_input_data)
    current_h5 = _write_file(
        Path(workspace.get_usim_mutable_data_dir()) / "custom_current.h5",
        contents="current",
    )
    state = _WorkflowStateStub(
        year=2020,
        forecast_year=2020,
        start_year=2020,
        settings=settings,
    )

    monkeypatch.setattr(
        vehicle_ownership_stage,
        "build_urbansim_inputs",
        lambda *_args, **_kwargs: (
            {
                USIM_DATASTORE_CURRENT_H5: str(current_h5),
                USIM_DATASTORE_BASE_H5: str(current_h5),
            },
            {},
        ),
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "atlas_static_input_keys_for_interval",
        lambda *_args, **_kwargs: ("psid_names", "modeaccessibility"),
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "merge_model_expected_inputs",
        lambda _model_name, inputs, *_args, **_kwargs: inputs,
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "enqueue_archive_copy",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "flush_archive_queue",
        lambda **_kwargs: None,
    )

    captured_run_inputs = {}

    def _fake_run_workflow(*, steps, state, workspace, outputs_holder, **_kwargs):
        atlas_run_step = next((step for step in steps if step.name == "atlas_run"), None)
        if atlas_run_step is not None:
            captured_run_inputs[state.year] = dict(atlas_run_step.inputs or {})
            raw_output = _write_file(
                Path(workspace.get_atlas_output_dir()) / f"households_{state.year}.csv"
            )
            outputs_holder.atlas_run = AtlasRunOutputs(
                atlas_output_dir=Path(workspace.get_atlas_output_dir()),
                raw_outputs={"atlas_households_csv": raw_output},
            )

    monkeypatch.setattr(vehicle_ownership_stage, "run_workflow", _fake_run_workflow)

    vehicle_ownership_stage.run_vehicle_ownership_stage(
        scenario=SimpleNamespace(),
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=_DictCoupler(),
        year=state.forecast_year,
        build_atlas_static_inputs_fallback=lambda _workspace: {
            "psid_names": str(tmp_path / "fallback-should-not-win.Rdat"),
            "modeaccessibility": str(fallback_static),
        },
    )

    atlas_run_inputs = captured_run_inputs[2020]
    assert atlas_run_inputs[USIM_DATASTORE_CURRENT_H5] == str(current_h5)
    assert atlas_run_inputs[USIM_DATASTORE_BASE_H5] == str(current_h5)
    assert atlas_run_inputs["psid_names"] == str(workspace_static)
    assert "modeaccessibility" not in atlas_run_inputs


def test_vehicle_ownership_stage_uses_static_fallback_when_workspace_registry_missing(
    tmp_path, monkeypatch
):
    settings = _vehicle_ownership_settings()
    workspace = _VehicleOwnershipWorkspace(tmp_path)
    current_h5 = _write_file(
        Path(workspace.get_usim_mutable_data_dir()) / "custom_current.h5",
        contents="current",
    )
    fallback_psid = _write_file(tmp_path / "atlas-fallback" / "psid_names.Rdat")
    fallback_mode = _write_file(
        tmp_path / "atlas-fallback" / "modeaccessibility.csv"
    )
    state = _WorkflowStateStub(
        year=2020,
        forecast_year=2020,
        start_year=2020,
        settings=settings,
    )

    monkeypatch.setattr(
        vehicle_ownership_stage,
        "build_urbansim_inputs",
        lambda *_args, **_kwargs: (
            {
                USIM_DATASTORE_CURRENT_H5: str(current_h5),
                USIM_DATASTORE_BASE_H5: str(current_h5),
            },
            {},
        ),
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "atlas_static_input_keys_for_interval",
        lambda *_args, **_kwargs: ("psid_names", "modeaccessibility"),
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "merge_model_expected_inputs",
        lambda _model_name, inputs, *_args, **_kwargs: inputs,
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "enqueue_archive_copy",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        vehicle_ownership_stage,
        "flush_archive_queue",
        lambda **_kwargs: None,
    )

    captured_run_inputs = {}

    def _fake_run_workflow(*, steps, state, workspace, outputs_holder, **_kwargs):
        atlas_run_step = next((step for step in steps if step.name == "atlas_run"), None)
        if atlas_run_step is not None:
            captured_run_inputs[state.year] = dict(atlas_run_step.inputs or {})
            raw_output = _write_file(
                Path(workspace.get_atlas_output_dir()) / f"households_{state.year}.csv"
            )
            outputs_holder.atlas_run = AtlasRunOutputs(
                atlas_output_dir=Path(workspace.get_atlas_output_dir()),
                raw_outputs={"atlas_households_csv": raw_output},
            )

    monkeypatch.setattr(vehicle_ownership_stage, "run_workflow", _fake_run_workflow)

    vehicle_ownership_stage.run_vehicle_ownership_stage(
        scenario=SimpleNamespace(),
        state=state,
        settings=settings,
        workspace=workspace,
        coupler=_DictCoupler(),
        year=state.forecast_year,
        build_atlas_static_inputs_fallback=lambda _workspace: {
            "psid_names": str(fallback_psid),
            "modeaccessibility": str(fallback_mode),
        },
    )

    atlas_run_inputs = captured_run_inputs[2020]
    assert atlas_run_inputs["psid_names"] == str(fallback_psid)
    assert atlas_run_inputs["modeaccessibility"] == str(fallback_mode)


@pytest.mark.parametrize(
    ("beam_preprocess_inputs", "expects_warmstart"),
    [
        (
            {
                BEAM_PLANS_IN: "plans.csv",
                BEAM_HOUSEHOLDS_IN: "households.csv",
                BEAM_PERSONS_IN: "persons.csv",
                LINKSTATS_WARMSTART: "init.linkstats.csv.gz",
            },
            True,
        ),
        (
            {
                BEAM_PLANS_IN: "plans.csv",
                BEAM_HOUSEHOLDS_IN: "households.csv",
                BEAM_PERSONS_IN: "persons.csv",
                "linkstats_parquet_2018_0": "history.parquet",
            },
            False,
        ),
    ],
)
def test_beam_run_input_keys_only_require_exact_warmstart_alias(
    beam_preprocess_inputs, expects_warmstart
):
    input_keys = _derive_beam_run_input_keys(
        beam_preprocess_inputs=beam_preprocess_inputs,
        activity_demand_outputs=None,
    )

    assert BEAM_PLANS_IN in input_keys
    assert BEAM_HOUSEHOLDS_IN in input_keys
    assert BEAM_PERSONS_IN in input_keys
    if expects_warmstart:
        assert LINKSTATS_WARMSTART in input_keys
    else:
        assert LINKSTATS_WARMSTART not in input_keys
