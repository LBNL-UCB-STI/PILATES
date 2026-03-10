from pathlib import Path
from types import SimpleNamespace
import yaml

from pilates.activitysim.outputs import ActivitySimPreprocessOutputs
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    BEAM_PLANS_OUT,
    BEAM_FULL_SKIMS,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_H5,
    USIM_DATASTORE_BASE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_H5_UPDATED,
    USIM_INPUT_MERGED_PREFIX,
    ZARR_SKIMS,
)
from pilates.workflows.orchestration import ManifestConfig, _recover_cached_outputs
from pilates.workflows.orchestration import run_manifested_steps, run_workflow
from pilates.workflows.orchestration import StepRef
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_preprocess_step,
    make_atlas_postprocess_step,
    make_beam_postprocess_step,
    make_urbansim_run_step,
)
from pilates.beam.outputs import BeamRunOutputs


class DummyScenario:
    def __init__(self, cache_hit: bool = True) -> None:
        self.cache_hit = cache_hit
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(cache_hit=self.cache_hit)


class DummyWorkspace:
    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def full_path(self) -> str:
        return str(self._root)

    def get_asim_mutable_data_dir(self) -> str:
        return str(self._root / "activitysim" / "data")

    def get_asim_output_dir(self) -> str:
        return str(self._root / "activitysim" / "output")

    def get_beam_mutable_data_dir(self) -> str:
        return str(self._root / "beam" / "input")

    def get_beam_output_dir(self) -> str:
        return str(self._root / "beam" / "output")

    def get_usim_mutable_data_dir(self) -> str:
        return str(self._root / "urbansim" / "data")

    def get_atlas_mutable_input_dir(self) -> str:
        return str(self._root / "atlas" / "input")

    def get_atlas_output_dir(self) -> str:
        return str(self._root / "atlas" / "output")


class DummyCoupler:
    def __init__(self) -> None:
        self.values = {}

    def get(self, _key, default=None):
        return self.values.get(_key, default)

    def set(self, _key, _value):
        self.values[_key] = _value

    def update(self, _mapping):
        self.values.update(_mapping)


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test")


def test_recover_activitysim_preprocess_outputs(tmp_path):
    workspace = DummyWorkspace(tmp_path)
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    _write_file(asim_dir / "households.csv")
    _write_file(asim_dir / "persons.csv")
    _write_file(asim_dir / "land_use.csv")
    _write_file(asim_dir / "skims.omx")

    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="activitysim_preprocess",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        step_inputs=None,
    )

    assert outputs is not None
    assert holder.activitysim_preprocess is not None
    assert holder.activitysim_preprocess.mutable_data_dir == asim_dir
    assert coupler.get(ASIM_HOUSEHOLDS_IN) is not None
    assert coupler.get(ASIM_PERSONS_IN) is not None
    assert coupler.get(ASIM_LAND_USE_IN) is not None
    holder.activitysim_preprocess.validate()


def test_recover_activitysim_preprocess_outputs_preserves_input_hashes(
    tmp_path, monkeypatch
):
    workspace = DummyWorkspace(tmp_path)
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    _write_file(asim_dir / "households.csv")
    _write_file(asim_dir / "persons.csv")
    _write_file(asim_dir / "land_use.csv")
    _write_file(asim_dir / "skims.omx")

    monkeypatch.setattr(
        "pilates.workflows.orchestration.resolve_artifact_from_value",
        lambda value, *, key=None, workspace=None: SimpleNamespace(hash=f"hash-{key}"),
    )

    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="activitysim_preprocess",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=DummyCoupler(),
        step_inputs=None,
    )

    assert outputs is not None
    assert holder.activitysim_preprocess.input_hashes[ASIM_HOUSEHOLDS_IN] == (
        f"hash-{ASIM_HOUSEHOLDS_IN}"
    )
    assert holder.activitysim_preprocess.input_hashes[ASIM_OMX_SKIMS] == (
        f"hash-{ASIM_OMX_SKIMS}"
    )


def test_recover_activitysim_preprocess_outputs_from_archive_only(tmp_path, monkeypatch):
    local_root = tmp_path / "local-run"
    archive_root = tmp_path / "archive-run"
    workspace = DummyWorkspace(local_root)
    local_asim_dir = Path(workspace.get_asim_mutable_data_dir())
    archive_asim_dir = archive_root / local_asim_dir.relative_to(local_root)

    _write_file(archive_asim_dir / "households.csv")
    _write_file(archive_asim_dir / "persons.csv")
    _write_file(archive_asim_dir / "land_use.csv")
    _write_file(archive_asim_dir / "skims.omx")

    monkeypatch.setenv("PILATES_LOCAL_RUN_DIR", str(local_root))
    monkeypatch.setenv("PILATES_ARCHIVE_RUN_DIR", str(archive_root))

    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="activitysim_preprocess",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        step_inputs=None,
    )

    assert outputs is not None
    assert holder.activitysim_preprocess is not None
    assert (local_asim_dir / "households.csv").exists()
    assert (local_asim_dir / "persons.csv").exists()
    assert (local_asim_dir / "land_use.csv").exists()
    assert (local_asim_dir / "skims.omx").exists()
    assert coupler.get(ASIM_HOUSEHOLDS_IN) is not None
    assert coupler.get(ASIM_PERSONS_IN) is not None
    assert coupler.get(ASIM_LAND_USE_IN) is not None
    holder.activitysim_preprocess.validate()


def test_recover_beam_preprocess_outputs(tmp_path):
    workspace = DummyWorkspace(tmp_path)
    beam_dir = Path(workspace.get_beam_mutable_data_dir())
    plans_path = beam_dir / "seattle" / "urbansim" / "plans.parquet"
    households_path = beam_dir / "seattle" / "urbansim" / "households.parquet"
    persons_path = beam_dir / "seattle" / "urbansim" / "persons.parquet"
    linkstats_path = beam_dir / "seattle" / "r5" / "init.linkstats.csv.gz"
    for path in (plans_path, households_path, persons_path, linkstats_path):
        _write_file(path)

    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="beam_preprocess",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        step_inputs={
            BEAM_PLANS_IN: str(plans_path),
            BEAM_HOUSEHOLDS_IN: str(households_path),
            BEAM_PERSONS_IN: str(persons_path),
            LINKSTATS_WARMSTART: str(linkstats_path),
        },
    )

    assert outputs is not None
    assert holder.beam_preprocess is not None
    assert holder.beam_preprocess.prepared_inputs[BEAM_PLANS_IN] == plans_path
    assert coupler.get(BEAM_PLANS_IN) is not None
    assert coupler.get(BEAM_HOUSEHOLDS_IN) is not None
    assert coupler.get(BEAM_PERSONS_IN) is not None
    assert coupler.get(LINKSTATS_WARMSTART) is not None
    holder.beam_preprocess.validate()


def test_recover_activitysim_run_outputs_carries_source_input_hashes(
    tmp_path, monkeypatch
):
    workspace = DummyWorkspace(tmp_path)
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    output_dir = Path(workspace.get_asim_output_dir())
    iter_dir = output_dir / "year-2018-iteration-0"
    _write_file(asim_dir / "land_use.csv")
    _write_file(asim_dir / "households.csv")
    _write_file(asim_dir / "persons.csv")
    _write_file(asim_dir / "skims.omx")
    _write_file(iter_dir / "households.parquet")
    _write_file(output_dir / "cache" / "skims.zarr")

    monkeypatch.setattr(
        "pilates.workflows.orchestration.resolve_artifact_from_value",
        lambda value, *, key=None, workspace=None: SimpleNamespace(hash=f"hash-{key}"),
    )

    holder = StepOutputsHolder()
    holder.activitysim_preprocess = ActivitySimPreprocessOutputs(
        mutable_data_dir=asim_dir,
        land_use_table=asim_dir / "land_use.csv",
        households_table=asim_dir / "households.csv",
        persons_table=asim_dir / "persons.csv",
        omx_skims=asim_dir / "skims.omx",
        input_hashes={
            ASIM_HOUSEHOLDS_IN: "hash-households",
            ASIM_PERSONS_IN: "hash-persons",
            ASIM_LAND_USE_IN: "hash-land-use",
            ASIM_OMX_SKIMS: "hash-omx",
        },
    )

    outputs = _recover_cached_outputs(
        step_name="activitysim_run",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2018, iteration=0),
        workspace=workspace,
        coupler=DummyCoupler(),
        step_inputs=None,
    )

    assert outputs is not None
    assert holder.activitysim_run.source_input_paths[ASIM_HOUSEHOLDS_IN] == (
        asim_dir / "households.csv"
    )
    assert holder.activitysim_run.source_input_hashes[ASIM_HOUSEHOLDS_IN] == (
        "hash-households"
    )
    assert holder.activitysim_run.source_input_paths[ZARR_SKIMS] == (
        output_dir / "cache" / "skims.zarr"
    )
    assert holder.activitysim_run.source_input_hashes[ZARR_SKIMS] == (
        f"hash-{ZARR_SKIMS}"
    )
    assert holder.activitysim_run.raw_output_hashes["households_asim_out_temp"] == (
        "hash-households_asim_out_temp"
    )


def test_recover_activitysim_run_outputs_does_not_require_source_input_paths_to_exist(
    tmp_path, monkeypatch
):
    workspace = DummyWorkspace(tmp_path)
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    output_dir = Path(workspace.get_asim_output_dir())
    iter_dir = output_dir / "year-2018-iteration-0"
    _write_file(iter_dir / "households.parquet")
    _write_file(output_dir / "cache" / "skims.zarr")

    monkeypatch.setattr(
        "pilates.workflows.orchestration.resolve_artifact_from_value",
        lambda value, *, key=None, workspace=None: SimpleNamespace(hash=f"hash-{key}"),
    )

    holder = StepOutputsHolder()
    holder.activitysim_preprocess = ActivitySimPreprocessOutputs(
        mutable_data_dir=asim_dir,
        land_use_table=asim_dir / "land_use.csv",
        households_table=asim_dir / "households.csv",
        persons_table=asim_dir / "persons.csv",
        omx_skims=asim_dir / "skims.omx",
        input_hashes={
            ASIM_HOUSEHOLDS_IN: "hash-households",
            ASIM_PERSONS_IN: "hash-persons",
            ASIM_LAND_USE_IN: "hash-land-use",
            ASIM_OMX_SKIMS: "hash-omx",
        },
    )

    outputs = _recover_cached_outputs(
        step_name="activitysim_run",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2018, iteration=0),
        workspace=workspace,
        coupler=DummyCoupler(),
        step_inputs=None,
    )

    assert outputs is not None
    assert holder.activitysim_run.raw_outputs["households_asim_out_temp"] == (
        iter_dir / "households.parquet"
    )
    assert holder.activitysim_run.source_input_paths[ASIM_HOUSEHOLDS_IN] == (
        asim_dir / "households.csv"
    )
    assert holder.activitysim_run.source_input_hashes[ASIM_HOUSEHOLDS_IN] == (
        "hash-households"
    )


def test_run_manifested_steps_recovers_cache_hit(tmp_path):
    workspace = DummyWorkspace(tmp_path)
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    _write_file(asim_dir / "households.csv")
    _write_file(asim_dir / "persons.csv")
    _write_file(asim_dir / "land_use.csv")

    def _noop_step(**_kwargs):
        raise AssertionError("step function should not execute on cache hit")
    _noop_step.__consist_step__ = object()
    _noop_step.__pilates_output_replayer__ = (
        lambda outputs, settings, state, workspace, holder: (
            coupler.set(ASIM_HOUSEHOLDS_IN, str(outputs.households_table)),
            coupler.set(ASIM_PERSONS_IN, str(outputs.persons_table)),
            coupler.set(ASIM_LAND_USE_IN, str(outputs.land_use_table)),
            coupler.set("replayed_cache_outputs", str(outputs.households_table)),
        )
    )

    holder = StepOutputsHolder()
    scenario = DummyScenario(cache_hit=True)
    coupler = DummyCoupler()
    manifest_path = tmp_path / "manifest.json"
    manifest_config = ManifestConfig(path=manifest_path)
    state = SimpleNamespace(year=2018, iteration=0)

    run_manifested_steps(
        stage_name="activity_demand_preprocess",
        steps=[
            StepRef(
                name="activitysim_preprocess",
                step_func=_noop_step,
                input_keys=None,
                inputs=None,
            )
        ],
        outputs_holder=holder,
        manifest_config=manifest_config,
        scenario=scenario,
        state=state,
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        name_suffix="2018_iter0",
        iteration=0,
    )

    assert holder.activitysim_preprocess is not None
    assert coupler.get(ASIM_HOUSEHOLDS_IN) is not None
    holder.activitysim_preprocess.validate()
    assert coupler.get("replayed_cache_outputs") == str(
        holder.activitysim_preprocess.households_table
    )
    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["activitysim_preprocess"]["cache_hit"]


def test_recover_activitysim_postprocess_outputs_preserves_hashes(
    tmp_path, monkeypatch
):
    workspace = DummyWorkspace(tmp_path)
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
    _write_file(inputs_dir / "skims.zarr")

    usim_h5 = Path(workspace.get_usim_mutable_data_dir()) / "base.h5"
    _write_file(usim_h5)

    monkeypatch.setattr(
        "pilates.workflows.orchestration.resolve_artifact_from_value",
        lambda value, *, key=None, workspace=None: SimpleNamespace(hash=f"hash-{key}"),
    )

    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="activitysim_postprocess",
        outputs_holder=holder,
        settings=SimpleNamespace(
            urbansim=SimpleNamespace(
                region_id="000",
                region_mappings={"region_to_region_id": {}},
                input_file_template="base_{region_id}.h5",
            ),
            run=SimpleNamespace(region="test"),
        ),
        state=SimpleNamespace(year=2018, forecast_year=2018, iteration=0),
        workspace=workspace,
        coupler=DummyCoupler(),
        step_inputs={USIM_DATASTORE_BASE_H5: str(usim_h5)},
    )

    assert outputs is not None
    assert holder.activitysim_postprocess.processed_output_hashes[
        "households_asim_out"
    ] == "hash-households_asim_out"
    assert holder.activitysim_postprocess.processed_output_hashes[
        "asim_input_households_csv_archived"
    ] == "hash-asim_input_households_csv_archived"
    assert holder.activitysim_postprocess.processed_output_hashes[
        "asim_input_skims_zarr_archived"
    ] == "hash-asim_input_skims_zarr_archived"


def test_recover_beam_run_outputs_from_cached_run_artifacts(tmp_path, monkeypatch):
    workspace = DummyWorkspace(tmp_path)
    beam_iter_dir = (
        Path(workspace.get_beam_output_dir())
        / "seattle"
        / "year-2018-iteration-0"
        / "ITERS"
        / "it.0"
    )
    linkstats_path = beam_iter_dir / "0.linkstats.parquet"
    plans_path = beam_iter_dir / "0.plans.csv.gz"
    events_path = beam_iter_dir / "0.events.parquet"
    for path in (linkstats_path, plans_path, events_path):
        _write_file(path)

    class DummyTracker:
        def get_run_outputs(self, run_id):
            assert run_id == "cached-run-id"
            return {
                "linkstats_parquet_2018_0": str(linkstats_path),
                "beam_plans_out_2018_0": str(plans_path),
                "events_parquet_2018_0": str(events_path),
            }

    monkeypatch.setattr(
        "pilates.workflows.orchestration.cr.current_tracker",
        lambda: DummyTracker(),
    )

    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="beam_run",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        step_inputs=None,
        cached_outputs={
            LINKSTATS: str(linkstats_path),
            BEAM_PLANS_OUT: str(plans_path),
        },
        run_id="cached-run-id",
    )

    assert outputs is not None
    assert holder.beam_run is not None
    assert holder.beam_run.raw_outputs["events_parquet_2018_0"] == events_path
    assert coupler.get("events_parquet_2018_0") is not None
    assert coupler.get("linkstats_parquet_2018_0") is not None


def test_recover_urbansim_run_outputs_from_cached_run_artifacts(tmp_path, monkeypatch):
    workspace = DummyWorkspace(tmp_path)
    usim_output = Path(workspace.get_usim_mutable_data_dir()) / "usim_output.h5"
    _write_file(usim_output)

    class DummyTracker:
        def get_run_outputs(self, run_id):
            assert run_id == "usim-run-id"
            return {USIM_FORECAST_OUTPUT: str(usim_output)}

    monkeypatch.setattr(
        "pilates.workflows.orchestration.cr.current_tracker",
        lambda: DummyTracker(),
    )

    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="urbansim_run",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        step_inputs=None,
        run_id="usim-run-id",
    )

    assert outputs is not None
    assert holder.urbansim_run is not None
    assert holder.urbansim_run.usim_datastore_h5 == usim_output
    assert holder.urbansim_run.raw_outputs[USIM_FORECAST_OUTPUT] == usim_output
    assert coupler.get(USIM_FORECAST_OUTPUT) is not None


def test_recover_urbansim_postprocess_outputs_from_cached_run_artifacts(
    tmp_path, monkeypatch
):
    workspace = DummyWorkspace(tmp_path)
    merged_path = (
        Path(workspace.get_usim_mutable_data_dir()) / "usim_input_merged_2018.h5"
    )
    archived_path = (
        Path(workspace.get_usim_mutable_data_dir()) / "usim_input_archive_2018.h5"
    )
    for path in (merged_path, archived_path):
        _write_file(path)

    merged_key = f"{USIM_INPUT_MERGED_PREFIX}2018"

    class DummyTracker:
        def get_run_outputs(self, run_id):
            assert run_id == "usim-post-id"
            return {
                merged_key: str(merged_path),
                "usim_input_archive_2018": str(archived_path),
            }

    monkeypatch.setattr(
        "pilates.workflows.orchestration.cr.current_tracker",
        lambda: DummyTracker(),
    )

    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="urbansim_postprocess",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        step_inputs=None,
        run_id="usim-post-id",
    )

    assert outputs is not None
    assert holder.urbansim_postprocess is not None
    assert holder.urbansim_postprocess.usim_datastore_h5 == merged_path
    assert holder.urbansim_postprocess.processed_outputs[merged_key] == merged_path
    assert coupler.get(merged_key) is not None


def test_recover_atlas_run_outputs_from_cached_run_artifacts(tmp_path, monkeypatch):
    workspace = DummyWorkspace(tmp_path)
    householdv_path = Path(workspace.get_atlas_output_dir()) / "householdv_2018.csv"
    vehicles_path = Path(workspace.get_atlas_output_dir()) / "vehicles_2018.csv"
    for path in (householdv_path, vehicles_path):
        _write_file(path)

    class DummyTracker:
        def get_run_outputs(self, run_id):
            assert run_id == "atlas-run-id"
            return {
                "householdv_2018": str(householdv_path),
                "vehicles_2018": str(vehicles_path),
            }

    monkeypatch.setattr(
        "pilates.workflows.orchestration.cr.current_tracker",
        lambda: DummyTracker(),
    )

    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="atlas_run",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        step_inputs=None,
        run_id="atlas-run-id",
    )

    assert outputs is not None
    assert holder.atlas_run is not None
    assert holder.atlas_run.raw_outputs["householdv_2018"] == householdv_path
    assert holder.atlas_run.raw_outputs["vehicles_2018"] == vehicles_path
    assert coupler.get("householdv_2018") is not None


def test_recover_atlas_postprocess_outputs_from_cached_run_artifacts(
    tmp_path, monkeypatch
):
    workspace = DummyWorkspace(tmp_path)
    updated_h5 = Path(workspace.get_usim_mutable_data_dir()) / "usim_2018.h5"
    vehicles2 = Path(workspace.get_atlas_output_dir()) / "vehicles2_2018.csv"
    for path in (updated_h5, vehicles2):
        _write_file(path)

    class DummyTracker:
        def get_run_outputs(self, run_id):
            assert run_id == "atlas-post-id"
            return {
                USIM_H5_UPDATED: str(updated_h5),
                "atlas_vehicles2_output": str(vehicles2),
            }

    monkeypatch.setattr(
        "pilates.workflows.orchestration.cr.current_tracker",
        lambda: DummyTracker(),
    )

    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="atlas_postprocess",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        step_inputs=None,
        run_id="atlas-post-id",
    )

    assert outputs is not None
    assert holder.atlas_postprocess is not None
    assert holder.atlas_postprocess.usim_datastore_h5 == updated_h5
    assert holder.atlas_postprocess.processed_outputs[USIM_H5_UPDATED] == updated_h5
    assert holder.atlas_postprocess.processed_outputs["atlas_vehicles2_output"] == vehicles2
    assert coupler.get(USIM_H5_UPDATED) is not None


def test_recover_beam_full_skim_outputs_from_cached_run_artifacts(tmp_path, monkeypatch):
    workspace = DummyWorkspace(tmp_path)
    full_skims = Path(workspace.get_beam_output_dir()) / "full-skims.omx"
    _write_file(full_skims)

    class DummyTracker:
        def get_run_outputs(self, run_id):
            assert run_id == "beam-skim-id"
            return {BEAM_FULL_SKIMS: str(full_skims)}

    monkeypatch.setattr(
        "pilates.workflows.orchestration.cr.current_tracker",
        lambda: DummyTracker(),
    )

    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    outputs = _recover_cached_outputs(
        step_name="beam_full_skim",
        outputs_holder=holder,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        step_inputs=None,
        run_id="beam-skim-id",
    )

    assert outputs is not None
    assert holder.beam_full_skim is not None
    assert holder.beam_full_skim.full_skims == full_skims
    assert coupler.get(BEAM_FULL_SKIMS) is not None


def test_run_workflow_cache_hit_uses_output_replayer(tmp_path):
    workspace = DummyWorkspace(tmp_path)
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    _write_file(asim_dir / "households.csv")
    _write_file(asim_dir / "persons.csv")
    _write_file(asim_dir / "land_use.csv")

    coupler = DummyCoupler()

    class CacheHitScenario:
        def run(self, **_kwargs):
            return SimpleNamespace(cache_hit=True)

    def _noop_step(**_kwargs):
        raise AssertionError("step function should not execute on cache hit")

    _noop_step.__consist_step__ = object()
    _noop_step.__pilates_output_replayer__ = (
        lambda outputs, settings, state, workspace, holder: (
            coupler.set(ASIM_HOUSEHOLDS_IN, str(outputs.households_table)),
            coupler.set(ASIM_PERSONS_IN, str(outputs.persons_table)),
            coupler.set(ASIM_LAND_USE_IN, str(outputs.land_use_table)),
            coupler.set("non_manifest_replay", str(outputs.households_table)),
        )
    )

    holder = StepOutputsHolder()
    run_workflow(
        stage_name="activity_demand_preprocess",
        steps=[StepRef(name="activitysim_preprocess", step_func=_noop_step)],
        scenario=CacheHitScenario(),
        state=SimpleNamespace(year=2018, iteration=0),
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        outputs_holder=holder,
        name_suffix="2018_iter0",
        iteration=0,
    )

    assert holder.activitysim_preprocess is not None
    assert coupler.get("non_manifest_replay") == str(
        holder.activitysim_preprocess.households_table
    )
    assert coupler.get(ASIM_HOUSEHOLDS_IN) is not None


def test_run_workflow_cache_hit_prefers_step_local_recoverer(tmp_path, monkeypatch):
    workspace = DummyWorkspace(tmp_path)
    asim_dir = Path(workspace.get_asim_mutable_data_dir())
    _write_file(asim_dir / "households.csv")
    _write_file(asim_dir / "persons.csv")
    _write_file(asim_dir / "land_use.csv")

    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    step_func = make_activitysim_preprocess_step(coupler=coupler, outputs_holder=holder)

    monkeypatch.setattr(
        "pilates.workflows.orchestration._recover_cached_outputs",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy orchestration recovery should not run")
        ),
    )

    class CacheHitScenario:
        def run(self, **_kwargs):
            return SimpleNamespace(cache_hit=True)

    run_workflow(
        stage_name="activity_demand_preprocess",
        steps=[StepRef(name="activitysim_preprocess", step_func=step_func)],
        scenario=CacheHitScenario(),
        state=SimpleNamespace(year=2018, iteration=0),
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        outputs_holder=holder,
        name_suffix="2018_iter0",
        iteration=0,
    )

    assert holder.activitysim_preprocess is not None
    assert coupler.get(ASIM_HOUSEHOLDS_IN) is not None


def test_run_workflow_cache_hit_beam_postprocess_replays_promoted_outputs(tmp_path):
    workspace = DummyWorkspace(tmp_path)
    coupler = DummyCoupler()
    holder = StepOutputsHolder()

    linkstats = Path(workspace.get_beam_output_dir()) / "linkstats.csv.gz"
    plans = Path(workspace.get_beam_output_dir()) / "plans.csv.gz"
    zarr = Path(workspace.get_beam_output_dir()) / "skims.zarr"
    for path in (linkstats, plans, zarr):
        _write_file(path)

    holder.beam_run = BeamRunOutputs(
        beam_output_dir=Path(workspace.get_beam_output_dir()),
        raw_outputs={
            LINKSTATS: linkstats,
            BEAM_PLANS_OUT: plans,
        },
    )
    step_func = make_beam_postprocess_step(coupler=coupler, outputs_holder=holder)

    class CacheHitScenario:
        def run(self, **_kwargs):
            return SimpleNamespace(cache_hit=True, outputs={ZARR_SKIMS: str(zarr)})

    run_workflow(
        stage_name="beam_postprocess",
        steps=[StepRef(name="beam_postprocess", step_func=step_func)],
        scenario=CacheHitScenario(),
        state=SimpleNamespace(year=2018, forecast_year=2018, iteration=0),
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        outputs_holder=holder,
        name_suffix="2018_iter0",
        iteration=0,
    )

    assert holder.beam_postprocess is not None
    assert coupler.get(ZARR_SKIMS) is not None
    assert coupler.get(LINKSTATS) is not None
    assert coupler.get(BEAM_PLANS_OUT) is not None


def test_run_workflow_cache_hit_urbansim_run_replays_canonical_datastore_key(tmp_path):
    workspace = DummyWorkspace(tmp_path)
    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    holder.urbansim_preprocess = object()

    usim_output = Path(workspace.get_usim_mutable_data_dir()) / "usim_output.h5"
    _write_file(usim_output)
    step_func = make_urbansim_run_step(coupler=coupler, outputs_holder=holder)

    class CacheHitScenario:
        def run(self, **_kwargs):
            return SimpleNamespace(
                cache_hit=True,
                outputs={USIM_FORECAST_OUTPUT: str(usim_output)},
            )

    run_workflow(
        stage_name="urbansim_run",
        steps=[StepRef(name="urbansim_run", step_func=step_func)],
        scenario=CacheHitScenario(),
        state=SimpleNamespace(year=2018, forecast_year=2018, iteration=0),
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        outputs_holder=holder,
        name_suffix="2018_iter0",
        iteration=0,
    )

    assert holder.urbansim_run is not None
    assert coupler.get(USIM_DATASTORE_H5) is not None
    assert coupler.get(USIM_FORECAST_OUTPUT) is None


def test_run_workflow_cache_hit_atlas_postprocess_replays_canonical_datastore_key(
    tmp_path,
):
    workspace = DummyWorkspace(tmp_path)
    coupler = DummyCoupler()
    holder = StepOutputsHolder()
    holder.atlas_run = object()

    updated_h5 = Path(workspace.get_usim_mutable_data_dir()) / "usim_2018.h5"
    vehicles2 = Path(workspace.get_atlas_output_dir()) / "vehicles2_2018.csv"
    for path in (updated_h5, vehicles2):
        _write_file(path)
    step_func = make_atlas_postprocess_step(coupler=coupler, outputs_holder=holder)

    class CacheHitScenario:
        def run(self, **_kwargs):
            return SimpleNamespace(
                cache_hit=True,
                outputs={
                    USIM_H5_UPDATED: str(updated_h5),
                    "atlas_vehicles2_output": str(vehicles2),
                },
            )

    state = SimpleNamespace(
        year=2017,
        forecast_year=2018,
        iteration=0,
        is_start_year=lambda: False,
    )

    run_workflow(
        stage_name="atlas_postprocess",
        steps=[StepRef(name="atlas_postprocess", step_func=step_func)],
        scenario=CacheHitScenario(),
        state=state,
        settings=SimpleNamespace(),
        workspace=workspace,
        coupler=coupler,
        outputs_holder=holder,
        name_suffix="2018_iter0",
        iteration=0,
    )

    assert holder.atlas_postprocess is not None
    assert coupler.get(USIM_DATASTORE_H5) is not None
    assert coupler.get(USIM_H5_UPDATED) is None
