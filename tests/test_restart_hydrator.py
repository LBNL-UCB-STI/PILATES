from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from consist import MaterializationResult

from pilates.runtime import restart as restart_runtime
from pilates.runtime.scenario_runtime import DEFAULT_CACHE_EPOCH
from pilates.workflows.catalog import restart_artifact_producers
from workflow_state import WorkflowState


class DummyWorkspace:
    def __init__(self, full_path: str):
        self.full_path = full_path
        self._asim_mutable_data_dir_override = None
        self._asim_runtime_cache_dir_override = None
        self._beam_mutable_data_dir_override = None

    def get_asim_mutable_data_dir(self):
        return self._asim_mutable_data_dir_override or str(
            Path(self.full_path) / "activitysim" / "data"
        )

    def get_asim_output_dir(self):
        return str(Path(self.full_path) / "activitysim" / "output")

    def get_asim_runtime_cache_dir(self):
        return self._asim_runtime_cache_dir_override or str(
            Path(self.get_asim_output_dir()) / "cache"
        )

    def get_beam_mutable_data_dir(self):
        return self._beam_mutable_data_dir_override or str(
            Path(self.full_path) / "beam" / "input"
        )

    def get_beam_output_dir(self):
        return str(Path(self.full_path) / "beam" / "output")

    def set_asim_mutable_data_dir_override(self, path):
        self._asim_mutable_data_dir_override = path

    def set_asim_runtime_cache_dir_override(self, path):
        self._asim_runtime_cache_dir_override = path

    def set_beam_mutable_data_dir_override(self, path):
        self._beam_mutable_data_dir_override = path


class DummyCoupler:
    def __init__(self):
        self._values = {}

    def get(self, key: str):
        return self._values.get(key)

    def set(self, key: str, value):
        self._values[key] = value


class DummyTracker:
    def __init__(self, *, runs_by_target, outputs_by_run, materialized_by_run=None):
        self.runs_by_target = dict(runs_by_target)
        self.outputs_by_run = dict(outputs_by_run)
        self.materialized_by_run = dict(materialized_by_run or {})
        self.find_latest_run_calls = []
        self.materialize_run_output_calls = []

    def find_latest_run(self, **kwargs):
        self.find_latest_run_calls.append(dict(kwargs))
        key = _query_key(**kwargs)
        run_id = self.runs_by_target.get(key)
        if run_id is None:
            raise ValueError(f"no run for target {kwargs}")
        return SimpleNamespace(id=run_id)

    def get_run_outputs(self, run_id):
        return self.outputs_by_run.get(run_id, {})

    def materialize_run_outputs(self, **kwargs):
        self.materialize_run_output_calls.append(dict(kwargs))
        run_id = kwargs["run_id"]
        materialized_path = self.materialized_by_run.get(run_id)
        return MaterializationResult(
            materialized_from_filesystem=(
                {run_id: materialized_path} if materialized_path is not None else {}
            )
        )


def _settings(*, activity_demand="activitysim", traffic_assignment="beam"):
    return SimpleNamespace(
        run=SimpleNamespace(
            region="test-region",
            models=SimpleNamespace(
                activity_demand=activity_demand,
                traffic_assignment=traffic_assignment,
            )
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="scenario",
        ),
    )


def _state(
    *,
    major_stage=WorkflowState.Stage.supply_demand_loop,
    sub_stage=WorkflowState.Stage.traffic_assignment,
    year=2018,
    iteration=1,
):
    return SimpleNamespace(
        current_major_stage=major_stage,
        current_sub_stage=sub_stage,
        current_year=year,
        current_inner_iter=iteration,
    )


def _query_key(**kwargs):
    items = dict(kwargs)
    items.setdefault("cache_epoch", DEFAULT_CACHE_EPOCH)
    if "facet" in items and items["facet"] is not None:
        items["facet"] = tuple(sorted(items["facet"].items()))
    return tuple(sorted(items.items()))


def test_restart_frontier_contract_scopes_v1_to_traffic_assignment():
    contract = restart_runtime.restart_frontier_contract(
        settings=_settings(),
        state=_state(),
        workflow_stage=WorkflowState.Stage,
    )

    assert contract is not None
    assert contract.frontier_stage == "traffic_assignment"
    assert contract.frontier_step == "beam_preprocess"
    assert contract.required_keys == (
        "beam_plans_asim_out",
        "households_asim_out",
        "persons_asim_out",
        "zarr_skims",
    )


def test_hydrate_missing_restart_artifacts_hydrates_traffic_assignment_inputs(tmp_path):
    workspace = DummyWorkspace(str(tmp_path / "run"))
    Path(workspace.full_path).mkdir(parents=True, exist_ok=True)
    coupler = DummyCoupler()

    beam_plans = tmp_path / "restored" / "beam_plans.parquet"
    households = tmp_path / "restored" / "households.parquet"
    persons = tmp_path / "restored" / "persons.parquet"
    zarr = tmp_path / "restored" / "skims.zarr"
    for path in (beam_plans, households, persons, zarr):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("restored", encoding="utf-8")

    facet = {"scenario_id": "scenario-a", "seed": 7}
    tracker = DummyTracker(
        runs_by_target={
            _query_key(
                year=2018,
                iteration=1,
                model="activitysim_postprocess",
                stage="activity_demand_postprocess",
                phase="postprocess",
                status="completed",
                facet=facet,
            ): "asim-post-1",
            _query_key(
                year=2018,
                iteration=1,
                model="activitysim_compile",
                stage="activity_demand_compile",
                phase="compile",
                status="completed",
                facet=facet,
            ): "asim-compile-1",
        },
        outputs_by_run={
            "asim-post-1": {
                "beam_plans_asim_out": str(beam_plans),
                "households_asim_out": str(households),
                "persons_asim_out": str(persons),
            },
            "asim-compile-1": {
                "zarr_skims": str(zarr),
            },
        },
        materialized_by_run={
            "asim-post-1": str(tmp_path / "restored"),
            "asim-compile-1": str(zarr),
        },
    )

    result = restart_runtime.hydrate_missing_restart_artifacts(
        tracker=tracker,
        settings=_settings(),
        state=_state(),
        workspace=workspace,
        coupler=coupler,
        local_run_dir=str(tmp_path / "run"),
        archive_run_dir=str(tmp_path / "archive"),
        workflow_stage=WorkflowState.Stage,
        query_facet=facet,
    )

    assert result["success"] is True
    assert set(result["hydrated_keys"]) == {
        "beam_plans_asim_out",
        "households_asim_out",
        "persons_asim_out",
        "zarr_skims",
    }
    assert result["producer_steps_by_key"] == {
        "beam_plans_asim_out": "activitysim_postprocess",
        "households_asim_out": "activitysim_postprocess",
        "persons_asim_out": "activitysim_postprocess",
        "zarr_skims": "activitysim_compile",
    }
    assert coupler.get("beam_plans_asim_out") == str(beam_plans)
    assert coupler.get("households_asim_out") == str(households)
    assert coupler.get("persons_asim_out") == str(persons)
    assert coupler.get("zarr_skims") == str(zarr)
    assert len(tracker.materialize_run_output_calls) == 4
    assert {tuple(call["keys"]) for call in tracker.materialize_run_output_calls} == {
        ("beam_plans_asim_out",),
        ("households_asim_out",),
        ("persons_asim_out",),
        ("zarr_skims",),
    }
    assert tracker.find_latest_run_calls
    assert all(
        call.get("cache_epoch") == DEFAULT_CACHE_EPOCH
        for call in tracker.find_latest_run_calls
    )


def test_hydrate_missing_restart_artifacts_prefers_exact_iteration_and_exact_facet(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "run"))
    Path(workspace.full_path).mkdir(parents=True, exist_ok=True)
    coupler = DummyCoupler()
    restored = tmp_path / "restored" / "beam_plans.parquet"
    restored.parent.mkdir(parents=True, exist_ok=True)
    restored.write_text("plans", encoding="utf-8")

    exact_facet = {"scenario_id": "scenario-a", "seed": 7}
    tracker = DummyTracker(
        runs_by_target={
            _query_key(
                year=2018,
                iteration=1,
                model="activitysim_postprocess",
                stage="activity_demand_postprocess",
                phase="postprocess",
                status="completed",
                facet=exact_facet,
            ): "asim-post-exact",
            _query_key(
                year=2018,
                iteration=2,
                model="activitysim_postprocess",
                stage="activity_demand_postprocess",
                phase="postprocess",
                status="completed",
                facet=exact_facet,
            ): "asim-post-wrong-iteration",
            _query_key(
                year=2018,
                iteration=1,
                model="activitysim_postprocess",
                stage="activity_demand_postprocess",
                phase="postprocess",
                status="completed",
                facet={"scenario_id": "scenario-b", "seed": 7},
            ): "asim-post-wrong-facet",
            _query_key(
                year=2018,
                model="activitysim_compile",
                stage="activity_demand_compile",
                phase="compile",
                status="completed",
                facet=exact_facet,
            ): "asim-compile-exact",
        },
        outputs_by_run={
            "asim-post-exact": {
                "beam_plans_asim_out": str(restored),
                "households_asim_out": str(restored),
                "persons_asim_out": str(restored),
            },
            "asim-compile-exact": {"zarr_skims": str(restored)},
        },
        materialized_by_run={
            "asim-post-exact": str(restored),
            "asim-compile-exact": str(restored),
        },
    )

    result = restart_runtime.hydrate_missing_restart_artifacts(
        tracker=tracker,
        settings=_settings(),
        state=_state(),
        workspace=workspace,
        coupler=coupler,
        local_run_dir=str(tmp_path / "run"),
        archive_run_dir=str(tmp_path / "archive"),
        workflow_stage=WorkflowState.Stage,
        query_facet=exact_facet,
    )

    assert result["success"] is True
    assert any(call.get("iteration") == 1 and call.get("facet") == exact_facet for call in tracker.find_latest_run_calls)
    assert any("iteration" not in call for call in tracker.find_latest_run_calls if call.get("model") == "activitysim_compile")
    assert all(call.get("facet") == exact_facet for call in tracker.find_latest_run_calls)
    assert all(
        call.get("cache_epoch") == DEFAULT_CACHE_EPOCH
        for call in tracker.find_latest_run_calls
    )
    assert not any(call.get("iteration") == 2 for call in tracker.find_latest_run_calls)
    materialized_run_ids = {
        call["run_id"] for call in tracker.materialize_run_output_calls
    }
    assert "asim-post-exact" in materialized_run_ids
    assert "asim-post-wrong-iteration" not in materialized_run_ids
    assert "asim-post-wrong-facet" not in materialized_run_ids


def test_restart_artifact_producers_applies_traffic_assignment_overrides():
    producers = restart_artifact_producers(
        frontier_stage="traffic_assignment",
        enabled_models=("activitysim", "beam"),
    )

    assert producers["zarr_skims"][0].step_name == "activitysim_compile"
    assert producers["beam_plans_asim_out"][0].step_name == "activitysim_postprocess"
    assert producers["households_asim_out"][0].step_name == "activitysim_postprocess"
    assert producers["persons_asim_out"][0].step_name == "activitysim_postprocess"


def test_hydrate_missing_restart_artifacts_fails_clearly_when_producer_run_missing(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "run"))
    Path(workspace.full_path).mkdir(parents=True, exist_ok=True)

    with pytest.raises(
        restart_runtime.RestartHydrationError,
        match=(
            "frontier_stage=traffic_assignment "
            "frontier_step=beam_preprocess "
            "missing_key=beam_plans_asim_out "
            "producer_step=activitysim_postprocess "
            "reason=no_completed_run_found"
        ),
    ):
        restart_runtime.hydrate_missing_restart_artifacts(
            tracker=DummyTracker(runs_by_target={}, outputs_by_run={}),
            settings=_settings(),
            state=_state(),
            workspace=workspace,
            coupler=DummyCoupler(),
            local_run_dir=str(tmp_path / "run"),
            archive_run_dir=str(tmp_path / "archive"),
            workflow_stage=WorkflowState.Stage,
            query_facet={"scenario_id": "scenario-a", "seed": 7},
        )


def test_hydrate_missing_restart_artifacts_skips_unsupported_frontiers(tmp_path):
    workspace = DummyWorkspace(str(tmp_path / "run"))
    Path(workspace.full_path).mkdir(parents=True, exist_ok=True)

    result = restart_runtime.hydrate_missing_restart_artifacts(
        tracker=DummyTracker(runs_by_target={}, outputs_by_run={}),
        settings=_settings(activity_demand=None, traffic_assignment="beam"),
        state=_state(sub_stage=WorkflowState.Stage.traffic_assignment),
        workspace=workspace,
        coupler=DummyCoupler(),
        local_run_dir=str(tmp_path / "run"),
        archive_run_dir=str(tmp_path / "archive"),
        workflow_stage=WorkflowState.Stage,
        query_facet={"scenario_id": "scenario-a"},
    )

    assert result == {
        "frontier_stage": None,
        "frontier_step": None,
        "success": True,
        "hydrated_keys": [],
        "missing_keys": [],
        "producer_steps_by_key": {},
        "fallback_reason": None,
        "rewind_restore": False,
        "overlay_root": None,
    }


def test_hydrate_missing_restart_artifacts_noops_when_coupler_already_has_contract(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "run"))
    Path(workspace.full_path).mkdir(parents=True, exist_ok=True)
    coupler = DummyCoupler()

    beam_plans = tmp_path / "restored" / "beam_plans.parquet"
    households = tmp_path / "restored" / "households.parquet"
    persons = tmp_path / "restored" / "persons.parquet"
    zarr = tmp_path / "restored" / "skims.zarr"
    for path in (beam_plans, households, persons, zarr):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ready", encoding="utf-8")

    coupler.set("beam_plans_asim_out", str(beam_plans))
    coupler.set("households_asim_out", str(households))
    coupler.set("persons_asim_out", str(persons))
    coupler.set("zarr_skims", str(zarr))

    tracker = DummyTracker(runs_by_target={}, outputs_by_run={})

    result = restart_runtime.hydrate_missing_restart_artifacts(
        tracker=tracker,
        settings=_settings(),
        state=_state(),
        workspace=workspace,
        coupler=coupler,
        local_run_dir=str(tmp_path / "run"),
        archive_run_dir=str(tmp_path / "archive"),
        workflow_stage=WorkflowState.Stage,
        query_facet={"scenario_id": "scenario-a", "seed": 7},
    )

    assert result == {
        "frontier_stage": "traffic_assignment",
        "frontier_step": "beam_preprocess",
        "success": True,
        "hydrated_keys": [],
        "missing_keys": [],
        "producer_steps_by_key": {},
        "fallback_reason": None,
        "rewind_restore": False,
        "overlay_root": None,
    }
    assert tracker.find_latest_run_calls == []
    assert tracker.materialize_run_output_calls == []
    assert coupler.get("beam_plans_asim_out") == str(beam_plans)
    assert coupler.get("households_asim_out") == str(households)
    assert coupler.get("persons_asim_out") == str(persons)
    assert coupler.get("zarr_skims") == str(zarr)


def test_hydrate_missing_restart_artifacts_remaps_stale_raw_paths_into_current_workspace(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "run" / "restart-run"))
    Path(workspace.full_path).mkdir(parents=True, exist_ok=True)
    coupler = DummyCoupler()

    restored_dir = Path(workspace.full_path) / "restored"
    restored_dir.mkdir(parents=True, exist_ok=True)
    beam_plans = restored_dir / "beam_plans.parquet"
    households = restored_dir / "households.parquet"
    persons = restored_dir / "persons.parquet"
    zarr = restored_dir / "skims.zarr"
    for path in (beam_plans, households, persons, zarr):
        path.write_text("materialized", encoding="utf-8")

    stale_root = tmp_path / "old-local" / "restart-run" / "restored"
    tracker = DummyTracker(
        runs_by_target={
            _query_key(
                year=2018,
                iteration=1,
                model="activitysim_postprocess",
                stage="activity_demand_postprocess",
                phase="postprocess",
                status="completed",
                facet={"scenario_id": "scenario-a", "seed": 7},
            ): "asim-post-1",
            _query_key(
                year=2018,
                iteration=1,
                model="activitysim_compile",
                stage="activity_demand_compile",
                phase="compile",
                status="completed",
                facet={"scenario_id": "scenario-a", "seed": 7},
            ): "asim-compile-1",
        },
        outputs_by_run={
            "asim-post-1": {
                "beam_plans_asim_out": str(stale_root / "beam_plans.parquet"),
                "households_asim_out": str(stale_root / "households.parquet"),
                "persons_asim_out": str(stale_root / "persons.parquet"),
            },
            "asim-compile-1": {
                "zarr_skims": str(stale_root / "skims.zarr"),
            },
        },
        materialized_by_run={
            "asim-post-1": str(restored_dir),
            "asim-compile-1": str(restored_dir),
        },
    )

    result = restart_runtime.hydrate_missing_restart_artifacts(
        tracker=tracker,
        settings=_settings(),
        state=_state(),
        workspace=workspace,
        coupler=coupler,
        local_run_dir=workspace.full_path,
        archive_run_dir=str(tmp_path / "archive"),
        workflow_stage=WorkflowState.Stage,
        query_facet={"scenario_id": "scenario-a", "seed": 7},
    )

    assert result["success"] is True
    assert coupler.get("beam_plans_asim_out") == str(beam_plans)
    assert coupler.get("households_asim_out") == str(households)
    assert coupler.get("persons_asim_out") == str(persons)
    assert coupler.get("zarr_skims") == str(zarr)


def test_hydrate_rewind_runner_inputs_restores_activitysim_overlay_with_zarr(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "run"))
    Path(workspace.full_path).mkdir(parents=True, exist_ok=True)
    coupler = DummyCoupler()

    restored_dir = tmp_path / "restored" / "asim"
    restored_dir.mkdir(parents=True, exist_ok=True)
    archived_paths = {
        "asim_input_households_csv_archived": restored_dir / "households.csv",
        "asim_input_persons_csv_archived": restored_dir / "persons.csv",
        "asim_input_land_use_csv_archived": restored_dir / "land_use.csv",
        "asim_input_skims_omx_archived": restored_dir / "skims.omx",
        "asim_input_skims_zarr_archived": restored_dir / "skims.zarr",
    }
    archived_paths["asim_input_households_csv_archived"].write_text(
        "households",
        encoding="utf-8",
    )
    archived_paths["asim_input_persons_csv_archived"].write_text(
        "persons",
        encoding="utf-8",
    )
    archived_paths["asim_input_land_use_csv_archived"].write_text(
        "land_use",
        encoding="utf-8",
    )
    archived_paths["asim_input_skims_omx_archived"].write_text("omx", encoding="utf-8")
    archived_paths["asim_input_skims_zarr_archived"].mkdir(parents=True, exist_ok=True)
    (archived_paths["asim_input_skims_zarr_archived"] / ".zgroup").write_text(
        "{}",
        encoding="utf-8",
    )

    tracker = DummyTracker(
        runs_by_target={
            _query_key(
                year=2018,
                iteration=1,
                model="activitysim_postprocess",
                stage="activity_demand_postprocess",
                phase="postprocess",
                status="completed",
                facet={"scenario_id": "scenario-a", "seed": 7},
            ): "asim-post-1",
        },
        outputs_by_run={
            "asim-post-1": {key: str(path) for key, path in archived_paths.items()},
        },
        materialized_by_run={"asim-post-1": str(restored_dir)},
    )

    result = restart_runtime.hydrate_rewind_runner_inputs(
        tracker=tracker,
        settings=_settings(),
        state=_state(sub_stage=WorkflowState.Stage.activity_demand),
        workspace=workspace,
        coupler=coupler,
        local_run_dir=workspace.full_path,
        archive_run_dir=str(tmp_path / "archive"),
        archive_state_path=str(tmp_path / "archive" / "run_state.yaml"),
        allow_rewind_resume=True,
        workflow_stage=WorkflowState.Stage,
        read_current_stage_fn=lambda _path: (
            2018,
            WorkflowState.Stage.traffic_assignment,
            2,
            False,
            None,
            None,
            True,
        ),
        query_facet={"scenario_id": "scenario-a", "seed": 7},
    )

    assert result is not None
    assert result["success"] is True
    assert result["rewind_restore"] is True
    assert workspace.get_asim_mutable_data_dir().endswith(
        "year-2018-iteration-1/data"
    )
    assert workspace.get_asim_runtime_cache_dir().endswith(
        "year-2018-iteration-1/cache"
    )
    assert Path(workspace.get_asim_mutable_data_dir(), "households.csv").read_text(
        encoding="utf-8"
    ) == "households"
    assert Path(workspace.get_asim_runtime_cache_dir(), "skims.zarr", ".zgroup").exists()
    assert coupler.get("households_asim_in") == str(
        Path(workspace.get_asim_mutable_data_dir()) / "households.csv"
    )
    assert coupler.get("zarr_skims") == str(
        Path(workspace.get_asim_runtime_cache_dir()) / "skims.zarr"
    )


def test_hydrate_rewind_runner_inputs_activitysim_falls_back_to_omx(tmp_path):
    workspace = DummyWorkspace(str(tmp_path / "run"))
    Path(workspace.full_path).mkdir(parents=True, exist_ok=True)
    coupler = DummyCoupler()
    stale_zarr = tmp_path / "stale" / "skims.zarr"
    stale_zarr.parent.mkdir(parents=True, exist_ok=True)
    stale_zarr.write_text("stale", encoding="utf-8")
    coupler.set("zarr_skims", str(stale_zarr))

    restored_dir = tmp_path / "restored" / "asim"
    restored_dir.mkdir(parents=True, exist_ok=True)
    archived_paths = {
        "asim_input_households_csv_archived": restored_dir / "households.csv",
        "asim_input_persons_csv_archived": restored_dir / "persons.csv",
        "asim_input_land_use_csv_archived": restored_dir / "land_use.csv",
        "asim_input_skims_omx_archived": restored_dir / "skims.omx",
    }
    for key, path in archived_paths.items():
        path.write_text(key, encoding="utf-8")

    tracker = DummyTracker(
        runs_by_target={
            _query_key(
                year=2018,
                iteration=1,
                model="activitysim_postprocess",
                stage="activity_demand_postprocess",
                phase="postprocess",
                status="completed",
            ): "asim-post-1",
        },
        outputs_by_run={
            "asim-post-1": {key: str(path) for key, path in archived_paths.items()},
        },
        materialized_by_run={"asim-post-1": str(restored_dir)},
    )

    result = restart_runtime.hydrate_rewind_runner_inputs(
        tracker=tracker,
        settings=_settings(),
        state=_state(sub_stage=WorkflowState.Stage.activity_demand),
        workspace=workspace,
        coupler=coupler,
        local_run_dir=workspace.full_path,
        archive_run_dir=str(tmp_path / "archive"),
        archive_state_path=str(tmp_path / "archive" / "run_state.yaml"),
        allow_rewind_resume=True,
        workflow_stage=WorkflowState.Stage,
        read_current_stage_fn=lambda _path: (
            2018,
            WorkflowState.Stage.traffic_assignment,
            2,
            False,
            None,
            None,
            True,
        ),
        query_facet=None,
    )

    assert result is not None
    assert result["success"] is True
    assert Path(workspace.get_asim_mutable_data_dir(), "skims.omx").read_text(
        encoding="utf-8"
    ) == "asim_input_skims_omx_archived"
    assert not Path(workspace.get_asim_runtime_cache_dir(), "skims.zarr").exists()
    assert coupler.get("zarr_skims") is None


def test_hydrate_rewind_runner_inputs_restores_beam_overlay_and_optional_inputs(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "run"))
    Path(workspace.full_path).mkdir(parents=True, exist_ok=True)
    coupler = DummyCoupler()

    restored_dir = tmp_path / "restored" / "beam"
    restored_dir.mkdir(parents=True, exist_ok=True)
    archived_paths = {
        "beam_input_plans_archived": restored_dir / "beam_input_plans_archived.csv.gz",
        "beam_input_households_archived": restored_dir / "beam_input_households_archived.csv.gz",
        "beam_input_persons_archived": restored_dir / "beam_input_persons_archived.csv.gz",
        "beam_input_config_archived": restored_dir / "beam_input_config_archived.conf",
        "beam_input_config_references_archived": restored_dir / "beam_input_config_references_archived",
        "beam_input_vehicles_archived": restored_dir / "beam_input_vehicles_archived.csv.gz",
        "beam_input_linkstats_warmstart_archived": restored_dir / "beam_input_linkstats_warmstart_archived.csv.gz",
        "beam_input_plans_warmstart_archived": restored_dir / "beam_input_plans_warmstart_archived.xml.gz",
        "beam_input_experienced_plans_warmstart_archived": restored_dir / "beam_input_experienced_plans_warmstart_archived.xml.gz",
    }
    archived_paths["beam_input_config_archived"].write_text(
        "beam config",
        encoding="utf-8",
    )
    archived_paths["beam_input_config_references_archived"].mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        archived_paths["beam_input_config_references_archived"] / "extra.conf"
    ).write_text("beam extra", encoding="utf-8")
    (
        archived_paths["beam_input_config_references_archived"]
        / "scenario"
        / "network.csv"
    ).parent.mkdir(parents=True, exist_ok=True)
    (
        archived_paths["beam_input_config_references_archived"]
        / "scenario"
        / "network.csv"
    ).write_text("network", encoding="utf-8")
    (
        archived_paths["beam_input_config_references_archived"]
        / "__archive_manifest.json"
    ).write_text("{}", encoding="utf-8")
    for key, path in archived_paths.items():
        if key in {
            "beam_input_config_archived",
            "beam_input_config_references_archived",
        }:
            continue
        path.write_text(key, encoding="utf-8")

    tracker = DummyTracker(
        runs_by_target={
            _query_key(
                year=2018,
                iteration=1,
                model="beam_run",
                stage="beam",
                phase="run",
                status="completed",
                facet={"scenario_id": "scenario-a"},
            ): "beam-run-1",
        },
        outputs_by_run={
            "beam-run-1": {key: str(path) for key, path in archived_paths.items()},
        },
        materialized_by_run={"beam-run-1": str(restored_dir)},
    )

    result = restart_runtime.hydrate_rewind_runner_inputs(
        tracker=tracker,
        settings=_settings(),
        state=_state(sub_stage=WorkflowState.Stage.traffic_assignment),
        workspace=workspace,
        coupler=coupler,
        local_run_dir=workspace.full_path,
        archive_run_dir=str(tmp_path / "archive"),
        archive_state_path=str(tmp_path / "archive" / "run_state.yaml"),
        allow_rewind_resume=True,
        workflow_stage=WorkflowState.Stage,
        read_current_stage_fn=lambda _path: (
            2018,
            WorkflowState.Stage.postprocessing,
            2,
            False,
            None,
            None,
            True,
        ),
        query_facet={"scenario_id": "scenario-a"},
    )

    assert result is not None
    assert result["success"] is True
    region_dir = Path(workspace.get_beam_mutable_data_dir()) / "test-region"
    assert (region_dir / "extra.conf").read_text(encoding="utf-8") == "beam extra"
    assert (region_dir / "scenario" / "network.csv").read_text(encoding="utf-8") == "network"
    assert not (region_dir / "__archive_manifest.json").exists()
    scenario_dir = (
        Path(workspace.get_beam_mutable_data_dir()) / "test-region" / "scenario"
    )
    assert (scenario_dir / "plans.csv.gz").read_text(encoding="utf-8") == (
        "beam_input_plans_archived"
    )
    assert (scenario_dir / "vehicles.csv.gz").read_text(encoding="utf-8") == (
        "beam_input_vehicles_archived"
    )
    assert coupler.get("plans_beam_in") == str(scenario_dir / "plans.csv.gz")
    assert coupler.get("vehicles_beam_in") == str(scenario_dir / "vehicles.csv.gz")
    assert coupler.get("atlas_vehicles2_output") == str(scenario_dir / "vehicles.csv.gz")
    assert coupler.get("beam_output_plans_xml").endswith(".xml.gz")
    assert coupler.get("beam_output_experienced_plans_xml").endswith(".xml.gz")


def test_hydrate_rewind_runner_inputs_fails_when_activitysim_snapshots_incomplete(
    tmp_path,
):
    workspace = DummyWorkspace(str(tmp_path / "run"))
    Path(workspace.full_path).mkdir(parents=True, exist_ok=True)

    restored_dir = tmp_path / "restored" / "asim"
    restored_dir.mkdir(parents=True, exist_ok=True)
    archived_paths = {
        "asim_input_households_csv_archived": restored_dir / "households.csv",
        "asim_input_persons_csv_archived": restored_dir / "persons.csv",
        "asim_input_land_use_csv_archived": restored_dir / "land_use.csv",
    }
    for path in archived_paths.values():
        path.write_text("x", encoding="utf-8")

    tracker = DummyTracker(
        runs_by_target={
            _query_key(
                year=2018,
                iteration=1,
                model="activitysim_postprocess",
                stage="activity_demand_postprocess",
                phase="postprocess",
                status="completed",
            ): "asim-post-1",
        },
        outputs_by_run={
            "asim-post-1": {key: str(path) for key, path in archived_paths.items()},
        },
        materialized_by_run={"asim-post-1": str(restored_dir)},
    )

    with pytest.raises(
        restart_runtime.RestartHydrationError,
        match="reason=producer_run_missing_declared_output",
    ):
        restart_runtime.hydrate_rewind_runner_inputs(
            tracker=tracker,
            settings=_settings(),
            state=_state(sub_stage=WorkflowState.Stage.activity_demand),
            workspace=workspace,
            coupler=DummyCoupler(),
            local_run_dir=workspace.full_path,
            archive_run_dir=str(tmp_path / "archive"),
            archive_state_path=str(tmp_path / "archive" / "run_state.yaml"),
            allow_rewind_resume=True,
            workflow_stage=WorkflowState.Stage,
            read_current_stage_fn=lambda _path: (
                2018,
                WorkflowState.Stage.traffic_assignment,
                2,
                False,
                None,
                None,
                True,
            ),
            query_facet=None,
        )
