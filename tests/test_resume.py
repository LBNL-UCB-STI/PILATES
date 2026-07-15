from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.workflows.resume import (
    HistoricalOutputRequest,
    ResumeBoundaryPolicy,
    ResumeDecision,
    ResumeDisposition,
    ResumePlanningError,
    ResumeProjectionError,
    build_resume_plan,
    execute_restore_decision,
)
from pilates.workflows.beam_checkpoint import (
    assert_committed_beam_run,
    mark_beam_postprocess_in_progress,
    publish_beam_run_checkpoint,
    read_beam_run_checkpoint,
    snapshot_and_publish_beam_run_checkpoint,
)


@dataclass
class _Run:
    id: str
    status: str = "completed"


class _Surface:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def step_enabled(self, _step_name: str) -> bool:
        return self.enabled


class _Tracker:
    def __init__(self, candidates=(), outputs=None, hydration=None) -> None:
        self.candidates = list(candidates)
        self.outputs = dict(outputs or {"linkstats": object()})
        self.hydration = hydration
        self.find_calls = []
        self.hydrate_calls = []

    def find_matching_runs(self, **kwargs):
        self.find_calls.append(kwargs)
        return self.candidates

    def get_run_outputs(self, _run_id: str):
        return self.outputs

    def hydrate_run_outputs_to_destinations(self, run_id: str, **kwargs):
        self.hydrate_calls.append((run_id, kwargs))
        return self.hydration


class _Hydration(dict):
    def __init__(self, source_run_id: str, items):
        super().__init__(items)
        self.source_run_id = source_run_id


def _policy(
    *, requests=(HistoricalOutputRequest("linkstats", Path("/tmp/linkstats"), True),)
):
    return ResumeBoundaryPolicy(
        step_name="beam_run",
        rerun_forbidden=True,
        allows_restore=lambda _state, _surface: True,
        output_requests=lambda _keys, _workspace, _year, _iteration: requests,
    )


def _target(*_args, **_kwargs):
    return {
        "model": "beam_run",
        "status": "completed",
        "run_scope": "campaign-a",
    }


def test_build_resume_plan_skips_disabled_step(monkeypatch):
    monkeypatch.setattr("pilates.workflows.resume.restart_target_for_step", _target)
    tracker = _Tracker(candidates=[_Run("run-1")])

    plan = build_resume_plan(
        state=object(),
        surface=_Surface(enabled=False),
        settings=object(),
        workspace=object(),
        tracker=tracker,
        year=2020,
        iteration=0,
        policy=_policy(),
    )

    decision = plan.decisions["beam_run"]
    assert decision.disposition is ResumeDisposition.SKIP
    assert decision.reason == "outside_enabled_surface"
    assert tracker.find_calls == []


def test_build_resume_plan_returns_run_when_no_completed_match(monkeypatch):
    monkeypatch.setattr("pilates.workflows.resume.restart_target_for_step", _target)
    tracker = _Tracker()

    plan = build_resume_plan(
        state=object(),
        surface=_Surface(),
        settings=object(),
        workspace=object(),
        tracker=tracker,
        year=2020,
        iteration=0,
        policy=_policy(),
    )

    decision = plan.decisions["beam_run"]
    assert decision.disposition is ResumeDisposition.RUN
    assert decision.reason == "no_completed_match"
    assert decision.source_run_id is None
    assert tracker.find_calls == [{**_target(), "limit": 2}]


@pytest.mark.parametrize(
    ("target", "candidates", "match"),
    [
        ({"status": "completed"}, [_Run("run-1")], "missing_workflow_scope"),
        (_target(), [_Run("run-1"), _Run("run-2")], "ambiguous_completed_match"),
        (_target(), [_Run("run-1", status="running")], "malformed_completed_match"),
    ],
)
def test_build_resume_plan_rejects_unsafe_candidates(
    monkeypatch, target, candidates, match
):
    monkeypatch.setattr(
        "pilates.workflows.resume.restart_target_for_step",
        lambda *_args, **_kwargs: target,
    )

    with pytest.raises(ResumePlanningError, match=match):
        build_resume_plan(
            state=object(),
            surface=_Surface(),
            settings=object(),
            workspace=object(),
            tracker=_Tracker(candidates=candidates),
            year=2020,
            iteration=0,
            policy=_policy(),
        )


def test_build_resume_plan_restores_one_completed_run(monkeypatch, tmp_path):
    monkeypatch.setattr("pilates.workflows.resume.restart_target_for_step", _target)
    request = HistoricalOutputRequest("linkstats", tmp_path / "linkstats.csv.gz", True)
    tracker = _Tracker(candidates=[_Run("run-1")])

    plan = build_resume_plan(
        state=object(),
        surface=_Surface(),
        settings=object(),
        workspace=object(),
        tracker=tracker,
        year=2020,
        iteration=0,
        policy=_policy(requests=(request,)),
    )

    assert plan.workflow_instance_scope == "campaign-a"
    assert plan.decisions["beam_run"] == ResumeDecision(
        step_name="beam_run",
        disposition=ResumeDisposition.RESTORE,
        reason="completed_match",
        semantic_target=_target(),
        source_run_id="run-1",
        outputs=(request,),
        rerun_forbidden=True,
    )


def test_build_resume_plan_rejects_empty_required_restore_contract(monkeypatch):
    monkeypatch.setattr("pilates.workflows.resume.restart_target_for_step", _target)

    with pytest.raises(ResumePlanningError, match="destination_contract_error"):
        build_resume_plan(
            state=object(),
            surface=_Surface(),
            settings=object(),
            workspace=object(),
            tracker=_Tracker(candidates=[_Run("run-1")]),
            year=2020,
            iteration=0,
            policy=_policy(requests=()),
        )


def test_execute_restore_decision_uses_exact_destinations_and_projects(tmp_path):
    destination = tmp_path / "linkstats.csv.gz"
    decision = ResumeDecision(
        step_name="beam_run",
        disposition=ResumeDisposition.RESTORE,
        reason="completed_match",
        semantic_target=_target(),
        source_run_id="run-1",
        outputs=(HistoricalOutputRequest("linkstats", destination, True),),
        rerun_forbidden=True,
    )
    item = SimpleNamespace(
        path=destination,
        status="materialized_from_filesystem",
        resolvable=True,
        artifact=object(),
    )
    tracker = _Tracker(hydration=_Hydration("run-1", {"linkstats": item}))
    projected = []

    result = execute_restore_decision(
        decision=decision,
        tracker=tracker,
        source_root=tmp_path / "archive",
        projection_adapter=lambda hydration: (
            projected.append(hydration) or ("outputs", ("linkstats",))
        ),
    )

    assert result.succeeded is True
    assert result.projected_outputs == "outputs"
    assert result.published_role_keys == ("linkstats",)
    assert projected == [tracker.hydration]
    assert tracker.hydrate_calls == [
        (
            "run-1",
            {
                "destinations_by_key": {"linkstats": destination},
                "source_root": tmp_path / "archive",
                "preserve_existing": False,
                "on_missing": "warn",
                "db_fallback": "never",
            },
        )
    ]


@pytest.mark.parametrize(
    ("item", "expected_category"),
    [
        (
            SimpleNamespace(
                path=Path("/wrong"),
                status="materialized_from_filesystem",
                resolvable=True,
                artifact=object(),
            ),
            "missing_required_output",
        ),
        (
            SimpleNamespace(
                path=None, status="missing_source", resolvable=False, artifact=object()
            ),
            "missing_required_output",
        ),
        (
            SimpleNamespace(
                path=Path("/tmp/linkstats"),
                status="materialized_from_db",
                resolvable=True,
                artifact=object(),
            ),
            "missing_required_output",
        ),
    ],
)
def test_execute_restore_decision_rejects_non_strict_required_results(
    tmp_path, item, expected_category
):
    destination = tmp_path / "linkstats"
    if item.path == Path("/tmp/linkstats"):
        item.path = destination
    decision = ResumeDecision(
        step_name="beam_run",
        disposition=ResumeDisposition.RESTORE,
        reason="completed_match",
        semantic_target=_target(),
        source_run_id="run-1",
        outputs=(HistoricalOutputRequest("linkstats", destination, True),),
        rerun_forbidden=True,
    )
    tracker = _Tracker(hydration=_Hydration("run-1", {"linkstats": item}))
    projected = []

    result = execute_restore_decision(
        decision=decision,
        tracker=tracker,
        source_root=None,
        projection_adapter=lambda hydration: projected.append(hydration),
    )

    assert result.succeeded is False
    assert result.failure_category == expected_category
    assert result.failed_keys == ("linkstats",)
    assert projected == []


def test_execute_restore_decision_rejects_preexisting_destination_before_hydration(
    tmp_path,
):
    destination = tmp_path / "linkstats"
    destination.write_text("stale", encoding="utf-8")
    decision = ResumeDecision(
        step_name="beam_run",
        disposition=ResumeDisposition.RESTORE,
        reason="completed_match",
        semantic_target=_target(),
        source_run_id="run-1",
        outputs=(HistoricalOutputRequest("linkstats", destination, True),),
        rerun_forbidden=True,
    )
    tracker = _Tracker()

    result = execute_restore_decision(
        decision=decision,
        tracker=tracker,
        source_root=None,
        projection_adapter=lambda _hydration: None,
    )

    assert result.failure_category == "preexisting_restore_destination"
    assert tracker.hydrate_calls == []


def test_execute_restore_decision_preserves_projection_error_category(tmp_path):
    destination = tmp_path / "linkstats"
    decision = ResumeDecision(
        step_name="beam_run",
        disposition=ResumeDisposition.RESTORE,
        reason="completed_match",
        semantic_target=_target(),
        source_run_id="run-1",
        outputs=(HistoricalOutputRequest("linkstats", destination, True),),
        rerun_forbidden=True,
    )
    item = SimpleNamespace(
        path=destination,
        status="materialized_from_filesystem",
        resolvable=True,
        artifact=object(),
    )
    tracker = _Tracker(hydration=_Hydration("run-1", {"linkstats": item}))

    result = execute_restore_decision(
        decision=decision,
        tracker=tracker,
        source_root=None,
        projection_adapter=lambda _hydration: (_ for _ in ()).throw(
            ResumeProjectionError("unsupported_output_representation", "directory")
        ),
    )

    assert result.failure_category == "unsupported_output_representation"


def test_beam_checkpoint_is_atomic_and_becomes_nonrestartable(tmp_path):
    checkpoint = publish_beam_run_checkpoint(
        archive_run_dir=tmp_path,
        producer_run_id="beam-run-1",
        scope={"year": 2019, "forecast_year": 2021, "iteration": 0},
        snapshot_ref=".consist/restart/checkpoints/pinned/tracker.duckdb",
        skim_variant="full",
        output_requests=(
            HistoricalOutputRequest("linkstats", tmp_path / "beam" / "0.linkstats", True),
        ),
    )

    assert checkpoint.producer_run_id == "beam-run-1"
    assert read_beam_run_checkpoint(tmp_path) == checkpoint

    mark_beam_postprocess_in_progress(tmp_path, checkpoint)
    assert read_beam_run_checkpoint(tmp_path) is None


def test_committed_beam_run_requires_completed_direct_run_and_selected_links(tmp_path):
    run = SimpleNamespace(status="completed", year=2019, iteration=0)
    tracker = SimpleNamespace(
        get_run=lambda run_id: run if run_id == "beam-run-1" else None,
        get_run_outputs=lambda _run_id: {"linkstats": object()},
    )
    checkpoint = publish_beam_run_checkpoint(
        archive_run_dir=tmp_path,
        producer_run_id="beam-run-1",
        scope={"year": 2019, "forecast_year": 2021, "iteration": 0},
        snapshot_ref=".consist/restart/checkpoints/pinned/tracker.duckdb",
        skim_variant="full",
        output_requests=(
            HistoricalOutputRequest("linkstats", tmp_path / "beam" / "0.linkstats", True),
        ),
    )

    assert assert_committed_beam_run(
        tracker=tracker,
        checkpoint=checkpoint,
        output_requests=(
            HistoricalOutputRequest("linkstats", tmp_path / "beam" / "0.linkstats", True),
        ),
    ) is run


def test_snapshot_publication_validates_the_pinned_snapshot_not_a_matching_query(tmp_path):
    run = SimpleNamespace(status="completed", year=2019, iteration=0)
    calls = []

    class _LiveTracker:
        def snapshot_db(self, destination, *, checkpoint):
            calls.append((Path(destination), checkpoint))
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            Path(destination).write_text("snapshot", encoding="utf-8")

    snapshot_tracker = SimpleNamespace(
        get_run=lambda run_id: run if run_id == "beam-run-1" else None,
        get_run_outputs=lambda _run_id: {"linkstats": object()},
    )
    checkpoint = snapshot_and_publish_beam_run_checkpoint(
        tracker=_LiveTracker(),
        open_snapshot=lambda path: snapshot_tracker,
        archive_run_dir=tmp_path,
        producer_run_id="beam-run-1",
        scope={"year": 2019, "forecast_year": 2021, "iteration": 0},
        skim_variant="full",
        output_requests=(
            HistoricalOutputRequest("linkstats", tmp_path / "beam" / "0.linkstats", True),
        ),
    )

    assert calls and calls[0][1] is True
    assert (tmp_path / checkpoint.snapshot_ref).is_file()


def test_failed_snapshot_does_not_publish_a_beam_checkpoint(tmp_path):
    class _Tracker:
        def snapshot_db(self, *_args, **_kwargs):
            raise RuntimeError("snapshot failure")

    with pytest.raises(RuntimeError, match="snapshot failure"):
        snapshot_and_publish_beam_run_checkpoint(
            tracker=_Tracker(),
            open_snapshot=lambda _path: pytest.fail("must not open snapshot"),
            archive_run_dir=tmp_path,
            producer_run_id="beam-run-1",
            scope={"year": 2019, "forecast_year": 2021, "iteration": 0},
            skim_variant="full",
            output_requests=(
                HistoricalOutputRequest("linkstats", tmp_path / "beam" / "0.linkstats", True),
            ),
        )

    assert read_beam_run_checkpoint(tmp_path) is None


def test_real_consist_snapshot_reopens_pinned_completed_beam_run(tmp_path):
    import consist

    archive_run_dir = tmp_path / "archive"
    db_path = archive_run_dir / ".consist" / "provenance.duckdb"
    output_path = archive_run_dir / "beam" / "0.linkstats.csv.gz"
    db_path.parent.mkdir(parents=True)
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"linkstats")
    tracker = consist.Tracker(run_dir=archive_run_dir, db_path=db_path)
    with tracker.start_run(
        "beam-run-1", "beam_run", year=2019, iteration=0
    ):
        tracker.log_output(output_path, key="linkstats")

    def _open_snapshot(snapshot_path: Path):
        return consist.Tracker(
            run_dir=archive_run_dir,
            db_path=snapshot_path,
            allow_external_paths=True,
            access_mode="read_only",
        )

    checkpoint = snapshot_and_publish_beam_run_checkpoint(
        tracker=tracker,
        open_snapshot=_open_snapshot,
        archive_run_dir=archive_run_dir,
        producer_run_id="beam-run-1",
        scope={"year": 2019, "forecast_year": 2021, "iteration": 0},
        skim_variant="disabled",
        output_requests=(
            HistoricalOutputRequest(
                "linkstats", tmp_path / "workspace" / "0.linkstats.csv.gz", True
            ),
        ),
    )

    snapshot = _open_snapshot(archive_run_dir / checkpoint.snapshot_ref)
    assert snapshot.get_run("beam-run-1").status == "completed"
    assert set(snapshot.get_run_outputs("beam-run-1")) == {"linkstats"}
