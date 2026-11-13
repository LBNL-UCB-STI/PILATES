import hashlib
from pathlib import Path

import consist
from consist import Tracker
from consist.core.directory_artifacts import build_directory_manifest
from consist.types import ExecutionOptions

from pilates.utils import consist_runtime as cr


def _content_identity(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if path.is_dir():
        return build_directory_manifest(path)["tree_hash"]
    raise AssertionError(f"Consist input has no consumable bytes: {path}")


def _input_content_identities(tracker: Tracker, run_id: str) -> dict[str, str]:
    artifacts = tracker.get_artifacts_for_run(run_id)
    identities: dict[str, str] = {}
    for artifact in sorted(
        artifacts.inputs.values(), key=lambda artifact: artifact.key
    ):
        content_identity = _content_identity(artifact.as_path(tracker=tracker))
        artifact_digest_hex = artifact.digest.hex if artifact.digest is not None else ""
        assert content_identity == artifact_digest_hex, (
            f"{artifact.key} bytes do not match Consist-selected artifact identity"
        )
        identities[artifact.key] = content_identity
    return identities


def _identity_lock_results(
    tmp_path: Path,
) -> tuple[dict[str, dict[str, dict[str, str]]], dict[str, str]]:
    tracker = Tracker(
        run_dir=tmp_path / "consist_runs",
        db_path=str(tmp_path / "identity.duckdb"),
        mounts={"workspace": str(tmp_path)},
    )
    plans = tmp_path / "beam" / "plans.parquet"
    events = tmp_path / "beam" / "events.parquet"
    postprocessed = tmp_path / "beam" / "postprocessed.txt"
    restart_destination = tmp_path / "restart" / "events.parquet"
    restart_postprocessed = tmp_path / "restart" / "postprocessed.txt"
    plans.parent.mkdir(parents=True)
    plans.write_bytes(b"selected plans bytes\n")
    consumed: dict[str, str] = {}

    def beam_run(beam_plans: str) -> None:
        consumed["fresh_beam_run"] = _content_identity(Path(beam_plans))
        events.write_bytes(Path(beam_plans).read_bytes())

    def beam_postprocess(beam_events: str, *, label: str) -> None:
        consumed[label] = _content_identity(Path(beam_events))
        target = (
            restart_postprocessed
            if label == "committed_beam_restart"
            else postprocessed
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(beam_events).read_bytes())

    execution_options = ExecutionOptions(input_binding="paths")
    with cr.scenario("identity-lock", tracker=tracker) as scenario:
        fresh_run = scenario.run(
            fn=beam_run,
            name="beam_run",
            model="beam_run",
            inputs={"beam_plans": str(plans)},
            output_paths={"beam_events": str(events)},
            execution_options=execution_options,
        )
        fresh_postprocess = scenario.run(
            fn=beam_postprocess,
            name="beam_postprocess",
            model="beam_postprocess",
            inputs={"beam_events": consist.ref(fresh_run, key="beam_events")},
            output_paths={"postprocessed": str(postprocessed)},
            execution_options=ExecutionOptions(
                input_binding="paths",
                runtime_kwargs={"label": "fresh_beam_postprocess"},
            ),
        )
        cache_hit = scenario.run(
            fn=beam_postprocess,
            name="beam_postprocess",
            model="beam_postprocess",
            inputs={"beam_events": consist.ref(fresh_run, key="beam_events")},
            output_paths={"postprocessed": str(postprocessed)},
            execution_options=ExecutionOptions(
                input_binding="paths",
                runtime_kwargs={"label": "cache_hit_must_not_execute"},
            ),
        )

        beam_events = tracker.get_artifacts_for_run(fresh_run.run.id).outputs[
            "beam_events"
        ]
        restart_destination.parent.mkdir(parents=True, exist_ok=True)
        restart_destination.write_bytes(
            beam_events.as_path(tracker=tracker).read_bytes()
        )
        committed_restart = scenario.run(
            fn=beam_postprocess,
            name="beam_postprocess_committed_restart",
            model="beam_postprocess",
            config={"restart_boundary": "beam_run_completed"},
            inputs={"beam_events": str(restart_destination)},
            output_paths={"postprocessed": str(restart_postprocessed)},
            execution_options=ExecutionOptions(
                input_binding="paths",
                runtime_kwargs={"label": "committed_beam_restart"},
            ),
        )

    assert fresh_run.cache_hit is False
    assert fresh_postprocess.cache_hit is False
    assert cache_hit.cache_hit is True
    assert "cache_hit_must_not_execute" not in consumed
    identities = {
        "fresh": {
            "beam_run": _input_content_identities(tracker, fresh_run.run.id),
            "beam_postprocess": _input_content_identities(
                tracker, fresh_postprocess.run.id
            ),
        },
        "cache_hit": {
            "beam_postprocess": _input_content_identities(tracker, cache_hit.run.id),
        },
        "committed_beam_restart": {
            "beam_postprocess": _input_content_identities(
                tracker, committed_restart.run.id
            ),
        },
    }
    return identities, consumed


def test_consist_identity_roles_match_the_bytes_consumed_and_restored(tmp_path) -> None:
    identities, consumed = _identity_lock_results(tmp_path)

    assert identities["fresh"]["beam_run"] == {"beam_plans": consumed["fresh_beam_run"]}
    assert identities["fresh"]["beam_postprocess"] == {
        "beam_events": consumed["fresh_beam_postprocess"]
    }
    assert (
        identities["cache_hit"]["beam_postprocess"]
        == identities["fresh"]["beam_postprocess"]
    )
    assert identities["committed_beam_restart"]["beam_postprocess"] == {
        "beam_events": consumed["committed_beam_restart"]
    }
    assert (
        identities["committed_beam_restart"]["beam_postprocess"]
        == identities["fresh"]["beam_postprocess"]
    )
