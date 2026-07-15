"""Archive-local committed checkpoint state for the BEAM run/postprocess boundary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence
from uuid import uuid4

import yaml

from pilates.workflows.resume import HistoricalOutputRequest

_CHECKPOINT_KEY = "restart_checkpoint"
_SCHEMA_VERSION = 1
_BOUNDARY_ID = "beam_run_completed"
_NEXT_BOUNDARY = "beam_postprocess"
_IN_PROGRESS = "beam_postprocess_in_progress"


@dataclass(frozen=True)
class BeamRunCheckpoint:
    """The only restart authority for a committed completed BEAM run."""

    scope: Mapping[str, int]
    snapshot_ref: str
    recovery_config_fingerprint: str
    producer_run_id: str
    boundary_id: str = _BOUNDARY_ID
    next_boundary: str = _NEXT_BOUNDARY


def beam_recovery_config_fingerprint(
    *,
    scope: Mapping[str, int],
    skim_variant: str,
    output_requests: Sequence[HistoricalOutputRequest],
) -> str:
    """Digest only the resolved BEAM restart contract."""

    payload = {
        "boundary_id": _BOUNDARY_ID,
        "scope": {key: int(scope[key]) for key in sorted(scope)},
        "skim_variant": str(skim_variant),
        "outputs": [
            {"key": request.key, "destination": str(request.destination)}
            for request in sorted(output_requests, key=lambda request: request.key)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _state_path(archive_run_dir: Path) -> Path:
    return archive_run_dir / "run_state.yaml"


def _write_state_atomically(path: Path, state: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(dict(state), stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Archive state must be a mapping: {path}")
    return data


def publish_beam_run_checkpoint(
    *,
    archive_run_dir: Path,
    producer_run_id: str,
    scope: Mapping[str, int],
    snapshot_ref: str,
    skim_variant: str,
    output_requests: Sequence[HistoricalOutputRequest],
) -> BeamRunCheckpoint:
    """Atomically publish a restartable completed BEAM run checkpoint."""

    checkpoint = BeamRunCheckpoint(
        scope=dict(scope),
        snapshot_ref=snapshot_ref,
        recovery_config_fingerprint=beam_recovery_config_fingerprint(
            scope=scope,
            skim_variant=skim_variant,
            output_requests=output_requests,
        ),
        producer_run_id=producer_run_id,
    )
    state_path = _state_path(archive_run_dir)
    state = _load_state(state_path)
    state[_CHECKPOINT_KEY] = {
        "schema_version": _SCHEMA_VERSION,
        "boundary_id": checkpoint.boundary_id,
        "next_boundary": checkpoint.next_boundary,
        "scope": dict(checkpoint.scope),
        "tracker_checkpoint_ref": checkpoint.snapshot_ref,
        "recovery_config_fingerprint": checkpoint.recovery_config_fingerprint,
        "producer_run_id": checkpoint.producer_run_id,
    }
    _write_state_atomically(state_path, state)
    return checkpoint


def read_beam_run_checkpoint(archive_run_dir: Path) -> BeamRunCheckpoint | None:
    """Return the valid restartable checkpoint, never an in-progress marker."""

    payload = _load_state(_state_path(archive_run_dir)).get(_CHECKPOINT_KEY)
    if not isinstance(payload, dict) or payload.get("boundary_id") != _BOUNDARY_ID:
        return None
    if payload.get("schema_version") != _SCHEMA_VERSION:
        return None
    if payload.get("next_boundary") == _IN_PROGRESS:
        return None
    if payload.get("next_boundary") != _NEXT_BOUNDARY:
        return None
    try:
        scope = payload["scope"]
        if not isinstance(scope, dict):
            return None
        return BeamRunCheckpoint(
            scope={key: int(value) for key, value in scope.items()},
            snapshot_ref=str(payload["tracker_checkpoint_ref"]),
            recovery_config_fingerprint=str(payload["recovery_config_fingerprint"]),
            producer_run_id=str(payload["producer_run_id"]),
            boundary_id=str(payload["boundary_id"]),
            next_boundary=str(payload["next_boundary"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def beam_postprocess_is_in_progress(archive_run_dir: Path) -> bool:
    """Whether the checkpoint was consumed by non-restartable postprocess."""

    payload = _load_state(_state_path(archive_run_dir)).get(_CHECKPOINT_KEY)
    return isinstance(payload, dict) and payload.get("next_boundary") == _IN_PROGRESS


def beam_checkpoint_record_present(archive_run_dir: Path) -> bool:
    """Whether archive state names any BEAM checkpoint record, valid or not."""

    return _CHECKPOINT_KEY in _load_state(_state_path(archive_run_dir))


def checkpoint_fingerprint_matches(
    checkpoint: BeamRunCheckpoint,
    *,
    scope: Mapping[str, int],
    skim_variant: str,
    output_requests: Sequence[HistoricalOutputRequest],
) -> bool:
    return checkpoint.recovery_config_fingerprint == beam_recovery_config_fingerprint(
        scope=scope,
        skim_variant=skim_variant,
        output_requests=output_requests,
    )


def mark_beam_postprocess_in_progress(
    archive_run_dir: Path, checkpoint: BeamRunCheckpoint
) -> None:
    """Atomically make a BEAM postprocess boundary explicitly non-restartable."""

    state_path = _state_path(archive_run_dir)
    state = _load_state(state_path)
    state[_CHECKPOINT_KEY] = {
        "schema_version": _SCHEMA_VERSION,
        "boundary_id": checkpoint.boundary_id,
        "next_boundary": _IN_PROGRESS,
        "scope": dict(checkpoint.scope),
        "tracker_checkpoint_ref": checkpoint.snapshot_ref,
        "recovery_config_fingerprint": checkpoint.recovery_config_fingerprint,
        "producer_run_id": checkpoint.producer_run_id,
    }
    _write_state_atomically(state_path, state)


def assert_committed_beam_run(
    *,
    tracker: object,
    checkpoint: BeamRunCheckpoint,
    output_requests: Sequence[HistoricalOutputRequest],
) -> object:
    """Validate the one pinned run without a historical matching query."""

    get_run = getattr(tracker, "get_run", None)
    get_run_outputs = getattr(tracker, "get_run_outputs", None)
    if not callable(get_run) or not callable(get_run_outputs):
        raise RuntimeError("Committed BEAM checkpoint requires direct tracker APIs.")
    run = get_run(checkpoint.producer_run_id)
    if run is None or getattr(run, "status", None) != "completed":
        raise RuntimeError("Committed BEAM checkpoint does not name a completed run.")
    expected_year = checkpoint.scope.get("year")
    if expected_year is not None and getattr(run, "year", None) != expected_year:
        raise RuntimeError("Committed BEAM checkpoint run scope does not match year.")
    if getattr(run, "iteration", None) != checkpoint.scope.get("iteration"):
        raise RuntimeError("Committed BEAM checkpoint run scope does not match iteration.")
    outputs = get_run_outputs(checkpoint.producer_run_id)
    required_keys = {request.key for request in output_requests if request.required}
    if not required_keys or not required_keys.issubset(outputs):
        raise RuntimeError("Committed BEAM checkpoint is missing selected output links.")
    return run


def snapshot_and_publish_beam_run_checkpoint(
    *,
    tracker: object,
    open_snapshot: object,
    archive_run_dir: Path,
    producer_run_id: str,
    scope: Mapping[str, int],
    skim_variant: str,
    output_requests: Sequence[HistoricalOutputRequest],
) -> BeamRunCheckpoint:
    """Pin a BEAM run in one new archive-local snapshot before publication."""

    snapshot_db = getattr(tracker, "snapshot_db", None)
    if not callable(snapshot_db) or not callable(open_snapshot):
        raise RuntimeError("Committed BEAM checkpoint requires tracker snapshot APIs.")
    snapshot_ref = Path(".consist") / "restart" / "checkpoints" / str(uuid4()) / "tracker.duckdb"
    snapshot_path = archive_run_dir / snapshot_ref
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_db(snapshot_path, checkpoint=True)
    if not snapshot_path.is_file():
        raise RuntimeError("Committed BEAM checkpoint snapshot was not created.")
    snapshot_tracker = open_snapshot(snapshot_path)
    assert_committed_beam_run(
        tracker=snapshot_tracker,
        checkpoint=BeamRunCheckpoint(
            scope=dict(scope),
            snapshot_ref=str(snapshot_ref),
            recovery_config_fingerprint=beam_recovery_config_fingerprint(
                scope=scope,
                skim_variant=skim_variant,
                output_requests=output_requests,
            ),
            producer_run_id=producer_run_id,
        ),
        output_requests=output_requests,
    )
    return publish_beam_run_checkpoint(
        archive_run_dir=archive_run_dir,
        producer_run_id=producer_run_id,
        scope=scope,
        snapshot_ref=str(snapshot_ref),
        skim_variant=skim_variant,
        output_requests=output_requests,
    )


def verify_archive_visible_recovery_bytes(
    *,
    tracker: object,
    archive_run_dir: Path,
    producer_run_id: str,
    output_requests: Sequence[HistoricalOutputRequest],
) -> None:
    """Prove each declared file can hydrate from the original archive namespace."""

    hydrate = getattr(tracker, "hydrate_run_outputs_to_destinations", None)
    if not callable(hydrate):
        raise RuntimeError("Committed BEAM checkpoint requires exact hydration APIs.")
    verification_root = archive_run_dir / ".consist" / "restart" / "verification" / str(uuid4())
    destinations = {
        request.key: verification_root / str(index)
        for index, request in enumerate(output_requests)
    }
    try:
        result = hydrate(
            producer_run_id,
            destinations_by_key=destinations,
            source_root=archive_run_dir,
            preserve_existing=False,
            on_missing="warn",
            db_fallback="never",
        )
        if result.source_run_id != producer_run_id:
            raise RuntimeError("Archive verification hydrated a different BEAM run.")
        for request in output_requests:
            item = result.get(request.key)
            if (
                item is None
                or item.status != "materialized_from_filesystem"
                or not item.resolvable
                or item.path != destinations[request.key]
                or not item.path.is_file()
            ):
                raise RuntimeError(
                    f"Committed BEAM checkpoint has no archive-visible bytes for {request.key}."
                )
    finally:
        if verification_root.exists():
            import shutil

            shutil.rmtree(verification_root)
