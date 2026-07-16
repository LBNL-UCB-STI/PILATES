"""Archive-local committed checkpoint state for the BEAM run/postprocess boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

import consist
import yaml

from pilates.workflows.resume import HistoricalOutputRequest

_CHECKPOINT_KEY = "restart_checkpoint"
_SCHEMA_VERSION = 2
_BOUNDARY_ID = "beam_run_completed"
_NEXT_BOUNDARY = "beam_postprocess"
_IN_PROGRESS = "beam_postprocess_in_progress"


@dataclass(frozen=True)
class PinnedClosureMember:
    """One immutable, exact-destination input to the successor operation."""

    member_id: str
    role: str
    producer_run_id: str
    output_key: str
    artifact_identity: str
    artifact_kind: str
    driver: str | None
    destination: Path
    required: bool


@dataclass(frozen=True)
class BeamRunCheckpoint:
    """The only restart authority for a committed completed BEAM run."""

    scope: Mapping[str, int]
    snapshot_ref: str
    recovery_config_fingerprint: str
    producer_run_id: str
    closure_members: tuple[PinnedClosureMember, ...] = ()
    boundary_id: str = _BOUNDARY_ID
    next_boundary: str = _NEXT_BOUNDARY


class PinnedClosureTracker(Protocol):
    """Direct pinned-snapshot APIs used by the closure executor."""

    def get_run(self, run_id: str) -> Any: ...

    def get_run_outputs(self, run_id: str) -> Mapping[str, Any]: ...

    def hydrate_run_outputs_to_destinations(
        self,
        run_id: str,
        *,
        destinations_by_key: Mapping[str, Path],
        source_root: Path,
        preserve_existing: bool,
        on_missing: str,
        db_fallback: str,
    ) -> Any: ...


def _pinned_artifact_kind(artifact: Any) -> str:
    meta = artifact.meta
    if not isinstance(meta, dict):
        raise RuntimeError(
            f"Pinned closure artifact {artifact.key!r} has invalid metadata."
        )
    if meta.get("directory_artifact") is True:
        return "directory"
    if meta.get("file_bundle_artifact") is True:
        return "file_bundle"
    return "file"


def _preflight_pinned_closure(
    members: Sequence[PinnedClosureMember],
    *,
    require_clean_destinations: bool = True,
) -> tuple[PinnedClosureMember, ...]:
    normalized = tuple(members)
    if not normalized:
        raise RuntimeError("Pinned successor input closure is empty.")

    member_ids: set[str] = set()
    producer_outputs: set[tuple[str, str]] = set()
    destinations: list[Path] = []
    for member in normalized:
        if not member.member_id or member.member_id in member_ids:
            raise RuntimeError(
                f"Pinned closure has a duplicate or empty member_id={member.member_id!r}."
            )
        member_ids.add(member.member_id)

        producer_output = (member.producer_run_id, member.output_key)
        if not all(producer_output) or producer_output in producer_outputs:
            raise RuntimeError(
                "Pinned closure has a duplicate or empty producer/output pair "
                f"{producer_output!r}."
            )
        producer_outputs.add(producer_output)

        if not member.artifact_identity:
            raise RuntimeError(
                f"Pinned closure member {member.member_id!r} has no artifact identity."
            )
        destination = member.destination.expanduser().resolve()
        if destination != member.destination:
            raise RuntimeError(
                f"Pinned closure destination is not exact and absolute: {member.destination}."
            )
        if require_clean_destinations and (
            destination.exists() or destination.is_symlink()
        ):
            raise RuntimeError(
                f"Pinned closure destination already exists: {destination}."
            )
        destinations.append(destination)

    for index, destination in enumerate(destinations):
        for other in destinations[index + 1 :]:
            if destination == other:
                raise RuntimeError(
                    f"Pinned closure has a duplicate destination: {destination}."
                )
            if destination.is_relative_to(other) or other.is_relative_to(destination):
                raise RuntimeError(
                    "Pinned closure has nested destinations: "
                    f"{destination} and {other}."
                )
    return normalized


def _validate_pinned_artifact(
    member: PinnedClosureMember,
    artifact: Any,
) -> None:
    if artifact.hash != member.artifact_identity:
        raise RuntimeError(
            f"Pinned closure artifact identity mismatch for {member.output_key}."
        )
    artifact_kind = _pinned_artifact_kind(artifact)
    if artifact_kind != member.artifact_kind:
        raise RuntimeError(
            f"Pinned closure artifact kind mismatch for {member.output_key}: "
            f"expected {member.artifact_kind}, found {artifact_kind}."
        )
    if artifact.driver != member.driver:
        raise RuntimeError(
            f"Pinned closure artifact driver mismatch for {member.output_key}: "
            f"expected {member.driver}, found {artifact.driver}."
        )
    if member.artifact_kind == "directory" and member.driver != "zarr":
        raise RuntimeError(
            f"Pinned closure directory {member.output_key} is not manifest-backed Zarr."
        )
    if member.driver == "zarr" and member.artifact_kind != "directory":
        raise RuntimeError(
            f"Pinned closure Zarr {member.output_key} is not manifest-backed Zarr."
        )
    if member.artifact_kind not in {"file", "directory"}:
        raise RuntimeError(
            f"Pinned closure artifact kind is unsupported for {member.output_key}."
        )


def _remove_hydration_destinations(members: Sequence[PinnedClosureMember]) -> None:
    for member in reversed(tuple(members)):
        destination = member.destination
        if destination.is_symlink() or destination.is_file():
            destination.unlink(missing_ok=True)
        elif destination.is_dir():
            shutil.rmtree(destination)


def hydrate_pinned_closure(
    *,
    tracker: PinnedClosureTracker,
    source_root: Path,
    members: Sequence[PinnedClosureMember],
) -> dict[str, Any]:
    """Strictly validate and hydrate one pinned multi-producer closure."""

    closure = _preflight_pinned_closure(members)
    members_by_run: dict[str, list[PinnedClosureMember]] = {}
    artifacts_by_member: dict[str, Any] = {}
    for member in closure:
        members_by_run.setdefault(member.producer_run_id, []).append(member)

    for run_id, run_members in members_by_run.items():
        run = tracker.get_run(run_id)
        if run is None or run.status != "completed":
            raise RuntimeError(
                f"Pinned closure producer run is not completed: {run_id}."
            )
        outputs = tracker.get_run_outputs(run_id)
        for member in run_members:
            artifact = outputs.get(member.output_key)
            if artifact is None:
                raise RuntimeError(
                    f"Pinned closure output link is missing: {member.output_key}."
                )
            _validate_pinned_artifact(member, artifact)
            artifacts_by_member[member.member_id] = artifact

    restored: dict[str, Any] = {}
    hydration_started = False
    try:
        for run_id, run_members in members_by_run.items():
            hydration_started = True
            result = tracker.hydrate_run_outputs_to_destinations(
                run_id,
                destinations_by_key={
                    member.output_key: member.destination for member in run_members
                },
                source_root=source_root,
                preserve_existing=False,
                on_missing="warn",
                db_fallback="never",
            )
            if result.source_run_id != run_id:
                raise RuntimeError(
                    f"Pinned closure hydration returned a different run for {run_id}."
                )
            for member in run_members:
                item = result.get(member.output_key)
                if item is None:
                    raise RuntimeError(
                        f"Pinned closure hydration omitted {member.output_key}."
                    )
                _validate_pinned_artifact(member, item.artifact)
                if item.artifact_kind != member.artifact_kind:
                    raise RuntimeError(
                        f"Pinned closure hydration changed artifact kind for {member.output_key}."
                    )
                if member.artifact_kind == "directory":
                    verified = is_verified_hydrated_zarr_directory(
                        item, destination=member.destination
                    )
                else:
                    verified = is_verified_hydrated_recovery_output(
                        item, destination=member.destination
                    )
                if not verified:
                    raise RuntimeError(
                        f"Pinned closure hydration did not verify {member.output_key}."
                    )
                if item.artifact is not artifacts_by_member[member.member_id]:
                    _validate_pinned_artifact(member, item.artifact)
                restored[member.member_id] = item
    except Exception:
        if hydration_started:
            _remove_hydration_destinations(closure)
        raise
    return restored


def validate_pinned_closure_snapshot(
    *,
    tracker: PinnedClosureTracker,
    members: Sequence[PinnedClosureMember],
    scope: Mapping[str, int],
) -> None:
    """Validate every closure link directly in one pinned tracker snapshot."""

    closure = _preflight_pinned_closure(
        members,
        require_clean_destinations=False,
    )
    members_by_run: dict[str, list[PinnedClosureMember]] = {}
    for member in closure:
        members_by_run.setdefault(member.producer_run_id, []).append(member)
    for run_id, run_members in members_by_run.items():
        run = tracker.get_run(run_id)
        if run is None or run.status != "completed":
            raise RuntimeError(
                f"Pinned closure producer run is not completed: {run_id}."
            )
        expected_run_year = scope.get("forecast_year", scope.get("year"))
        if run.year != expected_run_year or run.iteration != scope.get("iteration"):
            raise RuntimeError(
                f"Pinned closure producer run scope does not match: {run_id}."
            )
        outputs = tracker.get_run_outputs(run_id)
        for member in run_members:
            artifact = outputs.get(member.output_key)
            if artifact is None:
                raise RuntimeError(
                    f"Pinned closure output link is missing: {member.output_key}."
                )
            _validate_pinned_artifact(member, artifact)


def verify_archive_visible_pinned_closure_bytes(
    *,
    tracker: PinnedClosureTracker,
    archive_run_dir: Path,
    members: Sequence[PinnedClosureMember],
) -> None:
    """Prove every pinned closure member is hydratable from the archive root."""

    closure = _preflight_pinned_closure(
        members,
        require_clean_destinations=False,
    )
    verification_root = (
        archive_run_dir / ".consist" / "restart" / "verification" / str(uuid4())
    )
    verification_members = tuple(
        replace(
            member,
            destination=verification_root
            / f"{index}{'.zarr' if member.artifact_kind == 'directory' else ''}",
        )
        for index, member in enumerate(closure)
    )
    try:
        hydrate_pinned_closure(
            tracker=tracker,
            source_root=archive_run_dir,
            members=verification_members,
        )
    finally:
        if verification_root.exists():
            shutil.rmtree(verification_root)


def beam_recovery_config_fingerprint(
    *,
    scope: Mapping[str, int],
    skim_variant: str,
    output_requests: Sequence[HistoricalOutputRequest],
    closure_members: Sequence[PinnedClosureMember] = (),
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
        "successor_input_closure": [
            {
                "member_id": member.member_id,
                "role": member.role,
                "producer_run_id": member.producer_run_id,
                "output_key": member.output_key,
                "artifact_identity": member.artifact_identity,
                "artifact_kind": member.artifact_kind,
                "driver": member.driver,
                "destination": str(member.destination),
                "required": member.required,
            }
            for member in sorted(closure_members, key=lambda member: member.member_id)
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
    closure_members: Sequence[PinnedClosureMember] = (),
) -> BeamRunCheckpoint:
    """Atomically publish a restartable completed BEAM run checkpoint."""

    checkpoint = BeamRunCheckpoint(
        scope=dict(scope),
        snapshot_ref=snapshot_ref,
        recovery_config_fingerprint=beam_recovery_config_fingerprint(
            scope=scope,
            skim_variant=skim_variant,
            output_requests=output_requests,
            closure_members=closure_members,
        ),
        producer_run_id=producer_run_id,
        closure_members=tuple(closure_members),
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
        "successor_input_closure": [
            {
                "member_id": member.member_id,
                "role": member.role,
                "producer_run_id": member.producer_run_id,
                "output_key": member.output_key,
                "artifact_identity": member.artifact_identity,
                "artifact_kind": member.artifact_kind,
                "driver": member.driver,
                "destination": str(member.destination),
                "required": member.required,
            }
            for member in checkpoint.closure_members
        ],
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
        closure_payload = payload.get("successor_input_closure", [])
        if not isinstance(closure_payload, list):
            return None
        closure_members = tuple(
            PinnedClosureMember(
                member_id=str(member["member_id"]),
                role=str(member["role"]),
                producer_run_id=str(member["producer_run_id"]),
                output_key=str(member["output_key"]),
                artifact_identity=str(member["artifact_identity"]),
                artifact_kind=str(member["artifact_kind"]),
                driver=(str(member["driver"]) if member.get("driver") else None),
                destination=Path(str(member["destination"])),
                required=bool(member["required"]),
            )
            for member in closure_payload
            if isinstance(member, dict)
        )
        if len(closure_members) != len(closure_payload):
            return None
        return BeamRunCheckpoint(
            scope={key: int(value) for key, value in scope.items()},
            snapshot_ref=str(payload["tracker_checkpoint_ref"]),
            recovery_config_fingerprint=str(payload["recovery_config_fingerprint"]),
            producer_run_id=str(payload["producer_run_id"]),
            closure_members=closure_members,
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
    closure_members: Sequence[PinnedClosureMember] = (),
) -> bool:
    return checkpoint.recovery_config_fingerprint == beam_recovery_config_fingerprint(
        scope=scope,
        skim_variant=skim_variant,
        output_requests=output_requests,
        closure_members=closure_members,
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
        "successor_input_closure": [
            {
                "member_id": member.member_id,
                "role": member.role,
                "producer_run_id": member.producer_run_id,
                "output_key": member.output_key,
                "artifact_identity": member.artifact_identity,
                "artifact_kind": member.artifact_kind,
                "driver": member.driver,
                "destination": str(member.destination),
                "required": member.required,
            }
            for member in checkpoint.closure_members
        ],
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
        raise RuntimeError(
            "Committed BEAM checkpoint run scope does not match iteration."
        )
    outputs = get_run_outputs(checkpoint.producer_run_id)
    required_keys = {request.key for request in output_requests if request.required}
    if not required_keys or not required_keys.issubset(outputs):
        raise RuntimeError(
            "Committed BEAM checkpoint is missing selected output links."
        )
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
    closure_members: Sequence[PinnedClosureMember] = (),
) -> BeamRunCheckpoint:
    """Pin a BEAM run in one new archive-local snapshot before publication."""

    snapshot_db = getattr(tracker, "snapshot_db", None)
    if not callable(snapshot_db) or not callable(open_snapshot):
        raise RuntimeError("Committed BEAM checkpoint requires tracker snapshot APIs.")
    snapshot_ref = (
        Path(".consist") / "restart" / "checkpoints" / str(uuid4()) / "tracker.duckdb"
    )
    snapshot_path = archive_run_dir / snapshot_ref
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_db(snapshot_path, checkpoint=True)
    if not snapshot_path.is_file():
        raise RuntimeError("Committed BEAM checkpoint snapshot was not created.")
    snapshot_tracker = open_snapshot(snapshot_path)
    if closure_members:
        validate_pinned_closure_snapshot(
            tracker=snapshot_tracker,
            members=closure_members,
            scope=scope,
        )
    else:
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
        closure_members=closure_members,
    )


def verify_archive_visible_recovery_bytes(
    *,
    tracker: object,
    archive_run_dir: Path,
    producer_run_id: str,
    output_requests: Sequence[HistoricalOutputRequest],
) -> None:
    """Prove each declared output can hydrate from the original archive namespace."""

    hydrate = getattr(tracker, "hydrate_run_outputs_to_destinations", None)
    if not callable(hydrate):
        raise RuntimeError("Committed BEAM checkpoint requires exact hydration APIs.")
    verification_root = (
        archive_run_dir / ".consist" / "restart" / "verification" / str(uuid4())
    )
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
            if item is None or not is_verified_hydrated_recovery_output(
                item, destination=destinations[request.key]
            ):
                raise RuntimeError(
                    f"Committed BEAM checkpoint has no archive-visible bytes for {request.key}."
                )
    finally:
        if verification_root.exists():
            import shutil

            shutil.rmtree(verification_root)


def is_verified_hydrated_zarr_directory(item: Any, *, destination: Path) -> bool:
    """Return whether Consist exactly restored a native immutable Zarr directory."""

    path = item.path
    return (
        item.status == "materialized_directory_from_filesystem"
        and item.artifact_kind == "directory"
        and item.resolvable
        and path == destination
        and path is not None
        and path.is_dir()
        and consist.is_zarr_artifact(item.artifact)
    )


def is_verified_hydrated_recovery_output(item: Any, *, destination: Path) -> bool:
    """Accept regular files or Consist's strict immutable Zarr directories only."""

    path = item.path
    return (
        item.status == "materialized_from_filesystem"
        and item.resolvable
        and path == destination
        and path is not None
        and path.is_file()
    ) or is_verified_hydrated_zarr_directory(item, destination=destination)
