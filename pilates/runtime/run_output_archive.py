"""Archive completed native Consist child runs into the configured recovery root."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

from consist import RunResult
from consist.core.directory_artifacts import (
    materialize_directory_tree,
    validate_directory_manifest,
)
from consist.core.tracker import Tracker
from consist.models.artifact import Artifact

from pilates.runtime.archive_paths import archive_roots

_ARCHIVE_ENABLE_ENV = "PILATES_ENABLE_ARCHIVE_COPY"


def _archive_enabled() -> bool:
    return os.environ.get(_ARCHIVE_ENABLE_ENV, "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _configured_child_recovery_root(run_id: str) -> str | None:
    """Return the enabled archive location isolated to one completed child run."""

    roots = archive_roots()
    if roots is None:
        return None
    local_root, archive_root = roots
    if os.path.normcase(os.path.normpath(local_root)) == os.path.normcase(
        os.path.normpath(archive_root)
    ):
        return None
    return str(Path(archive_root) / "consist-recovery" / run_id)


def _output_set_represents_directory_artifact(
    *, tracker: Tracker, artifact: Artifact, output_set: Artifact
) -> bool:
    """Return whether one OutputSet manifest exactly represents a directory tree."""

    parent_uri = output_set.container_uri.rstrip("/")
    if not artifact.container_uri.startswith(f"{parent_uri}/"):
        return False
    metadata = output_set.meta if isinstance(output_set.meta, dict) else {}
    manifest_id = metadata.get("manifest_artifact_id")
    if not isinstance(manifest_id, str) or not manifest_id:
        return False
    manifest_artifact = tracker.get_artifact(manifest_id)
    if manifest_artifact is None:
        return False
    try:
        manifest_path = Path(tracker.resolve_uri(manifest_artifact.container_uri))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, Mapping):
        return False
    members = manifest.get("members")
    directory_metadata = artifact.meta if isinstance(artifact.meta, dict) else {}
    directory_manifest = directory_metadata.get("directory_manifest")
    if not isinstance(members, list) or not isinstance(directory_manifest, Mapping):
        return False

    prefix = artifact.container_uri[len(parent_uri) + 1 :].rstrip("/")
    expected_entries: list[dict[str, object]] = []
    expected_directories: set[str] = set()
    member_prefix = f"{prefix}/"
    for member in members:
        if not isinstance(member, Mapping):
            return False
        relative_path = member.get("relative_path")
        content_hash = member.get("content_hash")
        size_bytes = member.get("size_bytes")
        if (
            not isinstance(relative_path, str)
            or not relative_path.startswith(member_prefix)
            or not isinstance(content_hash, str)
            or type(size_bytes) is not int
        ):
            continue
        child_path = relative_path[len(member_prefix) :]
        parent = Path(child_path).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
        expected_entries.append(
            {
                "kind": "file",
                "path": child_path,
                "sha256": content_hash,
                "size": size_bytes,
            }
        )
    expected_entries.extend(
        {"kind": "directory", "path": path} for path in expected_directories
    )
    actual_entries = directory_manifest.get("entries")
    return actual_entries == sorted(
        expected_entries, key=lambda entry: (str(entry["path"]), str(entry["kind"]))
    )


def _is_direct_zarr_snapshot(
    *, tracker: Tracker, artifact: Artifact, output_sets: tuple[Artifact, ...]
) -> bool:
    """Return whether a direct Zarr output needs PILATES's recovery snapshot.

    An OutputSet records files, never a directory artifact's tree manifest.  A
    direct Zarr can therefore sit beneath an OutputSet root while still being
    deliberately excluded from that set.  Snapshot it only when no selected
    OutputSet manifest exactly represents its immutable tree.
    """

    metadata = artifact.meta if isinstance(artifact.meta, dict) else {}
    is_direct_zarr = artifact.driver == "zarr" and bool(
        metadata.get("directory_artifact")
    )
    return is_direct_zarr and not any(
        _output_set_represents_directory_artifact(
            tracker=tracker, artifact=artifact, output_set=output_set
        )
        for output_set in output_sets
    )


def _snapshot_directory_artifact(
    *, tracker: Tracker, artifact: Artifact, recovery_root: Path
) -> None:
    """Publish a manifest-verified direct-directory output as a recovery copy."""

    metadata = artifact.meta if isinstance(artifact.meta, dict) else {}
    manifest = metadata.get("directory_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError(
            f"directory artifact {artifact.key!r} has no persisted manifest"
        )
    normalized_manifest = validate_directory_manifest(manifest)
    if artifact.hash != normalized_manifest["tree_hash"]:
        raise ValueError(
            f"directory artifact {artifact.key!r} manifest does not match artifact identity"
        )
    relative_path = tracker.fs.get_remappable_relative_path(artifact.container_uri)
    if relative_path is None:
        raise ValueError(
            f"Artifact {artifact.key!r} does not have a rematerializable URI layout."
        )

    source = Path(tracker.resolve_uri(artifact.container_uri))
    destination = recovery_root / relative_path
    published = materialize_directory_tree(
        source, destination, normalized_manifest, preserve_existing=True
    )
    try:
        tracker.set_artifact_recovery_roots(artifact, [recovery_root], append=True)
    except Exception:
        if published and destination.exists():
            shutil.rmtree(destination)
        raise


def archive_completed_run(*, tracker: Tracker | None, result: RunResult) -> RunResult:
    """Archive enabled child-run outputs and return their refreshed artifacts."""

    if not _archive_enabled():
        return result
    archive_root = _configured_child_recovery_root(str(result.run.id))
    if archive_root is None:
        return result
    if tracker is None:
        raise RuntimeError("Cannot archive completed run without a Consist tracker.")
    recovery_root = Path(archive_root).resolve()
    output_sets = tuple(
        artifact
        for artifact in result.outputs.values()
        if artifact.driver == "artifact_set"
    )
    output_set_roots = tuple(
        artifact.container_uri.rstrip("/") for artifact in output_sets
    )
    snapshotted_keys = tuple(
        key
        for key, artifact in result.outputs.items()
        if _is_direct_zarr_snapshot(
            tracker=tracker, artifact=artifact, output_sets=output_sets
        )
    )
    if not snapshotted_keys:
        archived = tracker.archive_run_outputs(
            str(result.run.id), archive_root, mode="copy"
        )
        return RunResult(
            run=result.run,
            outputs=dict(archived.outputs),
            cache_hit=result.cache_hit,
        )

    # An excluded direct Zarr can live under an OutputSet root.  Archive the
    # set's file members first; otherwise the Zarr snapshot makes the set's
    # exact destination validation see an unexpected directory.
    deferred_snapshot_keys = tuple(
        key
        for key in snapshotted_keys
        if any(
            result.outputs[key].container_uri.startswith(f"{root}/")
            for root in output_set_roots
        )
    )
    immediate_snapshot_keys = tuple(
        key for key in snapshotted_keys if key not in deferred_snapshot_keys
    )
    for key in immediate_snapshot_keys:
        _snapshot_directory_artifact(
            tracker=tracker,
            artifact=result.outputs[key],
            recovery_root=recovery_root,
        )

    remaining_keys = tuple(key for key in result.outputs if key not in snapshotted_keys)
    archived_outputs: dict[str, Artifact] = {}
    if remaining_keys:
        archived = tracker.archive_run_outputs(
            str(result.run.id), archive_root, keys=remaining_keys, mode="copy"
        )
        archived_outputs.update(archived.outputs)
    for key in deferred_snapshot_keys:
        _snapshot_directory_artifact(
            tracker=tracker,
            artifact=result.outputs[key],
            recovery_root=recovery_root,
        )
    refreshed_outputs = tracker.get_run_outputs(str(result.run.id))
    archived_outputs.update({key: refreshed_outputs[key] for key in snapshotted_keys})
    return RunResult(
        run=result.run,
        outputs={key: archived_outputs[key] for key in result.outputs},
        cache_hit=result.cache_hit,
    )
