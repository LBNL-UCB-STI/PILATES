"""Archive completed native Consist child runs into the configured recovery root."""

from __future__ import annotations

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
from pilates.workflows.artifact_keys import ZARR_SKIMS

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


def _is_direct_zarr_snapshot(artifact: Artifact) -> bool:
    """Return whether an output needs PILATES's durable Zarr snapshot."""

    metadata = artifact.meta if isinstance(artifact.meta, dict) else {}
    return artifact.key == ZARR_SKIMS and artifact.driver == "zarr" and bool(
        metadata.get("directory_artifact")
    )


def _snapshot_directory_artifact(
    *, tracker: Tracker, artifact: Artifact, recovery_root: Path
) -> None:
    """Publish a manifest-verified direct-directory output as a recovery copy."""

    metadata = artifact.meta if isinstance(artifact.meta, dict) else {}
    manifest = metadata.get("directory_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError(f"directory artifact {artifact.key!r} has no persisted manifest")
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
    snapshotted_keys = tuple(
        key
        for key, artifact in result.outputs.items()
        if _is_direct_zarr_snapshot(artifact)
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

    for key in snapshotted_keys:
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
    refreshed_outputs = tracker.get_run_outputs(str(result.run.id))
    archived_outputs.update(
        {key: refreshed_outputs[key] for key in snapshotted_keys}
    )
    return RunResult(
        run=result.run,
        outputs={key: archived_outputs[key] for key in result.outputs},
        cache_hit=result.cache_hit,
    )
