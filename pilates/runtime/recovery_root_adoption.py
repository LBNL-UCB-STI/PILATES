from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pilates.runtime.archive_paths import archive_roots
from pilates.utils import consist_runtime as cr
from pilates.workflows.coupler_namespace import canonical_artifact_key_from_raw_key

logger = logging.getLogger(__name__)

CopySignature = tuple[int, int, bool]
CopiedArtifact = tuple[str, str, str, bool, Optional[CopySignature]]

_PHASE2_RECOVERY_ROOT_FAMILIES = {
    "usim_input_archive",
    "usim_population_source_h5",
}

_adoption_lock = threading.Lock()
_last_copied_details: Dict[str, CopiedArtifact] = {}
_last_registration_signature: Dict[str, CopySignature] = {}


def reset_recovery_root_adoption_state() -> None:
    with _adoption_lock:
        _last_copied_details.clear()
        _last_registration_signature.clear()


def record_archive_copy(
    *,
    key: str,
    src: str,
    dest: str,
    is_dir: bool,
    signature: Optional[CopySignature],
) -> None:
    with _adoption_lock:
        _last_copied_details[dest] = (key, src, dest, is_dir, signature)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase2_recovery_root_family(key: str) -> Optional[str]:
    family = canonical_artifact_key_from_raw_key(key)
    if family.startswith("usim_input_archive_"):
        family = "usim_input_archive"
    if family in _PHASE2_RECOVERY_ROOT_FAMILIES:
        return family
    return None


def _content_hash_for_recovery_copy(artifact: Any, dest: str) -> str:
    tracker = cr.current_tracker()
    if (
        tracker is not None
        and getattr(getattr(tracker, "identity", None), "hashing_strategy", None)
        == "full"
    ):
        artifact_hash = getattr(artifact, "hash", None)
        if artifact_hash:
            return str(artifact_hash)
    return _sha256_file(Path(dest))


def adopt_recovery_root_copy(
    *,
    key: str,
    src: str,
    dest: str,
    is_dir: bool,
    signature: Optional[CopySignature],
    find_artifact: Callable[[str, str], Optional[Any]],
    emit_lifecycle_event: Callable[..., None],
) -> int:
    family = _phase2_recovery_root_family(key)
    if family is None:
        return 0
    if is_dir:
        logger.debug(
            "[Archive] Skipping phase 2 recovery-root adoption for directory copy: %s (key=%s)",
            dest,
            key,
        )
        return 0
    if not os.path.isfile(dest):
        logger.warning(
            "[Archive] Skipping phase 2 recovery-root adoption for non-file copy: %s (key=%s)",
            dest,
            key,
        )
        return 0

    artifact = find_artifact(key, src)
    if artifact is None:
        logger.debug(
            "[Archive] No logged output artifact matched phase 2 recovery-root adoption for key=%s src=%s",
            key,
            src,
        )
        return 0

    tracker = cr.current_tracker()
    if tracker is None:
        logger.debug(
            "[Archive] No active tracker available for phase 2 recovery-root adoption (key=%s dest=%s)",
            key,
            dest,
        )
        return 0

    register_copies = getattr(tracker, "register_run_output_recovery_copies", None)
    if not callable(register_copies):
        logger.warning(
            "[Archive] Tracker does not expose register_run_output_recovery_copies; skipping phase 2 recovery-root adoption for key=%s",
            key,
        )
        return 0

    run_id = cr.current_run_id()
    if not run_id:
        logger.debug(
            "[Archive] No active run id available for phase 2 recovery-root adoption (key=%s dest=%s)",
            key,
            dest,
        )
        return 0

    roots = archive_roots()
    if roots is None:
        return 0
    _local_root, archive_root = roots

    with _adoption_lock:
        if (
            signature is not None
            and _last_registration_signature.get(dest) == signature
        ):
            logger.debug(
                "[Archive] Skipping duplicate phase 2 recovery-root adoption "
                "(already registered): %s (key=%s)",
                dest,
                key,
            )
            return 0

    content_hash = _content_hash_for_recovery_copy(artifact, dest)
    result = register_copies(
        str(run_id),
        str(archive_root),
        verify=True,
        append=True,
        content_hashes={str(key): content_hash},
    )
    registered = getattr(result, "registered", {}) or {}
    blocked = getattr(result, "blocked", {}) or {}
    if not registered:
        if blocked:
            logger.info(
                "[Archive] Phase 2 recovery-root adoption blocked for key=%s: %s",
                key,
                getattr(result, "summary", blocked),
            )
        return 0

    registered_count = len(registered)
    with _adoption_lock:
        if signature is not None:
            _last_registration_signature[dest] = signature
    logger.info(
        "[Archive] Adopted phase 2 recovery root for key=%s run_id=%s root=%s registered=%d",
        key,
        run_id,
        archive_root,
        registered_count,
    )
    emit_lifecycle_event(
        "archive_recovery_root_registered",
        key=key,
        src=src,
        path=src,
        dest=dest,
        recovery_root=str(archive_root),
        run_id=str(run_id),
        artifact_family=family,
        storage_event="local_to_scratch_recovery_root_adopted",
        local_to_scratch_recovery_roots_written=registered_count,
    )
    return registered_count


def adopt_pending_recovery_root_copies(
    *,
    find_artifact: Callable[[str, str], Optional[Any]],
    emit_lifecycle_event: Callable[..., None],
) -> int:
    registered = 0
    with _adoption_lock:
        copied_details = list(_last_copied_details.items())
    for dest, (key, src, recorded_dest, is_dir, signature) in copied_details:
        if dest != recorded_dest:
            continue
        registered += adopt_recovery_root_copy(
            key=key,
            src=src,
            dest=dest,
            is_dir=is_dir,
            signature=signature,
            find_artifact=find_artifact,
            emit_lifecycle_event=emit_lifecycle_event,
        )
    return registered
