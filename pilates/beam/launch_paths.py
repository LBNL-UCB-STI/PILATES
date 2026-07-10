"""Execution-exact BEAM launch-path resolution.

The functions here resolve the final staged HOCON values that PILATES mounts at
``/app/input``. They intentionally model PILATES/BEAM launch behavior rather
than creating a second portable Consist configuration identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from zipfile import BadZipFile, ZipFile

from pilates.beam.config_hocon import (
    beam_config_env_overrides,
    beam_primary_config_path,
    resolve_beam_config_value,
    update_staged_beam_config_value,
)


class BeamLaunchPathError(RuntimeError):
    """Raised when a final BEAM HOCON path cannot be mounted safely."""


@dataclass(frozen=True)
class BeamLaunchPathReference:
    """One final HOCON path and its mounted execution location."""

    config_key: str
    raw_value: str
    canonical_value: str
    configured_path: Path
    execution_path: Path
    physical_target_path: Path
    container_path: str


@dataclass(frozen=True)
class R5NetworkLaunchReference:
    """Raw R5 members selected from final HOCON after cache invalidation."""

    network_directory: BeamLaunchPathReference
    secondary_network_directory: Optional[BeamLaunchPathReference]
    selected_osm_path: Path
    selected_osm_physical_target_path: Path
    selected_osm_container_path: str
    gtfs_paths: tuple[Path, ...]
    ignored_osm_paths: tuple[Path, ...]


def configure_staged_linkstats_reference(
    *,
    settings: Any,
    workspace: Any,
    staged_path: Path,
) -> BeamLaunchPathReference:
    """Point final BEAM HOCON at a staged warm-start file.

    The persisted expression remains relative to ``beam.inputDirectory``:
    Consist resolves it to the staged host path during canonicalization, while
    BEAM resolves the same expression under its ``/app/input`` mount.
    """

    mutable_root = _mutable_input_root(workspace)
    staged_path = Path(staged_path)
    physical_target_path = _physical_target(
        staged_path,
        mutable_root=mutable_root,
        config_key="beam.warmStart.initialLinkstatsFilePath",
    )
    if not staged_path.is_file():
        raise BeamLaunchPathError(
            "Staged BEAM warm-start linkstats must be one regular file: "
            f"{staged_path}"
        )

    config_root = beam_primary_config_path(settings, workspace=workspace).parent.resolve()
    try:
        relative_path = staged_path.resolve().relative_to(config_root)
    except ValueError as exc:
        raise BeamLaunchPathError(
            "Staged BEAM warm-start linkstats must be under the region input root: "
            f"{staged_path}"
        ) from exc

    config_path = beam_primary_config_path(settings, workspace=workspace)
    env_overrides = beam_config_env_overrides(settings, workspace=workspace)
    hocon_value = f'${{beam.inputDirectory}}"/{relative_path.as_posix()}"'
    update_staged_beam_config_value(
        config_path,
        key="beam.warmStart.initialLinkstatsFilePath",
        value=hocon_value,
        env_overrides=env_overrides,
    )
    resolved_host_path = Path(
        str(
            resolve_beam_config_value(
                config_path,
                key="beam.warmStart.initialLinkstatsFilePath",
                env_overrides=env_overrides,
            )
        )
    )
    if resolved_host_path.resolve() != staged_path.resolve():
        raise BeamLaunchPathError(
            "Final BEAM HOCON warm-start linkstats path does not resolve to the staged "
            f"file: {resolved_host_path} != {staged_path}"
        )
    return BeamLaunchPathReference(
        config_key="beam.warmStart.initialLinkstatsFilePath",
        raw_value=hocon_value,
        canonical_value=str(physical_target_path),
        configured_path=staged_path,
        execution_path=staged_path,
        physical_target_path=physical_target_path,
        container_path=_container_path(staged_path, mutable_root=mutable_root),
    )


def resolve_r5_network_reference(*, settings: Any, workspace: Any) -> R5NetworkLaunchReference:
    """Resolve R5's raw-directory selection from final staged HOCON.

    PILATES invalidates ``network.dat`` before using this reference, so the
    selection follows R5's raw-build branch: lexically first ``.pbf``/``.vex``
    file and every top-level GTFS ZIP containing ``stop_times.txt``.
    """

    network_directory = _resolve_hocon_path_reference(
        settings=settings,
        workspace=workspace,
        config_key="beam.routing.r5.directory",
        required=True,
    )

    secondary_network_directory = _resolve_hocon_path_reference(
        settings=settings,
        workspace=workspace,
        config_key="beam.routing.r5.directory2",
        required=False,
    )
    if network_directory is None:  # pragma: no cover - required=True raises instead
        raise BeamLaunchPathError("R5 network directory unexpectedly missing.")
    selected_osm_path, ignored_osm_paths = _select_r5_osm(network_directory)
    gtfs_paths = _select_r5_gtfs(network_directory.execution_path)
    return R5NetworkLaunchReference(
        network_directory=network_directory,
        secondary_network_directory=secondary_network_directory,
        selected_osm_path=selected_osm_path,
        selected_osm_physical_target_path=_physical_target(
            selected_osm_path,
            mutable_root=_mutable_input_root(workspace),
            config_key="beam.routing.r5.directory",
        ),
        selected_osm_container_path=_container_path(
            selected_osm_path,
            mutable_root=_mutable_input_root(workspace),
        ),
        gtfs_paths=gtfs_paths,
        ignored_osm_paths=ignored_osm_paths,
    )


def validate_r5_execution_reference(
    *, settings: Any, workspace: Any, run_context: Any
) -> R5NetworkLaunchReference:
    """Verify Consist's selected R5 OSM member is the file BEAM will read."""

    snapshot = run_context.canonicalization
    if snapshot is None:
        raise BeamLaunchPathError("beam_run requires a Consist canonicalization snapshot.")
    r5_snapshot = next(
        (item for item in snapshot.references if item.reference.config_key == "beam.routing.r5.directory"),
        None,
    )
    if r5_snapshot is None:
        raise BeamLaunchPathError("Consist canonicalization did not observe beam.routing.r5.directory.")
    members = tuple(member for member in r5_snapshot.artifact_members if member.role == "r5_osm_source")
    if len(members) != 1:
        raise BeamLaunchPathError(f"Consist canonicalization must expose exactly one r5_osm_source member; observed {len(members)}.")
    member = members[0]
    if member.artifact_key not in r5_snapshot.artifact_keys:
        raise BeamLaunchPathError("Consist r5_osm_source member key is absent from its R5 directory reference.")
    execution_reference = resolve_r5_network_reference(settings=settings, workspace=workspace)
    if member.resolved_path.resolve() != execution_reference.selected_osm_physical_target_path:
        raise BeamLaunchPathError(
            "Consist selected R5 OSM member differs from the file BEAM will read: "
            f"{member.resolved_path} != {execution_reference.selected_osm_physical_target_path}"
        )
    return execution_reference


def prepare_r5_raw_rebuild(*, settings: Any, workspace: Any) -> R5NetworkLaunchReference:
    """Remove derived R5 and PhysSim caches from the mutable launch tree.

    BEAM uses an existing ``network.dat`` in preference to raw R5 source
    members.  PILATES therefore removes only the reproducible derived files
    before launch.  A lock is evidence of a live competing rebuild and is
    deliberately never removed.
    """

    _align_generated_network_paths(settings=settings, workspace=workspace)
    reference = resolve_r5_network_reference(settings=settings, workspace=workspace)
    mutable_root = _mutable_input_root(workspace)
    _remove_r5_derived_caches(reference.network_directory, mutable_root=mutable_root)
    if reference.secondary_network_directory is not None:
        _remove_r5_derived_caches(
            reference.secondary_network_directory,
            mutable_root=mutable_root,
        )
    return reference


def _align_generated_network_paths(*, settings: Any, workspace: Any) -> None:
    """Make both BEAM consumers of the generated PhysSim network agree.

    ``NetworkCoordinator`` creates ``physsim-network.xml`` beside the selected
    R5 raw inputs.  These keys otherwise permit stale or unrelated network
    files to be read by later BEAM phases.
    """

    config_path = beam_primary_config_path(settings, workspace=workspace)
    env_overrides = beam_config_env_overrides(settings, workspace=workspace)
    generated_network_path = '${beam.routing.r5.directory}"/physsim-network.xml"'
    for config_key in (
        "matsim.modules.network.inputNetworkFile",
        "beam.physsim.inputNetworkFilePath",
    ):
        update_staged_beam_config_value(
            config_path,
            key=config_key,
            value=generated_network_path,
            env_overrides=env_overrides,
        )


def _remove_r5_derived_caches(
    directory: BeamLaunchPathReference,
    *,
    mutable_root: Path,
) -> None:
    lock_path = directory.execution_path / "network.dat.lock"
    if lock_path.exists():
        raise BeamLaunchPathError(
            "Cannot prepare an R5 raw rebuild while an active network.dat.lock exists: "
            f"{lock_path}"
        )

    for filename in (
        "network.dat",
        "osm.mapdb",
        "osm.mapdb.p",
        "physsim-network.xml",
    ):
        cache_path = directory.execution_path / filename
        if not cache_path.exists() and not cache_path.is_symlink():
            continue
        _physical_target(
            cache_path,
            mutable_root=mutable_root,
            config_key=directory.config_key,
        )
        if not cache_path.is_file() and not cache_path.is_symlink():
            raise BeamLaunchPathError(
                "Expected derived R5 cache to be a file, not a directory: "
                f"{cache_path}"
            )
        cache_path.unlink()


def _resolve_hocon_path_reference(
    *,
    settings: Any,
    workspace: Any,
    config_key: str,
    required: bool,
) -> Optional[BeamLaunchPathReference]:
    config_path = beam_primary_config_path(settings, workspace=workspace)
    resolved_value = resolve_beam_config_value(
        config_path,
        key=config_key,
        env_overrides=beam_config_env_overrides(settings, workspace=workspace),
    )
    if resolved_value is None or not str(resolved_value).strip():
        if required:
            raise BeamLaunchPathError(
                f"Final BEAM HOCON key '{config_key}' is required for R5 launch."
            )
        return None

    configured_path = Path(str(resolved_value))
    if not configured_path.is_absolute():
        raise BeamLaunchPathError(
            f"Final BEAM HOCON key '{config_key}' did not resolve to an absolute host path: "
            f"{resolved_value!r}"
        )
    mutable_root = _mutable_input_root(workspace)
    physical_target_path = _physical_target(
        configured_path,
        mutable_root=mutable_root,
        config_key=config_key,
    )
    if not configured_path.exists():
        raise BeamLaunchPathError(
            f"Final BEAM HOCON key '{config_key}' does not exist: {configured_path}"
        )
    if not configured_path.is_dir():
        raise BeamLaunchPathError(
            f"Final BEAM HOCON key '{config_key}' must resolve to a directory: {configured_path}"
        )
    return BeamLaunchPathReference(
        config_key=config_key,
        raw_value=str(resolved_value),
        canonical_value=str(physical_target_path),
        configured_path=configured_path,
        execution_path=configured_path,
        physical_target_path=physical_target_path,
        container_path=_container_path(configured_path, mutable_root=mutable_root),
    )


def _select_r5_osm(
    directory: BeamLaunchPathReference,
) -> tuple[Path, tuple[Path, ...]]:
    candidates = tuple(
        path
        for path in sorted(directory.execution_path.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.suffix.lower() in {".pbf", ".vex"}
    )
    if not candidates:
        raise BeamLaunchPathError(
            "R5 raw rebuild requires one .pbf or .vex file in "
            f"{directory.execution_path}."
        )
    return candidates[0], candidates[1:]


def _select_r5_gtfs(directory: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.suffix.lower() == ".zip" and _is_gtfs_zip(path)
    )


def _is_gtfs_zip(path: Path) -> bool:
    try:
        with ZipFile(path) as archive:
            return archive.getinfo("stop_times.txt") is not None
    except (BadZipFile, KeyError, OSError):
        return False


def _mutable_input_root(workspace: Any) -> Path:
    return Path(workspace.get_beam_mutable_data_dir()).resolve()


def _physical_target(path: Path, *, mutable_root: Path, config_key: str) -> Path:
    physical_target = path.resolve()
    try:
        physical_target.relative_to(mutable_root)
    except ValueError as exc:
        raise BeamLaunchPathError(
            f"Final BEAM HOCON key '{config_key}' resolves outside the mutable BEAM "
            f"region input tree: {physical_target}"
        ) from exc
    return physical_target


def _container_path(path: Path, *, mutable_root: Path) -> str:
    try:
        relative_path = path.resolve().relative_to(mutable_root)
    except ValueError as exc:  # pragma: no cover - guarded by _physical_target
        raise BeamLaunchPathError(
            f"Cannot map path outside the BEAM input mount: {path}"
        ) from exc
    return str(Path("/app/input") / relative_path)
