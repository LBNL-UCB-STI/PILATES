"""Preflight materialization of the exact BEAM configuration tree to run.

The mutable workspace tree is an input to preprocessing, not the configuration
artifact passed to BEAM.  This module creates a fresh derived tree after
preprocessing and before ``beam_run`` so Consist can canonicalize, bind, and
the container can mount the same concrete files.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from consist.core.identity import IdentityManager
from consist.integrations.beam import BeamConfigAdapter, BeamConfigOverrides

from pilates.beam.admission import copy_linkstats_provenance_sidecar
from pilates.beam.config_hocon import (
    beam_config_env_overrides,
    resolve_beam_config_value,
)
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
)


_BEAM_CONTAINER_OUTPUT_DIR = "/app/output"
logger = logging.getLogger(__name__)


def _link_or_copy_static_file(source: str | Path, dest: str | Path) -> str:
    """Link static input files into the staged launch tree; copy mutable configs.

    Files with extension ``.conf`` (case-insensitive) are copied to ensure that
    any in-place modifications to HOCON configuration files never mutate the
    underlying source repository or shared input tree. All other static files are
    hardlinked where possible. If cross-device linking fails (e.g. ``EXDEV``
    across filesystems), it falls back directly to copying rather than symlinking,
    preventing broken container bind mounts and root-escape validation errors.
    """

    src = Path(source)
    dst = Path(dest)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)

    if src.suffix.lower() == ".conf":
        shutil.copy2(src, dst)
        return "copy"

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        pass

    dst.unlink(missing_ok=True)
    shutil.copy2(src, dst)
    return "copy"


_ARTIFACT_FORMAT_POLICY_KEYS = {
    "activitysim_skims": "beam.router.skim.activity-sim-skimmer.fileOutputFormat",
    "exchange": "beam.exchange.scenario.fileFormat",
    "events": "beam.outputs.events.fileOutputFormats",
    "linkstats": "beam.physsim.linkStatsOutputFileType",
}
_ACTIVITYSIM_SKIMS_FILE_BASE_NAME_KEY = (
    "beam.router.skim.activity-sim-skimmer.fileBaseName"
)


def _warmstart_destination_name(source: Path) -> str:
    """Return a BEAM reader-compatible warm-start name for ``source`` bytes.

    Native runs written before the format-neutral linkstats output migration can
    carry Parquet bytes at the legacy ``.csv.gz`` path.  BEAM chooses its reader
    from the filename, so retain the stem but correct a recognizable format
    suffix before the launch tree is compiled.
    """

    with source.open("rb") as stream:
        magic = stream.read(4)
    if magic == b"PAR1":
        format_suffix = ".parquet"
    elif magic.startswith(b"\x1f\x8b"):
        format_suffix = ".csv.gz"
    else:
        format_suffix = "".join(source.suffixes)

    source_suffix = "".join(source.suffixes)
    stem = source.name[: -len(source_suffix)] if source_suffix else source.name
    return f"{stem}{format_suffix}"


@dataclass(frozen=True)
class BeamLaunchConfig:
    """A fresh BEAM input tree and the primary config within that tree."""

    root: Path
    primary_config: Path

    def __post_init__(self) -> None:
        root = self.root.resolve()
        primary_config = self.primary_config.resolve()
        if primary_config == root or root not in primary_config.parents:
            raise ValueError(
                "Beam launch primary_config must be contained by the launch root."
            )


@dataclass(frozen=True)
class BeamLaunchConfigOverrides:
    """Structured HOCON overrides applied only to the derived config tree."""

    values: Mapping[str, Any]


@dataclass(frozen=True)
class BeamLaunchInput:
    """One prepared artifact copied into the exact derived BEAM input tree."""

    key: str
    source: Path
    destination: Path

    def __post_init__(self) -> None:
        if self.destination.is_absolute() or ".." in self.destination.parts:
            raise ValueError(
                "Beam launch input destination must be a relative path within the "
                f"launch tree: {self.destination}"
            )


def materialize_beam_launch_config(
    *,
    settings: Any,
    source_root: Path,
    output_dir: Path,
    identity: IdentityManager,
    overrides: BeamLaunchConfigOverrides,
    staged_inputs: tuple[BeamLaunchInput, ...] = (),
    allow_missing_override_keys: bool = False,
) -> BeamLaunchConfig:
    """Copy ``source_root`` and apply launch-only overrides to the copy.

    ``BeamConfigAdapter.materialize`` is deliberately used rather than PILATES
    HOCON rewriting: it canonicalizes the resulting config tree as the source
    of the downstream run identity.  Ordinary launch overrides must target an
    existing HOCON setting.  A first-class PILATES artifact-format policy may
    introduce a BEAM setting whose legacy source config relied on BEAM's own
    default instead.
    """

    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"BEAM source config root is not a directory: {source_root}"
        )

    primary_config = source_root / settings.beam.config
    if not primary_config.is_file():
        raise FileNotFoundError(f"BEAM primary config does not exist: {primary_config}")

    derived_root = output_dir / source_root.name
    materialized_values = {
        "beam.inputDirectory": str(derived_root),
        **dict(overrides.values),
    }
    if allow_missing_override_keys:
        _require_declared_non_policy_overrides(
            settings=settings,
            source_primary=primary_config,
            override_keys=tuple(materialized_values),
        )
    adapter = BeamConfigAdapter(
        root_dirs=[source_root],
        primary_config=primary_config,
        env_overrides=beam_config_env_overrides(settings, config_root=source_root),
    )
    source_roots = _materialization_source_roots(
        adapter=adapter,
        source_root=source_root,
        identity=identity,
    )
    orig_copytree = shutil.copytree

    def _hardlink_copytree(
        src: Any,
        dst: Any,
        symlinks: bool = False,
        ignore: Any = None,
        copy_function: Any = _link_or_copy_static_file,
        ignore_dangling_symlinks: bool = False,
        dirs_exist_ok: bool = True,
    ) -> Any:
        return orig_copytree(
            src,
            dst,
            symlinks=symlinks,
            ignore=ignore,
            copy_function=_link_or_copy_static_file,
            ignore_dangling_symlinks=ignore_dangling_symlinks,
            dirs_exist_ok=dirs_exist_ok,
        )

    try:
        shutil.copytree = _hardlink_copytree
        canonical_config = adapter.materialize(
            list(source_roots),
            BeamConfigOverrides(values=materialized_values),
            output_dir=output_dir,
            identity=identity,
            strict=not allow_missing_override_keys,
        )
    finally:
        shutil.copytree = orig_copytree

    derived_primary_config = derived_root / settings.beam.config
    if canonical_config.primary_config != derived_primary_config:
        raise RuntimeError(
            "Consist materialized BEAM config at an unexpected primary path: "
            f"{canonical_config.primary_config} (expected {derived_primary_config})."
        )
    for staged_input in staged_inputs:
        source = staged_input.source.resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"Prepared BEAM input {staged_input.key!r} is missing: {source}"
            )
        destination = derived_root / staged_input.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        shutil.copy2(source, destination)
        copy_linkstats_provenance_sidecar(
            source, destination, fallback_source_path=staged_input.source
        )
    return BeamLaunchConfig(root=derived_root, primary_config=derived_primary_config)


def _require_declared_non_policy_overrides(
    *,
    settings: Any,
    source_primary: Path,
    override_keys: tuple[str, ...],
) -> None:
    """Preserve strict validation when a legacy format key must be introduced."""

    policy_keys = frozenset(
        (
            _ACTIVITYSIM_SKIMS_FILE_BASE_NAME_KEY,
            *_ARTIFACT_FORMAT_POLICY_KEYS.values(),
        )
    )
    environment = beam_config_env_overrides(settings, config_root=source_primary.parent)
    for key in override_keys:
        if key == "beam.inputDirectory" or key in policy_keys:
            continue
        if (
            resolve_beam_config_value(
                source_primary,
                key=key,
                env_overrides=environment,
            )
            is None
        ):
            raise KeyError(f"Override key not found: {key}")


def build_beam_launch_config(
    *,
    settings: Any,
    source_root: Path,
    output_dir: Path,
    identity: IdentityManager,
    prepared_inputs: Mapping[str, Path],
) -> BeamLaunchConfig:
    """Compile the explicit post-preprocess BEAM launch tree.

    Each produced input is copied to a named location in the fresh tree and the
    derived primary config receives absolute paths to those copies.  This makes
    a cache-hydrated preprocess result just as usable as a fresh preprocessing
    result; no later runner path is allowed to consult the workspace input tree.
    """

    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    derived_root = output_dir / source_root.name
    env_overrides = beam_config_env_overrides(settings, config_root=source_root)
    source_primary = source_root / settings.beam.config
    exchange_relative = _relative_source_path(
        source_root=source_root,
        path=Path(
            str(
                resolve_beam_config_value(
                    source_primary,
                    key="beam.exchange.scenario.folder",
                    env_overrides=env_overrides,
                )
                or source_root / settings.beam.scenario_folder
            )
        ),
        key="beam.exchange.scenario.folder",
    )
    staged_inputs: list[BeamLaunchInput] = []
    overrides: dict[str, Any] = {
        "beam.outputs.baseOutputDirectory": _BEAM_CONTAINER_OUTPUT_DIR,
    }
    format_overrides, introduces_missing_format_key = _artifact_format_overrides(
        settings=settings,
        source_primary=source_primary,
        env_overrides=env_overrides,
    )
    overrides.update(format_overrides)
    skim_name_overrides, introduces_missing_skim_name_key = (
        _activitysim_skims_file_base_name_override(
            settings=settings,
            source_primary=source_primary,
            env_overrides=env_overrides,
        )
    )
    overrides.update(skim_name_overrides)
    population_stems = {
        BEAM_PLANS_IN: "plans",
        BEAM_HOUSEHOLDS_IN: "households",
        BEAM_PERSONS_IN: "persons",
    }
    for key, stem in population_stems.items():
        source = prepared_inputs.get(key)
        if source is None:
            continue
        source = Path(source)
        destination = exchange_relative / f"{stem}{''.join(source.suffixes)}"
        staged_inputs.append(
            BeamLaunchInput(key=key, source=source, destination=destination)
        )

    warmstart = prepared_inputs.get(LINKSTATS_WARMSTART)
    if warmstart is not None:
        warmstart_source = Path(warmstart)
        warmstart_destination = (
            Path(".pilates")
            / "warmstarts"
            / _warmstart_destination_name(warmstart_source)
        )
        staged_inputs.append(
            BeamLaunchInput(
                key=LINKSTATS_WARMSTART,
                source=warmstart_source,
                destination=warmstart_destination,
            )
        )
        overrides["beam.warmStart.initialLinkstatsFilePath"] = str(
            derived_root / warmstart_destination
        )

    vehicle_source = prepared_inputs.get("vehicles_beam_in")
    if vehicle_source is not None:
        vehicle_destination = exchange_relative / (
            "vehicles" + "".join(Path(vehicle_source).suffixes)
        )
        staged_inputs.append(
            BeamLaunchInput(
                key="vehicles_beam_in",
                source=Path(vehicle_source),
                destination=vehicle_destination,
            )
        )
        overrides["beam.agentsim.agents.vehicles.vehiclesFilePath"] = str(
            derived_root / vehicle_destination
        )
    else:
        configured_vehicles_path = resolve_beam_config_value(
            source_primary,
            key="beam.agentsim.agents.vehicles.vehiclesFilePath",
            env_overrides=env_overrides,
        )
        if (
            configured_vehicles_path
            and not Path(str(configured_vehicles_path)).exists()
        ):
            overrides["beam.agentsim.agents.vehicles.vehiclesFilePath"] = ""

    if settings.beam.discard_plans_every_year:
        overrides["beam.replanning.maxAgentPlanMemorySize"] = 0
    zones = settings.shared.geography.zones
    if zones is not None:
        zone_destination = Path("shape") / "canonical_zones_sorted.geojson"
        overrides["beam.agentsim.taz.filePath"] = str(derived_root / zone_destination)
        overrides["beam.agentsim.taz.tazIdFieldName"] = zones.activitysim_index_col

    r5_source = resolve_beam_config_value(
        source_primary,
        key="beam.routing.r5.directory",
        env_overrides=env_overrides,
    )
    if r5_source:
        r5_relative = _relative_source_path(
            source_root=source_root,
            path=Path(str(r5_source)),
            key="beam.routing.r5.directory",
        )
        generated_network = derived_root / r5_relative / "physsim-network.xml"
        overrides["matsim.modules.network.inputNetworkFile"] = str(generated_network)
        overrides["beam.physsim.inputNetworkFilePath"] = str(generated_network)

    launch_config = materialize_beam_launch_config(
        settings=settings,
        source_root=source_root,
        output_dir=output_dir,
        identity=identity,
        overrides=BeamLaunchConfigOverrides(values=overrides),
        staged_inputs=tuple(staged_inputs),
        allow_missing_override_keys=(
            introduces_missing_format_key or introduces_missing_skim_name_key
        ),
    )
    if overrides.get("beam.exchange.scenario.fileFormat") == "parquet":
        has_population_inputs = any(
            prepared_inputs.get(key) is not None for key in population_stems
        )
        _ensure_parquet_exchange_scenario_files(
            derived_root=launch_config.root,
            exchange_relative=exchange_relative,
            scenario_folder=settings.beam.scenario_folder,
            has_population_inputs=has_population_inputs,
        )
    return launch_config


def _ensure_parquet_exchange_scenario_files(
    *,
    derived_root: Path,
    exchange_relative: Path,
    scenario_folder: str,
    has_population_inputs: bool,
) -> None:
    """Ensure ``UrbansimReaderV2`` and ``ODRequester`` find valid Parquet scenario files.

    When ``beam.exchange.scenario.fileFormat`` is ``"parquet"``, BEAM's
    ``UrbansimReaderV2`` unconditionally requires ``persons.parquet``,
    ``plans.parquet``, ``households.parquet``, and ``blocks.parquet`` in
    ``beam.exchange.scenario.folder``, and ``ODRequester.createDummyPersonAttribute``
    requires at least one loaded household with a member and plan—even during
    standalone ``beam_full_skim`` where ``FullSkimsCreatorApp`` never simulates
    agents. Additionally, when ``beam.exchange.scenario.folder`` points to a
    nested subdirectory (such as ``urbansim/2018``) while ``blocks.parquet``
    lives in the parent ``scenario_folder`` (``urbansim/blocks.parquet``), copy
    ``blocks.parquet`` into ``exchange_dir`` if present, and write 1-row
    synthetic Parquet stubs referencing a valid ``block_id`` for any missing
    files when no population inputs were staged.
    """

    exchange_dir = derived_root / exchange_relative
    exchange_dir.mkdir(parents=True, exist_ok=True)

    blocks_dest = exchange_dir / "blocks.parquet"
    if not blocks_dest.is_file() and scenario_folder:
        fallback_blocks = derived_root / scenario_folder / "blocks.parquet"
        if fallback_blocks.is_file():
            shutil.copy2(fallback_blocks, blocks_dest)

    if not has_population_inputs:
        if not blocks_dest.is_file():
            _write_synthetic_parquet_stub(
                blocks_dest,
                stem="blocks",
                block_id=1,
                block_x=550000.0,
                block_y=4180000.0,
            )
        block_id, block_x, block_y = _first_block_row(blocks_dest)
        for stem in ("persons", "plans", "households"):
            target = exchange_dir / f"{stem}.parquet"
            if not target.is_file():
                _write_synthetic_parquet_stub(
                    target,
                    stem=stem,
                    block_id=block_id,
                    block_x=block_x,
                    block_y=block_y,
                )


def _first_block_row(blocks_path: Path) -> tuple[int, float, float]:
    """Read the first ``(block_id, x, y)`` row from ``blocks.parquet``."""

    import pyarrow.parquet as pq

    table = pq.read_table(blocks_path, columns=["block_id", "x", "y"])
    if table.num_rows == 0:
        return 1, 550000.0, 4180000.0
    row = table.slice(0, 1).to_pydict()
    return (
        int(row["block_id"][0]),
        float(row["x"][0]),
        float(row["y"][0]),
    )


def _write_synthetic_parquet_stub(
    path: Path,
    *,
    stem: str,
    block_id: int,
    block_x: float,
    block_y: float,
) -> None:
    """Write a minimal Parquet stub matching ``ParquetReaders.scala``, ``HOVModeTransformer``, and ``ODRequester``."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    tables: dict[str, pa.Table] = {
        "persons": pa.Table.from_pydict(
            {
                "person_id": [1],
                "household_id": [1],
                "age": [35],
                "sex": [1],
                "industry": ["stub"],
                "value_of_time": [18.0],
            },
            schema=pa.schema(
                [
                    ("person_id", pa.int64()),
                    ("household_id", pa.int64()),
                    ("age", pa.int64()),
                    ("sex", pa.int64()),
                    ("industry", pa.string()),
                    ("value_of_time", pa.float64()),
                ]
            ),
        ),
        "plans": pa.Table.from_pydict(
            {
                "trip_id": [None, None, None, None, None],
                "tour_id": [None, None, None, None, None],
                "person_id": [1, 1, 1, 1, 1],
                "PlanElementIndex": [1, 2, 3, 4, 5],
                "ActivityElement": [
                    "activity",
                    "leg",
                    "activity",
                    "leg",
                    "activity",
                ],
                "trip_mode": [None, "WALK", None, "WALK", None],
                "ActivityType": ["Home", None, "Work", None, "Home"],
                "x": [block_x, None, block_x, None, block_x],
                "y": [block_y, None, block_y, None, block_y],
                "departure_time": [8.0, None, 17.0, None, None],
                "trip_dur_min": [None, None, None, None, None],
                "trip_cost_dollars": [None, None, None, None, None],
            },
            schema=pa.schema(
                [
                    ("trip_id", pa.string()),
                    ("tour_id", pa.string()),
                    ("person_id", pa.int64()),
                    ("PlanElementIndex", pa.int64()),
                    ("ActivityElement", pa.string()),
                    ("trip_mode", pa.string()),
                    ("ActivityType", pa.string()),
                    ("x", pa.float64()),
                    ("y", pa.float64()),
                    ("departure_time", pa.float64()),
                    ("trip_dur_min", pa.float64()),
                    ("trip_cost_dollars", pa.float64()),
                ]
            ),
        ),
        "households": pa.Table.from_pydict(
            {
                "household_id": [1],
                "cars": [1],
                "income": [50000],
                "block_id": [block_id],
            },
            schema=pa.schema(
                [
                    ("household_id", pa.int64()),
                    ("cars", pa.int64()),
                    ("income", pa.int64()),
                    ("block_id", pa.int64()),
                ]
            ),
        ),
        "blocks": pa.Table.from_pydict(
            {
                "block_id": [block_id],
                "x": [block_x],
                "y": [block_y],
            },
            schema=pa.schema(
                [
                    ("block_id", pa.int64()),
                    ("x", pa.float64()),
                    ("y", pa.float64()),
                ]
            ),
        ),
    }
    pq.write_table(tables[stem], path)


def _activitysim_skims_file_base_name_override(
    *,
    settings: Any,
    source_primary: Path,
    env_overrides: Mapping[str, str],
) -> tuple[dict[str, str], bool]:
    """Return the configured BEAM activity-sim skimmer writer base name."""

    expected = settings.beam.activitysim_skims_file_base_name
    source_value = resolve_beam_config_value(
        source_primary,
        key=_ACTIVITYSIM_SKIMS_FILE_BASE_NAME_KEY,
        env_overrides=dict(env_overrides),
    )
    if source_value is None:
        logger.warning(
            "PILATES activity-sim skim filename policy adds source config key "
            "%s=%r; the legacy config relied on BEAM's implicit default.",
            _ACTIVITYSIM_SKIMS_FILE_BASE_NAME_KEY,
            expected,
        )
    elif str(source_value) != expected:
        logger.warning(
            "PILATES activity-sim skim filename policy overrides source config: "
            "%s=%r -> %r.",
            _ACTIVITYSIM_SKIMS_FILE_BASE_NAME_KEY,
            source_value,
            expected,
        )
    return {_ACTIVITYSIM_SKIMS_FILE_BASE_NAME_KEY: expected}, source_value is None


def _artifact_format_overrides(
    *,
    settings: Any,
    source_primary: Path,
    env_overrides: Mapping[str, str],
) -> tuple[dict[str, str], bool]:
    """Return typed format policy overrides and report legacy missing keys."""

    formats = settings.beam.artifact_formats
    field_sources = formats.model_fields_set
    format_values = formats.model_dump()
    overrides = {
        config_key: format_values[field_name]
        for field_name, config_key in _ARTIFACT_FORMAT_POLICY_KEYS.items()
    }
    introduces_missing_format_key = False
    policy_summary: list[str] = []
    for field_name, config_key in _ARTIFACT_FORMAT_POLICY_KEYS.items():
        expected = overrides[config_key]
        source_value = resolve_beam_config_value(
            source_primary,
            key=config_key,
            env_overrides=dict(env_overrides),
        )
        source_kind = "configured" if field_name in field_sources else "default"
        policy_summary.append(f"{field_name}={expected} ({source_kind})")
        if source_value is None:
            introduces_missing_format_key = True
            logger.warning(
                "PILATES %s BEAM artifact-format policy adds source config key "
                "%s=%r; the legacy config relied on BEAM's implicit default.",
                source_kind,
                config_key,
                expected,
            )
        elif str(source_value) != expected:
            logger.warning(
                "PILATES %s BEAM artifact-format policy overrides source config: "
                "%s=%r -> %r.",
                source_kind,
                config_key,
                source_value,
                expected,
            )
    logger.info(
        "[BEAM] Effective artifact-format policy: %s", ", ".join(policy_summary)
    )
    return overrides, introduces_missing_format_key


def _relative_source_path(*, source_root: Path, path: Path, key: str) -> Path:
    try:
        return path.resolve().relative_to(source_root)
    except ValueError as exc:
        raise ValueError(
            f"BEAM config key {key!r} resolves outside the source input tree: {path}"
        ) from exc


def _materialization_source_roots(
    *,
    adapter: BeamConfigAdapter,
    source_root: Path,
    identity: IdentityManager,
) -> tuple[Path, ...]:
    """Return the region tree plus any sibling roots used by its config graph."""

    discovered = adapter.discover([source_root], identity=identity, strict=True)
    source_parent = source_root.parent
    roots = [source_root]
    for config_file in discovered.config_files:
        resolved_config = config_file.resolve()
        if resolved_config.is_relative_to(source_root):
            continue
        try:
            sibling_name = resolved_config.relative_to(source_parent).parts[0]
        except ValueError as exc:
            raise ValueError(
                "BEAM config include resolves outside the staged production tree: "
                f"{resolved_config}"
            ) from exc
        sibling_root = source_parent / sibling_name
        if not sibling_root.is_dir():
            raise ValueError(
                "BEAM config include must be contained by a sibling directory: "
                f"{resolved_config}"
            )
        if sibling_root not in roots:
            roots.append(sibling_root)
    return tuple(roots)
