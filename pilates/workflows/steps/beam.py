"""BEAM workflow steps demonstrating the PILATES-Consist integration pattern.

The binding rules in `pilates.workflows.binding` declare BEAM's required
inputs, including exact-rewind snapshot artifacts for ActivitySim outputs,
vehicles, warm starts, and configuration references. The step factories in this
module convert those bindings into model execution, publish current-role
outputs through the Consist coupler, and log output-only diagnostic families
without expanding the handoff surface. Recovery roots remain storage metadata:
snapshot artifacts describe semantic model boundaries, while archive promotion
and future Consist recovery policy decide where the bytes can be restored from.
"""

from __future__ import annotations

import json
import inspect
import logging
import os
from pathlib import Path
import re
import shutil
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from consist import BindingResult, CacheOptions, ExecutionOptions, define_step

from pilates.beam.config_hocon import (
    BeamConfigHoconError,
    beam_config_env_overrides,
    beam_config_root,
    beam_primary_config_path,
    load_resolved_beam_config_tree,
)
from pilates.beam.admission import (
    preflight_staged_linkstats_admission,
    reject_or_warn_for_missing_staged_linkstats,
)
from pilates.beam.launch_paths import (
    validate_r5_execution_reference,
    validate_staged_linkstats_reference,
)
from pilates.beam.runner import BeamRunner
from pilates.config.models import PilatesConfig
from pilates.utils import consist_runtime as cr
from pilates.utils.coupler_helpers import (
    artifact_to_existing_path,
    artifact_to_path,
    enqueue_archive_copy,
    set_coupler_from_artifact,
)
from pilates.workflows.artifact_keys import (
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_CONFIG_FILE,
    BEAM_HOUSEHOLDS_IN,
    BEAM_INPUT_CONFIG_ARCHIVED,
    BEAM_INPUT_CONFIG_REFERENCES_ARCHIVED,
    BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED,
    BEAM_INPUT_HOUSEHOLDS_ARCHIVED,
    BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED,
    BEAM_INPUT_PERSONS_ARCHIVED,
    BEAM_INPUT_PLANS_ARCHIVED,
    BEAM_INPUT_PLANS_WARMSTART_ARCHIVED,
    BEAM_INPUT_VEHICLES_ARCHIVED,
    BEAM_MUTABLE_DATA_DIR,
    BEAM_NETWORK_FINAL,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
)
from pilates.workflows.state_helpers import resolve_forecast_year
from pilates.workflows.output_projection import require_output
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.step_consist_meta import consist_step_meta
from pilates.workflows.step_definition import StepDefinition
from pilates.workflows.outputs_base import ValidationContext
from pilates.workspace import Workspace

# Model-specific step factories for BEAM.
# Shared helpers/infrastructure are imported from shared.py.
from .shared import (
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_PLANS_XML,
    BEAM_PLANS_OUT,
    BeamFullSkimOutputs,
    BeamPostprocessOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
    CouplerProtocol,
    StandardStepSpec,
    StepOutputsHolder,
    WorkflowState,
    _beam_log_facet_meta,
    _beam_postprocess_split_facet_meta,
    build_standard_step,
    _log_step_records,
    make_default_recoverer,
    _schema_outputs_from_class,
    find_last_run_output_plans,
    log_and_set_output,
    log_input_only,
    log_output_only,
    recovered_cached_paths,
)

logger = logging.getLogger(__name__)

_BEAM_INCLUDE_RE = re.compile(r'^\s*include\s+(?:"([^"]+)"|file\("([^"]+)"\))')
_BEAM_CONFIG_REFERENCE_MANIFEST = "__archive_manifest.json"


def _primary_beam_config_path(
    settings: PilatesConfig,
    workspace: Workspace,
) -> Path:
    return beam_primary_config_path(settings, workspace=workspace)


def _require_primary_beam_config(
    settings: PilatesConfig,
    workspace: Workspace,
) -> Path:
    config_path = _primary_beam_config_path(settings, workspace)
    if not config_path.exists():
        raise FileNotFoundError(
            "BEAM primary config file is missing: "
            f"{config_path}. Expected from settings.beam.config="
            f"{settings.beam.config!r} under the mutable BEAM input dir for "
            f"region {settings.run.region!r}."
        )
    return config_path


def _is_beam_sub_iteration_key(short_name: Optional[str]) -> bool:
    return bool(
        short_name and ("_sub" in short_name or "__beam_sub_iter" in short_name)
    )


def _beam_linkstats_publication_meta(
    short_name: Optional[str],
    *,
    family: str,
) -> Dict[str, Any]:
    if not short_name:
        return {}
    for prefix in ("linkstats_parquet_", "linkstats_"):
        if not short_name.startswith(prefix):
            continue
        tail = short_name[len(prefix) :]
        parts = tail.split("_")
        if len(parts) < 2:
            continue
        try:
            year = int(parts[0])
            iteration = int(parts[1])
        except ValueError:
            continue
        facet: Dict[str, Any] = {
            "artifact_family": family,
            "year": year,
            "iteration": iteration,
        }
        if len(parts) > 2 and parts[2].startswith("sub"):
            try:
                facet["beam_sub_iteration"] = int(parts[2][3:])
            except ValueError:
                continue
        return {
            "facet": facet,
            "facet_schema_version": "v1",
            "facet_index": True,
        }
    return {}


_BEAM_RUN_ARCHIVE_KEY_MAP: Dict[str, str] = {
    BEAM_PLANS_IN: BEAM_INPUT_PLANS_ARCHIVED,
    BEAM_HOUSEHOLDS_IN: BEAM_INPUT_HOUSEHOLDS_ARCHIVED,
    BEAM_PERSONS_IN: BEAM_INPUT_PERSONS_ARCHIVED,
    BEAM_CONFIG_FILE: BEAM_INPUT_CONFIG_ARCHIVED,
    "vehicles_beam_in": BEAM_INPUT_VEHICLES_ARCHIVED,
    LINKSTATS_WARMSTART: BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED,
    BEAM_PLANS_OUT: BEAM_INPUT_PLANS_WARMSTART_ARCHIVED,
    BEAM_OUTPUT_PLANS_XML: BEAM_INPUT_PLANS_WARMSTART_ARCHIVED,
    BEAM_EXPERIENCED_PLANS_XML: BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML: (
        BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED
    ),
}

_BEAM_RUN_ARCHIVE_DESCRIPTION_MAP: Dict[str, str] = {
    BEAM_INPUT_PLANS_ARCHIVED: "Archived BEAM runner plans input snapshot",
    BEAM_INPUT_HOUSEHOLDS_ARCHIVED: "Archived BEAM runner households input snapshot",
    BEAM_INPUT_PERSONS_ARCHIVED: "Archived BEAM runner persons input snapshot",
    BEAM_INPUT_CONFIG_ARCHIVED: "Archived BEAM runner config input snapshot",
    BEAM_INPUT_CONFIG_REFERENCES_ARCHIVED: (
        "Archived BEAM config include/reference inputs snapshot"
    ),
    BEAM_INPUT_VEHICLES_ARCHIVED: "Archived BEAM runner vehicles input snapshot",
    BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED: (
        "Archived BEAM runner warm-start linkstats input snapshot"
    ),
    BEAM_INPUT_PLANS_WARMSTART_ARCHIVED: (
        "Archived BEAM runner warm-start plans input snapshot"
    ),
    BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED: (
        "Archived BEAM runner warm-start experienced plans input snapshot"
    ),
}

_BEAM_RUN_ARCHIVE_SOURCE_ROLE_MAP: Dict[str, str] = {
    BEAM_INPUT_PLANS_ARCHIVED: BEAM_PLANS_IN,
    BEAM_INPUT_HOUSEHOLDS_ARCHIVED: BEAM_HOUSEHOLDS_IN,
    BEAM_INPUT_PERSONS_ARCHIVED: BEAM_PERSONS_IN,
    BEAM_INPUT_CONFIG_ARCHIVED: BEAM_CONFIG_FILE,
    BEAM_INPUT_CONFIG_REFERENCES_ARCHIVED: "beam_config_references",
    BEAM_INPUT_VEHICLES_ARCHIVED: "vehicles_beam_in",
    BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED: LINKSTATS_WARMSTART,
    BEAM_INPUT_PLANS_WARMSTART_ARCHIVED: "beam_plans_warmstart",
    BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED: (
        "beam_experienced_plans_warmstart"
    ),
}


def _beam_preprocess_input_publication_meta(
    *,
    key: str,
    state: WorkflowState,
    input_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    if key != "vehicles_beam_in":
        return {}
    if input_metadata.get("source_semantic_key") != ATLAS_VEHICLES2_OUTPUT:
        return {}

    facet = {
        "artifact_family": "beam_preprocess_input",
        "source_role": key,
        "derived_from": ATLAS_VEHICLES2_OUTPUT,
        "year": getattr(state, "forecast_year", None),
        "iteration": getattr(state, "iteration", None),
        "source_year": input_metadata.get("source_year"),
        "source_resolution_mode": input_metadata.get("source_resolution_mode"),
        "source_storage_location": input_metadata.get("source_storage_location"),
    }
    for optional_key in (
        "filtered_to_staged_households",
        "staged_household_filter_removed_vehicle_rows",
    ):
        if optional_key in input_metadata:
            facet[optional_key] = input_metadata.get(optional_key)

    return {
        "facet": facet,
        "facet_schema_version": "v1",
        "facet_index": True,
        "source_path": input_metadata.get("source_path"),
    }


def _beam_run_snapshot_dir(
    *,
    workspace: Workspace,
    state: WorkflowState,
) -> Path:
    snapshot_year = resolve_forecast_year(state)
    return (
        Path(workspace.get_beam_output_dir())
        / f"inputs-year-{snapshot_year}-iteration-{state.iteration}"
    )


def _beam_input_archive_meta(
    *,
    archive_key: str,
    year: int,
    iteration: int,
) -> Dict[str, Any]:
    input_name = archive_key.removeprefix("beam_input_").removesuffix("_archived")
    return {
        "facet": {
            "artifact_family": "beam_input_archived",
            "input_name": input_name,
            "source_role": _BEAM_RUN_ARCHIVE_SOURCE_ROLE_MAP.get(
                archive_key,
                input_name,
            ),
            "snapshot_role": f"beam_input_{input_name}",
            "snapshot_reason": "exact_rewind",
            "storage_event": "snapshot_copy",
            "year": year,
            "iteration": iteration,
        },
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _scan_beam_config_includes(root: Path) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    queue = [root.resolve()]
    while queue:
        current = queue.pop(0)
        if current in seen or not current.exists() or not current.is_file():
            continue
        seen.add(current)
        ordered.append(current)
        try:
            lines = current.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            match = _BEAM_INCLUDE_RE.match(line)
            if not match:
                continue
            rel = match.group(1) or match.group(2)
            if not rel:
                continue
            queue.append((current.parent / rel).resolve())
    return ordered


def _collect_beam_config_path_references(config_tree: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            candidate = node.strip()
            if _looks_like_beam_path_reference(candidate):
                refs.add(candidate)

    walk(config_tree)
    return refs


def _looks_like_beam_path_reference(candidate: str) -> bool:
    ignore_tokens = {
        "csv",
        "csv.gz",
        "xml",
        "xml.gz",
        "parquet",
        "omx",
        "h5",
    }
    if not candidate or candidate in ignore_tokens:
        return False
    if candidate.startswith(("http://", "https://", "tcp://")):
        return False
    return "/" in candidate or candidate.endswith(
        (
            ".csv",
            ".csv.gz",
            ".xml",
            ".xml.gz",
            ".gz",
            ".parquet",
            ".zip",
            ".omx",
            ".h5",
        )
    )


def _resolve_beam_config_reference(value: str, root_dir: Path) -> Optional[Path]:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    resolved = (root_dir / candidate).resolve()
    return resolved


def _beam_config_reference_relative_path(path: Path, config_root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = config_root.resolve()
    if resolved.is_relative_to(root_resolved):
        return resolved.relative_to(root_resolved)
    parts = list(resolved.parts)
    if parts and parts[0] == resolved.anchor:
        parts = parts[1:]
    return Path("__external__", *parts)


def _beam_input_reference_relative_path(path: Path, beam_input_root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = beam_input_root.resolve()
    if resolved.is_relative_to(root_resolved):
        return resolved.relative_to(root_resolved)
    parts = list(resolved.parts)
    if parts and parts[0] == resolved.anchor:
        parts = parts[1:]
    return Path("__external__", *parts)


def _collect_beam_config_reference_sources(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
) -> set[Path]:
    config_path = _require_primary_beam_config(settings, workspace).resolve()
    config_root = beam_config_root(settings, workspace=workspace).resolve()
    sources: set[Path] = set()
    config_files = _scan_beam_config_includes(config_path)

    for include_path in config_files:
        if include_path != config_path:
            sources.add(include_path)

    try:
        config_tree = load_resolved_beam_config_tree(
            config_path,
            env_overrides=beam_config_env_overrides(
                settings,
                config_root=config_root,
            ),
        )
    except BeamConfigHoconError as exc:
        logger.warning(
            "Failed to resolve BEAM config references for archival from %s: %s",
            config_path,
            exc,
        )
        if sources:
            logger.debug(
                "pyhocon unavailable while archiving BEAM config references; "
                "archiving include files only."
            )
        return sources

    raw_refs = _collect_beam_config_path_references(config_tree)
    for raw_ref in sorted(raw_refs):
        resolved = _resolve_beam_config_reference(raw_ref, config_root)
        if resolved is None or not resolved.exists() or resolved == config_path:
            continue
        sources.add(resolved)
    return sources


def _copy_tree_contents(source_root: Path, target_root: Path) -> None:
    if not source_root.exists() or not source_root.is_dir():
        return
    for child in sorted(source_root.iterdir(), key=lambda path: path.name):
        target = target_root / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)


def _copy_path_for_beam_input_archive(source_path: Path, target_path: Path) -> None:
    source_resolved = source_path.resolve()
    target_resolved = target_path.resolve()
    if source_resolved == target_resolved:
        return
    if source_path.is_dir():
        if target_path.exists() and not target_path.is_dir():
            target_path.unlink()
        _copy_tree_contents(source_path, target_path)
        return
    if target_path.exists() and target_path.is_dir():
        shutil.rmtree(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def _materialize_beam_config_references_for_archive(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
) -> Dict[str, str]:
    """
    Ensure ConfigAdapter-discovered BEAM inputs are present under beam/input.

    Consist logs these paths as ``beam_input://...``. In local-to-archive runs,
    the archive worker only mirrors explicitly enqueued paths, so static BEAM
    production inputs must be enqueued even when they already exist in the
    node-local mutable input tree.
    """
    config_path = _require_primary_beam_config(settings, workspace).resolve()
    beam_input_root = Path(workspace.get_beam_mutable_data_dir()).resolve()
    sources = _collect_beam_config_reference_sources(
        settings=settings,
        workspace=workspace,
    )
    sources.add(config_path)

    materialized: Dict[str, str] = {}
    for source_path in sorted(
        sources,
        key=lambda path: (
            0 if path.is_dir() else 1,
            len(_beam_input_reference_relative_path(path, beam_input_root).parts),
            _beam_input_reference_relative_path(path, beam_input_root).as_posix(),
        ),
    ):
        if not source_path.exists():
            continue
        rel_target = _beam_input_reference_relative_path(source_path, beam_input_root)
        target_path = beam_input_root / rel_target
        _copy_path_for_beam_input_archive(source_path, target_path)
        enqueue_archive_copy(
            key=BEAM_INPUT_CONFIG_REFERENCES_ARCHIVED,
            path=target_path,
            workspace=workspace,
        )
        materialized[rel_target.as_posix()] = str(source_path)
    return materialized


def _archive_beam_config_references(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
    snapshot_dir: Path,
) -> Optional[Path]:
    config_root = beam_config_root(settings, workspace=workspace).resolve()
    archive_root = snapshot_dir / BEAM_INPUT_CONFIG_REFERENCES_ARCHIVED
    sources_by_target: Dict[Path, Path] = {}

    for include_path in _collect_beam_config_reference_sources(
        settings=settings,
        workspace=workspace,
    ):
        sources_by_target[
            _beam_config_reference_relative_path(include_path, config_root)
        ] = include_path

    if not sources_by_target:
        return None

    if archive_root.exists():
        shutil.rmtree(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, str] = {}
    for rel_target, source_path in sorted(
        sources_by_target.items(),
        key=lambda item: (
            0 if item[1].is_dir() else 1,
            len(item[0].parts),
            item[0].as_posix(),
        ),
    ):
        target_path = archive_root / rel_target
        if source_path.is_dir():
            _copy_tree_contents(source_path, target_path)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
        manifest[rel_target.as_posix()] = str(source_path)

    (archive_root / _BEAM_CONFIG_REFERENCE_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return archive_root


def _resolve_existing_coupler_input(
    *,
    coupler: CouplerProtocol,
    key: str,
    workspace: Workspace,
) -> Optional[tuple[str, str]]:
    get_value = getattr(coupler, "get", None)
    if not callable(get_value):
        return None
    resolved_path = artifact_to_existing_path(
        get_value(key),
        workspace=workspace,
    )
    if resolved_path is None:
        return None
    return key, resolved_path


def _resolve_beam_run_warmstart_inputs(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
    coupler: CouplerProtocol,
) -> tuple[Optional[tuple[str, str]], Optional[tuple[str, str]]]:
    plans_match = _resolve_existing_coupler_input(
        coupler=coupler,
        key=BEAM_OUTPUT_PLANS_XML,
        workspace=workspace,
    ) or _resolve_existing_coupler_input(
        coupler=coupler,
        key=BEAM_PLANS_OUT,
        workspace=workspace,
    )
    experienced_match = _resolve_existing_coupler_input(
        coupler=coupler,
        key=BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
        workspace=workspace,
    ) or _resolve_existing_coupler_input(
        coupler=coupler,
        key=BEAM_EXPERIENCED_PLANS_XML,
        workspace=workspace,
    )

    output_root = Path(workspace.get_beam_output_dir()) / settings.run.region
    if plans_match is None or experienced_match is None:
        scanned_plans_path, scanned_experienced_path = find_last_run_output_plans(
            output_root, "year-"
        )
        if (
            plans_match is None
            and scanned_plans_path is not None
            and scanned_plans_path.exists()
        ):
            scanned_plans_key = (
                BEAM_OUTPUT_PLANS_XML
                if scanned_plans_path.name == "output_plans.xml.gz"
                else BEAM_PLANS_OUT
            )
            plans_match = (scanned_plans_key, str(scanned_plans_path))
        if (
            experienced_match is None
            and scanned_experienced_path is not None
            and scanned_experienced_path.exists()
        ):
            scanned_experienced_key = (
                BEAM_OUTPUT_EXPERIENCED_PLANS_XML
                if scanned_experienced_path.name == "output_experienced_plans.xml.gz"
                else BEAM_EXPERIENCED_PLANS_XML
            )
            experienced_match = (scanned_experienced_key, str(scanned_experienced_path))
    return plans_match, experienced_match


def _collect_beam_run_snapshot_sources(
    *,
    settings: PilatesConfig,
    workspace: Workspace,
    holder: StepOutputsHolder,
    coupler: CouplerProtocol,
) -> Dict[str, Path]:
    upstream = holder.beam_preprocess
    if upstream is None:
        raise RuntimeError("BEAM preprocess must complete first")

    snapshot_sources: Dict[str, Path] = {
        BEAM_INPUT_CONFIG_ARCHIVED: _require_primary_beam_config(settings, workspace),
    }
    for short_name, path, _description in upstream._iter_record_items():
        archive_key = _BEAM_RUN_ARCHIVE_KEY_MAP.get(short_name)
        if archive_key is None:
            continue
        snapshot_sources[archive_key] = Path(path)

    plans_match, experienced_match = _resolve_beam_run_warmstart_inputs(
        settings=settings,
        workspace=workspace,
        coupler=coupler,
    )
    if plans_match is not None and Path(plans_match[1]).exists():
        snapshot_sources[BEAM_INPUT_PLANS_WARMSTART_ARCHIVED] = Path(plans_match[1])
    if experienced_match is not None and Path(experienced_match[1]).exists():
        snapshot_sources[BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED] = Path(
            experienced_match[1]
        )
    return snapshot_sources


def _archive_beam_run_inputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    holder: StepOutputsHolder,
    coupler: CouplerProtocol,
) -> None:
    snapshot_dir = _beam_run_snapshot_dir(workspace=workspace, state=state)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_year = resolve_forecast_year(state)

    config_reference_snapshot = _archive_beam_config_references(
        settings=settings,
        workspace=workspace,
        snapshot_dir=snapshot_dir,
    )
    if config_reference_snapshot is not None:
        log_output_only(
            key=BEAM_INPUT_CONFIG_REFERENCES_ARCHIVED,
            path=str(config_reference_snapshot),
            description=_BEAM_RUN_ARCHIVE_DESCRIPTION_MAP[
                BEAM_INPUT_CONFIG_REFERENCES_ARCHIVED
            ],
            step_name="beam_run",
            **_beam_input_archive_meta(
                archive_key=BEAM_INPUT_CONFIG_REFERENCES_ARCHIVED,
                year=snapshot_year,
                iteration=state.iteration,
            ),
        )
    materialized_references = _materialize_beam_config_references_for_archive(
        settings=settings,
        workspace=workspace,
    )
    if materialized_references:
        logger.info(
            "Enqueued %d BEAM config reference(s) for archive beam_input materialization",
            len(materialized_references),
        )

    for archive_key, source_path in _collect_beam_run_snapshot_sources(
        settings=settings,
        workspace=workspace,
        holder=holder,
        coupler=coupler,
    ).items():
        if not source_path.exists():
            raise FileNotFoundError(
                f"BEAM run input snapshot source is missing for {archive_key}: {source_path}"
            )
        target_path = snapshot_dir / f"{archive_key}{''.join(source_path.suffixes)}"
        if source_path.is_dir():
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)
        log_output_only(
            key=archive_key,
            path=str(target_path),
            description=_BEAM_RUN_ARCHIVE_DESCRIPTION_MAP[archive_key],
            step_name="beam_run",
            **_beam_input_archive_meta(
                archive_key=archive_key,
                year=snapshot_year,
                iteration=state.iteration,
            ),
        )


def _publish_beam_run_outputs(
    *,
    outputs: BeamRunOutputs,
    coupler: CouplerProtocol,
) -> None:
    promoted_linkstats = outputs.promoted_linkstats_for_publication()
    if promoted_linkstats is not None:
        source_key, path = promoted_linkstats
        linkstats_meta = _beam_linkstats_publication_meta(
            source_key,
            family="linkstats",
        )
        log_and_set_output(
            key=LINKSTATS,
            path=str(path),
            description="BEAM linkstats output for downstream runs",
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
            **linkstats_meta,
        )
        log_and_set_output(
            key=LINKSTATS_WARMSTART,
            path=str(path),
            description="BEAM warm-start linkstats for downstream runs",
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
            **linkstats_meta,
        )

    for short_name, path in outputs.iter_linkstats_parquet_outputs():
        linkstats_meta = _beam_log_facet_meta(short_name)
        if _is_beam_sub_iteration_key(short_name):
            log_output_only(
                key=short_name,
                path=str(path),
                description="BEAM linkstats parquet output for downstream runs",
                step_name="beam_run",
                profile_file_schema=True,
                **linkstats_meta,
            )
            continue
        log_and_set_output(
            key=short_name,
            path=str(path),
            description="BEAM linkstats parquet output for downstream runs",
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
            **linkstats_meta,
        )

    for short_name, path in outputs.iter_unmodified_phys_sim_outputs():
        record_meta = _beam_log_facet_meta(short_name)
        if _is_beam_sub_iteration_key(short_name):
            log_output_only(
                key=short_name,
                path=str(path),
                description=(
                    "BEAM unmodified linkstats parquet output for phys sim "
                    "sub-iteration"
                ),
                step_name="beam_run",
                profile_file_schema=True,
                **record_meta,
            )
            continue
        log_and_set_output(
            key=short_name,
            path=str(path),
            description=(
                "BEAM unmodified linkstats parquet output for phys sim sub-iteration"
            ),
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
            **record_meta,
        )

    promoted_plans = outputs.promoted_plans_for_publication()
    if promoted_plans is not None:
        _, path = promoted_plans
        log_and_set_output(
            key=BEAM_PLANS_OUT,
            path=str(path),
            description="BEAM plans output for downstream runs",
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
        )

    promoted_output_plans_xml = outputs.promoted_output_plans_xml_for_publication()
    if promoted_output_plans_xml is not None:
        _, path = promoted_output_plans_xml
        log_and_set_output(
            key=BEAM_OUTPUT_PLANS_XML,
            path=str(path),
            description="BEAM output plans XML for downstream warm-start reuse",
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
        )

    promoted_output_experienced_plans_xml = (
        outputs.promoted_output_experienced_plans_xml_for_publication()
    )
    if promoted_output_experienced_plans_xml is not None:
        _, path = promoted_output_experienced_plans_xml
        log_and_set_output(
            key=BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
            path=str(path),
            description=(
                "BEAM output experienced plans XML for downstream warm-start reuse"
            ),
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
        )

    promoted_experienced_plans_xml = (
        outputs.promoted_experienced_plans_xml_for_publication()
    )
    if promoted_experienced_plans_xml is not None:
        _, path = promoted_experienced_plans_xml
        log_and_set_output(
            key=BEAM_EXPERIENCED_PLANS_XML,
            path=str(path),
            description="BEAM experienced plans XML for downstream warm-start reuse",
            coupler=coupler,
            step_name="beam_run",
            profile_file_schema=True,
        )


def _execute_beam_preprocess(
    preprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    *,
    activity_demand_outputs: Optional[Dict[str, Any]] = None,
    previous_beam_outputs: Optional[Dict[str, Any]] = None,
    beam_preprocess_inputs: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> BeamPreprocessOutputs:
    return preprocessor.preprocess(
        workspace,
        activity_demand_outputs=activity_demand_outputs,
        previous_beam_outputs=previous_beam_outputs,
        beam_preprocess_inputs=beam_preprocess_inputs,
    )


def _execute_beam_run(
    runner: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    *,
    extra_inputs: Optional[Dict[str, Any]] = None,
    _consist_ctx: Any = None,
    **_: Any,
) -> BeamRunOutputs:
    upstream = outputs_holder.beam_preprocess
    if upstream is None:
        raise RuntimeError("BEAM preprocess must complete first")
    if not isinstance(upstream, BeamPreprocessOutputs):
        raise TypeError("beam_run requires BeamPreprocessOutputs from beam_preprocess")
    if isinstance(runner, BeamRunner):
        if _consist_ctx is None:
            raise RuntimeError(
                "beam_run requires the Consist run context for R5 validation."
            )
        validate_r5_execution_reference(
            settings=runner.state.full_settings,
            workspace=workspace,
            run_context=_consist_ctx,
        )
        staged_linkstats = upstream.prepared_inputs.get(LINKSTATS_WARMSTART)
        if staged_linkstats is not None:
            linkstats_reference = validate_staged_linkstats_reference(
                settings=runner.state.full_settings,
                workspace=workspace,
                run_context=_consist_ctx,
            )
            if linkstats_reference is None:
                raise RuntimeError(
                    "BEAM preprocess staged linkstats but final HOCON does not select it."
                )
            if (
                linkstats_reference.execution_path.resolve()
                != staged_linkstats.resolve()
            ):
                raise RuntimeError(
                    "BEAM final HOCON linkstats path differs from the staged warm-start "
                    f"input: {linkstats_reference.execution_path} != {staged_linkstats}"
                )
            tracker = cr.current_tracker()
            if tracker is None:
                raise RuntimeError("beam_run requires an active Consist tracker.")
            preflight_staged_linkstats_admission(
                tracker=tracker,
                settings=runner.state.full_settings,
                launch_reference=linkstats_reference,
                report_dir=_consist_ctx.run_dir,
            )
        else:
            reject_or_warn_for_missing_staged_linkstats(
                settings=runner.state.full_settings
            )
    return runner.run(
        upstream,
        workspace,
        extra_inputs=extra_inputs,
    )


def _execute_beam_postprocess(
    postprocessor: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    *,
    zarr_skims: Optional[Any] = None,
    **_: Any,
) -> BeamPostprocessOutputs:
    upstream = outputs_holder.beam_run
    if upstream is None:
        raise RuntimeError("BEAM run must complete first")
    if not isinstance(upstream, BeamRunOutputs):
        raise TypeError("beam_postprocess requires BeamRunOutputs from beam_run")
    if zarr_skims is not None:
        try:
            parameters = inspect.signature(postprocessor.postprocess).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_zarr_skims = "zarr_skims" in parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
        )
        if accepts_zarr_skims:
            return postprocessor.postprocess(
                upstream,
                workspace,
                zarr_skims=zarr_skims,
            )
    return postprocessor.postprocess(upstream, workspace)


def _execute_beam_full_skim(
    runner: Any,
    workspace: Workspace,
    outputs_holder: StepOutputsHolder,
    *,
    previous_beam_outputs: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> BeamFullSkimOutputs:
    upstream = outputs_holder.beam_preprocess
    if upstream is None:
        raise RuntimeError("BEAM preprocess must complete first")
    if not isinstance(upstream, BeamPreprocessOutputs):
        raise TypeError(
            "beam_full_skim requires BeamPreprocessOutputs from beam_preprocess"
        )
    return runner.run(
        upstream,
        workspace,
        previous_beam_outputs=previous_beam_outputs,
    )


_recover_beam_run_outputs = make_default_recoverer(
    outputs_class=BeamRunOutputs,
    mapping_field="raw_outputs",
    dir_field="beam_output_dir",
    dir_getter=lambda workspace: workspace.get_beam_output_dir(),
    step_logger=logger,
    log_context="BEAM cached output recovery",
)


def _recover_beam_preprocess_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[BeamPreprocessOutputs]:
    del settings, state, coupler, outputs_holder, run_id
    prepared_inputs: Dict[str, Path] = {}
    prepared_input_metadata: Dict[str, Dict[str, Any]] = {}
    if step_inputs:
        allowed_keys = {
            BEAM_CONFIG_FILE,
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
            LINKSTATS_WARMSTART,
            "vehicles_beam_in",
        }
        for key, value in step_inputs.items():
            if key not in allowed_keys:
                continue
            path = artifact_to_existing_path(
                value,
                workspace=workspace,
            )
            if path is not None:
                prepared_inputs[key] = Path(path)
    if not prepared_inputs:
        return None
    if cached_outputs:
        raw_metadata = cached_outputs.get("prepared_input_metadata")
        if isinstance(raw_metadata, Mapping):
            for key, value in raw_metadata.items():
                if key in prepared_inputs and isinstance(value, Mapping):
                    prepared_input_metadata[str(key)] = dict(value)
    return BeamPreprocessOutputs(
        beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
        prepared_inputs=prepared_inputs,
        prepared_input_metadata=prepared_input_metadata,
    )


def _recover_beam_postprocess_outputs(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    workspace: Workspace,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    step_inputs: Optional[Mapping[str, Any]],
    cached_outputs: Optional[Mapping[str, Any]],
    run_id: Optional[str],
) -> Optional[BeamPostprocessOutputs]:
    del settings, state, coupler, outputs_holder, step_inputs
    recovered_paths = recovered_cached_paths(
        cached_outputs=cached_outputs,
        run_id=run_id,
        workspace=workspace,
        step_logger=logger,
        log_context="BEAM cached output recovery",
    )
    if not recovered_paths:
        return None
    return BeamPostprocessOutputs(
        zarr_skims=recovered_paths.get("zarr_skims"),
        final_skims_omx=recovered_paths.get("final_skims_omx"),
        split_events={
            key: path
            for key, path in recovered_paths.items()
            if key.startswith("events_parquet_") and "_type_" in key
        },
        split_event_links={
            key: path
            for key, path in recovered_paths.items()
            if key.startswith("path_traversal_links_")
        },
    )


_recover_beam_full_skim_outputs = make_default_recoverer(
    outputs_class=BeamFullSkimOutputs,
    primary_path_field="full_skims",
    primary_path_resolver=lambda recovered_paths, _state: recovered_paths.get(
        "beam_full_skims"
    ),
    step_logger=logger,
    log_context="BEAM cached output recovery",
)


def _beam_step_runtime(ctx: Any) -> tuple[Any, Any, Any]:
    return (
        ctx.require_runtime("settings"),
        ctx.require_runtime("state"),
        ctx.require_runtime("workspace"),
    )


def _beam_preprocess_inputs(ctx: Any) -> Dict[str, Any]:
    from pilates.beam.preprocessor import BeamPreprocessor

    settings, state, workspace = _beam_step_runtime(ctx)
    return BeamPreprocessor.expected_inputs(settings, state, workspace)


def _beam_preprocess_output_paths(ctx: Any) -> Dict[str, Any]:
    from pilates.beam.preprocessor import BeamPreprocessor

    settings, state, workspace = _beam_step_runtime(ctx)
    return BeamPreprocessor.expected_outputs(settings, state, workspace)


def _beam_run_inputs(ctx: Any) -> Dict[str, Any]:
    from pilates.beam.runner import BeamRunner

    settings, state, workspace = _beam_step_runtime(ctx)
    inputs = dict(BeamRunner.runtime_expected_inputs(settings, state, workspace))
    # The BEAM input directory is a container mount, not a semantic cache input
    # for beam_run. Population inputs in that tree are logged as explicit
    # artifacts and the static network remains covered by the BEAM config adapter.
    inputs.pop(BEAM_MUTABLE_DATA_DIR, None)
    return inputs


def _beam_run_output_paths(ctx: Any) -> Dict[str, Any]:
    from pilates.beam.runner import BeamRunner

    settings, state, workspace = _beam_step_runtime(ctx)
    return BeamRunner.expected_outputs(settings, state, workspace)


def _beam_postprocess_inputs(ctx: Any) -> Dict[str, Any]:
    from pilates.beam.postprocessor import BeamPostprocessor

    settings, state, workspace = _beam_step_runtime(ctx)
    return BeamPostprocessor.expected_inputs(settings, state, workspace)


def _beam_postprocess_output_paths(ctx: Any) -> Dict[str, Any]:
    from pilates.beam.postprocessor import BeamPostprocessor

    settings, state, workspace = _beam_step_runtime(ctx)
    expected = BeamPostprocessor.expected_outputs(settings, state, workspace)
    if ZARR_SKIMS in expected:
        return {ZARR_SKIMS: expected[ZARR_SKIMS]}
    return {}


def _beam_full_skim_inputs(ctx: Any) -> Dict[str, Any]:
    from pilates.beam.runner import BeamFullSkimRunner

    settings, state, workspace = _beam_step_runtime(ctx)
    return BeamFullSkimRunner.expected_inputs(settings, state, workspace)


def _beam_full_skim_output_paths(ctx: Any) -> Dict[str, Any]:
    from pilates.beam.runner import BeamFullSkimRunner

    settings, state, workspace = _beam_step_runtime(ctx)
    return BeamFullSkimRunner.expected_outputs(settings, state, workspace)


def make_beam_preprocess_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the BEAM preprocess step function.

    This step builds the BEAM scenario inputs by transforming ActivitySim
    demand outputs, adding ATLAS vehicles (if enabled), and staging warm-start
    artifacts such as linkstats.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing preprocess outputs.

    Returns
    -------
    callable
        Step function for BEAM preprocess.

    Notes
    -----
    This step focuses on generating BEAM inputs and canonicalizing BEAM config.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        _require_primary_beam_config(settings, workspace)
        return {}

    def _log_outputs(
        outputs: BeamPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        """
        Log BEAM preprocess outputs and update the coupler.

        This helper logs prepared BEAM input artifacts into the coupler for
        downstream BEAM run and postprocess steps.

        Parameters
        ----------
        outputs : BeamPreprocessOutputs
            Typed outputs containing prepared BEAM inputs.
        settings : PilatesConfig
            Simulation settings for config root resolution.
        state : WorkflowState
            Current workflow state (used for log metadata only).
        workspace : Workspace
            Workspace used to resolve mutable BEAM config paths.
        holder : StepOutputsHolder
            Outputs holder (unused for this helper).
        """
        _log_step_records(
            record_items=(
                (
                    key,
                    path,
                    f"BEAM prepared input {key} for year {state.year}, iter {state.iteration}",
                )
                for key, path in outputs.prepared_inputs.items()
            ),
            log_fn=lambda key, path, description, **meta: log_and_set_output(
                key=key,
                path=path,
                description=description,
                coupler=coupler,
                step_name="beam_preprocess",
                **meta,
            ),
            profile_schema_keys={
                "plans_beam_in",
                "vehicles_beam_in",
            },
            extra_meta_fn=lambda key, _path, _description: (
                _beam_preprocess_input_publication_meta(
                    key=key,
                    state=state,
                    input_metadata=outputs.prepared_input_metadata.get(key, {}),
                )
            ),
        )

    step = build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="beam_preprocess",
            model_name="beam",
            phase="preprocess",
            outputs_class=BeamPreprocessOutputs,
            component_getter=lambda factory, state: factory.get_preprocessor(
                "beam", state
            ),
            component_executor=lambda component, workspace, outputs_holder, **kwargs: (
                _execute_beam_preprocess(
                    component,
                    workspace,
                    outputs_holder,
                    **kwargs,
                )
            ),
            input_logger=_log_inputs,
            output_logger=_log_outputs,
            output_recoverer=_recover_beam_preprocess_outputs,
            schema_outputs=_schema_outputs_from_class(BeamPreprocessOutputs),
            inputs=_beam_preprocess_inputs,
            output_paths=_beam_preprocess_output_paths,
            input_binding="paths",
            cache_hydration="metadata",
            use_logged_wrapper=False,
        ),
    )
    return step


def make_beam_run_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the BEAM run step function.

    This step performs the traffic assignment simulation for the current
    iteration and produces linkstats, skims, plans, and event outputs.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing run outputs.

    Returns
    -------
    callable
        Step function for BEAM run.
    """

    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        upstream = holder.beam_preprocess
        if upstream is None:
            raise RuntimeError("BEAM preprocess must complete first")

        config_path = _require_primary_beam_config(settings, workspace)
        log_input_only(
            key=BEAM_CONFIG_FILE,
            path=str(config_path),
            description="BEAM config file consumed by the BEAM run",
        )

        for short_name, path, description in upstream._iter_record_items():
            log_input_only(
                key=short_name,
                path=str(path),
                description=description,
            )

        plans_match, experienced_match = _resolve_beam_run_warmstart_inputs(
            settings=settings,
            workspace=workspace,
            coupler=coupler,
        )
        if plans_match is not None and Path(plans_match[1]).exists():
            log_input_only(
                key=plans_match[0],
                path=plans_match[1],
                description=(
                    "BEAM warm-start plans (selected by BEAM from previous outputs)"
                ),
            )
        if experienced_match is not None and Path(experienced_match[1]).exists():
            log_input_only(
                key=experienced_match[0],
                path=experienced_match[1],
                description=(
                    "BEAM warm-start experienced plans (selected by BEAM from previous outputs)"
                ),
            )
        return {}

    def _log_outputs(
        outputs: BeamRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        def _beam_run_extra_meta(
            short_name: str,
            _path: str,
            _description: str,
        ) -> Dict[str, Any]:
            meta: Dict[str, Any] = {}
            facet_meta = _beam_log_facet_meta(short_name)
            if facet_meta:
                meta.update(facet_meta)
            if short_name == BEAM_NETWORK_FINAL:
                meta.update(
                    {
                        "profile_file_schema": "if_changed",
                        "reuse_if_unchanged": True,
                        "reuse_scope": "any_uri",
                    }
                )
                beam_network_schema: Any = None
                try:
                    from pilates.database.schema.beam_schema import BeamNetworkFinal
                except Exception:
                    beam_network_schema = None
                else:
                    beam_network_schema = BeamNetworkFinal
                if beam_network_schema is not None:
                    meta["schema"] = beam_network_schema
            return meta

        _archive_beam_run_inputs(
            settings=settings,
            state=state,
            workspace=workspace,
            holder=holder,
            coupler=coupler,
        )

        _log_step_records(
            record_items=outputs._iter_record_items(),
            log_fn=lambda key, path, description, **meta: log_output_only(
                key=key,
                path=path,
                description=description,
                step_name="beam_run",
                **meta,
            ),
            extra_meta_fn=_beam_run_extra_meta,
        )

    step = build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="beam_run",
            model_name="beam",
            phase="run",
            outputs_class=BeamRunOutputs,
            component_getter=lambda factory, state: factory.get_runner("beam", state),
            component_executor=lambda component, workspace, outputs_holder, **kwargs: (
                _execute_beam_run(
                    component,
                    workspace,
                    outputs_holder,
                    **kwargs,
                )
            ),
            input_logger=_log_inputs,
            output_logger=_log_outputs,
            output_recoverer=_recover_beam_run_outputs,
            schema_outputs=_schema_outputs_from_class(BeamRunOutputs),
            inputs=_beam_run_inputs,
            output_paths=_beam_run_output_paths,
            input_binding="paths",
            inject_context="_consist_ctx",
            cache_hydration="inputs-missing",
            use_logged_wrapper=False,
        ),
    )
    return step


def make_beam_postprocess_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the BEAM postprocess step function.

    This step merges BEAM outputs into updated skims and produces final
    skim artifacts for ActivitySim and UrbanSim inputs.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing postprocess outputs.

    Returns
    -------
    callable
        Step function for BEAM postprocess.
    """

    def _log_outputs(
        outputs: BeamPostprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for short_name, path, description in outputs._iter_record_items():
            if (
                short_name == ZARR_SKIMS
                or short_name in outputs.split_events
                or short_name in outputs.split_event_links
            ):
                continue
            log_and_set_output(
                key=short_name,
                path=str(path),
                description=description,
                coupler=coupler,
                step_name="beam_postprocess",
            )
        if outputs.zarr_skims is not None:
            set_coupler_from_artifact(
                coupler,
                ZARR_SKIMS,
                None,
                fallback=str(outputs.zarr_skims),
            )
        for short_name, path in outputs.split_events.items():
            facet_meta = _beam_postprocess_split_facet_meta(short_name)
            log_output_only(
                key=short_name,
                path=str(path),
                description=f"BEAM events parquet split ({short_name})",
                step_name="beam_postprocess",
                profile_file_schema=True,
                **facet_meta,
            )
        for short_name, path in outputs.split_event_links.items():
            facet_meta = _beam_postprocess_split_facet_meta(short_name)
            log_output_only(
                key=short_name,
                path=str(path),
                description=f"BEAM events link table ({short_name})",
                step_name="beam_postprocess",
                profile_file_schema=True,
                **facet_meta,
            )
        upstream = holder.beam_run
        if upstream is None:
            return
        _publish_beam_run_outputs(outputs=upstream, coupler=coupler)

    step = build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="beam_postprocess",
            model_name="beam",
            phase="postprocess",
            outputs_class=BeamPostprocessOutputs,
            component_getter=lambda factory, state: factory.get_postprocessor(
                "beam", state
            ),
            component_executor=lambda component, workspace, outputs_holder, **kwargs: (
                _execute_beam_postprocess(
                    component,
                    workspace,
                    outputs_holder,
                    **kwargs,
                )
            ),
            declared_outputs=[ZARR_SKIMS],
            output_logger=_log_outputs,
            output_recoverer=_recover_beam_postprocess_outputs,
            schema_outputs=_schema_outputs_from_class(BeamPostprocessOutputs),
            inputs=_beam_postprocess_inputs,
            output_paths=_beam_postprocess_output_paths,
            input_binding="paths",
            cache_hydration="inputs-missing",
            use_logged_wrapper=False,
        ),
    )
    return step


def make_beam_full_skim_step(
    *,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the BEAM full-skim step function.

    This step runs BEAM's FullSkimsCreatorApp to produce background skims
    from prepared BEAM inputs and optional warm-start linkstats.
    """

    def _log_outputs(
        outputs: BeamFullSkimOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        for short_name, path, description in outputs._iter_record_items():
            set_coupler_from_artifact(coupler, short_name, None, fallback=str(path))

    step = build_standard_step(
        coupler=coupler,
        outputs_holder=outputs_holder,
        spec=StandardStepSpec(
            step_name="beam_full_skim",
            model_name="beam_full",
            phase="skim",
            outputs_class=BeamFullSkimOutputs,
            component_getter=lambda factory, state: factory.get_runner(
                "beam_full_skim", state
            ),
            component_executor=lambda component, workspace, outputs_holder, **kwargs: (
                _execute_beam_full_skim(
                    component,
                    workspace,
                    outputs_holder,
                    **kwargs,
                )
            ),
            output_logger=_log_outputs,
            output_recoverer=_recover_beam_full_skim_outputs,
            schema_outputs=_schema_outputs_from_class(BeamFullSkimOutputs),
            inputs=_beam_full_skim_inputs,
            output_paths=_beam_full_skim_output_paths,
            input_binding="paths",
            cache_hydration="inputs-missing",
            use_logged_wrapper=False,
        ),
    )
    return step


# Native Consist step definitions -------------------------------------------------
#
# These values deliberately sit beside the legacy factories until the coordinated
# stage cutover.  They do not capture a holder or coupler: all semantic selection is
# completed by their resolver and all declared outputs are persisted by Consist.


def _path_from_output(
    *,
    outputs: Mapping[str, Any],
    step_name: str,
    key: str,
    declared_outputs: Mapping[str, Any],
    workspace: Any,
) -> Path:
    """Return one output from the current invocation's declared path map.

    Cache-hit artifacts retain their original URI by design.  The typed
    projector must therefore validate the deterministic current destination
    requested for this invocation rather than consulting artifact metadata.
    """

    require_output(outputs, step_name=step_name, key=key)
    try:
        destination = declared_outputs[key]
    except KeyError as exc:
        raise RuntimeError(
            f"{step_name} output {key!r} has no declared current destination."
        ) from exc
    if isinstance(destination, os.PathLike):
        path = Path(destination)
    elif isinstance(destination, str) and destination.startswith("workspace://"):
        workspace_root = getattr(workspace, "full_path", None)
        if workspace_root is None:
            raise RuntimeError(
                f"{step_name} output {key!r} cannot resolve its workspace destination."
            )
        path = Path(workspace_root) / destination[len("workspace://") :].lstrip("/")
    elif isinstance(destination, str) and "://" in destination:
        raise RuntimeError(
            f"{step_name} output {key!r} has a non-local declared destination."
        )
    elif isinstance(destination, str):
        path = Path(destination)
        if not path.is_absolute():
            workspace_root = getattr(workspace, "full_path", None)
            if workspace_root is None:
                raise RuntimeError(
                    f"{step_name} output {key!r} cannot resolve its relative destination."
                )
            path = Path(workspace_root) / path
    else:
        raise RuntimeError(
            f"{step_name} output {key!r} has an invalid declared destination."
        )

    if not path.exists():
        raise RuntimeError(
            f"{step_name} output {key!r} is missing at declared destination {path}."
        )
    return path


def _native_output_destination(
    *, root: Path, step_name: str, key: str, suffix: str
) -> Path:
    """Return the individually keyed current destination for one BEAM output."""

    return root / ".pilates-consist-outputs" / step_name / f"{key}{suffix}"


def _beam_preprocess_native_output_paths(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> dict[str, Path]:
    del settings, state
    selected = (
        tuple(resolved_inputs.metadata.get("native_output_keys", ()))
        if resolved_inputs is not None
        else (
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
            LINKSTATS_WARMSTART,
            "vehicles_beam_in",
        )
    )
    suffixes = {
        BEAM_PLANS_IN: ".csv",
        BEAM_HOUSEHOLDS_IN: ".csv",
        BEAM_PERSONS_IN: ".csv",
        LINKSTATS_WARMSTART: ".csv.gz",
        "vehicles_beam_in": ".csv",
    }
    root = Path(workspace.get_beam_mutable_data_dir())
    return {
        key: _native_output_destination(
            root=root,
            step_name="beam_preprocess",
            key=key,
            suffix=suffixes.get(key, ""),
        )
        for key in selected
    }


def _beam_run_native_output_paths(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> dict[str, Path]:
    del settings, resolved_inputs
    year = resolve_forecast_year(state)
    if year is None:
        raise RuntimeError("beam_run requires a resolved forecast year.")
    iteration = int(state.iteration)
    keys_and_suffixes = {
        LINKSTATS: ".csv.gz",
        BEAM_PLANS_OUT: ".csv.gz",
        f"raw_od_skims_{year}_{iteration}": ".omx",
        f"raw_od_skims_zarr_{year}_{iteration}": ".zarr",
        f"events_parquet_{year}_{iteration}": ".parquet",
    }
    root = Path(workspace.get_beam_output_dir())
    return {
        key: _native_output_destination(
            root=root,
            step_name="beam_run",
            key=key,
            suffix=suffix,
        )
        for key, suffix in keys_and_suffixes.items()
    }


def _beam_postprocess_native_output_paths(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> dict[str, Path]:
    from pilates.beam.postprocessor import BeamPostprocessor

    static_outputs = {
        key: Path(path)
        for key, path in BeamPostprocessor.expected_outputs(
            settings, state, workspace
        ).items()
        if path is not None
    }
    if resolved_inputs is None:
        return static_outputs
    dynamic_outputs = resolved_inputs.metadata.get("beam_postprocess_output_paths", {})
    if not isinstance(dynamic_outputs, Mapping):
        raise RuntimeError(
            "beam_postprocess resolved output paths must be a key-to-path mapping."
        )
    duplicate_keys = set(static_outputs).intersection(dynamic_outputs)
    if duplicate_keys:
        raise RuntimeError(
            "beam_postprocess resolved output keys overlap static outputs: "
            + ", ".join(sorted(duplicate_keys))
        )
    resolved_output_paths = {
        **static_outputs,
        **{key: Path(path) for key, path in dynamic_outputs.items()},
    }
    if len(set(resolved_output_paths.values())) != len(resolved_output_paths):
        raise RuntimeError("beam_postprocess resolved output paths are not injective.")
    return resolved_output_paths


def _beam_full_skim_native_output_paths(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs | None = None,
) -> dict[str, Path]:
    del resolved_inputs
    from pilates.beam.runner import BeamFullSkimRunner

    return {
        key: Path(path)
        for key, path in BeamFullSkimRunner.expected_outputs(
            settings, state, workspace
        ).items()
        if path is not None
    }


def _materialize_native_outputs(
    *,
    source_paths: Mapping[str, Path],
    declared_outputs: Mapping[str, Path],
) -> None:
    """Copy selected semantic outputs to their declared individual paths."""

    for key, destination in declared_outputs.items():
        source = source_paths.get(key)
        if source is None:
            continue
        source = Path(source)
        if not source.exists():
            raise RuntimeError(
                f"BEAM native output {key!r} is missing before declared output logging: "
                f"{source}."
            )
        if source.resolve() == destination.resolve():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        elif source.resolve() != destination.resolve():
            shutil.copy2(source, destination)


def _beam_run_native_sources(produced: BeamRunOutputs) -> dict[str, Path]:
    """Select the closed BEAM run semantic surface from raw runner outputs."""

    sources = dict(produced.raw_outputs)
    latest_linkstats = produced._latest_raw_output_for_prefix(LINKSTATS)
    if latest_linkstats is None:
        latest_linkstats = produced._latest_raw_output_for_prefix("linkstats_parquet")
    if latest_linkstats is not None:
        sources[LINKSTATS] = latest_linkstats[1]
    latest_plans = produced._latest_raw_output_for_prefix(BEAM_PLANS_OUT)
    if latest_plans is not None:
        sources[BEAM_PLANS_OUT] = latest_plans[1]
    return sources


def _beam_postprocess_native_sources(
    produced: BeamPostprocessOutputs,
) -> dict[str, Path]:
    """Return every typed BEAM postprocess output by its semantic key."""

    sources = {
        **produced.split_events,
        **produced.split_event_links,
    }
    if produced.zarr_skims is not None:
        sources[ZARR_SKIMS] = produced.zarr_skims
    if produced.final_skims_omx is not None:
        sources["final_skims_omx"] = produced.final_skims_omx
    return sources


def _validate_native_outputs(
    outputs: Any,
    *,
    step_name: str,
    settings: Any,
    state: Any,
    workspace: Any,
) -> None:
    outputs.validate(
        context=ValidationContext(
            settings=settings,
            state=state,
            workspace=workspace,
            step_name=step_name,
        )
    )


def _project_beam_preprocess_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs,
) -> BeamPreprocessOutputs:
    declared_outputs = _beam_preprocess_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    prepared = {
        key: _path_from_output(
            outputs=outputs,
            step_name="beam_preprocess",
            key=key,
            declared_outputs=declared_outputs,
            workspace=workspace,
        )
        for key in (
            *BeamPreprocessOutputs.required_output_keys(),
            "vehicles_beam_in",
            LINKSTATS_WARMSTART,
        )
        if key in outputs and key in declared_outputs
    }
    projected = BeamPreprocessOutputs(
        beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
        prepared_inputs=prepared,
    )
    _validate_native_outputs(
        projected,
        step_name="beam_preprocess",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    return projected


def _project_beam_run_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs,
) -> BeamRunOutputs:
    declared_outputs = _beam_run_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    raw_outputs = {
        key: _path_from_output(
            outputs=outputs,
            step_name="beam_run",
            key=key,
            declared_outputs=declared_outputs,
            workspace=workspace,
        )
        for key in outputs
        if key in declared_outputs
    }
    projected = BeamRunOutputs(
        beam_output_dir=Path(workspace.get_beam_output_dir()),
        raw_outputs=raw_outputs,
    )
    _validate_native_outputs(
        projected,
        step_name="beam_run",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    return projected


def _project_beam_postprocess_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs,
) -> BeamPostprocessOutputs:
    dynamic_output_keys = {
        key
        for key in outputs
        if (
            key.startswith("events_parquet_")
            and "_type_" in key
            or key.startswith("path_traversal_links_")
        )
    }
    resolved_dynamic_outputs = resolved_inputs.metadata.get(
        "beam_postprocess_output_paths", {}
    )
    if not isinstance(resolved_dynamic_outputs, Mapping):
        raise RuntimeError(
            "beam_postprocess resolved output paths must be a key-to-path mapping."
        )
    if dynamic_output_keys != set(resolved_dynamic_outputs):
        raise RuntimeError(
            "beam_postprocess persisted typed output keys differ from its resolved "
            "closed output map: expected "
            f"{sorted(resolved_dynamic_outputs)}, got {sorted(dynamic_output_keys)}."
        )
    declared_outputs = _beam_postprocess_native_output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved_inputs,
    )
    projected = BeamPostprocessOutputs(
        zarr_skims=(
            _path_from_output(
                outputs=outputs,
                step_name="beam_postprocess",
                key=ZARR_SKIMS,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            if ZARR_SKIMS in outputs and ZARR_SKIMS in declared_outputs
            else None
        ),
        final_skims_omx=(
            _path_from_output(
                outputs=outputs,
                step_name="beam_postprocess",
                key="final_skims_omx",
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            if "final_skims_omx" in outputs and "final_skims_omx" in declared_outputs
            else None
        ),
        split_events={
            key: _path_from_output(
                outputs=outputs,
                step_name="beam_postprocess",
                key=key,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            for key in outputs
            if key in declared_outputs
            and key.startswith("events_parquet_")
            and "_type_" in key
        },
        split_event_links={
            key: _path_from_output(
                outputs=outputs,
                step_name="beam_postprocess",
                key=key,
                declared_outputs=declared_outputs,
                workspace=workspace,
            )
            for key in outputs
            if key in declared_outputs and key.startswith("path_traversal_links_")
        },
    )
    _validate_native_outputs(
        projected,
        step_name="beam_postprocess",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    return projected


def _project_beam_full_skim_outputs(
    outputs: Mapping[str, Any],
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs,
) -> BeamFullSkimOutputs:
    declared_outputs = _beam_full_skim_native_output_paths(
        settings=settings, state=state, workspace=workspace
    )
    projected = BeamFullSkimOutputs(
        full_skims=_path_from_output(
            outputs=outputs,
            step_name="beam_full_skim",
            key="beam_full_skims",
            declared_outputs=declared_outputs,
            workspace=workspace,
        )
    )
    _validate_native_outputs(
        projected,
        step_name="beam_full_skim",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    return projected


def _input_destination(*, workspace: Any, key: str, source: Any) -> Path:
    source_path = artifact_to_path(source, workspace=workspace)
    suffixes = "".join(Path(source_path).suffixes) if source_path is not None else ""
    return (
        Path(workspace.get_beam_mutable_data_dir())
        / ".consist-inputs"
        / f"{key}{suffixes}"
    )


def _resolved_beam_inputs(
    *,
    step_name: str,
    coupler: Any,
    workspace: Any,
    required_roles: Iterable[str],
    optional_roles: Iterable[str] = (),
    explicit_inputs: Mapping[str, Any] | None = None,
    logical_destinations: Mapping[str, Path] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ResolvedStepInputs:
    inputs: dict[str, Any] = dict(explicit_inputs or {})
    source_by_role = {key: "explicit" for key in inputs}
    selected_key_by_role = {key: key for key in inputs}
    destinations = dict(logical_destinations or {})
    for key in (*required_roles, *optional_roles):
        if key in inputs:
            continue
        value = coupler.get(key)
        if value is None:
            source_by_role[key] = "missing"
            continue
        inputs[key] = value
        source_by_role[key] = "coupler"
        selected_key_by_role[key] = key
        destinations.setdefault(
            key,
            _input_destination(workspace=workspace, key=key, source=value),
        )
    return ResolvedStepInputs(
        step_name=step_name,
        binding=BindingResult(inputs=inputs),
        required_roles=tuple(required_roles),
        optional_roles=tuple(optional_roles),
        source_by_role=source_by_role,
        selected_key_by_role=selected_key_by_role,
        logical_destinations=destinations,
        metadata=metadata or {},
    )


def _native_execution_options(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    resolved_inputs: ResolvedStepInputs,
) -> ExecutionOptions:
    del settings, state, workspace
    runtime_kwargs: dict[str, Any] = {}
    if "beam_postprocess_dynamic_paths" in resolved_inputs.metadata:
        runtime_kwargs["beam_run_dynamic_paths"] = dict(
            resolved_inputs.metadata["beam_postprocess_dynamic_paths"]
        )
    if "beam_postprocess_output_paths" in resolved_inputs.metadata:
        runtime_kwargs["beam_postprocess_output_paths"] = dict(
            resolved_inputs.metadata["beam_postprocess_output_paths"]
        )
    return ExecutionOptions(
        input_binding="paths",
        input_materialization="requested",
        input_paths=resolved_inputs.logical_destinations,
        runtime_kwargs=runtime_kwargs,
        inject_context="_consist_ctx",
    )


def _native_contract_output_paths(
    provider: Callable[..., Mapping[str, Any]],
) -> Callable[[Any], Mapping[str, Any]]:
    def resolve(context: Any) -> Mapping[str, Any]:
        settings = context.get_runtime("settings", default=None)
        state = context.get_runtime("state", default=None)
        workspace = context.get_runtime("workspace", default=None)
        if settings is None or state is None or workspace is None:
            return {}
        return provider(settings=settings, state=state, workspace=workspace)

    return resolve


def _strict_requested_output_cache(
    *, settings: Any, state: Any, workspace: Any
) -> CacheOptions:
    del settings, state, workspace
    return CacheOptions(
        cache_hydration="outputs-requested",
        cache_hydration_failure="miss",
    )


def _resolve_beam_preprocess_inputs(
    *, settings: Any, state: Any, workspace: Any, coupler: Any
) -> ResolvedStepInputs:
    config_path = _require_primary_beam_config(settings, workspace)
    resolved = _resolved_beam_inputs(
        step_name="beam_preprocess",
        coupler=coupler,
        workspace=workspace,
        required_roles=(
            BEAM_CONFIG_FILE,
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
        ),
        optional_roles=(LINKSTATS_WARMSTART, ATLAS_VEHICLES2_OUTPUT),
        explicit_inputs={BEAM_CONFIG_FILE: config_path},
        logical_destinations={BEAM_CONFIG_FILE: config_path},
    )
    return ResolvedStepInputs(
        step_name=resolved.step_name,
        binding=resolved.binding,
        required_roles=resolved.required_roles,
        optional_roles=resolved.optional_roles,
        source_by_role=resolved.source_by_role,
        selected_key_by_role=resolved.selected_key_by_role,
        logical_destinations=resolved.logical_destinations,
        metadata={
            "native_output_keys": tuple(
                key
                for key in (
                    BEAM_PLANS_IN,
                    BEAM_HOUSEHOLDS_IN,
                    BEAM_PERSONS_IN,
                    LINKSTATS_WARMSTART,
                    "vehicles_beam_in",
                )
                if key in (resolved.binding.inputs or {})
            )
        },
    )


def _resolve_beam_run_inputs(
    *, settings: Any, state: Any, workspace: Any, coupler: Any
) -> ResolvedStepInputs:
    config_path = _require_primary_beam_config(settings, workspace)
    return _resolved_beam_inputs(
        step_name="beam_run",
        coupler=coupler,
        workspace=workspace,
        required_roles=(
            BEAM_CONFIG_FILE,
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
        ),
        optional_roles=(LINKSTATS_WARMSTART, ZARR_SKIMS),
        explicit_inputs={BEAM_CONFIG_FILE: config_path},
        logical_destinations={BEAM_CONFIG_FILE: config_path},
    )


def _postprocess_dynamic_keys(
    *, coupler: Any, year: int, iteration: int
) -> tuple[str, ...]:
    keys = tuple(coupler.keys())
    selected = [
        key
        for key in keys
        if key.startswith(
            (f"events_parquet_{year}_{iteration}", f"raw_od_skims_{year}_{iteration}")
        )
    ]
    if not any(key.startswith("events_parquet_") for key in selected):
        selected.extend(key for key in keys if key.startswith("events_parquet_"))
    if not any(key.startswith("raw_od_skims") for key in selected):
        selected.extend(key for key in keys if key.startswith("raw_od_skims"))
    return tuple(dict.fromkeys(selected))


def _selected_postprocess_events_key(
    *, dynamic_keys: Iterable[str], year: int, iteration: int
) -> str | None:
    """Choose the exact events input consumed by ``BeamPostprocessor``.

    This mirrors the postprocessor's identity-bearing event selection: the
    canonical iteration key wins, otherwise its highest numeric sub-iteration
    is selected.  Older/fallback event keys remain valid inputs for other
    postprocess work but must not expand this invocation's output contract.
    """

    target = f"events_parquet_{year}_{iteration}"
    keys = tuple(dynamic_keys)
    if target in keys:
        return target
    selected: str | None = None
    selected_sub_iteration = -1
    for key in keys:
        if not key.startswith(f"{target}_sub"):
            continue
        suffix = key[len(f"{target}_sub") :]
        try:
            sub_iteration = int(suffix)
        except ValueError:
            continue
        if sub_iteration > selected_sub_iteration:
            selected = key
            selected_sub_iteration = sub_iteration
    return selected


def _beam_postprocess_split_output_paths(
    *,
    selected_events_key: str | None,
    inputs: Mapping[str, Any],
    year: int,
    iteration: int,
    workspace: Any,
) -> dict[str, Path]:
    """Close typed split outputs from the selected, locally readable event input.

    Event types are data-dependent, so the one semantic event input selected
    for this invocation is the only authority for the exact keyed output map.
    Refuse to run when that source cannot be inspected; accepting a partial
    output map would make a cache hit semantically different from a fresh run.
    """

    if selected_events_key is None:
        return {}
    source = inputs.get(selected_events_key)
    source_path = artifact_to_path(source, workspace=workspace)
    if source_path is None or "://" in source_path or not Path(source_path).is_file():
        raise RuntimeError(
            "beam_postprocess cannot inspect selected events input "
            f"{selected_events_key!r} to close typed outputs."
        )
    try:
        import pandas as pd

        event_types = pd.read_parquet(source_path, columns=["type"])["type"]
    except Exception as exc:
        raise RuntimeError(
            "beam_postprocess cannot inspect event types for selected input "
            f"{selected_events_key!r}: {source_path}."
        ) from exc
    if event_types.empty:
        return {}
    event_types = sorted({str(value) for value in event_types.dropna()})
    root = Path(workspace.get_beam_output_dir())
    event_keys = tuple(
        f"events_parquet_{year}_{iteration}_type_{_sanitize_beam_event_type(event_type)}"
        for event_type in event_types
    )
    if len(set(event_keys)) != len(event_keys):
        raise RuntimeError(
            "beam_postprocess selected event types do not map to injective "
            f"semantic output keys: {event_types}."
        )
    output_paths = {
        key: _native_output_destination(
            root=root,
            step_name="beam_postprocess",
            key=f"events_parquet_{year}_{iteration}_type_{_sanitize_beam_event_type(event_type)}",
            suffix=".parquet",
        )
        for key, event_type in zip(event_keys, event_types, strict=True)
    }
    if "PathTraversal" in event_types:
        key = f"path_traversal_links_{year}_{iteration}"
        output_paths[key] = _native_output_destination(
            root=root,
            step_name="beam_postprocess",
            key=key,
            suffix=".parquet",
        )
    if len(set(output_paths.values())) != len(output_paths):
        raise RuntimeError(
            "beam_postprocess resolved typed output paths are not injective."
        )
    return output_paths


def _sanitize_beam_event_type(event_type: str) -> str:
    """Match the postprocessor's semantic event-type key normalization."""

    safe = re.sub(r"[^A-Za-z0-9]+", "_", event_type).strip("_")
    return safe or "unknown"


def _postprocess_destination(*, key: str, workspace: Any, iteration: int) -> Path:
    output_dir = Path(workspace.get_beam_output_dir())
    if key.startswith("events_parquet_"):
        return output_dir / ".pilates-consist-inputs" / f"{key}.parquet"
    if key.startswith("raw_od_skims_zarr"):
        return output_dir / ".pilates-consist-inputs" / f"{key}.zarr"
    if key.startswith("raw_od_skims"):
        return output_dir / ".pilates-consist-inputs" / f"{key}.omx"
    raise RuntimeError(
        f"beam_postprocess has no deterministic destination for {key!r}."
    )


def _resolve_beam_postprocess_inputs(
    *, settings: Any, state: Any, workspace: Any, coupler: Any
) -> ResolvedStepInputs:
    year = resolve_forecast_year(state)
    if year is None:
        raise RuntimeError("beam_postprocess requires a resolved forecast year.")
    iteration = state.iteration
    dynamic_keys = _postprocess_dynamic_keys(
        coupler=coupler,
        year=int(year),
        iteration=int(iteration),
    )
    destinations = {
        key: _postprocess_destination(
            key=key, workspace=workspace, iteration=int(iteration)
        )
        for key in dynamic_keys
    }
    if settings.activitysim is not None and coupler.get(ZARR_SKIMS) is not None:
        destinations[ZARR_SKIMS] = (
            Path(workspace.get_asim_output_dir()) / "cache" / "skims.zarr"
        )
        optional_roles = (ZARR_SKIMS,)
    else:
        optional_roles = ()
    dynamic_paths = {key: destinations[key] for key in dynamic_keys}
    resolved = _resolved_beam_inputs(
        step_name="beam_postprocess",
        coupler=coupler,
        workspace=workspace,
        required_roles=dynamic_keys,
        optional_roles=optional_roles,
        logical_destinations=destinations,
        metadata={"beam_postprocess_dynamic_paths": dynamic_paths},
    )
    selected_events_key = _selected_postprocess_events_key(
        dynamic_keys=dynamic_keys,
        year=int(year),
        iteration=int(iteration),
    )
    split_outputs = _beam_postprocess_split_output_paths(
        selected_events_key=selected_events_key,
        inputs=resolved.binding.inputs or {},
        year=int(year),
        iteration=int(iteration),
        workspace=workspace,
    )
    return ResolvedStepInputs(
        step_name=resolved.step_name,
        binding=resolved.binding,
        required_roles=resolved.required_roles,
        optional_roles=resolved.optional_roles,
        source_by_role=resolved.source_by_role,
        selected_key_by_role=resolved.selected_key_by_role,
        logical_destinations=resolved.logical_destinations,
        metadata={
            **resolved.metadata,
            "beam_postprocess_output_paths": MappingProxyType(split_outputs),
        },
    )


def _resolve_beam_full_skim_inputs(
    *, settings: Any, state: Any, workspace: Any, coupler: Any
) -> ResolvedStepInputs:
    return _resolved_beam_inputs(
        step_name="beam_full_skim",
        coupler=coupler,
        workspace=workspace,
        required_roles=(BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN),
        optional_roles=(LINKSTATS_WARMSTART,),
    )


def _log_native_output_records(*, outputs: Any, context: Any) -> None:
    logged_keys: set[str] = set()
    for key, path, _description in outputs._iter_record_items():
        if key in logged_keys:
            continue
        logged_keys.add(key)
        context.log_output(
            path,
            key=key,
            artifact_kind="directory" if Path(path).is_dir() else "file",
        )


@define_step(
    model="beam_preprocess",
    name_template="beam_preprocess__y{year}__i{iteration}__phase_{phase}",
    inputs={
        BEAM_CONFIG_FILE: None,
        BEAM_PLANS_IN: None,
        BEAM_HOUSEHOLDS_IN: None,
        BEAM_PERSONS_IN: None,
    },
    optional_input_keys=(LINKSTATS_WARMSTART, ATLAS_VEHICLES2_OUTPUT),
    schema_outputs=[
        BEAM_PLANS_IN,
        BEAM_HOUSEHOLDS_IN,
        BEAM_PERSONS_IN,
        LINKSTATS_WARMSTART,
        "vehicles_beam_in",
    ],
    output_paths=_native_contract_output_paths(_beam_preprocess_native_output_paths),
    input_binding="paths",
    **consist_step_meta("beam_preprocess"),
)
def _native_beam_preprocess(
    beam_config_file: Path,
    plans_beam_in: Path,
    households_beam_in: Path,
    persons_beam_in: Path,
    linkstats_warmstart: Path | None = None,
    atlas_vehicles2_output: Path | None = None,
    *,
    settings: Any,
    state: Any,
    workspace: Workspace,
    _consist_ctx: Any,
) -> None:
    if not beam_config_file.exists():
        raise FileNotFoundError(
            f"beam_preprocess config is missing: {beam_config_file}"
        )
    from pilates.beam.preprocessor import BeamPreprocessor

    inputs = {
        BEAM_PLANS_IN: plans_beam_in,
        BEAM_HOUSEHOLDS_IN: households_beam_in,
        BEAM_PERSONS_IN: persons_beam_in,
    }
    if linkstats_warmstart is not None:
        inputs[LINKSTATS_WARMSTART] = linkstats_warmstart
    if atlas_vehicles2_output is not None:
        inputs[ATLAS_VEHICLES2_OUTPUT] = atlas_vehicles2_output
    produced = BeamPreprocessor("beam_preprocess", state).preprocess(
        workspace,
        beam_preprocess_inputs=inputs,
    )
    _validate_native_outputs(
        produced,
        step_name="beam_preprocess",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    _materialize_native_outputs(
        source_paths=produced.prepared_inputs,
        declared_outputs=_beam_preprocess_native_output_paths(
            settings=settings,
            state=state,
            workspace=workspace,
            resolved_inputs=ResolvedStepInputs(
                step_name="beam_preprocess",
                binding=BindingResult(inputs=inputs),
                metadata={"native_output_keys": tuple(produced.prepared_inputs)},
            ),
        ),
    )
    del _consist_ctx


@define_step(
    model="beam_run",
    name_template="beam_run__y{year}__i{iteration}__phase_{phase}",
    inputs={
        BEAM_CONFIG_FILE: None,
        BEAM_PLANS_IN: None,
        BEAM_HOUSEHOLDS_IN: None,
        BEAM_PERSONS_IN: None,
    },
    optional_input_keys=(LINKSTATS_WARMSTART, ZARR_SKIMS),
    schema_outputs=[LINKSTATS, BEAM_PLANS_OUT],
    output_paths=_native_contract_output_paths(_beam_run_native_output_paths),
    input_binding="paths",
    **consist_step_meta("beam_run"),
)
def _native_beam_run(
    beam_config_file: Path,
    plans_beam_in: Path,
    households_beam_in: Path,
    persons_beam_in: Path,
    linkstats_warmstart: Path | None = None,
    zarr_skims: Path | None = None,
    *,
    settings: Any,
    state: Any,
    workspace: Workspace,
    _consist_ctx: Any,
) -> None:
    if not beam_config_file.exists():
        raise FileNotFoundError(f"beam_run config is missing: {beam_config_file}")
    prepared = {
        BEAM_PLANS_IN: plans_beam_in,
        BEAM_HOUSEHOLDS_IN: households_beam_in,
        BEAM_PERSONS_IN: persons_beam_in,
    }
    if linkstats_warmstart is not None:
        prepared[LINKSTATS_WARMSTART] = linkstats_warmstart
    produced = BeamRunner("beam_run", state).run(
        BeamPreprocessOutputs(
            beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
            prepared_inputs=prepared,
        ),
        workspace,
        extra_inputs={ZARR_SKIMS: zarr_skims} if zarr_skims is not None else None,
    )
    _validate_native_outputs(
        produced,
        step_name="beam_run",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    _materialize_native_outputs(
        source_paths=_beam_run_native_sources(produced),
        declared_outputs=_beam_run_native_output_paths(
            settings=settings, state=state, workspace=workspace
        ),
    )
    del _consist_ctx


@define_step(
    model="beam_postprocess",
    name_template="beam_postprocess__y{year}__i{iteration}__phase_{phase}",
    optional_input_keys=(ZARR_SKIMS,),
    schema_outputs=[ZARR_SKIMS, "final_skims_omx"],
    output_paths=_native_contract_output_paths(_beam_postprocess_native_output_paths),
    input_binding="paths",
    **consist_step_meta("beam_postprocess"),
)
def _native_beam_postprocess(
    zarr_skims: Path | None = None,
    *,
    beam_run_dynamic_paths: Mapping[str, Path],
    beam_postprocess_output_paths: Mapping[str, Path],
    settings: Any,
    state: Any,
    workspace: Workspace,
    _consist_ctx: Any,
) -> None:
    from pilates.beam.postprocessor import BeamPostprocessor

    produced = BeamPostprocessor("beam_postprocess", state).postprocess(
        BeamRunOutputs(
            beam_output_dir=Path(workspace.get_beam_output_dir()),
            raw_outputs=dict(beam_run_dynamic_paths),
        ),
        workspace,
        zarr_skims=zarr_skims,
    )
    _validate_native_outputs(
        produced,
        step_name="beam_postprocess",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    declared_outputs = _beam_postprocess_native_output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
    )
    overlap = set(declared_outputs).intersection(beam_postprocess_output_paths)
    if overlap:
        raise RuntimeError(
            "beam_postprocess resolved output keys overlap static outputs: "
            + ", ".join(sorted(overlap))
        )
    declared_outputs.update(
        {key: Path(path) for key, path in beam_postprocess_output_paths.items()}
    )
    produced_sources = _beam_postprocess_native_sources(produced)
    expected_dynamic_keys = set(beam_postprocess_output_paths)
    produced_dynamic_keys = {
        key
        for key in produced_sources
        if (
            key.startswith("events_parquet_")
            and "_type_" in key
            or key.startswith("path_traversal_links_")
        )
    }
    if produced_dynamic_keys != expected_dynamic_keys:
        raise RuntimeError(
            "beam_postprocess produced typed output keys differ from its resolved "
            "closed output map: expected "
            f"{sorted(expected_dynamic_keys)}, got {sorted(produced_dynamic_keys)}."
        )
    _materialize_native_outputs(
        source_paths=produced_sources,
        declared_outputs=declared_outputs,
    )
    del _consist_ctx


@define_step(
    model="beam_full_skim",
    name_template="beam_full_skim__y{year}__i{iteration}__phase_{phase}",
    inputs={
        BEAM_PLANS_IN: None,
        BEAM_HOUSEHOLDS_IN: None,
        BEAM_PERSONS_IN: None,
    },
    optional_input_keys=(LINKSTATS_WARMSTART,),
    schema_outputs=["beam_full_skims"],
    output_paths=_native_contract_output_paths(_beam_full_skim_native_output_paths),
    input_binding="paths",
    **consist_step_meta("beam_full_skim"),
)
def _native_beam_full_skim(
    plans_beam_in: Path,
    households_beam_in: Path,
    persons_beam_in: Path,
    linkstats_warmstart: Path | None = None,
    *,
    settings: Any,
    state: Any,
    workspace: Workspace,
    _consist_ctx: Any,
) -> None:
    prepared = {
        BEAM_PLANS_IN: plans_beam_in,
        BEAM_HOUSEHOLDS_IN: households_beam_in,
        BEAM_PERSONS_IN: persons_beam_in,
    }
    if linkstats_warmstart is not None:
        prepared[LINKSTATS_WARMSTART] = linkstats_warmstart
    from pilates.beam.runner import BeamFullSkimRunner

    produced = BeamFullSkimRunner("beam_full_skim", state).run(
        BeamPreprocessOutputs(
            beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
            prepared_inputs=prepared,
        ),
        workspace,
    )
    _validate_native_outputs(
        produced,
        step_name="beam_full_skim",
        settings=settings,
        state=state,
        workspace=workspace,
    )
    del _consist_ctx


beam_preprocess = StepDefinition(
    name="beam_preprocess",
    function=_native_beam_preprocess,
    resolve_inputs=_resolve_beam_preprocess_inputs,
    project_outputs=_project_beam_preprocess_outputs,
    output_paths=_beam_preprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache,
)
beam_run = StepDefinition(
    name="beam_run",
    function=_native_beam_run,
    resolve_inputs=_resolve_beam_run_inputs,
    project_outputs=_project_beam_run_outputs,
    output_paths=_beam_run_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache,
)
beam_postprocess = StepDefinition(
    name="beam_postprocess",
    function=_native_beam_postprocess,
    resolve_inputs=_resolve_beam_postprocess_inputs,
    project_outputs=_project_beam_postprocess_outputs,
    output_paths=_beam_postprocess_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache,
)
beam_full_skim = StepDefinition(
    name="beam_full_skim",
    function=_native_beam_full_skim,
    resolve_inputs=_resolve_beam_full_skim_inputs,
    project_outputs=_project_beam_full_skim_outputs,
    output_paths=_beam_full_skim_native_output_paths,
    execution_options=_native_execution_options,
    cache_options=_strict_requested_output_cache,
)
