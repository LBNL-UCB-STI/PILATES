"""HPC-only cold-to-fresh acceptance harness for native ``activitysim_run``.

This is an operator evidence tool, not a contract promotion.  It deliberately
uses the ordinary ActivitySim binding while retaining the persisted run and
adapter identity needed to review one cache miss and one fresh-workspace hit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import shutil

import consist
from consist import Artifact, Tracker
from consist.types import OutputArtifactSpec
import pyarrow.parquet as pq

from pilates.activitysim.preprocessor import _ensure_required_asim_config_dirs
from pilates.activitysim.runner import validate_activitysim_zarr_skims
from pilates.activitysim.outputs import configured_asim_output_tables
from pilates.config import PilatesConfig, load_config
from pilates.runtime.activitysim_run_observations import OBSERVATION_PATH_ENV
from pilates.utils import consist_runtime as cr
from pilates.utils.coupler_helpers import artifact_to_path, set_coupler_from_artifact
from pilates.workspace import Workspace
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ZARR_SKIMS,
)
from pilates.workflows.step_execution import execute_step
from pilates.workflows.steps.activitysim import activitysim_run
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from workflow_state import WorkflowState


_COHORT = {"workflow_year": 2017, "forecast_year": 2019, "iteration": 0}
_REQUIRED_TABLE_ROLES = (ASIM_LAND_USE_IN, ASIM_HOUSEHOLDS_IN, ASIM_PERSONS_IN)
_SKIM_ROLES = (ZARR_SKIMS, ASIM_OMX_SKIMS)


@dataclass(frozen=True)
class AcceptanceManifest:
    """The four declared model-input roles for one ActivitySim acceptance run."""

    inputs: Mapping[str, Path]
    selected_skim_role: str
    workflow_year: int
    forecast_year: int
    iteration: int
    released_consist_version: str


@dataclass(frozen=True)
class PhaseExecution:
    """Reviewer-facing result of one exact native ActivitySim invocation."""

    cache_hit: bool
    requested_run_id: str
    execution_run_id: str
    source_run_id: str | None
    declared_outputs: Mapping[str, Path]
    selected_roles: Mapping[str, str]
    source_bindings: Mapping[str, str]
    input_identities: Mapping[str, str]
    config_staging: Mapping[str, str]
    persisted_run: Mapping[str, object]
    body_executions_before: int
    body_executions_after: int
    runner_preparation_attempts_before: int
    runner_preparation_attempts_after: int


def _progress(message: str) -> None:
    print(f"[activitysim-run-acceptance] {message}", flush=True)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _observation_counts(path: Path) -> tuple[int, int]:
    """Count only native body/preparation entries written by the opt-in hooks."""

    if not path.exists():
        return (0, 0)
    body_executions = 0
    runner_preparation_attempts = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"invalid ActivitySim acceptance observation at line {line_number}"
            ) from error
        if not isinstance(record, Mapping):
            raise RuntimeError(
                f"invalid ActivitySim acceptance observation at line {line_number}"
            )
        event = record.get("event")
        if event == "activitysim_run_body":
            body_executions += 1
        elif event == "activitysim_runner_preparation":
            runner_preparation_attempts += 1
        else:
            raise RuntimeError(
                "unexpected ActivitySim acceptance observation event "
                f"at line {line_number}: {event!r}"
            )
    return body_executions, runner_preparation_attempts


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    elif path.is_dir():
        for member in sorted(path.rglob("*")):
            digest.update(member.relative_to(path).as_posix().encode("utf-8"))
            if member.is_file():
                digest.update(member.read_bytes())
    else:
        raise ValueError(f"cannot checksum missing acceptance path: {path}")
    return digest.hexdigest()


def _validate_cohort(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_COHORT):
        raise ValueError(f"acceptance cohort must be exactly {_COHORT}")
    for field, expected in _COHORT.items():
        if type(value[field]) is not int or value[field] != expected:
            raise ValueError(f"acceptance cohort must be exactly {_COHORT}")


def _required_release_version(raw: Mapping[str, object]) -> str:
    value = raw.get("released_consist_version")
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(
            "ActivitySim acceptance manifest requires a non-empty "
            "released_consist_version"
        )
    return value.strip()


def load_manifest(path: Path) -> AcceptanceManifest:
    """Load exactly three table inputs and one validated selected skim source."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read ActivitySim acceptance manifest: {path}"
        ) from error
    if not isinstance(raw, Mapping):
        raise ValueError("ActivitySim acceptance manifest must be a JSON object")
    released_consist_version = _required_release_version(raw)
    inputs_value = raw.get("inputs")
    if not isinstance(inputs_value, Mapping):
        raise ValueError("ActivitySim acceptance manifest requires an inputs object")
    selected_skim_roles = [role for role in _SKIM_ROLES if role in inputs_value]
    if len(selected_skim_roles) != 1:
        raise ValueError("acceptance manifest requires exactly one selected skim")
    expected_roles = {*_REQUIRED_TABLE_ROLES, selected_skim_roles[0]}
    if set(inputs_value) != expected_roles:
        raise ValueError(
            "acceptance manifest allows only the three ActivitySim tables and "
            "one selected skim"
        )
    inputs: dict[str, Path] = {}
    for role, value in inputs_value.items():
        if not isinstance(value, str):
            raise ValueError(f"acceptance input {role!r} must be a path string")
        resolved = Path(os.path.expandvars(value)).expanduser().resolve()
        if role == ZARR_SKIMS:
            rejection = validate_activitysim_zarr_skims(resolved)
            if rejection is not None:
                raise ValueError(
                    f"acceptance input {role!r} is not a valid Zarr store: {rejection}"
                )
        elif not resolved.is_file():
            raise ValueError(
                f"acceptance input {role!r} is not a readable file: {resolved}"
            )
        inputs[str(role)] = resolved
    cohort = raw.get("cohort")
    _validate_cohort(cohort)
    return AcceptanceManifest(
        inputs=inputs,
        selected_skim_role=selected_skim_roles[0],
        workflow_year=_COHORT["workflow_year"],
        forecast_year=_COHORT["forecast_year"],
        iteration=_COHORT["iteration"],
        released_consist_version=released_consist_version,
    )


def _distribution_consist_import_paths(
    distribution: importlib_metadata.Distribution,
) -> tuple[Path, ...]:
    """Return wheel-recorded locations that can provide ``consist.__init__``."""

    package_files = distribution.files
    if not package_files:
        raise RuntimeError("installed Consist distribution has no package-file record")
    init_files = [
        member
        for member in package_files
        if tuple(Path(member).parts[-2:]) == ("consist", "__init__.py")
    ]
    if not init_files:
        raise RuntimeError(
            "installed Consist distribution does not record consist/__init__.py"
        )
    return tuple(
        Path(distribution.locate_file(member)).resolve() for member in init_files
    )


def preflight_released_consist(required_release_version: str) -> dict[str, object]:
    """Describe whether the imported Consist is the operator's release install."""

    if not isinstance(required_release_version, str) or required_release_version == "":
        raise ValueError("released Consist preflight requires a non-empty version")
    evidence: dict[str, object] = {
        "required_release_version": required_release_version,
        "installed_version": None,
        "importlib_metadata_version": None,
        "public_version": None,
        "import_path": None,
        "distribution_import_paths": [],
        "distribution_import_path_match": False,
        "editable_install": None,
        "release_install": False,
        "version_match": False,
        "public_api_matches_metadata": False,
        "valid": False,
    }
    try:
        distribution = importlib_metadata.distribution("consist")
        installed_version = distribution.version
        metadata_version = importlib_metadata.version("consist")
        public_version = consist.__version__
        import_file = consist.__file__
        if not isinstance(import_file, str):
            raise RuntimeError("consist has no importable module file")
        import_path = Path(import_file).resolve()
        evidence.update(
            {
                "installed_version": installed_version,
                "importlib_metadata_version": metadata_version,
                "public_version": public_version,
                "import_path": str(import_path),
            }
        )
        direct_url_text = distribution.read_text("direct_url.json")
        editable_install = False
        if direct_url_text is not None:
            direct_url = json.loads(direct_url_text)
            if not isinstance(direct_url, Mapping):
                raise RuntimeError("consist direct_url.json is not an object")
            directory_info = direct_url.get("dir_info")
            editable_install = (
                isinstance(directory_info, Mapping)
                and directory_info.get("editable") is True
            )
        evidence.update(
            {
                "editable_install": editable_install,
                "release_install": not editable_install,
            }
        )
        distribution_import_paths = _distribution_consist_import_paths(distribution)
    except (
        AttributeError,
        importlib_metadata.PackageNotFoundError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
        return evidence

    version_match = (
        installed_version == required_release_version
        and metadata_version == required_release_version
        and public_version == required_release_version
    )
    public_api_matches_metadata = (
        public_version == metadata_version == installed_version
    )
    distribution_import_path_match = import_path in distribution_import_paths
    release_install = not editable_install
    evidence.update(
        {
            "distribution_import_paths": [
                str(path) for path in distribution_import_paths
            ],
            "distribution_import_path_match": distribution_import_path_match,
            "editable_install": editable_install,
            "release_install": release_install,
            "version_match": version_match,
            "public_api_matches_metadata": public_api_matches_metadata,
            "valid": (
                release_install
                and version_match
                and public_api_matches_metadata
                and distribution_import_path_match
            ),
        }
    )
    return evidence


def _validate_state(state: WorkflowState) -> None:
    observed = {
        "workflow_year": state.year,
        "forecast_year": state.forecast_year,
        "iteration": state.current_inner_iter,
    }
    _validate_cohort(observed)


def _workspace_paths(
    paths: Mapping[str, object], *, workspace_root: Path
) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    resolved_root = workspace_root.resolve()
    for key, value in sorted(paths.items()):
        path = Path(str(value)).resolve()
        try:
            relative_path = path.relative_to(resolved_root).as_posix()
        except ValueError:
            normalized[str(key)] = {"scope": "external", "path": str(path)}
        else:
            normalized[str(key)] = {
                "scope": "workspace",
                "relative_path": relative_path,
            }
    return normalized


def _cache_source_run_id(meta: Mapping[str, object]) -> str | None:
    value = meta.get("cache_source")
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        source = value.get("run_id") or value.get("source_run_id")
        return str(source) if source is not None else None
    return None


def _artifact_evidence(artifact: Artifact) -> dict[str, object]:
    meta = dict(artifact.meta or {})
    content_identity = meta.get("content_identity")
    return {
        "artifact_id": str(artifact.id),
        "key": artifact.key,
        "fingerprint": artifact.hash,
        "content_identity": (
            str(content_identity) if content_identity is not None else artifact.hash
        ),
        "selector": {
            "driver": artifact.driver,
            "table_path": artifact.table_path,
            "array_path": artifact.array_path,
        },
        "container_uri": artifact.container_uri,
        "producer_run_id": artifact.run_id,
    }


def _artifact_links(artifacts: Mapping[str, Artifact]) -> list[dict[str, object]]:
    return [
        {"binding_key": key, **_artifact_evidence(artifact)}
        for key, artifact in sorted(artifacts.items())
    ]


def _adapter_identity(
    meta: Mapping[str, object], config: Mapping[str, object]
) -> dict[str, object]:
    plan = config.get("__consist_config_plan__")
    plan_mapping = plan if isinstance(plan, Mapping) else {}
    return {
        "name": meta.get("config_adapter") or plan_mapping.get("adapter"),
        "version": meta.get("config_adapter_version")
        or plan_mapping.get("adapter_version"),
        "hash": meta.get("config_bundle_hash") or plan_mapping.get("hash"),
    }


def _persisted_run_evidence(
    *,
    tracker: Tracker,
    requested_run_id: str,
    phase: str,
    workspace_root: Path,
    evidence_root: Path,
) -> dict[str, object]:
    run = tracker.get_run(requested_run_id)
    if run is None or run.status != "completed":
        status = None if run is None else run.status
        raise RuntimeError(
            f"acceptance run {requested_run_id} was not durably completed: {status}"
        )
    record = tracker.get_run_record(requested_run_id)
    if record is None or record.run.id != requested_run_id:
        raise RuntimeError(
            f"acceptance run snapshot was unavailable for {requested_run_id}"
        )
    meta = dict(run.meta or {})
    source_run_id = _cache_source_run_id(meta)
    cache_hit = meta.get("cache_hit") is True
    if cache_hit and source_run_id is None:
        raise RuntimeError(
            f"persisted cache hit {requested_run_id} has no cache source"
        )
    if not cache_hit and source_run_id is not None:
        raise RuntimeError(
            f"persisted cache miss {requested_run_id} unexpectedly has source {source_run_id}"
        )
    snapshot_path = evidence_root / "persisted-runs" / f"{phase}.json"
    _write_json(snapshot_path, record.model_dump(mode="json", warnings=False))
    requested_staging = meta.get("requested_input_staging")
    requested_staging_mapping = (
        dict(requested_staging) if isinstance(requested_staging, Mapping) else {}
    )
    materialized_outputs = meta.get("materialized_outputs")
    materialized_mapping = (
        dict(materialized_outputs) if isinstance(materialized_outputs, Mapping) else {}
    )
    input_binding = meta.get("input_binding")
    input_binding_mapping = (
        dict(input_binding) if isinstance(input_binding, Mapping) else {}
    )
    bindings = input_binding_mapping.get("bindings")
    binding_records = bindings if isinstance(bindings, list) else []
    action_input_artifact_ids = {
        str(binding["artifact_id"])
        for binding in binding_records
        if isinstance(binding, Mapping) and binding.get("artifact_id") is not None
    }
    linked = tracker.get_artifacts_for_run(requested_run_id)
    input_identity = meta.get("input_identity")
    config_identity_manifest = meta.get("config_identity_manifest")
    input_paths = requested_staging_mapping.get("input_paths")
    input_paths_mapping = input_paths if isinstance(input_paths, Mapping) else {}
    return {
        "binding_kind": "ordinary-binding",
        "requested_run_id": requested_run_id,
        "execution_run_id": source_run_id or requested_run_id,
        "source_run_id": source_run_id,
        "cache_outcome": "hit" if cache_hit else "miss",
        "run_snapshot_reference": snapshot_path.relative_to(evidence_root).as_posix(),
        "identity": {
            "config_hash": run.config_hash,
            "input_hash": run.input_hash,
            "git_hash": run.git_hash,
            "signature": run.signature,
            "year": run.year,
            "iteration": run.iteration,
            "stage": run.stage,
            "phase": run.phase,
            "config": dict(record.config),
            "facet": dict(record.facet),
            "config_adapter": _adapter_identity(meta, dict(record.config)),
            "input_binding": input_binding_mapping,
            "input_identity": (
                dict(input_identity) if isinstance(input_identity, Mapping) else {}
            ),
        },
        "config_identity_manifest": (
            dict(config_identity_manifest)
            if isinstance(config_identity_manifest, Mapping)
            else None
        ),
        "artifacts": {
            "inputs": _artifact_links(linked.inputs),
            "action_inputs": [
                link
                for link in _artifact_links(linked.inputs)
                if link["artifact_id"] in action_input_artifact_ids
            ],
            "outputs": _artifact_links(linked.outputs),
        },
        "requested_input_staging": {
            "persisted": requested_staging_mapping,
            "normalized_input_paths": _workspace_paths(
                input_paths_mapping, workspace_root=workspace_root
            ),
        },
        "materialized_outputs": {
            "persisted": materialized_mapping,
            "normalized_paths": _workspace_paths(
                materialized_mapping, workspace_root=workspace_root
            ),
        },
    }


def _config_source_root(settings: PilatesConfig) -> Path:
    activitysim_settings = settings.activitysim
    if activitysim_settings is None:
        raise ValueError("acceptance settings must define an ActivitySim section")
    local_configs_folder = activitysim_settings.local_configs_folder
    region = settings.run.region
    candidate = Path(os.path.expandvars(local_configs_folder)).expanduser()
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[2] / candidate
    source_root = (candidate / region).resolve()
    if not source_root.is_dir():
        raise ValueError(
            f"ActivitySim acceptance config root is not a directory: {source_root}"
        )
    return source_root


def _stage_workspace_config(
    *, settings: PilatesConfig, workspace: Workspace
) -> dict[str, str]:
    """Seed one empty acceptance workspace with adapter-tracked config sources."""

    source_root = _config_source_root(settings)
    destination = Path(workspace.get_asim_mutable_configs_dir())
    if destination.exists():
        raise RuntimeError(
            "acceptance workspace config root already exists; each phase must start empty"
        )
    shutil.copytree(source_root, destination)
    activitysim_settings = settings.activitysim
    if activitysim_settings is None:
        raise ValueError("acceptance settings must define an ActivitySim section")
    _ensure_required_asim_config_dirs(
        configs_dest_dir=str(destination),
        main_configs_dir=str(activitysim_settings.main_configs_dir),
    )
    return {
        "source_root": str(source_root),
        "staged_root": str(destination),
        "source_sha256": _sha256_path(source_root),
    }


def _selected_input_paths(resolved: ResolvedStepInputs) -> dict[str, Path]:
    inputs = resolved.binding.inputs or {}
    selected: dict[str, Path] = {}
    for key, artifact in inputs.items():
        path = artifact_to_path(artifact)
        if path is None:
            raise RuntimeError(f"selected acceptance input {key!r} has no local path")
        selected[str(key)] = Path(path)
    return selected


def _output_path(value: Path | str | OutputArtifactSpec) -> Path:
    path = value.path if isinstance(value, OutputArtifactSpec) else value
    if isinstance(path, Path):
        return path
    if isinstance(path, str):
        return Path(path)
    raise RuntimeError(f"ActivitySim acceptance output has invalid path: {value!r}")


def run_phase(
    *,
    phase: str,
    workspace_root: Path,
    settings: PilatesConfig,
    state: WorkflowState,
    tracker: Tracker,
    artifacts: Mapping[str, Artifact],
    evidence_root: Path,
    observation_log: Path,
) -> PhaseExecution:
    """Execute the native boundary once in one separately rooted workspace."""

    if workspace_root.exists():
        raise RuntimeError(f"acceptance workspace must start empty: {workspace_root}")
    workspace = Workspace(settings, str(workspace_root.parent), workspace_root.name)
    config_staging = _stage_workspace_config(settings=settings, workspace=workspace)
    body_before, preparation_before = _observation_counts(observation_log)
    with tracker.scenario(f"activitysim-run-acceptance-{phase}") as scenario:
        for key, artifact in artifacts.items():
            set_coupler_from_artifact(scenario.coupler, key, artifact)
        resolved = activitysim_run.resolve_inputs(
            settings=settings,
            state=state,
            workspace=workspace,
            coupler=scenario.coupler,
        )
        result, _ = execute_step(
            scenario=scenario,
            definition=activitysim_run,
            settings=settings,
            state=state,
            workspace=workspace,
            stage="activity_demand",
            year=state.year,
            iteration=state.current_inner_iter,
            phase="run",
            resolved_inputs=resolved,
        )
    body_after, preparation_after = _observation_counts(observation_log)
    declared = activitysim_run.output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )
    persisted = _persisted_run_evidence(
        tracker=tracker,
        requested_run_id=str(result.run.id),
        phase=phase,
        workspace_root=workspace_root,
        evidence_root=evidence_root,
    )
    expected_outcome = "hit" if result.cache_hit else "miss"
    if persisted["cache_outcome"] != expected_outcome:
        raise RuntimeError(
            "in-memory and persisted cache outcomes disagree for "
            f"{result.run.id}: {expected_outcome} != {persisted['cache_outcome']}"
        )
    return PhaseExecution(
        cache_hit=bool(result.cache_hit),
        requested_run_id=str(persisted["requested_run_id"]),
        execution_run_id=str(persisted["execution_run_id"]),
        source_run_id=(
            str(persisted["source_run_id"])
            if persisted["source_run_id"] is not None
            else None
        ),
        declared_outputs={key: _output_path(value) for key, value in declared.items()},
        selected_roles={
            key: str(value) for key, value in resolved.selected_key_by_role.items()
        },
        source_bindings={
            key: str(value) for key, value in resolved.source_by_role.items()
        },
        input_identities={
            key: _sha256_path(path)
            for key, path in _selected_input_paths(resolved).items()
        },
        config_staging=config_staging,
        persisted_run=persisted,
        body_executions_before=body_before,
        body_executions_after=body_after,
        runner_preparation_attempts_before=preparation_before,
        runner_preparation_attempts_after=preparation_after,
    )


def _semantic_product(
    *, key: str, path: Path, workspace_root: Path, configured_tables: Mapping[str, str]
) -> dict[str, object]:
    try:
        relative_path = path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return {"valid": False, "key": key, "reason": "output outside workspace"}
    if key == ZARR_SKIMS:
        rejection = validate_activitysim_zarr_skims(path)
        return {
            "valid": rejection is None,
            "kind": "zarr-skims",
            "relative_path": relative_path,
            "validation": "valid-zarr-root" if rejection is None else rejection,
        }
    expected_table = configured_tables.get(key)
    if expected_table is None:
        return {"valid": False, "key": key, "reason": "undeclared ActivitySim table"}
    if not path.is_file():
        return {
            "valid": False,
            "kind": "activitysim-final-pipeline-table",
            "relative_path": relative_path,
            "table": expected_table,
            "reason": "missing parquet output",
        }
    expected_suffix = Path("final_pipeline") / expected_table / "final.parquet"
    if tuple(path.parts[-3:]) != tuple(expected_suffix.parts):
        return {
            "valid": False,
            "kind": "activitysim-final-pipeline-table",
            "relative_path": relative_path,
            "table": expected_table,
            "reason": "unexpected final-pipeline location",
        }
    try:
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        if metadata is None:
            raise RuntimeError("Parquet footer metadata is unavailable")
        row_count = metadata.num_rows
        columns = list(parquet_file.schema_arrow.names)
    except Exception as error:
        return {
            "valid": False,
            "kind": "activitysim-final-pipeline-table",
            "relative_path": relative_path,
            "table": expected_table,
            "reason": f"unreadable parquet: {type(error).__name__}",
        }
    return {
        "valid": True,
        "kind": "activitysim-final-pipeline-table",
        "relative_path": relative_path,
        "table": expected_table,
        "row_count": row_count,
        "columns": columns,
    }


def _phase_record(
    execution: PhaseExecution, *, workspace_root: Path, settings: PilatesConfig
) -> dict[str, object]:
    configured_tables = configured_asim_output_tables(settings)
    products = {
        key: _semantic_product(
            key=key,
            path=path,
            workspace_root=workspace_root,
            configured_tables=configured_tables,
        )
        for key, path in execution.declared_outputs.items()
    }
    return {
        "cache_hit": execution.cache_hit,
        "requested_run_id": execution.requested_run_id,
        "execution_run_id": execution.execution_run_id,
        "source_run_id": execution.source_run_id,
        "workspace_root": str(workspace_root),
        "declared_outputs": products,
        "selected_roles": dict(execution.selected_roles),
        "source_bindings": dict(execution.source_bindings),
        "input_identities": dict(execution.input_identities),
        "config_staging": dict(execution.config_staging),
        "persisted_run": dict(execution.persisted_run),
        "body_executions_before": execution.body_executions_before,
        "body_executions_after": execution.body_executions_after,
        "runner_preparation_attempts_before": execution.runner_preparation_attempts_before,
        "runner_preparation_attempts_after": execution.runner_preparation_attempts_after,
        "cohort": dict(_COHORT),
    }


def validate(
    cold: Mapping[str, object], fresh: Mapping[str, object]
) -> dict[str, object]:
    """Validate the only admissible cold-miss/fresh-workspace-hit relationship."""

    identity_fields = (
        "selected_roles",
        "source_bindings",
        "input_identities",
    )
    differences = [
        field for field in identity_fields if cold.get(field) != fresh.get(field)
    ]
    cold_config_staging = cold.get("config_staging")
    fresh_config_staging = fresh.get("config_staging")
    config_staging_valid = (
        isinstance(cold_config_staging, Mapping)
        and isinstance(fresh_config_staging, Mapping)
        and cold_config_staging.get("source_root")
        == fresh_config_staging.get("source_root")
        and cold_config_staging.get("source_sha256")
        == fresh_config_staging.get("source_sha256")
        and cold_config_staging.get("staged_root")
        != fresh_config_staging.get("staged_root")
    )
    cold_products = cold.get("declared_outputs")
    fresh_products = fresh.get("declared_outputs")
    products_mapping = isinstance(cold_products, Mapping) and isinstance(
        fresh_products, Mapping
    )
    declared_output_keys_valid = (
        products_mapping
        and bool(cold_products)
        and set(cold_products) == set(fresh_products)
    )
    semantic_products_valid = products_mapping and all(
        isinstance(product, Mapping) and product.get("valid") is True
        for products in (cold_products, fresh_products)
        for product in products.values()
    )
    semantic_products_match = products_mapping and cold_products == fresh_products
    cold_persisted = cold.get("persisted_run")
    fresh_persisted = fresh.get("persisted_run")
    cold_persisted_mapping = (
        cold_persisted if isinstance(cold_persisted, Mapping) else {}
    )
    fresh_persisted_mapping = (
        fresh_persisted if isinstance(fresh_persisted, Mapping) else {}
    )
    ordinary_binding_valid = (
        cold_persisted_mapping.get("binding_kind") == "ordinary-binding"
        and fresh_persisted_mapping.get("binding_kind") == "ordinary-binding"
    )
    cold_requested_run_id = cold_persisted_mapping.get("requested_run_id")
    fresh_requested_run_id = fresh_persisted_mapping.get("requested_run_id")
    cache_relationship_valid = (
        cold_requested_run_id == cold.get("requested_run_id")
        and fresh_requested_run_id == fresh.get("requested_run_id")
        and cold_requested_run_id != fresh_requested_run_id
        and cold_persisted_mapping.get("cache_outcome") == "miss"
        and cold_persisted_mapping.get("execution_run_id") == cold_requested_run_id
        and cold_persisted_mapping.get("source_run_id") is None
        and fresh_persisted_mapping.get("cache_outcome") == "hit"
        and fresh_persisted_mapping.get("execution_run_id") == cold_requested_run_id
        and fresh_persisted_mapping.get("source_run_id") == cold_requested_run_id
        and fresh.get("source_run_id") == cold_requested_run_id
    )
    persisted_identity_valid = cold_persisted_mapping.get(
        "identity"
    ) == fresh_persisted_mapping.get("identity")
    cold_artifacts = cold_persisted_mapping.get("artifacts")
    fresh_artifacts = fresh_persisted_mapping.get("artifacts")
    persisted_artifact_links_valid = (
        isinstance(cold_artifacts, Mapping)
        and isinstance(fresh_artifacts, Mapping)
        and cold_artifacts.get("action_inputs") == fresh_artifacts.get("action_inputs")
        and cold_artifacts.get("outputs") == fresh_artifacts.get("outputs")
    )
    cold_staging = cold_persisted_mapping.get("requested_input_staging")
    fresh_staging = fresh_persisted_mapping.get("requested_input_staging")
    persisted_staging_valid = (
        isinstance(cold_staging, Mapping)
        and isinstance(fresh_staging, Mapping)
        and cold_staging.get("normalized_input_paths")
        == fresh_staging.get("normalized_input_paths")
    )
    expected_fresh_destinations = (
        {
            key: {
                "scope": "workspace",
                "relative_path": product.get("relative_path"),
            }
            for key, product in fresh_products.items()
            if isinstance(product, Mapping)
        }
        if isinstance(fresh_products, Mapping)
        else {}
    )
    fresh_outputs = fresh_persisted_mapping.get("materialized_outputs")
    fresh_hydration_destinations_valid = (
        isinstance(fresh_outputs, Mapping)
        and fresh_outputs.get("normalized_paths") == expected_fresh_destinations
    )
    body_and_preparation_valid = (
        cold.get("body_executions_after") == cold.get("body_executions_before", 0) + 1
        and fresh.get("body_executions_before") == cold.get("body_executions_after")
        and fresh.get("body_executions_after") == cold.get("body_executions_after")
        and cold.get("runner_preparation_attempts_after")
        == cold.get("runner_preparation_attempts_before", 0) + 1
        and fresh.get("runner_preparation_attempts_before")
        == cold.get("runner_preparation_attempts_after")
        and fresh.get("runner_preparation_attempts_after")
        == cold.get("runner_preparation_attempts_after")
    )
    distinct_workspaces = cold.get("workspace_root") != fresh.get("workspace_root")
    valid = all(
        (
            not differences,
            config_staging_valid,
            declared_output_keys_valid,
            semantic_products_valid,
            semantic_products_match,
            ordinary_binding_valid,
            cache_relationship_valid,
            persisted_identity_valid,
            persisted_artifact_links_valid,
            persisted_staging_valid,
            fresh_hydration_destinations_valid,
            body_and_preparation_valid,
            distinct_workspaces,
        )
    )
    return {
        "valid": valid,
        "identity_differences": differences,
        "config_staging_valid": config_staging_valid,
        "declared_output_keys_valid": declared_output_keys_valid,
        "semantic_products_valid": semantic_products_valid,
        "semantic_products_match": semantic_products_match,
        "ordinary_binding_valid": ordinary_binding_valid,
        "persisted_cache_relationship_valid": cache_relationship_valid,
        "persisted_identity_valid": persisted_identity_valid,
        "persisted_artifact_links_valid": persisted_artifact_links_valid,
        "persisted_staging_valid": persisted_staging_valid,
        "fresh_hydration_destinations_valid": fresh_hydration_destinations_valid,
        "body_and_preparation_valid": body_and_preparation_valid,
        "distinct_workspaces": distinct_workspaces,
    }


def _write_checksums(
    *,
    evidence_root: Path,
    manifest: AcceptanceManifest,
    cold: PhaseExecution,
    fresh: PhaseExecution,
    observation_log: Path,
) -> None:
    checksums: dict[str, str] = {
        f"input/{key}": _sha256_path(path) for key, path in manifest.inputs.items()
    }
    for phase, execution in (("cold", cold), ("fresh", fresh)):
        for key, path in execution.declared_outputs.items():
            checksums[f"{phase}/output/{key}"] = _sha256_path(path)
    checksums["activitysim-observations.jsonl"] = _sha256_path(observation_log)
    control_records = (
        "submitted-input-manifest.json",
        "effective-input-manifest.json",
        "generated-settings.yaml",
        "runtime-environment.json",
        "persisted-runs/cold.json",
        "persisted-runs/fresh.json",
        "phases/cold.json",
        "phases/fresh.json",
        "semantic-validation.json",
    )
    for relative_path in control_records:
        checksums[relative_path] = _sha256_path(evidence_root / relative_path)
    provenance_path = evidence_root / "provenance.duckdb"
    if provenance_path.is_file():
        checksums["provenance.duckdb"] = _sha256_path(provenance_path)
    _write_json(evidence_root / "checksums.json", {"sha256": checksums})


def _execute_acceptance_phases(
    *,
    settings: PilatesConfig,
    state: WorkflowState,
    tracker: Tracker,
    artifacts: Mapping[str, Artifact],
    evidence_root: Path,
) -> tuple[PhaseExecution, PhaseExecution, Path]:
    """Run both phases while the native hooks are explicitly enabled."""

    observation_log = evidence_root / "activitysim-observations.jsonl"
    if observation_log.exists():
        raise RuntimeError(
            "ActivitySim acceptance observation log already exists; "
            "use a fresh evidence root"
        )
    previous_observation_path = os.environ.get(OBSERVATION_PATH_ENV)
    os.environ[OBSERVATION_PATH_ENV] = str(observation_log)
    common = {
        "settings": settings,
        "state": state,
        "tracker": tracker,
        "artifacts": artifacts,
        "evidence_root": evidence_root,
        "observation_log": observation_log,
    }
    cold_root = evidence_root / "workspaces" / "cold"
    fresh_root = evidence_root / "workspaces" / "fresh"
    try:
        _progress("cold acceptance phase started")
        cold_execution = run_phase(phase="cold", workspace_root=cold_root, **common)
        if cold_execution.cache_hit:
            raise RuntimeError(
                "cold ActivitySim acceptance invocation unexpectedly hit cache"
            )
        _progress("cold acceptance phase completed")
        _progress("fresh acceptance phase started")
        fresh_execution = run_phase(phase="fresh", workspace_root=fresh_root, **common)
        if not fresh_execution.cache_hit:
            raise RuntimeError(
                "fresh ActivitySim acceptance invocation missed cache or entered the body"
            )
        _progress("fresh acceptance phase completed")
        return cold_execution, fresh_execution, observation_log
    finally:
        if previous_observation_path is None:
            os.environ.pop(OBSERVATION_PATH_ENV, None)
        else:
            os.environ[OBSERVATION_PATH_ENV] = previous_observation_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    args = parser.parse_args(argv)

    evidence_root = args.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    os.environ["PILATES_LOCAL_RUN_DIR"] = str(evidence_root)
    os.environ["PILATES_ARCHIVE_RUN_DIR"] = str(evidence_root)
    os.environ["PILATES_ENABLE_ARCHIVE_COPY"] = "1"
    _progress("acceptance driver started")
    manifest = load_manifest(args.manifest)
    _progress("four-role input manifest validated")
    shutil.copy2(args.settings, evidence_root / "generated-settings.yaml")
    submitted_manifest = evidence_root / "submitted-input-manifest.json"
    if (
        submitted_manifest.exists()
        and args.manifest.resolve() != submitted_manifest.resolve()
    ):
        if submitted_manifest.read_bytes() != args.manifest.read_bytes():
            raise RuntimeError(
                "ActivitySim acceptance retained submitted manifest does not match "
                f"the requested source: {submitted_manifest}"
            )
    elif args.manifest.resolve() != submitted_manifest.resolve():
        shutil.copy2(args.manifest, submitted_manifest)
    _write_json(
        evidence_root / "effective-input-manifest.json",
        {
            "inputs": {key: str(path) for key, path in manifest.inputs.items()},
            "selected_skim_role": manifest.selected_skim_role,
            "released_consist_version": manifest.released_consist_version,
            "cohort": dict(_COHORT),
        },
    )
    release_preflight = preflight_released_consist(manifest.released_consist_version)
    _write_json(evidence_root / "runtime-environment.json", release_preflight)
    if release_preflight["valid"] is not True:
        raise RuntimeError(
            "ActivitySim acceptance requires the requested non-editable released "
            "Consist installation; inspect runtime-environment.json"
        )
    _progress("released Consist preflight validated")
    settings = load_config(str(args.settings))
    state = WorkflowState.from_settings(settings)
    _validate_state(state)
    _progress("acceptance settings and cohort validated")
    tracker = cr.create_tracker(
        settings=settings,
        run_dir=str(evidence_root / "consist-runs"),
        db_path=str(evidence_root / "provenance.duckdb"),
        allow_external_paths=True,
        mounts={"evidence": str(evidence_root)},
    )
    if tracker is None:
        raise RuntimeError("acceptance harness could not create a Consist tracker")
    artifacts: dict[str, Artifact] = {}
    with tracker.start_run("activitysim-run-acceptance-inputs", "acceptance"):
        for key, path in manifest.inputs.items():
            artifacts[key] = tracker.log_artifact(path, key=key, direction="input")
    _progress("acceptance input artifacts logged")
    cold_root = evidence_root / "workspaces" / "cold"
    fresh_root = evidence_root / "workspaces" / "fresh"
    cold_execution, fresh_execution, observation_log = _execute_acceptance_phases(
        settings=settings,
        state=state,
        tracker=tracker,
        artifacts=artifacts,
        evidence_root=evidence_root,
    )
    cold = _phase_record(cold_execution, workspace_root=cold_root, settings=settings)
    _write_json(evidence_root / "phases" / "cold.json", cold)
    fresh = _phase_record(fresh_execution, workspace_root=fresh_root, settings=settings)
    _write_json(evidence_root / "phases" / "fresh.json", fresh)
    validation = validate(cold, fresh)
    _write_json(evidence_root / "semantic-validation.json", validation)
    if validation["valid"] is not True:
        raise RuntimeError("ActivitySim acceptance semantic validation failed")
    _write_checksums(
        evidence_root=evidence_root,
        manifest=manifest,
        cold=cold_execution,
        fresh=fresh_execution,
        observation_log=observation_log,
    )
    _progress("acceptance semantic validation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
