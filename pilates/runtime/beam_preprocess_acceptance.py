"""HPC-only cold-to-fresh acceptance harness for native ``beam_preprocess``."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from consist import Artifact, Tracker

from pilates.config import load_config
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.utils import consist_runtime as cr
from pilates.workspace import Workspace
from pilates.workflows.artifact_keys import (
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
)
from pilates.workflows.step_execution import execute_step
from pilates.workflows.steps.beam import beam_preprocess
from workflow_state import WorkflowState


_REQUIRED_INPUTS = (
    BEAM_PLANS_IN,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    ATLAS_VEHICLES2_OUTPUT,
)
_ACCEPTANCE_YEAR = 2019
_ACCEPTANCE_ITERATION = 0


def _progress(message: str) -> None:
    """Emit an immediately visible HPC acceptance milestone."""

    print(f"[beam-preprocess-acceptance] {message}", flush=True)


@dataclass(frozen=True)
class PhaseExecution:
    """Reviewer-facing result of one exact native boundary invocation."""

    cache_hit: bool
    run_id: str
    source_run_id: str | None
    declared_outputs: Mapping[str, Path]
    selected_roles: Mapping[str, str]
    source_bindings: Mapping[str, str]
    input_identities: Mapping[str, str]
    config_identity: str
    snapshot_reference: str
    requested_run_id: str | None = None
    persisted_run_meta: Mapping[str, Any] | None = None
    body_executions_before: int = 0
    body_executions_after: int = 0
    cohort_year: int = _ACCEPTANCE_YEAR
    cohort_iteration: int = _ACCEPTANCE_ITERATION
    persisted_run: Mapping[str, Any] | None = None


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    else:
        for member in sorted(path.rglob("*")):
            digest.update(member.relative_to(path).as_posix().encode("utf-8"))
            if member.is_file():
                digest.update(member.read_bytes())
    return digest.hexdigest()


def _semantic_product(path: Path, *, workspace_root: Path) -> dict[str, Any]:
    if not path.exists():
        return {"relative_path": str(path), "present": False, "type": "missing"}
    relative_path = path.relative_to(workspace_root).as_posix()
    if path.is_file():
        return {
            "relative_path": relative_path,
            "present": True,
            "type": "file",
            "size": path.stat().st_size,
        }
    if not path.is_dir():
        return {"relative_path": relative_path, "present": False, "type": "invalid"}
    members = [
        {
            "relative_path": member.relative_to(path).as_posix(),
            "type": "file" if member.is_file() else "directory",
            "size": member.stat().st_size if member.is_file() else None,
        }
        for member in sorted(path.rglob("*"))
    ]
    return {
        "relative_path": relative_path,
        "present": True,
        "type": "directory",
        "members": members,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_manifest(path: Path) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    beam_input_root = Path(os.path.expandvars(str(raw["beam_input_root"]))).expanduser()
    inputs = {
        key: Path(os.path.expandvars(str(value))).expanduser()
        for key, value in raw["inputs"].items()
    }
    missing = [
        key
        for key in _REQUIRED_INPUTS
        if key not in inputs or not inputs[key].is_file()
    ]
    if missing:
        raise ValueError(
            f"acceptance manifest is missing readable required inputs: {missing}"
        )
    if not beam_input_root.is_dir():
        raise ValueError(
            f"acceptance beam_input_root is not a directory: {beam_input_root}"
        )
    cohort = raw.get("cohort")
    if cohort != {"year": _ACCEPTANCE_YEAR, "iteration": _ACCEPTANCE_ITERATION}:
        raise ValueError(
            "acceptance manifest cohort must be {'year': 2019, 'iteration': 0}"
        )
    effective = {
        "beam_input_root": str(beam_input_root.resolve()),
        "inputs": {key: str(value.resolve()) for key, value in inputs.items()},
        "cohort": cohort,
    }
    return beam_input_root, inputs, effective


def _body_execution_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def _cache_source_run_id(meta: Mapping[str, Any]) -> str | None:
    cache_source = meta.get("cache_source")
    if isinstance(cache_source, str):
        return cache_source
    if isinstance(cache_source, Mapping):
        value = cache_source.get("run_id") or cache_source.get("source_run_id")
        return str(value) if value is not None else None
    return None


def _workspace_paths(
    paths: Mapping[str, Any], *, workspace_root: Path
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


def _artifact_evidence(artifact: Artifact) -> dict[str, Any]:
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
        "meta": meta,
    }


def _artifact_links(artifacts: Mapping[str, Artifact]) -> list[dict[str, Any]]:
    return [
        {"binding_key": key, **_artifact_evidence(artifact)}
        for key, artifact in sorted(artifacts.items())
    ]


def _adapter_identity(
    meta: Mapping[str, Any], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    plan = config.get("__consist_config_plan__")
    plan_mapping = plan if isinstance(plan, Mapping) else {}
    return {
        "name": meta.get("config_adapter") or plan_mapping.get("adapter"),
        "version": meta.get("config_adapter_version")
        or plan_mapping.get("adapter_version"),
        "hash": meta.get("config_bundle_hash") or plan_mapping.get("hash"),
    }


def _persisted_ordinary_run_evidence(
    *,
    tracker: Tracker,
    requested_run_id: str,
    phase: str,
    workspace_root: Path,
    evidence_root: Path,
) -> dict[str, Any]:
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
    linked = tracker.get_artifacts_for_run(requested_run_id)
    meta = dict(run.meta or {})
    source_run_id = _cache_source_run_id(meta)
    cache_hit = meta.get("cache_hit") is True
    if cache_hit and source_run_id is None:
        raise RuntimeError(
            f"persisted cache hit {requested_run_id} has no cache source run"
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
    input_paths = requested_staging_mapping.get("input_paths")
    input_paths_mapping = input_paths if isinstance(input_paths, Mapping) else {}
    materialized_outputs = meta.get("materialized_outputs")
    materialized_mapping = (
        dict(materialized_outputs) if isinstance(materialized_outputs, Mapping) else {}
    )
    input_binding = meta.get("input_binding")
    input_identity = meta.get("input_identity")
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
    input_links = _artifact_links(linked.inputs)
    output_links = _artifact_links(linked.outputs)
    config_identity_manifest = meta.get("config_identity_manifest")
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
            "config_adapter": _adapter_identity(meta, config=record.config),
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
            "inputs": input_links,
            "action_inputs": [
                link
                for link in input_links
                if link["artifact_id"] in action_input_artifact_ids
            ],
            "outputs": output_links,
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


def record_body_execution(*, step: str) -> None:
    """Record an opt-in body observation without changing ordinary execution."""

    destination = os.environ.get("PILATES_BEAM_PREPROCESS_ACCEPTANCE_BODY_LOG")
    if destination:
        with Path(destination).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": step}) + "\n")


def _stage_beam_input_tree(
    *, source_root: Path, settings: Any, workspace: Workspace
) -> None:
    destination = Path(workspace.get_beam_mutable_data_dir()) / settings.run.region
    shutil.copytree(source_root, destination, dirs_exist_ok=True)


def run_phase(
    *,
    phase: str,
    workspace_root: Path,
    settings: Any,
    state: Any,
    tracker: Tracker,
    artifacts: Mapping[str, Artifact],
    beam_input_root: Path,
    evidence_root: Path,
    body_log: Path,
) -> PhaseExecution:
    """Execute the committed step once in a separately rooted workspace."""

    workspace = Workspace(settings, str(workspace_root.parent), workspace_root.name)
    _stage_beam_input_tree(
        source_root=beam_input_root, settings=settings, workspace=workspace
    )
    before = _body_execution_count(body_log)
    with tracker.scenario(f"beam-preprocess-acceptance-{phase}") as scenario:
        for key, artifact in artifacts.items():
            scenario.coupler.set_from_artifact(key, artifact)
        resolved = beam_preprocess.resolve_inputs(
            settings=settings,
            state=state,
            workspace=workspace,
            coupler=scenario.coupler,
        )
        result, _ = execute_step(
            scenario=scenario,
            definition=beam_preprocess,
            settings=settings,
            state=state,
            workspace=workspace,
            stage="traffic_assignment",
            year=state.year,
            iteration=state.current_inner_iter,
            phase="preprocess",
            resolved_inputs=resolved,
        )
    declared = beam_preprocess.output_paths(
        settings=settings, state=state, workspace=workspace, resolved_inputs=resolved
    )
    after = _body_execution_count(body_log)
    run_meta = dict(result.run.meta)
    persisted = _persisted_ordinary_run_evidence(
        tracker=tracker,
        requested_run_id=str(result.run.id),
        phase=phase,
        workspace_root=workspace_root,
        evidence_root=evidence_root,
    )
    expected_cache_outcome = "hit" if result.cache_hit else "miss"
    if persisted["cache_outcome"] != expected_cache_outcome:
        raise RuntimeError(
            "in-memory and persisted cache outcomes disagree for "
            f"{result.run.id}: {expected_cache_outcome} != "
            f"{persisted['cache_outcome']}"
        )
    return PhaseExecution(
        cache_hit=bool(result.cache_hit),
        run_id=str(result.run.id),
        source_run_id=(
            str(persisted["source_run_id"])
            if persisted["source_run_id"] is not None
            else None
        ),
        declared_outputs={key: Path(value) for key, value in declared.items()},
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
        config_identity=_sha256_path(beam_input_root),
        snapshot_reference=str(evidence_root / "provenance.duckdb"),
        requested_run_id=str(persisted["requested_run_id"]),
        persisted_run_meta=run_meta,
        body_executions_before=before,
        body_executions_after=after,
        cohort_year=state.year,
        cohort_iteration=state.current_inner_iter,
        persisted_run=persisted,
    )


def _selected_input_paths(resolved: Any) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for key, artifact in (resolved.binding.inputs or {}).items():
        path = artifact_to_path(artifact)
        if path is None:
            raise RuntimeError(f"selected acceptance input {key!r} has no local path")
        selected[key] = Path(path)
    return selected


def _phase_record(execution: PhaseExecution, workspace_root: Path) -> dict[str, Any]:
    products = {
        key: _semantic_product(path, workspace_root=workspace_root)
        for key, path in execution.declared_outputs.items()
    }
    return {
        "cache_hit": execution.cache_hit,
        "run_id": execution.run_id,
        "source_run_id": execution.source_run_id,
        "workspace_root": str(workspace_root),
        "declared_outputs": products,
        "selected_roles": dict(execution.selected_roles),
        "source_bindings": dict(execution.source_bindings),
        "input_identities": dict(execution.input_identities),
        "config_identity": execution.config_identity,
        "consist_snapshot_reference": execution.snapshot_reference,
        "requested_run_id": execution.requested_run_id,
        "persisted_run_meta": dict(execution.persisted_run_meta or {}),
        "persisted_run": dict(execution.persisted_run or {}),
        "body_executions_before": execution.body_executions_before,
        "body_executions_after": execution.body_executions_after,
        "cohort": {
            "year": execution.cohort_year,
            "iteration": execution.cohort_iteration,
        },
    }


def _validate(cold: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    identity_fields = (
        "selected_roles",
        "source_bindings",
        "input_identities",
        "config_identity",
    )
    differences = [field for field in identity_fields if cold[field] != fresh[field]]
    keys_match = set(cold["declared_outputs"]) == set(fresh["declared_outputs"])
    present = all(
        item.get("present") is True for item in cold["declared_outputs"].values()
    ) and all(
        item.get("present") is True for item in fresh["declared_outputs"].values()
    )
    products_match = (
        keys_match and cold["declared_outputs"] == fresh["declared_outputs"]
    )
    body_valid = (
        cold["body_executions_after"] == cold["body_executions_before"] + 1
        and fresh["body_executions_after"] == cold["body_executions_after"]
    )
    cold_persisted = cold["persisted_run"]
    fresh_persisted = fresh["persisted_run"]
    ordinary_binding_valid = (
        cold_persisted.get("binding_kind") == "ordinary-binding"
        and fresh_persisted.get("binding_kind") == "ordinary-binding"
    )
    cold_requested_run_id = cold_persisted.get("requested_run_id")
    fresh_requested_run_id = fresh_persisted.get("requested_run_id")
    cache_relationship_valid = (
        cold_requested_run_id == cold.get("requested_run_id")
        and fresh_requested_run_id == fresh.get("requested_run_id")
        and cold_requested_run_id != fresh_requested_run_id
        and cold_persisted.get("cache_outcome") == "miss"
        and cold_persisted.get("execution_run_id") == cold_requested_run_id
        and cold_persisted.get("source_run_id") is None
        and fresh_persisted.get("cache_outcome") == "hit"
        and fresh_persisted.get("execution_run_id") == cold_requested_run_id
        and fresh_persisted.get("source_run_id") == cold_requested_run_id
        and fresh.get("source_run_id") == cold_requested_run_id
    )
    persisted_identity_valid = cold_persisted.get("identity") == fresh_persisted.get(
        "identity"
    )
    cold_artifacts = cold_persisted.get("artifacts", {})
    fresh_artifacts = fresh_persisted.get("artifacts", {})
    persisted_artifact_links_valid = cold_artifacts.get(
        "action_inputs"
    ) == fresh_artifacts.get("action_inputs") and cold_artifacts.get(
        "outputs"
    ) == fresh_artifacts.get("outputs")
    cold_staging = cold_persisted.get("requested_input_staging", {})
    fresh_staging = fresh_persisted.get("requested_input_staging", {})
    persisted_staging_valid = cold_staging.get(
        "normalized_input_paths"
    ) == fresh_staging.get("normalized_input_paths")
    expected_fresh_destinations = {
        key: {
            "scope": "workspace",
            "relative_path": item.get("relative_path"),
        }
        for key, item in fresh["declared_outputs"].items()
    }
    fresh_materialized = fresh_persisted.get("materialized_outputs", {})
    fresh_hydration_destinations_valid = (
        fresh_materialized.get("normalized_paths") == expected_fresh_destinations
    )
    valid = (
        not differences
        and present
        and products_match
        and body_valid
        and ordinary_binding_valid
        and cache_relationship_valid
        and persisted_identity_valid
        and persisted_artifact_links_valid
        and persisted_staging_valid
        and fresh_hydration_destinations_valid
        and cold["workspace_root"] != fresh["workspace_root"]
    )
    return {
        "valid": valid,
        "identity_differences": differences,
        "semantic_products_match": products_match,
        "declared_output_keys_match": keys_match,
        "declared_outputs_present": present,
        "body_execution_valid": body_valid,
        "ordinary_binding_valid": ordinary_binding_valid,
        "persisted_cache_relationship_valid": cache_relationship_valid,
        "persisted_identity_valid": persisted_identity_valid,
        "persisted_artifact_links_valid": persisted_artifact_links_valid,
        "persisted_staging_valid": persisted_staging_valid,
        "fresh_hydration_destinations_valid": fresh_hydration_destinations_valid,
        "expected_workspace_differences": [
            {
                "cold_workspace_root": cold["workspace_root"],
                "fresh_workspace_root": fresh["workspace_root"],
            }
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    args = parser.parse_args(argv)

    _progress("acceptance driver started")
    evidence_root = args.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    os.environ["PILATES_LOCAL_RUN_DIR"] = str(evidence_root)
    os.environ["PILATES_ARCHIVE_RUN_DIR"] = str(evidence_root)
    os.environ["PILATES_ENABLE_ARCHIVE_COPY"] = "1"
    beam_input_root, input_paths, effective_manifest = _load_manifest(args.manifest)
    _progress("acceptance manifest validated")
    shutil.copy2(args.settings, evidence_root / "generated-settings.yaml")
    submitted_manifest = evidence_root / "submitted-input-manifest.json"
    if args.manifest.resolve() != submitted_manifest.resolve():
        shutil.copy2(args.manifest, submitted_manifest)
    _write_json(evidence_root / "effective-input-manifest.json", effective_manifest)
    settings = load_config(str(args.settings))
    state = WorkflowState.from_settings(settings)
    if (
        state.year != _ACCEPTANCE_YEAR
        or state.current_inner_iter != _ACCEPTANCE_ITERATION
    ):
        raise ValueError(
            "acceptance settings must initialize WorkflowState at 2019 / iteration 0"
        )
    _progress("acceptance settings and state validated")
    tracker = cr.create_tracker(
        settings=settings,
        run_dir=str(evidence_root / "consist-runs"),
        db_path=str(evidence_root / "provenance.duckdb"),
        allow_external_paths=True,
        mounts={"evidence": str(evidence_root), "inputs": str(beam_input_root)},
    )
    if tracker is None:
        raise RuntimeError("acceptance harness could not create a Consist tracker")
    _progress("acceptance tracker created")
    artifacts: dict[str, Artifact] = {}
    with tracker.start_run("beam-preprocess-acceptance-inputs", "acceptance"):
        for key, path in input_paths.items():
            artifacts[key] = tracker.log_artifact(path, key=key, direction="input")
    _progress("acceptance input artifacts logged")

    cold_root = evidence_root / "workspaces" / "cold"
    fresh_root = evidence_root / "workspaces" / "fresh"
    body_log = evidence_root / "body-executions.jsonl"
    os.environ["PILATES_BEAM_PREPROCESS_ACCEPTANCE_BODY_LOG"] = str(body_log)
    common = {
        "settings": settings,
        "state": state,
        "tracker": tracker,
        "artifacts": artifacts,
        "beam_input_root": beam_input_root,
        "evidence_root": evidence_root,
        "body_log": body_log,
    }
    _progress("cold acceptance phase started")
    cold_execution = run_phase(phase="cold", workspace_root=cold_root, **common)
    cold = _phase_record(cold_execution, cold_root)
    _write_json(evidence_root / "phases" / "cold.json", cold)
    if cold_execution.cache_hit:
        raise RuntimeError(
            "cold beam_preprocess acceptance invocation unexpectedly hit cache"
        )
    _progress("cold acceptance phase completed")

    _progress("fresh acceptance phase started")
    fresh_execution = run_phase(phase="fresh", workspace_root=fresh_root, **common)
    fresh = _phase_record(fresh_execution, fresh_root)
    _write_json(evidence_root / "phases" / "fresh.json", fresh)
    if not fresh_execution.cache_hit:
        raise RuntimeError(
            "fresh beam_preprocess acceptance invocation missed cache or executed the body"
        )
    _progress("fresh acceptance phase completed")

    validation = _validate(cold, fresh)
    _write_json(evidence_root / "semantic-validation.json", validation)
    if not validation["valid"]:
        raise RuntimeError("beam_preprocess acceptance semantic validation failed")
    _progress("acceptance semantic validation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
