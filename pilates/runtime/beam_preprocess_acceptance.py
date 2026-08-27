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

from pilates.config import load_config
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.utils import consist_runtime as cr
from pilates.workspace import Workspace
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
)
from pilates.workflows.step_execution import execute_step
from pilates.workflows.steps.beam import beam_preprocess
from workflow_state import WorkflowState


_REQUIRED_INPUTS = (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN)


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
    relative_path = path.relative_to(workspace_root).as_posix()
    if path.is_file():
        return {"relative_path": relative_path, "type": "file", "size": path.stat().st_size}
    members = [
        {
            "relative_path": member.relative_to(path).as_posix(),
            "type": "file" if member.is_file() else "directory",
            "size": member.stat().st_size if member.is_file() else None,
        }
        for member in sorted(path.rglob("*"))
    ]
    return {"relative_path": relative_path, "type": "directory", "members": members}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_manifest(path: Path) -> tuple[Path, dict[str, Path]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    beam_input_root = Path(os.path.expandvars(str(raw["beam_input_root"]))).expanduser()
    inputs = {
        key: Path(os.path.expandvars(str(value))).expanduser()
        for key, value in raw["inputs"].items()
    }
    missing = [key for key in _REQUIRED_INPUTS if key not in inputs or not inputs[key].is_file()]
    if missing:
        raise ValueError(f"acceptance manifest is missing readable required inputs: {missing}")
    if not beam_input_root.is_dir():
        raise ValueError(f"acceptance beam_input_root is not a directory: {beam_input_root}")
    return beam_input_root, inputs


def _stage_beam_input_tree(*, source_root: Path, settings: Any, workspace: Workspace) -> None:
    destination = Path(workspace.get_beam_mutable_data_dir()) / settings.run.region
    shutil.copytree(source_root, destination, dirs_exist_ok=True)


def run_phase(
    *,
    phase: str,
    workspace_root: Path,
    settings: Any,
    state: Any,
    tracker: Any,
    artifacts: Mapping[str, Any],
    beam_input_root: Path,
    evidence_root: Path,
) -> PhaseExecution:
    """Execute the committed step once in a separately rooted workspace."""

    workspace = Workspace(settings, str(workspace_root.parent), workspace_root.name)
    _stage_beam_input_tree(source_root=beam_input_root, settings=settings, workspace=workspace)
    with cr.scenario(f"beam-preprocess-acceptance-{phase}", tracker=tracker) as scenario:
        for key, artifact in artifacts.items():
            scenario.coupler.set_from_artifact(key, artifact)
        resolved = beam_preprocess.resolve_inputs(
            settings=settings, state=state, workspace=workspace, coupler=scenario.coupler
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
    return PhaseExecution(
        cache_hit=bool(result.cache_hit),
        run_id=str(result.run.id),
        source_run_id=None,
        declared_outputs={key: Path(value) for key, value in declared.items()},
        selected_roles={key: str(value) for key, value in resolved.selected_key_by_role.items()},
        source_bindings={key: str(value) for key, value in resolved.source_by_role.items()},
        input_identities={key: _sha256_path(path) for key, path in _selected_input_paths(resolved).items()},
        config_identity=_sha256_path(beam_input_root),
        snapshot_reference=str(evidence_root / "provenance.duckdb"),
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
    }


def _validate(cold: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    identity_fields = ("selected_roles", "source_bindings", "input_identities", "config_identity")
    differences = [field for field in identity_fields if cold[field] != fresh[field]]
    products_match = cold["declared_outputs"] == fresh["declared_outputs"]
    valid = not differences and products_match and cold["workspace_root"] != fresh["workspace_root"]
    return {
        "valid": valid,
        "identity_differences": differences,
        "semantic_products_match": products_match,
        "expected_workspace_differences": [
            {"cold_workspace_root": cold["workspace_root"], "fresh_workspace_root": fresh["workspace_root"]}
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    args = parser.parse_args(argv)

    evidence_root = args.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    beam_input_root, input_paths = _load_manifest(args.manifest)
    shutil.copy2(args.settings, evidence_root / "generated-settings.yaml")
    shutil.copy2(args.manifest, evidence_root / "input-manifest.json")
    settings = load_config(str(args.settings))
    state = WorkflowState.from_settings(settings)
    tracker = cr.create_tracker(
        settings=settings,
        run_dir=str(evidence_root / "consist-runs"),
        db_path=str(evidence_root / "provenance.duckdb"),
        allow_external_paths=True,
        mounts={"workspace": str(evidence_root), "inputs": str(beam_input_root)},
    )
    if tracker is None:
        raise RuntimeError("acceptance harness could not create a Consist tracker")
    artifacts: dict[str, Any] = {}
    with tracker.start_run("beam-preprocess-acceptance-inputs", "acceptance"):
        for key, path in input_paths.items():
            artifacts[key] = tracker.log_artifact(path, key=key, direction="input")

    cold_root = evidence_root / "workspaces" / "cold"
    fresh_root = evidence_root / "workspaces" / "fresh"
    common = {
        "settings": settings,
        "state": state,
        "tracker": tracker,
        "artifacts": artifacts,
        "beam_input_root": beam_input_root,
        "evidence_root": evidence_root,
    }
    cold_execution = run_phase(phase="cold", workspace_root=cold_root, **common)
    cold = _phase_record(cold_execution, cold_root)
    _write_json(evidence_root / "phases" / "cold.json", cold)
    if cold_execution.cache_hit:
        raise RuntimeError("cold beam_preprocess acceptance invocation unexpectedly hit cache")

    fresh_execution = run_phase(phase="fresh", workspace_root=fresh_root, **common)
    fresh = _phase_record(fresh_execution, fresh_root)
    if fresh_execution.source_run_id is None:
        fresh["source_run_id"] = cold_execution.run_id
    _write_json(evidence_root / "phases" / "fresh.json", fresh)
    if not fresh_execution.cache_hit:
        raise RuntimeError("fresh beam_preprocess acceptance invocation missed cache or executed the body")

    validation = _validate(cold, fresh)
    _write_json(evidence_root / "semantic-validation.json", validation)
    if not validation["valid"]:
        raise RuntimeError("beam_preprocess acceptance semantic validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
