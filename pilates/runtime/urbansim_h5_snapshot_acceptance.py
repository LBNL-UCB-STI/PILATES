"""Fail-closed preflight helpers for UrbanSim HDF5 snapshot acceptance."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

import consist
import h5py
from consist import ExecutionOptions, Tracker

from pilates.config import load_config
from pilates.urbansim.outputs import UrbanSimPostprocessOutputs, UrbanSimRunOutputs
from pilates.workspace import Workspace
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_H5,
    USIM_INPUT_ARCHIVE_PREFIX,
    USIM_INPUT_MERGED_PREFIX,
)
from pilates.workflows.step_execution import execute_step
from pilates.workflows.steps import urbansim_atlas
from pilates.workflows.steps.urbansim_atlas import URBANSIM_POSTPROCESS
from workflow_state import WorkflowState


_COHORT = {"workflow_year": 2017, "forecast_year": 2019, "iteration": 0}
_REQUIRED_ROOT_TABLES = ("households", "persons", "jobs", "blocks")
_EXPECTED_COHORT = "workflow_year=2017, forecast_year=2019, iteration=0"


@dataclass(frozen=True)
class AcceptanceManifest:
    """The one admissible local UrbanSim HDF5 acceptance input."""

    usim_datastore_h5: Path
    workflow_year: int
    forecast_year: int
    iteration: int


def load_manifest(path: Path) -> AcceptanceManifest:
    """Load the exact acceptance cohort and a readable UrbanSim HDF5 source."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read acceptance manifest: {path}") from error

    if not isinstance(loaded, Mapping):
        raise ValueError("acceptance manifest must be a JSON object")
    inputs = loaded.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {"usim_datastore_h5"}:
        raise ValueError("acceptance manifest requires only inputs.usim_datastore_h5")
    value = inputs["usim_datastore_h5"]
    if not isinstance(value, str):
        raise ValueError("inputs.usim_datastore_h5 must be a path string")

    cohort = loaded.get("cohort")
    if not isinstance(cohort, Mapping):
        raise ValueError(f"acceptance cohort must be {_EXPECTED_COHORT}")
    _validate_state(cohort)

    source = Path(os.path.expandvars(value)).expanduser()
    if not source.is_file():
        raise ValueError(f"inputs.usim_datastore_h5 must be a readable file: {source}")
    try:
        with source.open("rb"):
            pass
    except OSError as error:
        raise ValueError(
            f"inputs.usim_datastore_h5 must be a readable file: {source}"
        ) from error

    return AcceptanceManifest(
        usim_datastore_h5=source.resolve(),
        workflow_year=_COHORT["workflow_year"],
        forecast_year=_COHORT["forecast_year"],
        iteration=_COHORT["iteration"],
    )


def _validate_state(state: Mapping[str, object]) -> None:
    """Reject every cohort except the one acceptance cohort."""
    if set(state) != set(_COHORT) or any(
        type(state[field_name]) is not int
        or state[field_name] != expected_value
        for field_name, expected_value in _COHORT.items()
    ):
        raise ValueError(f"acceptance cohort must be {_EXPECTED_COHORT}")


def describe_population_h5(
    path: Path, *, year: int, require_year_aliases: bool
) -> dict[str, object]:
    """Describe required root population tables without changing the source file."""
    resolved_path = path.resolve()
    if not resolved_path.is_file():
        raise ValueError(f"UrbanSim HDF5 input must be a readable file: {resolved_path}")

    try:
        with h5py.File(resolved_path, "r") as handle:
            root_tables = [
                table_name for table_name in _REQUIRED_ROOT_TABLES if table_name in handle
            ]
            missing_root_tables = [
                f"/{table_name}"
                for table_name in _REQUIRED_ROOT_TABLES
                if table_name not in handle
            ]
            if missing_root_tables:
                raise ValueError(
                    "UrbanSim HDF5 input is missing root population tables: "
                    + ", ".join(missing_root_tables)
                )

            descriptor: dict[str, object] = {
                "path": str(resolved_path),
                "size_bytes": resolved_path.stat().st_size,
                "root_tables": root_tables,
            }
            if require_year_aliases:
                aliases = [f"/{year}/{table_name}" for table_name in root_tables]
                missing_aliases = [
                    alias
                    for alias, table_name in zip(aliases, root_tables, strict=True)
                    if alias not in handle
                    or handle[alias].id != handle[table_name].id
                ]
                if missing_aliases:
                    raise ValueError(
                        "UrbanSim HDF5 input is missing exact year population aliases: "
                        + ", ".join(missing_aliases)
                    )
                descriptor["year_aliases"] = aliases
            return descriptor
    except OSError as error:
        raise ValueError(f"could not read UrbanSim HDF5 input: {resolved_path}") from error


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    """Write an acceptance record without creating unrelated directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git_revision(path: Path) -> str:
    """Return the checked-out revision for a repository-backed evidence record."""

    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unavailable"
    return completed.stdout.strip()


def _repository_root(path: Path) -> Path | None:
    """Find the closest repository root without assuming an editable layout."""

    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _runtime_environment(evidence_root: Path) -> dict[str, object]:
    """Record executable, source, and revision provenance without environment secrets."""

    pilates_root = Path(__file__).resolve().parents[2]
    consist_module = Path(consist.__file__).resolve()
    consist_root = _repository_root(consist_module.parent)
    runtime_environment: dict[str, object] = {
        "python": {"executable": sys.executable, "version": sys.version},
        "pilates": {
            "import_path": str(pilates_root),
            "revision": _git_revision(pilates_root),
        },
        "consist": {
            "import_path": str(consist_module),
            "revision": (
                _git_revision(consist_root)
                if consist_root is not None
                else "unavailable"
            ),
        },
    }
    hpc_runtime_record = evidence_root / "runtime-environment.json"
    if hpc_runtime_record.is_file():
        loaded = json.loads(hpc_runtime_record.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise RuntimeError("runtime-environment.json must be a JSON object")
        runtime_environment["hpc"] = dict(loaded)
    return runtime_environment


class _AcceptanceUrbansimPostprocessor(urbansim_atlas.UrbansimPostprocessor):
    """Materialize native postprocess outputs without invoking UrbanSim logic."""

    def postprocess(
        self,
        raw_outputs: UrbanSimRunOutputs,
        workspace: Workspace,
        model_run_hash: str | None = None,
    ) -> UrbanSimPostprocessOutputs:
        del model_run_hash
        settings = self.state.full_settings
        declared = self.expected_outputs(settings, self.state, workspace)
        merged = Path(declared[USIM_DATASTORE_H5])
        archive = merged.with_name(
            f"input_data_for_{self.state.forecast_year}_outputs.h5"
        )
        for destination in (merged, archive):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_outputs.usim_datastore_h5, destination)
        return UrbanSimPostprocessOutputs(
            usim_datastore_h5=merged,
            processed_outputs={
                f"{USIM_INPUT_MERGED_PREFIX}{self.state.forecast_year}": merged,
                f"{USIM_INPUT_ARCHIVE_PREFIX}{self.state.forecast_year}": archive,
            },
        )


def _relative_to_evidence(path: Path, *, evidence_root: Path) -> str:
    return path.resolve().relative_to(evidence_root.resolve()).as_posix()


def _record_output_descriptor(*, output: Path, year: int) -> dict[str, object]:
    return describe_population_h5(output, year=year, require_year_aliases=True)


def run_capture(*, settings_path: Path, manifest_path: Path, evidence_root: Path) -> None:
    """Execute one native UrbanSim postprocess wrapper and checkpoint its tracker."""

    evidence_root = evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    runtime_environment = _runtime_environment(evidence_root)
    manifest = load_manifest(manifest_path)
    source_descriptor = describe_population_h5(
        manifest.usim_datastore_h5,
        year=manifest.forecast_year,
        require_year_aliases=False,
    )
    settings = load_config(str(settings_path))
    state = WorkflowState.from_settings(settings)
    observed_cohort = {
        "workflow_year": state.year,
        "forecast_year": state.forecast_year,
        "iteration": state.current_inner_iter,
    }
    _validate_state(observed_cohort)
    workspace = Workspace(settings, str(evidence_root / "workspaces"), "capture")
    declared_outputs = {
        key: Path(value)
        for key, value in URBANSIM_POSTPROCESS.output_paths(
            settings=settings,
            state=state,
            workspace=workspace,
        ).items()
    }
    declared_outputs_before = {
        key: path.exists() for key, path in declared_outputs.items()
    }
    if any(declared_outputs_before.values()):
        raise RuntimeError("acceptance declared outputs unexpectedly exist before execution")
    tracker = Tracker(
        run_dir=evidence_root / "consist-runs",
        db_path=evidence_root / "provenance.duckdb",
        hashing_strategy="full",
        allow_external_paths=True,
    )
    try:
        with tracker.start_run("urbansim-h5-acceptance-inputs", "acceptance"):
            trusted = tracker.log_artifact(
                manifest.usim_datastore_h5,
                key=USIM_DATASTORE_H5,
                direction="input",
            )
        trusted_identity = str(consist.ArtifactIdentity.from_artifact(trusted))
        _write_json(
            evidence_root / "effective-input-manifest.json",
            {
                "cohort": {
                    "workflow_year": manifest.workflow_year,
                    "forecast_year": manifest.forecast_year,
                    "iteration": manifest.iteration,
                },
                "source_descriptor": source_descriptor,
                "trusted_artifact": {
                    "id": str(trusted.id),
                    "identity": trusted_identity,
                },
            },
        )
        original_postprocessor = urbansim_atlas.UrbansimPostprocessor
        urbansim_atlas.UrbansimPostprocessor = _AcceptanceUrbansimPostprocessor
        try:
            with tracker.scenario("urbansim-h5-acceptance") as scenario:
                scenario.coupler.set_from_artifact(USIM_DATASTORE_H5, trusted)
                step_identity = scenario.resolve_step_identity(
                    URBANSIM_POSTPROCESS.function,
                    year=state.year,
                    iteration=state.current_inner_iter,
                    phase="postprocess",
                    stage="land_use",
                    execution_options=ExecutionOptions(
                        input_binding="paths",
                        runtime_kwargs={
                            "settings": settings,
                            "state": state,
                            "workspace": workspace,
                        },
                    ),
                )
                resolved = URBANSIM_POSTPROCESS.resolve_inputs(
                    settings=settings,
                    state=state,
                    workspace=workspace,
                    coupler=scenario.coupler,
                    step_identity=step_identity,
                )
                result, _ = execute_step(
                    scenario=scenario,
                    definition=URBANSIM_POSTPROCESS,
                    settings=settings,
                    state=state,
                    workspace=workspace,
                    stage="land_use",
                    year=state.year,
                    iteration=state.current_inner_iter,
                    phase="postprocess",
                    resolved_inputs=resolved,
                )
        finally:
            urbansim_atlas.UrbansimPostprocessor = original_postprocessor

        input_identity = result.run.meta["input_identity"]
        if (
            input_identity.get("mode") != "action-v2"
            or input_identity.get("strict_input_count") != 1
            or not input_identity.get("strict_binding_identity")
        ):
            raise RuntimeError("capture did not persist one strict action-v2 HDF5 input")

        output_path = Path(
            URBANSIM_POSTPROCESS.output_paths(
                settings=settings,
                state=state,
                workspace=workspace,
                resolved_inputs=resolved,
            )["usim_population_source_h5"]
        )
        output_descriptor = _record_output_descriptor(
            output=output_path, year=manifest.forecast_year
        )
        declared_outputs_after = {
            key: path.is_file() for key, path in declared_outputs.items()
        }
        if not all(declared_outputs_after.values()):
            raise RuntimeError("native postprocess did not produce every declared output")
        with tracker.start_run("urbansim-h5-acceptance-override", "acceptance"):
            override = tracker.log_artifact(
                trusted,
                key=USIM_DATASTORE_H5,
                direction="input",
                content_hash=trusted.hash,
                force_hash_override=True,
            )
        if (
            override.id == trusted.id
            or override.run_id is not None
            or override.meta["hash_semantics"]["source"] != "caller_supplied"
        ):
            raise RuntimeError("capture did not create an unowned later HDF5 observation")

        snapshot_path = evidence_root / "snapshots" / "tracker.duckdb"
        tracker.snapshot_db(str(snapshot_path), checkpoint=True)
        if not snapshot_path.is_file():
            raise RuntimeError("capture tracker snapshot was not created")
        run_snapshot = tracker.get_run_record(str(result.run.id))
        if run_snapshot is None:
            raise RuntimeError("capture could not read the completed native run")
        _write_json(
            evidence_root / "capture.json",
            {
                "completed_run_id": str(result.run.id),
                "completed_run_snapshot": run_snapshot.model_dump(
                    mode="json", warnings=False
                ),
                "declared_outputs_after": declared_outputs_after,
                "declared_outputs_before": declared_outputs_before,
                "input_identity": dict(input_identity),
                "override_artifact_id": str(override.id),
                "output_descriptor": output_descriptor,
                "output_h5_path": _relative_to_evidence(
                    output_path, evidence_root=evidence_root
                ),
                "snapshot_path": _relative_to_evidence(
                    snapshot_path, evidence_root=evidence_root
                ),
                "source_descriptor": source_descriptor,
                "source_h5_valid": True,
                "strict_binding_valid": True,
                "trusted_artifact_id": str(trusted.id),
                "trusted_identity": trusted_identity,
                "override_is_unowned": True,
                "snapshot_created": True,
                "output_h5_aliases_valid": True,
                "process_id": os.getpid(),
                "runtime_environment": runtime_environment,
            },
        )
    finally:
        tracker.db.engine.dispose()


def validate(
    *, capture: Mapping[str, object], reconciliation: Mapping[str, object]
) -> dict[str, bool]:
    """Construct the acceptance verdict from capture and read-only evidence."""

    source_h5_valid = capture.get("source_h5_valid") is True
    strict_binding_valid = reconciliation.get("action_v2_strict_binding_valid") is True
    override_is_unowned = capture.get("override_is_unowned") is True
    snapshot_created = capture.get("snapshot_created") is True
    fresh_snapshot_readable = reconciliation.get("fresh_snapshot_readable") is True
    persisted_identity_unchanged = (
        reconciliation.get("persisted_identity") == capture.get("trusted_identity")
    )
    persisted_strict_link_trusted = (
        reconciliation.get("persisted_strict_link_trusted") is True
    )
    output_h5_aliases_valid = capture.get("output_h5_aliases_valid") is True
    valid = all(
        (
            source_h5_valid,
            strict_binding_valid,
            override_is_unowned,
            snapshot_created,
            fresh_snapshot_readable,
            persisted_identity_unchanged,
            persisted_strict_link_trusted,
            output_h5_aliases_valid,
        )
    )
    return {
        "source_h5_valid": source_h5_valid,
        "strict_binding_valid": strict_binding_valid,
        "override_is_unowned": override_is_unowned,
        "snapshot_created": snapshot_created,
        "fresh_snapshot_readable": fresh_snapshot_readable,
        "persisted_identity_unchanged": persisted_identity_unchanged,
        "persisted_strict_link_trusted": persisted_strict_link_trusted,
        "output_h5_aliases_valid": output_h5_aliases_valid,
        "valid": valid,
    }


def run_reconciliation(*, evidence_root: Path) -> None:
    """Read only the capture snapshot and verify the persisted strict input."""

    evidence_root = evidence_root.resolve()
    capture = json.loads((evidence_root / "capture.json").read_text(encoding="utf-8"))
    runtime_environment = _runtime_environment(evidence_root)
    snapshot_ref = capture.get("snapshot_path")
    if not isinstance(snapshot_ref, str):
        raise RuntimeError("capture.json is missing snapshot_path")
    snapshot_path = (evidence_root / snapshot_ref).resolve()
    if not snapshot_path.is_file():
        raise RuntimeError(f"required tracker snapshot is absent: {snapshot_ref}")

    snapshot_tracker = Tracker(
        run_dir=evidence_root / "consist-runs",
        db_path=snapshot_path,
        allow_external_paths=True,
        access_mode="read_only",
    )
    try:
        completed_run_id = capture.get("completed_run_id")
        trusted_artifact_id = capture.get("trusted_artifact_id")
        override_artifact_id = capture.get("override_artifact_id")
        if not all(
            isinstance(value, str)
            for value in (completed_run_id, trusted_artifact_id, override_artifact_id)
        ):
            raise RuntimeError("capture.json is missing required artifact identifiers")
        completed_run = snapshot_tracker.get_run(completed_run_id)
        if completed_run is None or completed_run.status != "completed":
            raise RuntimeError("capture snapshot does not contain a completed native run")
        input_identity = completed_run.meta.get("input_identity")
        action_v2_strict_binding_valid = (
            isinstance(input_identity, Mapping)
            and input_identity.get("mode") == "action-v2"
            and type(input_identity.get("strict_input_count")) is int
            and input_identity.get("strict_input_count") == 1
            and isinstance(input_identity.get("strict_binding_identity"), str)
            and bool(input_identity["strict_binding_identity"].strip())
            and input_identity.get("bindings") == []
        )
        inputs = snapshot_tracker.get_run_inputs(completed_run_id)
        artifact = inputs.get(USIM_DATASTORE_H5)
        if artifact is None or str(artifact.id) != trusted_artifact_id:
            raise RuntimeError("capture snapshot does not preserve the trusted HDF5 input")
        if snapshot_tracker.get_artifact(override_artifact_id) is None:
            raise RuntimeError("capture snapshot does not retain the later HDF5 override")
        if any(str(candidate.id) == override_artifact_id for candidate in inputs.values()):
            raise RuntimeError("capture snapshot selected the later HDF5 override")
        persisted_identity = str(consist.ArtifactIdentity.from_artifact(artifact))
        persisted_strict_link_trusted = (
            action_v2_strict_binding_valid
            and len(inputs) == 1
            and str(artifact.id) == trusted_artifact_id
            and persisted_identity == capture.get("trusted_identity")
        )
        reconciliation: dict[str, object] = {
            "action_v2_strict_binding_valid": action_v2_strict_binding_valid,
            "completed_run_id": completed_run_id,
            "fresh_snapshot_readable": True,
            "persisted_identity": persisted_identity,
            "persisted_strict_link_trusted": persisted_strict_link_trusted,
            "process_id": os.getpid(),
            "runtime_environment": runtime_environment,
            "trusted_artifact_id": trusted_artifact_id,
        }
    finally:
        snapshot_tracker.db.engine.dispose()
    _write_json(evidence_root / "reconciliation.json", reconciliation)
    validation = validate(capture=capture, reconciliation=reconciliation)
    _write_json(evidence_root / "validation.json", validation)
    if not validation["valid"]:
        raise RuntimeError("UrbanSim HDF5 snapshot acceptance validation failed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--settings", required=True, type=Path)
    capture.add_argument("--manifest", required=True, type=Path)
    capture.add_argument("--evidence-root", required=True, type=Path)
    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--evidence-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "capture":
        run_capture(
            settings_path=args.settings,
            manifest_path=args.manifest,
            evidence_root=args.evidence_root,
        )
    else:
        run_reconciliation(evidence_root=args.evidence_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
