"""Local evidence checker for the native structural HPC canary.

This module adds an opt-in sidecar to native workflow/container execution. It
collects resolver expectations and observed launch details into a JSON manifest,
then runs :func:`check_structural_canary` against the retained evidence directory
after the run.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import json
import os
import shlex
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence, cast

from consist import ResolvedBinding, RunResult
from consist.models.artifact import Artifact
from consist.models.run import RunBindingInvocation


_SCHEMA_VERSION = 1
STRUCTURAL_CANARY_MANIFEST_ENV = "PILATES_NATIVE_STRUCTURAL_CANARY_MANIFEST"
_ACTION_V2_CENSUS_RELATIVE_PATH = Path("evidence/action-v2.jsonl")
_ACTION_V2_CENSUS_SCHEMA_VERSION = 1


class _BindingInvocationStore(Protocol):
    """Minimal persisted-ledger surface needed by the canary census."""

    def get_binding_invocations(
        self, *, requested_run_id: str | None = None
    ) -> list[RunBindingInvocation]: ...


class _ActionV2CensusTracker(Protocol):
    """Minimal tracker surface needed to refresh one canary boundary."""

    db: _BindingInvocationStore | None

    def get_artifact(self, key_or_id: str) -> Artifact | None: ...


class _CensusConfigContract(Protocol):
    """Read-only config-contract fields retained in a census record."""

    kind: str
    adapter_name: str | None


class _CensusInputContract(Protocol):
    """Read-only input-contract fields retained in a census record."""

    status: str
    reason: str | None
    config_contract: _CensusConfigContract | None


@dataclass(frozen=True, slots=True)
class _ActiveCanaryStep:
    capture: "StructuralCanaryCapture"
    manifest_path: Path
    step: str
    roles: Mapping[str, str]
    launch_roots: Mapping[str, str]


_ACTIVE_CANARY_STEP: contextvars.ContextVar[_ActiveCanaryStep | None] = (
    contextvars.ContextVar("active_native_structural_canary_step", default=None)
)


@dataclass(frozen=True, slots=True)
class CanaryMount:
    """One host-to-container mount observed for a model launch."""

    source: str
    target: str
    mode: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target, "mode": self.mode}

    @classmethod
    def from_dict(cls, value: object, *, field: str) -> "CanaryMount":
        mapping = _mapping(value, field)
        return cls(
            source=_string(mapping, "source", field),
            target=_string(mapping, "target", field),
            mode=_string(mapping, "mode", field),
        )


@dataclass(frozen=True, slots=True)
class CanaryEvidence:
    """A retained artifact required to interpret the canary run."""

    name: str
    relative_path: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "relative_path": self.relative_path}

    @classmethod
    def from_dict(cls, value: object, *, field: str) -> "CanaryEvidence":
        mapping = _mapping(value, field)
        return cls(
            name=_string(mapping, "name", field),
            relative_path=_string(mapping, "relative_path", field),
        )


@dataclass(frozen=True, slots=True)
class CanaryLaunchObservation:
    """Expected or observed model-visible launch details."""

    model: str
    step: str
    roles: Mapping[str, str]
    launch_roots: Mapping[str, str]
    mounts: tuple[CanaryMount, ...]
    command: str
    working_dir: str
    output_roots: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.model}/{self.step}"

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "step": self.step,
            "roles": dict(sorted(self.roles.items())),
            "launch_roots": dict(sorted(self.launch_roots.items())),
            "mounts": [mount.to_dict() for mount in self.mounts],
            "command": self.command,
            "working_dir": self.working_dir,
            "output_roots": list(self.output_roots),
        }

    @classmethod
    def from_dict(cls, value: object, *, field: str) -> "CanaryLaunchObservation":
        mapping = _mapping(value, field)
        mounts_value = _sequence(mapping.get("mounts"), f"{field}.mounts")
        output_roots_value = _sequence(
            mapping.get("output_roots"), f"{field}.output_roots"
        )
        return cls(
            model=_string(mapping, "model", field),
            step=_string(mapping, "step", field),
            roles=_string_mapping(mapping.get("roles"), f"{field}.roles"),
            launch_roots=_string_mapping(
                mapping.get("launch_roots"), f"{field}.launch_roots"
            ),
            mounts=tuple(
                CanaryMount.from_dict(item, field=f"{field}.mounts[{index}]")
                for index, item in enumerate(mounts_value)
            ),
            command=_string(mapping, "command", field),
            working_dir=_string(mapping, "working_dir", field),
            output_roots=tuple(
                _sequence_strings(output_roots_value, f"{field}.output_roots")
            ),
        )


@dataclass(frozen=True, slots=True)
class StructuralCanaryManifest:
    """Expected and observed details plus retained run evidence."""

    expected_launches: tuple[CanaryLaunchObservation, ...]
    observed_launches: tuple[CanaryLaunchObservation, ...]
    required_evidence: tuple[CanaryEvidence, ...]
    evidence: tuple[CanaryEvidence, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "expected_launches": [item.to_dict() for item in self.expected_launches],
            "observed_launches": [item.to_dict() for item in self.observed_launches],
            "required_evidence": [item.to_dict() for item in self.required_evidence],
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, value: object) -> "StructuralCanaryManifest":
        mapping = _mapping(value, "manifest")
        version = mapping.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise ValueError(
                f"manifest.schema_version must be {_SCHEMA_VERSION}, got {version!r}"
            )
        return cls(
            expected_launches=_launches(mapping, "expected_launches"),
            observed_launches=_launches(mapping, "observed_launches"),
            required_evidence=_evidence(mapping, "required_evidence"),
            evidence=_evidence(mapping, "evidence"),
        )


class StructuralCanaryCapture:
    """Build a canary manifest without changing model execution behavior."""

    def __init__(
        self,
        *,
        expected_launches: Sequence[CanaryLaunchObservation],
        required_evidence: Sequence[CanaryEvidence],
    ) -> None:
        self.expected_launches = tuple(expected_launches)
        self.required_evidence = tuple(required_evidence)
        self._observed_launches: list[CanaryLaunchObservation] = []
        self._evidence: list[CanaryEvidence] = []
        _ensure_unique_keys(self.expected_launches, "expected launch observation")
        _ensure_unique_names(self.required_evidence, "required evidence")

    def record_launch(self, observation: CanaryLaunchObservation) -> None:
        if any(item.key == observation.key for item in self._observed_launches):
            raise ValueError(f"duplicate launch observation: {observation.key}")
        self._observed_launches.append(observation)

    def record_evidence(self, name: str, relative_path: str) -> None:
        if any(item.name == name for item in self._evidence):
            raise ValueError(f"duplicate evidence: {name}")
        self._evidence.append(CanaryEvidence(name, relative_path))

    def manifest(self) -> StructuralCanaryManifest:
        return StructuralCanaryManifest(
            expected_launches=self.expected_launches,
            observed_launches=tuple(self._observed_launches),
            required_evidence=self.required_evidence,
            evidence=tuple(self._evidence),
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.manifest().to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_manifest(
        cls, manifest: StructuralCanaryManifest
    ) -> "StructuralCanaryCapture":
        """Continue recording observations in an existing expectation manifest."""

        capture = cls(
            expected_launches=manifest.expected_launches,
            required_evidence=manifest.required_evidence,
        )
        capture._observed_launches.extend(manifest.observed_launches)
        capture._evidence.extend(manifest.evidence)
        return capture


@dataclass(frozen=True, slots=True)
class CanaryCheckReport:
    """Result of comparing a manifest and its retained evidence."""

    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def check_structural_canary(
    manifest: StructuralCanaryManifest, *, evidence_root: Path
) -> CanaryCheckReport:
    """Compare observations and verify required artifacts exist locally."""

    errors: list[str] = []
    expected = {item.key: item for item in manifest.expected_launches}
    observed = {item.key: item for item in manifest.observed_launches}

    for key in sorted(expected.keys() - observed.keys()):
        errors.append(f"missing launch observation: {key}")
    for key in sorted(observed.keys() - expected.keys()):
        errors.append(f"unexpected launch observation: {key}")
    for key in sorted(expected.keys() & observed.keys()):
        _compare_launch(expected[key], observed[key], errors)

    required = {item.name: item for item in manifest.required_evidence}
    retained = {item.name: item for item in manifest.evidence}
    for name in sorted(required.keys() - retained.keys()):
        errors.append(f"missing retained evidence: {name}")
    for name in sorted(required.keys() & retained.keys()):
        expected_item = required[name]
        actual_item = retained[name]
        if expected_item.relative_path != actual_item.relative_path:
            errors.append(
                f"{name} path mismatch: expected {expected_item.relative_path!r}, "
                f"observed {actual_item.relative_path!r}"
            )
        _check_evidence_path(actual_item, evidence_root, errors)

    return CanaryCheckReport(tuple(errors))


def load_structural_canary(path: Path) -> StructuralCanaryManifest:
    """Load a JSON manifest emitted by :class:`StructuralCanaryCapture`."""

    return StructuralCanaryManifest.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


@contextlib.contextmanager
def canary_step_capture(
    *,
    step: str,
    roles: Mapping[str, str],
    launch_roots: Mapping[str, str],
) -> Iterator[None]:
    """Activate opt-in collection for containers launched by one native step.

    The manifest path must name a pre-created expectation manifest.  Keeping
    expectations outside the runtime collector prevents a changed runner from
    silently approving its own mounts or command line.
    """

    path_value = os.environ.get(STRUCTURAL_CANARY_MANIFEST_ENV)
    if path_value is None:
        yield
        return
    manifest_path = Path(path_value)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"PILATES native structural canary manifest does not exist: {manifest_path}"
        )
    active = _ActiveCanaryStep(
        capture=StructuralCanaryCapture.from_manifest(
            load_structural_canary(manifest_path)
        ),
        manifest_path=manifest_path,
        step=step,
        roles=dict(roles),
        launch_roots=dict(launch_roots),
    )
    token = _ACTIVE_CANARY_STEP.set(active)
    try:
        yield
    finally:
        active.capture.write(manifest_path)
        _ACTIVE_CANARY_STEP.reset(token)


def record_active_container_launch(
    *,
    model: str,
    volumes: Mapping[str, object],
    command: Sequence[str],
    working_dir: str | None,
    output_paths: Sequence[object] | Mapping[str, object],
) -> None:
    """Append one observed container invocation when canary capture is active."""

    active = _ACTIVE_CANARY_STEP.get()
    if active is None:
        return
    active.capture.record_launch(
        CanaryLaunchObservation(
            model=model,
            step=active.step,
            roles=active.roles,
            launch_roots=active.launch_roots,
            mounts=_canary_mounts(volumes),
            command=shlex.join(command),
            working_dir=working_dir or "",
            output_roots=_output_roots(output_paths),
        )
    )


def resolved_launch_roots(runtime_kwargs: Mapping[str, object]) -> dict[str, str]:
    """Flatten typed runtime contexts into explicit resolver-owned path roots."""

    roots: dict[str, str] = {}
    for name, value in runtime_kwargs.items():
        if not (name.endswith("_launch_context") or name == "beam_launch_config"):
            continue
        if not is_dataclass(value):
            continue
        for field_name, field_value in asdict(value).items():
            if isinstance(field_value, Path):
                roots[f"{name}.{field_name}"] = str(field_value)
            elif isinstance(field_value, str):
                roots[f"{name}.{field_name}"] = field_value
    return roots


def refresh_action_v2_census(
    *,
    tracker: _ActionV2CensusTracker,
    result: RunResult,
    step: str,
    input_contract: _CensusInputContract,
    binding: object,
    selected_roles: Sequence[str],
    selected_key_by_role: Mapping[str, str],
    source_by_role: Mapping[str, str],
) -> None:
    """Refresh the opt-in canary census immediately after one step completes.

    The JSONL file is a current snapshot, not an event log: an interrupted run
    still leaves exactly one up-to-date record for each completed requested run.
    It is derived solely from Consist's persisted binding invocation ledger and
    the already-resolved PILATES contract; it cannot influence cache selection.
    """

    path_value = os.environ.get(STRUCTURAL_CANARY_MANIFEST_ENV)
    if path_value is None:
        return
    database = tracker.db
    if database is None:
        raise RuntimeError(
            "native structural canary census requires Consist metadata persistence"
        )
    requested_run_id = result.run.id
    invocations = database.get_binding_invocations(requested_run_id=requested_run_id)
    record = _census_record(
        tracker=tracker,
        result=result,
        step=step,
        input_contract=input_contract,
        binding=binding,
        invocations=invocations,
        selected_roles=selected_roles,
        selected_key_by_role=selected_key_by_role,
        source_by_role=source_by_role,
    )
    census_path = Path(path_value).parent / _ACTION_V2_CENSUS_RELATIVE_PATH
    records = _load_census_records(census_path)
    records[requested_run_id] = record
    _write_census_records(census_path, records)


def _census_record(
    *,
    tracker: _ActionV2CensusTracker,
    result: RunResult,
    step: str,
    input_contract: _CensusInputContract,
    binding: object,
    invocations: Sequence[RunBindingInvocation],
    selected_roles: Sequence[str],
    selected_key_by_role: Mapping[str, str],
    source_by_role: Mapping[str, str],
) -> dict[str, object]:
    if isinstance(binding, ResolvedBinding):
        if len(invocations) != 1:
            raise RuntimeError(
                "native structural canary expected exactly one persisted "
                f"binding invocation for {result.run.id}, found {len(invocations)}"
            )
        invocation = invocations[0]
        binding_evidence = _mapping(
            json.loads(invocation.binding_json), "binding invocation evidence"
        )
        return {
            "schema_version": _ACTION_V2_CENSUS_SCHEMA_VERSION,
            "requested_run_id": invocation.requested_run_id,
            "model": result.run.model_name,
            "step": step,
            "binding_kind": "resolved-binding-content-v1",
            "cache": {
                "outcome": invocation.cache_outcome,
                "execution_run_id": invocation.execution_run_id,
                "source_run_id": invocation.cache_source_run_id,
            },
            "input_contract": _input_contract_record(input_contract),
            "roles": _strict_census_roles(
                tracker=tracker, binding_evidence=binding_evidence
            ),
        }
    if invocations:
        raise RuntimeError(
            "ordinary native binding unexpectedly wrote persisted binding evidence "
            f"for {result.run.id}"
        )
    return {
        "schema_version": _ACTION_V2_CENSUS_SCHEMA_VERSION,
        "requested_run_id": result.run.id,
        "model": result.run.model_name,
        "step": step,
        "binding_kind": "ordinary-binding",
        "cache": {
            "outcome": "hit" if result.cache_hit else "miss",
            "execution_run_id": result.run.id,
            "source_run_id": None,
        },
        "input_contract": _input_contract_record(input_contract),
        "roles": [
            {
                "role": role,
                "selected_key": selected_key_by_role.get(role, role),
                "source": source_by_role.get(role),
                "fallback_reason": None,
                "artifact": None,
            }
            for role in sorted(selected_roles)
        ],
    }


def _input_contract_record(input_contract: _CensusInputContract) -> dict[str, object]:
    config_contract = input_contract.config_contract
    return {
        "status": input_contract.status,
        "reason": input_contract.reason,
        "config": (
            {
                "kind": config_contract.kind,
                "adapter_name": config_contract.adapter_name,
            }
            if config_contract is not None
            else None
        ),
    }


def _strict_census_roles(
    *, tracker: _ActionV2CensusTracker, binding_evidence: Mapping[str, object]
) -> list[dict[str, object]]:
    inputs = _mapping(binding_evidence.get("inputs"), "binding evidence.inputs")
    diagnostics = _mapping_or_empty(
        binding_evidence.get("diagnostics"), "binding evidence.diagnostics"
    )
    selection = _mapping_or_empty(
        diagnostics.get("selection"), "binding evidence.diagnostics.selection"
    )
    fallback_reason = selection.get("reason")
    if fallback_reason is not None and not isinstance(fallback_reason, str):
        raise ValueError(
            "binding evidence diagnostics.selection.reason must be a string"
        )
    roles: list[dict[str, object]] = []
    for parameter in sorted(inputs):
        entry = _mapping(inputs[parameter], f"binding evidence.inputs.{parameter}")
        artifact = _mapping(
            entry.get("artifact"), f"binding evidence.inputs.{parameter}.artifact"
        )
        artifact_id = _string(
            artifact, "artifact_id", f"binding evidence.inputs.{parameter}.artifact"
        )
        persisted_artifact = tracker.get_artifact(artifact_id)
        if persisted_artifact is None:
            raise RuntimeError(
                "native structural canary cannot reload bound artifact "
                f"{artifact_id} for {parameter}"
            )
        roles.append(
            {
                "role": entry.get("selected_role") or parameter,
                "source": entry.get("source"),
                "fallback_reason": fallback_reason,
                "destination": entry.get("destination"),
                "artifact": {
                    "artifact_id": artifact_id,
                    "artifact_kind": artifact.get("artifact_kind"),
                    "identity": artifact.get("identity"),
                    "selector": {
                        "driver": persisted_artifact.driver,
                        "table_path": persisted_artifact.table_path,
                        "array_path": persisted_artifact.array_path,
                    },
                },
            }
        )
    return roles


def _load_census_records(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            continue
        record = _mapping(json.loads(line), f"action-v2 census line {line_number}")
        requested_run_id = _string(
            record, "requested_run_id", f"action-v2 census line {line_number}"
        )
        records[requested_run_id] = dict(record)
    return records


def _write_census_records(
    path: Path, records: Mapping[str, Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(records[run_id], sort_keys=True) + "\n" for run_id in sorted(records)
    )
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _mapping_or_empty(value: object, field: str) -> Mapping[str, object]:
    if value is None:
        return {}
    return _mapping(value, field)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--record-evidence",
        action="append",
        nargs=2,
        metavar=("NAME", "RELATIVE_PATH"),
        default=[],
        help="append one retained evidence path before checking the manifest",
    )
    args = parser.parse_args(argv)
    manifest = load_structural_canary(args.manifest)
    if args.record_evidence:
        capture = StructuralCanaryCapture.from_manifest(manifest)
        for name, relative_path in args.record_evidence:
            capture.record_evidence(name, relative_path)
        capture.write(args.manifest)
        manifest = capture.manifest()
    report = check_structural_canary(manifest, evidence_root=args.evidence_root)
    if report.ok:
        print("native structural canary evidence: OK")
        return 0
    for error in report.errors:
        print(f"ERROR: {error}")
    return 1


def _compare_launch(
    expected: CanaryLaunchObservation,
    observed: CanaryLaunchObservation,
    errors: list[str],
) -> None:
    if dict(expected.roles) != dict(observed.roles):
        errors.append(f"{expected.key} roles mismatch")
    if dict(expected.launch_roots) != dict(observed.launch_roots):
        errors.append(f"{expected.key} launch roots mismatch")
    expected_mounts = sorted(
        (mount.source, mount.target, mount.mode) for mount in expected.mounts
    )
    observed_mounts = sorted(
        (mount.source, mount.target, mount.mode) for mount in observed.mounts
    )
    if expected_mounts != observed_mounts:
        errors.append(f"{expected.key} mounts mismatch")
    if expected.command != observed.command:
        errors.append(f"{expected.key} command mismatch")
    if expected.working_dir != observed.working_dir:
        errors.append(f"{expected.key} working directory mismatch")
    if tuple(expected.output_roots) != tuple(observed.output_roots):
        errors.append(f"{expected.key} output roots mismatch")


def _canary_mounts(volumes: Mapping[str, object]) -> tuple[CanaryMount, ...]:
    mounts: list[CanaryMount] = []
    for source, value in volumes.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"invalid canary mount mapping for {source!r}")
        target = value.get("bind")
        mode = value.get("mode", "rw")
        if not isinstance(target, str) or not isinstance(mode, str):
            raise ValueError(f"invalid canary mount mapping for {source!r}")
        mounts.append(CanaryMount(os.path.abspath(source), target, mode))
    return tuple(mounts)


def _output_roots(
    output_paths: Sequence[object] | Mapping[str, object],
) -> tuple[str, ...]:
    values = (
        output_paths.values() if isinstance(output_paths, Mapping) else output_paths
    )
    return tuple(str(value) for value in values)


def _check_evidence_path(
    evidence: CanaryEvidence, evidence_root: Path, errors: list[str]
) -> None:
    relative = Path(evidence.relative_path)
    safe = not relative.is_absolute() and ".." not in relative.parts
    if not safe:
        errors.append(f"{evidence.name} path must be relative to evidence root")
        return
    if not (evidence_root / relative).exists():
        errors.append(f"{evidence.name} does not exist: {evidence.relative_path}")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _string(mapping: Mapping[str, object], key: str, field: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}.{key} must be a non-empty string")
    return value


def _string_mapping(value: object, field: str) -> dict[str, str]:
    mapping = _mapping(value, field)
    result: dict[str, str] = {}
    for key, item in mapping.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{field} must map strings to strings")
        result[key] = item
    return result


def _sequence(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _sequence_strings(values: Sequence[object], field: str) -> list[str]:
    result: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field}[{index}] must be a non-empty string")
        result.append(value)
    return result


def _launches(
    mapping: Mapping[str, object], key: str
) -> tuple[CanaryLaunchObservation, ...]:
    values = _sequence(mapping.get(key), f"manifest.{key}")
    launches = tuple(
        CanaryLaunchObservation.from_dict(item, field=f"manifest.{key}[{index}]")
        for index, item in enumerate(values)
    )
    _ensure_unique_keys(launches, f"manifest.{key} launch observation")
    return launches


def _evidence(mapping: Mapping[str, object], key: str) -> tuple[CanaryEvidence, ...]:
    values = _sequence(mapping.get(key), f"manifest.{key}")
    evidence = tuple(
        CanaryEvidence.from_dict(item, field=f"manifest.{key}[{index}]")
        for index, item in enumerate(values)
    )
    _ensure_unique_names(evidence, f"manifest.{key} evidence")
    return evidence


def _ensure_unique_keys(values: Sequence[CanaryLaunchObservation], label: str) -> None:
    seen: set[str] = set()
    for item in values:
        if item.key in seen:
            raise ValueError(f"duplicate {label}: {item.key}")
        seen.add(item.key)


def _ensure_unique_names(values: Sequence[CanaryEvidence], label: str) -> None:
    seen: set[str] = set()
    for item in values:
        if item.name in seen:
            raise ValueError(f"duplicate {label}: {item.name}")
        seen.add(item.name)


if __name__ == "__main__":
    raise SystemExit(main())
