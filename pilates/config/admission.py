"""Inbound-admission policy configuration schemas and fingerprinting."""

from __future__ import annotations

import re
from typing import Annotated, Any, Dict, Literal, Mapping, Optional, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class PriorRunExpectation(BaseModel):
    """Select one persisted artifact from a completed Consist run."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["prior_run"] = "prior_run"
    expected_run_id: str = Field(
        ...,
        validation_alias=AliasChoices("expected_run_id", "run_id"),
        description="Completed Consist run supplying the expected artifact.",
    )
    artifact_key: str = Field(
        ..., description="Exact persisted artifact key on expected_run_id."
    )
    direction: Literal["input", "output", "any"] = Field(
        "any",
        validation_alias=AliasChoices("expected_direction", "direction"),
        description="Whether the expected artifact on expected_run_id is an input, output, or either.",
    )
    source_tracker_db: Optional[str] = Field(
        None,
        description="Optional path to an external Consist DuckDB database when expected_run_id lives in another tracker.",
    )
    expected_bytes_path: Optional[str] = Field(
        None,
        description="Optional immutable historical bytes for hash re-verification.",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_conflicting_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        if "run_id" in payload and "expected_run_id" in payload:
            if payload["run_id"] != payload["expected_run_id"]:
                raise ValueError(
                    "Conflicting 'run_id' and 'expected_run_id' values in PriorRunExpectation."
                )
            payload.pop("run_id")
        if "direction" in payload and "expected_direction" in payload:
            if payload["direction"] != payload["expected_direction"]:
                raise ValueError(
                    "Conflicting 'direction' and 'expected_direction' values in PriorRunExpectation."
                )
            payload.pop("expected_direction")
        return payload

    @field_validator("expected_run_id", "artifact_key")
    @classmethod
    def _validate_non_empty_identifier(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError(
                "PriorRunExpectation requires non-empty 'expected_run_id' and 'artifact_key'."
            )
        return text

    @property
    def run_id(self) -> str:
        """Alias matching Consist's PriorRunExpectation.run_id."""
        return self.expected_run_id

    @property
    def expected_direction(self) -> Literal["input", "output", "any"]:
        """Alias matching Consist's PriorRunExpectation.expected_direction."""
        return self.direction


class DeclaredDigestExpectation(BaseModel):
    """Select one immutable raw-file identity declared outside a prior run."""

    kind: Literal["declared_digest"] = "declared_digest"
    identity: str = Field(
        ...,
        description="Canonical sha256:file identity for the expected raw bytes.",
    )
    source_uri: Optional[str] = Field(
        None,
        description="Optional opaque source location retained outside cache identity.",
    )
    source_label: Optional[str] = Field(
        None,
        description="Optional human-readable source label retained outside cache identity.",
    )
    provenance_sidecar_path: Optional[str] = Field(
        None,
        description="Optional path to a .provenance.json sidecar retained outside cache identity.",
    )

    @field_validator("identity")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        """Require one canonical raw-file or logical-matrices SHA-256 identity before runtime."""
        if not re.fullmatch(r"sha256:(file|logical_matrices):[0-9a-f]{64}", value):
            raise ValueError(
                "identity must be sha256:file:<64 lowercase hexadecimal characters> or "
                "sha256:logical_matrices:<64 lowercase hexadecimal characters>"
            )
        return value


AdmissionExpectation = Annotated[
    Union[PriorRunExpectation, DeclaredDigestExpectation],
    Field(discriminator="kind"),
]


class AdmissionPolicyConfig(BaseModel):
    """Typed inbound-admission policy shared by model-specific configuration."""

    mode: Literal["warn", "strict"] = Field(
        "warn", description="Warn or stop when the configured expectation differs."
    )
    expectation: AdmissionExpectation


def admission_policy_fingerprint(
    policy: Optional[AdmissionPolicyConfig],
) -> Optional[Dict[str, str]]:
    """Return the portable cache selector for an admission policy.

    Runtime, report, and host-path details deliberately remain outside the
    fingerprint. Only policy mode and immutable expectation selectors affect
    cache identity.
    """
    if policy is None:
        return None

    expectation = policy.expectation
    fingerprint = {"mode": policy.mode, "kind": expectation.kind}
    if isinstance(expectation, DeclaredDigestExpectation):
        fingerprint["identity"] = expectation.identity
    else:
        fingerprint["expected_run_id"] = expectation.expected_run_id
        fingerprint["artifact_key"] = expectation.artifact_key
        if expectation.direction != "any":
            fingerprint["direction"] = expectation.direction
    return fingerprint
