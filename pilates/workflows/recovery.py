from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from pilates.utils.step_manifest import load_step_manifest, save_step_manifest


@dataclass(frozen=True)
class StepRecoveryDecision:
    """
    Decision returned by a recovery store for an individual step.
    """

    should_restore: bool
    entry: Mapping[str, Any] | None
    reason: str | None = None


class StepRecoveryStore(Protocol):
    """
    Small recovery boundary for workflow step replay and persistence.
    """

    def load(self) -> Mapping[str, Any]:
        """Load the current persisted recovery state."""

    def decision_for_step(
        self,
        *,
        step_name: str,
        can_restore: bool,
    ) -> StepRecoveryDecision:
        """Decide whether a step should be restored from persisted state."""

    def prune(self, step_names: Sequence[str]) -> None:
        """Remove stale or downstream entries from persisted state."""

    def record_step(
        self,
        *,
        step_name: str,
        completed_at: str,
        cache_hit: bool,
        run_id: str | None,
        outputs: Mapping[str, Any] | None,
    ) -> None:
        """Record completed step metadata back to persisted state."""

    def uses_persisted_entries(self) -> bool:
        """Return whether this store restores or records persisted entries."""


@dataclass
class YamlManifestRecoveryStore:
    """
    Recovery store backed by the existing YAML step-manifest helpers.
    """

    manifest_path: Path
    _manifest: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _loaded: bool = field(default=False, init=False, repr=False)

    def load(self) -> Mapping[str, Any]:
        manifest = load_step_manifest(self.manifest_path)
        self._manifest = dict(manifest or {})
        self._loaded = True
        return self._manifest

    def decision_for_step(
        self,
        *,
        step_name: str,
        can_restore: bool,
    ) -> StepRecoveryDecision:
        manifest = self._ensure_loaded()
        entry = manifest.get(step_name)
        if entry is None:
            return StepRecoveryDecision(
                should_restore=False,
                entry=None,
                reason="no manifest entry",
            )
        if not isinstance(entry, Mapping):
            return StepRecoveryDecision(
                should_restore=False,
                entry=None,
                reason="manifest entry is not a mapping",
            )
        if not can_restore:
            return StepRecoveryDecision(
                should_restore=False,
                entry=entry,
                reason="step is not eligible for manifest restore",
            )
        return StepRecoveryDecision(
            should_restore=True,
            entry=entry,
            reason=None,
        )

    def prune(self, step_names: Sequence[str]) -> None:
        manifest = self._ensure_loaded()
        changed = False
        for step_name in step_names:
            key = str(step_name)
            if key in manifest:
                del manifest[key]
                changed = True
        if changed:
            save_step_manifest(manifest, self.manifest_path)

    def record_step(
        self,
        *,
        step_name: str,
        completed_at: str,
        cache_hit: bool,
        run_id: str | None,
        outputs: Mapping[str, Any] | None,
    ) -> None:
        manifest = self._ensure_loaded()
        manifest[step_name] = {
            "completed_at": completed_at,
            "cache_hit": bool(cache_hit),
            "run_id": run_id,
            "outputs": dict(outputs or {}),
        }
        save_step_manifest(manifest, self.manifest_path)

    def uses_persisted_entries(self) -> bool:
        return True

    def _ensure_loaded(self) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        return self._manifest


class NoManifestRecoveryStore:
    """
    No-op recovery store for stages that should never restore from manifests.
    """

    def load(self) -> Mapping[str, Any]:
        return {}

    def decision_for_step(
        self,
        *,
        step_name: str,
        can_restore: bool,
    ) -> StepRecoveryDecision:
        return StepRecoveryDecision(
            should_restore=False,
            entry=None,
            reason="manifest recovery is disabled",
        )

    def prune(self, step_names: Sequence[str]) -> None:
        return None

    def record_step(
        self,
        *,
        step_name: str,
        completed_at: str,
        cache_hit: bool,
        run_id: str | None,
        outputs: Mapping[str, Any] | None,
    ) -> None:
        return None

    def uses_persisted_entries(self) -> bool:
        return False
