from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml

from pilates.config.models import WorkflowConfig
from pilates.workflows import recovery


def test_yaml_recovery_store_loads_missing_manifest_as_empty_state(tmp_path):
    manifest_path = tmp_path / ".workflow" / "year_2030_iteration_0.yaml"

    store = recovery.YamlManifestRecoveryStore(manifest_path)

    assert store.load() == {}

    decision = store.decision_for_step(step_name="missing_step", can_restore=True)
    assert decision.should_restore is False
    assert decision.entry is None
    assert decision.reason == "no manifest entry"


def test_yaml_recovery_store_preserves_schema_when_recording_and_pruning(
    monkeypatch, tmp_path
):
    manifest_path = tmp_path / ".workflow" / "year_2030_iteration_0.yaml"
    seeded_manifest = {
        "restorable_step": {
            "completed_at": "2026-01-01T00:00:00",
            "cache_hit": True,
            "run_id": "run-111",
            "outputs": {
                "foo": "bar",
                "nested": {"count": 1},
            },
        },
        "stale_step": {
            "completed_at": "2026-01-01T00:00:00",
            "cache_hit": False,
            "run_id": "run-222",
            "outputs": {},
        },
        "downstream_step": {
            "completed_at": "2026-01-01T00:00:00",
            "cache_hit": False,
            "run_id": "run-333",
            "outputs": {},
        },
    }
    load_calls: list[Path] = []
    save_calls: list[tuple[Path, dict[str, dict[str, object]]]] = []

    def fake_load_step_manifest(path: Path):
        load_calls.append(path)
        return deepcopy(seeded_manifest)

    def fake_save_step_manifest(manifest, path: Path):
        save_calls.append((path, deepcopy(manifest)))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(recovery, "load_step_manifest", fake_load_step_manifest)
    monkeypatch.setattr(recovery, "save_step_manifest", fake_save_step_manifest)

    store = recovery.YamlManifestRecoveryStore(manifest_path)

    loaded = store.load()
    assert loaded == seeded_manifest
    assert load_calls == [manifest_path]

    decision = store.decision_for_step(
        step_name="restorable_step",
        can_restore=True,
    )
    assert decision.should_restore is True
    assert decision.entry == seeded_manifest["restorable_step"]
    assert decision.reason is None

    store.prune(["stale_step", "downstream_step"])
    assert save_calls[0][0] == manifest_path
    assert save_calls[0][1] == {
        "restorable_step": seeded_manifest["restorable_step"],
    }

    store.record_step(
        step_name="new_step",
        completed_at="2026-02-02T03:04:05",
        cache_hit=False,
        run_id="run-444",
        outputs={
            "alpha": "beta",
            "count": 7,
        },
    )
    assert save_calls[1][0] == manifest_path
    assert save_calls[1][1]["restorable_step"] == seeded_manifest["restorable_step"]
    assert save_calls[1][1]["new_step"] == {
        "completed_at": "2026-02-02T03:04:05",
        "cache_hit": False,
        "run_id": "run-444",
        "outputs": {
            "alpha": "beta",
            "count": 7,
        },
    }
    assert yaml.safe_load(manifest_path.read_text(encoding="utf-8")) == save_calls[1][1]


def test_yaml_recovery_store_rejects_non_mapping_entries(monkeypatch, tmp_path):
    manifest_path = tmp_path / ".workflow" / "year_2030_iteration_0.yaml"

    monkeypatch.setattr(
        recovery,
        "load_step_manifest",
        lambda _path: {"bad_step": "not-an-entry"},
    )

    store = recovery.YamlManifestRecoveryStore(manifest_path)

    assert store.load() == {"bad_step": "not-an-entry"}
    decision = store.decision_for_step(step_name="bad_step", can_restore=True)
    assert decision.should_restore is False
    assert decision.entry is None
    assert decision.reason == "manifest entry is not a mapping"


def test_no_manifest_recovery_store_never_restores_or_persists(monkeypatch, tmp_path):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("NoManifestRecoveryStore should not persist manifest data")

    monkeypatch.setattr(recovery, "load_step_manifest", fail_if_called)
    monkeypatch.setattr(recovery, "save_step_manifest", fail_if_called)

    store = recovery.NoManifestRecoveryStore()

    assert store.load() == {}

    decision = store.decision_for_step(step_name="anything", can_restore=True)
    assert decision.should_restore is False
    assert decision.entry is None
    assert decision.reason == "manifest recovery is disabled"

    store.prune(["anything", "downstream"])
    store.record_step(
        step_name="anything",
        completed_at="2026-02-02T03:04:05",
        cache_hit=True,
        run_id="run-555",
        outputs={"ignored": True},
    )

    assert list(tmp_path.rglob("*.yaml")) == []


def test_recovery_store_for_stage_defaults_to_yaml_manifest_store(tmp_path):
    settings = SimpleNamespace(workflow=WorkflowConfig())
    manifest_path = tmp_path / ".workflow" / "land_use_year_2030.yaml"

    store = recovery.recovery_store_for_stage(
        stage_name="land_use",
        settings=settings,
        manifest_path=manifest_path,
    )

    assert isinstance(store, recovery.YamlManifestRecoveryStore)
    assert store.manifest_path == manifest_path


def test_recovery_store_for_stage_honors_disabled_stage_policy(tmp_path):
    settings = SimpleNamespace(
        workflow=WorkflowConfig(manifests={"disabled_stages": ["postprocessing"]})
    )

    disabled_store = recovery.recovery_store_for_stage(
        stage_name="postprocessing",
        settings=settings,
        manifest_path=tmp_path / ".workflow" / "postprocessing_year_2030.yaml",
    )
    enabled_store = recovery.recovery_store_for_stage(
        stage_name="land_use",
        settings=settings,
        manifest_path=tmp_path / ".workflow" / "land_use_year_2030.yaml",
    )

    assert isinstance(disabled_store, recovery.NoManifestRecoveryStore)
    assert isinstance(enabled_store, recovery.YamlManifestRecoveryStore)
