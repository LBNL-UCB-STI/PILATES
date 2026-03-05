from pathlib import Path

from pilates.utils import step_manifest


def test_save_step_manifest_enqueues_archive_copy(monkeypatch, tmp_path):
    calls = []

    def _enqueue_archive_copy(*, key, path, workspace=None):
        calls.append((key, Path(path)))

    monkeypatch.setattr(
        "pilates.utils.coupler_helpers.enqueue_archive_copy",
        _enqueue_archive_copy,
    )

    manifest_path = tmp_path / ".workflow" / "year_2030_iteration_0.yaml"
    step_manifest.save_step_manifest({"step": {"status": "done"}}, manifest_path)

    assert manifest_path.exists()
    assert calls == [("workflow_manifest", manifest_path)]
