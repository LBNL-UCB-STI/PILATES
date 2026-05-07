from pilates.runtime import storage_probe


def test_storage_probe_is_disabled_by_default(monkeypatch):
    calls = []
    monkeypatch.delenv(storage_probe.STORAGE_PROBE_ENV, raising=False)
    monkeypatch.setattr(
        storage_probe,
        "log_local_storage_info",
        lambda: calls.append("called"),
    )

    storage_probe.log_local_storage_info_if_enabled()

    assert calls == []


def test_storage_probe_runs_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setenv(storage_probe.STORAGE_PROBE_ENV, "1")
    monkeypatch.setattr(
        storage_probe,
        "log_local_storage_info",
        lambda: calls.append("called"),
    )

    storage_probe.log_local_storage_info_if_enabled()

    assert calls == ["called"]
