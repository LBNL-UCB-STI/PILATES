from pilates.workflows.step_runner import common_runtime_kwargs


def test_common_runtime_kwargs_includes_basics() -> None:
    runtime_kwargs = common_runtime_kwargs(
        settings="settings",
        state="state",
        workspace="workspace",
        extra_key=123,
    )
    assert runtime_kwargs == {
        "settings": "settings",
        "state": "state",
        "workspace": "workspace",
        "extra_key": 123,
    }
