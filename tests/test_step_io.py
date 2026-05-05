from types import SimpleNamespace

from pilates.workflows import step_io


def test_merge_expected_inputs_no_override() -> None:
    target = {"a": "keep", "b": "old"}
    expected = {"b": "new", "c": "add"}

    merged = step_io.merge_expected_inputs(target, expected, prefer_expected=False)

    assert merged == {"a": "keep", "b": "old", "c": "add"}


def test_merge_expected_inputs_with_override() -> None:
    target = {"a": "keep", "b": "old"}
    expected = {"b": "new"}

    merged = step_io.merge_expected_inputs(target, expected, prefer_expected=True)

    assert merged == {"a": "keep", "b": "new"}


def test_merge_expected_outputs_filters_none() -> None:
    target = {"a": "keep"}
    expected = {"b": None, "c": "/tmp/path"}

    merged = step_io.merge_expected_outputs(target, expected)

    assert merged == {"a": "keep", "c": "/tmp/path"}


def test_merge_expected_model_inputs_uses_expected(monkeypatch) -> None:
    def _expected_inputs_for(model_name, settings, state, workspace):
        return {"a": "from_expected", "b": "expected"}

    monkeypatch.setattr(step_io, "expected_inputs_for", _expected_inputs_for)

    base = {"a": "base"}
    merged = step_io.merge_model_expected_inputs(
        "demo",
        base,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(),
    )

    assert merged == {"a": "base", "b": "expected"}
