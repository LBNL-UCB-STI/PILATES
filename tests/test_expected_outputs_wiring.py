import os

from pilates.utils.coupler_helpers import artifact_to_path, clean_expected_outputs


def test_clean_expected_outputs_drops_none_and_keeps_paths():
    outputs = {
        "a": None,
        "b": "/tmp/example",
    }
    cleaned = clean_expected_outputs(outputs)
    assert cleaned == {"b": "/tmp/example"}


def test_artifact_to_path_ignores_uri(tmp_path):
    workspace = type("W", (), {"full_path": str(tmp_path)})()
    uri = "s3://bucket/data/file"
    assert artifact_to_path(uri, workspace) == uri


def test_artifact_to_path_joins_relative(tmp_path):
    workspace = type("W", (), {"full_path": str(tmp_path)})()
    rel_path = os.path.join("data", "file.txt")
    expected = os.path.join(str(tmp_path), rel_path)
    assert artifact_to_path(rel_path, workspace) == expected
