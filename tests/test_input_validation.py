import os
import time

import pytest

from pilates.utils.input_validation import resolve_input_path, validate_inputs


def test_missing_required_input_fails_before_model(tmp_path):
    missing_path = tmp_path / "missing.h5"
    inputs = {"usim_datastore_h5": str(missing_path)}

    with pytest.raises(FileNotFoundError, match="usim_datastore_h5"):
        validate_inputs(inputs, required_keys=["usim_datastore_h5"])


def test_optional_input_missing_logs_warning(caplog):
    caplog.set_level("WARNING")
    validate_inputs({}, optional_keys=["zarr_skims"], context="beam")

    assert "Optional input 'zarr_skims' missing for beam." in caplog.text


def test_stale_input_detection(tmp_path):
    path = tmp_path / "input.h5"
    path.write_text("data")
    old_time = time.time() - 3600
    os.utime(path, (old_time, old_time))

    with pytest.raises(RuntimeError, match="stale"):
        validate_inputs(
            {"usim_datastore_h5": str(path)},
            required_keys=["usim_datastore_h5"],
            min_mtime=time.time(),
        )


def test_uri_inputs_skip_filesystem_check():
    inputs = {"asim_output_dir": "s3://bucket/output"}
    validate_inputs(inputs, required_keys=["asim_output_dir"])


def test_resolve_input_path_prefers_path_over_uri():
    class Artifact:
        path = "/tmp/path"
        uri = "s3://bucket/path"

    assert resolve_input_path(Artifact()) == "/tmp/path"
