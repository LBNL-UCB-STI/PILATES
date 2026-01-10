import os
from types import SimpleNamespace

from pilates.utils.coupler_helpers import artifact_to_path
from pilates.utils.input_validation import resolve_input_path
from pilates.workspace import Workspace


def _make_settings():
    return {
        "urbansim": {"local_mutable_data_folder": "usim_mutable"},
        "activitysim": {
            "local_mutable_data_folder": "asim_mutable",
            "local_mutable_configs_folder": "asim_configs",
            "local_output_folder": "asim_output",
        },
        "beam": {
            "local_mutable_data_folder": "beam_mutable",
            "local_output_folder": "beam_output",
        },
        "atlas": {
            "host_mutable_input_folder": "atlas_input",
            "host_output_folder": "atlas_output",
        },
    }


def test_workspace_path_resolution(tmp_path):
    settings = _make_settings()
    workspace = Workspace(settings, str(tmp_path), folder_name="run")

    assert workspace.get_usim_mutable_data_dir() == os.path.join(
        workspace.full_path, "usim_mutable"
    )
    assert workspace.get_asim_output_dir() == os.path.join(
        workspace.full_path, "asim_output"
    )
    assert workspace.get_beam_output_dir() == os.path.join(
        workspace.full_path, "beam_output"
    )
    assert workspace.get_atlas_output_dir() == os.path.join(
        workspace.full_path, "atlas_output"
    )


def test_artifact_path_resolution(tmp_path):
    workspace = SimpleNamespace(full_path=str(tmp_path))

    class Artifact:
        uri = "s3://bucket/path"
        path = "relative/path"

    assert resolve_input_path(Artifact()) == "relative/path"
    assert artifact_to_path(Artifact(), workspace) == os.path.join(
        str(tmp_path), "relative/path"
    )
