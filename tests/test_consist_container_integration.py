"""
Unit tests for Consist container integration in GenericRunner.run_container().

Tests the Consist delegation path and validates argument mapping and
fallback behavior when Consist is disabled or unavailable.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from pilates.generic.runner import GenericRunner


class TestRunContainerConsistDelegation:
    """Tests for GenericRunner.run_container delegation to Consist."""

    @patch("pilates.generic.runner.get_setting")
    @patch("consist.integrations.containers.run_container")
    @patch("pilates.generic.runner.cr.current_tracker")
    def test_delegation_to_consist_success(
        self,
        mock_current_tracker,
        mock_consist_run_container,
        mock_get_setting,
    ):
        """Test that run_container calls Consist with properly mapped arguments."""
        mock_consist_run_container.return_value = True

        # Setup inputs
        settings = MagicMock()

        # Configure get_setting mock to return specific values for this test
        def get_setting_side_effect(obj, key, default=None):
            if key == "infrastructure.container_manager":
                return "docker"
            if key == "infrastructure.docker_config.pull_latest":
                return True
            return default

        mock_get_setting.side_effect = get_setting_side_effect

        image = "test/image:tag"
        volumes = {
            "/host/data": {"bind": "/container/data", "mode": "rw"},
            "/host/output": "/container/output",  # Simple string format
        }
        command = "python script.py"
        args = ["--arg", "val"]
        model_name = "test_model"

        # Mock tracker
        tracker = Mock()
        mock_current_tracker.return_value = tracker

        # Execute
        result = GenericRunner.run_container(
            client=None,  # Ignored
            settings=settings,
            image=image,
            volumes=volumes,
            command=command,
            model_name=model_name,
            args=args,
        )

        assert result is True
        assert mock_consist_run_container.called

        # Verify arguments
        call_kwargs = mock_consist_run_container.call_args.kwargs

        # 1. Volumes adapted correctly (stripped 'bind'/'mode')
        assert call_kwargs["volumes"] == {
            "/host/data": "/container/data",
            "/host/output": "/container/output",
        }

        # 2. Command split and extended with args
        expected_cmd = ["python", "script.py", "--arg", "val"]
        assert call_kwargs["command"] == expected_cmd

        # 3. Settings passed correctly
        assert call_kwargs["backend_type"] == "docker"
        assert call_kwargs["pull_latest"] is True
        assert call_kwargs["run_id"] == "test_model_container"
        assert call_kwargs["tracker"] == tracker

    @patch("pilates.generic.runner.cr.current_tracker")
    def test_missing_tracker_raises_error(self, mock_current_tracker):
        """Test that missing tracker raises RuntimeError when Consist is enabled."""
        mock_current_tracker.return_value = None
        with pytest.raises(RuntimeError, match="Consist tracker must be active"):
            GenericRunner.run_container(
                client=None,
                settings=MagicMock(),
                image="img",
                volumes={},
                command="cmd",
                model_name="model",
            )

    @patch("consist.integrations.containers.run_container")
    @patch("pilates.generic.runner.cr.current_tracker")
    def test_command_string_args_combination(
        self,
        mock_current_tracker,
        mock_consist_run_container,
    ):
        """Test that string args are correctly split and combined."""
        mock_consist_run_container.return_value = True

        # Setup inputs
        tracker = Mock()
        mock_current_tracker.return_value = tracker

        GenericRunner.run_container(
            client=None,
            settings=MagicMock(),
            image="img",
            volumes={},
            command="python script.py",
            model_name="model",
            args="--arg val",  # String format args
        )

        call_kwargs = mock_consist_run_container.call_args.kwargs
        expected_cmd = ["python", "script.py", "--arg", "val"]
        assert call_kwargs["command"] == expected_cmd

    @patch("pilates.generic.runner.GenericRunner._run_container_direct")
    @patch("consist.integrations.containers.run_container")
    @patch("pilates.generic.runner.cr.current_tracker")
    def test_exception_handling_returns_false(
        self,
        mock_current_tracker,
        mock_consist_run_container,
        mock_run_container_direct,
    ):
        """Test that exceptions from Consist fall back to direct execution."""
        mock_consist_run_container.side_effect = Exception("Consist failed")
        mock_run_container_direct.return_value = False

        tracker = Mock()
        mock_current_tracker.return_value = tracker

        result = GenericRunner.run_container(
            client=None,
            settings=MagicMock(),
            image="img",
            volumes={},
            command="cmd",
            model_name="model",
        )

        assert result is False
        assert mock_run_container_direct.called

    @patch("pilates.generic.runner.GenericRunner._run_container_direct")
    @patch("consist.integrations.containers.run_container")
    @patch("pilates.generic.runner.cr.current_tracker")
    def test_exception_handling_runs_pre_fallback_hook(
        self,
        mock_current_tracker,
        mock_consist_run_container,
        mock_run_container_direct,
    ):
        call_order = []

        def _cleanup():
            call_order.append("cleanup")

        def _direct(**_kwargs):
            call_order.append("direct")
            return True

        mock_consist_run_container.side_effect = Exception("Consist failed")
        mock_run_container_direct.side_effect = _direct
        mock_current_tracker.return_value = Mock()

        result = GenericRunner.run_container(
            client=None,
            settings=MagicMock(),
            image="img",
            volumes={},
            command="cmd",
            model_name="model",
            before_direct_fallback=_cleanup,
        )

        assert result is True
        assert call_order == ["cleanup", "direct"]

    @patch("pilates.generic.runner.GenericRunner._run_container_direct")
    @patch("consist.integrations.containers.run_container")
    @patch("pilates.generic.runner.cr.current_tracker")
    def test_non_capable_tracker_skips_consist_delegation(
        self,
        mock_current_tracker,
        mock_consist_run_container,
        mock_run_container_direct,
    ):
        """Trackers without mount/start_run support should bypass Consist container API."""
        mock_run_container_direct.return_value = True

        class NoopLikeTracker:
            pass

        mock_current_tracker.return_value = NoopLikeTracker()

        result = GenericRunner.run_container(
            client=None,
            settings=MagicMock(),
            image="img",
            volumes={},
            command="cmd",
            model_name="model",
        )

        assert result is True
        assert not mock_consist_run_container.called
        assert mock_run_container_direct.called

    @patch("pilates.generic.runner.get_setting")
    @patch("consist.integrations.containers.run_container")
    @patch("pilates.generic.runner.cr.current_tracker")
    def test_backend_defaults(
        self,
        mock_current_tracker,
        mock_consist_run_container,
        mock_get_setting,
    ):
        """Test default backend selection."""
        mock_consist_run_container.return_value = True

        settings = MagicMock()

        # Configure get_setting mock to return singularity
        def get_setting_side_effect(obj, key, default=None):
            if key == "infrastructure.container_manager":
                return "singularity"
            return default

        mock_get_setting.side_effect = get_setting_side_effect

        tracker = Mock()
        mock_current_tracker.return_value = tracker

        GenericRunner.run_container(
            client=None,
            settings=settings,
            image="img",
            volumes={},
            command="cmd",
            model_name="model",
        )

        assert (
            mock_consist_run_container.call_args.kwargs["backend_type"] == "singularity"
        )

    @patch("pilates.generic.runner.get_setting")
    @patch("consist.integrations.containers.run_container")
    @patch("pilates.generic.runner.cr.current_tracker")
    def test_delegation_enables_debug_stream_when_stdout_requested(
        self,
        mock_current_tracker,
        mock_consist_run_container,
        mock_get_setting,
    ):
        tracker = Mock()
        mock_current_tracker.return_value = tracker

        def get_setting_side_effect(_obj, key, default=None):
            if key == "infrastructure.container_manager":
                return "docker"
            if key == "infrastructure.docker_config.pull_latest":
                return False
            if key == "infrastructure.docker_config.stdout":
                return True
            return default

        mock_get_setting.side_effect = get_setting_side_effect

        def _check_env(**_kwargs):
            assert os.environ.get("CONSIST_CONTAINER_DEBUG_STREAM") == "1"
            return True

        mock_consist_run_container.side_effect = _check_env

        os.environ.pop("CONSIST_CONTAINER_DEBUG_STREAM", None)

        result = GenericRunner.run_container(
            client=None,
            settings=MagicMock(),
            image="img",
            volumes={},
            command="cmd",
            model_name="model",
        )

        assert result is True
        assert "CONSIST_CONTAINER_DEBUG_STREAM" not in os.environ


@patch("pilates.generic.runner.subprocess.run")
@patch("pilates.generic.runner.shutil.which")
def test_direct_singularity_prefers_singularity_and_uses_writable_tmpfs(
    mock_which, mock_subprocess_run, tmp_path: Path
):
    def _which(name):
        if name == "singularity":
            return "/usr/bin/singularity"
        if name == "apptainer":
            return "/usr/bin/apptainer"
        return None

    mock_which.side_effect = _which

    host_mount = tmp_path / "mount"
    host_mount.mkdir()

    result = GenericRunner._run_singularity_container(
        image="docker://example/image:tag",
        command=["python", "script.py"],
        mounts=[(str(host_mount), "/container/mount", "rw")],
        environment={"PYTHONNOUSERSITE": "1"},
        working_dir="/workdir",
    )

    assert result is True
    cmd = mock_subprocess_run.call_args.args[0]
    env = mock_subprocess_run.call_args.kwargs["env"]
    assert cmd[:4] == ["singularity", "run", "--cleanenv", "--writable-tmpfs"]
    assert "--pwd" in cmd
    assert env["SINGULARITYENV_PYTHONNOUSERSITE"] == "1"


@patch("pilates.generic.runner.subprocess.run")
@patch("pilates.generic.runner.shutil.which")
def test_direct_singularity_falls_back_to_apptainer_when_needed(
    mock_which, mock_subprocess_run, tmp_path: Path
):
    def _which(name):
        if name == "apptainer":
            return "/usr/bin/apptainer"
        return None

    mock_which.side_effect = _which

    host_mount = tmp_path / "mount"
    host_mount.mkdir()

    result = GenericRunner._run_singularity_container(
        image="docker://example/image:tag",
        command=["python", "script.py"],
        mounts=[(str(host_mount), "/container/mount", "rw")],
        environment={"PYTHONNOUSERSITE": "1"},
        working_dir="/workdir",
    )

    assert result is True
    cmd = mock_subprocess_run.call_args.args[0]
    env = mock_subprocess_run.call_args.kwargs["env"]
    assert cmd[:4] == ["apptainer", "run", "--cleanenv", "--writable-tmpfs"]
    assert env["APPTAINERENV_PYTHONNOUSERSITE"] == "1"
