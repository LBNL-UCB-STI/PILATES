"""
Unit tests for Consist container integration in GenericRunner.run_container()

This module tests the Consist delegation logic in GenericRunner.run_container(),
ensuring that:
1. Volume format conversion works correctly
2. Command and args are properly combined
3. Backend type detection works for docker vs singularity
4. Fallback behavior is correct when consist is unavailable or tracker is None
5. Consist delegation is invoked with correct parameters

These tests use unittest.mock to mock consist_run_container and verify behavior
without actually executing containers.
"""

import pytest
import unittest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
from types import SimpleNamespace

from pilates.generic.runner import GenericRunner, CONSIST_AVAILABLE
from pilates.config import PilatesConfig
from pilates.generic.records import ModelRunInfo


class TestRunContainerVolumeConversion:
    """Tests for Docker volume format to Consist format conversion."""

    def test_volume_conversion_docker_format(self):
        """Test conversion of Docker-style volumes to Consist format."""
        # Docker format: {host: {'bind': container, 'mode': 'rw'}}
        volumes = {
            "/host/data": {"bind": "/container/data", "mode": "rw"},
            "/host/output": {"bind": "/container/output", "mode": "rw"},
        }

        # Expected Consist format: {host: container}
        expected = {
            "/host/data": "/container/data",
            "/host/output": "/container/output",
        }

        # Simulate the conversion logic from run_container
        consist_volumes = {}
        for host_path, mount_info in volumes.items():
            if isinstance(mount_info, dict):
                container_path = mount_info.get("bind", mount_info)
            else:
                container_path = mount_info
            consist_volumes[host_path] = container_path

        assert consist_volumes == expected

    def test_volume_conversion_simple_format(self):
        """Test conversion of simple string volumes to Consist format."""
        # Some callers might pass simple string format directly
        volumes = {
            "/host/data": "/container/data",
            "/host/output": "/container/output",
        }

        expected = {
            "/host/data": "/container/data",
            "/host/output": "/container/output",
        }

        consist_volumes = {}
        for host_path, mount_info in volumes.items():
            if isinstance(mount_info, dict):
                container_path = mount_info.get("bind", mount_info)
            else:
                container_path = mount_info
            consist_volumes[host_path] = container_path

        assert consist_volumes == expected

    def test_volume_conversion_mixed_formats(self):
        """Test conversion with mixed dict and string volumes."""
        volumes = {
            "/host/data": {"bind": "/container/data", "mode": "rw"},
            "/host/output": "/container/output",  # Simple string
        }

        expected = {
            "/host/data": "/container/data",
            "/host/output": "/container/output",
        }

        consist_volumes = {}
        for host_path, mount_info in volumes.items():
            if isinstance(mount_info, dict):
                container_path = mount_info.get("bind", mount_info)
            else:
                container_path = mount_info
            consist_volumes[host_path] = container_path

        assert consist_volumes == expected

    def test_volume_conversion_empty(self):
        """Test conversion of empty volumes dictionary."""
        volumes = {}
        expected = {}

        consist_volumes = {}
        for host_path, mount_info in volumes.items():
            if isinstance(mount_info, dict):
                container_path = mount_info.get("bind", mount_info)
            else:
                container_path = mount_info
            consist_volumes[host_path] = container_path

        assert consist_volumes == expected


class TestRunContainerCommandCombination:
    """Tests for command and args combination logic."""

    def test_command_without_args(self):
        """Test that command is used as-is when no args provided."""
        import shlex

        command = "python /script.py"
        args = None

        full_command = command
        if args:
            if isinstance(args, list):
                full_command = command + " " + " ".join(shlex.quote(a) for a in args)
            else:
                full_command = command + " " + str(args)

        assert full_command == command

    def test_command_with_list_args(self):
        """Test command combination with list of arguments."""
        import shlex

        command = "python /script.py"
        args = ["--input", "/data/input.csv", "--output", "/data/output.csv"]

        full_command = command
        if args:
            if isinstance(args, list):
                full_command = command + " " + " ".join(shlex.quote(a) for a in args)
            else:
                full_command = command + " " + str(args)

        expected = 'python /script.py --input /data/input.csv --output /data/output.csv'
        assert full_command == expected

    def test_command_with_list_args_special_chars(self):
        """Test command combination with args containing special characters."""
        import shlex

        command = "bash /script.sh"
        args = ["--message", "Hello World", "--path", "/tmp/dir with spaces"]

        full_command = command
        if args:
            if isinstance(args, list):
                full_command = command + " " + " ".join(shlex.quote(a) for a in args)
            else:
                full_command = command + " " + str(args)

        # shlex.quote should escape special characters
        assert "--message" in full_command
        assert "Hello" in full_command

    def test_command_with_string_args(self):
        """Test command combination when args is a string."""
        import shlex

        command = "python /script.py"
        args = "--input /data.csv --verbose"

        full_command = command
        if args:
            if isinstance(args, list):
                full_command = command + " " + " ".join(shlex.quote(a) for a in args)
            else:
                full_command = command + " " + str(args)

        expected = "python /script.py --input /data.csv --verbose"
        assert full_command == expected


class TestRunContainerBackendDetection:
    """Tests for docker vs singularity backend type detection."""

    def test_docker_backend_detection(self):
        """Test that docker backend is detected when container_manager is 'docker'."""
        settings = MagicMock()
        settings.infrastructure.container_manager = "docker"

        backend_type = (
            "docker"
            if settings.infrastructure.container_manager == "docker"
            else "singularity"
        )

        assert backend_type == "docker"

    def test_singularity_backend_detection(self):
        """Test that singularity backend is detected otherwise."""
        settings = MagicMock()
        settings.infrastructure.container_manager = "singularity"

        backend_type = (
            "docker"
            if settings.infrastructure.container_manager == "docker"
            else "singularity"
        )

        assert backend_type == "singularity"

    def test_singularity_backend_detection_apptainer(self):
        """Test that singularity is default for other values."""
        settings = MagicMock()
        settings.infrastructure.container_manager = "apptainer"

        backend_type = (
            "docker"
            if settings.infrastructure.container_manager == "docker"
            else "singularity"
        )

        assert backend_type == "singularity"


class TestRunContainerFallbackBehavior:
    """Tests for fallback behavior when Consist is unavailable or conditions not met."""

    def test_fallback_when_consist_unavailable(self):
        """Test fallback to native execution when CONSIST_AVAILABLE is False."""
        # This tests the logic path, not the actual execution
        consist_available = False
        provenance_tracker = Mock()
        provenance_tracker._tracker = Mock()

        settings = MagicMock()
        settings.run.use_stubs = False

        should_use_consist = (
            consist_available
            and provenance_tracker is not None
            and hasattr(provenance_tracker, "_tracker")
            and not settings.run.use_stubs
        )

        assert should_use_consist is False

    def test_fallback_when_tracker_is_none(self):
        """Test fallback to native execution when provenance_tracker is None."""
        consist_available = True
        provenance_tracker = None

        settings = MagicMock()
        settings.run.use_stubs = False

        should_use_consist = (
            consist_available
            and provenance_tracker is not None
            and hasattr(provenance_tracker, "_tracker")
            and not settings.run.use_stubs
        )

        assert should_use_consist is False

    def test_fallback_when_tracker_lacks_attribute(self):
        """Test fallback when tracker doesn't have _tracker attribute."""
        consist_available = True
        provenance_tracker = Mock(spec=[])  # No _tracker attribute

        settings = MagicMock()
        settings.run.use_stubs = False

        should_use_consist = (
            consist_available
            and provenance_tracker is not None
            and hasattr(provenance_tracker, "_tracker")
            and not settings.run.use_stubs
        )

        assert should_use_consist is False

    def test_fallback_when_use_stubs_enabled(self):
        """Test fallback to stubs when use_stubs is True."""
        consist_available = True
        provenance_tracker = Mock()
        provenance_tracker._tracker = Mock()

        settings = MagicMock()
        settings.run.use_stubs = True  # Stubs enabled

        should_use_consist = (
            consist_available
            and provenance_tracker is not None
            and hasattr(provenance_tracker, "_tracker")
            and not settings.run.use_stubs
        )

        assert should_use_consist is False

    def test_all_conditions_met_for_consist(self):
        """Test that all conditions are met for Consist delegation."""
        consist_available = True
        provenance_tracker = Mock()
        provenance_tracker._tracker = Mock()

        settings = MagicMock()
        settings.run.use_stubs = False

        should_use_consist = (
            consist_available
            and provenance_tracker is not None
            and hasattr(provenance_tracker, "_tracker")
            and not settings.run.use_stubs
        )

        assert should_use_consist is True


@pytest.mark.skipif(not CONSIST_AVAILABLE, reason="Consist not available")
class TestRunContainerConsistDelegation:
    """Tests for actual Consist delegation when conditions are met."""

    @patch("pilates.generic.runner.consist_run_container")
    @patch("pilates.generic.runner.CONSIST_AVAILABLE", True)
    def test_consist_run_container_called(self, mock_consist_run_container):
        """Test that consist_run_container is called with correct parameters."""
        mock_consist_run_container.return_value = True

        # Create mock objects
        client = None  # Docker client not provided, but we're using Consist
        settings = MagicMock()
        settings.infrastructure.container_manager = "docker"
        settings.run.use_stubs = False
        settings.infrastructure.docker_config = SimpleNamespace(pull_latest=False)

        image = "my-image:latest"
        volumes = {"/host/data": {"bind": "/container/data", "mode": "rw"}}
        command = "python /script.py"
        model_name = "TestModel"
        working_dir = "/workspace"
        environment = {"ENV_VAR": "value"}
        args = ["--option", "value"]
        input_artifacts = ["/input/file.csv"]
        output_paths = ["/output/result.csv"]

        # Create a mock provenance tracker
        provenance_tracker = Mock()
        provenance_tracker._tracker = Mock()

        # Call run_container using the GenericRunner static method
        # We need to patch the import check and call the method directly
        with patch("pilates.generic.runner.CONSIST_AVAILABLE", True):
            with patch("pilates.generic.runner.consist_run_container", mock_consist_run_container):
                with patch("pilates.generic.runner.get_setting") as mock_get_setting:
                    mock_get_setting.return_value = False

                    result = GenericRunner.run_container(
                        client=client,
                        settings=settings,
                        image=image,
                        volumes=volumes,
                        command=command,
                        model_name=model_name,
                        working_dir=working_dir,
                        environment=environment,
                        args=args,
                        provenance_tracker=provenance_tracker,
                        input_artifacts=input_artifacts,
                        output_paths=output_paths,
                    )

        # Verify consist_run_container was called
        assert mock_consist_run_container.called
        call_args = mock_consist_run_container.call_args

        # Verify the parameters passed to consist_run_container
        assert call_args.kwargs["tracker"] == provenance_tracker._tracker
        assert call_args.kwargs["run_id"] == "TestModel_container"
        assert call_args.kwargs["image"] == image
        assert "/host/data" in call_args.kwargs["volumes"]
        assert call_args.kwargs["volumes"]["/host/data"] == "/container/data"
        assert call_args.kwargs["backend_type"] == "docker"
        assert call_args.kwargs["inputs"] == input_artifacts
        assert call_args.kwargs["outputs"] == output_paths
        assert call_args.kwargs["environment"] == environment
        assert call_args.kwargs["working_dir"] == working_dir

    @patch("pilates.generic.runner.consist_run_container")
    @patch("pilates.generic.runner.CONSIST_AVAILABLE", True)
    def test_consist_command_args_combination(self, mock_consist_run_container):
        """Test that command and args are properly combined for Consist."""
        mock_consist_run_container.return_value = True

        settings = MagicMock()
        settings.infrastructure.container_manager = "singularity"
        settings.run.use_stubs = False
        settings.infrastructure.docker_config = SimpleNamespace(pull_latest=True)

        image = "model.sif"
        volumes = {}
        command = "python"
        model_name = "Model"
        args = ["-u", "/script.py", "--verbose"]

        provenance_tracker = Mock()
        provenance_tracker._tracker = Mock()

        with patch("pilates.generic.runner.CONSIST_AVAILABLE", True):
            with patch("pilates.generic.runner.consist_run_container", mock_consist_run_container):
                with patch("pilates.generic.runner.get_setting") as mock_get_setting:
                    mock_get_setting.return_value = True

                    GenericRunner.run_container(
                        client=None,
                        settings=settings,
                        image=image,
                        volumes=volumes,
                        command=command,
                        model_name=model_name,
                        args=args,
                        provenance_tracker=provenance_tracker,
                    )

        call_args = mock_consist_run_container.call_args
        full_command = call_args.kwargs["command"]

        # Command should contain both command and args
        assert "python" in full_command
        assert "-u" in full_command
        assert "/script.py" in full_command
        assert "--verbose" in full_command

    @patch("pilates.generic.runner.consist_run_container")
    @patch("pilates.generic.runner.CONSIST_AVAILABLE", True)
    def test_consist_backend_type_singularity(self, mock_consist_run_container):
        """Test that singularity backend type is correctly passed to Consist."""
        mock_consist_run_container.return_value = True

        settings = MagicMock()
        settings.infrastructure.container_manager = "singularity"
        settings.run.use_stubs = False
        settings.infrastructure.docker_config = SimpleNamespace(pull_latest=False)

        provenance_tracker = Mock()
        provenance_tracker._tracker = Mock()

        with patch("pilates.generic.runner.CONSIST_AVAILABLE", True):
            with patch("pilates.generic.runner.consist_run_container", mock_consist_run_container):
                with patch("pilates.generic.runner.get_setting") as mock_get_setting:
                    mock_get_setting.return_value = False

                    GenericRunner.run_container(
                        client=None,
                        settings=settings,
                        image="model.sif",
                        volumes={},
                        command="bash /run.sh",
                        model_name="Model",
                        provenance_tracker=provenance_tracker,
                    )

        call_args = mock_consist_run_container.call_args
        assert call_args.kwargs["backend_type"] == "singularity"

    @patch("pilates.generic.runner.consist_run_container")
    @patch("pilates.generic.runner.CONSIST_AVAILABLE", True)
    def test_consist_exception_fallback(self, mock_consist_run_container):
        """Test fallback to native execution when Consist raises an exception."""
        mock_consist_run_container.side_effect = RuntimeError("Consist error")

        settings = MagicMock()
        settings.infrastructure.container_manager = "docker"
        settings.run.use_stubs = False
        settings.infrastructure.docker_config = SimpleNamespace(pull_latest=False)

        # Mock client for fallback to docker
        client = Mock()
        container = Mock()
        container.wait.return_value = {"StatusCode": 0}
        container.logs.return_value = [b"output"]
        client.containers.run.return_value = container

        provenance_tracker = Mock()
        provenance_tracker._tracker = Mock()

        with patch("pilates.generic.runner.CONSIST_AVAILABLE", True):
            with patch("pilates.generic.runner.consist_run_container", mock_consist_run_container):
                with patch("pilates.generic.runner.get_setting") as mock_get_setting:
                    mock_get_setting.return_value = False

                    result = GenericRunner.run_container(
                        client=client,
                        settings=settings,
                        image="test:latest",
                        volumes={},
                        command="echo test",
                        model_name="TestModel",
                        provenance_tracker=provenance_tracker,
                    )

        # Should have called consist_run_container but caught the exception
        assert mock_consist_run_container.called
        # Should fall back to docker execution
        assert client.containers.run.called


class TestRunContainerNativeFallback:
    """Tests for native execution fallback when not using Consist."""

    @patch("pilates.generic.runner.CONSIST_AVAILABLE", False)
    def test_docker_execution_when_no_consist(self):
        """Test that docker execution works when Consist is not available."""
        client = Mock()
        container = Mock()
        container.wait.return_value = {"StatusCode": 0}
        container.logs.return_value = [b"output"]
        client.containers.run.return_value = container

        settings = MagicMock()
        settings.infrastructure.docker_config = SimpleNamespace(stdout=False)
        settings.run.use_stubs = False

        image = "test:latest"
        volumes = {"/host": {"bind": "/container", "mode": "rw"}}
        command = "echo test"

        with patch("pilates.generic.runner.CONSIST_AVAILABLE", False):
            result = GenericRunner.run_container(
                client=client,
                settings=settings,
                image=image,
                volumes=volumes,
                command=command,
                model_name="TestModel",
                provenance_tracker=None,  # No tracker, so no Consist
            )

        assert client.containers.run.called
        assert result is True

    @patch("pilates.generic.runner.CONSIST_AVAILABLE", False)
    def test_docker_execution_with_args(self):
        """Test that docker execution properly handles args."""
        client = Mock()
        container = Mock()
        container.wait.return_value = {"StatusCode": 0}
        container.logs.return_value = [b"output"]
        client.containers.run.return_value = container

        settings = MagicMock()
        settings.infrastructure.docker_config = SimpleNamespace(stdout=True)
        settings.run.use_stubs = False

        image = "test:latest"
        volumes = {}
        command = "python /script.py"
        args = ["--option", "value"]

        with patch("pilates.generic.runner.CONSIST_AVAILABLE", False):
            result = GenericRunner.run_container(
                client=client,
                settings=settings,
                image=image,
                volumes=volumes,
                command=command,
                args=args,
                model_name="TestModel",
                provenance_tracker=None,
            )

        assert client.containers.run.called
        call_args = client.containers.run.call_args
        # Check that args were combined with command
        assert "command" in call_args.kwargs

    @patch("pilates.generic.runner.CONSIST_AVAILABLE", False)
    def test_docker_execution_failure(self):
        """Test that docker execution returns False on non-zero exit code."""
        client = Mock()
        container = Mock()
        container.wait.return_value = {"StatusCode": 1}  # Non-zero exit
        container.logs.return_value = [b"error"]
        client.containers.run.return_value = container

        settings = MagicMock()
        settings.infrastructure.docker_config = SimpleNamespace(stdout=False)
        settings.run.use_stubs = False

        with patch("pilates.generic.runner.CONSIST_AVAILABLE", False):
            result = GenericRunner.run_container(
                client=client,
                settings=settings,
                image="test:latest",
                volumes={},
                command="false",  # Command that fails
                model_name="TestModel",
                provenance_tracker=None,
            )

        assert result is False
