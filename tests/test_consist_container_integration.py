"""
Unit tests for Consist container integration in GenericRunner.run_container()

Tests the pure delegation logic in GenericRunner.run_container().
Since legacy fallback logic has been removed, these tests strictly enforce:
1. Consist availability
2. Provenance tracker presence
3. Correct argument mapping to consist.integrations.containers.run_container
"""

import pytest
import shlex
from unittest.mock import Mock, MagicMock, patch
from types import SimpleNamespace

from pilates.generic.runner import GenericRunner

class TestRunContainerConsistDelegation:
    """Tests for GenericRunner.run_container delegation to Consist."""

    @patch("pilates.generic.runner.get_setting")
    @patch("pilates.generic.runner.consist_run_container")
    def test_delegation_to_consist_success(self, mock_consist_run_container, mock_get_setting):
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
            "/host/output": "/container/output"  # Simple string format
        }
        command = "python script.py"
        args = ["--arg", "val"]
        model_name = "test_model"

        # Mock tracker
        tracker = Mock()
        tracker._tracker = Mock()

        # Execute
        result = GenericRunner.run_container(
            client=None,  # Ignored
            settings=settings,
            image=image,
            volumes=volumes,
            command=command,
            model_name=model_name,
            args=args,
            provenance_tracker=tracker
        )

        assert result is True
        assert mock_consist_run_container.called

        # Verify arguments
        call_kwargs = mock_consist_run_container.call_args.kwargs

        # 1. Volumes adapted correctly (stripped 'bind'/'mode')
        assert call_kwargs["volumes"] == {
            "/host/data": "/container/data",
            "/host/output": "/container/output"
        }

        # 2. Command split and extended with args
        expected_cmd = ["python", "script.py", "--arg", "val"]
        assert call_kwargs["command"] == expected_cmd

        # 3. Settings passed correctly
        assert call_kwargs["backend_type"] == "docker"
        assert call_kwargs["pull_latest"] is True
        assert call_kwargs["run_id"] == "test_model_container"
        assert call_kwargs["tracker"] == tracker._tracker

    def test_missing_tracker_raises_error(self):
        """Test that missing provenance_tracker raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Consist-backed provenance_tracker is required"):
            GenericRunner.run_container(
                client=None,
                settings=MagicMock(),
                image="img",
                volumes={},
                command="cmd",
                model_name="model",
                provenance_tracker=None # Missing
            )

    @patch("pilates.generic.runner.consist_run_container")
    def test_command_string_args_combination(self, mock_consist_run_container):
        """Test that string args are correctly split and combined."""
        mock_consist_run_container.return_value = True

        # Setup inputs
        tracker = Mock()
        tracker._tracker = Mock()

        GenericRunner.run_container(
            client=None,
            settings=MagicMock(),
            image="img",
            volumes={},
            command="python script.py",
            model_name="model",
            args="--arg val", # String format args
            provenance_tracker=tracker
        )

        call_kwargs = mock_consist_run_container.call_args.kwargs
        expected_cmd = ["python", "script.py", "--arg", "val"]
        assert call_kwargs["command"] == expected_cmd

    @patch("pilates.generic.runner.consist_run_container")
    def test_exception_handling_returns_false(self, mock_consist_run_container):
        """Test that exceptions from Consist are caught and return False."""
        mock_consist_run_container.side_effect = Exception("Consist failed")

        tracker = Mock()
        tracker._tracker = Mock()

        result = GenericRunner.run_container(
            client=None,
            settings=MagicMock(),
            image="img",
            volumes={},
            command="cmd",
            model_name="model",
            provenance_tracker=tracker
        )

        assert result is False

    @patch("pilates.generic.runner.get_setting")
    @patch("pilates.generic.runner.consist_run_container")
    def test_backend_defaults(self, mock_consist_run_container, mock_get_setting):
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
        tracker._tracker = Mock()

        GenericRunner.run_container(
            client=None,
            settings=settings,
            image="img",
            volumes={},
            command="cmd",
            model_name="model",
            provenance_tracker=tracker
        )

        assert mock_consist_run_container.call_args.kwargs["backend_type"] == "singularity"