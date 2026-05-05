"""
Unit tests for Consist container integration in GenericRunner.run_container().

Tests the Consist delegation path and validates argument mapping and
strict failure behavior when Consist support is unavailable.
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

    @patch("consist.integrations.containers.run_container")
    @patch("pilates.generic.runner.cr.current_tracker")
    def test_output_mapping_passes_canonical_keys_through(
        self,
        mock_current_tracker,
        mock_consist_run_container,
    ):
        mock_consist_run_container.return_value = True

        tracker = Mock()
        mock_current_tracker.return_value = tracker

        GenericRunner.run_container(
            client=None,
            settings=MagicMock(),
            image="img",
            volumes={},
            command="cmd",
            model_name="model",
            output_paths={"usim_datastore_h5": "/tmp/model_data_2023.h5"},
        )

        call_kwargs = mock_consist_run_container.call_args.kwargs
        assert call_kwargs["outputs"] == {
            "usim_datastore_h5": "/tmp/model_data_2023.h5"
        }

    @patch("consist.integrations.containers.run_container")
    @patch("pilates.generic.runner.cr.current_tracker")
    def test_exception_from_consist_propagates(
        self,
        mock_current_tracker,
        mock_consist_run_container,
    ):
        """Consist failures should fail closed rather than using a direct backend."""
        mock_consist_run_container.side_effect = Exception("Consist failed")

        tracker = Mock()
        mock_current_tracker.return_value = tracker

        with pytest.raises(Exception, match="Consist failed"):
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
    def test_active_tracker_is_passed_through_without_local_capability_probe(
        self,
        mock_current_tracker,
        mock_consist_run_container,
    ):
        """GenericRunner no longer performs a local mount/start_run capability probe."""

        class NoopLikeTracker:
            pass

        mock_consist_run_container.return_value = True
        mock_current_tracker.return_value = NoopLikeTracker()

        assert (
            GenericRunner.run_container(
                client=None,
                settings=MagicMock(),
                image="img",
                volumes={},
                command="cmd",
                model_name="model",
            )
            is True
        )

        assert mock_consist_run_container.called

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

    @patch("pilates.generic.runner.get_setting")
    @patch("consist.integrations.containers.run_container")
    @patch("pilates.generic.runner.cr.current_tracker")
    def test_delegation_uses_per_run_tmpdir_for_singularity(
        self,
        mock_current_tracker,
        mock_consist_run_container,
        mock_get_setting,
        tmp_path: Path,
    ):
        run_tmp = tmp_path / "run-tmp"
        run_tmp.mkdir()

        def get_setting_side_effect(_obj, key, default=None):
            if key == "infrastructure.container_manager":
                return "singularity"
            return default

        mock_get_setting.side_effect = get_setting_side_effect
        mock_current_tracker.return_value = Mock()

        def _check_env(**_kwargs):
            expected_base = str(run_tmp / ".container_runtime")
            assert os.environ["TMPDIR"] == expected_base
            assert os.environ["APPTAINER_CACHEDIR"] == expected_base + "/.apptainer/cache"
            assert os.environ["APPTAINER_TMPDIR"] == expected_base + "/.apptainer/tmp"
            assert os.environ["SINGULARITY_CACHEDIR"] == expected_base + "/.apptainer/cache"
            assert os.environ["SINGULARITY_TMPDIR"] == expected_base + "/.apptainer/tmp"
            return True

        mock_consist_run_container.side_effect = _check_env

        result = GenericRunner.run_container(
            client=None,
            settings=MagicMock(),
            image="img",
            volumes={str(run_tmp): {"bind": "/tmp", "mode": "rw"}},
            command="cmd",
            model_name="model",
        )

        assert result is True
