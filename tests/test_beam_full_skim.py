"""
Unit tests for BEAM full-skim mode (BackgroundSkimsCreatorApp).

Tests verify that BeamRunner correctly detects full-skim configuration and
constructs the appropriate command and environment for the BackgroundSkimsCreatorApp.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestBeamFullSkimMode:
    """Tests for BEAM full-skim mode configuration and command construction."""

    def _make_skim_config(
        self,
        run_schedule="standalone",
        router_type="r5+gh",
        skims_geo_type="taz",
        skims_kind="od",
        peak_hours=None,
        modes_to_build=None,
        parallelism_ratio=0.8,
        output_filename="background_skims.csv",
        linkstats_file=None,
    ):
        """Helper to create a full_skim config object."""
        if peak_hours is None:
            peak_hours = [8.5]
        if modes_to_build is None:
            modes_to_build = {"drive": True, "walk": False, "transit": False}

        return SimpleNamespace(
            run_schedule=run_schedule,
            router_type=router_type,
            skims_geo_type=skims_geo_type,
            skims_kind=skims_kind,
            peak_hours=peak_hours,
            modes_to_build=modes_to_build,
            parallelism_thread_ratio=parallelism_ratio,
            output_filename=output_filename,
            linkstats_file=linkstats_file,
        )

    def _make_settings(self, full_skim_cfg=None):
        """Helper to create settings with beam config."""
        run_cfg = SimpleNamespace(region="test_region")
        beam_cfg = SimpleNamespace(
            config="beam.conf",
            memory="180g",
            full_skim=full_skim_cfg,
        )
        shared_cfg = SimpleNamespace(skims=SimpleNamespace(fname="skims.omx"))
        return SimpleNamespace(run=run_cfg, beam=beam_cfg, shared=shared_cfg)

    def _setup_runner_test(self, skim_cfg, *, full_skim=False):
        """Common test setup for runner tests."""
        from pilates.beam.runner import BeamFullSkimRunner, BeamRunner
        from pilates.generic.records import RecordStore

        settings = self._make_settings(full_skim_cfg=skim_cfg)
        state = MagicMock()
        state.full_settings = settings
        state.current_year = 2020
        state.current_inner_iter = 0
        runner_cls = BeamFullSkimRunner if full_skim else BeamRunner
        model_name = "beam_full_skim" if full_skim else "beam"
        runner = runner_cls(model_name, state)

        workspace = MagicMock()
        workspace.get_beam_mutable_data_dir.return_value = "/tmp/beam_input"
        workspace.get_beam_output_dir.return_value = "/tmp/beam_output"

        store = RecordStore(recordList=[])

        return runner, workspace, store

    @patch("pilates.beam.runner._calculate_optimal_parallelism")
    @patch("pilates.beam.runner.get_setting")
    @patch("pilates.beam.runner.BeamFullSkimRunner.run_container")
    @patch("pilates.beam.runner.os.makedirs")
    @patch("pilates.beam.runner.os.path.exists")
    def test_full_skim_command_construction_all_params(
        self,
        mock_exists,
        mock_makedirs,
        mock_run_container,
        mock_get_setting,
        mock_calc_parallelism,
    ):
        """Test that all full-skim config parameters are included in command."""
        mock_run_container.return_value = True
        mock_exists.return_value = True
        mock_get_setting.return_value = "skims.omx"
        mock_calc_parallelism.return_value = 16

        skim_cfg = self._make_skim_config(
            run_schedule="standalone",
            router_type="r5+gh",
            skims_geo_type="taz",
            skims_kind="od",
            peak_hours=[8.5, 17.5],
            modes_to_build={"drive": True, "walk": True, "transit": False},
            parallelism_ratio=0.5,
            linkstats_file=None,
        )

        runner, workspace, store = self._setup_runner_test(skim_cfg, full_skim=True)

        with patch.object(runner, "get_model_and_image", return_value=("beam", "beam:latest")):
            runner._run(store, workspace)

        # Verify run_container was called
        assert mock_run_container.called

        # Extract the command argument
        call_kwargs = mock_run_container.call_args.kwargs
        command = call_kwargs["command"]

        # Verify all parameters are present in command
        assert "--configPath=" in command
        assert (
            "--output=/app/output/test_region/year-2020-iteration-0/skimsODFull.csv.gz"
            in command
        )
        assert "--parallelism=16" in command
        assert "--routerType=r5+gh" in command
        assert "--skimsGeoType=taz" in command
        assert "--skimsKind=od" in command
        assert "--peakHours=8.5,17.5" in command
        assert "--modesToBuild=" in command
        assert "drive" in command and "walk" in command
        assert "--linkstatsPath=" not in command

        # Verify environment has BEAM_MAIN_CLASS override
        environment = call_kwargs["environment"]
        assert environment["BEAM_MAIN_CLASS"] == "scripts.FullSkimsCreatorApp"

    @patch("pilates.beam.runner._calculate_optimal_parallelism")
    @patch("pilates.beam.runner.get_setting")
    @patch("pilates.beam.runner.BeamFullSkimRunner.run_container")
    @patch("pilates.beam.runner.os.makedirs")
    @patch("pilates.beam.runner.os.path.exists")
    def test_full_skim_without_optional_linkstats(
        self,
        mock_exists,
        mock_makedirs,
        mock_run_container,
        mock_get_setting,
        mock_calc_parallelism,
    ):
        """Test command construction without optional linkstats_file."""
        mock_run_container.return_value = True
        mock_exists.return_value = True
        mock_get_setting.return_value = "skims.omx"
        mock_calc_parallelism.return_value = 12

        skim_cfg = self._make_skim_config(
            run_schedule="standalone",
            linkstats_file=None,  # No linkstats
        )

        runner, workspace, store = self._setup_runner_test(skim_cfg, full_skim=True)

        with patch.object(runner, "get_model_and_image", return_value=("beam", "beam:latest")):
            runner._run(store, workspace)

        call_kwargs = mock_run_container.call_args.kwargs
        command = call_kwargs["command"]

        # Verify linkstats is NOT in command
        assert "--linkstatsPath=" not in command

        # Verify other required params are present
        assert "--configPath=" in command
        assert "--output=" in command
        assert "--parallelism=" in command

    @patch("pilates.beam.runner._calculate_optimal_parallelism")
    @patch("pilates.beam.runner.get_setting")
    @patch("pilates.beam.runner.BeamFullSkimRunner.run_container")
    @patch("pilates.beam.runner.os.makedirs")
    @patch("pilates.beam.runner.os.path.exists")
    def test_full_skim_single_mode_enabled(
        self,
        mock_exists,
        mock_makedirs,
        mock_run_container,
        mock_get_setting,
        mock_calc_parallelism,
    ):
        """Test modes_to_build with only one mode enabled."""
        mock_run_container.return_value = True
        mock_exists.return_value = True
        mock_get_setting.return_value = "skims.omx"
        mock_calc_parallelism.return_value = 12

        skim_cfg = self._make_skim_config(
            run_schedule="standalone",
            modes_to_build={"drive": True, "walk": False, "transit": False},
        )

        runner, workspace, store = self._setup_runner_test(skim_cfg, full_skim=True)

        with patch.object(runner, "get_model_and_image", return_value=("beam", "beam:latest")):
            runner._run(store, workspace)

        call_kwargs = mock_run_container.call_args.kwargs
        command = call_kwargs["command"]

        # Verify only drive mode is included
        assert "--modesToBuild=drive" in command

    @patch("pilates.beam.runner._calculate_optimal_parallelism")
    @patch("pilates.beam.runner.get_setting")
    @patch("pilates.beam.runner.BeamFullSkimRunner.run_container")
    @patch("pilates.beam.runner.os.makedirs")
    @patch("pilates.beam.runner.os.path.exists")
    def test_full_skim_all_modes_enabled(
        self,
        mock_exists,
        mock_makedirs,
        mock_run_container,
        mock_get_setting,
        mock_calc_parallelism,
    ):
        """Test modes_to_build with all modes enabled."""
        mock_run_container.return_value = True
        mock_exists.return_value = True
        mock_get_setting.return_value = "skims.omx"
        mock_calc_parallelism.return_value = 12

        skim_cfg = self._make_skim_config(
            run_schedule="standalone",
            modes_to_build={"drive": True, "walk": True, "transit": True},
        )

        runner, workspace, store = self._setup_runner_test(skim_cfg, full_skim=True)

        with patch.object(runner, "get_model_and_image", return_value=("beam", "beam:latest")):
            runner._run(store, workspace)

        call_kwargs = mock_run_container.call_args.kwargs
        command = call_kwargs["command"]

        # Verify all modes are included (order may vary)
        assert "--modesToBuild=" in command
        assert "drive" in command
        assert "walk" in command
        assert "transit" in command

    @patch("pilates.beam.runner.get_setting")
    @patch("pilates.beam.runner.BeamRunner.run_container")
    @patch("pilates.beam.runner.rename_beam_output_directory")
    @patch("pilates.beam.runner.os.makedirs")
    @patch("pilates.beam.runner.os.path.exists")
    def test_normal_beam_run_without_full_skim(
        self, mock_exists, mock_makedirs, mock_rename, mock_run_container, mock_get_setting
    ):
        """Test that normal BEAM run works when full_skim is disabled."""
        mock_run_container.return_value = True
        mock_exists.return_value = True
        mock_rename.return_value = ("/old", "/new")
        mock_get_setting.return_value = "skims.omx"

        skim_cfg = self._make_skim_config(run_schedule="disabled")
        runner, workspace, store = self._setup_runner_test(skim_cfg)

        with patch.object(runner, "get_model_and_image", return_value=("beam", "beam:latest")):
            with patch.object(runner, "gather_outputs", return_value=[]):
                runner._run(store, workspace)

        call_kwargs = mock_run_container.call_args.kwargs
        command = call_kwargs["command"]
        environment = call_kwargs["environment"]

        # Verify normal BEAM command format (not full-skim)
        assert command.startswith("--config=")
        assert "--output=" not in command
        assert "--routerType=" not in command
        assert "--skimsGeoType=" not in command

        # Verify BEAM_MAIN_CLASS is NOT overridden
        assert "BEAM_MAIN_CLASS" not in environment

    @patch("pilates.beam.runner._calculate_optimal_parallelism")
    @patch("pilates.beam.runner.get_setting")
    @patch("pilates.beam.runner.BeamFullSkimRunner.run_container")
    @patch("pilates.beam.runner.os.makedirs")
    @patch("pilates.beam.runner.os.path.exists")
    def test_full_skim_no_modes_enabled(
        self,
        mock_exists,
        mock_makedirs,
        mock_run_container,
        mock_get_setting,
        mock_calc_parallelism,
    ):
        """Test behavior when no modes are enabled."""
        mock_run_container.return_value = True
        mock_exists.return_value = True
        mock_get_setting.return_value = "skims.omx"
        mock_calc_parallelism.return_value = 12

        skim_cfg = self._make_skim_config(
            run_schedule="standalone",
            modes_to_build={"drive": False, "walk": False, "transit": False},
        )

        runner, workspace, store = self._setup_runner_test(skim_cfg, full_skim=True)

        with patch.object(runner, "get_model_and_image", return_value=("beam", "beam:latest")):
            runner._run(store, workspace)

        call_kwargs = mock_run_container.call_args.kwargs
        command = call_kwargs["command"]

        # When no modes are enabled, modesToBuild should not appear
        assert "--modesToBuild=" not in command

    @patch("pilates.beam.runner._calculate_optimal_parallelism")
    @patch("pilates.beam.runner.get_setting")
    @patch("pilates.beam.runner.BeamFullSkimRunner.run_container")
    @patch("pilates.beam.runner.os.makedirs")
    @patch("pilates.beam.runner.os.path.exists")
    def test_full_skim_multiple_peak_hours(
        self,
        mock_exists,
        mock_makedirs,
        mock_run_container,
        mock_get_setting,
        mock_calc_parallelism,
    ):
        """Test peak hours formatting with multiple values."""
        mock_run_container.return_value = True
        mock_exists.return_value = True
        mock_get_setting.return_value = "skims.omx"
        mock_calc_parallelism.return_value = 12

        skim_cfg = self._make_skim_config(
            run_schedule="standalone",
            peak_hours=[6.0, 8.5, 12.0, 17.5, 20.0],
        )

        runner, workspace, store = self._setup_runner_test(skim_cfg, full_skim=True)

        with patch.object(runner, "get_model_and_image", return_value=("beam", "beam:latest")):
            runner._run(store, workspace)

        call_kwargs = mock_run_container.call_args.kwargs
        command = call_kwargs["command"]

        # Verify all peak hours are formatted correctly
        assert "--peakHours=6.0,8.5,12.0,17.5,20.0" in command
