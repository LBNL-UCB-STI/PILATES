"""
Unit tests for BEAM full-skim mode (BackgroundSkimsCreatorApp).

Tests verify that BeamRunner correctly detects full-skim configuration and
constructs the appropriate command and environment for the BackgroundSkimsCreatorApp.
"""

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch


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
        from pilates.beam.launch_config import BeamLaunchConfig
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
        launch_root = Path("/tmp/beam_launch") / ("full_skim" if full_skim else "beam")
        launch_config = BeamLaunchConfig(
            root=launch_root,
            primary_config=launch_root / "beam.conf",
        )

        return runner, workspace, store, launch_config

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

        runner, workspace, store, launch_config = self._setup_runner_test(
            skim_cfg, full_skim=True
        )

        with patch.object(
            runner, "get_model_and_image", return_value=("beam", "beam:latest")
        ):
            runner._run(store, workspace, launch_config)

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

        runner, workspace, store, launch_config = self._setup_runner_test(
            skim_cfg, full_skim=True
        )

        with patch.object(
            runner, "get_model_and_image", return_value=("beam", "beam:latest")
        ):
            runner._run(store, workspace, launch_config)

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

        runner, workspace, store, launch_config = self._setup_runner_test(
            skim_cfg, full_skim=True
        )

        with patch.object(
            runner, "get_model_and_image", return_value=("beam", "beam:latest")
        ):
            runner._run(store, workspace, launch_config)

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

        runner, workspace, store, launch_config = self._setup_runner_test(
            skim_cfg, full_skim=True
        )

        with patch.object(
            runner, "get_model_and_image", return_value=("beam", "beam:latest")
        ):
            runner._run(store, workspace, launch_config)

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
        self,
        mock_exists,
        mock_makedirs,
        mock_rename,
        mock_run_container,
        mock_get_setting,
    ):
        """Test that normal BEAM run works when full_skim is disabled."""
        mock_run_container.return_value = True
        mock_exists.return_value = True
        mock_rename.return_value = ("/old", "/new")
        mock_get_setting.return_value = "skims.omx"

        skim_cfg = self._make_skim_config(run_schedule="disabled")
        runner, workspace, store, launch_config = self._setup_runner_test(skim_cfg)

        with patch.object(
            runner, "get_model_and_image", return_value=("beam", "beam:latest")
        ):
            with patch.object(runner, "gather_outputs", return_value=[]):
                runner._run(store, workspace, launch_config)

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

        runner, workspace, store, launch_config = self._setup_runner_test(
            skim_cfg, full_skim=True
        )

        with patch.object(
            runner, "get_model_and_image", return_value=("beam", "beam:latest")
        ):
            runner._run(store, workspace, launch_config)

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

        runner, workspace, store, launch_config = self._setup_runner_test(
            skim_cfg, full_skim=True
        )

        with patch.object(
            runner, "get_model_and_image", return_value=("beam", "beam:latest")
        ):
            runner._run(store, workspace, launch_config)

        call_kwargs = mock_run_container.call_args.kwargs
        command = call_kwargs["command"]

        # Verify all peak hours are formatted correctly
        assert "--peakHours=6.0,8.5,12.0,17.5,20.0" in command

    def test_full_skims_creator_config_defaults_and_facet(self):
        """FullSkimsCreatorConfig defaults to activitySimOmx and serializes to facet."""
        from pilates.config.models import FullSkimsCreatorConfig

        cfg = FullSkimsCreatorConfig()
        assert cfg.run_schedule == "standalone"
        assert cfg.skims_kind == "activitySimOmx"
        assert cfg.modes_to_build == {"drive": True, "walk": True, "transit": True}
        assert cfg.bifurcate_tolls is True
        assert cfg.generate_return_trips is True
        assert cfg.toll_file_path is None
        assert cfg.directory2 is None
        assert cfg.od_skims_path is None
        assert cfg.input_path is None
        assert cfg.linkstats_path is None

        facet = cfg.to_consist_facet()
        assert facet["skims_kind"] == "activitySimOmx"
        assert facet["bifurcate_tolls"] is True
        assert facet["generate_return_trips"] is True

    def test_build_beam_identity_config_includes_full_skim_when_present(self):
        """build_beam_identity_config includes full_skim facet when configured."""
        from pilates.config.models import FullSkimsCreatorConfig
        from pilates.utils.consist_config import build_beam_identity_config

        full_skim = FullSkimsCreatorConfig(
            skims_kind="activitySimOmx",
            bifurcate_tolls=False,
            toll_file_path="toll-prices.csv",
        )
        settings = SimpleNamespace(
            write_skims_to_omx=True,
            beam=SimpleNamespace(
                config="beam.conf",
                sample=1.0,
                replanning_portion=0.4,
                memory="180g",
                discard_plans_every_year=False,
                max_plans_memory=5,
                router_directory="r5",
                scenario_folder="scenario",
                admission=None,
                full_skim=full_skim,
            ),
        )
        identity_cfg = build_beam_identity_config(settings)
        assert "full_skim" in identity_cfg
        assert identity_cfg["full_skim"]["skims_kind"] == "activitySimOmx"
        assert identity_cfg["full_skim"]["bifurcate_tolls"] is False
        assert identity_cfg["full_skim"]["toll_file_path"] == "toll-prices.csv"

    def test_expected_outputs_maps_skims_kind_to_canonical_filename(self):
        """BeamFullSkimRunner.expected_outputs maps skims_kind to .omx, .zarr, or .csv.gz."""
        from pilates.beam.runner import BeamFullSkimRunner
        from pilates.config.models import FullSkimsCreatorConfig
        from pilates.workflows.artifact_keys import BEAM_FULL_SKIMS

        workspace = MagicMock()
        workspace.get_beam_output_dir.return_value = "/tmp/beam_output"
        state = SimpleNamespace(current_year=2025, current_inner_iter=1)

        for skims_kind, expected_name in (
            ("activitySimOmx", "skims.omx"),
            ("activitySimZarr", "skims.zarr"),
            ("activitySim", "skimsODFull.csv.gz"),
            ("od", "skimsODFull.csv.gz"),
        ):
            settings = self._make_settings(
                full_skim_cfg=FullSkimsCreatorConfig(skims_kind=skims_kind)
            )
            outputs = BeamFullSkimRunner.expected_outputs(settings, state, workspace)
            assert outputs[BEAM_FULL_SKIMS] == (
                Path("/tmp/beam_output/test_region/year-2025-iteration-1")
                / expected_name
            )

    @patch("pilates.beam.runner._calculate_optimal_parallelism")
    @patch("pilates.beam.runner.BeamFullSkimRunner.run_container")
    @patch("pilates.beam.runner.os.makedirs")
    @patch("pilates.beam.runner.os.path.exists")
    def test_full_skim_new_cli_options_and_input_artifacts(
        self,
        mock_exists,
        mock_makedirs,
        mock_run_container,
        mock_calc_parallelism,
    ):
        """New FullSkimsCreatorConfig CLI options and input artifacts are wired."""
        from pilates.config.models import FullSkimsCreatorConfig

        mock_run_container.return_value = True
        mock_exists.return_value = True
        mock_calc_parallelism.return_value = 8

        skim_cfg = FullSkimsCreatorConfig(
            run_schedule="standalone",
            skims_kind="activitySimOmx",
            bifurcate_tolls=True,
            generate_return_trips=False,
            toll_file_path="toll-prices.csv",
            directory2="r5-ferry",
            od_skims_path="warmstart_od.csv.gz",
            input_path="od_pairs.csv",
            linkstats_path="congested_linkstats.csv.gz",
        )

        runner, workspace, store, launch_config = self._setup_runner_test(
            skim_cfg, full_skim=True
        )

        with patch.object(
            runner, "get_model_and_image", return_value=("beam", "beam:latest")
        ):
            result_store = runner._run(store, workspace, launch_config)

        call_kwargs = mock_run_container.call_args.kwargs
        command = call_kwargs["command"]

        assert (
            "--output=/app/output/test_region/year-2020-iteration-0/skims.omx"
            in command
        )
        assert "--skimsKind=activitySimOmx" in command
        assert "--bifurcateTolls=true" in command
        assert "--generateReturnTrips=false" in command
        assert "--tollFilePath=/app/input/test_region/toll-prices.csv" in command
        assert "--directory2=/app/input/test_region/r5-ferry" in command
        assert "--ODSkimsPath=/app/input/test_region/warmstart_od.csv.gz" in command
        assert "--input=/app/input/test_region/od_pairs.csv" in command
        assert (
            "--linkstatsPath=/app/input/test_region/congested_linkstats.csv.gz"
            in command
        )

        input_artifacts = call_kwargs["input_artifacts"]
        launch_region = str(launch_config.root / "test_region")
        assert f"{launch_region}/toll-prices.csv" in input_artifacts
        assert f"{launch_region}/r5-ferry" in input_artifacts
        assert f"{launch_region}/warmstart_od.csv.gz" in input_artifacts
        assert f"{launch_region}/od_pairs.csv" in input_artifacts
        assert f"{launch_region}/congested_linkstats.csv.gz" in input_artifacts

        output_path = result_store.to_mapping()["beam_full_skims"]
        assert str(output_path).endswith("test_region/year-2020-iteration-0/skims.omx")

    def test_select_latest_linkstats_supports_linkstats_warmstart(self):
        """_select_latest_linkstats_path selects linkstats_warmstart when present."""
        from pilates.beam.runner import _select_latest_linkstats_path
        from pilates.generic.records import FileRecord, RecordStore

        store = RecordStore(
            recordList=[
                FileRecord(
                    file_path="/tmp/beam_input/test_region/warmstart.linkstats.csv.gz",
                    short_name="linkstats_warmstart",
                    description="Warmstart linkstats",
                )
            ]
        )
        selected = _select_latest_linkstats_path(
            store, "/tmp/beam_input", "/tmp/beam_output"
        )
        assert selected == "/app/input/test_region/warmstart.linkstats.csv.gz"
