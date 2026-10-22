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

    def test_expected_outputs_emits_final_skims_omx_for_activitysim_omx(self):
        """BeamFullSkimRunner.expected_outputs includes final_skims_omx when skims_kind is activitySimOmx."""
        from pilates.beam.runner import BeamFullSkimRunner
        from pilates.config.models import FullSkimsCreatorConfig
        from pilates.workflows.artifact_keys import BEAM_FULL_SKIMS, FINAL_SKIMS_OMX

        workspace = MagicMock()
        workspace.get_beam_mutable_data_dir.return_value = "/tmp/beam_input"
        workspace.get_beam_output_dir.return_value = "/tmp/beam_output"
        state = SimpleNamespace(current_year=2025, current_inner_iter=1)

        omx_settings = self._make_settings(
            full_skim_cfg=FullSkimsCreatorConfig(skims_kind="activitySimOmx")
        )
        omx_outputs = BeamFullSkimRunner.expected_outputs(
            omx_settings, state, workspace
        )
        assert FINAL_SKIMS_OMX in omx_outputs
        assert omx_outputs[FINAL_SKIMS_OMX] == Path(
            "/tmp/beam_input/test_region/skims.omx"
        )
        assert omx_outputs[BEAM_FULL_SKIMS] == Path(
            "/tmp/beam_output/test_region/year-2025-iteration-1/skims.omx"
        )

        csv_settings = self._make_settings(
            full_skim_cfg=FullSkimsCreatorConfig(skims_kind="od")
        )
        csv_outputs = BeamFullSkimRunner.expected_outputs(
            csv_settings, state, workspace
        )
        assert FINAL_SKIMS_OMX not in csv_outputs

    def test_resolve_beam_full_skim_inputs_without_population_inputs(self):
        """_resolve_beam_full_skim_inputs succeeds when population artifacts are absent from coupler."""
        from pilates.workflows.steps.beam import _resolve_beam_full_skim_inputs

        settings = self._make_settings()
        state = SimpleNamespace(current_year=2020, current_inner_iter=0)
        workspace = MagicMock()
        workspace.get_beam_mutable_data_dir.return_value = "/tmp/beam_input"
        workspace.get_beam_output_dir.return_value = "/tmp/beam_output"
        coupler = {}

        resolved = _resolve_beam_full_skim_inputs(
            settings=settings,
            state=state,
            workspace=workspace,
            coupler=coupler,
        )
        assert resolved.step_name == "beam_full_skim"
        assert resolved.required_roles == ()
        assert resolved.binding.inputs == {}

    def test_resolve_beam_full_skim_inputs_standalone_bypasses_coupler_population_inputs(
        self, tmp_path
    ):
        """_resolve_beam_full_skim_inputs skips coupler population inputs in standalone mode."""
        from pilates.workflows.artifact_keys import (
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
            BEAM_PLANS_IN,
            LINKSTATS_WARMSTART,
        )
        from pilates.workflows.steps.beam import _resolve_beam_full_skim_inputs

        settings = self._make_settings(
            full_skim_cfg=self._make_skim_config(run_schedule="standalone")
        )
        state = SimpleNamespace(current_year=2020, current_inner_iter=0)
        workspace = MagicMock()
        workspace.full_path = str(tmp_path)
        workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam_input")
        workspace.get_beam_output_dir.return_value = str(tmp_path / "beam_output")

        plans_path = tmp_path / "plans.csv.gz"
        plans_path.write_bytes(b"plans")
        hh_path = tmp_path / "households.csv.gz"
        hh_path.write_bytes(b"hh")
        persons_path = tmp_path / "persons.csv.gz"
        persons_path.write_bytes(b"persons")
        warmstart_path = tmp_path / "warmstart.parquet"
        warmstart_path.write_bytes(b"warmstart")

        coupler = {
            BEAM_PLANS_IN: str(plans_path),
            BEAM_HOUSEHOLDS_IN: str(hh_path),
            BEAM_PERSONS_IN: str(persons_path),
            LINKSTATS_WARMSTART: str(warmstart_path),
        }

        resolved = _resolve_beam_full_skim_inputs(
            settings=settings,
            state=state,
            workspace=workspace,
            coupler=coupler,
        )
        assert resolved.step_name == "beam_full_skim"
        assert resolved.required_roles == ()
        assert LINKSTATS_WARMSTART in resolved.binding.inputs
        assert BEAM_PLANS_IN not in resolved.binding.inputs
        assert BEAM_HOUSEHOLDS_IN not in resolved.binding.inputs
        assert BEAM_PERSONS_IN not in resolved.binding.inputs

    def test_resolve_beam_full_skim_inputs_final_iter_binds_coupler_population_inputs(
        self, tmp_path
    ):
        """_resolve_beam_full_skim_inputs binds coupler population inputs when not standalone."""
        from pilates.workflows.artifact_keys import (
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
            BEAM_PLANS_IN,
            LINKSTATS_WARMSTART,
        )
        from pilates.workflows.steps.beam import _resolve_beam_full_skim_inputs

        settings = self._make_settings(
            full_skim_cfg=self._make_skim_config(run_schedule="after_final_iteration")
        )
        state = SimpleNamespace(current_year=2020, current_inner_iter=0)
        workspace = MagicMock()
        workspace.full_path = str(tmp_path)
        workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam_input")
        workspace.get_beam_output_dir.return_value = str(tmp_path / "beam_output")

        plans_path = tmp_path / "plans.csv.gz"
        plans_path.write_bytes(b"plans")
        hh_path = tmp_path / "households.csv.gz"
        hh_path.write_bytes(b"hh")
        persons_path = tmp_path / "persons.csv.gz"
        persons_path.write_bytes(b"persons")
        warmstart_path = tmp_path / "warmstart.parquet"
        warmstart_path.write_bytes(b"warmstart")

        coupler = {
            BEAM_PLANS_IN: str(plans_path),
            BEAM_HOUSEHOLDS_IN: str(hh_path),
            BEAM_PERSONS_IN: str(persons_path),
            LINKSTATS_WARMSTART: str(warmstart_path),
        }

        resolved = _resolve_beam_full_skim_inputs(
            settings=settings,
            state=state,
            workspace=workspace,
            coupler=coupler,
        )
        assert resolved.step_name == "beam_full_skim"
        assert resolved.required_roles == ()
        assert LINKSTATS_WARMSTART in resolved.binding.inputs
        assert BEAM_PLANS_IN in resolved.binding.inputs
        assert BEAM_HOUSEHOLDS_IN in resolved.binding.inputs
        assert BEAM_PERSONS_IN in resolved.binding.inputs

    def test_prepare_r5_raw_rebuild_removes_tolls_dat_and_directory2_caches(
        self, tmp_path
    ):
        """prepare_r5_raw_rebuild cleans tolls.dat in primary R5 dir and derived caches in directory2."""
        from pilates.beam.launch_paths import prepare_r5_raw_rebuild
        from pilates.config.models import FullSkimsCreatorConfig

        region_dir = tmp_path / "beam" / "test_region"
        r5_primary = region_dir / "r5-simple"
        r5_secondary = region_dir / "r5-ferry"
        r5_primary.mkdir(parents=True)
        r5_secondary.mkdir(parents=True)
        (r5_primary / "osm.pbf").write_bytes(b"osm-primary")
        (r5_secondary / "osm.pbf").write_bytes(b"osm-secondary")

        for cache_file in ("network.dat", "osm.mapdb", "osm.mapdb.p", "tolls.dat"):
            (r5_primary / cache_file).write_bytes(b"derived")
            (r5_secondary / cache_file).write_bytes(b"derived2")

        conf_path = region_dir / "beam.conf"
        conf_path.write_text(
            "\n".join(
                [
                    "beam.routing.r5 {",
                    '  directory = ${beam.inputDirectory}"/r5-simple"',
                    '  directory2 = ""',
                    '  osmMapdbFile = ${beam.routing.r5.directory}"/osm.mapdb"',
                    "}",
                    'matsim.modules.network.inputNetworkFile = ${beam.routing.r5.directory}"/physsim-network.xml"',
                    'beam.physsim.inputNetworkFilePath = ${beam.routing.r5.directory}"/physsim-network.xml"',
                ]
            ),
            encoding="utf-8",
        )

        workspace = MagicMock()
        workspace.full_path = str(tmp_path)
        workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam")
        workspace.get_beam_output_dir.return_value = str(tmp_path / "beam_out")

        settings = SimpleNamespace(
            run=SimpleNamespace(region="test_region"),
            beam=SimpleNamespace(
                config="beam.conf",
                local_input_folder=str(tmp_path / "beam_immutable"),
                local_mutable_data_folder="beam",
                container_input_dir="/app/input",
                container_output_dir="/app/output",
                full_skim=FullSkimsCreatorConfig(directory2="r5-ferry"),
            ),
        )

        prepare_r5_raw_rebuild(
            settings=settings,
            workspace=workspace,
        )

        for cache_file in ("network.dat", "osm.mapdb", "osm.mapdb.p", "tolls.dat"):
            assert not (r5_primary / cache_file).exists()
            assert not (r5_secondary / cache_file).exists()
        assert (r5_primary / "osm.pbf").exists()
        assert (r5_secondary / "osm.pbf").exists()

    def test_skim_provenance_sidecar_and_inspection(self, tmp_path):
        """write_skim_provenance_sidecar and inspect_skim_artifact inspect OMX with and without root attrs."""
        import h5py
        import numpy as np
        from pilates.beam.skim_provenance import (
            SKIM_PROVENANCE_SCHEMA_VERSION,
            inspect_skim_artifact,
            read_skim_provenance_sidecar,
            write_skim_provenance_sidecar,
        )

        omx_path = tmp_path / "skims.omx"
        with h5py.File(omx_path, "w") as h5:
            h5.attrs["OMX_VERSION"] = np.bytes_("0.2")
            h5.attrs["SHAPE"] = np.array([3, 3], dtype=np.int32)
            h5.attrs["data_model_version"] = "beam_omx_v1"
            h5.attrs["skim_matrix_version"] = "asim_omx_v1"
            h5.attrs["beam_run_id"] = "run-123"
            h5.attrs["created_at"] = "2026-10-14T12:00:00Z"
            data_grp = h5.create_group("data")
            data_grp.create_dataset("SOV_TIME__AM", data=np.zeros((3, 3)))
            lookup_grp = h5.create_group("lookup")
            lookup_grp.create_dataset("zone_number", data=np.array([1, 2, 3]))

        sidecar_path, written_payload = write_skim_provenance_sidecar(
            omx_path,
            producer_step="beam_full_skim",
            run_id="run-123",
            scenario="test_scenario",
            region="test_region",
            year=2020,
            iteration=0,
        )
        assert sidecar_path.is_file()
        payload = read_skim_provenance_sidecar(sidecar_path)
        assert payload == written_payload
        assert payload["schema_version"] == SKIM_PROVENANCE_SCHEMA_VERSION
        assert payload["producer"]["step"] == "beam_full_skim"
        assert payload["identity"].startswith("sha256:file:")
        assert payload["skim_metadata"]["data_model_version"] == "beam_omx_v1"
        assert payload["skim_metadata"]["skim_matrix_version"] == "asim_omx_v1"
        assert payload["skim_metadata"]["beam_run_id"] == "run-123"
        assert payload["skim_metadata"]["matrix_count"] == 1
        assert payload["skim_metadata"]["zone_count"] == 3
        assert payload["skim_metadata"]["time_periods"] == ["AM"]

        # Legacy OMX without custom root provenance attributes still inspects cleanly
        legacy_omx = tmp_path / "legacy_skims.omx"
        with h5py.File(legacy_omx, "w") as h5:
            data_grp = h5.create_group("data")
            data_grp.create_dataset("DIST", data=np.ones((2, 2)))
        legacy_info = inspect_skim_artifact(legacy_omx)
        assert legacy_info["skim_format"] == "omx"
        assert legacy_info["has_embedded_provenance"] is False
        assert legacy_info["data_model_version"] is None
        assert legacy_info["matrix_count"] == 1
        assert legacy_info["zone_count"] == 2

    def test_select_latest_linkstats_maps_staged_warmstart_in_launch_config(
        self, tmp_path
    ):
        """_select_latest_linkstats_path maps unmounted workspace warmstart to staged .pilates/warmstarts inside launch_config."""
        from pilates.beam.runner import _select_latest_linkstats_path
        from pilates.generic.records import FileRecord, RecordStore

        workspace_warmstart = (
            tmp_path
            / "workspace"
            / "beam"
            / "input"
            / "sfbay"
            / "r5"
            / "init.linkstats.parquet"
        )
        workspace_warmstart.parent.mkdir(parents=True)
        workspace_warmstart.write_bytes(b"PAR1-fake-parquet")

        launch_root = (
            tmp_path
            / "workspace"
            / ".pilates-beam-launch-config"
            / "y2019"
            / "i0"
            / "sfbay"
        )
        staged_warmstart = (
            launch_root / ".pilates" / "warmstarts" / "init.linkstats.parquet"
        )
        staged_warmstart.parent.mkdir(parents=True)
        staged_warmstart.write_bytes(b"PAR1-fake-parquet")

        store = RecordStore(
            recordList=[
                FileRecord(
                    file_path=str(workspace_warmstart),
                    short_name="linkstats_warmstart",
                    description="Warmstart linkstats",
                )
            ]
        )
        selected = _select_latest_linkstats_path(
            store,
            str(launch_root),
            str(tmp_path / "workspace" / "beam" / "output"),
        )
        assert selected == "/app/input/.pilates/warmstarts/init.linkstats.parquet"

    def test_resolve_full_skim_input_path_does_not_double_region_for_launch_root(
        self, tmp_path
    ):
        """_resolve_full_skim_input_path does not double region when abs_beam_input already ends with region."""
        from pilates.beam.runner import _resolve_full_skim_input_path

        launch_root = (
            tmp_path / ".pilates-beam-launch-config" / "y2019" / "i0" / "sfbay"
        )
        launch_root.mkdir(parents=True)
        toll_file = launch_root / "toll-prices.csv"
        toll_file.write_text("linkId,toll\n", encoding="utf-8")

        container_path, host_path = _resolve_full_skim_input_path(
            "toll-prices.csv",
            abs_beam_input=str(launch_root),
            abs_beam_output=str(tmp_path / "output"),
            region="sfbay",
        )
        assert container_path == "/app/input/toll-prices.csv"
        assert host_path == str(toll_file)

    def test_standalone_full_skim_preprocess_outputs_resolves_warmstart_and_zones(
        self, tmp_path
    ):
        """_standalone_full_skim_preprocess_outputs resolves initial warmstart linkstats and prepares canonical zones."""
        from pilates.workflows.artifact_keys import LINKSTATS_WARMSTART
        from pilates.workflows.stages.supply_demand_beam import (
            _standalone_full_skim_preprocess_outputs,
        )

        beam_input_dir = tmp_path / "beam" / "input"
        warmstart_file = (
            beam_input_dir
            / "sfbay"
            / "r5"
            / "sfbay-cbg5500-weakConn-network"
            / "init.linkstats.parquet"
        )
        warmstart_file.parent.mkdir(parents=True)
        warmstart_file.write_bytes(b"PAR1-warmstart")

        workspace = MagicMock()
        workspace.get_beam_mutable_data_dir.return_value = str(beam_input_dir)
        scenario = SimpleNamespace(coupler={})
        settings = SimpleNamespace(
            run=SimpleNamespace(region="sfbay"),
            beam=SimpleNamespace(
                router_directory="r5/sfbay-cbg5500-weakConn-network",
                warmstart_linkstats_path="{router_directory}/init.linkstats.parquet",
            ),
            shared=SimpleNamespace(
                geography=SimpleNamespace(
                    zones=SimpleNamespace(
                        source_file="pilates/activitysim/data/sfbay/taz_sfbay.geojson"
                    )
                )
            ),
        )

        with patch(
            "pilates.beam.preprocessor._prepare_beam_zone_shapefile"
        ) as mock_prepare_zones:
            outputs = _standalone_full_skim_preprocess_outputs(
                scenario=scenario,
                workspace=workspace,
                settings=settings,
            )

        assert outputs.prepared_inputs[LINKSTATS_WARMSTART] == warmstart_file
        assert scenario.coupler[LINKSTATS_WARMSTART] == warmstart_file
        mock_prepare_zones.assert_called_once_with(workspace, settings)

    def test_standalone_full_skim_bypasses_beam_preprocess_even_when_population_in_coupler(
        self, tmp_path
    ):
        """In standalone mode, _standalone_full_skim_preprocess_outputs and _run_traffic_assignment_phase ignore population inputs in coupler."""
        from pilates.workflows.artifact_keys import (
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
            BEAM_PLANS_IN,
            LINKSTATS_WARMSTART,
        )
        from pilates.workflows.stages.supply_demand_beam import (
            TrafficAssignmentPhaseInputs,
            _run_traffic_assignment_phase,
            _standalone_full_skim_preprocess_outputs,
        )

        beam_input_dir = tmp_path / "beam" / "input"
        warmstart_file = beam_input_dir / "init.linkstats.parquet"
        warmstart_file.parent.mkdir(parents=True)
        warmstart_file.write_bytes(b"warmstart")

        plans_file = tmp_path / "plans.csv.gz"
        plans_file.write_bytes(b"plans")
        hh_file = tmp_path / "households.csv.gz"
        hh_file.write_bytes(b"hh")
        persons_file = tmp_path / "persons.csv.gz"
        persons_file.write_bytes(b"persons")

        workspace = MagicMock()
        workspace.get_beam_mutable_data_dir.return_value = str(beam_input_dir)
        coupler = {
            BEAM_PLANS_IN: str(plans_file),
            BEAM_HOUSEHOLDS_IN: str(hh_file),
            BEAM_PERSONS_IN: str(persons_file),
            LINKSTATS_WARMSTART: str(warmstart_file),
        }
        scenario = SimpleNamespace(coupler=coupler)
        settings = SimpleNamespace(
            run=SimpleNamespace(region="sfbay"),
            beam=SimpleNamespace(
                router_directory="r5",
                warmstart_linkstats_path="{router_directory}/init.linkstats.parquet",
                full_skim=SimpleNamespace(run_schedule="standalone"),
            ),
            shared=SimpleNamespace(
                geography=SimpleNamespace(
                    zones=SimpleNamespace(
                        source_file="pilates/activitysim/data/sfbay/taz_sfbay.geojson"
                    )
                )
            ),
        )

        # 1. Verify _standalone_full_skim_preprocess_outputs ignores population
        with patch("pilates.beam.preprocessor._prepare_beam_zone_shapefile"):
            outputs = _standalone_full_skim_preprocess_outputs(
                scenario=scenario,
                workspace=workspace,
                settings=settings,
            )

        assert LINKSTATS_WARMSTART in outputs.prepared_inputs
        assert outputs.prepared_inputs[LINKSTATS_WARMSTART] == warmstart_file
        assert BEAM_PLANS_IN not in outputs.prepared_inputs
        assert BEAM_HOUSEHOLDS_IN not in outputs.prepared_inputs
        assert BEAM_PERSONS_IN not in outputs.prepared_inputs

        # Standalone mode unconditionally bypasses beam_preprocess
        # 2. Verify _run_traffic_assignment_phase calls _standalone_full_skim_preprocess_outputs and never _run_beam_preprocess_step
        state = MagicMock()
        context = SimpleNamespace(
            settings=settings,
            state=state,
            workspace=workspace,
        )
        inputs = TrafficAssignmentPhaseInputs(year=2020, iteration=0)

        from contextlib import ExitStack

        with ExitStack() as stack:
            mock_beam_steps = stack.enter_context(
                patch("pilates.workflows.stages.supply_demand_beam._run_beam_steps")
            )
            mock_standalone_outputs = stack.enter_context(
                patch(
                    "pilates.workflows.stages.supply_demand_beam._standalone_full_skim_preprocess_outputs",
                    return_value=outputs,
                )
            )
            mock_launch_config = stack.enter_context(
                patch(
                    "pilates.workflows.stages.supply_demand_beam._compile_beam_launch_config"
                )
            )
            mock_full_skim_step = stack.enter_context(
                patch(
                    "pilates.workflows.stages.supply_demand_beam._run_beam_full_skim_step",
                    return_value=None,
                )
            )
            _run_traffic_assignment_phase(
                scenario=scenario,
                inputs=inputs,
                context=context,
            )

        mock_standalone_outputs.assert_called_once_with(
            scenario=scenario,
            workspace=workspace,
            settings=settings,
        )
        mock_beam_steps.assert_not_called()
        mock_launch_config.assert_called_once()
        mock_full_skim_step.assert_called_once()

    def test_sfbay_standalone_fullskim_hpc_config_loads_validly(self):
        """scenarios/sfbay/settings-sfbay-consist-fullskim-hpc-2019.yaml validates as a standalone activitySimOmx full-skim config."""
        from pilates.config.models import load_config

        cfg_path = (
            Path(__file__).resolve().parents[1]
            / "scenarios"
            / "sfbay"
            / "settings-sfbay-consist-fullskim-hpc-2019.yaml"
        )
        cfg = load_config(cfg_path)
        assert cfg.run.region == "sfbay"
        assert cfg.run.models.travel == "beam"
        assert cfg.run.models.land_use is None
        assert cfg.run.models.activity_demand is None
        assert cfg.run.models.vehicle_ownership is None
        assert cfg.beam is not None
        assert cfg.beam.config == "scenarios/sfbay-pilates-base-calibrated.conf"
        assert cfg.beam.full_skim is not None
        assert cfg.beam.full_skim.run_schedule == "standalone"
        assert cfg.beam.full_skim.skims_kind == "activitySimOmx"
        assert cfg.beam.full_skim.peak_hours == [5.0, 8.5, 12.0, 17.5, 22.1]
        assert cfg.beam.full_skim.modes_to_build == {
            "drive": True,
            "walk": True,
            "transit": True,
        }
        assert cfg.beam.full_skim.toll_file_path == "toll-prices.csv"

    def test_native_beam_full_skim_runs_linkstats_and_r5_preflight(self, tmp_path):
        """_native_beam_full_skim runs linkstats admission and R5 execution reference preflight before launching BeamFullSkimRunner."""
        from pilates.beam.launch_config import BeamLaunchConfig
        from pilates.workflows.steps.beam import _native_beam_full_skim

        launch_root = tmp_path / "launch"
        launch_root.mkdir()
        primary_conf = launch_root / "sfbay.conf"
        primary_conf.write_text("# beam conf\n", encoding="utf-8")
        launch_config = BeamLaunchConfig(
            root=launch_root,
            primary_config=primary_conf,
        )

        workspace = MagicMock()
        workspace.get_beam_mutable_data_dir.return_value = str(
            tmp_path / "beam" / "input"
        )
        state = SimpleNamespace(
            run_info_path=str(tmp_path / "run_info.json"),
            year=2019,
            iteration=0,
        )
        settings = SimpleNamespace()
        consist_ctx = SimpleNamespace(
            run_metadata={"admission_reports": {"existing": True}}
        )
        fake_ref = SimpleNamespace(staged_path=tmp_path / "init.linkstats.parquet")

        from contextlib import ExitStack

        fn = getattr(_native_beam_full_skim, "__wrapped__", _native_beam_full_skim)
        with ExitStack() as stack:
            mock_validate_linkstats = stack.enter_context(
                patch(
                    "pilates.workflows.steps.beam.validate_staged_linkstats_reference",
                    return_value=fake_ref,
                )
            )
            stack.enter_context(
                patch(
                    "pilates.workflows.steps.beam.has_staged_linkstats_admission_policy",
                    return_value=True,
                )
            )
            mock_preflight_admission = stack.enter_context(
                patch(
                    "pilates.workflows.steps.beam.preflight_staged_linkstats_admission"
                )
            )
            mock_validate_r5 = stack.enter_context(
                patch("pilates.workflows.steps.beam.validate_r5_execution_reference")
            )
            mock_runner_cls = stack.enter_context(
                patch("pilates.beam.runner.BeamFullSkimRunner")
            )
            stack.enter_context(
                patch("pilates.workflows.steps.beam._validate_native_outputs")
            )
            mock_runner_cls.return_value.run.return_value = SimpleNamespace(
                full_skims=None,
                final_skims_omx=None,
            )
            fn(
                settings=settings,
                state=state,
                workspace=workspace,
                beam_launch_config=launch_config,
                _consist_ctx=consist_ctx,
            )

        mock_validate_linkstats.assert_called_once_with(
            settings=settings,
            workspace=workspace,
            run_context=consist_ctx,
            config_root=launch_root,
        )
        mock_preflight_admission.assert_called_once()
        assert mock_preflight_admission.call_args.kwargs["launch_reference"] is fake_ref
        mock_validate_r5.assert_called_once_with(
            settings=settings,
            workspace=workspace,
            run_context=consist_ctx,
            config_root=launch_root,
        )
        mock_runner_cls.return_value.run.assert_called_once()
