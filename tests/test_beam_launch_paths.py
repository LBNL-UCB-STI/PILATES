from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest


class _Workspace:
    def __init__(self, beam_input_dir: Path) -> None:
        self._beam_input_dir = beam_input_dir
        self.full_path = str(beam_input_dir.parent.parent)

    def get_beam_mutable_data_dir(self) -> str:
        return str(self._beam_input_dir)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        beam=SimpleNamespace(
            config="beam.conf",
            local_mutable_data_folder="beam/input",
        ),
    )


def _write_config(
    config_path: Path,
    *,
    directory: str,
    physsim_directory: str | None = None,
) -> None:
    physsim_directory = physsim_directory or directory
    config_path.write_text(
        "\n".join(
            [
                "beam.routing.r5 {",
                f"  directory = {directory}",
                '  directory2 = ""',
                '  osmMapdbFile = ${beam.routing.r5.directory}"/osm.mapdb"',
                "}",
                (
                    "matsim.modules.network.inputNetworkFile = "
                    f'{physsim_directory}"/physsim-network.xml"'
                ),
                (
                    "beam.physsim.inputNetworkFilePath = "
                    f'{physsim_directory}"/physsim-network.xml"'
                ),
            ]
        ),
        encoding="utf-8",
    )


def _write_gtfs_zip(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("stop_times.txt", "trip_id,arrival_time\n")


def test_resolve_r5_network_reference_prefers_first_pbf_and_collects_gtfs(
    tmp_path: Path,
) -> None:
    from pilates.beam.launch_paths import resolve_r5_network_reference

    beam_input_dir = tmp_path / "beam" / "input"
    region_dir = beam_input_dir / "seattle"
    r5_dir = region_dir / "r5" / "network"
    r5_dir.mkdir(parents=True)
    vex = r5_dir / "00.vex"
    vex.write_bytes(b"vex")
    (r5_dir / "z.osm.pbf").write_bytes(b"z")
    selected_osm = r5_dir / "a.osm.pbf"
    selected_osm.write_bytes(b"a")
    _write_gtfs_zip(r5_dir / "transit.zip")
    (r5_dir / "not-a-feed.zip").write_bytes(b"not a zip")
    _write_config(
        region_dir / "beam.conf",
        directory='${beam.inputDirectory}"/r5/network"',
        physsim_directory='${beam.inputDirectory}"/other-network"',
    )

    reference = resolve_r5_network_reference(
        settings=_settings(),
        workspace=_Workspace(beam_input_dir),
    )

    assert reference.network_directory.config_key == "beam.routing.r5.directory"
    assert reference.selected_osm_path == selected_osm
    assert reference.selected_osm_container_path == (
        "/app/input/seattle/r5/network/a.osm.pbf"
    )
    assert reference.gtfs_paths == (r5_dir / "transit.zip",)
    assert reference.ignored_osm_paths == (vex, r5_dir / "z.osm.pbf")


def test_r5_vex_only_is_allowed_for_beam_and_rejected_before_presim(
    tmp_path: Path,
    monkeypatch,
):
    from pilates.beam.launch_paths import (
        BeamLaunchPathError,
        resolve_r5_network_reference,
    )
    from pilates.workflows.steps.impacts import impacts_presim

    settings = _settings()
    workspace = _Workspace(tmp_path / "beam/input")
    region_dir = Path(workspace.get_beam_mutable_data_dir()) / "seattle"
    r5_dir = region_dir / "r5"
    r5_dir.mkdir(parents=True)
    vex = r5_dir / "only.vex"
    vex.touch()
    _write_config(region_dir / "beam.conf", directory='${beam.inputDirectory}"/r5"')

    reference = resolve_r5_network_reference(settings=settings, workspace=workspace)
    assert reference.selected_osm_path == vex

    settings.run.models = SimpleNamespace(impacts="impacts")
    monkeypatch.setattr(
        "pilates.beam.preprocessor.ensure_beam_mutable_workspace", lambda *_: None
    )
    with pytest.raises(BeamLaunchPathError, match=r"IMPACTS requires an OSM \.pbf"):
        impacts_presim.resolve_inputs(settings=settings, workspace=workspace)


def test_prepare_r5_omits_vex_from_consist_view_and_compiled_launch(tmp_path, caplog):
    from pilates.beam.launch_paths import (
        prepare_r5_canonicalization_directory,
        prepare_r5_raw_rebuild,
    )

    source = tmp_path / "source"
    source.mkdir()
    (source / "00.vex").write_bytes(b"vex")
    pbf = source / "road.osm.pbf"
    pbf.write_bytes(b"pbf")
    feed = source / "transit.zip"
    _write_gtfs_zip(feed)
    view = prepare_r5_canonicalization_directory(source, tmp_path / "view")
    assert {path.resolve() for path in view.iterdir()} == {pbf, feed}

    launch = tmp_path / "compiled/seattle"
    r5 = launch / "r5"
    r5.mkdir(parents=True)
    for path in source.iterdir():
        (r5 / path.name).hardlink_to(path)
    _write_config(launch / "beam.conf", directory='${beam.inputDirectory}"/r5"')
    reference = prepare_r5_raw_rebuild(
        settings=_settings(),
        workspace=_Workspace(tmp_path / "beam/input"),
        config_root=launch,
    )

    assert reference.selected_osm_path == r5 / pbf.name
    assert reference.gtfs_paths == (r5 / feed.name,)
    assert not (r5 / "00.vex").exists()
    assert (source / "00.vex").read_bytes() == b"vex"
    assert "R5 found VEX 00.vex and will not load it" in caplog.text


def test_resolve_r5_network_reference_rejects_an_external_directory(
    tmp_path: Path,
) -> None:
    from pilates.beam.launch_paths import (
        BeamLaunchPathError,
        resolve_r5_network_reference,
    )

    beam_input_dir = tmp_path / "beam" / "input"
    region_dir = beam_input_dir / "seattle"
    region_dir.mkdir(parents=True)
    external_r5_dir = tmp_path / "external-r5"
    external_r5_dir.mkdir()
    (external_r5_dir / "network.osm.pbf").write_bytes(b"osm")
    _write_config(
        region_dir / "beam.conf",
        directory=f'"{external_r5_dir}"',
    )

    with pytest.raises(
        BeamLaunchPathError, match="outside the mutable BEAM region input tree"
    ):
        resolve_r5_network_reference(
            settings=_settings(),
            workspace=_Workspace(beam_input_dir),
        )


def test_prepare_r5_raw_rebuild_removes_only_derived_caches(tmp_path: Path) -> None:
    from pilates.beam.config_hocon import (
        beam_config_env_overrides,
        resolve_beam_config_value,
    )
    from pilates.beam.launch_paths import prepare_r5_raw_rebuild

    beam_input_dir = tmp_path / "beam" / "input"
    region_dir = beam_input_dir / "seattle"
    r5_dir = region_dir / "r5" / "network"
    r5_dir.mkdir(parents=True)
    selected_osm = r5_dir / "network.osm.pbf"
    selected_osm.write_bytes(b"osm source")
    for name in ("network.dat", "osm.mapdb", "osm.mapdb.p", "physsim-network.xml"):
        (r5_dir / name).write_bytes(b"derived cache")
    _write_config(
        region_dir / "beam.conf",
        directory='${beam.inputDirectory}"/r5/network"',
        physsim_directory='${beam.inputDirectory}"/other-network"',
    )

    reference = prepare_r5_raw_rebuild(
        settings=_settings(),
        workspace=_Workspace(beam_input_dir),
    )

    assert reference.selected_osm_path == selected_osm
    assert selected_osm.exists()
    assert not (r5_dir / "network.dat").exists()
    assert not (r5_dir / "osm.mapdb").exists()
    assert not (r5_dir / "osm.mapdb.p").exists()
    assert not (r5_dir / "physsim-network.xml").exists()
    config_path = region_dir / "beam.conf"
    expected_physsim_path = str(r5_dir / "physsim-network.xml")
    env_overrides = beam_config_env_overrides(
        _settings(), workspace=_Workspace(beam_input_dir)
    )
    assert (
        resolve_beam_config_value(
            config_path,
            key="matsim.modules.network.inputNetworkFile",
            env_overrides=env_overrides,
        )
        == expected_physsim_path
    )
    assert (
        resolve_beam_config_value(
            config_path,
            key="beam.physsim.inputNetworkFilePath",
            env_overrides=env_overrides,
        )
        == expected_physsim_path
    )


def test_prepare_r5_raw_rebuild_refuses_an_active_network_lock(tmp_path: Path) -> None:
    from pilates.beam.launch_paths import BeamLaunchPathError, prepare_r5_raw_rebuild

    beam_input_dir = tmp_path / "beam" / "input"
    region_dir = beam_input_dir / "seattle"
    r5_dir = region_dir / "r5" / "network"
    r5_dir.mkdir(parents=True)
    (r5_dir / "network.osm.pbf").write_bytes(b"osm source")
    (r5_dir / "network.dat.lock").write_text("busy", encoding="utf-8")
    _write_config(
        region_dir / "beam.conf",
        directory='${beam.inputDirectory}"/r5/network"',
    )

    with pytest.raises(BeamLaunchPathError, match="active network.dat.lock"):
        prepare_r5_raw_rebuild(
            settings=_settings(),
            workspace=_Workspace(beam_input_dir),
        )


def test_configure_staged_linkstats_reference_rewrites_final_hocon(
    tmp_path: Path,
) -> None:
    from pilates.beam.config_hocon import (
        beam_config_env_overrides,
        resolve_beam_config_value,
    )
    from pilates.beam.launch_paths import configure_staged_linkstats_reference

    beam_input_dir = tmp_path / "beam" / "input"
    region_dir = beam_input_dir / "seattle"
    r5_dir = region_dir / "r5" / "network"
    r5_dir.mkdir(parents=True)
    (r5_dir / "network.osm.pbf").write_bytes(b"osm source")
    staged_linkstats = region_dir / "_pilates" / "linkstats" / "warmstart.csv.gz"
    staged_linkstats.parent.mkdir(parents=True)
    staged_linkstats.write_bytes(b"linkstats")
    _write_config(
        region_dir / "beam.conf",
        directory='${beam.inputDirectory}"/r5/network"',
    )

    reference = configure_staged_linkstats_reference(
        settings=_settings(),
        workspace=_Workspace(beam_input_dir),
        staged_path=staged_linkstats,
    )

    assert reference.execution_path == staged_linkstats
    assert (
        reference.container_path
        == "/app/input/seattle/_pilates/linkstats/warmstart.csv.gz"
    )
    config_path = region_dir / "beam.conf"
    assert '${beam.inputDirectory}"/_pilates/linkstats/warmstart.csv.gz"' in (
        config_path.read_text(encoding="utf-8")
    )
    assert resolve_beam_config_value(
        config_path,
        key="beam.warmStart.initialLinkstatsFilePath",
        env_overrides=beam_config_env_overrides(
            _settings(), workspace=_Workspace(beam_input_dir)
        ),
    ) == str(staged_linkstats)


def test_validate_staged_linkstats_reference_matches_canonicalization(
    tmp_path: Path,
) -> None:
    from pilates.beam.launch_paths import validate_staged_linkstats_reference

    beam_input_dir = tmp_path / "beam" / "input"
    region_dir = beam_input_dir / "seattle"
    r5_dir = region_dir / "r5" / "network"
    r5_dir.mkdir(parents=True)
    (r5_dir / "network.osm.pbf").write_bytes(b"osm source")
    staged_linkstats = region_dir / "_pilates" / "linkstats" / "warmstart.csv.gz"
    staged_linkstats.parent.mkdir(parents=True)
    staged_linkstats.write_bytes(b"linkstats")
    _write_config(
        region_dir / "beam.conf",
        directory='${beam.inputDirectory}"/r5/network"',
    )
    from pilates.beam.launch_paths import configure_staged_linkstats_reference

    configure_staged_linkstats_reference(
        settings=_settings(),
        workspace=_Workspace(beam_input_dir),
        staged_path=staged_linkstats,
    )
    snapshot_reference = SimpleNamespace(
        reference=SimpleNamespace(
            config_key="beam.warmStart.initialLinkstatsFilePath",
            delegated_artifact_keys=("linkstats_warmstart",),
        ),
        resolved_path=staged_linkstats.resolve(),
    )
    context = SimpleNamespace(
        canonicalization=SimpleNamespace(references=(snapshot_reference,))
    )

    reference = validate_staged_linkstats_reference(
        settings=_settings(),
        workspace=_Workspace(beam_input_dir),
        run_context=context,
    )

    assert reference is not None
    assert reference.execution_path == staged_linkstats


def test_validate_r5_execution_reference_matches_consist_member(tmp_path: Path) -> None:
    from pilates.beam.launch_paths import (
        BeamLaunchPathError,
        validate_r5_execution_reference,
    )

    beam_input_dir = tmp_path / "beam" / "input"
    region_dir = beam_input_dir / "seattle"
    r5_dir = region_dir / "r5" / "network"
    r5_dir.mkdir(parents=True)
    selected_osm = r5_dir / "network.osm.pbf"
    selected_osm.write_bytes(b"osm")
    _write_config(
        region_dir / "beam.conf", directory='${beam.inputDirectory}"/r5/network"'
    )
    artifact_key = "config:seattle/r5/network/network.osm.pbf"
    member = SimpleNamespace(
        role="r5_osm_source",
        resolved_path=selected_osm.resolve(),
        artifact_key=artifact_key,
    )
    reference = SimpleNamespace(
        reference=SimpleNamespace(config_key="beam.routing.r5.directory"),
        artifact_keys=(artifact_key,),
        artifact_members=(member,),
    )
    context = SimpleNamespace(canonicalization=SimpleNamespace(references=(reference,)))

    execution_reference = validate_r5_execution_reference(
        settings=_settings(), workspace=_Workspace(beam_input_dir), run_context=context
    )

    assert execution_reference.selected_osm_path == selected_osm

    (r5_dir / "00.vex").touch()
    with pytest.raises(BeamLaunchPathError, match="VEX that sorts before"):
        validate_r5_execution_reference(
            settings=_settings(),
            workspace=_Workspace(beam_input_dir),
            run_context=context,
        )


@pytest.mark.parametrize(
    "variant",
    [
        "iter0_base_renamed_to_warmstart",
        "iter1_warmstart_differs_from_base",
        "no_warmstart_artifact_uses_base_copy",
        "tampered_staged_warmstart_fails_closed",
    ],
)
def test_linkstats_and_r5_validation_across_compiled_launch_variants(
    tmp_path: Path,
    variant: str,
) -> None:
    """Real BeamConfigAdapter canonicalization + build_beam_launch_config preflight."""
    from consist.core.identity import IdentityManager
    from consist.integrations.beam import BeamConfigAdapter, BeamReferencePolicy
    from pilates.beam.launch_config import build_beam_launch_config
    from pilates.beam.launch_paths import (
        BeamLaunchPathError,
        beam_config_env_overrides,
        validate_r5_execution_reference,
        validate_staged_linkstats_reference,
    )
    from pilates.config.models import BeamArtifactFormatsConfig
    from pilates.workflows.artifact_keys import LINKSTATS_WARMSTART

    beam_input_dir = tmp_path / "beam" / "input"
    region_dir = beam_input_dir / "seattle"
    r5_dir = region_dir / "r5" / "network"
    r5_dir.mkdir(parents=True)
    osm_path = r5_dir / "network.osm.pbf"
    osm_path.write_bytes(b"osm-source-bytes")
    base_linkstats = r5_dir / "init.linkstats.parquet"
    base_linkstats.write_bytes(b"base-init-linkstats")
    (region_dir / "urbansim").mkdir(parents=True)
    vehicles_path = region_dir / "urbansim" / "vehicles.csv"
    vehicles_path.write_text("vehicleId,vehicleTypeId\n", encoding="utf-8")

    primary_conf = region_dir / "beam.conf"
    primary_conf.write_text(
        "\n".join(
            [
                'beam.inputDirectory = "."',
                'beam.outputs.baseOutputDirectory = "output"',
                "beam.replanning.maxAgentPlanMemorySize = 4",
                "beam.exchange.scenario {",
                '  folder = ${beam.inputDirectory}"/urbansim"',
                "  convertWgs2Utm = true",
                "}",
                'beam.agentsim.agents.vehicles.vehiclesFilePath = ${beam.inputDirectory}"/urbansim/vehicles.csv"',
                'beam.warmStart.initialLinkstatsFilePath = ${beam.inputDirectory}"/r5/network/init.linkstats.parquet"',
                "beam.routing.r5 {",
                '  directory = ${beam.inputDirectory}"/r5/network"',
                '  directory2 = ""',
                '  osmMapdbFile = ${beam.routing.r5.directory}"/osm.mapdb"',
                "}",
                'beam.physsim.inputNetworkFilePath = ${beam.routing.r5.directory}"/physsim.xml"',
                'matsim.modules.network.inputNetworkFile = ""',
                'beam.router.skim.activity-sim-skimmer.fileBaseName = "skims"',
                "beam.exchange.output.activity_sim_skims_format = parquet",
                "beam.exchange.output.exchange_format = parquet",
                "beam.exchange.output.events_format = parquet",
                "beam.exchange.output.linkstats_format = parquet",
                "beam.exchange.output.full_skim_timeout_hours = 72",
                "",
            ]
        ),
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="seattle", output_directory=str(tmp_path / "archive")
        ),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="urbansim",
            local_mutable_data_folder="beam/input",
            discard_plans_every_year=True,
            sample=1.0,
            replanning_portion=0.0,
            memory="4g",
            max_plans_memory=4,
            router_directory=None,
            admission=None,
            artifact_formats=BeamArtifactFormatsConfig(),
            activitysim_skims_file_base_name="skimsActivitySimOD",
        ),
        activitysim=None,
        shared=SimpleNamespace(geography=SimpleNamespace(zones=None)),
    )
    workspace = _Workspace(beam_input_dir)
    identity = IdentityManager()

    adapter = BeamConfigAdapter(
        root_dirs=[region_dir],
        primary_config=primary_conf,
        env_overrides=beam_config_env_overrides(settings, config_root=region_dir),
        path_aliases={"beam_region_input": region_dir, "beam_input": beam_input_dir},
        reference_policies={
            "beam.routing.r5.directory": BeamReferencePolicy(
                identity_policy="delegated_to_artifacts",
                role="beam_r5_raw_network_directory",
                required=True,
                delegated_artifact_keys=("r5_network",),
            ),
            "beam.warmStart.initialLinkstatsFilePath": BeamReferencePolicy(
                identity_policy="delegated_to_artifacts",
                role="beam_linkstats_warmstart",
                required=False,
                delegated_artifact_keys=(LINKSTATS_WARMSTART,),
            ),
            "beam.exchange.scenario.folder": BeamReferencePolicy(
                identity_policy="delegated_to_artifacts",
                role="beam_population_input_root",
                required=False,
                delegated_artifact_keys=("plans_beam_in",),
            ),
            "beam.agentsim.agents.vehicles.vehiclesFilePath": BeamReferencePolicy(
                identity_policy="delegated_to_artifacts",
                role="beam_vehicle_input",
                required=False,
                delegated_artifact_keys=("vehicles_beam_in",),
            ),
            "beam.physsim.inputNetworkFilePath": BeamReferencePolicy(
                identity_policy="output_or_runtime_ignored",
                required=False,
            ),
            "beam.routing.r5.osmMapdbFile": BeamReferencePolicy(
                identity_policy="output_or_runtime_ignored",
                required=False,
            ),
            "beam.router.skim.activity-sim-skimmer.fileBaseName": BeamReferencePolicy(
                identity_policy="output_or_runtime_ignored",
                required=False,
            ),
        },
    )
    discovered = adapter.discover(
        [region_dir],
        identity=identity,
        strict=True,
    )
    canonical = adapter.canonicalize(
        discovered,
        run=None,
        plan_only=True,
        strict=True,
    )
    run_context = SimpleNamespace(canonicalization=canonical.canonicalization)

    prepared_inputs: dict[str, Path] = {}
    expected_warmstart_path: Path | None = None
    if variant == "iter0_base_renamed_to_warmstart":
        staged = (
            beam_input_dir
            / ".pilates-consist-outputs"
            / "linkstats_warmstart.linkstats.parquet"
        )
        staged.parent.mkdir(parents=True)
        staged.write_bytes(b"base-init-linkstats")
        prepared_inputs[LINKSTATS_WARMSTART] = staged
        expected_warmstart_path = staged
    elif variant in (
        "iter1_warmstart_differs_from_base",
        "tampered_staged_warmstart_fails_closed",
    ):
        iter0_out = tmp_path / "iter0_linkstats.parquet"
        iter0_out.write_bytes(b"iter0-warmstart-linkstats")
        prepared_inputs[LINKSTATS_WARMSTART] = iter0_out
        expected_warmstart_path = iter0_out

    launch_config = build_beam_launch_config(
        settings=settings,
        source_root=region_dir,
        output_dir=tmp_path / ".pilates-beam-launch-config" / "y2019" / "i0",
        identity=identity,
        prepared_inputs=prepared_inputs,
    )

    if variant == "tampered_staged_warmstart_fails_closed":
        staged_target = validate_staged_linkstats_reference(
            settings=settings,
            workspace=workspace,
            run_context=run_context,
            config_root=launch_config.root,
            expected_warmstart_path=expected_warmstart_path,
        )
        assert staged_target is not None
        staged_target.physical_target_path.write_bytes(b"tampered-bytes")
        with pytest.raises(
            BeamLaunchPathError,
            match="Consist canonicalized linkstats differs from the staged file BEAM will read",
        ):
            validate_staged_linkstats_reference(
                settings=settings,
                workspace=workspace,
                run_context=run_context,
                config_root=launch_config.root,
                expected_warmstart_path=expected_warmstart_path,
            )
        return

    linkstats_ref = validate_staged_linkstats_reference(
        settings=settings,
        workspace=workspace,
        run_context=run_context,
        config_root=launch_config.root,
        expected_warmstart_path=expected_warmstart_path,
    )
    r5_ref = validate_r5_execution_reference(
        settings=settings,
        workspace=workspace,
        run_context=run_context,
        config_root=launch_config.root,
    )
    assert linkstats_ref is not None
    assert (
        r5_ref.selected_osm_path.resolve()
        == (launch_config.root / "r5" / "network" / "network.osm.pbf").resolve()
    )
