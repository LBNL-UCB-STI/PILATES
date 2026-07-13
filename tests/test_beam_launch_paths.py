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


def test_resolve_r5_network_reference_selects_lexically_first_osm_and_gtfs(
    tmp_path: Path,
) -> None:
    from pilates.beam.launch_paths import resolve_r5_network_reference

    beam_input_dir = tmp_path / "beam" / "input"
    region_dir = beam_input_dir / "seattle"
    r5_dir = region_dir / "r5" / "network"
    r5_dir.mkdir(parents=True)
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
    assert reference.ignored_osm_paths == (r5_dir / "z.osm.pbf",)


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


def test_validate_r5_execution_reference_matches_consist_member(tmp_path: Path) -> None:
    from pilates.beam.launch_paths import validate_r5_execution_reference

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
