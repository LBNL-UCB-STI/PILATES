from __future__ import annotations

import gzip
import logging
from pathlib import Path
from types import SimpleNamespace

from consist import Tracker
from pilates.config.models import BeamArtifactFormatsConfig


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="scenario",
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
        shared=SimpleNamespace(geography=SimpleNamespace(zones=None)),
    )


def _write_base_config(root: Path) -> Path:
    root.mkdir(parents=True)
    primary = root / "beam.conf"
    primary.write_text(
        "\n".join(
            (
                "beam {",
                "  inputDirectory = ${?inputDirectory}",
                '  outputs.baseOutputDirectory = "/app/output"',
                "  replanning.maxAgentPlanMemorySize = 4",
                '  warmStart.initialLinkstatsFilePath = ""',
                '  agentsim.taz.filePath = ""',
                '  agentsim.taz.tazIdFieldName = "taz"',
                '  exchange.scenario.folder = ${beam.inputDirectory}"/scenario"',
                '  exchange.scenario.fileFormat = "csv"',
                '  router.skim.activity-sim-skimmer.fileBaseName = "activitySimODSkims"',
                '  router.skim.activity-sim-skimmer.fileOutputFormat = "omx"',
                '  outputs.events.fileOutputFormats = "csv.gz"',
                '  physsim.linkStatsOutputFileType = "csv.gz"',
                '  routing.r5.directory = ${beam.inputDirectory}"/r5/network"',
                '  physsim.inputNetworkFilePath = ""',
                "}",
                'matsim.modules.network.inputNetworkFile = ""',
            )
        ),
        encoding="utf-8",
    )
    (root / "r5" / "network").mkdir(parents=True)
    return primary


def _write_shared_config_tree(root: Path) -> Path:
    """Create a region config that includes its staged sibling ``common`` tree."""

    root.mkdir(parents=True)
    primary = root / "beam.conf"
    primary.write_text(
        "\n".join(
            (
                'include "../common/matsim.conf"',
                "beam {",
                "  inputDirectory = ${?inputDirectory}",
                "  outputs.baseOutputDirectory = ${?BEAM_OUTPUT}",
                "  replanning.maxAgentPlanMemorySize = 4",
                '  warmStart.initialLinkstatsFilePath = ""',
                '  agentsim.taz.filePath = ""',
                '  agentsim.taz.tazIdFieldName = "taz"',
                '  exchange.scenario.folder = ${beam.inputDirectory}"/scenario"',
                '  exchange.scenario.fileFormat = "csv"',
                '  router.skim.activity-sim-skimmer.fileBaseName = "activitySimODSkims"',
                '  router.skim.activity-sim-skimmer.fileOutputFormat = "omx"',
                '  outputs.events.fileOutputFormats = "csv.gz"',
                '  physsim.linkStatsOutputFileType = "csv.gz"',
                '  routing.r5.directory = ${beam.inputDirectory}"/r5/network"',
                '  physsim.inputNetworkFilePath = ""',
                "}",
            )
        ),
        encoding="utf-8",
    )
    common = root.parent / "common"
    common.mkdir()
    (common / "matsim.conf").write_text(
        'matsim.modules.network.inputNetworkFile = ""\n', encoding="utf-8"
    )
    (root / "r5" / "network").mkdir(parents=True)
    return primary


def test_materialize_beam_launch_config_creates_a_fresh_config_tree(
    tmp_path: Path,
) -> None:
    from pilates.beam.config_hocon import (
        beam_config_env_overrides,
        resolve_beam_config_value,
    )
    from pilates.beam.launch_config import (
        BeamLaunchConfigOverrides,
        materialize_beam_launch_config,
    )

    settings = _settings()
    source_root = tmp_path / "beam" / "input" / "seattle"
    source_primary = _write_base_config(source_root)
    source_bytes = source_primary.read_bytes()
    warmstart = source_root / "_pilates" / "linkstats" / "warmstart.csv.gz"
    warmstart.parent.mkdir(parents=True)
    warmstart.write_bytes(b"warmstart")
    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )

    launch_config = materialize_beam_launch_config(
        settings=settings,
        source_root=source_root,
        output_dir=tmp_path / "launch-config",
        identity=tracker.identity,
        overrides=BeamLaunchConfigOverrides(
            values={
                "beam.replanning.maxAgentPlanMemorySize": 0,
                "beam.warmStart.initialLinkstatsFilePath": str(
                    tmp_path
                    / "launch-config"
                    / "seattle"
                    / "_pilates"
                    / "linkstats"
                    / "warmstart.csv.gz"
                ),
                "matsim.modules.network.inputNetworkFile": str(
                    tmp_path
                    / "launch-config"
                    / "seattle"
                    / "r5"
                    / "network"
                    / "physsim-network.xml"
                ),
                "beam.physsim.inputNetworkFilePath": str(
                    tmp_path
                    / "launch-config"
                    / "seattle"
                    / "r5"
                    / "network"
                    / "physsim-network.xml"
                ),
            }
        ),
    )

    assert source_primary.read_bytes() == source_bytes
    assert launch_config.root == tmp_path / "launch-config" / "seattle"
    assert launch_config.primary_config == launch_config.root / "beam.conf"
    assert launch_config.primary_config.is_file()
    env = beam_config_env_overrides(settings, config_root=launch_config.root)
    assert (
        resolve_beam_config_value(
            launch_config.primary_config,
            key="beam.replanning.maxAgentPlanMemorySize",
            env_overrides=env,
        )
        == 0
    )
    assert resolve_beam_config_value(
        launch_config.primary_config,
        key="beam.warmStart.initialLinkstatsFilePath",
        env_overrides=env,
    ) == str(launch_config.root / "_pilates" / "linkstats" / "warmstart.csv.gz")


def test_build_beam_launch_config_rebinds_preprocess_outputs_without_mutating_source(
    tmp_path: Path,
) -> None:
    from pilates.beam.config_hocon import (
        beam_config_env_overrides,
        resolve_beam_config_value,
    )
    from pilates.beam.launch_config import build_beam_launch_config

    settings = _settings()
    source_root = tmp_path / "beam" / "input" / "seattle"
    source_primary = _write_base_config(source_root)
    source_bytes = source_primary.read_bytes()
    prepared_root = tmp_path / "preprocess-cache"
    prepared: dict[str, Path] = {}
    for key in ("plans_beam_in", "households_beam_in", "persons_beam_in"):
        path = prepared_root / f"{key}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key, encoding="utf-8")
        prepared[key] = path
    # A prior native run before the output-layout migration may have written
    # Parquet bytes under the historical ``.csv.gz`` destination. The launch
    # compiler must recover the reader-compatible filename from the bytes.
    warmstart = prepared_root / "warmstart.csv.gz"
    warmstart.write_bytes(b"PAR1simulated-parquet-linkstats")
    prepared["linkstats_warmstart"] = warmstart
    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )

    launch_config = build_beam_launch_config(
        settings=settings,
        source_root=source_root,
        output_dir=tmp_path / "launch-config",
        identity=tracker.identity,
        prepared_inputs=prepared,
    )

    assert source_primary.read_bytes() == source_bytes
    assert (
        launch_config.root / "scenario" / "plans.csv"
    ).read_text() == "plans_beam_in"
    assert (
        launch_config.root / "scenario" / "households.csv"
    ).read_text() == "households_beam_in"
    assert (
        launch_config.root / "scenario" / "persons.csv"
    ).read_text() == "persons_beam_in"
    env = beam_config_env_overrides(settings, config_root=launch_config.root)
    assert (
        resolve_beam_config_value(
            launch_config.primary_config,
            key="beam.replanning.maxAgentPlanMemorySize",
            env_overrides=env,
        )
        == 0
    )
    assert resolve_beam_config_value(
        launch_config.primary_config,
        key="beam.warmStart.initialLinkstatsFilePath",
        env_overrides=env,
    ) == str(launch_config.root / ".pilates" / "warmstarts" / "warmstart.parquet")
    assert (
        launch_config.root / ".pilates" / "warmstarts" / "warmstart.parquet"
    ).read_bytes() == warmstart.read_bytes()


def test_build_beam_launch_config_applies_default_artifact_format_policy(
    tmp_path: Path, caplog
) -> None:
    from pilates.beam.config_hocon import resolve_beam_config_value
    from pilates.beam.launch_config import build_beam_launch_config

    settings = _settings()
    source_root = tmp_path / "beam" / "input" / "seattle"
    _write_base_config(source_root)
    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )

    with caplog.at_level(logging.WARNING, logger="pilates.beam.launch_config"):
        launch_config = build_beam_launch_config(
            settings=settings,
            source_root=source_root,
            output_dir=tmp_path / "launch-config",
            identity=tracker.identity,
            prepared_inputs={},
        )

    expected = {
        "beam.router.skim.activity-sim-skimmer.fileBaseName": "skimsActivitySimOD",
        "beam.router.skim.activity-sim-skimmer.fileOutputFormat": "zarr",
        "beam.exchange.scenario.fileFormat": "parquet",
        "beam.outputs.events.fileOutputFormats": "parquet",
        "beam.physsim.linkStatsOutputFileType": "parquet",
    }
    actual = {
        key: resolve_beam_config_value(
            launch_config.primary_config,
            key=key,
            env_overrides={},
        )
        for key in expected
    }

    assert actual == expected
    assert (
        "PILATES default BEAM artifact-format policy overrides source config"
        in caplog.text
    )


def test_build_beam_launch_config_keeps_gzip_warmstart_with_parquet_output_policy(
    tmp_path: Path,
) -> None:
    from pilates.beam.config_hocon import resolve_beam_config_value
    from pilates.beam.launch_config import build_beam_launch_config

    settings = _settings()
    source_root = tmp_path / "beam" / "input" / "seattle"
    _write_base_config(source_root)
    warmstart = tmp_path / "preprocess-cache" / "linkstats.csv.gz"
    warmstart.parent.mkdir(parents=True)
    warmstart.write_bytes(gzip.compress(b"legacy linkstats"))
    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )

    launch_config = build_beam_launch_config(
        settings=settings,
        source_root=source_root,
        output_dir=tmp_path / "launch-config",
        identity=tracker.identity,
        prepared_inputs={"linkstats_warmstart": warmstart},
    )

    assert (
        resolve_beam_config_value(
            launch_config.primary_config,
            key="beam.physsim.linkStatsOutputFileType",
            env_overrides={},
        )
        == "parquet"
    )
    assert resolve_beam_config_value(
        launch_config.primary_config,
        key="beam.warmStart.initialLinkstatsFilePath",
        env_overrides={},
    ) == str(launch_config.root / ".pilates" / "warmstarts" / "linkstats.csv.gz")
    assert (
        launch_config.root / ".pilates" / "warmstarts" / "linkstats.csv.gz"
    ).read_bytes() == warmstart.read_bytes()


def test_build_beam_launch_config_keeps_non_policy_overrides_strict(
    tmp_path: Path,
) -> None:
    import pytest

    from pilates.beam.launch_config import build_beam_launch_config

    settings = _settings()
    source_root = tmp_path / "beam" / "input" / "seattle"
    primary = _write_base_config(source_root)
    primary.write_text(
        primary.read_text(encoding="utf-8")
        .replace('  outputs.baseOutputDirectory = "/app/output"\n', "")
        .replace('  physsim.linkStatsOutputFileType = "csv.gz"\n', ""),
        encoding="utf-8",
    )
    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )

    with pytest.raises(KeyError, match="beam.outputs.baseOutputDirectory"):
        build_beam_launch_config(
            settings=settings,
            source_root=source_root,
            output_dir=tmp_path / "launch-config",
            identity=tracker.identity,
            prepared_inputs={},
        )


def test_build_beam_launch_config_stages_shared_includes_and_canonicalizes_output(
    tmp_path: Path,
) -> None:
    from pilates.beam.config_hocon import resolve_beam_config_value
    from pilates.beam.launch_config import build_beam_launch_config

    settings = _settings()
    source_root = tmp_path / "beam" / "input" / "seattle"
    source_primary = _write_shared_config_tree(source_root)
    source_bytes = source_primary.read_bytes()
    prepared = tmp_path / "preprocess-cache" / "plans.csv"
    prepared.parent.mkdir(parents=True)
    prepared.write_text("plans", encoding="utf-8")
    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )

    launch_config = build_beam_launch_config(
        settings=settings,
        source_root=source_root,
        output_dir=tmp_path / "launch-config",
        identity=tracker.identity,
        prepared_inputs={"plans_beam_in": prepared},
    )

    assert source_primary.read_bytes() == source_bytes
    assert (launch_config.root.parent / "common" / "matsim.conf").is_file()
    assert (
        resolve_beam_config_value(
            launch_config.primary_config,
            key="beam.outputs.baseOutputDirectory",
            env_overrides={},
        )
        == "/app/output"
    )
    assert resolve_beam_config_value(
        launch_config.primary_config,
        key="matsim.modules.network.inputNetworkFile",
        env_overrides={},
    ) == str(launch_config.root / "r5" / "network" / "physsim-network.xml")


def test_beam_run_adapter_uses_the_compiled_launch_tree(tmp_path: Path) -> None:
    from consist.core.step_context import StepContext

    from pilates.beam.launch_config import BeamLaunchConfig
    from pilates.workflows.step_consist_meta import consist_step_meta

    settings = _settings()
    launch_root = tmp_path / "launch" / "seattle"
    launch_primary = _write_base_config(launch_root)
    launch_config = BeamLaunchConfig(root=launch_root, primary_config=launch_primary)
    context = StepContext(
        func_name="beam_run",
        model="beam_run",
        runtime_kwargs={
            "settings": settings,
            "workspace": SimpleNamespace(full_path=tmp_path / "workspace"),
            "beam_launch_config": launch_config,
        },
    )

    adapter = consist_step_meta("beam_run")["adapter"](context)

    assert adapter is not None
    assert adapter.primary_config == launch_primary
    assert adapter.root_dirs == [launch_root]
    assert (
        adapter.reference_policies["beam.physsim.inputNetworkFilePath"].identity_policy
        == "output_or_runtime_ignored"
    )
    assert (
        adapter.reference_policies[
            "matsim.modules.network.inputNetworkFile"
        ].identity_policy
        == "output_or_runtime_ignored"
    )
    assert (
        adapter.reference_policies[
            "beam.agentsim.agents.rideHail.managers[0].initialization.filePath"
        ].identity_policy
        == "ignored"
    )


def test_beam_preprocess_adapter_defers_generated_vehicles_destination(
    tmp_path: Path,
) -> None:
    import pytest

    from consist.core.step_context import StepContext

    from pilates.beam.launch_config import BeamLaunchConfig
    from pilates.workflows.step_consist_meta import consist_step_meta

    settings = _settings()
    workspace = SimpleNamespace(
        full_path=tmp_path / "workspace",
        get_beam_mutable_data_dir=lambda: str(tmp_path / "workspace" / "beam-input"),
    )
    launch_root = Path(workspace.get_beam_mutable_data_dir()) / "seattle"
    launch_primary = _write_base_config(launch_root)
    (launch_root / "scenario").mkdir()
    with launch_primary.open("a", encoding="utf-8") as config_file:
        config_file.write(
            '\nbeam.agentsim.agents.vehicles.vehiclesFilePath = '
            '${beam.inputDirectory}"/urbansim/vehicles.parquet"\n'
        )

    preprocess_adapter = consist_step_meta("beam_preprocess")["adapter"](
        StepContext(
            func_name="beam_preprocess",
            model="beam_preprocess",
            runtime_kwargs={"settings": settings, "workspace": workspace},
        )
    )
    run_adapter = consist_step_meta("beam_run")["adapter"](
        StepContext(
            func_name="beam_run",
            model="beam_run",
            runtime_kwargs={
                "settings": settings,
                "workspace": workspace,
                "beam_launch_config": BeamLaunchConfig(
                    root=launch_root,
                    primary_config=launch_primary,
                ),
            },
        )
    )

    assert preprocess_adapter is not None
    assert run_adapter is not None
    preprocess_policy = preprocess_adapter.reference_policies[
        "beam.agentsim.agents.vehicles.vehiclesFilePath"
    ]
    assert preprocess_policy.identity_policy == "output_or_runtime_ignored"
    assert not preprocess_policy.required
    run_policy = run_adapter.reference_policies[
        "beam.agentsim.agents.vehicles.vehiclesFilePath"
    ]
    assert run_policy.identity_policy == "delegated_to_artifacts"
    assert run_policy.required

    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )
    preprocess_config = preprocess_adapter.discover(
        [launch_root], identity=tracker.identity
    )
    preprocess_adapter.canonicalize(
        preprocess_config,
        tracker=tracker,
        plan_only=True,
        strict=True,
    )
    run_config = run_adapter.discover([launch_root], identity=tracker.identity)
    with pytest.raises(FileNotFoundError, match="vehiclesFilePath"):
        run_adapter.canonicalize(
            run_config,
            tracker=tracker,
            plan_only=True,
            strict=True,
        )
