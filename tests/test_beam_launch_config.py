from __future__ import annotations

import gzip
import logging
from pathlib import Path
from types import SimpleNamespace

from consist import BindingResult, Tracker
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
        activitysim=None,
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


def test_beam_adapter_omits_activitysim_alias_without_activitysim_config(
    tmp_path: Path,
) -> None:
    from consist.core.step_context import StepContext

    from pilates.beam.launch_config import BeamLaunchConfig
    from pilates.workspace import Workspace
    from pilates.workflows.step_consist_meta import consist_step_meta

    settings = _settings()
    settings.activitysim = None
    settings.beam.local_output_folder = "beam/output"
    workspace = Workspace(settings, str(tmp_path), "run")
    launch_root = tmp_path / "launch" / "seattle"
    launch_primary = _write_base_config(launch_root)
    context = StepContext(
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

    adapter = consist_step_meta("beam_run")["adapter"](context)

    assert adapter is not None
    assert "activitysim_output" not in adapter.path_aliases


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
            "\nbeam.agentsim.agents.vehicles.vehiclesFilePath = "
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


def test_beam_preprocess_adapter_identity_is_portable_and_content_sensitive(
    tmp_path: Path,
) -> None:
    """Equivalent staged config trees retain one adapter identity across roots."""
    from consist.core.step_context import StepContext

    from pilates.workflows.step_consist_meta import consist_step_meta

    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )

    def adapter_identity(root: Path) -> str:
        workspace = SimpleNamespace(
            full_path=root.parent.parent.parent,
            get_beam_mutable_data_dir=lambda: str(root.parent),
        )
        adapter = consist_step_meta("beam_preprocess")["adapter"](
            StepContext(
                func_name="beam_preprocess",
                model="beam_preprocess",
                runtime_kwargs={"settings": _settings(), "workspace": workspace},
            )
        )
        assert adapter is not None
        canonical = adapter.canonicalize(
            adapter.discover([root], identity=tracker.identity),
            tracker=tracker,
            plan_only=True,
            strict=True,
        )
        return canonical.identity.identity_hash

    first_root = tmp_path / "workspace-a" / "beam" / "input" / "seattle"
    second_root = tmp_path / "workspace-b" / "beam" / "input" / "seattle"
    for root in (first_root, second_root):
        _write_base_config(root)
        (root / "scenario").mkdir()

    first_identity = adapter_identity(first_root)
    assert first_identity == adapter_identity(second_root)

    with (second_root / "beam.conf").open("a", encoding="utf-8") as config_file:
        config_file.write("\nbeam.sampleSizeAsFractionOfPopulation = 0.5\n")
    assert first_identity != adapter_identity(second_root)


def test_beam_preprocess_step_identity_uses_portable_closure_shape(
    tmp_path: Path,
) -> None:
    """Role shape and launch choices matter; workspace/scheduler paths do not."""
    from consist.core.step_context import StepContext

    from pilates.workflows.resolved_inputs import ResolvedStepInputs
    from pilates.workflows.step_consist_meta import consist_step_meta
    from pilates.workflows.steps.beam import (
        _beam_preprocess_identity_payload,
    )

    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )
    roots = []
    for workspace_name in ("workspace-a", "workspace-b"):
        root = tmp_path / workspace_name / "beam" / "input" / "seattle"
        _write_base_config(root)
        (root / "scenario").mkdir()
        roots.append(root)

    def step_identity(
        root: Path,
        *,
        include_warmstart: bool = False,
        file_format: str = "parquet",
        scheduler_allocation: str,
        output_destination: Path,
    ) -> dict[str, object]:
        settings = _settings()
        settings.activitysim = SimpleNamespace(file_format=file_format)
        settings.beam.skim_zone_geoid_col = "TAZ"
        inputs = {
            "plans_beam_in": root / "plans.parquet",
            "households_beam_in": root / "households.parquet",
            "persons_beam_in": root / "persons.parquet",
        }
        if include_warmstart:
            inputs["linkstats_warmstart"] = root / "linkstats.parquet"
        resolved = ResolvedStepInputs(
            step_name="beam_preprocess",
            binding=BindingResult(inputs=inputs),
            metadata={
                "beam_preprocess_context": {
                    "primary_config": root / "beam.conf",
                    "exchange_scenario_folder": root / "scenario",
                }
            },
        )
        workspace = SimpleNamespace(
            full_path=root.parent.parent.parent,
            get_beam_mutable_data_dir=lambda: str(root.parent),
        )
        runtime_kwargs = {
            "settings": settings,
            "workspace": workspace,
            "beam_preprocess_context": dict(
                resolved.metadata["beam_preprocess_context"]
            ),
            "beam_preprocess_identity_closure": _beam_preprocess_identity_payload(
                settings=settings,
                resolved_inputs=resolved,
            ),
            "scheduler_allocation": scheduler_allocation,
            "output_destination": output_destination,
        }
        config = consist_step_meta("beam_preprocess")["config"](
            StepContext(
                func_name="beam_preprocess",
                model="beam_preprocess",
                runtime_kwargs=runtime_kwargs,
            )
        )
        return config["beam_preprocess_identity"]

    first = step_identity(
        roots[0],
        scheduler_allocation="12345",
        output_destination=tmp_path / "workspace-a" / "outputs",
    )
    assert first == step_identity(
        roots[1],
        scheduler_allocation="67890",
        output_destination=tmp_path / "workspace-b" / "other-outputs",
    )
    assert first != step_identity(
        roots[1],
        include_warmstart=True,
        scheduler_allocation="67890",
        output_destination=tmp_path / "workspace-b" / "other-outputs",
    )
    assert first != step_identity(
        roots[1],
        file_format="csv",
        scheduler_allocation="67890",
        output_destination=tmp_path / "workspace-b" / "other-outputs",
    )


def test_build_beam_launch_config_clears_missing_configured_vehicles_path(
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
    source_primary.write_text(
        source_primary.read_text(encoding="utf-8")
        + '\nbeam.agentsim.agents.vehicles.vehiclesFilePath = ${beam.inputDirectory}"/urbansim/missing_vehicles.parquet"\n',
        encoding="utf-8",
    )
    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )

    launch_config = build_beam_launch_config(
        settings=settings,
        source_root=source_root,
        output_dir=tmp_path / "launch-config",
        identity=tracker.identity,
        prepared_inputs={},
    )

    env = beam_config_env_overrides(settings, config_root=launch_config.root)
    assert (
        resolve_beam_config_value(
            launch_config.primary_config,
            key="beam.agentsim.agents.vehicles.vehiclesFilePath",
            env_overrides=env,
        )
        == ""
    )


def test_build_beam_launch_config_writes_empty_parquet_exchange_stubs_when_no_population_inputs(
    tmp_path: Path,
) -> None:
    import pyarrow.parquet as pq

    from pilates.beam.launch_config import build_beam_launch_config

    settings = _settings()
    source_root = tmp_path / "beam" / "input" / "seattle"
    _write_base_config(source_root)
    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )

    launch_config = build_beam_launch_config(
        settings=settings,
        source_root=source_root,
        output_dir=tmp_path / "launch-config",
        identity=tracker.identity,
        prepared_inputs={},
    )

    exchange_dir = launch_config.root / "scenario"
    expected_columns = {
        "persons": [
            "person_id",
            "household_id",
            "age",
            "sex",
            "industry",
            "value_of_time",
        ],
        "plans": [
            "trip_id",
            "tour_id",
            "person_id",
            "PlanElementIndex",
            "ActivityElement",
            "trip_mode",
            "ActivityType",
            "x",
            "y",
            "departure_time",
            "trip_dur_min",
            "trip_cost_dollars",
        ],
        "households": ["household_id", "cars", "income", "block_id"],
        "blocks": ["block_id", "x", "y"],
    }
    for stem, cols in expected_columns.items():
        parquet_path = exchange_dir / f"{stem}.parquet"
        assert parquet_path.is_file()
        table = pq.read_table(parquet_path)
        assert table.num_rows == 0
        assert table.column_names == cols


def test_build_beam_launch_config_copies_fallback_blocks_parquet_to_nested_exchange_folder(
    tmp_path: Path,
) -> None:
    from pilates.beam.launch_config import build_beam_launch_config

    settings = _settings()
    settings.beam.scenario_folder = "urbansim"
    source_root = tmp_path / "beam" / "input" / "seattle"
    source_primary = _write_base_config(source_root)
    source_primary.write_text(
        source_primary.read_text(encoding="utf-8").replace(
            '  exchange.scenario.folder = ${beam.inputDirectory}"/scenario"\n',
            '  exchange.scenario.folder = ${beam.inputDirectory}"/urbansim/2018"\n',
        ),
        encoding="utf-8",
    )
    fallback_blocks = source_root / "urbansim" / "blocks.parquet"
    fallback_blocks.parent.mkdir(parents=True, exist_ok=True)
    fallback_blocks.write_bytes(b"PAR1blocks")

    prepared_root = tmp_path / "preprocess-cache"
    prepared_root.mkdir(parents=True)
    plans_in = prepared_root / "plans.parquet"
    plans_in.write_bytes(b"PAR1plans")

    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )

    launch_config = build_beam_launch_config(
        settings=settings,
        source_root=source_root,
        output_dir=tmp_path / "launch-config",
        identity=tracker.identity,
        prepared_inputs={"plans_beam_in": plans_in},
    )

    nested_blocks = launch_config.root / "urbansim" / "2018" / "blocks.parquet"
    assert nested_blocks.is_file()
    assert nested_blocks.read_bytes() == b"PAR1blocks"


def test_build_beam_launch_config_hardlinks_static_files_and_copies_configs(
    tmp_path: Path,
) -> None:
    import os
    from pilates.beam.launch_config import build_beam_launch_config

    settings = _settings()
    source_root = tmp_path / "beam" / "input" / "seattle"
    source_primary = _write_base_config(source_root)

    # Create static assets in the source tree: a network file and a scenario table
    static_net = source_root / "r5" / "network" / "osm.mapdb"
    static_net.write_bytes(b"mock_mapdb_binary_content_12345")
    static_table = source_root / "scenario" / "transit_vehicles.csv"
    static_table.parent.mkdir(parents=True, exist_ok=True)
    static_table.write_text("vehicle_id,type\nv1,bus\n", encoding="utf-8")

    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )

    launch_config = build_beam_launch_config(
        settings=settings,
        source_root=source_root,
        output_dir=tmp_path / "launch-config",
        identity=tracker.identity,
        prepared_inputs={},
    )

    staged_primary = launch_config.primary_config
    staged_net = launch_config.root / "r5" / "network" / "osm.mapdb"
    staged_table = launch_config.root / "scenario" / "transit_vehicles.csv"

    # 1. Config (.conf) files must be copied (distinct inodes) so modifying the staged config
    # never mutates the original source tree.
    assert staged_primary.exists()
    assert os.stat(staged_primary).st_ino != os.stat(source_primary).st_ino

    # 2. Static non-config files must be hardlinked (identical inode numbers).
    assert staged_net.exists()
    assert os.stat(staged_net).st_ino == os.stat(static_net).st_ino
    assert staged_table.exists()
    assert os.stat(staged_table).st_ino == os.stat(static_table).st_ino

    # 3. Overwriting a file in staged launch tree (or staging inputs) must not mutate source
    # because destination is unlinked before writing or copying.
    staged_table.unlink()
    staged_table.write_text("vehicle_id,type\nv2,train\n", encoding="utf-8")
    assert static_table.read_text(encoding="utf-8") == "vehicle_id,type\nv1,bus\n"


def test_link_or_copy_static_file_falls_back_on_exdev(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import errno
    import os
    from pilates.beam.launch_config import _link_or_copy_static_file

    src_file = tmp_path / "data" / "large_network.dat"
    src_file.parent.mkdir(parents=True, exist_ok=True)
    src_file.write_bytes(b"payload_data")

    dst_fallback = tmp_path / "stage_fallback" / "large_network.dat"

    # Simulate cross-device link failure on os.link
    def mock_link_fail(src, dst):
        raise OSError(errno.EXDEV, "Cross-device link")

    monkeypatch.setattr(os, "link", mock_link_fail)

    # When os.link raises EXDEV, it must fall back directly to copy rather than symlink,
    # preventing broken container bind mounts and root-escape validation errors.
    result = _link_or_copy_static_file(src_file, dst_fallback)
    assert result == "copy"
    assert not dst_fallback.is_symlink()
    assert dst_fallback.is_file()
    assert os.stat(dst_fallback).st_ino != os.stat(src_file).st_ino
    assert dst_fallback.read_bytes() == b"payload_data"
