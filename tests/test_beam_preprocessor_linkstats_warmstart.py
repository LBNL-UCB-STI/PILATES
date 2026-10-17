from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from consist import Tracker
from pilates.beam.launch_config import build_beam_launch_config
from pilates.beam.outputs import BeamPreprocessOutputs
from pilates.beam.preprocessor import BeamPreprocessor
from pilates.config.models import BeamArtifactFormatsConfig
from pilates.generic.records import FileRecord, RecordStore
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
)


def _make_preprocessor():
    settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        beam=SimpleNamespace(
            config="beam.conf",
            scenario_folder="scenario",
            router_directory="router",
            discard_plans_every_year=True,
            artifact_formats=BeamArtifactFormatsConfig(),
            activitysim_skims_file_base_name="skimsActivitySimOD",
        ),
        shared=SimpleNamespace(geography=SimpleNamespace(zones=None)),
    )
    state = SimpleNamespace(
        full_settings=settings,
        current_year=2018,
        forecast_year=2018,
        current_inner_iter=1,
        set_sub_stage_progress=lambda _progress: None,
    )
    return BeamPreprocessor("beam", state)


def test_handle_linkstats_accepts_versioned_parquet_warmstart(tmp_path):
    preprocessor = _make_preprocessor()
    workspace = MagicMock()
    workspace.full_path = str(tmp_path)
    workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam" / "input")

    parquet_path = tmp_path / "beam_output" / "2.linkstats.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.write_bytes(b"PAR1stub")

    ignored_path = tmp_path / "beam_output" / "2.events.PathTraversal.links.parquet"
    ignored_path.write_bytes(b"PAR1stub")

    previous_records = [
        FileRecord(
            file_path=str(ignored_path),
            short_name="path_traversal_links_2018_0",
        ),
        FileRecord(
            file_path=str(parquet_path),
            short_name="linkstats_parquet_2018_0",
        ),
    ]

    store = RecordStore()
    preprocessor._handle_linkstats(workspace, previous_records, store)

    warmstart = [
        rec
        for rec in store.all_records()
        if getattr(rec, "short_name", "") == "linkstats_warmstart"
    ]
    assert len(warmstart) == 1
    assert warmstart[0].file_path == str(parquet_path.relative_to(tmp_path))
    staged_path = tmp_path / warmstart[0].file_path
    assert staged_path.read_bytes() == b"PAR1stub"
    assert warmstart[0].metadata["source"] == "previous_beam_output"
    assert not (
        tmp_path / "beam" / "input" / "seattle" / "_pilates" / "linkstats"
    ).exists()


def test_preprocess_accepts_artifact_mappings_and_returns_typed_outputs(tmp_path):
    preprocessor = _make_preprocessor()
    workspace = MagicMock()
    workspace.full_path = str(tmp_path)
    workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam" / "input")

    activity_path = tmp_path / "beam_plans.parquet"
    previous_path = tmp_path / "previous_linkstats.parquet"
    fallback_path = tmp_path / "fallback_households.csv"
    activity_path.write_text("plans")
    previous_path.write_text("linkstats")
    fallback_path.write_text("households")

    captured = {}

    def _fake_preprocess(_workspace, previous_records):
        captured["mapping"] = previous_records.to_mapping()
        return previous_records

    preprocessor._preprocess = _fake_preprocess

    outputs = preprocessor.preprocess(
        workspace,
        beam_preprocess_inputs={
            BEAM_PLANS_IN: activity_path,
            BEAM_HOUSEHOLDS_IN: fallback_path,
            LINKSTATS_WARMSTART: previous_path,
        },
    )

    assert isinstance(outputs, BeamPreprocessOutputs)
    assert captured["mapping"]["beam_plans"] == str(activity_path)
    assert captured["mapping"]["households"] == str(fallback_path)
    assert captured["mapping"]["linkstats_warmstart"] == str(previous_path)


def test_handle_linkstats_explicit_warmstart_iter0_sets_initial_inputs_and_no_warning(
    tmp_path, caplog
):
    preprocessor = _make_preprocessor()
    preprocessor.state.current_inner_iter = 0
    workspace = MagicMock()
    workspace.full_path = str(tmp_path)
    workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam" / "input")

    warmstart_path = tmp_path / "initial_linkstats.parquet"
    warmstart_path.write_bytes(b"PAR1initial")

    record = FileRecord(
        file_path=str(warmstart_path),
        short_name=LINKSTATS_WARMSTART,
        description="Explicit BEAM preprocess input: linkstats_warmstart",
    )
    store = RecordStore()
    with caplog.at_level(logging.WARNING):
        preprocessor._handle_linkstats(workspace, [record], store)

    warmstart = [
        rec
        for rec in store.all_records()
        if getattr(rec, "short_name", "") == LINKSTATS_WARMSTART
    ]
    assert len(warmstart) == 1
    assert warmstart[0].metadata["source"] == "initial_inputs"
    assert warmstart[0].file_path == str(warmstart_path.relative_to(tmp_path))
    assert "[NOT IDEAL] Using an unversioned 'linkstats' record" not in caplog.text


def test_build_beam_launch_config_preserves_warmstart_provenance_sidecar(tmp_path):
    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "test_tracker.duckdb"),
    )
    settings = _make_preprocessor().settings

    region_dir = tmp_path / "source_tree" / "seattle"
    region_dir.mkdir(parents=True)
    primary = region_dir / "beam.conf"
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
    (region_dir / "r5" / "network").mkdir(parents=True)

    warmstart_path = tmp_path / "inputs" / "linkstats_warmstart.parquet"
    warmstart_path.parent.mkdir(parents=True, exist_ok=True)
    warmstart_path.write_bytes(b"PAR1warmstart_content")
    sidecar_path = Path(f"{warmstart_path}.provenance.json")
    sidecar_path.write_text(
        '{"lineage": "test_warmstart_provenance"}', encoding="utf-8"
    )

    output_dir = tmp_path / "launch_configs"
    launch_config = build_beam_launch_config(
        settings=settings,
        source_root=region_dir,
        output_dir=output_dir,
        identity=tracker.identity,
        prepared_inputs={LINKSTATS_WARMSTART: warmstart_path},
    )

    staged_warmstart = (
        launch_config.root / ".pilates" / "warmstarts" / "linkstats_warmstart.parquet"
    )
    assert staged_warmstart.is_file()
    assert staged_warmstart.read_bytes() == b"PAR1warmstart_content"
    staged_sidecar = Path(f"{staged_warmstart}.provenance.json")
    assert staged_sidecar.is_file()
    assert (
        staged_sidecar.read_text(encoding="utf-8")
        == '{"lineage": "test_warmstart_provenance"}'
    )


def test_project_beam_preprocess_outputs_retains_fallback_initial_warmstart(tmp_path):
    from consist import BindingResult
    from pilates.workflows.steps.beam import (
        ResolvedStepInputs,
        _project_beam_preprocess_outputs,
    )

    settings = _make_preprocessor().settings
    initial_warmstart_file = tmp_path / "warmstart" / "initial.parquet"
    initial_warmstart_file.parent.mkdir(parents=True, exist_ok=True)
    initial_warmstart_file.write_bytes(b"PAR1initial_warmstart")
    settings.beam.warmstart_linkstats_path = str(initial_warmstart_file)

    state = SimpleNamespace(iteration=0)
    workspace = MagicMock()
    workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam_mutable")
    (tmp_path / "beam_mutable").mkdir(parents=True, exist_ok=True)

    declared_dest = (
        tmp_path
        / "beam_mutable"
        / ".pilates-consist-outputs"
        / "beam_preprocess"
        / "linkstats_warmstart.parquet"
    )
    declared_dest.parent.mkdir(parents=True, exist_ok=True)
    declared_dest.write_bytes(b"PAR1declared_dest")

    plans_path = tmp_path / "plans.csv.gz"
    plans_path.write_text("plans")
    households_path = tmp_path / "households.csv.gz"
    households_path.write_text("households")
    persons_path = tmp_path / "persons.csv.gz"
    persons_path.write_text("persons")

    resolved_inputs = ResolvedStepInputs(
        step_name="beam_preprocess",
        binding=BindingResult(inputs={}),
        metadata={"native_output_keys": (LINKSTATS_WARMSTART,)},
    )

    outputs = {
        BEAM_PLANS_IN: plans_path,
        BEAM_HOUSEHOLDS_IN: households_path,
        BEAM_PERSONS_IN: persons_path,
    }

    projected = _project_beam_preprocess_outputs(
        outputs,
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved_inputs,
    )

    assert LINKSTATS_WARMSTART in projected.prepared_inputs
    assert projected.prepared_inputs[LINKSTATS_WARMSTART] == declared_dest
    assert declared_dest.suffix == ".parquet"


def test_project_beam_preprocess_outputs_restores_provenance_sidecar_from_artifact_metadata(
    tmp_path,
):
    from consist import Artifact, BindingResult
    from pilates.workflows.steps.beam import (
        ResolvedStepInputs,
        _project_beam_preprocess_outputs,
    )

    settings = _make_preprocessor().settings
    state = SimpleNamespace(iteration=0)
    workspace = MagicMock()
    workspace.full_path = str(tmp_path)
    workspace.get_beam_output_dir.return_value = str(tmp_path / "beam_output")
    workspace.get_beam_mutable_data_dir.return_value = str(tmp_path / "beam_mutable")
    (tmp_path / "beam_mutable").mkdir(parents=True, exist_ok=True)

    warmstart_path = (
        tmp_path
        / "beam_mutable"
        / ".pilates-consist-outputs"
        / "beam_preprocess"
        / "linkstats_warmstart.parquet"
    )
    warmstart_path.parent.mkdir(parents=True, exist_ok=True)
    warmstart_path.write_bytes(b"PAR1cached_warmstart")

    sidecar_path = Path(f"{warmstart_path}.provenance.json")
    assert not sidecar_path.exists()

    prov_metadata = {"source": "prior_run", "run_id": "test-123"}
    artifact = Artifact(
        container_uri=str(warmstart_path),
        driver="parquet",
        meta={"linkstats_provenance": prov_metadata},
    )

    plans_path = tmp_path / "plans.csv.gz"
    plans_path.write_text("plans")
    households_path = tmp_path / "households.csv.gz"
    households_path.write_text("households")
    persons_path = tmp_path / "persons.csv.gz"
    persons_path.write_text("persons")

    resolved_inputs = ResolvedStepInputs(
        step_name="beam_preprocess",
        binding=BindingResult(inputs={LINKSTATS_WARMSTART: artifact}),
        metadata={"native_output_keys": (LINKSTATS_WARMSTART,)},
    )

    outputs = {
        BEAM_PLANS_IN: plans_path,
        BEAM_HOUSEHOLDS_IN: households_path,
        BEAM_PERSONS_IN: persons_path,
        LINKSTATS_WARMSTART: artifact,
    }

    projected = _project_beam_preprocess_outputs(
        outputs,
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved_inputs,
    )

    assert LINKSTATS_WARMSTART in projected.prepared_inputs
    assert sidecar_path.is_file()
    import json

    assert json.loads(sidecar_path.read_text(encoding="utf-8")) == prov_metadata
    assert (
        projected.prepared_input_metadata[LINKSTATS_WARMSTART]["linkstats_provenance"]
        == prov_metadata
    )
