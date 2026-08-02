import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from consist import (
    BindingResult,
    Tracker,
    resolve_step_contract,
)
from pilates.config.models import BeamArtifactFormatsConfig
from pilates.runtime.archive_paths import resolve_workspace_uri_path

from pilates.workflows.artifact_keys import (
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_CONFIG_FILE,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
)
from pilates.workflows.steps.beam import (
    _beam_preprocess_native_output_paths,
    _beam_run_native_output_paths,
    _materialize_native_outputs,
    _native_beam_run,
    _project_beam_run_outputs,
    _resolve_beam_preprocess_inputs,
    _resolved_beam_inputs,
    _native_beam_postprocess,
    beam_full_skim,
    beam_postprocess,
    beam_preprocess,
    beam_run,
)
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.steps import beam as beam_steps
from pilates.workflows.steps.shared import BeamRunOutputs
from pilates.beam.launch_config import BeamLaunchConfig


def _empty_resolved_inputs() -> ResolvedStepInputs:
    return ResolvedStepInputs(
        step_name="test",
        binding=BindingResult(inputs={}),
    )


def _beam_settings(*, skim_format: str = "zarr") -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(region="test-region"),
        beam=SimpleNamespace(
            artifact_formats=BeamArtifactFormatsConfig(activitysim_skims=skim_format)
        ),
    )


def _write_event_types(path: Path, event_types: list[str]) -> None:
    import pandas as pd

    pd.DataFrame({"type": event_types}).to_parquet(path, index=False)


def test_materialize_native_outputs_preserves_same_directory_source(
    tmp_path: Path,
) -> None:
    skims = tmp_path / "skims.zarr"
    marker = skims / ".zgroup"
    marker.parent.mkdir()
    marker.write_text('{"zarr_format": 2}\n', encoding="utf-8")

    _materialize_native_outputs(
        source_paths={ZARR_SKIMS: skims},
        declared_outputs={ZARR_SKIMS: skims},
    )

    assert marker.read_text(encoding="utf-8") == '{"zarr_format": 2}\n'


def test_beam_preprocess_output_paths_preserve_parquet_warmstart_suffix(
    tmp_path: Path,
) -> None:
    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
    )
    resolved_inputs = ResolvedStepInputs(
        step_name="beam_preprocess",
        binding=BindingResult(
            inputs={LINKSTATS_WARMSTART: tmp_path / "linkstats.parquet"}
        ),
        metadata={"native_output_keys": (LINKSTATS_WARMSTART,)},
    )

    outputs = _beam_preprocess_native_output_paths(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        resolved_inputs=resolved_inputs,
    )

    assert outputs[LINKSTATS_WARMSTART] == (
        tmp_path
        / "beam-input"
        / ".pilates-consist-outputs"
        / "beam_preprocess"
        / "linkstats_warmstart.parquet"
    )


def test_beam_run_output_paths_keep_linkstats_format_neutral(tmp_path: Path) -> None:
    workspace = SimpleNamespace(
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
    )

    outputs = _beam_run_native_output_paths(
        settings=_beam_settings(),
        state=SimpleNamespace(forecast_year=2030, iteration=1),
        workspace=workspace,
    )

    assert outputs["linkstats"] == (
        tmp_path
        / "beam-output"
        / "test-region"
        / "year-2030-iteration-1"
        / ".pilates-consist-outputs"
        / "beam_run"
        / "linkstats"
    )


@pytest.mark.parametrize(
    ("skim_format", "expected_key", "expected_suffix"),
    (
        ("zarr", None, None),
        ("omx", "raw_od_skims_2030_2", ".omx"),
    ),
)
def test_beam_run_output_paths_declare_only_configured_activitysim_skim_format(
    tmp_path: Path,
    skim_format: str,
    expected_key: str | None,
    expected_suffix: str | None,
) -> None:
    workspace = SimpleNamespace(
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
    )

    outputs = _beam_run_native_output_paths(
        settings=_beam_settings(skim_format=skim_format),
        state=SimpleNamespace(forecast_year=2030, iteration=2),
        workspace=workspace,
    )

    skim_keys = {key for key in outputs if key.startswith("raw_od_skims")}
    if expected_key is None:
        assert skim_keys == set()
    else:
        assert skim_keys == {expected_key}
        assert outputs[expected_key].suffix == expected_suffix


def test_beam_run_declares_only_exact_region_year_iteration_output_set(
    tmp_path: Path,
) -> None:
    workspace = SimpleNamespace(
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
    )

    output_set = beam_run.output_sets(
        settings=_beam_settings(),
        state=SimpleNamespace(forecast_year=2030, iteration=2),
        workspace=workspace,
    )["beam_run_outputs"]
    scalar_paths = beam_run.output_paths(
        settings=_beam_settings(),
        state=SimpleNamespace(forecast_year=2030, iteration=2),
        workspace=workspace,
        resolved_inputs=None,
    )

    assert output_set.root == (
        tmp_path / "beam-output" / "test-region" / "year-2030-iteration-2"
    )
    assert output_set.include == "**/*"
    assert output_set.exclude == "**/*.zarr/**"
    assert output_set.recursive is True
    assert all(Path(path).is_relative_to(Path(output_set.root)) for path in scalar_paths.values())
    assert not any(key.startswith("raw_od_skims_zarr_") for key in scalar_paths)


def test_native_beam_run_does_not_enqueue_declared_tree_members() -> None:
    assert "enqueue_archive_copy" not in inspect.getsource(beam_steps._native_beam_run)


def test_archive_paths_leaves_legacy_beam_input_uri_unrewritten() -> None:
    assert resolve_workspace_uri_path("beam_input://test-region/beam.conf") == (
        "beam_input://test-region/beam.conf"
    )


def test_beam_output_layout_cache_versions_reject_pre_migration_artifacts() -> None:
    options_kwargs = {
        "settings": SimpleNamespace(),
        "state": SimpleNamespace(),
        "workspace": SimpleNamespace(),
    }

    assert beam_preprocess.cache_options(**options_kwargs).cache_version == 1
    assert beam_run.cache_options(**options_kwargs).cache_version == 1


def test_native_beam_definitions_resolve_consist_contracts(
    tmp_path: Path, monkeypatch
) -> None:
    definitions = (
        beam_preprocess,
        beam_run,
        beam_postprocess,
        beam_full_skim,
    )

    mutable_data_dir = tmp_path / "beam-input"
    output_dir = tmp_path / "beam-output"
    mutable_data_dir.mkdir()
    output_dir.mkdir()
    workspace = SimpleNamespace(
        full_path=str(tmp_path),
        get_beam_mutable_data_dir=lambda: str(mutable_data_dir),
        get_beam_output_dir=lambda: str(output_dir),
        get_asim_output_dir=lambda: str(tmp_path / "asim-output"),
    )
    settings = SimpleNamespace(
        activitysim=None,
        beam=SimpleNamespace(
            config="beam.conf",
            full_skim=None,
            artifact_formats=BeamArtifactFormatsConfig(),
        ),
        run=SimpleNamespace(
            region="test-region",
            models=SimpleNamespace(land_use=None),
        ),
        write_skims_to_omx=False,
    )
    state = SimpleNamespace(
        year=2030, current_year=2030, iteration=2, current_inner_iter=2
    )
    monkeypatch.setenv("PILATES_DISABLE_BEAM_CONFIG_ADAPTER", "1")
    monkeypatch.setattr(
        "pilates.workflows.step_consist_meta.build_step_consist_kwargs",
        lambda *_args, **_kwargs: {},
    )

    contracts = {
        definition.name: resolve_step_contract(
            definition.function,
            year=2030,
            iteration=2,
            phase="run",
            stage="traffic_assignment",
            runtime_kwargs={
                "settings": settings,
                "state": state,
                "workspace": workspace,
            },
        )
        for definition in definitions
    }

    assert set(contracts) == {
        "beam_preprocess",
        "beam_run",
        "beam_postprocess",
        "beam_full_skim",
    }
    assert {name: contract.model for name, contract in contracts.items()} == {
        name: name for name in contracts
    }
    assert {name: contract.name for name, contract in contracts.items()} == {
        "beam_preprocess": "beam_preprocess__y2030__i2__phase_run",
        "beam_run": "beam_run__y2030__i2__phase_run",
        "beam_postprocess": "beam_postprocess__y2030__i2__phase_run",
        "beam_full_skim": "beam_full_skim__y2030__i2__phase_run",
    }
    assert all(contract.input_binding == "paths" for contract in contracts.values())
    assert contracts["beam_preprocess"].output_paths
    assert contracts["beam_run"].output_paths
    assert contracts["beam_full_skim"].output_paths


def test_native_beam_artifact_inputs_match_callable_parameter_names() -> None:
    for definition in (beam_preprocess, beam_run, beam_full_skim):
        declared = definition.function.__consist_step__
        declared_artifact_inputs = set(declared.inputs or {})
        declared_artifact_inputs.update(declared.input_keys or ())
        declared_artifact_inputs.update(declared.optional_input_keys or ())

        assert declared_artifact_inputs <= set(
            inspect.signature(definition.function).parameters
        )


def test_native_beam_run_validates_canonical_launch_references_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "beam.conf"
    config.write_text("beam {}\n", encoding="utf-8")
    inputs = {}
    for key in (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN):
        path = tmp_path / f"{key}.csv"
        path.write_text(f"{key}\n", encoding="utf-8")
        inputs[key] = path
    mutable_data_dir = tmp_path / "beam-input"
    mutable_data_dir.mkdir()
    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(mutable_data_dir),
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
    )
    context = SimpleNamespace(canonicalization=object())
    events: list[tuple[str, object]] = []

    launch_config = BeamLaunchConfig(root=tmp_path, primary_config=config)

    def validate_linkstats(*, settings, workspace, run_context, config_root) -> None:
        assert workspace is not None
        assert config_root == launch_config.root
        events.append(("linkstats", run_context))

    def validate_r5(*, settings, workspace, run_context, config_root) -> None:
        assert workspace is not None
        assert config_root == launch_config.root
        events.append(("r5", run_context))

    class Runner:
        def __init__(self, name: str, state: object) -> None:
            assert name == "beam_run"

        def run(self, *_args, **_kwargs) -> object:
            events.append(("runner", context))
            return BeamRunOutputs(
                beam_output_dir=tmp_path / "beam-output",
                raw_outputs={},
            )

    monkeypatch.setattr(
        beam_steps,
        "validate_staged_linkstats_reference",
        validate_linkstats,
        raising=False,
    )
    monkeypatch.setattr(
        beam_steps,
        "validate_r5_execution_reference",
        validate_r5,
        raising=False,
    )
    monkeypatch.setattr(beam_steps, "BeamRunner", Runner)
    monkeypatch.setattr(
        beam_steps, "_validate_native_outputs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        beam_steps, "_materialize_native_outputs", lambda **_kwargs: None
    )

    beam_steps._native_beam_run(
        config,
        inputs[BEAM_PLANS_IN],
        inputs[BEAM_HOUSEHOLDS_IN],
        inputs[BEAM_PERSONS_IN],
        settings=_beam_settings(),
        state=SimpleNamespace(year=2030, iteration=1),
        workspace=workspace,
        beam_launch_config=launch_config,
        _consist_ctx=context,
    )

    assert events == [
        ("linkstats", context),
        ("r5", context),
        ("runner", context),
    ]


def test_native_beam_run_fails_closed_before_starting_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "beam.conf"
    config.write_text("beam {}\n", encoding="utf-8")
    inputs = []
    for key in (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN):
        path = tmp_path / f"{key}.csv"
        path.write_text(f"{key}\n", encoding="utf-8")
        inputs.append(path)
    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
    )

    def reject_linkstats(**_kwargs) -> None:
        raise RuntimeError("linkstats launch proof failed")

    class Runner:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("BeamRunner must not start after failed launch proof")

    monkeypatch.setattr(
        beam_steps, "validate_staged_linkstats_reference", reject_linkstats
    )
    monkeypatch.setattr(beam_steps, "BeamRunner", Runner)

    with pytest.raises(RuntimeError, match="linkstats launch proof failed"):
        beam_steps._native_beam_run(
            config,
            *inputs,
            settings=object(),
            state=SimpleNamespace(year=2030, iteration=1),
            workspace=workspace,
            beam_launch_config=BeamLaunchConfig(root=tmp_path, primary_config=config),
            _consist_ctx=SimpleNamespace(canonicalization=object()),
        )


def test_beam_full_skim_resolver_keeps_ordinary_binding_for_workspace_runner(
    tmp_path: Path,
) -> None:
    assert beam_full_skim.preflight_identity is False

    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )
    values = {}
    with tracker.start_run("seed", "test"):
        for key in (
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
            "linkstats_warmstart",
        ):
            source = tmp_path / f"{key}.csv"
            source.write_text(f"{key}\n", encoding="utf-8")
            values[key] = tracker.log_artifact(source, key=key, direction="input")

    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            return values.get(key, default)

    launch_config_path = tmp_path / "beam-launch" / "beam.conf"
    launch_config_path.parent.mkdir()
    launch_config_path.write_text("beam {}\n", encoding="utf-8")
    launch_config = BeamLaunchConfig(
        root=launch_config_path.parent,
        primary_config=launch_config_path,
    )
    settings = object()
    state = SimpleNamespace()
    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input")
    )
    resolved = beam_full_skim.resolve_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=Coupler(),
        launch_config=launch_config,
    )

    assert isinstance(resolved.binding, BindingResult)
    assert dict(resolved.binding.inputs or {}) == {
        "beam_config_file": launch_config_path,
        **values,
    }
    assert set(resolved.logical_destinations) == {"beam_config_file", *values}


def test_beam_preprocess_binds_activitysim_outputs_without_duplicate_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "beam.conf"
    config_path.write_text("beam {}\n", encoding="utf-8")
    activitysim_outputs = {
        "beam_plans_asim_out": tmp_path / "beam_plans.parquet",
        "households_asim_out": tmp_path / "households.parquet",
        "persons_asim_out": tmp_path / "persons.parquet",
    }

    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            return activitysim_outputs.get(key, default)

    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(traffic_assignment=None, travel=None),
        )
    )
    monkeypatch.setattr(
        beam_steps,
        "_require_primary_beam_config",
        lambda _settings, _workspace: config_path,
    )
    monkeypatch.setattr(
        beam_steps,
        "build_enabled_workflow_surface",
        lambda _settings: SimpleNamespace(
            profile=SimpleNamespace(vehicle_ownership_model_enabled=False)
        ),
    )

    resolved = _resolve_beam_preprocess_inputs(
        settings=settings,
        state=SimpleNamespace(year=2030),
        workspace=workspace,
        coupler=Coupler(),
    )

    inputs = resolved.binding.inputs or {}
    assert inputs[BEAM_PLANS_IN] is activitysim_outputs["beam_plans_asim_out"]
    assert inputs[BEAM_HOUSEHOLDS_IN] is activitysim_outputs["households_asim_out"]
    assert inputs[BEAM_PERSONS_IN] is activitysim_outputs["persons_asim_out"]
    assert resolved.source_by_role[BEAM_PLANS_IN] == "coupler"
    assert resolved.selected_key_by_role == {
        BEAM_CONFIG_FILE: BEAM_CONFIG_FILE,
        BEAM_PLANS_IN: "beam_plans_asim_out",
        BEAM_HOUSEHOLDS_IN: "households_asim_out",
        BEAM_PERSONS_IN: "persons_asim_out",
    }
    assert set(inputs) == {
        BEAM_CONFIG_FILE,
        BEAM_PLANS_IN,
        BEAM_HOUSEHOLDS_IN,
        BEAM_PERSONS_IN,
    }


def test_beam_preprocess_preserves_atlas_vehicles_producer_basename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "beam.conf"
    config_path.write_text("beam {}\n", encoding="utf-8")
    inputs = {
        "beam_plans_asim_out": tmp_path / "beam_plans.parquet",
        "households_asim_out": tmp_path / "households.parquet",
        "persons_asim_out": tmp_path / "persons.parquet",
        ATLAS_VEHICLES2_OUTPUT: tmp_path / "vehicles2_2019.csv",
    }

    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            return inputs.get(key, default)

    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
    )
    monkeypatch.setattr(
        beam_steps,
        "_require_primary_beam_config",
        lambda _settings, _workspace: config_path,
    )
    monkeypatch.setattr(
        beam_steps,
        "build_enabled_workflow_surface",
        lambda _settings: SimpleNamespace(
            profile=SimpleNamespace(vehicle_ownership_model_enabled=True)
        ),
    )

    resolved = _resolve_beam_preprocess_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2017, forecast_year=2019, current_inner_iter=0),
        workspace=workspace,
        coupler=Coupler(),
    )

    assert resolved.logical_destinations[ATLAS_VEHICLES2_OUTPUT] == (
        tmp_path / "beam-input" / ".consist-inputs" / "vehicles2_2019.csv"
    )
    assert resolved.logical_destinations[BEAM_PLANS_IN] == (
        tmp_path / "beam-input" / ".consist-inputs" / "plans_beam_in.parquet"
    )


def test_beam_preprocess_rejects_unqualified_atlas_vehicles_basename(
    tmp_path: Path,
) -> None:
    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
    )

    with pytest.raises(ValueError, match="year-qualified ATLAS vehicles2"):
        beam_steps._input_destination(
            workspace=workspace,
            key=ATLAS_VEHICLES2_OUTPUT,
            source=tmp_path / "atlas_vehicles2_output.csv",
        )


def test_beam_preprocess_requires_atlas_vehicles_when_vehicle_ownership_is_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "beam.conf"
    config_path.write_text("beam {}\n", encoding="utf-8")
    inputs = {
        BEAM_PLANS_IN: tmp_path / "plans.parquet",
        BEAM_HOUSEHOLDS_IN: tmp_path / "households.parquet",
        BEAM_PERSONS_IN: tmp_path / "persons.parquet",
    }

    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            return inputs.get(key, default)

    monkeypatch.setattr(
        beam_steps,
        "_require_primary_beam_config",
        lambda _settings, _workspace: config_path,
    )
    monkeypatch.setattr(
        beam_steps,
        "build_enabled_workflow_surface",
        lambda _settings: SimpleNamespace(
            profile=SimpleNamespace(vehicle_ownership_model_enabled=True)
        ),
    )

    resolved = _resolve_beam_preprocess_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2030, current_inner_iter=0),
        workspace=SimpleNamespace(
            get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
        ),
        coupler=Coupler(),
    )

    with pytest.raises(RuntimeError, match="atlas_vehicles2_output"):
        resolved.require_complete()


def test_beam_postprocess_resolver_stages_dynamic_closure_at_exact_destinations(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "source-events.parquet"
    _write_event_types(events_path, ["Event A", "PathTraversal"])

    class Coupler:
        def __init__(self) -> None:
            self.values = {
                "events_parquet_2030_2": events_path,
                "raw_od_skims_2030_2": tmp_path / "source-skims.omx",
                ZARR_SKIMS: tmp_path / "source-skims.zarr",
            }

        def get(self, key: str, default: object = None) -> object:
            return self.values.get(key, default)

        def keys(self):
            return self.values.keys()

    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
        get_asim_output_dir=lambda: str(tmp_path / "asim-output"),
    )
    state = SimpleNamespace(year=2030, iteration=2)
    resolved = beam_postprocess.resolve_inputs(
        settings=SimpleNamespace(activitysim=object()),
        state=state,
        workspace=workspace,
        coupler=Coupler(),
    )
    options = beam_postprocess.execution_options(
        settings=object(),
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )

    assert set(resolved.binding.inputs or {}) == {
        "events_parquet_2030_2",
        "raw_od_skims_2030_2",
        ZARR_SKIMS,
    }
    assert options.input_materialization == "requested"
    assert options.input_paths == resolved.logical_destinations
    assert options.runtime_kwargs["beam_run_dynamic_paths"] == {
        "events_parquet_2030_2": resolved.logical_destinations["events_parquet_2030_2"],
        "raw_od_skims_2030_2": resolved.logical_destinations["raw_od_skims_2030_2"],
    }
    assert (
        resolved.metadata["beam_postprocess_dynamic_paths"]
        == (options.runtime_kwargs["beam_run_dynamic_paths"])
    )
    assert resolved.logical_destinations["events_parquet_2030_2"] == (
        tmp_path
        / "beam-output"
        / ".pilates-consist-inputs"
        / "events_parquet_2030_2.parquet"
    )
    assert resolved.logical_destinations["raw_od_skims_2030_2"] == (
        tmp_path / "beam-output" / ".pilates-consist-inputs" / "raw_od_skims_2030_2.omx"
    )
    assert resolved.logical_destinations[ZARR_SKIMS] == (
        tmp_path / "asim-output" / "cache" / "skims.zarr"
    )
    assert resolved.metadata["beam_postprocess_output_paths"] == {
        "events_parquet_2030_2_type_Event_A": (
            tmp_path
            / "beam-output"
            / ".pilates-consist-outputs"
            / "beam_postprocess"
            / "events_parquet_2030_2_type_Event_A.parquet"
        ),
        "events_parquet_2030_2_type_PathTraversal": (
            tmp_path
            / "beam-output"
            / ".pilates-consist-outputs"
            / "beam_postprocess"
            / "events_parquet_2030_2_type_PathTraversal.parquet"
        ),
        "path_traversal_links_2030_2": (
            tmp_path
            / "beam-output"
            / ".pilates-consist-outputs"
            / "beam_postprocess"
            / "path_traversal_links_2030_2.parquet"
        ),
    }


def test_beam_postprocess_dynamic_keys_exclude_prior_iteration_zarr() -> None:
    keys = (
        "events_parquet_2018_1",
        "raw_od_skims_zarr_2018_0",
        "raw_od_skims_zarr_2018_1",
    )

    assert beam_steps._postprocess_dynamic_keys(
        storage_keys=keys,
        year=2018,
        iteration=1,
    ) == (
        "events_parquet_2018_1",
        "raw_od_skims_zarr_2018_1",
    )


def test_beam_postprocess_resolver_uses_completed_beam_outputs_over_coupler_history(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "events.parquet"
    _write_event_types(events_path, ["PathTraversal"])
    prior_skims = tmp_path / "prior.zarr"
    current_skims = tmp_path / "current.zarr"
    prior_skims.mkdir()
    current_skims.mkdir()

    class Coupler:
        def __init__(self) -> None:
            self.values = {
                "events_parquet_2018_0": tmp_path / "prior-events.parquet",
                "raw_od_skims_zarr_2018_0": prior_skims,
                ZARR_SKIMS: tmp_path / "asim-skims.zarr",
            }

        def get(self, key: str, default: object = None) -> object:
            return self.values.get(key, default)

        def keys(self):
            return self.values.keys()

    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
        get_asim_output_dir=lambda: str(tmp_path / "asim-output"),
    )
    resolved = beam_steps._resolve_beam_postprocess_inputs(
        settings=SimpleNamespace(activitysim=object()),
        state=SimpleNamespace(year=2018, iteration=1),
        workspace=workspace,
        coupler=Coupler(),
        beam_run_outputs={
            "events_parquet_2018_1": events_path,
            "raw_od_skims_zarr_2018_1": current_skims,
        },
    )

    assert set(resolved.binding.inputs or {}) == {
        "events_parquet_2018_1",
        "raw_od_skims_zarr_2018_1",
        ZARR_SKIMS,
    }
    assert resolved.source_by_role["events_parquet_2018_1"] == "explicit"
    assert resolved.source_by_role["raw_od_skims_zarr_2018_1"] == "explicit"


def test_beam_required_roles_do_not_resolve_from_a_namespace_view(
    tmp_path: Path,
) -> None:
    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            return default

        def view(self, namespace: str) -> object:
            assert namespace == "beam"
            return type(
                "BeamView",
                (),
                {
                    "get": lambda _self, key, default=None: (
                        tmp_path / "view-only-plans.csv"
                    )
                },
            )()

    resolved = _resolved_beam_inputs(
        step_name="beam_run",
        coupler=Coupler(),
        workspace=SimpleNamespace(
            get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input")
        ),
        required_roles=(BEAM_PLANS_IN,),
    )

    assert resolved.binding.inputs == {}
    assert resolved.source_by_role[BEAM_PLANS_IN] == "missing"
    assert BEAM_PLANS_IN not in resolved.logical_destinations


def test_beam_required_roles_prefer_the_exact_global_storage_value(
    tmp_path: Path,
) -> None:
    global_plans = tmp_path / "global-plans.csv"

    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            return {BEAM_PLANS_IN: global_plans}.get(key, default)

        def view(self, namespace: str) -> object:
            assert namespace == "beam"
            return type(
                "BeamView",
                (),
                {"get": lambda _self, key, default=None: tmp_path / "view-plans.csv"},
            )()

    resolved = _resolved_beam_inputs(
        step_name="beam_run",
        coupler=Coupler(),
        workspace=SimpleNamespace(
            get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input")
        ),
        required_roles=(BEAM_PLANS_IN,),
    )

    assert resolved.binding.inputs == {BEAM_PLANS_IN: global_plans}
    assert resolved.source_by_role[BEAM_PLANS_IN] == "coupler"


def test_beam_postprocess_does_not_admit_view_only_zarr_skims(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "source-events.parquet"
    _write_event_types(events_path, ["Event A"])

    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            return {
                "events_parquet_2030_2": events_path,
                "raw_od_skims_2030_2": tmp_path / "source-skims.omx",
            }.get(key, default)

        def keys(self):
            return ("events_parquet_2030_2", "raw_od_skims_2030_2")

        def view(self, namespace: str) -> object:
            assert namespace == "activitysim"
            return type(
                "ActivitySimView",
                (),
                {
                    "get": lambda _self, key, default=None: (
                        tmp_path / "view-only-skims.zarr"
                    )
                },
            )()

    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
        get_asim_output_dir=lambda: str(tmp_path / "asim-output"),
    )
    state = SimpleNamespace(year=2030, iteration=2)
    resolved = beam_postprocess.resolve_inputs(
        settings=SimpleNamespace(activitysim=object()),
        state=state,
        workspace=workspace,
        coupler=Coupler(),
    )
    options = beam_postprocess.execution_options(
        settings=object(),
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )

    assert ZARR_SKIMS not in (resolved.binding.inputs or {})
    assert ZARR_SKIMS not in resolved.optional_roles
    assert ZARR_SKIMS not in resolved.logical_destinations
    assert ZARR_SKIMS not in options.input_paths


def test_beam_postprocess_projector_has_fresh_hit_parity_for_closed_split_outputs(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "source-events.parquet"
    _write_event_types(events_path, ["Event A", "PathTraversal"])
    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
        get_asim_output_dir=lambda: str(tmp_path / "asim-output"),
    )
    settings = SimpleNamespace(
        activitysim=None,
        run=SimpleNamespace(models=SimpleNamespace(land_use=None)),
        write_skims_to_omx=False,
    )
    state = SimpleNamespace(year=2030, iteration=2)

    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            if key == "events_parquet_2030_2":
                return events_path
            if key == "raw_od_skims_2030_2":
                return tmp_path / "source-skims.omx"
            return default

        def keys(self):
            return ("events_parquet_2030_2", "raw_od_skims_2030_2")

    resolved = beam_postprocess.resolve_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=Coupler(),
    )
    declared = beam_postprocess.output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )
    dynamic_paths = resolved.metadata["beam_postprocess_output_paths"]
    assert set(dynamic_paths).issubset(declared)
    for path in dynamic_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    fresh = beam_postprocess.project_outputs(
        {
            key: SimpleNamespace(container_uri=str(path))
            for key, path in dynamic_paths.items()
        },
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )
    hit = beam_postprocess.project_outputs(
        {
            key: SimpleNamespace(container_uri=f"archive://prior/{path.name}")
            for key, path in dynamic_paths.items()
        },
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )

    assert (
        fresh.split_events
        == hit.split_events
        == {
            key: path
            for key, path in dynamic_paths.items()
            if key.startswith("events_parquet_")
        }
    )
    assert (
        fresh.split_event_links
        == hit.split_event_links
        == {
            key: path
            for key, path in dynamic_paths.items()
            if key.startswith("path_traversal_links_")
        }
    )


def test_beam_postprocess_resolver_refuses_uninspectable_selected_events(
    tmp_path: Path,
) -> None:
    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
        get_asim_output_dir=lambda: str(tmp_path / "asim-output"),
    )

    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            return {
                "events_parquet_2030_2": tmp_path / "not-present.parquet",
                "raw_od_skims_2030_2": tmp_path / "source-skims.omx",
            }.get(key, default)

        def keys(self):
            return ("events_parquet_2030_2", "raw_od_skims_2030_2")

    with pytest.raises(
        RuntimeError,
        match="cannot inspect selected events input 'events_parquet_2030_2'",
    ):
        beam_postprocess.resolve_inputs(
            settings=SimpleNamespace(activitysim=None),
            state=SimpleNamespace(year=2030, iteration=2),
            workspace=workspace,
            coupler=Coupler(),
        )


def test_beam_postprocess_resolver_rejects_sanitized_event_key_collision(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "source-events.parquet"
    _write_event_types(events_path, ["Event A", "Event-A"])
    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
        get_asim_output_dir=lambda: str(tmp_path / "asim-output"),
    )

    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            return {
                "events_parquet_2030_2": events_path,
                "raw_od_skims_2030_2": tmp_path / "source-skims.omx",
            }.get(key, default)

        def keys(self):
            return ("events_parquet_2030_2", "raw_od_skims_2030_2")

    with pytest.raises(
        RuntimeError, match="do not map to injective semantic output keys"
    ):
        beam_postprocess.resolve_inputs(
            settings=SimpleNamespace(activitysim=None),
            state=SimpleNamespace(year=2030, iteration=2),
            workspace=workspace,
            coupler=Coupler(),
        )


def test_native_beam_postprocess_promotes_every_closed_typed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pilates.beam.outputs import BeamPostprocessOutputs

    source_events = tmp_path / "source-events.parquet"
    source_links = tmp_path / "source-links.parquet"
    zarr_skims = tmp_path / "asim-output" / "cache" / "skims.zarr"
    source_events.write_text("events", encoding="utf-8")
    source_links.write_text("links", encoding="utf-8")
    (zarr_skims / ".zgroup").parent.mkdir(parents=True)
    (zarr_skims / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")
    output_paths = {
        "events_parquet_2030_2_type_PathTraversal": (
            tmp_path
            / "beam-output"
            / ".pilates-consist-outputs"
            / "beam_postprocess"
            / "events_parquet_2030_2_type_PathTraversal.parquet"
        ),
        "path_traversal_links_2030_2": (
            tmp_path
            / "beam-output"
            / ".pilates-consist-outputs"
            / "beam_postprocess"
            / "path_traversal_links_2030_2.parquet"
        ),
    }

    class FakePostprocessor:
        def __init__(self, *_args: object) -> None:
            pass

        @staticmethod
        def expected_outputs(*_args: object) -> dict[str, Path]:
            return {ZARR_SKIMS: zarr_skims}

        def postprocess(
            self, *_args: object, **_kwargs: object
        ) -> BeamPostprocessOutputs:
            return BeamPostprocessOutputs(
                zarr_skims=zarr_skims,
                split_events={
                    "events_parquet_2030_2_type_PathTraversal": source_events
                },
                split_event_links={"path_traversal_links_2030_2": source_links},
            )

    monkeypatch.setattr(
        "pilates.beam.postprocessor.BeamPostprocessor", FakePostprocessor
    )
    archive_keys: list[str] = []
    monkeypatch.setattr(
        beam_steps,
        "enqueue_archive_copy",
        lambda *, key, path, workspace: archive_keys.append(key),
    )
    settings = SimpleNamespace(
        activitysim=None,
        run=SimpleNamespace(models=SimpleNamespace(land_use=None)),
        write_skims_to_omx=False,
    )
    workspace = SimpleNamespace(
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
    )

    _native_beam_postprocess(
        beam_run_dynamic_paths={},
        beam_postprocess_output_paths=output_paths,
        settings=settings,
        state=SimpleNamespace(year=2030, iteration=2),
        workspace=workspace,
        _consist_ctx=object(),
    )

    assert {
        key: path.read_text(encoding="utf-8") for key, path in output_paths.items()
    } == {
        "events_parquet_2030_2_type_PathTraversal": "events",
        "path_traversal_links_2030_2": "links",
    }
    assert archive_keys == [
        "events_parquet_2030_2_type_PathTraversal",
        "path_traversal_links_2030_2",
    ]


def test_beam_preprocess_projector_is_pure_and_validates_persisted_outputs(
    tmp_path: Path,
) -> None:
    mutable_data_dir = tmp_path / "beam-input"
    mutable_data_dir.mkdir()
    workspace = SimpleNamespace(get_beam_mutable_data_dir=lambda: str(mutable_data_dir))
    declared = beam_preprocess.output_paths(
        settings=object(), state=object(), workspace=workspace, resolved_inputs=None
    )
    paths = {
        key: declared[key]
        for key in (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN)
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    projected = beam_preprocess.project_outputs(
        {
            key: SimpleNamespace(
                container_uri=f"archive://historical/{path.name}",
                path=tmp_path / "historical" / path.name,
            )
            for key, path in paths.items()
        },
        settings=object(),
        state=object(),
        workspace=workspace,
        resolved_inputs=_empty_resolved_inputs(),
    )

    assert projected.prepared_inputs == paths


def test_beam_preprocess_projector_has_fresh_hit_path_parity(
    tmp_path: Path,
) -> None:
    mutable_data_dir = tmp_path / "beam-input"
    mutable_data_dir.mkdir()
    workspace = SimpleNamespace(get_beam_mutable_data_dir=lambda: str(mutable_data_dir))
    declared = beam_preprocess.output_paths(
        settings=object(), state=object(), workspace=workspace, resolved_inputs=None
    )
    paths = {
        key: declared[key]
        for key in (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN)
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    fresh = beam_preprocess.project_outputs(
        {key: SimpleNamespace(container_uri=str(path)) for key, path in paths.items()},
        settings=object(),
        state=object(),
        workspace=workspace,
        resolved_inputs=_empty_resolved_inputs(),
    )
    hit = beam_preprocess.project_outputs(
        {
            key: SimpleNamespace(container_uri=f"archive://prior/{path.name}")
            for key, path in paths.items()
        },
        settings=object(),
        state=object(),
        workspace=workspace,
        resolved_inputs=_empty_resolved_inputs(),
    )

    assert fresh.prepared_inputs == hit.prepared_inputs == paths


def test_beam_run_declares_closed_individually_keyed_handoff_outputs(
    tmp_path: Path,
) -> None:
    workspace = SimpleNamespace(
        get_beam_output_dir=lambda: str(tmp_path / "beam-output")
    )
    paths = beam_run.output_paths(
        settings=_beam_settings(),
        state=SimpleNamespace(year=2030, iteration=2),
        workspace=workspace,
        resolved_inputs=None,
    )

    assert "beam_output_dir" not in paths
    assert set(paths) == {
        "linkstats",
        "beam_plans_out",
        "events_parquet_2030_2",
    }
    assert len(set(paths.values())) == len(paths)
    assert all(
        ".pilates-consist-outputs/beam_run" in str(path) for path in paths.values()
    )


def test_native_beam_run_logs_raw_zarr_directly_without_materializing_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "beam.conf"
    config.write_text("beam {}\n", encoding="utf-8")
    inputs = {}
    for key in (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN):
        path = tmp_path / f"{key}.csv"
        path.write_text(f"{key}\n", encoding="utf-8")
        inputs[key] = path
    raw_zarr = tmp_path / "beam-output" / "ITERS" / "it.1" / "1.skims.zarr"
    (raw_zarr / ".zgroup").parent.mkdir(parents=True)
    (raw_zarr / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")
    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
    )
    logged: list[tuple[Path, str, str]] = []
    materialized: dict[str, object] = {}

    class Runner:
        def __init__(self, *_args: object) -> None:
            pass

        def run(self, *_args: object, **_kwargs: object) -> BeamRunOutputs:
            return BeamRunOutputs(
                beam_output_dir=Path(workspace.get_beam_output_dir()),
                raw_outputs={"raw_od_skims_zarr_2030_2": raw_zarr},
            )

    context = SimpleNamespace(
        canonicalization=object(),
        log_output=lambda path, key, artifact_kind: logged.append(
            (Path(path), key, artifact_kind)
        ),
    )
    monkeypatch.setattr(beam_steps, "BeamRunner", Runner)
    monkeypatch.setattr(
        beam_steps, "validate_staged_linkstats_reference", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        beam_steps, "validate_r5_execution_reference", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        beam_steps, "_validate_native_outputs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        beam_steps,
        "_materialize_native_outputs",
        lambda **kwargs: materialized.update(kwargs),
    )

    _native_beam_run(
        config,
        inputs[BEAM_PLANS_IN],
        inputs[BEAM_HOUSEHOLDS_IN],
        inputs[BEAM_PERSONS_IN],
        settings=_beam_settings(),
        state=SimpleNamespace(year=2030, iteration=2),
        workspace=workspace,
        beam_launch_config=BeamLaunchConfig(root=tmp_path, primary_config=config),
        _consist_ctx=context,
    )

    assert logged == [(raw_zarr, "raw_od_skims_zarr_2030_2", "directory")]
    assert "raw_od_skims_zarr_2030_2" not in materialized["source_paths"]
    assert "raw_od_skims_zarr_2030_2" not in materialized["declared_outputs"]


def test_project_beam_run_outputs_uses_direct_logged_zarr_path(tmp_path: Path) -> None:
    workspace = SimpleNamespace(
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
    )
    zarr = tmp_path / "beam-output" / "ITERS" / "it.1" / "1.skims.zarr"
    (zarr / ".zgroup").parent.mkdir(parents=True)
    (zarr / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")

    projected = _project_beam_run_outputs(
        {"raw_od_skims_zarr_2030_2": SimpleNamespace(path=zarr)},
        settings=_beam_settings(),
        state=SimpleNamespace(year=2030, iteration=2),
        workspace=workspace,
        resolved_inputs=_empty_resolved_inputs(),
    )

    assert projected.raw_outputs == {"raw_od_skims_zarr_2030_2": zarr}


def test_beam_postprocess_dynamic_closure_excludes_non_numeric_suffixes(
    tmp_path: Path,
) -> None:
    selected_events = tmp_path / "source-events_parquet_2030_2.parquet"
    _write_event_types(selected_events, ["PathTraversal"])

    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            if key == "events_parquet_2030_2":
                return selected_events
            return tmp_path / f"source-{key}"

        def keys(self):
            return (
                "events_parquet_2030_2",
                "events_parquet_2030_2_retry",
                "raw_od_skims_2030_2",
            )

    resolved = beam_postprocess.resolve_inputs(
        settings=SimpleNamespace(activitysim=None),
        state=SimpleNamespace(year=2030, iteration=2),
        workspace=SimpleNamespace(
            get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
            get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
            get_asim_output_dir=lambda: str(tmp_path / "asim-output"),
        ),
        coupler=Coupler(),
    )
    closure = resolved.metadata["beam_postprocess_dynamic_paths"]

    assert set(closure) == {
        "events_parquet_2030_2",
        "raw_od_skims_2030_2",
    }
    assert len(set(closure.values())) == len(closure)
    assert all(key in str(path) for key, path in closure.items())


def test_beam_preprocess_projector_rejects_stale_mount_when_current_destination_missing(
    tmp_path: Path,
) -> None:
    source_mount = tmp_path / "source-mount" / "plans.csv"
    source_mount.parent.mkdir()
    source_mount.write_text("historical plans", encoding="utf-8")
    current_dir = tmp_path / "current"
    current_households = current_dir / "households.csv"
    current_persons = current_dir / "persons.csv"
    current_dir.mkdir()
    (tmp_path / "beam-input").mkdir()
    current_households.write_text("current households", encoding="utf-8")
    current_persons.write_text("current persons", encoding="utf-8")

    workspace = SimpleNamespace(
        full_path=str(tmp_path),
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
    )
    outputs = {
        BEAM_PLANS_IN: SimpleNamespace(
            container_uri="workspace://current/plans.csv",
            path=source_mount,
        ),
        BEAM_HOUSEHOLDS_IN: SimpleNamespace(
            container_uri="workspace://current/households.csv",
            path=current_households,
        ),
        BEAM_PERSONS_IN: SimpleNamespace(
            container_uri="workspace://current/persons.csv",
            path=current_persons,
        ),
    }

    with pytest.raises(
        RuntimeError,
        match="beam_preprocess output 'plans_beam_in' is missing at declared destination",
    ):
        beam_preprocess.project_outputs(
            outputs,
            settings=object(),
            state=object(),
            workspace=workspace,
            resolved_inputs=_empty_resolved_inputs(),
        )
