from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from consist import (
    BindingResult,
    ExecutionOptions,
    ResolvedBinding,
    Tracker,
    resolve_step_contract,
)

from pilates.activitysim.outputs import (
    ASIM_REQUIRED_RUN_OUTPUT_KEYS,
    configured_asim_output_tables,
)
from pilates.activitysim.postprocessor import _activitysim_iteration_output_paths
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_BASE_H5,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.step_execution import execute_step
from pilates.workflows.steps import activitysim


def _tracked_artifacts(tmp_path: Path, *keys: str) -> dict[str, object]:
    """Log local artifacts suitable for a V1 strict-binding resolver test."""

    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        hashing_strategy="full",
    )
    artifacts: dict[str, object] = {}
    with tracker.start_run("seed_activitysim_inputs", "test"):
        for key in keys:
            source = tmp_path / "sources" / key
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"{key}\n", encoding="utf-8")
            artifacts[key] = tracker.log_artifact(source, key=key, direction="input")
    return artifacts


def _activitysim_run_resolution(*, produces_zarr: bool) -> ResolvedStepInputs:
    """Build the immutable native skim decision for direct projector tests."""

    return ResolvedStepInputs(
        step_name="activitysim_run",
        binding=BindingResult(),
        metadata={
            "activitysim_skim_mode": "omx" if produces_zarr else "zarr",
            "activitysim_produces_zarr": produces_zarr,
        },
    )


def _activitysim_test_settings() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(region="test"),
        activitysim=SimpleNamespace(
            output_tables={
                "tables": [
                    "accessibility",
                    "beam_plans",
                    "disaggregate_accessibility",
                    "households",
                    "joint_tour_participants",
                    "land_use",
                    "non_mandatory_tour_destination_accessibility",
                    "persons",
                    "tours",
                    "trips",
                ]
            }
        ),
    )


def test_activitysim_output_contract_uses_configured_source_table_names(
    tmp_path: Path,
) -> None:
    """Only configured tables are declared, retaining the real ``plans`` path."""

    settings = SimpleNamespace(
        activitysim=SimpleNamespace(
            output_tables={
                "tables": [
                    "checkpoints",
                    "households",
                    "persons",
                    "tours",
                    "trips",
                    "plans",
                ]
            }
        )
    )
    state = SimpleNamespace(year=2025, forecast_year=2025, iteration=0)
    workspace = SimpleNamespace(
        get_asim_output_dir=lambda: str(tmp_path / "activitysim" / "output")
    )

    configured = configured_asim_output_tables(settings)
    assert configured == {
        "households_asim_out": "households",
        "persons_asim_out": "persons",
        "tours_asim_out": "tours",
        "trips_asim_out": "trips",
        "beam_plans_asim_out": "plans",
    }

    run_outputs = activitysim.activitysim_run_output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        produces_zarr=False,
    )
    postprocess_outputs = _activitysim_iteration_output_paths(
        settings,
        state,
        workspace,
    )

    assert set(run_outputs) == set(configured)
    assert set(postprocess_outputs) == set(configured)
    assert run_outputs["beam_plans_asim_out"].path.endswith(
        "final_pipeline/plans/final.parquet"
    )
    assert str(postprocess_outputs["beam_plans_asim_out"]).endswith(
        "year-2025-iteration-0/beam_plans.parquet"
    )
    assert "accessibility_asim_out" not in run_outputs


def test_activitysim_definitions_resolve_native_consist_contracts(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "pilates.workflows.step_consist_meta.build_step_consist_kwargs",
        lambda model, settings, workspace_path=None: {
            "config": {"model": model},
            "identity_inputs": [],
        },
    )
    settings = SimpleNamespace(run=SimpleNamespace(region="test"))
    state = SimpleNamespace(year=2025, forecast_year=2025, iteration=0)
    workspace = SimpleNamespace(full_path=str(tmp_path))

    for definition in (
        activitysim.activitysim_preprocess,
        activitysim.activitysim_run,
        activitysim.activitysim_postprocess,
    ):
        contract = resolve_step_contract(
            definition.function,
            year=2025,
            iteration=0,
            phase="activity_demand",
            stage="supply_demand",
            runtime_kwargs={
                "settings": settings,
                "state": state,
                "workspace": workspace,
            },
        )

        assert contract.model == "activitysim"
        assert contract.name.startswith(f"{definition.name}__y2025__i0")
        assert contract.input_binding == "paths"


def test_activitysim_preprocess_resolver_keeps_one_semantic_binding(
    monkeypatch, tmp_path: Path
) -> None:
    selected_artifacts = _tracked_artifacts(
        tmp_path, USIM_POPULATION_SOURCE_H5, "final_skims_omx"
    )
    captured: dict[str, object] = {}

    def resolve_roles(**kwargs: object) -> ResolvedStepInputs:
        captured.update(kwargs)
        return ResolvedStepInputs(
            step_name="activitysim_preprocess",
            binding=BindingResult(
                inputs={
                    USIM_POPULATION_SOURCE_H5: selected_artifacts[
                        USIM_POPULATION_SOURCE_H5
                    ],
                    "final_skims_omx": selected_artifacts["final_skims_omx"],
                }
            ),
            required_roles=(USIM_POPULATION_SOURCE_H5,),
            optional_roles=("final_skims_omx",),
            source_by_role={
                USIM_POPULATION_SOURCE_H5: "explicit",
                "final_skims_omx": "explicit",
            },
            logical_destinations=kwargs["logical_destinations"],
            metadata={"candidate_paths_by_semantic_key": {}},
        )

    monkeypatch.setattr(activitysim, "resolve_artifact_roles", resolve_roles)
    workspace = SimpleNamespace(
        get_asim_mutable_configs_dir=lambda: str(tmp_path / "configs"),
        get_usim_mutable_data_dir=lambda: str(tmp_path / "usim"),
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam"),
    )

    resolved = activitysim.activitysim_preprocess.resolve_inputs(
        settings=SimpleNamespace(run=SimpleNamespace(region="test")),
        state=SimpleNamespace(year=2025),
        workspace=workspace,
        coupler=object(),
    )

    assert isinstance(resolved.binding, BindingResult)
    assert dict(resolved.binding.inputs or {}) == {
        USIM_POPULATION_SOURCE_H5: selected_artifacts[USIM_POPULATION_SOURCE_H5],
        "final_skims_omx": selected_artifacts["final_skims_omx"],
    }
    assert resolved.required_roles == (USIM_POPULATION_SOURCE_H5,)
    assert resolved.source_by_role[USIM_POPULATION_SOURCE_H5] == "explicit"
    assert set(resolved.logical_destinations) == {
        USIM_POPULATION_SOURCE_H5,
        "final_skims_omx",
    }
    rules = {rule.semantic_key: rule for rule in captured["artifact_rules"]}
    assert rules[USIM_POPULATION_SOURCE_H5].allow_fallback is True
    assert rules[USIM_POPULATION_SOURCE_H5].fallback_provider == (
        "activitysim_population_source"
    )
    assert rules["final_skims_omx"].required is False


def test_activitysim_postprocess_resolver_omits_current_alias_of_population_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    population_source = _tracked_artifacts(tmp_path, USIM_POPULATION_SOURCE_H5)[
        USIM_POPULATION_SOURCE_H5
    ]
    base_resolution = ResolvedStepInputs(
        step_name="activitysim_postprocess",
        binding=BindingResult(
            inputs={
                USIM_POPULATION_SOURCE_H5: population_source,
                USIM_DATASTORE_CURRENT_H5: population_source,
            }
        ),
        optional_roles=(
            USIM_POPULATION_SOURCE_H5,
            USIM_DATASTORE_CURRENT_H5,
        ),
        source_by_role={
            USIM_POPULATION_SOURCE_H5: "coupler",
            USIM_DATASTORE_CURRENT_H5: "coupler",
        },
        logical_destinations={
            USIM_POPULATION_SOURCE_H5: tmp_path / "population-source.h5",
            USIM_DATASTORE_CURRENT_H5: tmp_path / "current.h5",
        },
    )
    monkeypatch.setattr(
        activitysim,
        "_native_activitysim_resolved_inputs",
        lambda **_kwargs: base_resolution,
    )
    monkeypatch.setattr(
        activitysim.ActivitysimPostprocessor,
        "declared_expected_inputs",
        staticmethod(lambda *_args: {}),
    )

    resolved = activitysim._activitysim_postprocess_resolver(
        settings=_activitysim_test_settings(),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(),
        coupler=object(),
    )

    assert dict(resolved.binding.inputs or {}) == {
        USIM_POPULATION_SOURCE_H5: population_source,
    }
    assert resolved.selected_roles() == (USIM_POPULATION_SOURCE_H5,)


def test_activitysim_postprocess_resolver_requires_only_configured_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disabled ActivitySim table cannot become a postprocess input requirement."""

    captured: dict[str, tuple[str, ...]] = {}

    def _resolve_native(**kwargs: object) -> ResolvedStepInputs:
        roles = kwargs["required_roles"]
        assert isinstance(roles, tuple)
        captured["required_roles"] = roles
        return ResolvedStepInputs(
            step_name="activitysim_postprocess",
            binding=BindingResult(),
        )

    monkeypatch.setattr(
        activitysim, "_native_activitysim_resolved_inputs", _resolve_native
    )
    monkeypatch.setattr(
        activitysim.ActivitysimPostprocessor,
        "declared_expected_inputs",
        staticmethod(lambda *_args: {}),
    )
    settings = SimpleNamespace(
        activitysim=SimpleNamespace(
            output_tables={"tables": ["households", "persons", "plans"]}
        )
    )

    activitysim._activitysim_postprocess_resolver(
        settings=settings,
        state=SimpleNamespace(),
        workspace=SimpleNamespace(),
        coupler=object(),
    )

    assert captured["required_roles"] == (
        ASIM_HOUSEHOLDS_IN,
        ASIM_PERSONS_IN,
        ASIM_LAND_USE_IN,
        ASIM_OMX_SKIMS,
        ZARR_SKIMS,
        "households_asim_out",
        "persons_asim_out",
        "beam_plans_asim_out",
    )


def test_activitysim_postprocess_resolver_omits_current_artifact_with_population_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        hashing_strategy="full",
    )
    source_paths = [tmp_path / "population-a.h5", tmp_path / "population-b.h5"]
    for index, path in enumerate(source_paths):
        path.write_text(f"population {index}\n", encoding="utf-8")
    with tracker.start_run("seed_duplicate_population_key", "test"):
        current_artifact = tracker.log_artifact(
            source_paths[0], key=USIM_POPULATION_SOURCE_H5, direction="input"
        )
        population_artifact = tracker.log_artifact(
            source_paths[1], key=USIM_POPULATION_SOURCE_H5, direction="input"
        )
    base_resolution = ResolvedStepInputs(
        step_name="activitysim_postprocess",
        binding=BindingResult(
            inputs={
                USIM_POPULATION_SOURCE_H5: population_artifact,
                USIM_DATASTORE_CURRENT_H5: current_artifact,
            }
        ),
        optional_roles=(
            USIM_POPULATION_SOURCE_H5,
            USIM_DATASTORE_CURRENT_H5,
        ),
        logical_destinations={
            USIM_POPULATION_SOURCE_H5: tmp_path / "population-source.h5",
            USIM_DATASTORE_CURRENT_H5: tmp_path / "current.h5",
        },
    )
    monkeypatch.setattr(
        activitysim,
        "_native_activitysim_resolved_inputs",
        lambda **_kwargs: base_resolution,
    )
    monkeypatch.setattr(
        activitysim.ActivitysimPostprocessor,
        "declared_expected_inputs",
        staticmethod(lambda *_args: {}),
    )

    resolved = activitysim._activitysim_postprocess_resolver(
        settings=_activitysim_test_settings(),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(),
        coupler=object(),
    )

    assert dict(resolved.binding.inputs or {}) == {
        USIM_POPULATION_SOURCE_H5: population_artifact,
    }


def test_activitysim_postprocess_resolver_freezes_tracked_h5_aliases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One tracked H5 may safely satisfy all three named postprocess parameters."""

    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        hashing_strategy="full",
    )
    source = tmp_path / "sources" / "model_data.h5"
    source.parent.mkdir(parents=True)
    source.write_text("tracked h5\n", encoding="utf-8")
    with tracker.start_run("seed_postprocess_h5", "test"):
        artifact = tracker.log_artifact(
            source,
            key=USIM_DATASTORE_BASE_H5,
            direction="input",
        )

    base_resolution = ResolvedStepInputs(
        step_name="activitysim_postprocess",
        binding=BindingResult(
            inputs={
                USIM_POPULATION_SOURCE_H5: artifact,
                USIM_DATASTORE_CURRENT_H5: artifact,
                USIM_DATASTORE_BASE_H5: artifact,
            }
        ),
        optional_roles=(
            USIM_POPULATION_SOURCE_H5,
            USIM_DATASTORE_CURRENT_H5,
            USIM_DATASTORE_BASE_H5,
        ),
        source_by_role={
            USIM_POPULATION_SOURCE_H5: "coupler",
            USIM_DATASTORE_CURRENT_H5: "coupler",
            USIM_DATASTORE_BASE_H5: "coupler",
        },
        logical_destinations={
            USIM_POPULATION_SOURCE_H5: tmp_path / "population-source.h5",
            USIM_DATASTORE_CURRENT_H5: tmp_path / "current.h5",
            USIM_DATASTORE_BASE_H5: tmp_path / "base.h5",
        },
    )
    monkeypatch.setattr(
        activitysim,
        "_native_activitysim_resolved_inputs",
        lambda **_kwargs: base_resolution,
    )
    monkeypatch.setattr(
        activitysim.ActivitysimPostprocessor,
        "declared_expected_inputs",
        staticmethod(lambda *_args: {}),
    )

    monkeypatch.setattr(
        activitysim,
        "configured_asim_output_keys",
        lambda _settings: ASIM_REQUIRED_RUN_OUTPUT_KEYS,
    )
    settings = SimpleNamespace(run=SimpleNamespace(region="test"))
    state = SimpleNamespace(year=2025, forecast_year=2025, iteration=0)
    workspace = SimpleNamespace(full_path=str(tmp_path))
    with tracker.scenario("activitysim-postprocess") as scenario:
        identity = scenario.resolve_step_identity(
            activitysim._activitysim_postprocess_callable,
            year=2025,
            iteration=0,
            phase="postprocess",
            stage="supply_demand",
            execution_options=ExecutionOptions(
                input_binding="paths",
                runtime_kwargs={
                    "settings": settings,
                    "state": state,
                    "workspace": workspace,
                },
            ),
        )
        resolved = activitysim._activitysim_postprocess_resolver(
            settings=settings,
            state=state,
            workspace=workspace,
            coupler=object(),
            step_identity=identity,
        )

    assert isinstance(resolved.binding, ResolvedBinding)
    assert set(resolved.binding.inputs) == {
        USIM_POPULATION_SOURCE_H5,
        USIM_DATASTORE_CURRENT_H5,
        USIM_DATASTORE_BASE_H5,
    }
    assert {
        input.artifact.artifact_id for input in resolved.binding.inputs.values()
    } == {artifact.id}


def test_activitysim_postprocess_executes_h5_aliases_from_strict_snapshots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The native postprocessor receives distinct strict snapshots for H5 aliases."""

    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        hashing_strategy="full",
    )
    required_roles = (
        ASIM_HOUSEHOLDS_IN,
        ASIM_PERSONS_IN,
        ASIM_LAND_USE_IN,
        ASIM_OMX_SKIMS,
        ZARR_SKIMS,
        *ASIM_REQUIRED_RUN_OUTPUT_KEYS,
    )
    with tracker.start_run("seed_postprocess_inputs", "test"):
        h5_source = tmp_path / "sources" / "model_data.h5"
        h5_source.parent.mkdir(parents=True)
        h5_source.write_text("tracked h5\n", encoding="utf-8")
        h5_artifact = tracker.log_artifact(
            h5_source,
            key=USIM_DATASTORE_BASE_H5,
            direction="input",
        )
        inputs = {
            USIM_POPULATION_SOURCE_H5: h5_artifact,
            USIM_DATASTORE_CURRENT_H5: h5_artifact,
            USIM_DATASTORE_BASE_H5: h5_artifact,
        }
        for role in required_roles:
            source = tmp_path / "sources" / role
            source.write_text(f"{role}\n", encoding="utf-8")
            inputs[role] = tracker.log_artifact(source, key=role, direction="input")

    base_resolution = ResolvedStepInputs(
        step_name="activitysim_postprocess",
        binding=BindingResult(inputs=inputs),
        required_roles=required_roles,
        optional_roles=(
            USIM_POPULATION_SOURCE_H5,
            USIM_DATASTORE_CURRENT_H5,
            USIM_DATASTORE_BASE_H5,
        ),
        source_by_role={role: "coupler" for role in inputs},
        logical_destinations={
            role: tmp_path / "logical-inputs" / role for role in inputs
        },
    )
    monkeypatch.setattr(
        activitysim,
        "_native_activitysim_resolved_inputs",
        lambda **_kwargs: base_resolution,
    )
    monkeypatch.setattr(
        activitysim.ActivitysimPostprocessor,
        "declared_expected_inputs",
        staticmethod(lambda *_args: {}),
    )

    output_paths = {
        key: tmp_path / "outputs" / key
        for key in (USIM_DATASTORE_CURRENT_H5, *ASIM_REQUIRED_RUN_OUTPUT_KEYS)
    }
    monkeypatch.setattr(
        activitysim,
        "activitysim_postprocess_output_paths",
        lambda **_kwargs: output_paths,
    )
    received: dict[str, str] = {}

    class _Postprocessor:
        def postprocess(self, *_args: object, **kwargs: object) -> None:
            received.update(
                {
                    key: str(kwargs[key])
                    for key in (
                        "population_source_h5_path",
                        "current_input_h5_path",
                    )
                }
            )
            for path in output_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("persisted output\n", encoding="utf-8")

    monkeypatch.setattr(
        activitysim.ModelFactory,
        "get_postprocessor",
        lambda _self, *_args: _Postprocessor(),
    )
    monkeypatch.setattr(
        "pilates.workflows.step_consist_meta.build_step_consist_kwargs",
        lambda **_kwargs: {
            "config": {"model": "activitysim"},
            "identity_inputs": [],
        },
    )
    monkeypatch.setattr(
        activitysim,
        "configured_asim_output_keys",
        lambda _settings: ASIM_REQUIRED_RUN_OUTPUT_KEYS,
    )
    settings = SimpleNamespace(run=SimpleNamespace(region="test"))
    state = SimpleNamespace(year=2025, forecast_year=2025, iteration=0)
    asim_output_dir = tmp_path / "asim-output"
    asim_output_dir.mkdir()
    workspace = SimpleNamespace(
        full_path=str(tmp_path),
        get_asim_output_dir=lambda: str(asim_output_dir),
        get_asim_mutable_data_dir=lambda: str(tmp_path / "asim-inputs"),
    )

    with tracker.scenario("activitysim-postprocess-aliases") as scenario:
        result, projected = execute_step(
            scenario=scenario,
            definition=replace(
                activitysim.activitysim_postprocess,
                output_paths=lambda **_kwargs: output_paths,
            ),
            settings=settings,
            state=state,
            workspace=workspace,
            stage="supply_demand",
            year=2025,
            iteration=0,
            phase="postprocess",
        )

    snapshot_root = (
        tracker.run_dir / ".resolved-bindings" / result.run.id / "inputs"
    ).resolve()
    population_snapshot = snapshot_root / USIM_POPULATION_SOURCE_H5
    current_snapshot = snapshot_root / USIM_DATASTORE_CURRENT_H5
    base_snapshot = snapshot_root / USIM_DATASTORE_BASE_H5
    assert Path(received["population_source_h5_path"]) == population_snapshot
    assert Path(received["current_input_h5_path"]) == current_snapshot
    assert len({population_snapshot, current_snapshot, base_snapshot}) == 3
    assert all(
        path.read_text(encoding="utf-8") == "tracked h5\n"
        for path in (
            population_snapshot,
            current_snapshot,
            base_snapshot,
        )
    )
    assert projected.usim_datastore_h5 == output_paths[USIM_DATASTORE_CURRENT_H5]
    assert result.outputs[USIM_DATASTORE_CURRENT_H5].container_uri == str(
        output_paths[USIM_DATASTORE_CURRENT_H5]
    )


def test_activitysim_postprocess_uses_population_source_when_current_alias_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Postprocessor:
        def postprocess(self, *_args: object, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        activitysim.ModelFactory,
        "get_postprocessor",
        lambda _self, *_args: _Postprocessor(),
    )
    population_source = Path("/inputs/population-source.h5")
    base_datastore = Path("/inputs/base.h5")
    output_paths = {
        key: Path(f"/outputs/{key}") for key in ASIM_REQUIRED_RUN_OUTPUT_KEYS
    }

    activitysim._activitysim_postprocess_callable(
        households_asim_in=Path("/inputs/households.csv"),
        persons_asim_in=Path("/inputs/persons.csv"),
        land_use_asim_in=Path("/inputs/land-use.csv"),
        omx_skims=Path("/inputs/skims.omx"),
        zarr_skims=Path("/inputs/skims.zarr"),
        usim_population_source_h5=population_source,
        usim_datastore_h5=None,
        usim_datastore_base_h5=base_datastore,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(get_asim_output_dir=lambda: "/outputs"),
        **output_paths,
    )

    assert captured["population_source_h5_path"] == str(population_source)
    assert captured["current_input_h5_path"] == str(population_source)


def test_activitysim_projectors_validate_persisted_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    preprocess_paths = {
        ASIM_LAND_USE_IN: tmp_path / "land_use.csv",
        ASIM_HOUSEHOLDS_IN: tmp_path / "households.csv",
        ASIM_PERSONS_IN: tmp_path / "persons.csv",
    }
    run_paths = {
        key: tmp_path / f"{key}.parquet" for key in ASIM_REQUIRED_RUN_OUTPUT_KEYS
    }
    postprocess_paths = {
        key: tmp_path / f"processed-{key}.parquet"
        for key in ASIM_REQUIRED_RUN_OUTPUT_KEYS
    }
    for path in (
        *preprocess_paths.values(),
        *run_paths.values(),
        *postprocess_paths.values(),
    ):
        path.write_text("output", encoding="utf-8")

    workspace = SimpleNamespace(
        full_path=str(tmp_path),
        get_asim_mutable_data_dir=lambda: str(tmp_path),
        get_asim_output_dir=lambda: str(tmp_path),
    )
    artifact = lambda path: SimpleNamespace(container_uri=str(path))

    expected_preprocess_outputs = dict(preprocess_paths)
    expected_run_outputs = dict(run_paths)
    expected_postprocess_outputs = dict(postprocess_paths)
    monkeypatch.setattr(
        activitysim,
        "activitysim_preprocess_output_paths",
        lambda **_kwargs: expected_preprocess_outputs,
    )
    monkeypatch.setattr(
        activitysim,
        "activitysim_run_output_paths",
        lambda **_kwargs: expected_run_outputs,
    )
    monkeypatch.setattr(
        activitysim,
        "activitysim_postprocess_output_paths",
        lambda **_kwargs: expected_postprocess_outputs,
    )

    preprocess = activitysim.activitysim_preprocess.project_outputs(
        {key: artifact(path) for key, path in preprocess_paths.items()},
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
    )
    run = activitysim.activitysim_run.project_outputs(
        {key: artifact(path) for key, path in run_paths.items()},
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        resolved_inputs=_activitysim_run_resolution(produces_zarr=False),
    )
    postprocess = activitysim.activitysim_postprocess.project_outputs(
        {key: artifact(path) for key, path in postprocess_paths.items()},
        settings=SimpleNamespace(land_use_enabled=False),
        state=SimpleNamespace(),
        workspace=workspace,
    )

    assert preprocess.households_table == preprocess_paths[ASIM_HOUSEHOLDS_IN]
    assert set(run.raw_outputs) == set(ASIM_REQUIRED_RUN_OUTPUT_KEYS)
    assert set(postprocess.processed_outputs) == set(ASIM_REQUIRED_RUN_OUTPUT_KEYS)


@pytest.mark.parametrize(
    ("published_roles", "expected_skim_role", "expected_mode", "produces_zarr"),
    [
        (
            {activitysim.ZARR_SKIMS: "coupler", activitysim.ASIM_OMX_SKIMS: "coupler"},
            activitysim.ZARR_SKIMS,
            "zarr",
            False,
        ),
        (
            {activitysim.ASIM_OMX_SKIMS: "coupler"},
            activitysim.ASIM_OMX_SKIMS,
            "omx",
            True,
        ),
    ],
)
def test_activitysim_run_resolver_selects_exactly_one_published_skim_source(
    monkeypatch,
    tmp_path: Path,
    published_roles: dict[str, str],
    expected_skim_role: str,
    expected_mode: str,
    produces_zarr: bool,
) -> None:
    all_roles = (
        activitysim.ASIM_LAND_USE_IN,
        activitysim.ASIM_HOUSEHOLDS_IN,
        activitysim.ASIM_PERSONS_IN,
        activitysim.ZARR_SKIMS,
        activitysim.ASIM_OMX_SKIMS,
    )
    selected_keys = tuple(
        key
        for key in all_roles
        if key in published_roles or key not in activitysim._ACTIVITYSIM_RUN_SKIM_ROLES
    )
    selected_artifacts = _tracked_artifacts(tmp_path, *selected_keys)
    base_resolution = ResolvedStepInputs(
        step_name="activitysim_run",
        binding=BindingResult(inputs=selected_artifacts),
        required_roles=activitysim._ACTIVITYSIM_RUN_REQUIRED_ROLES,
        optional_roles=activitysim._ACTIVITYSIM_RUN_SKIM_ROLES,
        source_by_role={
            key: published_roles.get(
                key,
                "coupler"
                if key not in activitysim._ACTIVITYSIM_RUN_SKIM_ROLES
                else "missing",
            )
            for key in all_roles
        },
        selected_key_by_role={key: key for key in selected_keys},
        logical_destinations={key: Path(f"/native/{key}") for key in all_roles},
    )
    monkeypatch.setattr(
        activitysim,
        "_native_activitysim_resolved_inputs",
        lambda **_kwargs: base_resolution,
    )
    monkeypatch.setattr(
        activitysim.ActivitysimRunner,
        "declared_expected_inputs",
        staticmethod(lambda *_args: {key: Path(f"/native/{key}") for key in all_roles}),
    )
    coupler = SimpleNamespace(
        get=lambda key, default=None: pytest.fail(
            f"activitysim_run reread frozen native role {key!r}"
        )
    )
    resolved = activitysim._activitysim_run_resolver(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(),
        coupler=coupler,
    )

    assert resolved.selected_roles() == (
        activitysim.ASIM_LAND_USE_IN,
        activitysim.ASIM_HOUSEHOLDS_IN,
        activitysim.ASIM_PERSONS_IN,
        expected_skim_role,
    )
    assert resolved.metadata["activitysim_skim_mode"] == expected_mode
    assert resolved.metadata["activitysim_produces_zarr"] is produces_zarr
    assert set(resolved.logical_destinations) == set(resolved.selected_roles())
    assert isinstance(resolved.binding, BindingResult)
    assert dict(resolved.binding.inputs or {}) == {
        key: selected_artifacts[key] for key in resolved.selected_roles()
    }


def test_activitysim_run_resolver_rejects_when_no_published_skim_source(
    monkeypatch,
) -> None:
    base_resolution = ResolvedStepInputs(
        step_name="activitysim_run",
        binding=BindingResult(input_keys=activitysim._ACTIVITYSIM_RUN_REQUIRED_ROLES),
        required_roles=activitysim._ACTIVITYSIM_RUN_REQUIRED_ROLES,
        optional_roles=activitysim._ACTIVITYSIM_RUN_SKIM_ROLES,
        source_by_role={
            **{key: "coupler" for key in activitysim._ACTIVITYSIM_RUN_REQUIRED_ROLES},
            activitysim.ZARR_SKIMS: "missing",
            activitysim.ASIM_OMX_SKIMS: "missing",
        },
    )
    monkeypatch.setattr(
        activitysim,
        "_native_activitysim_resolved_inputs",
        lambda **_kwargs: base_resolution,
    )
    monkeypatch.setattr(
        activitysim.ActivitysimRunner,
        "declared_expected_inputs",
        staticmethod(
            lambda *_args: {
                key: Path(f"/native/{key}")
                for key in (
                    *activitysim._ACTIVITYSIM_RUN_REQUIRED_ROLES,
                    *activitysim._ACTIVITYSIM_RUN_SKIM_ROLES,
                )
            }
        ),
    )

    with pytest.raises(RuntimeError, match="requires one published skim role"):
        activitysim._activitysim_run_resolver(
            settings=SimpleNamespace(),
            state=SimpleNamespace(),
            workspace=SimpleNamespace(),
            coupler=object(),
        )


@pytest.mark.parametrize(
    ("zarr_skims", "omx_skims", "expected_mode", "expected_extra_key"),
    [
        (Path("/inputs/skims.zarr"), None, "zarr", activitysim.ZARR_SKIMS),
        (None, Path("/inputs/skims.omx"), "omx", None),
    ],
)
def test_activitysim_run_callable_binds_only_the_resolved_skim_source(
    monkeypatch,
    zarr_skims: Path | None,
    omx_skims: Path | None,
    expected_mode: str,
    expected_extra_key: str | None,
) -> None:
    captured: dict[str, object] = {}

    class Runner:
        def run(self, inputs, workspace, *, skim_mode, extra_inputs):
            captured.update(
                inputs=inputs,
                workspace=workspace,
                skim_mode=skim_mode,
                extra_inputs=extra_inputs,
            )

    monkeypatch.setattr(
        activitysim.ModelFactory,
        "get_runner",
        lambda _self, *_args: Runner(),
    )
    workspace = SimpleNamespace(get_asim_mutable_data_dir=lambda: "/inputs")

    activitysim._activitysim_run_callable(
        Path("/inputs/land_use.csv"),
        Path("/inputs/households.csv"),
        Path("/inputs/persons.csv"),
        zarr_skims=zarr_skims,
        omx_skims=omx_skims,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
    )

    assert captured["skim_mode"] == expected_mode
    if expected_extra_key is None:
        assert captured["extra_inputs"] == {}
        assert captured["inputs"].omx_skims == omx_skims
    else:
        assert captured["extra_inputs"] == {expected_extra_key: zarr_skims}
        assert captured["inputs"].omx_skims is None


@pytest.mark.parametrize(
    ("zarr_skims", "omx_skims"),
    [
        (None, None),
        (Path("/inputs/skims.zarr"), Path("/inputs/skims.omx")),
    ],
)
def test_activitysim_run_callable_rejects_ambiguous_skim_sources(
    zarr_skims: Path | None,
    omx_skims: Path | None,
) -> None:
    with pytest.raises(RuntimeError, match="exactly one materialized skim input"):
        activitysim._activitysim_run_callable(
            Path("/inputs/land_use.csv"),
            Path("/inputs/households.csv"),
            Path("/inputs/persons.csv"),
            zarr_skims=zarr_skims,
            omx_skims=omx_skims,
            settings=SimpleNamespace(),
            state=SimpleNamespace(),
            workspace=SimpleNamespace(get_asim_mutable_data_dir=lambda: "/inputs"),
        )


def test_activitysim_projector_rejects_source_mount_when_declared_destination_missing(
    monkeypatch, tmp_path: Path
) -> None:
    source_mount = tmp_path / "source-mount" / "land_use.csv"
    source_mount.parent.mkdir()
    source_mount.write_text("historical output", encoding="utf-8")
    declared_land_use = tmp_path / "current" / "land_use.csv"
    declared_households = tmp_path / "current" / "households.csv"
    declared_persons = tmp_path / "current" / "persons.csv"
    for path in (declared_households, declared_persons):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("current output", encoding="utf-8")

    workspace = SimpleNamespace(
        full_path=str(tmp_path),
        get_asim_mutable_data_dir=lambda: str(tmp_path),
    )
    declared_outputs = {
        ASIM_LAND_USE_IN: declared_land_use,
        ASIM_HOUSEHOLDS_IN: declared_households,
        ASIM_PERSONS_IN: declared_persons,
    }
    monkeypatch.setattr(
        activitysim,
        "activitysim_preprocess_output_paths",
        lambda **_kwargs: declared_outputs,
    )
    outputs = {
        ASIM_LAND_USE_IN: SimpleNamespace(
            container_uri="workspace://current/land_use.csv",
            path=source_mount,
            abs_path=str(source_mount),
        ),
        ASIM_HOUSEHOLDS_IN: SimpleNamespace(
            container_uri="workspace://current/households.csv",
            path=declared_households,
        ),
        ASIM_PERSONS_IN: SimpleNamespace(
            container_uri="workspace://current/persons.csv",
            path=declared_persons,
        ),
    }

    with pytest.raises(
        RuntimeError,
        match="activitysim_preprocess output 'land_use_asim_in' is missing at declared destination",
    ):
        activitysim.activitysim_preprocess.project_outputs(
            outputs,
            settings=SimpleNamespace(),
            state=SimpleNamespace(),
            workspace=workspace,
        )
