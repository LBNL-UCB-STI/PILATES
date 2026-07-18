from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from consist import BindingResult, resolve_step_contract

from pilates.activitysim.outputs import ASIM_REQUIRED_RUN_OUTPUT_KEYS
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_PERSONS_IN,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.steps import activitysim


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
    population = tmp_path / "population.h5"
    population.write_text("population", encoding="utf-8")
    final_skims = tmp_path / "final_skims.omx"
    final_skims.write_text("skims", encoding="utf-8")
    captured: dict[str, object] = {}

    def resolve_roles(**kwargs: object) -> ResolvedStepInputs:
        captured.update(kwargs)
        return ResolvedStepInputs(
            step_name="activitysim_preprocess",
            binding=BindingResult(
                inputs={
                    USIM_POPULATION_SOURCE_H5: population,
                    "final_skims_omx": final_skims,
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

    assert resolved.binding.inputs == {
        USIM_POPULATION_SOURCE_H5: population,
        "final_skims_omx": final_skims,
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
