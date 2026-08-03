from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
from consist import BindingResult

from pilates.workflows.artifact_keys import (
    FINAL_SKIMS_OMX,
    USIM_DATASTORE_H5,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.atlas_state import AtlasSubState
from pilates.workflows.input_authority import requires_prior_beam_skim_handoff
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.steps import activitysim, urbansim_atlas
from pilates.beam.beam_input_staging import copy_vehicles_from_atlas


def _settings(*, traffic_assignment: str | None = "beam") -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(
            start_year=2020,
            models=SimpleNamespace(traffic_assignment=traffic_assignment),
        )
    )


def _state(*, current_year: int, iteration: int = 0) -> SimpleNamespace:
    return SimpleNamespace(current_year=current_year, iteration=iteration)


def test_prior_beam_skim_handoff_is_required_only_after_bootstrap() -> None:
    settings = _settings()

    assert not requires_prior_beam_skim_handoff(
        settings=settings,
        state=_state(current_year=2020),
    )
    assert requires_prior_beam_skim_handoff(
        settings=settings,
        state=_state(current_year=2022),
    )
    assert requires_prior_beam_skim_handoff(
        settings=settings,
        state=_state(current_year=2020, iteration=1),
    )
    assert not requires_prior_beam_skim_handoff(
        settings=_settings(traffic_assignment=None),
        state=_state(current_year=2022),
    )


def test_atlas_subyear_requires_prior_beam_skim_at_later_parent_iteration() -> None:
    """ATLAS sub-years retain the parent iteration used by input authority."""

    class ParentState:
        year = 2020
        forecast_year = 2022
        start_year = 2020
        full_settings = None

        def __init__(self) -> None:
            self.current_inner_iter = 1

        @property
        def iteration(self) -> int:
            return self.current_inner_iter

    parent = ParentState()
    atlas_state = AtlasSubState(parent, 2020)

    assert requires_prior_beam_skim_handoff(
        settings=_settings(),
        state=atlas_state,
    )


def test_atlas_later_subyear_does_not_require_beam_skim_before_first_beam_run() -> None:
    """ATLAS sub-years do not themselves advance the BEAM handoff frontier."""

    class ParentState:
        year = 2020
        forecast_year = 2022
        start_year = 2020
        full_settings = None

        @property
        def iteration(self) -> int:
            return 0

    atlas_state = AtlasSubState(ParentState(), 2022)

    assert not requires_prior_beam_skim_handoff(
        settings=_settings(),
        state=atlas_state,
    )


def test_native_resolvers_require_a_beam_skim_handoff_after_bootstrap(
    monkeypatch,
) -> None:
    settings = _settings()
    settings.run.region = "test"
    settings.urbansim = SimpleNamespace(
        input_file_template="input_{region_id}.h5",
        region_mappings={"region_to_region_id": {"test": "001"}},
    )
    state = _state(current_year=2022)
    workspace = SimpleNamespace(
        full_path=".",
        get_usim_mutable_data_dir=lambda: "urbansim/data",
        get_atlas_mutable_input_dir=lambda: "atlas/input",
    )
    captured: dict[str, tuple[str, ...]] = {}

    def resolve_activitysim(**kwargs):
        captured["activitysim"] = kwargs["required_roles"]
        return ResolvedStepInputs(
            step_name="activitysim_preprocess", binding=BindingResult()
        )

    def resolve_urbansim(**kwargs):
        captured["urbansim"] = kwargs["required_roles"]
        return ResolvedStepInputs(step_name="urbansim_run", binding=BindingResult())

    def resolve_atlas(**kwargs):
        captured["atlas"] = kwargs["required_roles"]
        return ResolvedStepInputs(step_name="atlas_preprocess", binding=BindingResult())

    monkeypatch.setattr(
        activitysim.ActivitysimPreprocessor,
        "declared_expected_inputs",
        staticmethod(lambda *_args: {}),
    )
    monkeypatch.setattr(activitysim, "resolve_artifact_roles", resolve_activitysim)
    monkeypatch.setattr(urbansim_atlas, "_resolve_native_inputs", resolve_urbansim)

    activitysim._activitysim_preprocess_resolver(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=object(),
    )
    urbansim_atlas._resolve_urbansim_run_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=object(),
    )

    monkeypatch.setattr(urbansim_atlas, "_resolve_native_inputs", resolve_atlas)
    monkeypatch.setattr(
        urbansim_atlas.AtlasPreprocessor,
        "expected_inputs",
        staticmethod(lambda *_args: {USIM_DATASTORE_H5: "urbansim/data/input.h5"}),
    )
    urbansim_atlas._resolve_atlas_preprocess_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=object(),
    )

    assert FINAL_SKIMS_OMX not in captured["activitysim"]
    assert FINAL_SKIMS_OMX in captured["urbansim"]
    assert FINAL_SKIMS_OMX in captured["atlas"]


def test_activitysim_uses_zarr_only_after_beam_bootstrap(monkeypatch) -> None:
    """Later ActivitySim iterations must not select an OMX skim handoff."""

    captured: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    zarr = object()
    run_inputs = {
        "land_use_asim_in": object(),
        "households_asim_in": object(),
        "persons_asim_in": object(),
        ZARR_SKIMS: zarr,
    }

    def resolve_native(**kwargs):
        step_name = kwargs["step_name"]
        captured[step_name] = (
            kwargs["required_roles"],
            kwargs["optional_roles"],
        )
        if step_name == "activitysim_run":
            return ResolvedStepInputs(
                step_name=step_name,
                binding=BindingResult(inputs=run_inputs),
                required_roles=kwargs["required_roles"],
                optional_roles=kwargs["optional_roles"],
                source_by_role={ZARR_SKIMS: "coupler"},
                selected_key_by_role={ZARR_SKIMS: ZARR_SKIMS},
                logical_destinations={ZARR_SKIMS: "activitysim/cache/skims.zarr"},
            )
        return ResolvedStepInputs(
            step_name=step_name,
            binding=BindingResult(inputs={}),
        )

    monkeypatch.setattr(
        activitysim,
        "_native_activitysim_resolved_inputs",
        resolve_native,
    )
    monkeypatch.setattr(
        activitysim.ActivitysimRunner,
        "declared_expected_inputs",
        staticmethod(lambda *_args: {}),
    )
    state = _state(current_year=2020, iteration=1)

    activitysim._activitysim_preprocess_resolver(
        settings=_settings(),
        state=state,
        workspace=SimpleNamespace(),
        coupler=object(),
    )
    resolved_run = activitysim._activitysim_run_resolver(
        settings=_settings(),
        state=state,
        workspace=SimpleNamespace(),
        coupler=object(),
    )

    assert captured["activitysim_preprocess"] == (
        (USIM_POPULATION_SOURCE_H5,),
        (),
    )
    assert captured["activitysim_run"] == (
        (
            "land_use_asim_in",
            "households_asim_in",
            "persons_asim_in",
            ZARR_SKIMS,
        ),
        (),
    )
    assert dict(resolved_run.binding.inputs or {}) == run_inputs


def test_activitysim_preprocess_skips_omx_preparation_after_beam_bootstrap(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _Preprocessor:
        def preprocess(self, _workspace, **kwargs):
            captured.update(kwargs)

    class _Factory:
        def get_preprocessor(self, _model, _state):
            return _Preprocessor()

    monkeypatch.setattr(activitysim, "ModelFactory", _Factory)
    activitysim._activitysim_preprocess_callable(
        Path("/bound/population.h5"),
        settings=_settings(),
        state=_state(current_year=2020, iteration=1),
        workspace=SimpleNamespace(),
    )

    assert captured["prepare_omx_skims"] is False
    assert "final_skims_omx" not in captured


def test_activitysim_preprocess_does_not_declare_stale_omx_after_bootstrap(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        activitysim.ActivitysimPreprocessor,
        "expected_outputs",
        staticmethod(
            lambda *_args: {
                "land_use_asim_in": "activitysim/data/land_use.csv",
                "households_asim_in": "activitysim/data/households.csv",
                "persons_asim_in": "activitysim/data/persons.csv",
                "omx_skims": "activitysim/data/skims.omx",
            }
        ),
    )

    declared = activitysim.activitysim_preprocess_output_paths(
        settings=_settings(),
        state=_state(current_year=2020, iteration=1),
        workspace=SimpleNamespace(),
    )

    assert "omx_skims" not in declared


def test_activitysim_population_source_cannot_fall_back_after_land_use(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    state = SimpleNamespace(
        current_year=2022,
        iteration=0,
        year=2022,
        is_enabled=lambda _stage: True,
    )

    def resolve_roles(**kwargs):
        captured.update(kwargs)
        return ResolvedStepInputs(
            step_name="activitysim_preprocess", binding=BindingResult()
        )

    monkeypatch.setattr(
        activitysim.ActivitysimPreprocessor,
        "declared_expected_inputs",
        staticmethod(lambda *_args: {}),
    )
    monkeypatch.setattr(activitysim, "resolve_artifact_roles", resolve_roles)

    activitysim._activitysim_preprocess_resolver(
        settings=_settings(),
        state=state,
        workspace=SimpleNamespace(full_path="."),
        coupler=object(),
    )

    rules = {rule.semantic_key: rule for rule in captured["artifact_rules"]}
    population = rules["usim_population_source_h5"]
    assert population.allow_fallback is False
    assert population.fallback_provider is None


def test_native_callables_disable_adapter_skim_rediscovery_after_beam(
    monkeypatch,
) -> None:
    settings = _settings()
    state = _state(current_year=2022)
    workspace = SimpleNamespace()
    captured: dict[str, dict[str, object]] = {}

    class _ActivitysimPreprocessor:
        def preprocess(self, _workspace, **kwargs):
            captured["activitysim"] = kwargs

    class _Factory:
        def get_preprocessor(self, _model, _state):
            return _ActivitysimPreprocessor()

    class _UrbansimRunner:
        def __init__(self, *_args):
            pass

        def run(self, **kwargs):
            captured["urbansim"] = kwargs

    class _AtlasPreprocessor:
        def __init__(self, *_args):
            pass

        def preprocess(self, _workspace, **kwargs):
            captured["atlas"] = kwargs

    monkeypatch.setattr(activitysim, "ModelFactory", _Factory)
    monkeypatch.setattr(urbansim_atlas, "UrbansimRunner", _UrbansimRunner)
    monkeypatch.setattr(urbansim_atlas, "AtlasPreprocessor", _AtlasPreprocessor)

    population = Path("/bound/population.h5")
    skims = Path("/bound/final-skims.omx")
    activitysim._activitysim_preprocess_callable(
        population,
        skims,
        settings=settings,
        state=state,
        workspace=workspace,
    )
    urbansim_atlas._native_urbansim_run(
        population,
        skims,
        settings=settings,
        state=state,
        workspace=workspace,
    )
    urbansim_atlas._native_atlas_preprocess(
        population,
        skims,
        settings=settings,
        state=state,
        workspace=workspace,
    )

    assert captured["activitysim"]["prepare_omx_skims"] is False
    assert "final_skims_omx" not in captured["activitysim"]
    assert captured["atlas"]["allow_workspace_skim_fallback"] is False
    assert captured["urbansim"] == {
        "usim_datastore_h5": population,
        "final_skims_omx": skims,
        "workspace": workspace,
    }
    assert captured["atlas"]["final_skims_omx"] == skims


def test_direct_beam_vehicle_staging_requires_an_explicit_atlas_artifact() -> None:
    with pytest.raises(ValueError, match="explicit ATLAS vehicles2 source_path"):
        copy_vehicles_from_atlas(
            workspace=object(),
            state=object(),
            resolve_beam_exchange_scenario_folder_fn=lambda _workspace: ".",
        )


def test_direct_beam_vehicle_staging_rejects_wrong_forecast_year(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "vehicles2_2018.csv"
    source_path.write_text("vehicle_id\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forecast_year=2019.*parsed_source_year=2018"):
        copy_vehicles_from_atlas(
            workspace=SimpleNamespace(full_path=str(tmp_path)),
            state=SimpleNamespace(forecast_year=2019),
            resolve_beam_exchange_scenario_folder_fn=lambda _workspace: str(
                tmp_path / "beam-input"
            ),
            source_path=str(source_path),
            require_exact_year=True,
        )
