from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
from consist import BindingResult

from pilates.workflows.artifact_keys import FINAL_SKIMS_OMX, USIM_DATASTORE_H5
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


def test_native_resolvers_require_a_beam_skim_handoff_after_bootstrap(
    monkeypatch,
) -> None:
    settings = _settings()
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
        return ResolvedStepInputs(
            step_name="urbansim_preprocess", binding=BindingResult()
        )

    def resolve_atlas(**kwargs):
        captured["atlas"] = kwargs["required_roles"]
        return ResolvedStepInputs(step_name="atlas_preprocess", binding=BindingResult())

    monkeypatch.setattr(
        activitysim.ActivitysimPreprocessor,
        "declared_expected_inputs",
        staticmethod(lambda *_args: {}),
    )
    monkeypatch.setattr(activitysim, "resolve_artifact_roles", resolve_activitysim)
    monkeypatch.setattr(
        urbansim_atlas,
        "_urbansim_preprocess_native_output_paths",
        lambda **_kwargs: {USIM_DATASTORE_H5: "urbansim/data/input.h5"},
    )
    monkeypatch.setattr(urbansim_atlas, "_resolve_native_inputs", resolve_urbansim)

    activitysim._activitysim_preprocess_resolver(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=object(),
    )
    urbansim_atlas._resolve_urbansim_preprocess_inputs(
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

    assert FINAL_SKIMS_OMX in captured["activitysim"]
    assert FINAL_SKIMS_OMX in captured["urbansim"]
    assert FINAL_SKIMS_OMX in captured["atlas"]


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

    class _UrbansimPreprocessor:
        def __init__(self, *_args):
            pass

        def preprocess(self, _workspace, **kwargs):
            captured["urbansim"] = kwargs

    class _AtlasPreprocessor:
        def __init__(self, *_args):
            pass

        def preprocess(self, _workspace, **kwargs):
            captured["atlas"] = kwargs

    monkeypatch.setattr(activitysim, "ModelFactory", _Factory)
    monkeypatch.setattr(urbansim_atlas, "UrbansimPreprocessor", _UrbansimPreprocessor)
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
    urbansim_atlas._native_urbansim_preprocess(
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

    assert all(
        values["allow_workspace_skim_fallback"] is False for values in captured.values()
    )
    assert captured["activitysim"]["final_skims_omx"] == skims
    assert captured["urbansim"]["final_skims_omx"] == skims
    assert captured["atlas"]["final_skims_omx"] == skims


def test_direct_beam_vehicle_staging_requires_an_explicit_atlas_artifact() -> None:
    with pytest.raises(ValueError, match="explicit ATLAS vehicles2 source_path"):
        copy_vehicles_from_atlas(
            workspace=object(),
            state=object(),
            resolve_beam_exchange_scenario_folder_fn=lambda _workspace: ".",
        )
