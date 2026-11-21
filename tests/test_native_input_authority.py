from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from pilates.workflows.artifact_keys import (
    ASIM_OMX_SKIMS,
    FINAL_SKIMS_OMX,
    ZARR_SKIMS,
)
from pilates.workflows.orchestrator import AtlasSubState
from pilates.workflows.state_helpers import (
    requires_prior_beam_linkstats_handoff,
    requires_prior_beam_skim_handoff,
)
from pilates.workflows.sim_state import (
    DemandState,
    LandUseState,
    NetworkState,
    SimState,
)
from pilates.workflows.steps import activitysim, atlas, urbansim
from pilates.urbansim.outputs import UrbanSimPreprocessOutputs
from pilates.urbansim.runner import UrbanSimLaunchContext
from pilates.beam.beam_input_staging import copy_vehicles_from_atlas
from tests.conftest import make_conforming_workspace


def _settings(*, traffic_assignment: str | None = "beam") -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(
            start_year=2020,
            models=SimpleNamespace(traffic_assignment=traffic_assignment),
        )
    )


def _state(*, current_year: int, iteration: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        current_year=current_year,
        iteration=iteration,
        is_start_year=lambda: current_year == 2020,
    )


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


def test_prior_beam_linkstats_handoff_is_required_only_after_bootstrap() -> None:
    settings = _settings()

    assert not requires_prior_beam_linkstats_handoff(
        settings=settings,
        state=_state(current_year=2020),
    )
    assert requires_prior_beam_linkstats_handoff(
        settings=settings,
        state=_state(current_year=2022),
    )
    assert requires_prior_beam_linkstats_handoff(
        settings=settings,
        state=_state(current_year=2020, iteration=1),
    )
    assert not requires_prior_beam_linkstats_handoff(
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
        output_file_template="output_{year}.h5",
        region_mappings={"region_to_region_id": {"test": "001"}},
    )
    state = _state(current_year=2022)
    state.forecast_year = 2022
    workspace = make_conforming_workspace(
        ".",
        full_path=".",
        usim_mutable_data_dir="urbansim/data",
        atlas_mutable_input_dir="atlas/input",
    )
    monkeypatch.setattr(
        activitysim.ActivitysimPreprocessor,
        "declared_expected_inputs",
        staticmethod(lambda *_args: {}),
    )
    sim_state = SimState(
        land_use=LandUseState(
            current_h5=Path("curr.h5"),
            population_source=Path("pop.h5"),
        ),
        network=NetworkState(skims_omx=Path("final.omx")),
    )

    asim_inputs = activitysim._activitysim_preprocess_resolver(
        settings=settings,
        state=state,
        workspace=workspace,
        sim_state=sim_state,
    )
    usim_inputs = urbansim._resolve_urbansim_run_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        sim_state=sim_state,
    )
    atlas_inputs = atlas._resolve_atlas_preprocess_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        sim_state=sim_state,
    )

    assert FINAL_SKIMS_OMX not in asim_inputs.required_keys
    assert FINAL_SKIMS_OMX not in asim_inputs.optional_keys
    assert FINAL_SKIMS_OMX in usim_inputs.required_keys
    assert FINAL_SKIMS_OMX in atlas_inputs.required_keys


def test_activitysim_uses_zarr_only_after_beam_bootstrap(
    monkeypatch, tmp_path: Path
) -> None:
    """Later ActivitySim iterations must not select an OMX skim handoff."""

    zarr = tmp_path / "selected-skims.zarr"
    zarr.mkdir()
    (zarr / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")
    sim_state = SimState(
        land_use=LandUseState(population_source=tmp_path / "pop.h5"),
        demand=DemandState(
            land_use_csv=tmp_path / "lu.csv",
            households_in=tmp_path / "hh.csv",
            persons_in=tmp_path / "per.csv",
        ),
        network=NetworkState(skims_zarr=zarr),
    )

    monkeypatch.setattr(
        activitysim.ActivitysimRunner,
        "declared_expected_inputs",
        staticmethod(lambda *_args: {}),
    )
    state = _state(current_year=2020, iteration=1)
    workspace = make_conforming_workspace(
        "/workspace",
        full_path="/workspace",
        asim_mutable_data_dir="/workspace/activitysim/data",
        asim_output_dir="/workspace/activitysim/output",
        asim_mutable_configs_dir="/workspace/activitysim/configs",
        asim_runtime_cache_dir="/workspace/activitysim/output/cache",
    )

    asim_prep = activitysim._activitysim_preprocess_resolver(
        settings=_settings(),
        state=state,
        workspace=workspace,
        sim_state=sim_state,
    )
    resolved_run = activitysim._activitysim_run_resolver(
        settings=_settings(),
        state=state,
        workspace=workspace,
        sim_state=sim_state,
    )

    assert FINAL_SKIMS_OMX not in asim_prep.optional_keys
    assert ZARR_SKIMS in resolved_run.inputs
    assert ASIM_OMX_SKIMS not in resolved_run.inputs


def test_activitysim_preprocess_skips_omx_preparation_after_beam_bootstrap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    settings = _settings()
    state = _state(current_year=2020, iteration=1)
    workspace = make_conforming_workspace(
        tmp_path,
        asim_mutable_data_dir=str(tmp_path / "asim_data"),
    )

    class _Preprocessor:
        def preprocess(self, _workspace, **kwargs):
            captured.update(kwargs)
            from consist import OutputArtifactSpec

            for spec in activitysim.activitysim_preprocess_output_paths(
                settings=settings,
                state=state,
                workspace=workspace,
            ).values():
                out = Path(spec.path if isinstance(spec, OutputArtifactSpec) else spec)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.touch()

    monkeypatch.setattr(
        activitysim,
        "ActivitysimPreprocessor",
        lambda *_args: _Preprocessor(),
    )
    activitysim._activitysim_preprocess_callable(
        Path("/bound/population.h5"),
        settings=settings,
        state=state,
        workspace=workspace,
    )

    assert captured["prepare_omx_skims"] is False
    assert "final_skims_omx" not in captured


def test_activitysim_preprocess_does_not_declare_stale_omx_after_bootstrap() -> None:
    declared = activitysim.activitysim_preprocess_output_paths(
        settings=_settings(),
        state=_state(current_year=2020, iteration=1),
        workspace=make_conforming_workspace(
            ".", asim_mutable_data_dir="activitysim/data"
        ),
    )

    assert "omx_skims" not in declared


def test_activitysim_population_source_cannot_fall_back_after_land_use(
    monkeypatch,
) -> None:
    state = SimpleNamespace(
        current_year=2022,
        iteration=0,
        year=2022,
        is_enabled=lambda _stage: True,
    )
    monkeypatch.setattr(
        activitysim.ActivitysimPreprocessor,
        "declared_expected_inputs",
        staticmethod(lambda *_args: {}),
    )
    # With land_use enabled and population_source missing, current_h5 is NOT used as fallback
    resolved = activitysim._activitysim_preprocess_resolver(
        settings=_settings(),
        state=state,
        workspace=make_conforming_workspace(".", full_path="."),
        sim_state=SimState(land_use=LandUseState(current_h5=Path("curr.h5"))),
    )
    assert "usim_population_source_h5" not in resolved.inputs


def test_native_callables_disable_adapter_skim_rediscovery_after_beam(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from consist import OutputArtifactSpec

    settings = _settings()
    settings.urbansim = SimpleNamespace(output_file_template="output_{year}.h5")
    state = _state(current_year=2022)
    state.forecast_year = 2022
    state.year = 2022
    workspace = make_conforming_workspace(
        tmp_path,
        full_path=str(tmp_path),
        asim_mutable_data_dir=str(tmp_path / "asim_data"),
        atlas_output_dir=str(tmp_path / "atlas_output"),
        usim_mutable_data_dir=str(tmp_path / "usim_data"),
    )
    captured: dict[str, dict[str, object]] = {}

    class _ActivitysimPreprocessor:
        def preprocess(self, _workspace, **kwargs):
            captured["activitysim"] = kwargs
            for spec in activitysim.activitysim_preprocess_output_paths(
                settings=settings,
                state=state,
                workspace=workspace,
            ).values():
                out = Path(spec.path if isinstance(spec, OutputArtifactSpec) else spec)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.touch()

    class _UrbansimRunner:
        def __init__(self, *_args):
            pass

        def run(self, inputs, launch_context):
            captured["urbansim"] = {
                "inputs": inputs,
                "launch_context": launch_context,
            }
            for target in (
                launch_context.output_datastore,
                Path(workspace.get_usim_mutable_data_dir()) / "output_2022.h5",
            ):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()

    class _AtlasPreprocessor:
        def __init__(self, *_args):
            pass

        def preprocess(self, _workspace, **kwargs):
            captured["atlas"] = kwargs

    monkeypatch.setattr(
        activitysim,
        "ActivitysimPreprocessor",
        lambda *_args: _ActivitysimPreprocessor(),
    )
    monkeypatch.setattr(urbansim, "UrbansimRunner", _UrbansimRunner)
    monkeypatch.setattr(atlas, "AtlasPreprocessor", _AtlasPreprocessor)

    population = Path("/bound/population.h5")
    skims = Path("/bound/final-skims.omx")
    urbansim_launch_context = UrbanSimLaunchContext(
        mutable_data_dir=tmp_path / "urbansim",
        output_datastore=tmp_path / "urbansim" / "output_2022.h5",
    )
    staged = UrbanSimPreprocessOutputs(
        usim_mutable_data_dir=urbansim_launch_context.mutable_data_dir,
        prepared_inputs={"usim_datastore_h5": population},
    )
    monkeypatch.setattr(
        urbansim,
        "stage_urbansim_run_workspace",
        lambda **_kwargs: staged,
    )
    activitysim._activitysim_preprocess_callable(
        population,
        skims,
        settings=settings,
        state=state,
        workspace=workspace,
    )
    urbansim._native_urbansim_run(
        population,
        skims,
        settings=settings,
        state=state,
        workspace=workspace,
        urbansim_launch_context=urbansim_launch_context,
    )
    atlas._native_atlas_preprocess(
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
        "inputs": staged,
        "launch_context": urbansim_launch_context,
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
            workspace=make_conforming_workspace(tmp_path, full_path=str(tmp_path)),
            state=SimpleNamespace(forecast_year=2019),
            resolve_beam_exchange_scenario_folder_fn=lambda _workspace: str(
                tmp_path / "beam-input"
            ),
            source_path=str(source_path),
            require_exact_year=True,
        )
