from types import SimpleNamespace

import pytest

from pilates.activitysim.inputs import build_activitysim_inputs
from pilates.workflows.binding import BindingPlan
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_PERSONS_IN,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)


class DummyWorkspace:
    def __init__(self, root):
        self.full_path = str(root)
        self._root = root

    def get_asim_mutable_data_dir(self):
        return str(self._root / "activitysim" / "data")

    def get_usim_mutable_data_dir(self):
        return str(self._root / "urbansim" / "data")

    def get_asim_output_dir(self):
        return str(self._root / "activitysim" / "output")


def _surface(*, land_use_enabled: bool = False):
    return SimpleNamespace(
        profile=SimpleNamespace(land_use_enabled=land_use_enabled),
        step_surface=lambda _name: None,
    )


def test_build_activitysim_inputs_merges_coupler_and_usim(tmp_path) -> None:
    workspace = DummyWorkspace(tmp_path)
    asim_dir = tmp_path / "activitysim" / "data"
    asim_dir.mkdir(parents=True)
    (asim_dir / "households.csv").write_text("")
    (asim_dir / "persons.csv").write_text("")
    (asim_dir / "land_use.csv").write_text("")

    coupler = {"zarr_skims": "skims.zarr"}
    usim_inputs = {USIM_DATASTORE_CURRENT_H5: "/tmp/usim.h5"}

    inputs, descriptions = build_activitysim_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        year=2018,
        iteration=0,
        coupler=coupler,
        usim_inputs=usim_inputs,
        surface=_surface(),
    )

    assert inputs[ASIM_HOUSEHOLDS_IN] == str(asim_dir / "households.csv")
    assert inputs[ASIM_PERSONS_IN] == str(asim_dir / "persons.csv")
    assert inputs[ASIM_LAND_USE_IN] == str(asim_dir / "land_use.csv")
    assert inputs[USIM_POPULATION_SOURCE_H5] == "/tmp/usim.h5"
    assert inputs[ZARR_SKIMS] == "skims.zarr"
    assert ASIM_HOUSEHOLDS_IN in descriptions


def test_build_activitysim_inputs_uses_base_datastore_fallback(tmp_path) -> None:
    workspace = DummyWorkspace(tmp_path)
    asim_dir = tmp_path / "activitysim" / "data"
    asim_dir.mkdir(parents=True)
    (asim_dir / "households.csv").write_text("")
    (asim_dir / "persons.csv").write_text("")
    (asim_dir / "land_use.csv").write_text("")

    coupler = {}
    usim_inputs = {USIM_DATASTORE_BASE_H5: "/tmp/usim_base.h5"}

    inputs, _ = build_activitysim_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        year=2018,
        iteration=0,
        coupler=coupler,
        usim_inputs=usim_inputs,
        surface=_surface(),
    )

    assert inputs[USIM_POPULATION_SOURCE_H5] == "/tmp/usim_base.h5"


def test_build_activitysim_inputs_prefers_explicit_base_over_stale_coupler_current(
    tmp_path,
) -> None:
    workspace = DummyWorkspace(tmp_path)
    asim_dir = tmp_path / "activitysim" / "data"
    asim_dir.mkdir(parents=True)
    (asim_dir / "households.csv").write_text("")
    (asim_dir / "persons.csv").write_text("")
    (asim_dir / "land_use.csv").write_text("")

    coupler = {USIM_DATASTORE_CURRENT_H5: "/tmp/coupler_current.h5"}
    usim_inputs = {
        USIM_DATASTORE_CURRENT_H5: "/tmp/explicit_current.h5",
        USIM_DATASTORE_BASE_H5: "/tmp/explicit_base.h5",
    }

    inputs, _ = build_activitysim_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        year=2018,
        iteration=0,
        coupler=coupler,
        usim_inputs=usim_inputs,
        surface=_surface(),
    )

    assert inputs[USIM_POPULATION_SOURCE_H5] == "/tmp/explicit_base.h5"


def test_build_activitysim_inputs_resolves_population_source_for_forecast_year(
    monkeypatch,
    tmp_path,
) -> None:
    workspace = DummyWorkspace(tmp_path)
    asim_dir = tmp_path / "activitysim" / "data"
    asim_dir.mkdir(parents=True)
    (asim_dir / "households.csv").write_text("")
    (asim_dir / "persons.csv").write_text("")
    (asim_dir / "land_use.csv").write_text("")

    population_h5 = tmp_path / "urbansim" / "data" / "model_data_2021.h5"
    population_h5.parent.mkdir(parents=True)
    population_h5.write_text("population")
    captured_years = []

    def _fake_build_binding_plan(**kwargs):
        captured_years.append(kwargs["year"])
        return BindingPlan(
            inputs={USIM_POPULATION_SOURCE_H5: str(population_h5)},
            source_by_key={USIM_POPULATION_SOURCE_H5: "fallback"},
            metadata={
                "selected_key_by_semantic_key": {
                    USIM_POPULATION_SOURCE_H5: USIM_POPULATION_SOURCE_H5
                }
            },
        )

    monkeypatch.setattr(
        "pilates.activitysim.inputs.build_binding_plan",
        _fake_build_binding_plan,
    )

    inputs, descriptions = build_activitysim_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2019, forecast_year=2021),
        workspace=workspace,
        year=2019,
        iteration=0,
        coupler={},
        usim_inputs={},
        surface=_surface(land_use_enabled=True),
    )

    assert captured_years == [2021]
    assert inputs[USIM_POPULATION_SOURCE_H5] == str(population_h5)
    assert "population year 2021" in descriptions[USIM_POPULATION_SOURCE_H5]


def test_build_activitysim_inputs_requires_surface(tmp_path) -> None:
    workspace = DummyWorkspace(tmp_path)
    asim_dir = tmp_path / "activitysim" / "data"
    asim_dir.mkdir(parents=True)
    (asim_dir / "households.csv").write_text("")
    (asim_dir / "persons.csv").write_text("")
    (asim_dir / "land_use.csv").write_text("")

    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'surface'"):
        build_activitysim_inputs(
            settings=SimpleNamespace(),
            state=SimpleNamespace(),
            workspace=workspace,
            year=2018,
            iteration=0,
            coupler={},
            usim_inputs={USIM_DATASTORE_BASE_H5: "/tmp/usim_base.h5"},
        )
