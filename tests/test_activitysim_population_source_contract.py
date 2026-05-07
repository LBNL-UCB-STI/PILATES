from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pandas as pd
import pytest

from pilates.workflows.artifact_keys import (
    USIM_POPULATION_HOUSEHOLDS_TABLE,
    USIM_POPULATION_PERSONS_TABLE,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.binding import BindingPlan, build_binding_plan
from pilates.workflows.stages.vehicle_ownership import (
    _validate_population_h5_for_activitysim_year,
)
from pilates.workflows.steps import activitysim as steps_activitysim
from pilates.utils.usim_h5 import (
    reconcile_usim_population_table_paths,
    should_require_exact_population_year_tables,
)


class _CouplerStub:
    def get(self, _key, default=None):
        return default


class _WorkspaceStub:
    def __init__(self, root: Path):
        self.full_path = str(root)
        self._usim_dir = root / "urbansim" / "data"

    def get_usim_mutable_data_dir(self) -> str:
        return str(self._usim_dir)


def _write_root_population_h5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.HDFStore(path, mode="w") as store:
        for table_name in ("households", "persons", "jobs", "blocks"):
            store.put(f"/{table_name}", pd.DataFrame({"value": [1]}))


def _state(year: int) -> SimpleNamespace:
    return SimpleNamespace(
        year=year,
        forecast_year=year,
        Stage=SimpleNamespace(land_use="land_use"),
        is_enabled=lambda stage: stage == "land_use",
        is_start_year=lambda: False,
    )


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(models=SimpleNamespace(land_use="urbansim")),
        urbansim=SimpleNamespace(output_file_template="model_data_{year}.h5"),
    )


def _binding_entrypoint(step_name: str) -> Callable[[Path, int, Path], BindingPlan]:
    def _run(h5_path: Path, year: int, tmp_path: Path) -> BindingPlan:
        return build_binding_plan(
            step_name=step_name,
            fallback_inputs={USIM_POPULATION_SOURCE_H5: str(h5_path)},
            required_keys=(),
            optional_keys=(
                USIM_POPULATION_SOURCE_H5,
                USIM_POPULATION_HOUSEHOLDS_TABLE,
                USIM_POPULATION_PERSONS_TABLE,
            ),
            settings=_settings(),
            state=_state(year),
            workspace=_WorkspaceStub(tmp_path),
            year=year,
        )

    return _run


def _preprocess_runtime_entrypoint(h5_path: Path, year: int, tmp_path: Path) -> dict:
    return steps_activitysim._resolve_activitysim_preprocess_runtime_inputs(
        settings=_settings(),
        state=_state(year),
        workspace=_WorkspaceStub(tmp_path),
        coupler=_CouplerStub(),
        step_inputs={USIM_POPULATION_SOURCE_H5: str(h5_path)},
    )


def _preprocessor_table_policy_entrypoint(
    h5_path: Path,
    year: int,
    _tmp_path: Path,
) -> dict:
    return reconcile_usim_population_table_paths(
        h5_path=str(h5_path),
        year=year,
        require_exact_year=should_require_exact_population_year_tables(
            h5_path=str(h5_path),
            year=year,
            require_exact_year=True,
        ),
    )


def _vehicle_ownership_validation_entrypoint(
    h5_path: Path,
    year: int,
    _tmp_path: Path,
) -> dict:
    _validate_population_h5_for_activitysim_year(
        path=h5_path,
        year=year,
        context="contract test",
    )
    return {}


@pytest.mark.parametrize(
    "entrypoint",
    [
        _binding_entrypoint("activitysim_preprocess"),
        _binding_entrypoint("activitysim_postprocess"),
        _preprocess_runtime_entrypoint,
        _preprocessor_table_policy_entrypoint,
        _vehicle_ownership_validation_entrypoint,
    ],
    ids=[
        "activitysim_preprocess_binding",
        "activitysim_postprocess_binding",
        "activitysim_preprocess_runtime",
        "activitysim_preprocessor_table_policy",
        "vehicle_ownership_validation",
    ],
)
def test_activitysim_population_source_entrypoints_reject_stale_root_only_h5(
    tmp_path,
    entrypoint,
) -> None:
    h5_path = tmp_path / "urbansim" / "data" / "model_data_2019.h5"
    _write_root_population_h5(h5_path)

    with pytest.raises(KeyError, match="require_exact_year=True"):
        entrypoint(h5_path, 2021, tmp_path)


@pytest.mark.parametrize(
    "entrypoint",
    [
        _binding_entrypoint("activitysim_preprocess"),
        _binding_entrypoint("activitysim_postprocess"),
        _preprocess_runtime_entrypoint,
        _preprocessor_table_policy_entrypoint,
        _vehicle_ownership_validation_entrypoint,
    ],
    ids=[
        "activitysim_preprocess_binding",
        "activitysim_postprocess_binding",
        "activitysim_preprocess_runtime",
        "activitysim_preprocessor_table_policy",
        "vehicle_ownership_validation",
    ],
)
def test_activitysim_population_source_entrypoints_accept_same_year_root_only_h5(
    tmp_path,
    entrypoint,
) -> None:
    h5_path = tmp_path / "urbansim" / "data" / "model_data_2021.h5"
    _write_root_population_h5(h5_path)

    result = entrypoint(h5_path, 2021, tmp_path)

    if isinstance(result, BindingPlan):
        assert result.inputs[USIM_POPULATION_SOURCE_H5] == str(h5_path)
        resolved = result.metadata.get("resolved_values_by_semantic_key", {})
        if resolved:
            assert resolved[USIM_POPULATION_HOUSEHOLDS_TABLE] == "/households"
            assert resolved[USIM_POPULATION_PERSONS_TABLE] == "/persons"
    elif result:
        if "population_source_h5_path" in result:
            assert result["population_source_h5_path"] == str(h5_path)
        assert result[USIM_POPULATION_HOUSEHOLDS_TABLE] == "/households"
        assert result[USIM_POPULATION_PERSONS_TABLE] == "/persons"
