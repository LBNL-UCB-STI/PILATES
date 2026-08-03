"""Freeze the ResolvedBinding V1 eligibility decision for native steps."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from consist import ExecutionOptions, ResolvedBinding, Tracker

from pilates.activitysim.outputs import ASIM_REQUIRED_RUN_OUTPUT_KEYS
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ATLAS_OUTPUT_DIR,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    FINAL_SKIMS_OMX,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.steps import STEP_DEFINITIONS


@dataclass(frozen=True)
class EligibilityEntry:
    """One reviewed V1 classification, independent of resolver implementation."""

    step_name: str
    strict_binding_required_after_task3: bool
    v1_limitation_reason: str | None = None


ELIGIBILITY_MATRIX = (
    EligibilityEntry(
        "urbansim_run",
        False,
        "configured static identity inputs are staged outside the strict semantic binding",
    ),
    EligibilityEntry("urbansim_postprocess", True),
    EligibilityEntry(
        "atlas_preprocess",
        False,
        "required datastore is read from the mutable workspace instead of its strict parameter",
    ),
    EligibilityEntry(
        "atlas_run",
        False,
        "runner mounts the workspace input directory instead of the strict parameter",
    ),
    EligibilityEntry(
        "atlas_postprocess",
        False,
        "required datastore is read from the mutable workspace instead of its strict parameter",
    ),
    EligibilityEntry(
        "activitysim_preprocess",
        False,
        "population-source fallback can be an untracked workspace file",
    ),
    EligibilityEntry(
        "activitysim_run",
        False,
        "runner mounts the mutable ActivitySim workspace instead of strict parameter paths",
    ),
    EligibilityEntry("activitysim_postprocess", True),
    EligibilityEntry(
        "beam_preprocess",
        False,
        "untracked external config parameter",
    ),
    EligibilityEntry(
        "beam_run",
        False,
        "untracked external config parameter",
    ),
    EligibilityEntry(
        "beam_postprocess",
        False,
        "dynamic workspace-only inputs and restart closure",
    ),
    EligibilityEntry(
        "beam_full_skim",
        False,
        "runner mounts the mutable BEAM workspace instead of strict snapshot paths",
    ),
    EligibilityEntry(
        "postprocessing",
        False,
        "workspace-only execution inputs",
    ),
)


def test_matrix_covers_each_native_step_definition_once() -> None:
    matrix_names = [entry.step_name for entry in ELIGIBILITY_MATRIX]

    assert len(matrix_names) == len(set(matrix_names))
    assert set(matrix_names) == set(STEP_DEFINITIONS)
    assert len(matrix_names) == 13


@pytest.mark.parametrize(
    "entry",
    [
        entry
        for entry in ELIGIBILITY_MATRIX
        if not entry.strict_binding_required_after_task3
    ],
    ids=lambda entry: entry.step_name,
)
def test_deferred_definitions_keep_an_explicit_v1_limitation(
    entry: EligibilityEntry,
) -> None:
    assert entry.v1_limitation_reason
    assert not entry.strict_binding_required_after_task3
    assert STEP_DEFINITIONS[entry.step_name].preflight_identity is False
    assert (
        "step_identity"
        not in inspect.signature(
            STEP_DEFINITIONS[entry.step_name].resolve_inputs
        ).parameters
    )


class _TrackedCoupler:
    """Provide local Consist artifacts for each role a resolver selects."""

    def __init__(self, *, tracker: Tracker, root: Path) -> None:
        self._tracker = tracker
        self._root = root
        self._values: dict[str, Any] = {}

    def seed(self, *keys: str) -> None:
        with self._tracker.start_run("seed_resolver_inputs", "test"):
            for key in keys:
                source = self._root / f"{key.replace('/', '_')}.txt"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(f"{key} input\\n", encoding="utf-8")
                self._values[key] = self._tracker.log_artifact(
                    source,
                    key=key,
                    direction="input",
                )

    def get(self, key: str, default: object = None) -> object:
        return self._values.get(key, default)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._values)


@dataclass(frozen=True)
class _ResolverEnvironment:
    settings: SimpleNamespace
    state: SimpleNamespace
    workspace: SimpleNamespace
    coupler: _TrackedCoupler
    tracker: Tracker


@pytest.fixture(scope="module")
def resolver_environment(
    tmp_path_factory: pytest.TempPathFactory,
) -> _ResolverEnvironment:
    tmp_path = tmp_path_factory.mktemp("resolved-binding-v1")
    workspace_root = tmp_path / "workspace"
    beam_input_dir = workspace_root / "beam" / "input"
    beam_input_dir.mkdir(parents=True)
    (beam_input_dir / "beam.conf").write_text("beam {}\\n", encoding="utf-8")
    workspace = SimpleNamespace(
        full_path=str(workspace_root),
        get_usim_mutable_data_dir=lambda: str(workspace_root / "urbansim" / "data"),
        get_atlas_mutable_input_dir=lambda: str(workspace_root / "atlas" / "input"),
        get_atlas_output_dir=lambda: str(workspace_root / "atlas" / "output"),
        get_asim_mutable_data_dir=lambda: str(workspace_root / "activitysim" / "data"),
        get_asim_mutable_configs_dir=lambda: str(
            workspace_root / "activitysim" / "configs"
        ),
        get_asim_output_dir=lambda: str(workspace_root / "activitysim" / "output"),
        get_beam_mutable_data_dir=lambda: str(beam_input_dir),
        get_beam_output_dir=lambda: str(workspace_root / "beam" / "output"),
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(region="test", models=SimpleNamespace(land_use=None)),
        urbansim=SimpleNamespace(
            command_template="urbansim {0}",
            input_file_template="input_{region_id}.h5",
            input_file_template_year=None,
            output_file_template="output_{year}.h5",
            region_id="001",
            region_mappings={"region_to_region_id": {"test": "001"}},
        ),
        atlas=SimpleNamespace(),
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
        beam=SimpleNamespace(config="beam.conf", full_skim=None),
        shared=SimpleNamespace(skims=SimpleNamespace(fname="final_skims.omx")),
        write_skims_to_omx=False,
    )
    state = SimpleNamespace(
        year=2030,
        current_year=2030,
        forecast_year=2030,
        iteration=1,
        current_inner_iter=1,
        start_year=2020,
        is_start_year=lambda: False,
    )
    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )
    coupler = _TrackedCoupler(tracker=tracker, root=tmp_path / "sources")
    coupler.seed(
        ASIM_HOUSEHOLDS_IN,
        ASIM_LAND_USE_IN,
        ASIM_OMX_SKIMS,
        ASIM_PERSONS_IN,
        ATLAS_OUTPUT_DIR,
        BEAM_HOUSEHOLDS_IN,
        BEAM_PERSONS_IN,
        BEAM_PLANS_IN,
        FINAL_SKIMS_OMX,
        LINKSTATS_WARMSTART,
        USIM_DATASTORE_BASE_H5,
        USIM_DATASTORE_CURRENT_H5,
        USIM_DATASTORE_H5,
        USIM_POPULATION_SOURCE_H5,
        ZARR_SKIMS,
        "atlas_mutable_input_dir",
        *ASIM_REQUIRED_RUN_OUTPUT_KEYS,
    )
    return _ResolverEnvironment(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
        tracker=tracker,
    )


@pytest.mark.parametrize(
    "entry",
    [
        entry
        for entry in ELIGIBILITY_MATRIX
        if entry.strict_binding_required_after_task3
    ],
    ids=lambda entry: entry.step_name,
)
def test_eligible_definitions_must_emit_resolved_binding_after_task3(
    entry: EligibilityEntry,
    resolver_environment: _ResolverEnvironment,
) -> None:
    assert entry.v1_limitation_reason is None
    assert entry.strict_binding_required_after_task3
    definition = STEP_DEFINITIONS[entry.step_name]
    with resolver_environment.tracker.scenario("eligibility") as scenario:
        identity = scenario.resolve_step_identity(
            definition.function,
            year=resolver_environment.state.year,
            iteration=resolver_environment.state.iteration,
            phase="run",
            stage="test",
            execution_options=ExecutionOptions(
                input_binding="paths",
                runtime_kwargs={
                    "settings": resolver_environment.settings,
                    "state": resolver_environment.state,
                    "workspace": resolver_environment.workspace,
                },
            ),
        )
        resolved = definition.resolve_inputs(
            settings=resolver_environment.settings,
            state=resolver_environment.state,
            workspace=resolver_environment.workspace,
            coupler=resolver_environment.coupler,
            step_identity=identity,
        )
    assert isinstance(resolved.binding, ResolvedBinding)
