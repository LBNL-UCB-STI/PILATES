"""Freeze the ResolvedBinding V1 eligibility decision for native steps."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from consist import BindingResult, ExecutionOptions, ResolvedBinding, Tracker
from consist.core.identity import IdentityManager
from consist.integrations.activitysim import ActivitySimConfigAdapter

from pilates.activitysim.outputs import (
    ASIM_REQUIRED_RUN_OUTPUT_KEYS,
    ActivitySimPreprocessOutputs,
)
from pilates.activitysim.runner import (
    ActivitySimConfigStagingPlan,
    ActivitySimLaunchContext,
    ActivitysimRunner,
)
from pilates.activitysim.config_roots import required_activitysim_config_roots
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
import pilates.workflows.steps.activitysim as activitysim_steps
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.step_consist_meta import activitysim_config_root_dirs


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
        "runtime cache, warmup, and model output closure remain outside strict binding",
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


def test_activitysim_config_identity_roots_use_stable_adapter_order(
    tmp_path: Path,
) -> None:
    mutable_root = tmp_path / "activitysim" / "configs"
    roots = activitysim_config_root_dirs(
        SimpleNamespace(main_configs_dir="scenarios/sfbay/configs"),
        mutable_root,
    )

    assert roots == (
        mutable_root / "scenarios" / "sfbay" / "configs",
        mutable_root / "configs",
        mutable_root / "configs_extended",
        mutable_root / "configs_mp",
        mutable_root / "configs_sh_compile",
    )


def test_activitysim_config_adapter_identity_is_portable_and_content_sensitive(
    tmp_path: Path,
) -> None:
    """Equivalent staged config trees retain identity across workspace roots."""
    settings = SimpleNamespace(
        activitysim=SimpleNamespace(main_configs_dir="configs_extended")
    )

    def make_roots(workspace_name: str) -> tuple[Path, ...]:
        root = tmp_path / workspace_name / "activitysim" / "configs"
        roots = activitysim_config_root_dirs(settings.activitysim, root)
        for config_root in roots:
            config_root.mkdir(parents=True)
            (config_root / "settings.yaml").write_text(
                "models: []\nchunk_size: 0\n", encoding="utf-8"
            )
        return roots

    first_roots = make_roots("workspace-a")
    second_roots = make_roots("workspace-b")
    adapter = ActivitySimConfigAdapter()
    first = adapter.discover(
        list(first_roots), identity=IdentityManager(project_root=tmp_path)
    )
    second = adapter.discover(
        list(second_roots), identity=IdentityManager(project_root=tmp_path)
    )
    assert first.content_hash == second.content_hash
    (second_roots[0] / "settings.yaml").write_text(
        "models: []\nchunk_size: 1\n", encoding="utf-8"
    )
    changed = adapter.discover(
        list(second_roots), identity=IdentityManager(project_root=tmp_path)
    )
    assert changed.content_hash != first.content_hash


@pytest.mark.parametrize("selected_skim_role", (ASIM_OMX_SKIMS, ZARR_SKIMS))
def test_activitysim_run_uses_only_staged_model_inputs_after_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, selected_skim_role: str
) -> None:
    """A native body must not rediscover poisoned mutable ActivitySim roots.

    This catches a regression that would make the launch tree look correctly
    materialized to Consist but mount or project a former workspace input/output
    location during actual ActivitySim execution.
    """

    poisoned_data_root = tmp_path / "activitysim" / "data"
    poisoned_config_root = tmp_path / "activitysim" / "configs"
    for root in (poisoned_data_root, poisoned_config_root):
        root.mkdir(parents=True)
        (root / "POISONED").write_text("must not be read\n", encoding="utf-8")

    staged_data_root = tmp_path / "activitysim" / "native-launch" / "data"
    staged_data_root.mkdir(parents=True)
    staged_inputs = {
        ASIM_LAND_USE_IN: staged_data_root / "land_use.csv",
        ASIM_HOUSEHOLDS_IN: staged_data_root / "households.csv",
        ASIM_PERSONS_IN: staged_data_root / "persons.csv",
    }
    for role, path in staged_inputs.items():
        path.write_text(f"{role}\n", encoding="utf-8")
    selected_skim = staged_data_root / (
        "skims.zarr" if selected_skim_role == ZARR_SKIMS else "skims.omx"
    )
    if selected_skim_role == ZARR_SKIMS:
        selected_skim.mkdir()
        (selected_skim / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")
    else:
        selected_skim.write_text("omx skims\n", encoding="utf-8")
    staged_inputs[selected_skim_role] = selected_skim

    adapter_roots: list[Path] = []
    adapter_root = tmp_path / "adapter-selected-configs"
    for dirname in ("configs_extended", "configs", "configs_mp", "configs_sh_compile"):
        source = adapter_root / dirname
        source.mkdir(parents=True)
        (source / "settings.yaml").write_text(f"source: {dirname}\n", encoding="utf-8")
        adapter_roots.append(source)
    adapter = ActivitySimConfigAdapter()
    assert adapter.discover(
        adapter_roots, identity=IdentityManager(project_root=tmp_path)
    ).content_hash

    staged_config_root = tmp_path / "activitysim" / "native-launch" / "configs"
    output_root = tmp_path / "activitysim" / "output"
    config_roots = required_activitysim_config_roots("configs_extended")
    launch_context = ActivitySimLaunchContext(
        workspace_root=tmp_path,
        mutable_data_dir=staged_data_root,
        output_dir=output_root,
        compile_output_dir=tmp_path / "activitysim" / "compile-output",
        mutable_configs_dir=staged_config_root,
        runtime_cache_dir=output_root / "cache",
        runtime_zarr_path=output_root / "cache" / "skims.zarr",
        shared_cache_dir=tmp_path / "shared-cache",
        shared_tmp_dir=tmp_path / "tmp",
        requires_staged_config_dirs=True,
        config_roots=config_roots,
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(region="test"),
        activitysim=SimpleNamespace(
            main_configs_dir="configs_extended",
            region_mappings={"region_to_subdir": {"test": "example"}},
            output_tables={"tables": ["households", "persons", "land_use"]},
        ),
    )
    captured: dict[str, object] = {}
    native_state = SimpleNamespace(
        full_settings=settings,
        current_year=2019,
        current_inner_iter=0,
    )
    body_runner = ActivitysimRunner("activitysim", native_state)

    def capture_container(**kwargs: object) -> bool:
        captured["mounts"] = kwargs["volumes"]
        return True

    monkeypatch.setattr(
        body_runner, "get_model_and_image", lambda *_args: ("activitysim", "image")
    )
    monkeypatch.setattr(
        body_runner, "get_base_asim_cmd", lambda *_args: ["activitysim"]
    )
    monkeypatch.setattr(body_runner, "get_asim_additional_args", lambda *_args: [])
    monkeypatch.setattr(body_runner, "run_container", capture_container)

    class Runner(ActivitysimRunner):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def run(
            self, inputs: object, context: ActivitySimLaunchContext, **kwargs: object
        ) -> None:
            captured["inputs"] = inputs
            captured["context"] = context
            captured["kwargs"] = kwargs
            body_runner._run(
                inputs,
                context,
                skim_mode="zarr" if selected_skim_role == ZARR_SKIMS else "omx",
                extra_inputs=(
                    {ZARR_SKIMS: selected_skim}
                    if selected_skim_role == ZARR_SKIMS
                    else {}
                ),
            )

    monkeypatch.setattr(
        activitysim_steps,
        "ActivitysimRunner",
        Runner,
    )
    workspace = SimpleNamespace(
        full_path=str(tmp_path),
        get_asim_mutable_data_dir=lambda: pytest.fail("ambient data root was read"),
        get_asim_mutable_configs_dir=lambda: pytest.fail(
            "ambient config root was read"
        ),
        get_asim_output_dir=lambda: str(output_root),
    )

    skim_kwargs = (
        {"zarr_skims": selected_skim}
        if selected_skim_role == ZARR_SKIMS
        else {"omx_skims": selected_skim}
    )
    activitysim_steps._activitysim_run_callable(
        staged_inputs[ASIM_LAND_USE_IN],
        staged_inputs[ASIM_HOUSEHOLDS_IN],
        staged_inputs[ASIM_PERSONS_IN],
        **skim_kwargs,
        settings=settings,
        state=native_state,
        workspace=workspace,
        activitysim_launch_context=launch_context,
        activitysim_config_staging_plan=ActivitySimConfigStagingPlan(
            source_root=adapter_root,
            config_roots=config_roots,
        ),
    )

    mounts = captured["mounts"]
    assert isinstance(mounts, dict)
    mounted_paths = tuple(mounts)
    assert str(poisoned_data_root) not in mounted_paths
    assert str(poisoned_config_root) not in mounted_paths
    remote_root = "/activitysim/example"
    expected_mounts = {
        str(staged_data_root.resolve()): {
            "bind": f"{remote_root}/data",
            "mode": "ro",
        },
        str(output_root.resolve()): {"bind": f"{remote_root}/output", "mode": "rw"},
        str((staged_config_root / "configs_extended").resolve()): {
            "bind": f"{remote_root}/configs",
            "mode": "ro",
        },
        str((staged_config_root / "configs_mp").resolve()): {
            "bind": f"{remote_root}/configs_mp",
            "mode": "ro",
        },
        str((staged_config_root / "configs_sh_compile").resolve()): {
            "bind": f"{remote_root}/configs_sh_compile",
            "mode": "ro",
        },
        str(launch_context.shared_tmp_dir.resolve()): {"bind": "/tmp", "mode": "rw"},
        str(launch_context.shared_cache_dir.resolve()): {
            "bind": "/app/numba_cache",
            "mode": "rw",
        },
    }
    assert mounts == expected_mounts
    assert str(launch_context.compile_output_dir.resolve()) not in mounts
    captured_inputs = captured["inputs"]
    assert isinstance(captured_inputs, ActivitySimPreprocessOutputs)
    assert captured_inputs.mutable_data_dir == staged_data_root
    assert captured_inputs.land_use_table == staged_inputs[ASIM_LAND_USE_IN]
    assert captured_inputs.households_table == staged_inputs[ASIM_HOUSEHOLDS_IN]
    assert captured_inputs.persons_table == staged_inputs[ASIM_PERSONS_IN]
    assert captured["context"] is launch_context
    captured_kwargs = captured["kwargs"]
    assert isinstance(captured_kwargs, dict)
    if selected_skim_role == ZARR_SKIMS:
        assert captured_inputs.omx_skims is None
        assert captured_kwargs["skim_mode"] == "zarr"
        assert captured_kwargs["extra_inputs"] == {ZARR_SKIMS: selected_skim}
    else:
        assert captured_inputs.omx_skims == selected_skim
        assert captured_kwargs["skim_mode"] == "omx"
        assert captured_kwargs["extra_inputs"] == {}
    assert (staged_config_root / "configs_extended" / "settings.yaml").read_text(
        encoding="utf-8"
    ) == "source: configs_extended\n"

    state = SimpleNamespace(year=2030, forecast_year=2030, iteration=1)
    resolved = ResolvedStepInputs(
        step_name="activitysim_run",
        binding=BindingResult(),
        metadata={"activitysim_produces_zarr": False},
    )
    declared = activitysim_steps.activitysim_run.output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )
    for output_spec in declared.values():
        output_path = Path(output_spec.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("declared output\n", encoding="utf-8")
    projected = activitysim_steps.activitysim_run.project_outputs(
        {key: object() for key in declared},
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )

    assert set(projected.raw_outputs) == set(declared)
    assert all(
        path.is_relative_to(output_root) for path in projected.raw_outputs.values()
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
