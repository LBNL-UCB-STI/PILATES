"""Freeze the ResolvedBinding V1 eligibility decision for native steps."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
from pilates.runtime import activitysim_run_acceptance


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
    observation_log = tmp_path / "activitysim-observations.jsonl"
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

    class Runner:
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
        activitysim_steps.ModelFactory,
        "get_runner",
        lambda _factory, *_args: Runner(),
    )
    monkeypatch.setenv(
        "PILATES_ACTIVITYSIM_RUN_ACCEPTANCE_OBSERVATIONS", str(observation_log)
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

    assert observation_log.exists()
    assert [
        json.loads(line)["event"] for line in observation_log.read_text().splitlines()
    ] == ["activitysim_run_body"]

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
    if selected_skim_role == ZARR_SKIMS:
        expected_mounts[str(launch_context.runtime_cache_dir.resolve())] = {
            "bind": f"{remote_root}/output/cache",
            "mode": "ro",
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


def test_activitysim_run_acceptance_manifest_allows_only_three_tables_and_one_skim(
    tmp_path: Path,
) -> None:
    """The operator manifest cannot silently add ambient ActivitySim inputs."""

    inputs = {
        ASIM_LAND_USE_IN: tmp_path / "land_use.csv",
        ASIM_HOUSEHOLDS_IN: tmp_path / "households.csv",
        ASIM_PERSONS_IN: tmp_path / "persons.csv",
        ZARR_SKIMS: tmp_path / "skims.zarr",
    }
    for key, path in inputs.items():
        if key == ZARR_SKIMS:
            path.mkdir()
            (path / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")
        else:
            path.write_text(f"{key}\n", encoding="utf-8")
    manifest_path = tmp_path / "activitysim-inputs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "released_consist_version": "9.8.7",
                "inputs": {key: str(path) for key, path in inputs.items()},
                "cohort": {
                    "workflow_year": 2017,
                    "forecast_year": 2019,
                    "iteration": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = activitysim_run_acceptance.load_manifest(manifest_path)

    assert manifest.selected_skim_role == ZARR_SKIMS
    assert manifest.inputs == {key: path.resolve() for key, path in inputs.items()}
    assert manifest.released_consist_version == "9.8.7"

    invalid = json.loads(manifest_path.read_text(encoding="utf-8"))
    invalid["inputs"][ASIM_OMX_SKIMS] = str(tmp_path / "skims.omx")
    (tmp_path / "skims.omx").write_text("extra skim\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one selected skim"):
        activitysim_run_acceptance.load_manifest(manifest_path)


def test_activitysim_run_acceptance_requires_operator_released_consist_version(
    tmp_path: Path,
) -> None:
    """The release gate cannot fall back to a repository-adjacent version."""

    manifest_path = tmp_path / "activitysim-inputs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "inputs": {},
                "cohort": {
                    "workflow_year": 2017,
                    "forecast_year": 2019,
                    "iteration": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="released_consist_version"):
        activitysim_run_acceptance.load_manifest(manifest_path)


def test_activitysim_run_acceptance_allows_explicit_editable_consist_mode(
    tmp_path: Path,
) -> None:
    """Pre-merge evidence selects editable Consist without a release version."""

    inputs = {
        ASIM_LAND_USE_IN: tmp_path / "land_use.csv",
        ASIM_HOUSEHOLDS_IN: tmp_path / "households.csv",
        ASIM_PERSONS_IN: tmp_path / "persons.csv",
        ZARR_SKIMS: tmp_path / "skims.zarr",
    }
    for key, path in inputs.items():
        if key == ZARR_SKIMS:
            path.mkdir()
            (path / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")
        else:
            path.write_text(f"{key}\n", encoding="utf-8")
    manifest_path = tmp_path / "activitysim-editable-inputs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "consist_install_mode": "editable",
                "inputs": {key: str(path) for key, path in inputs.items()},
                "cohort": {
                    "workflow_year": 2017,
                    "forecast_year": 2019,
                    "iteration": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = activitysim_run_acceptance.load_manifest(manifest_path)

    assert manifest.consist_install_mode == "editable"
    assert manifest.released_consist_version is None


def test_activitysim_run_editable_preflight_records_exact_checkout_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pre-merge evidence proves the imported module belongs to its checkout."""

    source = tmp_path / "consist"
    module = source / "src" / "consist" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        activitysim_run_acceptance,
        "consist",
        SimpleNamespace(__file__=str(module)),
    )

    def git_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="feedface\n")
        if command[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(stdout="")
        raise AssertionError(command)

    monkeypatch.setattr(activitysim_run_acceptance.subprocess, "run", git_run)

    evidence = activitysim_run_acceptance.preflight_editable_consist(source)

    assert evidence == {
        "consist_install_mode": "editable",
        "evidence_kind": "pre_merge_editable_integration",
        "editable_source": str(source.resolve()),
        "editable_revision": "feedface",
        "editable_dirty": False,
        "import_path": str(module.resolve()),
        "import_within_editable_source": True,
        "editable_install": True,
        "release_install": False,
        "valid": True,
    }


def test_activitysim_run_driver_rejects_relative_editable_consist_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The driver must not turn a relative checkout argument into an absolute one."""

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("run: {}\n", encoding="utf-8")
    manifest_path = tmp_path / "inputs.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest = activitysim_run_acceptance.AcceptanceManifest(
        inputs={},
        selected_skim_role=ZARR_SKIMS,
        workflow_year=2017,
        forecast_year=2019,
        iteration=0,
        consist_install_mode="editable",
    )
    monkeypatch.setattr(
        activitysim_run_acceptance, "load_manifest", lambda _path: manifest
    )
    monkeypatch.setattr(
        activitysim_run_acceptance,
        "preflight_editable_consist",
        lambda _path: pytest.fail("relative source reached editable preflight"),
    )

    with pytest.raises(ValueError, match="must be an absolute path"):
        activitysim_run_acceptance.main(
            [
                "--settings",
                str(settings_path),
                "--manifest",
                str(manifest_path),
                "--evidence-root",
                str(tmp_path / "evidence"),
                "--editable-consist",
                "relative-consist",
            ]
        )


@pytest.mark.parametrize(
    (
        "direct_url",
        "installed_version",
        "metadata_version",
        "public_version",
        "expected_valid",
    ),
    [
        (None, "9.8.7", "9.8.7", "9.8.7", True),
        ('{"dir_info": {"editable": true}}', "9.8.7", "9.8.7", "9.8.7", False),
        (None, "9.8.6", "9.8.7", "9.8.7", False),
        (None, "9.8.7", "9.8.6", "9.8.7", False),
        (None, "9.8.7", "9.8.7", "9.8.6", False),
    ],
)
def test_activitysim_run_release_preflight_records_and_enforces_release_identity(
    monkeypatch: pytest.MonkeyPatch,
    direct_url: str | None,
    installed_version: str,
    metadata_version: str,
    public_version: str,
    expected_valid: bool,
) -> None:
    """Editable or inconsistent Consist metadata invalidates the acceptance."""

    class Distribution:
        version = installed_version
        files = (Path("consist/__init__.py"),)

        def read_text(self, name: str) -> str | None:
            assert name == "direct_url.json"
            return direct_url

        def locate_file(self, member: Path) -> Path:
            assert member == Path("consist/__init__.py")
            return Path("/released/site-packages") / member

    monkeypatch.setattr(
        activitysim_run_acceptance,
        "consist",
        SimpleNamespace(
            __version__=public_version,
            __file__="/released/site-packages/consist/__init__.py",
        ),
    )
    monkeypatch.setattr(
        activitysim_run_acceptance.importlib_metadata,
        "distribution",
        lambda package_name: Distribution(),
    )
    monkeypatch.setattr(
        activitysim_run_acceptance.importlib_metadata,
        "version",
        lambda package_name: metadata_version,
    )

    evidence = activitysim_run_acceptance.preflight_released_consist("9.8.7")

    assert evidence["required_release_version"] == "9.8.7"
    assert evidence["installed_version"] == installed_version
    assert evidence["public_version"] == public_version
    assert evidence["import_path"] == "/released/site-packages/consist/__init__.py"
    assert evidence["distribution_import_paths"] == [
        "/released/site-packages/consist/__init__.py"
    ]
    assert evidence["distribution_import_path_match"] is True
    assert evidence["editable_install"] is (direct_url is not None)
    assert evidence["release_install"] is (direct_url is None)
    assert evidence["valid"] is expected_valid


def test_activitysim_run_release_preflight_rejects_same_version_shadow_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PYTHONPATH checkout cannot impersonate the selected same-version wheel."""

    class Distribution:
        version = "9.8.7"
        files = (Path("consist/__init__.py"),)

        def read_text(self, name: str) -> None:
            assert name == "direct_url.json"
            return None

        def locate_file(self, member: Path) -> Path:
            assert member == Path("consist/__init__.py")
            return Path("/released/site-packages") / member

    monkeypatch.setattr(
        activitysim_run_acceptance,
        "consist",
        SimpleNamespace(
            __version__="9.8.7",
            __file__="/shadow-checkout/consist/__init__.py",
        ),
    )
    monkeypatch.setattr(
        activitysim_run_acceptance.importlib_metadata,
        "distribution",
        lambda _package_name: Distribution(),
    )
    monkeypatch.setattr(
        activitysim_run_acceptance.importlib_metadata,
        "version",
        lambda _package_name: "9.8.7",
    )

    evidence = activitysim_run_acceptance.preflight_released_consist("9.8.7")

    assert evidence["valid"] is False
    assert evidence["import_path"] == "/shadow-checkout/consist/__init__.py"
    assert evidence["distribution_import_path_match"] is False


def test_activitysim_run_retains_failed_release_preflight_before_any_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rejected release check still leaves its evidence for operator review."""

    for name in (
        "PILATES_LOCAL_RUN_DIR",
        "PILATES_ARCHIVE_RUN_DIR",
        "PILATES_ENABLE_ARCHIVE_COPY",
    ):
        monkeypatch.delenv(name, raising=False)
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("run: {}\n", encoding="utf-8")
    manifest_path = tmp_path / "inputs.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    evidence_root = tmp_path / "evidence"
    manifest = activitysim_run_acceptance.AcceptanceManifest(
        inputs={},
        selected_skim_role=ZARR_SKIMS,
        workflow_year=2017,
        forecast_year=2019,
        iteration=0,
        released_consist_version="9.8.7",
    )
    preflight = {
        "required_release_version": "9.8.7",
        "installed_version": "9.8.7",
        "public_version": "9.8.7",
        "import_path": "/adjacent/consist/__init__.py",
        "editable_install": True,
        "release_install": False,
        "valid": False,
    }
    monkeypatch.setattr(
        activitysim_run_acceptance, "load_manifest", lambda _path: manifest
    )
    monkeypatch.setattr(
        activitysim_run_acceptance,
        "preflight_released_consist",
        lambda _version: preflight,
    )

    try:
        with pytest.raises(RuntimeError, match="requires the requested non-editable"):
            activitysim_run_acceptance.main(
                [
                    "--settings",
                    str(settings_path),
                    "--manifest",
                    str(manifest_path),
                    "--evidence-root",
                    str(evidence_root),
                ]
            )

        assert (
            json.loads((evidence_root / "runtime-environment.json").read_text())
            == preflight
        )
    finally:
        for name in (
            "PILATES_LOCAL_RUN_DIR",
            "PILATES_ARCHIVE_RUN_DIR",
            "PILATES_ENABLE_ARCHIVE_COPY",
        ):
            os.environ.pop(name, None)


def test_activitysim_run_driver_does_not_overwrite_retained_submitted_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A mutable source cannot replace the submission record after retention."""

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("run: {}\n", encoding="utf-8")
    manifest_path = tmp_path / "mutable-inputs.json"
    manifest_path.write_text('{"source": "changed"}\n', encoding="utf-8")
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    retained_manifest = evidence_root / "submitted-input-manifest.json"
    retained_manifest.write_text('{"source": "submitted"}\n', encoding="utf-8")
    manifest = activitysim_run_acceptance.AcceptanceManifest(
        inputs={},
        selected_skim_role=ZARR_SKIMS,
        workflow_year=2017,
        forecast_year=2019,
        iteration=0,
        released_consist_version="9.8.7",
    )
    monkeypatch.setattr(
        activitysim_run_acceptance, "load_manifest", lambda _path: manifest
    )
    monkeypatch.setattr(
        activitysim_run_acceptance,
        "preflight_released_consist",
        lambda _version: {"valid": False},
    )

    try:
        with pytest.raises(RuntimeError, match="retained submitted manifest"):
            activitysim_run_acceptance.main(
                [
                    "--settings",
                    str(settings_path),
                    "--manifest",
                    str(manifest_path),
                    "--evidence-root",
                    str(evidence_root),
                ]
            )
        assert retained_manifest.read_text(encoding="utf-8") == (
            '{"source": "submitted"}\n'
        )
    finally:
        for name in (
            "PILATES_LOCAL_RUN_DIR",
            "PILATES_ARCHIVE_RUN_DIR",
            "PILATES_ENABLE_ARCHIVE_COPY",
        ):
            os.environ.pop(name, None)


def test_activitysim_run_checksums_seal_evidence_control_records(
    tmp_path: Path,
) -> None:
    """A reviewer can detect mutation of every formal control record."""

    evidence_root = tmp_path / "evidence"
    controls = {
        "submitted-input-manifest.json": "submitted\n",
        "effective-input-manifest.json": "effective\n",
        "generated-settings.yaml": "settings\n",
        "runtime-environment.json": "runtime\n",
        "persisted-runs/cold.json": "persisted cold\n",
        "persisted-runs/fresh.json": "persisted fresh\n",
        "phases/cold.json": "phase cold\n",
        "phases/fresh.json": "phase fresh\n",
        "semantic-validation.json": "semantic\n",
    }
    for relative_path, content in controls.items():
        path = evidence_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    observation_log = evidence_root / "activitysim-observations.jsonl"
    observation_log.write_text("{}\n", encoding="utf-8")
    input_path = tmp_path / "input.parquet"
    input_path.write_text("input\n", encoding="utf-8")
    output_path = tmp_path / "output.parquet"
    output_path.write_text("output\n", encoding="utf-8")
    manifest = activitysim_run_acceptance.AcceptanceManifest(
        inputs={ASIM_LAND_USE_IN: input_path},
        selected_skim_role=ZARR_SKIMS,
        workflow_year=2017,
        forecast_year=2019,
        iteration=0,
        released_consist_version="9.8.7",
    )

    def execution() -> activitysim_run_acceptance.PhaseExecution:
        return activitysim_run_acceptance.PhaseExecution(
            cache_hit=False,
            requested_run_id="run",
            execution_run_id="run",
            source_run_id=None,
            declared_outputs={"activitysim_trips": output_path},
            selected_roles={},
            source_bindings={},
            input_identities={},
            config_staging={},
            persisted_run={},
            body_executions_before=0,
            body_executions_after=1,
            runner_preparation_attempts_before=0,
            runner_preparation_attempts_after=1,
        )

    activitysim_run_acceptance._write_checksums(
        evidence_root=evidence_root,
        manifest=manifest,
        cold=execution(),
        fresh=execution(),
        observation_log=observation_log,
    )

    checksums = json.loads((evidence_root / "checksums.json").read_text())["sha256"]
    assert set(controls).issubset(checksums)
    assert checksums["submitted-input-manifest.json"] == (
        "93577775f201d3c610a22609a1a97f6c86355f55055a979271ce3cb7fbbec558"
    )


def test_activitysim_run_phase_derives_counts_from_observation_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Phase evidence uses native observation entries, never cache-status inference."""

    from pilates.runtime.activitysim_run_observations import (
        record_activitysim_observation,
    )

    observation_log = tmp_path / "activitysim-observations.jsonl"
    monkeypatch.setenv(
        "PILATES_ACTIVITYSIM_RUN_ACCEPTANCE_OBSERVATIONS", str(observation_log)
    )

    class Scenario:
        coupler = SimpleNamespace(set_from_artifact=lambda *_args: None)

        def __enter__(self) -> Scenario:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class TrackerStub:
        def scenario(self, _name: str) -> Scenario:
            return Scenario()

    resolved = ResolvedStepInputs(
        step_name="activitysim_run",
        binding=BindingResult(inputs=None),
    )
    result = SimpleNamespace(
        cache_hit=False,
        run=SimpleNamespace(id="cold-run"),
    )
    native_step = SimpleNamespace(
        resolve_inputs=lambda **_kwargs: resolved,
        output_paths=lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        activitysim_run_acceptance, "Workspace", lambda *_args: object()
    )
    monkeypatch.setattr(
        activitysim_run_acceptance,
        "_stage_workspace_config",
        lambda **_kwargs: {"source_root": "source", "staged_root": "staged"},
    )
    monkeypatch.setattr(activitysim_run_acceptance, "activitysim_run", native_step)
    monkeypatch.setattr(
        activitysim_run_acceptance,
        "_persisted_run_evidence",
        lambda **_kwargs: {
            "cache_outcome": "miss",
            "requested_run_id": "cold-run",
            "execution_run_id": "cold-run",
            "source_run_id": None,
        },
    )

    def execute_native_body(**_kwargs: object) -> tuple[SimpleNamespace, None]:
        record_activitysim_observation("activitysim_run_body")
        record_activitysim_observation("activitysim_runner_preparation")
        return result, None

    monkeypatch.setattr(activitysim_run_acceptance, "execute_step", execute_native_body)

    phase = activitysim_run_acceptance.run_phase(
        phase="cold",
        workspace_root=tmp_path / "workspace",
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2017, current_inner_iter=0),
        tracker=TrackerStub(),
        artifacts={},
        evidence_root=tmp_path / "evidence",
        observation_log=observation_log,
    )

    assert phase.body_executions_before == 0
    assert phase.body_executions_after == 1
    assert phase.runner_preparation_attempts_before == 0
    assert phase.runner_preparation_attempts_after == 1


def test_activitysim_run_semantic_validation_reads_parquet_footer_not_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Large model outputs are checked from schema and row-count metadata only."""

    path = (
        tmp_path
        / "activitysim"
        / "output"
        / "final_pipeline"
        / "households"
        / "final.parquet"
    )
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"household_id": [1, 2], "income": [100, 200]}), path)

    def fail_full_table_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("semantic validation must not load a full parquet table")

    monkeypatch.setattr(pd, "read_parquet", fail_full_table_load)

    product = activitysim_run_acceptance._semantic_product(
        key="households_asim_out",
        path=path,
        workspace_root=tmp_path,
        configured_tables={"households_asim_out": "households"},
    )

    assert product == {
        "valid": True,
        "kind": "activitysim-final-pipeline-table",
        "relative_path": "activitysim/output/final_pipeline/households/final.parquet",
        "table": "households",
        "row_count": 2,
        "columns": ["household_id", "income"],
    }


def test_activitysim_run_acceptance_requires_one_body_then_model_aware_hydration() -> (
    None
):
    """A fresh cache hit needs the cold run's persisted identity and products."""

    product = {
        "valid": True,
        "kind": "activitysim-final-pipeline-table",
        "relative_path": "activitysim/output/final_pipeline/households/final.parquet",
        "table": "households",
        "row_count": 2,
        "columns": ["household_id", "income"],
    }
    input_paths = {
        ASIM_LAND_USE_IN: {
            "scope": "workspace",
            "relative_path": "activitysim/native-launch/data/land_use.csv",
        },
        ASIM_HOUSEHOLDS_IN: {
            "scope": "workspace",
            "relative_path": "activitysim/native-launch/data/households.csv",
        },
        ASIM_PERSONS_IN: {
            "scope": "workspace",
            "relative_path": "activitysim/native-launch/data/persons.csv",
        },
        ZARR_SKIMS: {
            "scope": "workspace",
            "relative_path": "activitysim/native-launch/data/skims.zarr",
        },
    }
    artifacts = {
        "action_inputs": [{"artifact_id": "input-1"}],
        "outputs": [{"artifact_id": "output-1"}],
    }

    def persisted(
        *, requested: str, source: str | None, outcome: str
    ) -> dict[str, object]:
        return {
            "binding_kind": "ordinary-binding",
            "requested_run_id": requested,
            "execution_run_id": source or requested,
            "source_run_id": source,
            "cache_outcome": outcome,
            "identity": {"config": "same", "input": "same"},
            "artifacts": artifacts,
            "requested_input_staging": {"normalized_input_paths": input_paths},
            "materialized_outputs": {
                "normalized_paths": {
                    "households_asim_out": {
                        "scope": "workspace",
                        "relative_path": product["relative_path"],
                    }
                }
            },
        }

    common = {
        "declared_outputs": {"households_asim_out": product},
        "selected_roles": {ASIM_HOUSEHOLDS_IN: ASIM_HOUSEHOLDS_IN},
        "source_bindings": {ASIM_HOUSEHOLDS_IN: "coupler"},
        "input_identities": {ASIM_HOUSEHOLDS_IN: "content-identity"},
    }
    cold = {
        **common,
        "workspace_root": "/evidence/workspaces/cold",
        "requested_run_id": "cold-run",
        "source_run_id": None,
        "persisted_run": persisted(requested="cold-run", source=None, outcome="miss"),
        "config_staging": {
            "source_root": "/project/configs/sfbay",
            "source_sha256": "config-identity",
            "staged_root": "/evidence/workspaces/cold/activitysim/configs",
        },
        "body_executions_before": 0,
        "body_executions_after": 1,
        "runner_preparation_attempts_before": 0,
        "runner_preparation_attempts_after": 1,
    }
    fresh = {
        **common,
        "workspace_root": "/evidence/workspaces/fresh",
        "requested_run_id": "fresh-run",
        "source_run_id": "cold-run",
        "persisted_run": persisted(
            requested="fresh-run", source="cold-run", outcome="hit"
        ),
        "config_staging": {
            "source_root": "/project/configs/sfbay",
            "source_sha256": "config-identity",
            "staged_root": "/evidence/workspaces/fresh/activitysim/configs",
        },
        "body_executions_before": 1,
        "body_executions_after": 1,
        "runner_preparation_attempts_before": 1,
        "runner_preparation_attempts_after": 1,
    }

    validation = activitysim_run_acceptance.validate(cold, fresh)

    assert validation["valid"] is True
    assert validation["semantic_products_valid"] is True
    assert validation["ordinary_binding_valid"] is True
    assert validation["body_and_preparation_valid"] is True
    assert validation["fresh_hydration_destinations_valid"] is True

    cold["body_executions_after"] = 2
    assert activitysim_run_acceptance.validate(cold, fresh)["valid"] is False

    cold["body_executions_after"] = 1
    fresh["runner_preparation_attempts_after"] = 2
    assert activitysim_run_acceptance.validate(cold, fresh)["valid"] is False


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
