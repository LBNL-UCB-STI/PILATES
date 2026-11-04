"""Focused BEAM restart diagnostics and explicit canary policy.

Whole-stage and whole-year execution are exercised through the native golden
workflow.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from consist import BindingResult

from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.stages import supply_demand_beam as beam_stage
from pilates.workflows.stages.supply_demand_beam import (
    _FAIL_AFTER_BEAM_RUN_ENV,
    _maybe_fail_after_beam_run_for_canary,
)


def test_beam_restart_canary_failpoint_requires_explicit_env(monkeypatch):
    monkeypatch.delenv(_FAIL_AFTER_BEAM_RUN_ENV, raising=False)

    _maybe_fail_after_beam_run_for_canary(year=2021, iteration=0)

    monkeypatch.setenv(_FAIL_AFTER_BEAM_RUN_ENV, "1")
    with pytest.raises(RuntimeError, match="Injected failure after completed beam_run"):
        _maybe_fail_after_beam_run_for_canary(year=2021, iteration=0)


def test_beam_stage_passes_one_compiled_launch_config_to_binding_and_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The config Consist binds is the same config the runner receives at launch."""

    from pilates.beam.launch_config import BeamLaunchConfig

    launch_root = tmp_path / "launch"
    launch_primary = launch_root / "beam.conf"
    launch_root.mkdir()
    launch_primary.write_text("beam {}\n", encoding="utf-8")
    launch_config = BeamLaunchConfig(root=launch_root, primary_config=launch_primary)
    preprocess_outputs = SimpleNamespace(
        prepared_inputs={"plans_beam_in": tmp_path / "plans.csv"}
    )
    run_inputs = ResolvedStepInputs(
        step_name="beam_run",
        binding=BindingResult(inputs={"beam_config_file": launch_primary}),
    )
    postprocess_inputs = ResolvedStepInputs(
        step_name="beam_postprocess",
        binding=BindingResult(inputs={}),
    )
    observed: dict[str, object] = {}

    class Coupler:
        def get(self, _key, default=None):
            return default

    scenario = SimpleNamespace(
        coupler=Coupler(), tracker=SimpleNamespace(run_dir=tmp_path)
    )
    context = SimpleNamespace(
        settings=SimpleNamespace(beam=SimpleNamespace(full_skim=None)),
        state=SimpleNamespace(
            is_restart_run=False,
            sub_stage_progress=None,
            run_info_path=None,
            archive_run_dir=None,
            set_sub_stage_progress=lambda _: None,
        ),
        workspace=SimpleNamespace(get_beam_output_dir=lambda: tmp_path),
    )

    def fake_execute_step(
        *, phase, runtime_kwargs=None, resolved_inputs=None, **_kwargs
    ):
        if phase == "preprocess":
            return SimpleNamespace(), preprocess_outputs
        if phase == "run":
            observed["runtime"] = runtime_kwargs
            observed["resolved"] = resolved_inputs
            return (
                SimpleNamespace(run=SimpleNamespace(id="run-1"), outputs={}),
                SimpleNamespace(),
            )
        if phase == "postprocess":
            return SimpleNamespace(), SimpleNamespace()
        pytest.fail(f"unexpected phase {phase}")

    def resolve_run_inputs(**kwargs):
        observed["resolver"] = kwargs
        return run_inputs

    monkeypatch.setattr(
        beam_stage, "_compile_beam_launch_config", lambda **_: launch_config
    )
    monkeypatch.setattr(
        beam_stage, "beam_run", SimpleNamespace(resolve_inputs=resolve_run_inputs)
    )
    monkeypatch.setattr(
        beam_stage,
        "beam_postprocess",
        SimpleNamespace(resolve_inputs=lambda **_kwargs: postprocess_inputs),
    )
    monkeypatch.setattr(beam_stage, "execute_step", fake_execute_step)
    monkeypatch.setattr(
        beam_stage, "_maybe_fail_after_beam_run_for_canary", lambda **_: None
    )

    beam_stage._run_beam_steps(
        scenario=scenario,
        year=2030,
        iteration=1,
        context=context,
    )

    assert observed["resolved"] is run_inputs
    assert observed["runtime"] == {"beam_launch_config": launch_config}


def test_native_archive_is_called_with_verify_before_canary_failpoint(
    monkeypatch, tmp_path: Path
):
    """Step-level RunCompletionArchivePolicy executes during beam_run and before canary failpoint."""
    from dataclasses import replace
    from consist import RunCompletionArchivePolicy

    assert beam_stage.beam_run.archive_outputs is True
    assert beam_stage.beam_postprocess.archive_outputs is True

    local_root = tmp_path / "local"
    archive_root = tmp_path / "archive"
    calls: list[str] = []
    tracked_run_outputs = {"linkstats": object()}

    def fake_scenario_run(**kwargs):
        phase = kwargs["phase"]
        calls.append(phase)
        output_policy = kwargs["output_policy"]
        assert isinstance(
            output_policy.run_completion_archive, RunCompletionArchivePolicy
        )
        if phase == "preprocess":
            return SimpleNamespace(run=SimpleNamespace(id="beam-prep-1"), outputs={})
        if phase == "run":
            assert (
                output_policy.run_completion_archive.archive_root
                == (
                    archive_root
                    / "consist-recovery"
                    / "beam_run__supply_demand__y2021__i0__run"
                ).resolve()
            )
            calls.append("archived")
            return SimpleNamespace(
                run=SimpleNamespace(id="beam-run-1"), outputs=tracked_run_outputs
            )
        pytest.fail("canary failure must happen before native postprocess")

    scenario = SimpleNamespace(
        coupler=object(),
        tracker=SimpleNamespace(run_dir=tmp_path),
        run=fake_scenario_run,
    )
    context = SimpleNamespace(
        settings=SimpleNamespace(
            run=SimpleNamespace(enable_archive_copy=True),
            beam=SimpleNamespace(full_skim=None),
        ),
        state=SimpleNamespace(
            is_restart_run=False,
            sub_stage_progress=None,
            run_info_path=None,
            archive_run_dir=str(archive_root),
            set_sub_stage_progress=lambda _: None,
        ),
        workspace=SimpleNamespace(
            full_path=str(local_root),
            archive_run_dir=str(archive_root),
            get_beam_output_dir=lambda: local_root,
        ),
    )

    def failpoint(**_kwargs):
        calls.append("failpoint")
        raise RuntimeError("canary")

    monkeypatch.setattr(
        beam_stage,
        "_compile_beam_launch_config",
        lambda **_kwargs: SimpleNamespace(
            root=tmp_path / "launch", primary_config=tmp_path / "launch" / "beam.conf"
        ),
    )
    monkeypatch.setattr(
        beam_stage,
        "beam_preprocess",
        replace(
            beam_stage.beam_preprocess,
            resolve_inputs=lambda **_kwargs: ResolvedStepInputs(
                "beam_preprocess", BindingResult(inputs={})
            ),
            project_outputs=lambda *_args, **_kwargs: SimpleNamespace(),
            output_paths=None,
            output_sets=None,
            execution_options=None,
            input_contract=None,
            preflight_identity=False,
        ),
    )
    monkeypatch.setattr(
        beam_stage,
        "beam_run",
        replace(
            beam_stage.beam_run,
            resolve_inputs=lambda **_kwargs: ResolvedStepInputs(
                "beam_run", BindingResult(inputs={})
            ),
            project_outputs=lambda *_args, **_kwargs: SimpleNamespace(),
            output_paths=None,
            output_sets=None,
            execution_options=None,
            input_contract=None,
            preflight_identity=False,
        ),
    )
    monkeypatch.setattr(beam_stage, "_maybe_fail_after_beam_run_for_canary", failpoint)

    with pytest.raises(RuntimeError, match="canary"):
        beam_stage._run_beam_steps(
            scenario=scenario,
            year=2021,
            iteration=0,
            context=context,
        )

    assert calls == ["preprocess", "run", "archived", "failpoint"]


def test_run_completion_archive_handles_nested_scalar_and_enclosing_directory_artifacts(
    tmp_path: Path,
) -> None:
    """Archiving BEAM outputs succeeds when scalar files sit inside directory_artifact trees."""
    from consist import OutputPolicyOptions, RunCompletionArchivePolicy, Tracker
    from consist.types import OutputArtifactSpec

    workspace_root = tmp_path / "workspace"
    iter_dir = workspace_root / "beam" / "year-2020-iteration-0" / "ITERS" / "it.0"
    iter_dir.mkdir(parents=True, exist_ok=True)
    raw_skim = iter_dir / "0.skimsOD_Aggregated.csv.gz"
    raw_skim.write_text("origin,destination,travel_time\n1,2,10\n", encoding="utf-8")
    final_omx = workspace_root / "beam" / "year-2020-iteration-0" / "skims.omx"
    final_omx.write_bytes(b"omx-bytes")
    root_dir = workspace_root / "beam" / "year-2020-iteration-0"

    tracker = Tracker(
        run_dir=tmp_path / "consist_runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        mounts={"workspace": str(workspace_root)},
    )

    def _produce_postprocess() -> None:
        return None

    recovery_root = tmp_path / "recovery" / "beam_postprocess_archive_test"
    run_result = tracker.run(
        fn=_produce_postprocess,
        name="beam_postprocess_archive_test",
        output_paths={
            "beam_skims_raw": raw_skim,
            "final_skims_omx": final_omx,
            "beam_iteration_output_dir": OutputArtifactSpec.for_directory(iter_dir),
            "beam_output_root_dir": OutputArtifactSpec.for_directory(root_dir),
        },
        output_policy=OutputPolicyOptions(
            run_completion_archive=RunCompletionArchivePolicy(
                archive_root=recovery_root,
                mode="copy",
            )
        ),
    )

    expected_roots = {str(workspace_root.resolve()), str(recovery_root.resolve())}
    for key, expected_path in (
        ("beam_skims_raw", raw_skim),
        ("final_skims_omx", final_omx),
        ("beam_iteration_output_dir", iter_dir),
        ("beam_output_root_dir", root_dir),
    ):
        art = run_result.outputs[key]
        assert Path(art.abs_path).resolve() == expected_path.resolve()
        roots = {str(Path(r).resolve()) for r in (art.meta.get("recovery_roots") or [])}
        assert expected_roots.issubset(roots)
    assert (
        recovery_root
        / "beam"
        / "year-2020-iteration-0"
        / "ITERS"
        / "it.0"
        / "0.skimsOD_Aggregated.csv.gz"
    ).is_file()
    assert (recovery_root / "beam" / "year-2020-iteration-0" / "skims.omx").is_file()


def test_interrupted_postprocess_restart_proceeds_directly_through_step_execution(
    monkeypatch, tmp_path: Path
):
    """Restarting after an interrupted postprocess proceeds directly through step execution without refusal."""
    launch_root = tmp_path / "launch"
    launch_primary = launch_root / "beam.conf"
    launch_root.mkdir()
    launch_primary.write_text("beam {}\n", encoding="utf-8")

    class Coupler:
        def get(self, _key, default=None):
            return default

    scenario = SimpleNamespace(
        coupler=Coupler(), tracker=SimpleNamespace(run_dir=tmp_path)
    )
    context = SimpleNamespace(
        settings=SimpleNamespace(beam=SimpleNamespace(full_skim=None)),
        state=SimpleNamespace(
            is_restart_run=True,
            sub_stage_progress="postprocessor",
            run_info_path=None,
            archive_run_dir=None,
            set_sub_stage_progress=lambda _: None,
        ),
        workspace=SimpleNamespace(get_beam_output_dir=lambda: tmp_path),
    )
    calls: list[str] = []

    def fake_execute_step(*, phase, **_kwargs):
        calls.append(phase)
        if phase == "preprocess":
            return SimpleNamespace(), SimpleNamespace(prepared_inputs={})
        if phase == "run":
            return (
                SimpleNamespace(run=SimpleNamespace(id="run-1"), outputs={}),
                SimpleNamespace(),
            )
        if phase == "postprocess":
            return SimpleNamespace(), SimpleNamespace()
        pytest.fail(f"unexpected phase {phase}")

    monkeypatch.setattr(
        beam_stage,
        "_compile_beam_launch_config",
        lambda **_: SimpleNamespace(root=launch_root, primary_config=launch_primary),
    )
    monkeypatch.setattr(
        beam_stage,
        "beam_run",
        SimpleNamespace(
            resolve_inputs=lambda **_kwargs: ResolvedStepInputs(
                "beam_run", BindingResult(inputs={})
            )
        ),
    )
    monkeypatch.setattr(
        beam_stage,
        "beam_postprocess",
        SimpleNamespace(
            resolve_inputs=lambda **_kwargs: ResolvedStepInputs(
                "beam_postprocess", BindingResult(inputs={})
            )
        ),
    )
    monkeypatch.setattr(beam_stage, "execute_step", fake_execute_step)
    monkeypatch.setattr(
        beam_stage, "_maybe_fail_after_beam_run_for_canary", lambda **_: None
    )

    beam_stage._run_beam_steps(
        scenario=scenario,
        year=2021,
        iteration=0,
        context=context,
    )

    assert calls == ["preprocess", "run", "postprocess"]


# ==============================================================================
# Semantic Gaps for Removed beam_checkpoint Tests (per spike brief instruction)
# ==============================================================================
# The following tests encoded beam_checkpoint semantics that have no direct native
# equivalent in Consist, and are recorded here as architectural gaps:
#
# 1. test_beam_checkpoint_closure_uses_frozen_zarr_producer_not_scenario_cache:
#    GAP: Consist lacks pinned producer closures; cache resolution dynamically
#    queries scenario coupler rather than freezing the producer run's lineage.
#
# 2. test_fresh_zarr_postprocess_closure_keeps_current_run_artifact_provenance:
#    GAP: Consist does not pin run-artifact bindings; Coupler re-resolves artifacts
#    dynamically.
#
# 3. test_beam_checkpoint_closure_keeps_requested_beam_run_on_cache_hit:
#    GAP: Stochastic BEAM output collisions cause Consist find_matching_run() to
#    select the newest run (epoch_us DESC) rather than retaining the original producer.
#
# 4. test_public_beam_handoff_is_postprocess_owned_across_run_modes_and_resume:
#    GAP: Checkpoint-based handoff bypass is replaced by Consist's standard step
#    output mappings.
#
# 5. test_rebound_checkpoint_requires_normal_resolver_identity_and_destination:
#    GAP: Rebound destination verification was part of checkpoint deserialization;
#    native Consist relies on declared output_paths and step contracts.
# ==============================================================================
